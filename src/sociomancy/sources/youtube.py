"""YouTube comments for a study, split across the windows in its config.

Reuses the topic pipeline's search and hydration helpers. Search order and the
minimum comment count come from the study JSON. Videos are then balanced
across the configured channel tiers. Shorts under min_duration_sec are dropped.
Comment threads stay on YouTube's relevance sort, which is how that API returns
top comments.
"""

from __future__ import annotations

import pandas as pd
from googleapiclient.errors import HttpError

from sociomancy.sources.cache import cached_fetch, clean_text
from sociomancy.sources.windows import iso_day
from sociomancy.studies.config import Study
from sociomancy.youtube_algo_analyzer import (
    get_youtube_client,
    hydrate_videos_and_channels,
    parse_duration_seconds,
    search_paginated,
)


def _pick_balanced(videos: list[dict], limit: int, tier_names: list[str]) -> list[dict]:
    ordered = list(videos)
    buckets: dict[str, list[dict]] = {name: [] for name in tier_names}
    buckets.setdefault("Unknown", [])
    for row in ordered:
        buckets.setdefault(row["channel_tier"], []).append(row)

    per_tier = max(limit // max(len(tier_names), 1), 1)
    picked: list[dict] = []
    seen: set[str] = set()
    for tier in tier_names:
        for row in buckets.get(tier, [])[:per_tier]:
            if row["video_id"] not in seen:
                picked.append(row)
                seen.add(row["video_id"])
    if len(picked) < limit:
        for row in ordered:
            if row["video_id"] in seen:
                continue
            picked.append(row)
            seen.add(row["video_id"])
            if len(picked) >= limit:
                break
    return picked[:limit]


def _video_record(study: Study, item: dict, channel: dict, window: str) -> dict | None:
    snippet = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    details = item.get("contentDetails") or {}
    duration = parse_duration_seconds(details.get("duration", ""))
    if duration and duration < study.min_duration_sec:
        return None
    ch_stats = (channel or {}).get("statistics") or {}
    hidden = bool(ch_stats.get("hiddenSubscriberCount"))
    subs = 0 if hidden else int(ch_stats.get("subscriberCount") or 0)
    published = snippet.get("publishedAt") or ""
    return {
        "video_id": item.get("id") or "",
        "title": snippet.get("title") or "",
        "channel_id": snippet.get("channelId") or "",
        "channel_title": snippet.get("channelTitle") or "",
        "subscriber_count": subs,
        "subscribers_hidden": hidden,
        "channel_tier": study.channel_tier(subs, hidden),
        "published_at": published,
        "day": iso_day(published),
        "window": window,
        "view_count": int(stats.get("viewCount") or 0),
        "comment_count": int(stats.get("commentCount") or 0),
        "duration_sec": duration,
    }


def _fetch(study: Study, videos_per_window: int, comments_per_video: int) -> dict:
    youtube = get_youtube_client()
    windows: dict[str, dict] = {}
    tier_names = [name for name, _ in study.tiers]

    for window in study.windows:
        print(f"  [youtube] {window.name} (order={study.youtube_order})")
        # One extra page when a comment floor will throw videos out.
        found = search_paginated(
            youtube,
            query=study.youtube_query,
            published_after=window.published_after,
            published_before=window.published_before,
            order=study.youtube_order,
            max_pages=2 if study.min_comments else 1,
        )
        video_ids = [item["id"]["videoId"] for item in found["items"] if item.get("id", {}).get("videoId")]
        videos, channels = hydrate_videos_and_channels(youtube, video_ids)
        by_id = {item.get("id"): item for item in videos}
        records = []
        for video_id in video_ids:
            item = by_id.get(video_id)
            if not item:
                continue
            channel = channels.get((item.get("snippet") or {}).get("channelId", ""), {})
            record = _video_record(study, item, channel, window.name)
            if record and record["video_id"]:
                records.append(record)
        if study.min_comments:
            before = len(records)
            records = [row for row in records if row["comment_count"] >= study.min_comments]
            print(
                f"  [youtube] {len(records)}/{before} videos have at least {study.min_comments} comments"
            )
        if study.youtube_order == "viewCount":
            records.sort(key=lambda row: row["view_count"], reverse=True)
        selected = _pick_balanced(records, videos_per_window, tier_names)

        comments = []
        for video in selected:
            try:
                resp = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video["video_id"],
                        order="relevance",
                        maxResults=min(comments_per_video, 100),
                        textFormat="plainText",
                    )
                    .execute()
                )
            except HttpError as err:
                print(f"  [skip] comments for {video['video_id']}: {err.reason}")
                continue
            for item in resp.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = clean_text(snippet.get("textDisplay"))
                if not text:
                    continue
                published = snippet.get("publishedAt") or ""
                comments.append(
                    {
                        "item_id": f"youtube:{item['id']}",
                        "source": "youtube",
                        "kind": "comment",
                        "text": text,
                        "engagement": int(snippet.get("likeCount") or 0),
                        "created_at": published,
                        "day": video["day"],
                        "window": study.window_for(video["published_at"]),
                        "group": video["channel_tier"],
                        "thread_id": video["video_id"],
                        "video_title": video["title"],
                        "channel_title": video["channel_title"],
                        "subscriber_count": video["subscriber_count"],
                        "comment_published_at": published,
                    }
                )
        windows[window.name] = {"videos": selected, "comments": comments}

    return {"query": study.youtube_query, "windows": windows}


def collect_youtube(
    study: Study,
    refresh: bool = False,
    videos_per_window: int | None = None,
    comments_per_video: int | None = None,
) -> pd.DataFrame:
    if not study.youtube_query:
        print("  [skip] No youtube.query in the study config.")
        return pd.DataFrame()

    videos_per_window = study.videos_per_window if videos_per_window is None else videos_per_window
    comments_per_video = study.comments_per_video if comments_per_video is None else comments_per_video
    print(f"Collecting YouTube comments for {study.title}...")
    study.ensure_dirs()
    raw = cached_fetch(
        study.raw_dir / "youtube_raw.json",
        refresh,
        lambda: _fetch(study, videos_per_window, comments_per_video),
    )
    windows = raw.get("windows") or {}
    videos = [row for payload in windows.values() for row in payload.get("videos") or []]
    comments = [row for payload in windows.values() for row in payload.get("comments") or []]

    video_frame = pd.DataFrame(videos)
    comment_frame = pd.DataFrame(comments)
    if not video_frame.empty:
        video_path = study.processed_dir / "youtube_videos.csv"
        video_frame.to_csv(video_path, index=False)
        print(f"  [saved] {video_path} ({len(video_frame)} videos)")
    if comment_frame.empty:
        print("  [warn] YouTube returned no comments.")
        return comment_frame

    out = study.processed_dir / "youtube_comments.csv"
    comment_frame.to_csv(out, index=False)
    print(f"  [saved] {out} ({len(comment_frame)} comments)")
    return comment_frame
