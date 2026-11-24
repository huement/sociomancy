import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from trendspy import Trends

from sociomancy.text_scores import (
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    classify_zero_shot_onnx,
    score_texts_with_sociomancy_onnx,
)

try:
    import isodate
except ImportError:
    isodate = None

# Load your YouTube API Key from .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Output directories aligned with Sociomancy's data pipeline
RAW_DIR = Path("data/raw/topics")
PROCESSED_DIR = Path("data/processed/topics")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Title pattern matchers for Act 1 & Act 3
CONTROVERSY_RE = re.compile(
    r"\b(dead|killed|ruined|disaster|lawsuit|banned|outrage|backlash|worst|scam|lie|lied|exposed|boycott|slop)\b",
    re.I,
)
BREAKING_RE = re.compile(
    r"\b(just happened|breaking|officially|it's over|happening now|announced|confirmed)\b",
    re.I,
)
ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")

# ---------------------------------------------------------
# Helpers: Caching, Duration Parsing, & Batch Hydration
# ---------------------------------------------------------
def get_youtube_client():
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is missing from your .env file.")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def parse_duration_seconds(duration_str: str) -> int:
    if not duration_str:
        return 0
    if isodate is not None:
        try:
            return int(isodate.parse_duration(duration_str).total_seconds())
        except Exception:
            pass
    match = ISO_DURATION_RE.match(duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def cached_fetch(cache_file: Path, refresh: bool, fetch_fn) -> Any:
    if cache_file.exists() and not refresh:
        print(f"  [cache] Loaded {cache_file}")
        return json.loads(cache_file.read_text(encoding="utf-8"))
    payload = fetch_fn()
    cache_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  [saved] Cached raw response to {cache_file}")
    return payload


def search_paginated(
    youtube,
    query: str,
    published_after: str | None = None,
    published_before: str | None = None,
    order: str = "relevance",
    max_pages: int = 1,
) -> dict[str, Any]:
    """Paginates YouTube search.list (100 quota units per page) and captures totalResults."""
    items = []
    page_token = None
    total_results = 0

    for _ in range(max_pages):
        kwargs = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": order,
            "maxResults": 50,
            "relevanceLanguage": "en",
        }
        if published_after:
            kwargs["publishedAfter"] = published_after
        if published_before:
            kwargs["publishedBefore"] = published_before
        if page_token:
            kwargs["pageToken"] = page_token

        resp = youtube.search().list(**kwargs).execute()
        if total_results == 0:
            total_results = resp.get("pageInfo", {}).get("totalResults", 0)
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return {"total_results": total_results, "items": items}


def hydrate_videos_and_channels(youtube, video_ids: list[str]) -> tuple[list[dict], dict[str, dict]]:
    """Fetches full video & channel stats in batches of 50 (1 quota unit per batch)."""
    unique_ids = list(dict.fromkeys(vid for vid in video_ids if vid))
    videos = []

    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i : i + 50]
        resp = (
            youtube.videos()
            .list(part="snippet,contentDetails,statistics", id=",".join(batch))
            .execute()
        )
        videos.extend(resp.get("items", []))

    channel_ids = list(
        dict.fromkeys(v["snippet"]["channelId"] for v in videos if v.get("snippet", {}).get("channelId"))
    )
    channels: dict[str, dict] = {}
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        resp = youtube.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        for ch in resp.get("items", []):
            channels[ch["id"]] = ch

    return videos, channels


def build_video_rows(videos: list[dict], channels: dict[str, dict], tag: str = "") -> list[dict]:
    now = datetime.now(timezone.utc)
    rows = []

    for item in videos:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        ch_id = snippet.get("channelId", "")
        ch_stats = channels.get(ch_id, {}).get("statistics", {})

        title = snippet.get("title", "")
        pub_str = snippet.get("publishedAt", "")
        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        age_days = max((now - pub_dt).total_seconds() / 86400.0, 0.04)

        dur_sec = parse_duration_seconds(details.get("duration", ""))
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        subs = int(ch_stats.get("subscriberCount", 0))

        tier = (
            "Large (500K+)"
            if subs >= 500_000
            else "Mid (50K-500K)"
            if subs >= 50_000
            else "Small (<50K)"
        )

        rows.append(
            {
                "Video_ID": item["id"],
                "Tag": tag,
                "Title": title,
                "Description": snippet.get("description", "")[:250],
                "Channel_ID": ch_id,
                "Channel_Title": snippet.get("channelTitle", ""),
                "Subscriber_Count": subs,
                "Channel_Tier": tier,
                "Published_At": pub_str,
                "Published_Date": pub_dt.strftime("%Y-%m-%d"),
                "Age_Days": round(age_days, 2),
                "Duration_Sec": dur_sec,
                "Duration_Min": round(dur_sec / 60.0, 2),
                "View_Count": views,
                "Like_Count": likes,
                "Comment_Count": comments,
                "Views_Per_Day": round(views / age_days, 1),
                "View_to_Sub_Ratio": round(views / subs, 3) if subs > 0 else np.nan,
                "Like_Rate_Pct": round((likes / views) * 100, 3) if views > 0 else 0.0,
                "Comment_Rate_Pct": round((comments / views) * 100, 3) if views > 0 else 0.0,
                "Has_Question": "?" in title,
                "Has_Number": bool(re.search(r"\b\d+\b", title)),
                "Has_Breaking_Hook": bool(BREAKING_RE.search(title)),
                "Has_Controversy_Word": bool(CONTROVERSY_RE.search(title)),
                "Thumbnail_URL": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            }
        )
    return rows


# ---------------------------------------------------------
# ACT 1: The Packaging Formula (~500 Videos, Last 45 Days)
# ---------------------------------------------------------
def run_act1_packaging_data(youtube, refresh: bool = False, with_onnx: bool = False) -> pd.DataFrame:
    print("Running Act 1: Packaging Formula (~500 Videos)...")
    published_after = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = ["gaming news", "game industry news", "software news", "AI in games"]

    def _fetch():
        all_videos = []
        all_channels = {}
        for q in queries:
            res = search_paginated(
                youtube,
                query=q,
                published_after=published_after,
                order="relevance",
                max_pages=3,
            )
            vids, ch_map = hydrate_videos_and_channels(
                youtube, [item["id"]["videoId"] for item in res["items"]]
            )
            all_channels.update(ch_map)
            for v in vids:
                v["_tag"] = q
                all_videos.append(v)
        return {"videos": all_videos, "channels": all_channels}

    raw = cached_fetch(RAW_DIR / "act1_raw.json", refresh, _fetch)

    seen_ids = set()
    deduped = []
    for v in raw["videos"]:
        if v["id"] not in seen_ids:
            seen_ids.add(v["id"])
            deduped.append(v)

    df = pd.DataFrame(build_video_rows(deduped, raw["channels"]))
    df = df[(df["Duration_Sec"] >= 65) & (df["View_Count"] >= 500)].copy()
    df = df.sort_values(by="View_Count", ascending=False).reset_index(drop=True)

    if with_onnx and not df.empty:
        print(f"  [onnx] Classifying formats & title toxicity across {len(df)} videos...")
        prompts = [f"{t} — {d}" for t, d in zip(df["Title"], df["Description"])]
        df["Video_Format"] = classify_zero_shot_onnx(
            prompts, ["News Recap", "Opinion or Rant", "Developer Interview"]
        )
        tox, is_tox, emo = score_texts_with_sociomancy_onnx(df["Title"].tolist())
        df["Title_Toxicity"] = tox
        df["Is_Toxic_Title"] = is_tox
        df["Title_Emotion"] = emo

    out_csv = PROCESSED_DIR / "act1_packaging_formula.csv"
    df.to_csv(out_csv, index=False)
    print(f"Act 1 Data ({len(df)} videos) saved to '{out_csv}'\n")
    return df


# ---------------------------------------------------------
# ACT 2: The Comment Engine (Comment Rates & Resonance Delta)
# ---------------------------------------------------------
def run_act2_comment_engine(youtube, refresh: bool = False, with_onnx: bool = False):
    print("Running Act 2: Comment Engine...")
    act1_path = PROCESSED_DIR / "act1_packaging_formula.csv"
    if not act1_path.exists():
        print("Run Act 1 first to generate the 500-video dataset.")
        return

    df_videos = pd.read_csv(act1_path)
    df_videos["View_Count"] = pd.to_numeric(df_videos["View_Count"], errors="coerce").fillna(0)
    df_videos["Comment_Count"] = pd.to_numeric(df_videos["Comment_Count"], errors="coerce").fillna(0)

    df_videos["Comment_Rate_Pct"] = np.where(
        df_videos["View_Count"] > 0,
        (df_videos["Comment_Count"] / df_videos["View_Count"]) * 100,
        0.0,
    ).round(3)

    df_sorted = df_videos.sort_values(by="Comment_Rate_Pct", ascending=False).reset_index(drop=True)
    rates_csv = PROCESSED_DIR / "act2_comment_rates.csv"
    df_sorted.to_csv(rates_csv, index=False)

    # Pull top 40 comments across top 12 highest-comment-rate videos (>= 10K views)
    top_candidates = df_sorted[df_sorted["View_Count"] >= 10_000].head(12)

    def _fetch_comments():
        comments = []
        for row in top_candidates.itertuples():
            try:
                resp = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=row.Video_ID,
                        order="relevance",
                        maxResults=40,
                        textFormat="plainText",
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    c = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(
                        {
                            "Comment_ID": item["id"],
                            "Video_ID": row.Video_ID,
                            "Video_Title": row.Title,
                            "Video_Format": getattr(row, "Video_Format", "Unclassified"),
                            "Author": c.get("authorDisplayName", ""),
                            "Text": c.get("textDisplay", ""),
                            "Likes": int(c.get("likeCount", 0)),
                        }
                    )
            except HttpError as err:
                print(f"  [skip] Could not fetch comments for {row.Video_ID}: {err.reason}")
        return comments

    raw_comments = cached_fetch(RAW_DIR / "act2_comments_raw.json", refresh, _fetch_comments)
    cdf = pd.DataFrame(raw_comments)

    # Sync latest Video_Format from Act 1 in case ONNX was run after initial comment fetch
    if not cdf.empty and "Video_Format" in df_sorted.columns:
        fmt_map = df_sorted.set_index("Video_ID")["Video_Format"].to_dict()
        cdf["Video_Format"] = cdf["Video_ID"].map(fmt_map).fillna("Unclassified")

    if with_onnx and not cdf.empty:
        print(f"  [onnx] Running SociomancyONNXPipeline on {len(cdf)} top comments...")
        tox, is_tox, emo = score_texts_with_sociomancy_onnx(cdf["Text"].tolist())
        cdf["Toxicity_Score"] = tox
        cdf["Is_Toxic"] = is_tox
        cdf["Top_Emotion"] = emo

        print("  [onnx] Categorizing comment intent (Supportive / Critical / Industry)...")
        cdf["Comment_Category"] = classify_zero_shot_onnx(
            cdf["Text"].tolist(),
            ["Supportive", "Critical or Outraged", "Industry or Developer Insight"],
        )

        # Compute Pillar 1 Resonance Toxicity Delta (ΔR_tox) grouped by Video Format
        summary_rows = []
        for fmt, grp in cdf.groupby("Video_Format"):
            baseline_avg = float(grp["Toxicity_Score"].mean())
            baseline_pct = float(grp["Is_Toxic"].mean() * 100)
            top_per_vid = (
                grp.sort_values(["Video_ID", "Likes"], ascending=[True, False])
                .groupby("Video_ID")
                .head(3)
            )
            top_avg = float(top_per_vid["Toxicity_Score"].mean())
            summary_rows.append(
                {
                    "Video_Format": fmt,
                    "Comments_Analyzed": len(grp),
                    "Baseline_Toxicity_Pct": round(baseline_pct, 2),
                    "Baseline_Avg_Toxicity": round(baseline_avg, 4),
                    "Top3_Avg_Toxicity": round(top_avg, 4),
                    "Resonance_Delta_Toxicity": round(top_avg - baseline_avg, 4),
                }
            )
        summary_csv = PROCESSED_DIR / "act2_format_toxicity_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"  [saved] Format toxicity & ΔR_tox summary written to '{summary_csv}'")

    comments_csv = PROCESSED_DIR / "act2_top_comments_sample.csv"
    cdf.to_csv(comments_csv, index=False)
    print(f"Act 2 Data saved to '{rates_csv}' and '{comments_csv}'\n")


# ---------------------------------------------------------
# ACT 3: The AI Narrative (12-Month Trend & Toxicity Overlay)
# ---------------------------------------------------------
def run_act3_ai_narrative(youtube, refresh: bool = False, with_onnx: bool = False):
    print("Running Act 3: AI Narrative with Trendspy + YouTube...")
    now = datetime.now(timezone.utc)

    # 1. Google Web & YouTube Search Interest via trendspy
    def _fetch_trends():
        tr = Trends()
        kw = ["AI in games", "AI game dev"]
        web_df = tr.interest_over_time(kw, timeframe="today 12-m").drop(
            columns=["isPartial"], errors="ignore"
        )
        time.sleep(1.5)
        yt_df = tr.interest_over_time(kw, timeframe="today 12-m", gprop="youtube").drop(
            columns=["isPartial"], errors="ignore"
        )
        combined = web_df.add_suffix(" (Web)").join(yt_df.add_suffix(" (YouTube)"), how="outer")
        combined.index.name = "Date"
        combined = combined.reset_index()
        combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime("%Y-%m-%d")
        return combined.to_dict(orient="records")

    try:
        trend_rows = cached_fetch(RAW_DIR / "act3_trends_raw.json", refresh, _fetch_trends)
        pd.DataFrame(trend_rows).to_csv(PROCESSED_DIR / "act3_ai_search_trends.csv", index=False)
    except Exception as exc:
        print(f"  [warn] Trendspy request failed ({exc}); continuing with YouTube data.")

    # 2. Monthly YouTube Upload Volume Estimate + Video Sentiment/Toxicity
    def _fetch_monthly_ai():
        buckets = []
        for i in range(12):
            end_dt = now - timedelta(days=30 * i)
            start_dt = now - timedelta(days=30 * (i + 1))
            res = search_paginated(
                youtube,
                query='"AI in games" OR "AI game dev"',
                published_after=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                published_before=end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                order="viewCount",
                max_pages=1,
            )
            vids, ch_map = hydrate_videos_and_channels(
                youtube, [item["id"]["videoId"] for item in res["items"]]
            )
            buckets.append(
                {
                    "month_label": start_dt.strftime("%Y-%m"),
                    "month_offset": f"-{i} Months",
                    "total_results_estimate": res["total_results"],
                    "videos": vids,
                    "channels": ch_map,
                }
            )
        return buckets

    raw_months = cached_fetch(RAW_DIR / "act3_youtube_raw.json", refresh, _fetch_monthly_ai)

    all_ai_videos = []
    for b in raw_months:
        v_rows = build_video_rows(b["videos"], b["channels"], tag=b["month_label"])
        for r in v_rows:
            r["Month"] = b["month_label"]
            r["Month_Offset"] = b["month_offset"]
            r["Month_Upload_Estimate"] = b["total_results_estimate"]
        all_ai_videos.extend(v_rows)

    ai_videos_df = pd.DataFrame(all_ai_videos)

    if not ai_videos_df.empty:
        if with_onnx:
            print(f"  [onnx] Scoring {len(ai_videos_df)} AI video titles with SociomancyONNXPipeline...")
            tox, is_tox, emo = score_texts_with_sociomancy_onnx(ai_videos_df["Title"].tolist())
            ai_videos_df["Title_Toxicity"] = tox
            ai_videos_df["Is_Toxic_Title"] = is_tox
            ai_videos_df["Title_Emotion"] = emo

            ai_videos_df["Sentiment_Bucket"] = np.where(
                ai_videos_df["Is_Toxic_Title"]
                | ai_videos_df["Title_Emotion"].isin(NEGATIVE_EMOTIONS)
                | ai_videos_df["Has_Controversy_Word"],
                "Negative / Outrage",
                np.where(
                    ai_videos_df["Title_Emotion"].isin(POSITIVE_EMOTIONS),
                    "Positive / Optimistic",
                    "Neutral / Informational",
                ),
            )
        else:
            ai_videos_df["Title_Toxicity"] = np.nan
            ai_videos_df["Is_Toxic_Title"] = False
            ai_videos_df["Title_Emotion"] = "unscored"
            ai_videos_df["Sentiment_Bucket"] = np.where(
                ai_videos_df["Has_Controversy_Word"],
                "Negative / Outrage",
                "Neutral / Informational",
            )

    monthly_summary = []
    for b in raw_months:
        m_label = b["month_label"]
        mdf = ai_videos_df[ai_videos_df["Month"] == m_label] if not ai_videos_df.empty else pd.DataFrame()
        monthly_summary.append(
            {
                "Month": m_label,
                "Month_Offset": b["month_offset"],
                "Upload_Volume_Estimate": b["total_results_estimate"],
                "Sampled_Videos": len(mdf),
                "Avg_Views": round(mdf["View_Count"].mean(), 1) if not mdf.empty else 0,
                "Median_Views": round(mdf["View_Count"].median(), 1) if not mdf.empty else 0,
                "Avg_Title_Toxicity": (
                    round(float(mdf["Title_Toxicity"].mean()), 4)
                    if (not mdf.empty and mdf["Title_Toxicity"].notna().any())
                    else 0.0
                ),
                "Negative_Title_Pct": (
                    round((mdf["Sentiment_Bucket"] == "Negative / Outrage").mean() * 100, 1)
                    if not mdf.empty
                    else 0.0
                ),
            }
        )

    ai_videos_df.to_csv(PROCESSED_DIR / "act3_ai_videos_detailed.csv", index=False)
    pd.DataFrame(monthly_summary).sort_values("Month").to_csv(
        PROCESSED_DIR / "act3_ai_youtube_volume.csv", index=False
    )
    print("Act 3 Data saved to 'act3_ai_search_trends.csv' and 'act3_ai_youtube_volume.csv'\n")


# ---------------------------------------------------------
# ACT 4: How Fast Do Controversies Die? (Decay Curves + GDELT)
# ---------------------------------------------------------
def fetch_gdelt_daily_volume(query: str, start_dt: datetime, days: int) -> dict[int, int]:
    """Queries free GDELT 2.0 DOC API for daily news-article counts (0 API keys required)."""
    end_dt = start_dt + timedelta(days=days)
    params = {
        "query": f'"{query}"',
        "mode": "TimelineVolRaw",
        "startdatetime": start_dt.strftime("%Y%m%d000000"),
        "enddatetime": end_dt.strftime("%Y%m%d235959"),
        "format": "json",
    }
    resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=12)
    resp.raise_for_status()
    timeline = resp.json().get("timeline", [{}])[0].get("data", [])
    return {idx: int(pt.get("value", 0)) for idx, pt in enumerate(timeline)}


def run_act4_decay_curves(youtube, refresh: bool = False):
    print("Running Act 4: Controversy Decay Curves...")
    now = datetime.now(timezone.utc)

    controversies = {
        "GTA 6 Launch": {
            "query": "GTA 6",
            "start": "2026-11-19T00:00:00Z",
            "type": "Player / Launch",
        },
        "Binding of Isaac AI": {
            "query": "Binding of Isaac AI",
            "start": "2026-11-19T00:00:00Z",
            "type": "Player Outrage",
        },
        "EU Digital Fairness": {
            "query": "EU Digital Fairness Act games",
            "start": "2026-11-11T00:00:00Z",
            "type": "Industry Regulation",
        },
        "Gamescom 2026 Backlash": {
            "query": "Gamescom apology",
            "start": "2026-09-05T00:00:00Z",
            "type": "Industry Outrage",
        },
        "Steam UK Lawsuit": {
            "query": "Valve Steam lawsuit",
            "start": "2026-01-26T00:00:00Z",
            "type": "Industry Regulation",
        },
    }

    def _fetch_decay():
        events_payload = {}
        for event, cfg in controversies.items():
            peak_date = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            days_payload = []

            for day in range(14):
                day_start = peak_date + timedelta(days=day)
                if day_start >= now:
                    break
                day_end = min(day_start + timedelta(days=1), now)

                res = search_paginated(
                    youtube,
                    query=cfg["query"],
                    published_after=day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    published_before=day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    order="viewCount",
                    max_pages=1,
                )
                vids, ch_map = hydrate_videos_and_channels(
                    youtube, [item["id"]["videoId"] for item in res["items"]]
                )
                days_payload.append(
                    {
                        "day_offset": day,
                        "date": day_start.strftime("%Y-%m-%d"),
                        "total_results_estimate": res["total_results"],
                        "videos": vids,
                        "channels": ch_map,
                    }
                )

            try:
                gdelt_counts = fetch_gdelt_daily_volume(cfg["query"], peak_date, len(days_payload))
            except Exception as exc:
                print(f"  [warn] GDELT lookup skipped for {event}: {exc}")
                gdelt_counts = {}

            events_payload[event] = {
                "type": cfg["type"],
                "days": days_payload,
                "gdelt": gdelt_counts,
            }
        return events_payload

    raw_decay = cached_fetch(RAW_DIR / "act4_decay_raw.json", refresh, _fetch_decay)

    decay_rows = []
    for event, payload in raw_decay.items():
        gdelt_map = {int(k): v for k, v in payload.get("gdelt", {}).items()}
        for d in payload["days"]:
            vdf = pd.DataFrame(build_video_rows(d["videos"], d["channels"], tag=event))
            small_share = (
                round((vdf["Channel_Tier"] == "Small (<50K)").mean() * 100, 1)
                if not vdf.empty
                else 0.0
            )
            decay_rows.append(
                {
                    "Event": event,
                    "Topic_Type": payload["type"],
                    "Day_Offset": d["day_offset"],
                    "Date": d["date"],
                    "Upload_Volume_Estimate": d["total_results_estimate"],
                    "Sampled_Videos": len(vdf),
                    "Total_Sampled_Views": int(vdf["View_Count"].sum()) if not vdf.empty else 0,
                    "Avg_Views_Per_Video": round(vdf["View_Count"].mean(), 1) if not vdf.empty else 0,
                    "Small_Channel_Pct": small_share,
                    "GDELT_News_Articles": gdelt_map.get(d["day_offset"], 0),
                }
            )

    out_csv = PROCESSED_DIR / "act4_decay_curves.csv"
    pd.DataFrame(decay_rows).to_csv(out_csv, index=False)
    print(f"Act 4 Data saved to '{out_csv}'\n")


# ---------------------------------------------------------
# Export Combined JSON for Sociomancy ECharts Templates
# ---------------------------------------------------------
def export_echarts_bundle():
    bundle = {}
    for csv_file in sorted(PROCESSED_DIR.glob("*.csv")):
        bundle[csv_file.stem] = pd.read_csv(csv_file).fillna(0).to_dict(orient="records")
    out_json = PROCESSED_DIR / "echarts_payload.json"
    out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Combined ECharts payload saved to '{out_json}'")


def main():
    parser = argparse.ArgumentParser(description="Sociomancy YouTube Algorithm & Trend Analyzer")
    parser.add_argument("--act", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--refresh", action="store_true", help="Bypass cache and re-query APIs")
    parser.add_argument(
        "--with-onnx",
        action="store_true",
        help="Run SociomancyONNXPipeline (toxicity + GoEmotions) & zero-shot classifiers",
    )
    args = parser.parse_args()

    print("Starting Sociomancy YouTube Data Pull...\n")
    youtube = get_youtube_client()

    if args.act in ("1", "all"):
        run_act1_packaging_data(youtube, refresh=args.refresh, with_onnx=args.with_onnx)
    if args.act in ("2", "all"):
        run_act2_comment_engine(youtube, refresh=args.refresh, with_onnx=args.with_onnx)
    if args.act in ("3", "all"):
        run_act3_ai_narrative(youtube, refresh=args.refresh, with_onnx=args.with_onnx)
    if args.act in ("4", "all"):
        run_act4_decay_curves(youtube, refresh=args.refresh)

    export_echarts_bundle()
    print("All data extraction complete. Ready for visualization.")


if __name__ == "__main__":
    main()