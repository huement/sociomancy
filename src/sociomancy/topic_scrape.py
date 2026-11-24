"""
Topic-first YouTube & external trend collector for Sociomancy.
Builds datasets for the 4-act gaming/software algorithm video essay:
  Act 1: Packaging formula (last 45 days top videos, duration, title syntax, breakout ratio)
  Act 2: Comment engine (500-video sample, comment rates, zero-shot format & comment tagging)
  Act 3: AI in games narrative (12-month volume, emotion/toxicity split, Google Trends overlay)
  Act 4: Controversy decay curves (daily upload/view velocity, channel tiers, GDELT news volume)
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import isodate
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from trendspy import Trends

load_dotenv()

RAW_DIR = Path("data/raw/topics")
PROCESSED_DIR = Path("data/processed/topics")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration: Queries & 2026 Controversy Windows
# ---------------------------------------------------------------------------
ACT1_QUERIES = [
    "gaming news",
    "game industry news",
    "software news",
    "AI in games",
]

ACT3_AI_QUERY = '"AI in games" OR "AI game dev" OR "generative AI games"'

# Mix of acute player/gaming flashpoints and slower industry/regulation stories
CONTROVERSIES = {
    "GTA 6 Launch": {
        "query": "GTA 6",
        "start_date": "2026-11-19",
        "days": 4,  # Nov 19 - Nov 22, 2026
        "category": "Player / Launch",
    },
    "Binding of Isaac AI Music": {
        "query": "Binding of Isaac AI",
        "start_date": "2026-11-19",
        "days": 4,
        "category": "Player Outrage",
    },
    "EU Digital Fairness Act": {
        "query": "EU Digital Fairness Act games",
        "start_date": "2026-11-11",
        "days": 12,
        "category": "Industry Regulation",
    },
    "EU KIDS Act": {
        "query": "EU KIDS Act games",
        "start_date": "2026-09-29",
        "days": 14,
        "category": "Industry Regulation",
    },
    "Rockstar Union Dispute": {
        "query": "Rockstar Games union",
        "start_date": "2026-09-10",
        "days": 14,
        "category": "Industry / Labor",
    },
}


# ---------------------------------------------------------------------------
# Local ONNX Model Suite (Reuses Sociomancy's Cached Transformers)
# ---------------------------------------------------------------------------
class ONNXSuite:
    """Lazy-loads Sociomancy's local ONNX models via optimum.pipelines."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._zero_shot = None
        self._toxicity = None
        self._emotion = None

    def _load_zero_shot(self):
        if self._zero_shot is None:
            from optimum.pipelines import pipeline

            print("Loading ONNX zero-shot model (facebook/bart-large-mnli)...")
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                accelerator="ort",
            )
        return self._zero_shot

    def _load_toxicity(self):
        if self._toxicity is None:
            from optimum.pipelines import pipeline

            print("Loading ONNX toxicity model (unitary/unbiased-toxic-roberta)...")
            self._toxicity = pipeline(
                "text-classification",
                model="unitary/unbiased-toxic-roberta",
                accelerator="ort",
                truncation=True,
                max_length=512,
            )
        return self._toxicity

    def _load_emotion(self):
        if self._emotion is None:
            from optimum.pipelines import pipeline

            print("Loading ONNX emotion model (SamLowe/roberta-base-go_emotions)...")
            self._emotion = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                accelerator="ort",
                truncation=True,
                max_length=512,
            )
        return self._emotion

    def classify_zero_shot(self, texts: list[str], labels: list[str]) -> list[str]:
        if not self.enabled or not texts:
            return ["Unclassified"] * len(texts)
        clf = self._load_zero_shot()
        results = []
        for text in texts:
            clean = (text or "").strip()[:400]
            if not clean:
                results.append("Unclassified")
                continue
            out = clf(clean, candidate_labels=labels)
            results.append(out["labels"][0])
        return results

    def score_toxicity(self, texts: list[str]) -> list[float]:
        if not self.enabled or not texts:
            return [0.0] * len(texts)
        clf = self._load_toxicity()
        scores = []
        for text in texts:
            clean = (text or "").strip()[:500]
            if not clean:
                scores.append(0.0)
                continue
            out = clf(clean)[0]
            # Handle models returning 'toxicity' vs non-toxic label
            score = out["score"] if "toxic" in out["label"].lower() else 1.0 - out["score"]
            scores.append(round(float(score), 4))
        return scores

    def classify_emotion(self, texts: list[str]) -> list[str]:
        if not self.enabled or not texts:
            return ["neutral"] * len(texts)
        clf = self._load_emotion()
        emotions = []
        for text in texts:
            clean = (text or "").strip()[:500]
            if not clean:
                emotions.append("neutral")
                continue
            out = clf(clean)[0]
            emotions.append(out["label"])
        return emotions


# ---------------------------------------------------------------------------
# YouTube Data API v3 Helpers (Batched & Cached)
# ---------------------------------------------------------------------------
def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY missing from .env")
    return build("youtube", "v3", developerKey=api_key)


def load_or_fetch_json(cache_path: Path, refresh: bool, fetch_fn) -> Any:
    if cache_path.exists() and not refresh:
        print(f"  [cache hit] {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = fetch_fn()
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  [saved raw] {cache_path}")
    return data


def search_videos_paginated(
    youtube,
    query: str,
    published_after: str | None = None,
    published_before: str | None = None,
    order: str = "relevance",
    max_pages: int = 2,
    video_category_id: str | None = None,
) -> dict[str, Any]:
    """Paginates search.list (100 quota units per page) and captures totalResults."""
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
        if video_category_id:
            kwargs["videoCategoryId"] = video_category_id
        if page_token:
            kwargs["pageToken"] = page_token

        resp = youtube.search().list(**kwargs).execute()
        if total_results == 0:
            total_results = resp.get("pageInfo", {}).get("totalResults", 0)
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return {"total_results_estimate": total_results, "items": items}


def hydrate_videos_and_channels(youtube, video_ids: list[str]) -> tuple[list[dict], dict[str, dict]]:
    """Batches videos.list and channels.list in chunks of 50 (1 unit per call)."""
    unique_vids = list(dict.fromkeys(video_ids))
    video_items = []

    for i in range(0, len(unique_vids), 50):
        chunk = unique_vids[i : i + 50]
        resp = (
            youtube.videos()
            .list(part="snippet,contentDetails,statistics", id=",".join(chunk))
            .execute()
        )
        video_items.extend(resp.get("items", []))

    channel_ids = list(
        dict.fromkeys(v["snippet"]["channelId"] for v in video_items if "snippet" in v)
    )
    channels_map: dict[str, dict] = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i : i + 50]
        resp = youtube.channels().list(part="snippet,statistics", id=",".join(chunk)).execute()
        for ch in resp.get("items", []):
            channels_map[ch["id"]] = ch

    return video_items, channels_map


# ---------------------------------------------------------------------------
# Feature Engineering Helpers
# ---------------------------------------------------------------------------
CONTROVERSY_WORDS = re.compile(
    r"\b(dead|killed|ruined|disaster|lawsuit|banned|outrage|backlash|worst|scam|lie|lied|exposed|boycott|slop)\b",
    re.I,
)
BREAKING_HOOK = re.compile(
    r"\b(just happened|breaking|officially|it's over|happening now|announced|confirmed)\b",
    re.I,
)


def parse_video_record(v: dict, channels_map: dict[str, dict], query_tag: str = "") -> dict:
    snippet = v.get("snippet", {})
    stats = v.get("statistics", {})
    details = v.get("contentDetails", {})
    ch_id = snippet.get("channelId", "")
    ch_stats = channels_map.get(ch_id, {}).get("statistics", {})

    title = snippet.get("title", "")
    pub_dt = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
    age_days = max((datetime.now(timezone.utc) - pub_dt).total_seconds() / 86400.0, 0.04)

    try:
        duration_sec = int(isodate.parse_duration(details.get("duration", "PT0S")).total_seconds())
    except Exception:
        duration_sec = 0

    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0))
    comments = int(stats.get("commentCount", 0))
    subs = int(ch_stats.get("subscriberCount", 0))

    return {
        "video_id": v["id"],
        "query_tag": query_tag,
        "title": title,
        "description": snippet.get("description", "")[:300],
        "channel_id": ch_id,
        "channel_title": snippet.get("channelTitle", ""),
        "subscriber_count": subs,
        "channel_tier": (
            "Large (500K+)"
            if subs >= 500_000
            else "Mid (50K-500K)"
            if subs >= 50_000
            else "Small (<50K)"
        ),
        "published_at": snippet.get("publishedAt", ""),
        "published_date": pub_dt.strftime("%Y-%m-%d"),
        "age_days": round(age_days, 2),
        "duration_sec": duration_sec,
        "duration_min": round(duration_sec / 60.0, 2),
        "view_count": views,
        "like_count": likes,
        "comment_count": comments,
        "views_per_day": round(views / age_days, 1),
        "view_to_sub_ratio": round(views / subs, 3) if subs > 0 else np.nan,
        "like_rate_pct": round((likes / views) * 100, 3) if views > 0 else 0.0,
        "comment_rate_pct": round((comments / views) * 100, 3) if views > 0 else 0.0,
        "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        "title_len_chars": len(title),
        "has_question": "?" in title,
        "has_number": bool(re.search(r"\b\d+\b", title)),
        "has_breaking_hook": bool(BREAKING_HOOK.search(title)),
        "has_controversy_word": bool(CONTROVERSY_WORDS.search(title)),
    }


# ---------------------------------------------------------------------------
# ACT 1 & ACT 2: Packaging Formula + 500-Video Comment Engine
# ---------------------------------------------------------------------------
def run_acts_1_and_2(youtube, onnx: ONNXSuite, refresh: bool = False) -> pd.DataFrame:
    print("\n=== ACT 1 & 2: Scraping ~500 Recent Gaming/Software Videos ===")
    published_after = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fetch():
        all_vids = []
        all_channels = {}
        for q in ACT1_QUERIES:
            print(f"  Searching query='{q}' (last 45 days)...")
            res = search_videos_paginated(
                youtube,
                query=q,
                published_after=published_after,
                order="relevance",
                max_pages=3,  # ~150 per query -> ~500-600 raw before dedup
            )
            vids_raw, ch_map = hydrate_videos_and_channels(
                youtube, [i["id"]["videoId"] for i in res["items"]]
            )
            all_channels.update(ch_map)
            for v in vids_raw:
                v["_query_tag"] = q
                all_vids.append(v)
        return {"videos": all_vids, "channels": all_channels}

    raw = load_or_fetch_json(RAW_DIR / "act1_act2_videos_raw.json", refresh, _fetch)

    seen = set()
    rows = []
    for v in raw["videos"]:
        if v["id"] in seen:
            continue
        seen.add(v["id"])
        rec = parse_video_record(v, raw["channels"], query_tag=v.get("_query_tag", ""))
        # Exclude YouTube Shorts (<65s) so runtime scatter reflects long-form videos
        if rec["duration_sec"] >= 65 and rec["view_count"] >= 500:
            rows.append(rec)

    df = pd.DataFrame(rows)

    # Zero-shot classify video format for Act 2 ("News Recap" vs "Opinion/Rant" vs "Dev Interview")
    print(f"  Classifying formats across {len(df)} long-form videos...")
    prompts = [f"{r.title} — {r.description[:150]}" for r in df.itertuples()]
    df["video_format"] = onnx.classify_zero_shot(
        prompts, ["News Recap", "Opinion or Rant", "Developer Interview"]
    )

    # Save Act 1 (sorted by views) & Act 2 (sorted by comment rate)
    df.sort_values("view_count", ascending=False).to_csv(
        PROCESSED_DIR / "act1_packaging.csv", index=False
    )
    df.sort_values("comment_rate_pct", ascending=False).to_csv(
        PROCESSED_DIR / "act2_comment_rates.csv", index=False
    )

    # Pull top 40 comments from the top 12 highest-comment-rate videos (12 quota units)
    top_discussed = (
        df[df["view_count"] >= 10_000]
        .sort_values("comment_rate_pct", ascending=False)
        .head(12)
    )

    def _fetch_comments():
        comment_rows = []
        for row in top_discussed.itertuples():
            try:
                resp = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=row.video_id,
                        order="relevance",
                        maxResults=40,
                        textFormat="plainText",
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    c = item["snippet"]["topLevelComment"]["snippet"]
                    comment_rows.append(
                        {
                            "video_id": row.video_id,
                            "video_title": row.title,
                            "video_format": row.video_format,
                            "comment_id": item["id"],
                            "author": c.get("authorDisplayName", ""),
                            "text": c.get("textDisplay", ""),
                            "like_count": int(c.get("likeCount", 0)),
                        }
                    )
            except Exception as exc:
                print(f"  [warn] comments disabled/failed for {row.video_id}: {exc}")
        return comment_rows

    comments_raw = load_or_fetch_json(
        RAW_DIR / "act2_comments_raw.json", refresh, _fetch_comments
    )
    cdf = pd.DataFrame(comments_raw)
    if not cdf.empty:
        print(f"  Tagging {len(cdf)} top comments with local ONNX models...")
        cdf["comment_category"] = onnx.classify_zero_shot(
            cdf["text"].tolist(),
            ["Supportive", "Critical or Outraged", "Industry or Developer Insight"],
        )
        cdf["toxicity_score"] = onnx.score_toxicity(cdf["text"].tolist())
        cdf.to_csv(PROCESSED_DIR / "act2_comments_classified.csv", index=False)

    print("  [done] Act 1 & Act 2 CSVs written.")
    return df


# ---------------------------------------------------------------------------
# ACT 3: Real State of AI in Games (12-Month Trend & Sentiment)
# ---------------------------------------------------------------------------
def run_act_3(youtube, onnx: ONNXSuite, refresh: bool = False):
    print("\n=== ACT 3: AI in Game Dev (12-Month YouTube + Google Trends) ===")
    now = datetime.now(timezone.utc)

    def _fetch_ai_youtube():
        buckets = []
        for m in range(12):
            end_dt = now - timedelta(days=30 * m)
            start_dt = now - timedelta(days=30 * (m + 1))
            res = search_videos_paginated(
                youtube,
                query=ACT3_AI_QUERY,
                published_after=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                published_before=end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                order="viewCount",
                max_pages=1,
            )
            vids_raw, ch_map = hydrate_videos_and_channels(
                youtube, [i["id"]["videoId"] for i in res["items"]]
            )
            buckets.append(
                {
                    "month_label": start_dt.strftime("%Y-%m"),
                    "total_results_estimate": res["total_results_estimate"],
                    "videos": vids_raw,
                    "channels": ch_map,
                }
            )
        return buckets

    raw_buckets = load_or_fetch_json(RAW_DIR / "act3_ai_raw.json", refresh, _fetch_ai_youtube)

    all_ai_videos = []
    monthly_summary = []

    for b in raw_buckets:
        recs = [parse_video_record(v, b["channels"], query_tag=b["month_label"]) for v in b["videos"]]
        if recs:
            titles = [r["title"] for r in recs]
            tox = onnx.score_toxicity(titles)
            emo = onnx.classify_emotion(titles)
            for r, t_score, e_label in zip(recs, tox, emo):
                r["title_toxicity"] = t_score
                r["title_emotion"] = e_label
                r["sentiment_bucket"] = (
                    "Negative / Outrage"
                    if (
                        t_score >= 0.45
                        or e_label in {"anger", "annoyance", "disapproval", "disgust", "fear"}
                        or r["has_controversy_word"]
                    )
                    else "Positive / Optimistic"
                    if e_label in {"admiration", "excitement", "joy", "optimism", "curiosity"}
                    else "Neutral / Informational"
                )
            all_ai_videos.extend(recs)

        mdf = pd.DataFrame(recs)
        monthly_summary.append(
            {
                "month": b["month_label"],
                "api_upload_estimate": b["total_results_estimate"],
                "sample_count": len(mdf),
                "avg_views": round(mdf["view_count"].mean(), 1) if not mdf.empty else 0,
                "median_views": round(mdf["view_count"].median(), 1) if not mdf.empty else 0,
                "pct_negative_titles": (
                    round((mdf["sentiment_bucket"] == "Negative / Outrage").mean() * 100, 1)
                    if not mdf.empty
                    else 0.0
                ),
            }
        )

    pd.DataFrame(all_ai_videos).to_csv(PROCESSED_DIR / "act3_ai_videos.csv", index=False)
    pd.DataFrame(monthly_summary).sort_values("month").to_csv(
        PROCESSED_DIR / "act3_ai_monthly.csv", index=False
    )

    # External Search Popularity via trendspy (no API key)
    def _fetch_trends():
        tr = Trends()
        time.sleep(2)
        df_tr = tr.interest_over_time(["AI in games", "AI game dev"], timeframe="today 12-m")
        df_tr = df_tr.reset_index()
        df_tr.iloc[:, 0] = df_tr.iloc[:, 0].astype(str)
        return df_tr.to_dict(orient="records")

    try:
        trends_records = load_or_fetch_json(
            RAW_DIR / "act3_trends_raw.json", refresh, _fetch_trends
        )
        pd.DataFrame(trends_records).to_csv(PROCESSED_DIR / "act3_ai_trends.csv", index=False)
    except Exception as exc:
        print(f"  [warn] trendspy rate-limited or failed ({exc}); YouTube data still saved.")

    print("  [done] Act 3 AI CSVs written.")


# ---------------------------------------------------------------------------
# ACT 4: Controversy Decay Curves + GDELT News Volume
# ---------------------------------------------------------------------------
def fetch_gdelt_daily_volume(query: str, start_date_str: str, days: int) -> list[dict]:
    """Queries free GDELT 2.0 DOC API for daily news-article volume."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=days)
    params = {
        "query": f'"{query}"',
        "mode": "TimelineVolRaw",
        "startdatetime": start_dt.strftime("%Y%m%d000000"),
        "enddatetime": end_dt.strftime("%Y%m%d235959"),
        "format": "json",
    }
    resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    timeline = payload.get("timeline", [{}])[0].get("data", [])
    out = []
    for idx, point in enumerate(timeline):
        out.append(
            {
                "day_offset": idx,
                "date": point.get("date", "")[:8],
                "news_article_count": point.get("value", 0),
            }
        )
    return out


def run_act_4(youtube, refresh: bool = False):
    print("\n=== ACT 4: Controversy Decay Curves (YouTube + GDELT News) ===")

    def _fetch_decay():
        all_events = {}
        for event_name, cfg in CONTROVERSIES.items():
            print(f"  Tracking '{event_name}' ({cfg['days']} days from {cfg['start_date']})...")
            base_dt = datetime.strptime(cfg["start_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            daily_buckets = []
            for d in range(cfg["days"]):
                d_start = base_dt + timedelta(days=d)
                d_end = d_start + timedelta(days=1)
                res = search_videos_paginated(
                    youtube,
                    query=cfg["query"],
                    published_after=d_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    published_before=d_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    order="viewCount",
                    max_pages=1,
                )
                vids_raw, ch_map = hydrate_videos_and_channels(
                    youtube, [i["id"]["videoId"] for i in res["items"]]
                )
                daily_buckets.append(
                    {
                        "day_offset": d,
                        "date": d_start.strftime("%Y-%m-%d"),
                        "total_results_estimate": res["total_results_estimate"],
                        "videos": vids_raw,
                        "channels": ch_map,
                    }
                )
            try:
                gdelt = fetch_gdelt_daily_volume(cfg["query"], cfg["start_date"], cfg["days"])
            except Exception as exc:
                print(f"  [warn] GDELT lookup skipped for {event_name}: {exc}")
                gdelt = []

            all_events[event_name] = {
                "category": cfg["category"],
                "daily_youtube": daily_buckets,
                "daily_gdelt": gdelt,
            }
        return all_events

    raw_events = load_or_fetch_json(RAW_DIR / "act4_decay_raw.json", refresh, _fetch_decay)

    decay_rows = []
    for event_name, payload in raw_events.items():
        gdelt_map = {g["day_offset"]: g["news_article_count"] for g in payload.get("daily_gdelt", [])}
        for b in payload["daily_youtube"]:
            recs = [parse_video_record(v, b["channels"], query_tag=event_name) for v in b["videos"]]
            vdf = pd.DataFrame(recs)
            small_share = (
                round((vdf["channel_tier"] == "Small (<50K)").mean() * 100, 1)
                if not vdf.empty
                else 0.0
            )
            decay_rows.append(
                {
                    "event": event_name,
                    "category": payload["category"],
                    "day_offset": b["day_offset"],
                    "date": b["date"],
                    "youtube_upload_estimate": b["total_results_estimate"],
                    "sampled_videos": len(vdf),
                    "total_sampled_views": int(vdf["view_count"].sum()) if not vdf.empty else 0,
                    "avg_views_per_video": round(vdf["view_count"].mean(), 1) if not vdf.empty else 0,
                    "small_channel_pct": small_share,
                    "gdelt_news_articles": gdelt_map.get(b["day_offset"], np.nan),
                }
            )

    pd.DataFrame(decay_rows).to_csv(PROCESSED_DIR / "act4_decay_curves.csv", index=False)
    print("  [done] Act 4 decay CSV written.")


# ---------------------------------------------------------------------------
# Export Combined JSON Payload for Apache ECharts (src/charts/build_charts.py)
# ---------------------------------------------------------------------------
def build_echarts_bundle():
    bundle = {}
    for csv_name in [
        "act1_packaging.csv",
        "act2_comment_rates.csv",
        "act2_comments_classified.csv",
        "act3_ai_monthly.csv",
        "act3_ai_trends.csv",
        "act4_decay_curves.csv",
    ]:
        path = PROCESSED_DIR / csv_name
        if path.exists():
            bundle[csv_name.replace(".csv", "")] = pd.read_csv(path).fillna(0).to_dict(orient="records")

    out_path = PROCESSED_DIR / "echarts_payload.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\n[ready] Combined ECharts payload saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sociomancy Topic & Trend Scraper")
    parser.add_argument("--act", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--refresh", action="store_true", help="Ignore data/raw cache and re-query APIs")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip local ONNX model inference")
    args = parser.parse_args()

    youtube = get_youtube_client()
    onnx = ONNXSuite(enabled=not args.skip_onnx)

    if args.act in ("1", "2", "all"):
        run_acts_1_and_2(youtube, onnx, refresh=args.refresh)
    if args.act in ("3", "all"):
        run_act_3(youtube, onnx, refresh=args.refresh)
    if args.act in ("4", "all"):
        run_act_4(youtube, refresh=args.refresh)

    build_echarts_bundle()


if __name__ == "__main__":
    main()