"""Run the local ONNX models on text collected for a study and write the chart tables.

Toxicity and GoEmotions run on every new row. bart-large-mnli runs only on
negative rows, because that model is one text at a time. Already scored
item_ids are skipped unless refresh=True. Complaint hypotheses come from the
study JSON.
"""

from __future__ import annotations

import pandas as pd

from sociomancy.studies.config import Study
from sociomancy.text_scores import (
    classify_complaints,
    score_texts_with_sociomancy_onnx,
    sentiment_bucket,
)

TEXT_FILES = {
    "youtube": "youtube_comments.csv",
    "metacritic": "metacritic_reviews.csv",
    "reddit": "reddit_items.csv",
}
REQUIRED = [
    "item_id",
    "source",
    "kind",
    "text",
    "engagement",
    "created_at",
    "day",
    "window",
    "group",
    "thread_id",
]


def _load_texts(study: Study) -> pd.DataFrame:
    frames = []
    for name, filename in TEXT_FILES.items():
        path = study.processed_dir / filename
        if not path.exists():
            print(f"  [skip] {path} not found. Run collect for {name} first.")
            continue
        frame = pd.read_csv(path)
        missing = [col for col in REQUIRED if col not in frame.columns]
        if missing:
            raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["item_id"] = combined["item_id"].astype(str)
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"] != ""].drop_duplicates(subset=["item_id"])
    combined["engagement"] = pd.to_numeric(combined["engagement"], errors="coerce").fillna(0)
    return combined.reset_index(drop=True)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _mark_top(study: Study, frame: pd.DataFrame) -> pd.DataFrame:
    """Flag upvoted rows: top N per thread, or the top fraction of a Metacritic platform."""
    frame = frame.copy()
    frame["is_top"] = False
    if frame.empty:
        return frame
    for (_, _), part in frame.groupby(["source", "thread_id"], dropna=False):
        if part.empty:
            continue
        if str(part["source"].iloc[0]) == "metacritic":
            take = max(int(round(len(part) * study.metacritic_top_fraction)), 1)
        else:
            take = min(study.top_per_thread, len(part))
        frame.loc[part.nlargest(take, "engagement").index, "is_top"] = True
    return frame


def _needs_complaint(study: Study, row: pd.Series) -> bool:
    if row.get("sentiment") == "Negative":
        return True
    if row.get("source") == "metacritic":
        try:
            return float(row.get("user_score")) <= study.negative_score_at_most
        except (TypeError, ValueError):
            return False
    return False


def _write_summaries(study: Study, frame: pd.DataFrame) -> None:
    scored = frame.copy()
    scored["is_toxic"] = _as_bool(scored["is_toxic"])
    scored["is_top"] = _as_bool(scored["is_top"]) if "is_top" in scored.columns else False

    daily = (
        scored.groupby(["source", "day", "window"], dropna=False)
        .agg(
            texts=("item_id", "size"),
            avg_toxicity=("toxicity_score", "mean"),
            toxic_pct=("is_toxic", "mean"),
            positive_pct=("sentiment", lambda s: (s == "Positive").mean()),
            neutral_pct=("sentiment", lambda s: (s == "Neutral").mean()),
            negative_pct=("sentiment", lambda s: (s == "Negative").mean()),
        )
        .reset_index()
    )
    for col in ("avg_toxicity", "toxic_pct", "positive_pct", "neutral_pct", "negative_pct"):
        if col.endswith("_pct"):
            daily[col] = (daily[col] * 100).round(2)
        else:
            daily[col] = daily[col].round(4)
    daily["avg_toxicity_pct"] = (daily["avg_toxicity"] * 100).round(2)
    daily_path = study.processed_dir / "daily_by_source.csv"
    daily.to_csv(daily_path, index=False)

    youtube_daily = daily[daily["source"] == "youtube"][
        [
            "day",
            "window",
            "texts",
            "avg_toxicity",
            "avg_toxicity_pct",
            "toxic_pct",
            "positive_pct",
            "neutral_pct",
            "negative_pct",
        ]
    ].rename(columns={"texts": "comments"})
    trends_path = study.processed_dir / "google_trends.csv"
    if trends_path.exists():
        trends = pd.read_csv(trends_path)
        keep = [col for col in ("date", "interest", "window") if col in trends.columns]
        trends = trends[keep].rename(columns={"date": "day", "window": "trends_window"})
        if youtube_daily.empty:
            youtube_daily = trends.rename(columns={"trends_window": "window"})
        else:
            youtube_daily = youtube_daily.merge(trends, on="day", how="outer")
            if "trends_window" in youtube_daily.columns:
                youtube_daily["window"] = youtube_daily["window"].fillna(youtube_daily["trends_window"])
                youtube_daily = youtube_daily.drop(columns=["trends_window"])
    timeline_path = study.processed_dir / "act1_youtube_timeline.csv"
    youtube_daily.sort_values("day").to_csv(timeline_path, index=False)
    comment_path = _write_comment_timeline(study, scored)
    same_video_path = _write_same_video(study, scored)

    launch = scored[(scored["source"] == "youtube") & (scored["window"] == study.focus_window)]
    tier_rows = []
    if not launch.empty:
        for tier, part in launch.groupby("group"):
            top = (
                part.sort_values(["thread_id", "engagement"], ascending=[True, False])
                .groupby("thread_id", dropna=False)
                .head(study.top_per_thread)
            )
            baseline = float(part["toxicity_score"].mean())
            top_avg = float(top["toxicity_score"].mean()) if not top.empty else baseline
            tier_rows.append(
                {
                    "channel_tier": tier,
                    "comments": len(part),
                    "videos": part["thread_id"].nunique(),
                    "baseline_avg_toxicity": round(baseline, 4),
                    "baseline_toxic_pct": round(float(part["is_toxic"].mean()) * 100, 2),
                    "top_avg_toxicity": round(top_avg, 4),
                    "resonance_delta_toxicity": round(top_avg - baseline, 4),
                    "positive_pct": round(float((part["sentiment"] == "Positive").mean()) * 100, 2),
                    "neutral_pct": round(float((part["sentiment"] == "Neutral").mean()) * 100, 2),
                    "negative_pct": round(float((part["sentiment"] == "Negative").mean()) * 100, 2),
                }
            )
    tier_path = study.processed_dir / "act2_channel_tiers.csv"
    pd.DataFrame(tier_rows).to_csv(tier_path, index=False)

    complaint_rows = []
    if "complaint_category" in scored.columns:
        negative = scored[scored["complaint_category"].fillna("").astype(str).str.len() > 0]
        for source, part in negative.groupby("source"):
            for scope, subset in (
                ("all_negative", part),
                ("top_engagement", part[part["is_top"]]),
            ):
                total = len(subset)
                if total == 0:
                    continue
                counts = subset["complaint_category"].value_counts()
                for category, count in counts.items():
                    complaint_rows.append(
                        {
                            "source": source,
                            "scope": scope,
                            "complaint_category": category,
                            "texts": int(count),
                            "share_pct": round(count / total * 100, 2),
                        }
                    )
    complaints_path = study.processed_dir / "act3_complaints.csv"
    pd.DataFrame(complaint_rows).to_csv(complaints_path, index=False)
    print(f"  [saved] {daily_path}")
    print(f"  [saved] {timeline_path}")
    print(f"  [saved] {comment_path}")
    print(f"  [saved] {same_video_path}")
    print(f"  [saved] {tier_path}")
    print(f"  [saved] {complaints_path}")


def _sentiment_daily(frame: pd.DataFrame, day_col: str) -> pd.DataFrame:
    daily = (
        frame.groupby(day_col, dropna=False)
        .agg(
            comments=("item_id", "size"),
            avg_toxicity=("toxicity_score", "mean"),
            toxic_pct=("is_toxic", "mean"),
            positive_pct=("sentiment", lambda s: (s == "Positive").mean()),
            neutral_pct=("sentiment", lambda s: (s == "Neutral").mean()),
            negative_pct=("sentiment", lambda s: (s == "Negative").mean()),
        )
        .reset_index()
        .rename(columns={day_col: "day"})
    )
    for col in ("toxic_pct", "positive_pct", "neutral_pct", "negative_pct"):
        daily[col] = (daily[col] * 100).round(2)
    daily["avg_toxicity"] = daily["avg_toxicity"].round(4)
    daily["avg_toxicity_pct"] = (daily["avg_toxicity"] * 100).round(2)
    return daily


def _join_trends(study: Study, daily: pd.DataFrame) -> pd.DataFrame:
    trends_path = study.processed_dir / "google_trends.csv"
    if not trends_path.exists():
        return daily
    trends = pd.read_csv(trends_path).rename(columns={"date": "day", "window": "trends_window"})
    merged = daily.merge(trends, on="day", how="outer")
    if "trends_window" in merged.columns:
        if "window" in merged.columns:
            merged["window"] = merged["window"].fillna(merged["trends_window"])
        else:
            merged["window"] = merged["trends_window"]
        merged = merged.drop(columns=["trends_window"])
    return merged


def _write_comment_timeline(study: Study, scored: pd.DataFrame):
    """YouTube daily stats by the day the comment was posted."""
    path = study.processed_dir / "act1_comment_timeline.csv"
    youtube = scored[scored["source"] == "youtube"].copy()
    if youtube.empty or "comment_published_at" not in youtube.columns:
        pd.DataFrame().to_csv(path, index=False)
        return path
    youtube["comment_day"] = pd.to_datetime(
        youtube["comment_published_at"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    youtube = youtube[youtube["comment_day"].notna() & (youtube["comment_day"] != "NaT")]
    daily = _sentiment_daily(youtube, "comment_day")
    daily["window"] = daily["day"].map(study.window_for)
    daily = _join_trends(study, daily)
    daily.sort_values("day").to_csv(path, index=False)
    return path


def _write_same_video(study: Study, scored: pd.DataFrame):
    """Comments on pre-launch videos, split by whether they were written after launch."""
    path = study.processed_dir / "act1_same_video.csv"
    focus = next(window for window in study.windows if window.name == study.focus_window)
    cutoff = focus.start.isoformat()
    youtube = scored[scored["source"] == "youtube"].copy()
    pre_names = {window.name for window in study.windows if window.end < focus.start}
    videos = youtube[youtube["window"].isin(pre_names)].copy()
    if videos.empty or "comment_published_at" not in videos.columns:
        pd.DataFrame().to_csv(path, index=False)
        return path
    videos["comment_day"] = pd.to_datetime(
        videos["comment_published_at"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    videos = videos[videos["comment_day"].notna()]
    videos["after_launch"] = videos["comment_day"] >= cutoff
    both = int(videos.groupby("thread_id")["after_launch"].nunique().ge(2).sum())
    rows = []
    for after, label in ((False, "before_launch"), (True, "launch_and_after")):
        part = videos[videos["after_launch"] == after]
        if part.empty:
            continue
        rows.append(
            {
                "slice": label,
                "comments": int(len(part)),
                "videos": int(part["thread_id"].nunique()),
                "avg_toxicity": round(float(part["toxicity_score"].mean()), 4),
                "avg_toxicity_pct": round(float(part["toxicity_score"].mean()) * 100, 2),
                "positive_pct": round(float((part["sentiment"] == "Positive").mean()) * 100, 2),
                "neutral_pct": round(float((part["sentiment"] == "Neutral").mean()) * 100, 2),
                "negative_pct": round(float((part["sentiment"] == "Negative").mean()) * 100, 2),
                "videos_with_both": both,
                "prelaunch_videos": int(videos["thread_id"].nunique()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def score_collected(study: Study, refresh: bool = False, complaints: bool = True) -> pd.DataFrame:
    print(f"Scoring {study.title} text with the local ONNX models...")
    study.ensure_dirs()
    incoming = _load_texts(study)
    if incoming.empty:
        print("  [warn] Nothing to score. Collect YouTube, Metacritic, or Reddit first.")
        return incoming

    scored_path = study.processed_dir / "scored_texts.csv"
    previous = pd.DataFrame()
    if scored_path.exists() and not refresh:
        previous = pd.read_csv(scored_path).drop_duplicates(subset=["item_id"], keep="last")
        previous["item_id"] = previous["item_id"].astype(str)
        done = set(previous["item_id"])
        pending = incoming[~incoming["item_id"].isin(done)].copy()
        print(f"  [cache] {len(done & set(incoming['item_id']))} texts already scored; {len(pending)} new.")
    else:
        pending = incoming.copy()

    if not pending.empty:
        print(f"  [onnx] Toxicity and emotion on {len(pending)} texts...")
        tox, is_toxic, emotions = score_texts_with_sociomancy_onnx(pending["text"].tolist())
        pending["toxicity_score"] = tox
        pending["is_toxic"] = is_toxic
        pending["top_emotion"] = emotions
        pending["sentiment"] = [
            sentiment_bucket(emotion, toxic) for emotion, toxic in zip(emotions, is_toxic)
        ]
        pending["complaint_category"] = ""

    if previous.empty:
        combined = pending
    elif pending.empty:
        combined = previous
    else:
        previous = previous[~previous["item_id"].isin(set(pending["item_id"]))]
        combined = pd.concat([previous, pending], ignore_index=True)

    # Keep the latest collector columns for ids we just loaded, without dropping scores.
    score_cols = ["toxicity_score", "is_toxic", "top_emotion", "sentiment", "complaint_category"]
    fresh = incoming.merge(
        combined[["item_id", *[col for col in score_cols if col in combined.columns]]],
        on="item_id",
        how="left",
    )
    combined = _mark_top(study, fresh)

    if complaints:
        if not study.complaint_labels:
            print("  [skip] No complaints in the study config.")
        else:
            if "complaint_category" not in combined.columns:
                combined["complaint_category"] = ""
            combined["complaint_category"] = combined["complaint_category"].fillna("").astype(str)
            mask = combined.apply(lambda row: _needs_complaint(study, row), axis=1)
            mask = mask & (combined["complaint_category"] == "")
            targets = combined.index[mask].tolist()
            if targets:
                print(f"  [onnx] Complaint labels on {len(targets)} negative texts...")
                labels = classify_complaints(combined.loc[targets, "text"].tolist(), study.complaint_labels)
                combined.loc[targets, "complaint_category"] = labels
            else:
                print("  [cache] Complaint labels already present for negative texts.")

    combined.to_csv(scored_path, index=False)
    print(f"  [saved] {scored_path} ({len(combined)} texts)")
    _write_summaries(study, combined)
    return combined
