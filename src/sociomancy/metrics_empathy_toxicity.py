import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


def load_channel_comments(json_path: Path) -> tuple[dict, list[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_comments = []
    for video in data.get("videos", []):
        for c in video.get("comments", []):
            flat_comments.append({
                "video_id": video["video_id"],
                "comment_id": c["comment_id"],
                "text": str(c.get("text", "")),
                "author_id": c["author_id"],
                "like_count": c.get("like_count", 0)
            })
    return data, flat_comments


def process_empathy_and_toxicity(json_path: Path,
                                 top_n_per_video: int = 1) -> dict:
    metadata, raw_comments = load_channel_comments(json_path)
    if not raw_comments:
        return {"error": "No comments found"}

    clean_handle = metadata["channel_handle"]
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / f"{clean_handle}_metrics.csv"

    existing_df = pd.DataFrame()
    cached_comment_ids = set()

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if "comment_id" in existing_df.columns:
            cached_comment_ids = set(existing_df["comment_id"].astype(str))

    uncached_comments = [
        c for c in raw_comments
        if str(c["comment_id"]) not in cached_comment_ids
    ]

    if uncached_comments:
        print(
            f"[*] Found {len(uncached_comments)} new comments requiring ONNX inference..."
        )
        from sociomancy.models_onnx import SociomancyONNXPipeline

        nlp = SociomancyONNXPipeline()
        texts = [c["text"] for c in uncached_comments]
        evaluations = nlp.analyze_comments(texts)

        new_df = pd.DataFrame(uncached_comments)
        new_df["toxicity_score"] = [e["toxicity_score"] for e in evaluations]
        new_df["is_toxic"] = [e["is_toxic"] for e in evaluations]
        new_df["top_emotion"] = [
            e["top_emotions"][0]["label"] if e["top_emotions"] else "neutral"
            for e in evaluations
        ]

        df = pd.concat([existing_df, new_df],
                       ignore_index=True) if not existing_df.empty else new_df
        df.to_csv(csv_path, index=False)
        print(f"[+] Updated CSV cache at {csv_path}")
    else:
        print(
            f"[*] [CACHE HIT] All {len(raw_comments)} comments already processed in CSV. Skipping ONNX inference."
        )
        df = existing_df

    # Standardize types
    df["like_count"] = pd.to_numeric(df["like_count"],
                                     errors="coerce").fillna(0)

    # 1. Baseline Analytics (All Comments)
    total_comments = len(df)
    toxic_count = int(df["is_toxic"].sum())
    toxicity_pct = round(
        (toxic_count / total_comments) * 100, 2) if total_comments > 0 else 0.0
    avg_toxicity = round(float(df["toxicity_score"].mean()),
                         4) if total_comments > 0 else 0.0
    emotion_counts = Counter(df["top_emotion"].fillna("neutral"))

    # 2. Top Comments Analytics (Top N Most Liked Per Video)
    top_df = (df.sort_values(
        by=["video_id", "like_count"],
        ascending=[True, False]).groupby("video_id").head(top_n_per_video))

    total_top_comments = len(top_df)
    top_toxic_count = int(top_df["is_toxic"].sum())
    top_toxicity_pct = round((top_toxic_count / total_top_comments) *
                             100, 2) if total_top_comments > 0 else 0.0
    top_avg_toxicity = round(float(top_df["toxicity_score"].mean()),
                             4) if total_top_comments > 0 else 0.0
    top_emotion_counts = Counter(top_df["top_emotion"].fillna("neutral"))

    # Resonance Delta (Top Comment Toxicity - Baseline Toxicity)
    # A negative delta means community upvotes are active buffers against baseline toxicity
    delta_toxicity = round(top_avg_toxicity - avg_toxicity, 4)

    return {
        "channel_title": metadata["channel_title"],
        "total_comments_analyzed": total_comments,
        "baseline": {
            "toxic_comments_count": toxic_count,
            "toxicity_percentage": toxicity_pct,
            "average_toxicity_score": avg_toxicity,
            "dominant_emotions": emotion_counts.most_common(5),
        },
        "top_comments": {
            "analyzed_count": total_top_comments,
            "toxic_comments_count": top_toxic_count,
            "toxicity_percentage": top_toxicity_pct,
            "average_toxicity_score": top_avg_toxicity,
            "dominant_emotions": top_emotion_counts.most_common(5),
        },
        "resonance_delta_toxicity": delta_toxicity,
        "csv_saved_to": str(csv_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Run Pillar 1 ONNX Empathy/Toxicity model on cached raw JSON.")
    parser.add_argument(
        "handle",
        nargs="?",
        help=
        "Channel handle (e.g., donflurgundy). If omitted, processes all raw files in data/raw/."
    )
    args = parser.parse_args()

    raw_dir = Path("data/raw")

    if args.handle:
        clean_handle = args.handle.lstrip("@").lower()
        target_file = raw_dir / f"{clean_handle}.json"
        if target_file.exists():
            res = process_empathy_and_toxicity(target_file)
            print(f"\n--- PILLAR 1: EMPATHY & TOXICITY (@{clean_handle}) ---")
            print(
                f"Baseline Toxicity: {res['baseline']['toxicity_percentage']}% (Avg: {res['baseline']['average_toxicity_score']})"
            )
            print(
                f"Top-Comment Toxicity: {res['top_comments']['toxicity_percentage']}% (Avg: {res['top_comments']['average_toxicity_score']})"
            )
            print(
                f"Resonance Toxicity Delta (ΔR_tox): {res['resonance_delta_toxicity']:+} points"
            )
            print(
                f"Top-Comment Emotions: {res['top_comments']['dominant_emotions']}"
            )
        else:
            print(f"[!] File {target_file} not found. Run fetcher.py first.")
    else:
        raw_files = list(raw_dir.glob("*.json"))
        if not raw_files:
            print("[!] No raw JSON files found in data/raw/.")
        for f in raw_files:
            print(f"\n[*] Processing {f.name}...")
            res = process_empathy_and_toxicity(f)
            print(
                f"Resonance Toxicity Delta (ΔR_tox): {res.get('resonance_delta_toxicity', 0.0):+} points"
            )
