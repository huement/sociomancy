import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd

from sociomancy.analysis.parasocial import ParasocialConfig, ParasocialDetector
from sociomancy.analysis.models import SociomancyONNXSignalProvider

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "parasocial" / "en.yaml"

def calculate_parasocial_impact(channel_handle: str, top_n_per_video: int = 1) -> dict:
    clean_handle = channel_handle.lstrip("@").lower()
    csv_path = Path(f"data/processed/{clean_handle}_metrics.csv")
    raw_json_path = Path(f"data/raw/{clean_handle}.json")

    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found at {csv_path}. Run Pillar 1 first.")

    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

    # Basic comment statistics
    df["word_count"] = df["text"].str.split().str.len()
    total_comments = len(df)
    avg_word_count = round(float(df["word_count"].mean()), 2) if total_comments else 0.0
    word_count_std = round(float(df["word_count"].std()), 2) if total_comments > 1 else 0.0
    longform_comments_pct = round((len(df[df["word_count"] >= 50]) / total_comments) * 100, 2) if total_comments else 0.0

    # Initialize Detector with ONNX Signal Provider
    config = ParasocialConfig(CONFIG_PATH)
    onnx_provider = SociomancyONNXSignalProvider()
    detector = ParasocialDetector(config, onnx_provider=onnx_provider)

    results = []
    for _, row in df.iterrows():
        emotion_data = {
            "top_emotion": row.get("top_emotion", "neutral"),
            "toxicity_score": row.get("toxicity_score", 0.0)
        }
        res = detector.analyze(row["text"], emotion_data=emotion_data)
        results.append(res)

    # Store detailed results in dataframe
    df["parasocial_score"] = [r.score for r in results]
    df["parasocial_classification"] = [r.classification for r in results]
    df["parasocial_categories"] = [", ".join(r.categories()) for r in results]

    # Save enriched dataframe back to CSV
    df.to_csv(csv_path, index=False)

    # 1. Baseline Analytics (All Comments)
    baseline_avg_score = round(float(df["parasocial_score"].mean()), 4) if total_comments else 0.0
    moderate_or_higher = df[df["parasocial_score"] >= config.thresholds["moderate"]]
    strong_or_higher = df[df["parasocial_score"] >= config.thresholds["strong"]]
    very_strong = df[df["parasocial_score"] >= config.thresholds["very_strong"]]

    parasocial_count = len(moderate_or_higher)
    parasocial_density = round((parasocial_count / total_comments) * 1000, 2) if total_comments else 0.0

    category_counts = {}
    for r in results:
        for cat in r.categories():
            category_counts[cat] = category_counts.get(cat, 0) + 1

    category_density = {
        cat: round((count / total_comments) * 1000, 2)
        for cat, count in category_counts.items()
    }

    # 2. Top Comments Parasocial Analytics (Top N Most Liked Per Video)
    top_df = pd.DataFrame()
    top_avg_score = 0.0
    top_parasocial_density = 0.0
    top_moderate_count = 0
    top_category_counts = {}

    if "video_id" in df.columns and "like_count" in df.columns:
        top_df = (
            df.sort_values(by=["video_id", "like_count"], ascending=[True, False])
            .groupby("video_id")
            .head(top_n_per_video)
        )
        total_top = len(top_df)
        if total_top > 0:
            top_avg_score = round(float(top_df["parasocial_score"].mean()), 4)
            top_moderate = top_df[top_df["parasocial_score"] >= config.thresholds["moderate"]]
            top_moderate_count = len(top_moderate)
            top_parasocial_density = round((top_moderate_count / total_top) * 1000, 2)

            for cats_str in top_df["parasocial_categories"].dropna():
                if cats_str:
                    for cat in cats_str.split(", "):
                        if cat:
                            top_category_counts[cat] = top_category_counts.get(cat, 0) + 1

    # Parasocial Resonance Delta (Top Comment Score - Baseline Score)
    # A positive delta indicates community upvotes disproportionately favor deep parasocial connection
    delta_parasocial = round(top_avg_score - baseline_avg_score, 4)

    # Extract editorial samples
    sorted_df = df.sort_values("parasocial_score", ascending=False)
    strongest_samples = sorted_df[sorted_df["parasocial_score"] >= config.thresholds["strong"]].head(3).to_dict("records")
    moderate_samples = sorted_df[
        (sorted_df["parasocial_score"] >= config.thresholds["moderate"]) &
        (sorted_df["parasocial_score"] < config.thresholds["strong"])
    ].head(3).to_dict("records")

    channel_title = clean_handle
    if raw_json_path.exists():
        with open(raw_json_path, "r", encoding="utf-8") as f:
            channel_title = json.load(f).get("channel_title", clean_handle)

    return {
        "channel_title": channel_title,
        "total_comments_analyzed": total_comments,
        "avg_comment_length_words": avg_word_count,
        "comment_length_std_dev": word_count_std,
        "longform_comment_rate_pct": longform_comments_pct,
        "baseline": {
            "avg_parasocial_score": baseline_avg_score,
            "parasocial_comments": parasocial_count,
            "parasocial_density_per_1k": parasocial_density,
            "strong_or_higher_count": len(strong_or_higher),
            "very_strong_count": len(very_strong),
            "category_density_per_1k": category_density,
        },
        "top_comments": {
            "analyzed_count": len(top_df),
            "avg_parasocial_score": top_avg_score,
            "parasocial_comments": top_moderate_count,
            "parasocial_density_per_1k": top_parasocial_density,
            "category_counts": top_category_counts,
        },
        "resonance_delta_parasocial": delta_parasocial,
        "sample_comments": {
            "strongest": strongest_samples,
            "moderate": moderate_samples,
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pillar 3 Parasocial Signal Detector on processed channel data.")
    parser.add_argument("handle", nargs="?", help="Channel handle (e.g. donflurgundy). If omitted, processes all CSVs in data/processed/.")
    args = parser.parse_args()

    processed_dir = Path("data/processed")

    if args.handle:
        clean_handle = args.handle.lstrip("@").lower()
        try:
            report = calculate_parasocial_impact(clean_handle)
            print(f"\n--- PILLAR 3: PARASOCIAL SIGNAL DENSITY (@{clean_handle}) ---")
            print(f"Baseline Avg Parasocial Score: {report['baseline']['avg_parasocial_score']}")
            print(f"Top-Comment Avg Parasocial Score: {report['top_comments']['avg_parasocial_score']}")
            print(f"Resonance Parasocial Delta (ΔR_parasocial): {report['resonance_delta_parasocial']:+} points")
            print(f"Top-Comment Parasocial Density: {report['top_comments']['parasocial_density_per_1k']} / 1,000 comments")
        except FileNotFoundError as e:
            print(f"[!] Error: {e}")
    else:
        csv_files = list(processed_dir.glob("*_metrics.csv"))
        if not csv_files:
            print("[!] No processed CSV files found in data/processed/. Run Pillar 1 first.")
        for csv_file in csv_files:
            handle_name = csv_file.stem.replace("_metrics", "")
            try:
                report = calculate_parasocial_impact(handle_name)
                print(f"\n--- PILLAR 3: PARASOCIAL SIGNAL DENSITY (@{handle_name}) ---")
                print(f"Baseline Avg Parasocial Score: {report['baseline']['avg_parasocial_score']}")
                print(f"Top-Comment Avg Parasocial Score: {report['top_comments']['avg_parasocial_score']}")
                print(f"Resonance Parasocial Delta (ΔR_parasocial): {report['resonance_delta_parasocial']:+} points")
            except Exception as e:
                print(f"[!] Error processing {handle_name}: {e}")