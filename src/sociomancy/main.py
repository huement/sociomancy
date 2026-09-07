import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings
import os

from sociomancy.fetcher import collect_and_save_channel_data
from sociomancy.metrics_empathy_toxicity import process_empathy_and_toxicity
from sociomancy.metrics_tribe import load_channel_dataset, calculate_core_tribe_ratio
from sociomancy.metrics_parasocial import calculate_parasocial_impact
from sociomancy.metrics_stress_test import calculate_cold_audience_stress_test
from sociomancy.metrics_authenticity import calculate_channel_authenticity
from sociomancy.scorecard import generate_channel_scorecard

# Suppress PyTorch ONNX exporter & Hugging Face deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.onnx")

# Mute Hugging Face verbosity
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def parse_pillars(pillar_str: str) -> set[int]:
    """Parses comma-separated pillar numbers (e.g., '1,3,4') into a set of integers."""
    if not pillar_str or pillar_str.lower() == "all":
        return {1, 2, 3, 4, 5}

    try:
        pillars = {int(p.strip()) for p in pillar_str.split(",") if p.strip()}
        invalid = pillars - {1, 2, 3, 4, 5}
        if invalid:
            raise ValueError(
                f"Invalid pillar numbers: {invalid}. Must be between 1 and 5.")
        return pillars
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid pillar selection format: {e}")


def run_pipeline(channel_identifier: str,
                 video_limit: int = 15,
                 comments_per_video: int = 300,
                 since_date: str = None,
                 until_date: str = None,
                 pillars: set[int] = None):
    if pillars is None:
        pillars = {1, 2, 3, 4, 5}

    clean_handle = channel_identifier.lstrip("@").lower()
    raw_json_path = Path(f"data/raw/{clean_handle}.json")
    csv_path = Path(f"data/processed/{clean_handle}_metrics.csv")

    print(f"\n==================================================")
    print(f"      SOCIOMANCY ANALYSIS PIPELINE")
    print(f"==================================================")
    print(f"Target: @{clean_handle}")
    print(f"Pillars Selected: {sorted(list(pillars))}")
    if since_date or until_date:
        print(
            f"Date Filter: {since_date or 'Beginning'} -> {until_date or 'Present'}"
        )
    print(f"Video Limit: {video_limit} | Comments/Video: {comments_per_video}")
    print(f"==================================================\n")

    # STEP 1: Ingestion (Incremental Cache Check)
    print("[1/3] Executing YouTube API Data Ingestion...")
    collect_and_save_channel_data(clean_handle,
                                  video_limit=video_limit,
                                  comments_per_video=comments_per_video)

    # Note: Pillars 3, 4, and 5 depend on Pillar 1's processed CSV feature vectors.
    requires_p1_csv = bool({1, 3, 4, 5}.intersection(pillars))

    # STEP 2: ML Feature Extraction (Pillar 1 Incremental Cache Check)
    if requires_p1_csv:
        print("\n[2/3] Running ONNX Local ML Model Pipeline (Pillar 1)...")
        process_empathy_and_toxicity(raw_json_path)

    # STEP 3: Report & Metrics Aggregation
    print("\n[3/3] Generating Pillar Reports...")
    print("\n--------------------------------------------------")

    if 1 in pillars:
        p1_stats = process_empathy_and_toxicity(raw_json_path)
        base = p1_stats["baseline"]
        top_c = p1_stats["top_comments"]
        base_emo = base["dominant_emotions"][0][0] if base[
            "dominant_emotions"] else "neutral"
        top_emo = top_c["dominant_emotions"][0][0] if top_c[
            "dominant_emotions"] else "neutral"

        print(f"[*] PILLAR 1 (Empathy vs Toxicity):")
        print(
            f"    • Baseline Toxicity: {base['toxicity_percentage']}% (Avg Score: {base['average_toxicity_score']})"
        )
        print(
            f"    • Top-Comment Toxicity: {top_c['toxicity_percentage']}% (Avg Score: {top_c['average_toxicity_score']})"
        )
        print(
            f"    • Resonance Toxicity Delta (ΔR_tox): {p1_stats['resonance_delta_toxicity']:+} pts"
        )
        print(f"    • Dominant Emotion (Base / Top): {base_emo} / {top_emo}")

    if 2 in pillars:
        metadata, comments_df = load_channel_dataset(raw_json_path)
        tribe_stats = calculate_core_tribe_ratio(comments_df, min_videos=2)
        print(f"[*] PILLAR 2 (Core Tribe Ratio):")
        print(
            f"    • Core Tribe Ratio: {tribe_stats['core_tribe_ratio_pct']}%")
        print(
            f"    • Repeat Commenters: {tribe_stats['core_tribe_count']} unique handles"
        )

    if 3 in pillars:
        para_stats = calculate_parasocial_impact(clean_handle)
        print(f"[*] PILLAR 3 (Parasocial Impact):")
        print(
            f"    • Impact Density: {para_stats['baseline']['parasocial_density_per_1k']} / 1,000 comments"
        )
        print(
            f"    • Top-Comment Density: {para_stats['top_comments']['parasocial_density_per_1k']} / 1,000 comments"
        )
        print(
            f"    • Resonance Delta (ΔR_parasocial): {para_stats['resonance_delta_parasocial']:+} pts"
        )
        print(
            f"    • Avg Comment Length: {para_stats['avg_comment_length_words']} words"
        )

    if 4 in pillars:
        stress_stats = calculate_cold_audience_stress_test(clean_handle)
        print(f"[*] PILLAR 4 (Algorithmic Stress Test):")
        print(
            f"    • Stress Test Delta (ΔT): {stress_stats['delta_toxicity']:+} points"
        )
        print(
            f"    • Outlier Push Videos Detected: {stress_stats['push_videos_count']}"
        )

    if 5 in pillars:
        bot_stats = calculate_channel_authenticity(clean_handle)
        print(f"[*] PILLAR 5 (Bot & Authenticity Estimation):")
        print(
            f"    • Comment Authenticity Score: {bot_stats['comment_authenticity_score']}%"
        )
        print(
            f"    • Suspicious Comments: {bot_stats['suspicious_rate_pct']}% (Avg Bot Confidence: {bot_stats['avg_bot_confidence']}/100)"
        )

    # Export consolidated JSON scorecard when all 5 pillars are run
    if pillars == {1, 2, 3, 4, 5}:
        scorecard = generate_channel_scorecard(clean_handle)
        print(
            f"\n[+] Scorecard exported to data/processed/{clean_handle}_scorecard.json"
        )

    print("--------------------------------------------------")
    print("[+] Pipeline execution complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description=
        "SOCIOMANCY: Data-Driven YouTube Channel Growth & Sentiment Analyzer")

    parser.add_argument(
        "channel",
        type=str,
        help="YouTube Channel Handle (e.g., @mkbhd) or Channel ID")
    parser.add_argument(
        "-p",
        "--pillars",
        type=parse_pillars,
        default={1, 2, 3, 4, 5},
        help=
        "Comma-separated pillars to run (e.g. --pillars 1,3,4,5). Defaults to all (1,2,3,4,5)."
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=15,
        help="Maximum number of recent video uploads to analyze (default: 15)."
    )
    parser.add_argument(
        "-c",
        "--comments-limit",
        type=int,
        default=300,
        help="Maximum comments to fetch per video upload (default: 300).")
    parser.add_argument("--since",
                        type=str,
                        default=None,
                        help="Filter uploads since date (YYYY-MM-DD).")
    parser.add_argument("--until",
                        type=str,
                        default=None,
                        help="Filter uploads until date (YYYY-MM-DD).")

    args = parser.parse_args()

    run_pipeline(channel_identifier=args.channel,
                 video_limit=args.limit,
                 comments_per_video=args.comments_limit,
                 since_date=args.since,
                 until_date=args.until,
                 pillars=args.pillars)


if __name__ == "__main__":
    main()
