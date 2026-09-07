import json
from pathlib import Path
import pandas as pd

from sociomancy.metrics_empathy_toxicity import process_empathy_and_toxicity
from sociomancy.metrics_tribe import load_channel_dataset, calculate_core_tribe_ratio
from sociomancy.metrics_parasocial import calculate_parasocial_impact
from sociomancy.metrics_stress_test import calculate_cold_audience_stress_test
from sociomancy.metrics_authenticity import calculate_channel_authenticity


def generate_channel_scorecard(channel_handle: str) -> dict:
    clean_handle = channel_handle.lstrip("@").lower()
    raw_json_path = Path(f"data/raw/{clean_handle}.json")
    csv_path = Path(f"data/processed/{clean_handle}_metrics.csv")

    if not raw_json_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"Missing required data for @{clean_handle}. "
            "Ensure fetcher.py and process_empathy_and_toxicity have been executed first."
        )

    # 1. Load Raw JSON & Datasets
    metadata, comments_df = load_channel_dataset(raw_json_path)

    # 2. Compute Metrics Across All 5 Pillars
    p1_metrics = process_empathy_and_toxicity(raw_json_path)
    p2_metrics = calculate_core_tribe_ratio(comments_df, min_videos=2)
    p3_metrics = calculate_parasocial_impact(clean_handle)
    p4_metrics = calculate_cold_audience_stress_test(clean_handle)
    p5_metrics = calculate_channel_authenticity(clean_handle)

    # Extract emotion trends
    base_dom_emo = (p1_metrics["baseline"]["dominant_emotions"][0][0] if
                    p1_metrics["baseline"]["dominant_emotions"] else "neutral")
    top_dom_emo = (p1_metrics["top_comments"]["dominant_emotions"][0][0]
                   if p1_metrics["top_comments"]["dominant_emotions"] else
                   "neutral")

    # 3. Compile Unified Scorecard
    scorecard = {
        "channel_title": metadata.get("channel_title", clean_handle),
        "handle": clean_handle,
        "subscribers": metadata.get("subscribers", 0),
        "total_comments_sampled": len(comments_df),
        "pillars": {
            "p1_empathy_vs_toxicity": {
                "toxicity_rate_pct":
                p1_metrics["baseline"]["toxicity_percentage"],
                "avg_toxicity_score":
                p1_metrics["baseline"]["average_toxicity_score"],
                "dominant_emotion":
                base_dom_emo,
                "top_comment_toxicity_pct":
                p1_metrics["top_comments"]["toxicity_percentage"],
                "top_comment_avg_toxicity":
                p1_metrics["top_comments"]["average_toxicity_score"],
                "top_comment_dominant_emotion":
                top_dom_emo,
                "resonance_delta_toxicity":
                p1_metrics["resonance_delta_toxicity"],
            },
            "p2_core_tribe_ratio": {
                "tribe_ratio_pct": p2_metrics["core_tribe_ratio_pct"],
                "repeat_commenters_count": p2_metrics["core_tribe_count"],
            },
            "p3_parasocial_impact": {
                "parasocial_density_per_1k":
                p3_metrics["baseline"]["parasocial_density_per_1k"],
                "avg_parasocial_score":
                p3_metrics["baseline"]["avg_parasocial_score"],
                "longform_take_rate_pct":
                p3_metrics["longform_comment_rate_pct"],
                "avg_comment_length_words":
                p3_metrics["avg_comment_length_words"],
                "top_comment_parasocial_density":
                p3_metrics["top_comments"]["parasocial_density_per_1k"],
                "top_comment_avg_parasocial_score":
                p3_metrics["top_comments"]["avg_parasocial_score"],
                "resonance_delta_parasocial":
                p3_metrics["resonance_delta_parasocial"],
            },
            "p4_cold_audience_stress_test": {
                "delta_toxicity": p4_metrics["delta_toxicity"],
                "baseline_toxicity_pct": p4_metrics["baseline_toxic_rate_pct"],
                "push_toxicity_pct": p4_metrics["push_toxic_rate_pct"],
                "push_videos_count": p4_metrics["push_videos_count"],
            },
            "p5_bot_and_authenticity": {
                "comment_authenticity_score":
                p5_metrics["comment_authenticity_score"],
                "suspicious_rate_pct":
                p5_metrics["suspicious_rate_pct"],
                "avg_bot_confidence":
                p5_metrics["avg_bot_confidence"],
                "exact_duplicate_groups":
                p5_metrics["exact_duplicate_groups"],
                "semantic_copy_clusters":
                p5_metrics["semantic_copy_clusters"],
                "top_flagged_reasons":
                p5_metrics.get("flagged_reasons_breakdown", {}),
            }
        }
    }

    # Save Scorecard JSON
    output_path = Path(f"data/processed/{clean_handle}_scorecard.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)

    return scorecard


if __name__ == "__main__":
    handle = "donflurgundy"
    try:
        card = generate_channel_scorecard(handle)
        p = card["pillars"]

        print("==================================================")
        print(f"      SOCIOMANCY SCORECARD: {card['channel_title'].upper()}")
        print("==================================================")
        print(
            f"Subscribers: {card['subscribers']:,} | Sample Size: {card['total_comments_sampled']} comments\n"
        )
        print(
            f"1. Toxicity Rate (Base / Top): {p['p1_empathy_vs_toxicity']['toxicity_rate_pct']}% / {p['p1_empathy_vs_toxicity']['top_comment_toxicity_pct']}% (ΔR_tox: {p['p1_empathy_vs_toxicity']['resonance_delta_toxicity']:+} pts)"
        )
        print(
            f"2. Core Tribe Ratio:           {p['p2_core_tribe_ratio']['tribe_ratio_pct']}%"
        )
        print(
            f"3. Parasocial Density:         {p['p3_parasocial_impact']['parasocial_density_per_1k']} / 1k (Top: {p['p3_parasocial_impact']['top_comment_parasocial_density']} / 1k, ΔR_parasocial: {p['p3_parasocial_impact']['resonance_delta_parasocial']:+} pts)"
        )
        print(
            f"4. Stress Test Delta (ΔT):     {p['p4_cold_audience_stress_test']['delta_toxicity']:+} points"
        )
        print(
            f"5. Comment Authenticity:       {p['p5_bot_and_authenticity']['comment_authenticity_score']}% (Suspicious: {p['p5_bot_and_authenticity']['suspicious_rate_pct']}%)"
        )
        print("==================================================")
        print(
            f"[+] Consolidated scorecard exported to data/processed/{card['handle']}_scorecard.json"
        )

    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
