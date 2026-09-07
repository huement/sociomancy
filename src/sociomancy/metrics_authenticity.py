import json
import argparse
from pathlib import Path
import pandas as pd

from sociomancy.analysis.bot_authenticity import BotAuthenticityDetector

def calculate_channel_authenticity(channel_handle: str) -> dict:
    clean_handle = channel_handle.lstrip("@").lower()
    csv_path = Path(f"data/processed/{clean_handle}_metrics.csv")
    raw_json_path = Path(f"data/raw/{clean_handle}.json")

    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found at {csv_path}. Run Pillar 1 first.")

    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)

    detector = BotAuthenticityDetector(suspicious_threshold=60.0)
    enriched_df, summary = detector.analyze_dataframe(df)

    # Save enriched dataframe with bot_confidence and bot_reasons back to CSV
    enriched_df.to_csv(csv_path, index=False)

    channel_title = clean_handle
    if raw_json_path.exists():
        with open(raw_json_path, "r", encoding="utf-8") as f:
            channel_title = json.load(f).get("channel_title", clean_handle)

    summary["channel_title"] = channel_title
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pillar 5 Bot & Authenticity Estimation with Flag Reasons.")
    parser.add_argument("handle", nargs="?", help="Channel handle (e.g. donflurgundy). If omitted, processes all CSVs in data/processed/.")
    args = parser.parse_args()

    processed_dir = Path("data/processed")

    if args.handle:
        clean_handle = args.handle.lstrip("@").lower()
        try:
            res = calculate_channel_authenticity(clean_handle)
            print(f"\n--- PILLAR 5: BOT & COMMENT AUTHENTICITY (@{clean_handle}) ---")
            print(f"Overall Comment Authenticity Score: {res['comment_authenticity_score']}%")
            print(f"Suspicious Comment Rate: {res['suspicious_rate_pct']}% ({res['suspicious_comments_count']}/{res['total_comments_analyzed']} comments)")
            print(f"Exact Duplicate Groups: {res['exact_duplicate_groups']} | Semantic Clusters: {res['semantic_copy_clusters']}")
            print(f"Average Bot Confidence: {res['avg_bot_confidence']} / 100")
            print(f"Top Flagged Anomaly Reasons: {res['flagged_reasons_breakdown']}")
        except FileNotFoundError as e:
            print(f"[!] Error: {e}")
    else:
        csv_files = list(processed_dir.glob("*_metrics.csv"))
        if not csv_files:
            print("[!] No processed CSV files found in data/processed/. Run Pillar 1 first.")
        for csv_file in csv_files:
            handle_name = csv_file.stem.replace("_metrics", "")
            try:
                res = calculate_channel_authenticity(handle_name)
                print(f"\n--- PILLAR 5: BOT & COMMENT AUTHENTICITY (@{handle_name}) ---")
                print(f"Comment Authenticity Score: {res['comment_authenticity_score']}%")
                print(f"Top Flagged Anomaly Reasons: {res['flagged_reasons_breakdown']}")
            except Exception as e:
                print(f"[!] Error processing {handle_name}: {e}")