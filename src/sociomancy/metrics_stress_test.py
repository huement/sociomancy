import json
from pathlib import Path
import pandas as pd

def calculate_cold_audience_stress_test(channel_handle: str) -> dict:
    clean_handle = channel_handle.lstrip("@").lower()
    raw_json_path = Path(f"data/raw/{clean_handle}.json")
    csv_path = Path(f"data/processed/{clean_handle}_metrics.csv")

    if not raw_json_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"Missing required data files for handle '@{clean_handle}'. "
            f"Ensure data/raw/{clean_handle}.json and data/processed/{clean_handle}_metrics.csv exist."
        )

    # 1. Load raw metadata
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    subscribers = raw_data.get("subscribers", 0)
    videos_meta = {v["video_id"]: v for v in raw_data.get("videos", [])}

    # 2. Load processed comment CSV
    comments_df = pd.read_csv(csv_path)

    # Map video view counts to comment dataframe
    comments_df["view_count"] = comments_df["video_id"].map(
        lambda v_id: videos_meta.get(v_id, {}).get("view_count", 0)
    )
    comments_df["video_title"] = comments_df["video_id"].map(
        lambda v_id: videos_meta.get(v_id, {}).get("title", "Unknown")
    )

    # Calculate baseline performance metrics
    unique_videos = comments_df[["video_id", "view_count", "video_title"]].drop_duplicates()
    median_views = unique_videos["view_count"].median()

    # Flag Algorithmic Push / Outlier Videos:
    # Defined as views >= 3.0x subscribers OR views >= 1.5x channel median view count
    def classify_video(views):
        if subscribers > 0 and views >= (3.0 * subscribers):
            return True
        if median_views > 0 and views >= (1.5 * median_views):
            return True
        return False

    comments_df["is_algorithmic_push"] = comments_df["view_count"].apply(classify_video)

    # Separate baseline comments from push comments
    baseline_df = comments_df[~comments_df["is_algorithmic_push"]]
    push_df = comments_df[comments_df["is_algorithmic_push"]]

    # Calculate Toxicity Averages
    baseline_avg_tox = baseline_df["toxicity_score"].mean() if not baseline_df.empty else 0.0
    push_avg_tox = push_df["toxicity_score"].mean() if not push_df.empty else 0.0
    delta_t = push_avg_tox - baseline_avg_tox

    baseline_toxic_pct = (baseline_df["is_toxic"].sum() / len(baseline_df) * 100) if not baseline_df.empty else 0.0
    push_toxic_pct = (push_df["is_toxic"].sum() / len(push_df) * 100) if not push_df.empty else 0.0

    return {
        "channel_title": raw_data.get("channel_title"),
        "subscribers": subscribers,
        "median_views": int(median_views),
        "total_videos_analyzed": len(unique_videos),
        "push_videos_count": int(unique_videos["view_count"].apply(classify_video).sum()),
        "baseline_comments_count": len(baseline_df),
        "push_comments_count": len(push_df),
        "baseline_avg_toxicity": round(float(baseline_avg_tox), 4),
        "push_avg_toxicity": round(float(push_avg_tox), 4),
        "baseline_toxic_rate_pct": round(float(baseline_toxic_pct), 2),
        "push_toxic_rate_pct": round(float(push_toxic_pct), 2),
        "delta_toxicity": round(float(delta_t), 4)
    }

if __name__ == "__main__":
    handle = "mkbhd"
    try:
        report = calculate_cold_audience_stress_test(handle)
        
        print("\n--- PILLAR 4: COLD-AUDIENCE STRESS TEST (ΔT) ---")
        print(f"Channel: {report['channel_title']}")
        print(f"Sample Median Views: {report['median_views']:,}")
        print(f"Algorithmic Push Videos Detected: {report['push_videos_count']} of {report['total_videos_analyzed']}")
        print(f"\nBaseline Toxicity Rate: {report['baseline_toxic_rate_pct']}% (Avg Score: {report['baseline_avg_toxicity']})")
        print(f"Algorithmic Push Toxicity Rate: {report['push_toxic_rate_pct']}% (Avg Score: {report['push_avg_toxicity']})")
        print(f"-----------------------------------------------")
        print(f"Delta Toxicity (ΔT): {report['delta_toxicity']:+} points")
        
        if report['delta_toxicity'] <= 0:
            print("Interpretation: EXCELLENT. Community sentiment holds stable during cold audience surges.")
        elif report['delta_toxicity'] < 0.05:
            print("Interpretation: MODERATE. Minor increase in friction during viral spikes.")
        else:
            print("Interpretation: HIGH FRICTION. Channel receives significant toxic backlash on cold-audience hits.")

    except FileNotFoundError as e:
        print(f"[!] Error: {e}")