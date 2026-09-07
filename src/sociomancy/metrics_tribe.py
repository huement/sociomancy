import json
from pathlib import Path
import pandas as pd

def load_channel_dataset(file_path: str) -> tuple[dict, pd.DataFrame]:
    """Loads a raw channel JSON payload and flattens comments into a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    comments_list = []
    for video in data.get("videos", []):
        video_id = video["video_id"]
        video_title = video["title"]
        
        for comment in video.get("comments", []):
            comments_list.append({
                "video_id": video_id,
                "video_title": video_title,
                "comment_id": comment["comment_id"],
                "author_id": comment["author_id"],
                "author_name": comment["author_name"],
                "published_at": comment["published_at"],
                "like_count": comment["like_count"],
            })
            
    df = pd.DataFrame(comments_list)
    return data, df

def calculate_core_tribe_ratio(df: pd.DataFrame, min_videos: int = 2) -> dict:
    """
    Computes repeat author participation across distinct video uploads.
    """
    # Filter out anonymous or missing author IDs
    valid_comments = df[df["author_id"] != "ANONYMOUS"].copy()
    
    # Count distinct videos each author commented on
    author_video_counts = (
        valid_comments.groupby("author_id")["video_id"]
        .nunique()
        .reset_index(name="distinct_videos_commented")
    )
    
    total_unique_authors = len(author_video_counts)
    if total_unique_authors == 0:
        return {"core_tribe_ratio": 0.0, "total_authors": 0, "core_tribe_count": 0}
        
    # Filter authors active on >= min_videos distinct uploads
    core_tribe_members = author_video_counts[
        author_video_counts["distinct_videos_commented"] >= min_videos
    ]
    core_tribe_count = len(core_tribe_members)
    
    core_tribe_ratio = (core_tribe_count / total_unique_authors) * 100
    
    return {
        "total_unique_authors": total_unique_authors,
        "core_tribe_count": core_tribe_count,
        "core_tribe_ratio_pct": round(core_tribe_ratio, 2),
        "min_video_threshold": min_videos,
        "top_community_members": author_video_counts.sort_values(
            by="distinct_videos_commented", ascending=False
        ).head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    # Test on cached MKBHD data (or your target channel handle)
    json_path = Path("data/raw/mkbhd.json")
    
    if not json_path.exists():
        print(f"[!] Target file {json_path} does not exist. Run fetcher.py first.")
    else:
        metadata, comments_df = load_channel_dataset(json_path)
        print(f"[*] Loaded {len(comments_df)} comments across {metadata['channel_title']}")
        
        tribe_stats = calculate_core_tribe_ratio(comments_df, min_videos=2)
        
        print("\n--- PILLAR 2: CORE TRIBE RATIO ---")
        print(f"Total Unique Commenters: {tribe_stats['total_unique_authors']}")
        print(f"Repeat Frequenters (>= {tribe_stats['min_video_threshold']} videos): {tribe_stats['core_tribe_count']}")
        print(f"Core Tribe Ratio: {tribe_stats['core_tribe_ratio_pct']}%")