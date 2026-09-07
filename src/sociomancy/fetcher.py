import json
import os
from pathlib import Path
from datetime import datetime
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)

def fetch_channel_stats(youtube, channel_handle):
    handle = channel_handle.lstrip("@")
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        forHandle=handle
    )
    response = request.execute()
    if not response.get("items"):
        raise ValueError(f"Channel handle @{handle} not found.")
    return response["items"][0]

def fetch_recent_videos(youtube, uploads_playlist_id, max_results=15):
    videos = []
    next_page_token = None
    
    while len(videos) < max_results:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response.get("items", []):
            video_id = item["contentDetails"]["videoId"]
            stats_request = youtube.videos().list(
                part="statistics,snippet",
                id=video_id
            )
            stats_response = stats_request.execute()
            if stats_response.get("items"):
                v_data = stats_response["items"][0]
                videos.append({
                    "video_id": video_id,
                    "title": v_data["snippet"]["title"],
                    "published_at": v_data["snippet"]["publishedAt"],
                    "view_count": int(v_data["statistics"].get("viewCount", 0)),
                    "like_count": int(v_data["statistics"].get("likeCount", 0)),
                    "comment_count": int(v_data["statistics"].get("commentCount", 0)),
                })
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
            
    return videos

def fetch_video_comments(youtube, video_id, max_comments=300):
    comments = []
    next_page_token = None
    
    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id": item["id"],
                    "author_id": comment.get("authorChannelId", {}).get("value", "ANONYMOUS"),
                    "author_name": comment.get("authorDisplayName"),
                    "text": comment.get("textDisplay"),
                    "like_count": comment.get("likeCount", 0),
                    "published_at": comment.get("publishedAt")
                })
                
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            print(f"   [!] Warning: Could not fetch comments for video {video_id}: {e}")
            break
            
    return comments

def collect_and_save_channel_data(channel_handle: str, video_limit: int = 15, comments_per_video: int = 300):
    youtube = get_youtube_client()
    clean_handle = channel_handle.lstrip("@").lower()
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{clean_handle}.json"

    # Load existing cache if present
    cached_videos_map = {}
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                cached_videos_map = {v["video_id"]: v for v in existing_data.get("videos", [])}
            print(f"[*] Loaded local cache with {len(cached_videos_map)} existing videos.")
        except Exception as e:
            print(f"[!] Warning reading cache: {e}. Re-indexing.")

    print(f"[*] Fetching metadata for @{clean_handle}...")
    channel_info = fetch_channel_stats(youtube, clean_handle)
    uploads_id = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]
    
    print(f"[*] Fetching last {video_limit} videos...")
    videos = fetch_recent_videos(youtube, uploads_id, max_results=video_limit)
    
    for idx, video in enumerate(videos, 1):
        v_id = video["video_id"]
        cached_v = cached_videos_map.get(v_id)
        
        # Check if video comments are already cached
        if cached_v and len(cached_v.get("comments", [])) >= comments_per_video:
            print(f"   ({idx}/{len(videos)}) [CACHE HIT] Skipping API call for: {video['title'][:40]}...")
            video["comments"] = cached_v["comments"]
        else:
            print(f"   ({idx}/{len(videos)}) [API FETCH] Fetching comments for: {video['title'][:40]}...")
            video["comments"] = fetch_video_comments(youtube, v_id, max_comments=comments_per_video)
            print(f"       -> Got {len(video['comments'])} comments.")
        
    payload = {
        "channel_handle": clean_handle,
        "channel_title": channel_info["snippet"]["title"],
        "subscribers": int(channel_info["statistics"]["subscriberCount"]),
        "total_views": int(channel_info["statistics"]["viewCount"]),
        "total_videos": int(channel_info["statistics"]["videoCount"]),
        "fetched_at": datetime.utcnow().isoformat(),
        "videos": videos
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Data successfully updated at {file_path}")