import json
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from sociomancy.scorecard import generate_channel_scorecard
from sociomancy.embeddings import EmbeddingService

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Configuration Controls
MIN_SUBSCRIBERS = 1000                  # Disqualify tiny, 0-sub, or dead channels
MIN_AUDIENCE_AFFINITY_THRESHOLD = 0.001 # Minimum overlap required for pure Approach B candidates

SEMANTIC_SIMILARITY_WEIGHT = 0.55
AUDIENCE_AFFINITY_WEIGHT = 0.25

def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)

def load_target_commenters(clean_handle: str) -> tuple[set[str], dict]:
    """Loads target channel raw data and extracts set of unique author IDs."""
    raw_path = Path(f"data/raw/{clean_handle}.json")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data for @{clean_handle} not found. Run sociomancy pipeline first.")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    author_ids = set()
    for video in raw_data.get("videos", []):
        for c in video.get("comments", []):
            aid = c.get("author_id")
            if aid and aid != "ANONYMOUS":
                author_ids.add(aid)

    return author_ids, raw_data

def sample_candidate_commenters(youtube, channel_id: str, max_videos: int = 3, max_comments_per_video: int = 100) -> set[str]:
    """Samples recent video comments from a candidate channel to measure audience overlap."""
    author_ids = set()
    try:
        ch_res = youtube.channels().list(part="contentDetails", id=channel_id).execute()
        if not ch_res.get("items"):
            return author_ids

        uploads_id = ch_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        pl_res = youtube.playlistItems().list(part="contentDetails", playlistId=uploads_id, maxResults=max_videos).execute()

        for item in pl_res.get("items", []):
            v_id = item["contentDetails"]["videoId"]
            try:
                c_res = youtube.commentThreads().list(
                    part="snippet",
                    videoId=v_id,
                    maxResults=min(max_comments_per_video, 100),
                    textFormat="plainText"
                ).execute()

                for c_item in c_res.get("items", []):
                    aid = c_item["snippet"]["topLevelComment"]["snippet"].get("authorChannelId", {}).get("value")
                    if aid:
                        author_ids.add(aid)
            except HttpError:
                continue
    except HttpError:
        pass

    return author_ids

def generate_cards_for_channels(youtube,
                                target_id: str,
                                candidates: list[dict],
                                clean_handle: str = "",
                                output_dir_base: str = "./output_cards",
                                ):
    """Fetches channel details, preserves all discovery metrics, and builds HTML cards into output_cards/<clean_handle>/."""
    if not candidates and not target_id:
        return

    output_dir = Path(output_dir_base) / clean_handle
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_ids = [c["channel_id"] for c in candidates if "channel_id" in c]
    candidates_map = {c["channel_id"]: c for c in candidates if "channel_id" in c}

    all_channel_ids = [target_id] + channel_ids

    res = youtube.channels().list(
        part="snippet,statistics,brandingSettings",
        id=",".join(all_channel_ids[:50])
    ).execute()

    cards_payload = []
    now = datetime.now(timezone.utc)

    for item in res.get("items", []):
        channel_id = item["id"]
        cand_data = candidates_map.get(channel_id, {})

        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        branding = item.get("brandingSettings", {}).get("image", {})

        raw_handle = snippet.get("customUrl", "")
        handle_str = f"@{raw_handle.lstrip('@')}" if raw_handle else f"@{snippet.get('title', '').replace(' ', '')}"

        pub_str = snippet.get("publishedAt", "")
        lifespan_str = "N/A"
        if pub_str:
            pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            days = (now - pub_date).days
            years = days // 365
            lifespan_str = f"{years} Year{'s' if years != 1 else ''}" if years >= 1 else f"{days // 30} Month{'s' if (days // 30) != 1 else ''}"

        thumbnails = snippet.get("thumbnails", {})
        avatar_url = (
            thumbnails.get("high", {}).get("url") or 
            thumbnails.get("medium", {}).get("url") or 
            thumbnails.get("default", {}).get("url", "")
        )

        payload = {
            "channel_id": channel_id,
            "title": snippet.get("title", ""),
            "handle": handle_str,
            "avatar_url": avatar_url,
            "subs": stats.get("subscriberCount", "0"),
            "lifespan": lifespan_str,
            "banner_url": branding.get("bannerExternalUrl", ""),
            "peer_score": cand_data.get("peer_score", 0.0),
            "semantic_similarity": cand_data.get("semantic_similarity", 0.0),
            "audience_affinity": cand_data.get("audience_affinity", 0.0),
            "jaccard_similarity": cand_data.get("jaccard_similarity", 0.0),
            "shared_commenters_count": cand_data.get("shared_commenters_count", 0),
            "target_audience_overlap_pct": cand_data.get("target_audience_overlap_pct", 0.0),
            "selection_rationale": cand_data.get("selection_rationale", ""),
            **cand_data
        }
        cards_payload.append(payload)

    cards_input_path = output_dir / "related.json"
    with open(cards_input_path, "w", encoding="utf-8") as f:
        json.dump(cards_payload, f, indent=2, ensure_ascii=False)

    print(f"[*] Saved complete related channels payload to {cards_input_path}")
    print(f"[*] Building HTML cards into {output_dir}...")

    subprocess.run([
        "uv", "run", "python", "src/cards/build_cards.py",
        str(cards_input_path), "-o", str(output_dir)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_channel_profile(channel_item: dict, videos: list[dict]) -> str:
    """Builds a textual profile for a channel from its metadata and recent videos."""
    snippet = channel_item.get("snippet", {})

    profile_parts = [
        f"CHANNEL: {snippet.get('title', '')}",
        f"DESCRIPTION: {snippet.get('description', '')}",
        f"KEYWORDS: {channel_item.get('brandingSettings', {}).get('channel', {}).get('keywords', '')}",
        f"TOPICS: {', '.join(channel_item.get('topicDetails', {}).get('topicCategories', []))}",
        "RECENT VIDEOS:"
    ]

    for video in videos[:20]:
        profile_parts.append(f"- {video.get('snippet', {}).get('title', '')}")

    return "\n".join(profile_parts)

def fetch_candidate_pools(youtube, search_query: str, target_id: str) -> tuple[list[dict], list[dict]]:
    """Generates candidate pools separated by Approach A (Channel Graph) and Approach B (Video Search)."""
    candidates_a = {}
    candidates_b = {}
    seen_ids = {target_id}

    # Approach A: Channel Search (Primary)
    ch_search_req = youtube.search().list(
        part="snippet", q=search_query, type="channel", maxResults=15
    ).execute()

    for item in ch_search_req.get("items", []):
        cid = item["snippet"]["channelId"]
        if cid not in seen_ids:
            candidates_a[cid] = {
                "channel_id": cid,
                "channel_title": item["snippet"]["title"],
                "handle": item["snippet"].get("customUrl", item["snippet"]["title"].replace(" ", "")).lstrip("@"),
                "discovery_source": "Approach A (Topic & Channel Graph)"
            }
            seen_ids.add(cid)

    # Approach B: Video Search (Secondary / Padding)
    vid_search_req = youtube.search().list(
        part="snippet", q=search_query, type="video", maxResults=15
    ).execute()

    for item in vid_search_req.get("items", []):
        cid = item["snippet"]["channelId"]
        if cid not in seen_ids and cid not in candidates_a:
            candidates_b[cid] = {
                "channel_id": cid,
                "channel_title": item["snippet"]["channelTitle"],
                "handle": item["snippet"].get("customUrl", item["snippet"]["channelTitle"].replace(" ", "")).lstrip("@"),
                "discovery_source": "Approach B (Video Search Padding)"
            }
            seen_ids.add(cid)

    return list(candidates_a.values()), list(candidates_b.values())

def calculate_peer_score(cand: dict) -> float:
    """Calculates a composite peer score based on weighted similarities."""
    return (
        cand.get("semantic_similarity", 0.0) * SEMANTIC_SIMILARITY_WEIGHT +
        cand.get("audience_affinity", 0.0) * AUDIENCE_AFFINITY_WEIGHT
    )

def run_discovery(channel_handle: str, max_results: int = 8) -> dict:
    youtube = get_youtube_client()
    embed_service = EmbeddingService()

    clean_handle = channel_handle.lstrip("@").lower()
    target_commenters, target_raw = load_target_commenters(clean_handle)

    # 1. Fetch target channel details
    ch_req = youtube.channels().list(
        part="snippet,topicDetails,brandingSettings",
        forHandle=clean_handle
    )
    ch_res = ch_req.execute()
    if not ch_res.get("items"):
        raise ValueError(f"Channel @{clean_handle} not found on YouTube.")

    target_item = ch_res["items"][0]
    target_id = target_item["id"]
    channel_title = target_item["snippet"]["title"]
    keywords = target_item.get("brandingSettings", {}).get("channel", {}).get("keywords", "")

    search_query = keywords if keywords else channel_title

    # Build target profile & embedding
    uploads_id = target_item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    target_videos = []
    if uploads_id:
        pl_res = youtube.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=20).execute()
        target_videos = pl_res.get("items", [])

    target_profile = build_channel_profile(target_item, target_videos)
    target_embedding = embed_service.generate_embeddings([target_profile])[0]

    # 2. Fetch raw pools for A and B
    pool_a, pool_b = fetch_candidate_pools(youtube, search_query, target_id)
    all_raw_candidates = pool_a + pool_b

    if not all_raw_candidates:
        print("[!] No candidate channels discovered.")
        return {}

    # 3. Batch fetch candidate details & enforce MIN_SUBSCRIBERS floor
    cand_ids = [c["channel_id"] for c in all_raw_candidates]
    ch_details_req = youtube.channels().list(
        part="snippet,statistics,topicDetails,brandingSettings,contentDetails",
        id=",".join(cand_ids[:50])
    ).execute()

    qualified_map = {}
    for item in ch_details_req.get("items", []):
        cid = item["id"]
        subs = int(item.get("statistics", {}).get("subscriberCount", 0))

        # Disqualify channels below subscriber floor
        if subs < MIN_SUBSCRIBERS:
            continue

        qualified_map[cid] = item

    # Filter pools to only subscriber-qualified channels
    valid_a = [c for c in pool_a if c["channel_id"] in qualified_map]
    valid_b = [c for c in pool_b if c["channel_id"] in qualified_map]

    # 4. Semantic Ranking for Qualified Approach A Candidates
    print(f"[*] Ranking {len(valid_a)} qualified Approach A candidates...")
    for cand in valid_a:
        cid = cand["channel_id"]
        cand_item = qualified_map[cid]

        cand_uploads_id = cand_item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        cand_videos = []
        if cand_uploads_id:
            try:
                cand_pl_res = youtube.playlistItems().list(part="snippet", playlistId=cand_uploads_id, maxResults=20).execute()
                cand_videos = cand_pl_res.get("items", [])
            except Exception:
                pass

        cand_profile = build_channel_profile(cand_item, cand_videos)
        cand_embedding = embed_service.generate_embeddings([cand_profile])[0]
        cand["semantic_similarity"] = float(embed_service.cosine_similarity(target_embedding, cand_embedding))

    # Sort Approach A by semantic similarity
    valid_a.sort(key=lambda x: x.get("semantic_similarity", 0.0), reverse=True)

    # 5. Measure Audience Overlap on Approach A Candidates
    print(f"[*] Measuring commenter cross-pollination on Approach A channels...")
    final_candidates = []

    for cand in valid_a:
        cand_commenters = sample_candidate_commenters(youtube, cand["channel_id"])
        shared = target_commenters.intersection(cand_commenters)
        union = target_commenters.union(cand_commenters)

        jaccard = round(len(shared) / len(union), 4) if union else 0.0
        target_overlap_pct = round((len(shared) / len(target_commenters)) * 100, 2) if target_commenters else 0.0
        cand_overlap_pct = round((len(shared) / len(cand_commenters)) * 100, 2) if cand_commenters else 0.0

        if (target_overlap_pct + cand_overlap_pct) > 0:
            audience_affinity = (2 * target_overlap_pct * cand_overlap_pct) / (target_overlap_pct + cand_overlap_pct)
        else:
            audience_affinity = 0.0

        cand["jaccard_similarity"] = jaccard
        cand["shared_commenters_count"] = len(shared)
        cand["target_audience_overlap_pct"] = target_overlap_pct
        cand["candidate_audience_overlap_pct"] = cand_overlap_pct
        cand["audience_affinity"] = round(audience_affinity, 4)
        cand["peer_score"] = calculate_peer_score(cand)
        cand["selection_rationale"] = (
            f"Selected via Approach A (Channel Graph). Semantic Sim: {cand['semantic_similarity']:.2f}, "
            f"Audience Overlap: {len(shared)} shared users ({target_overlap_pct}%)."
        )
        final_candidates.append(cand)

    # Take top 5 Approach A candidates
    selected_candidates = final_candidates[:5]

    # 6. Evaluate Approach B candidates ONLY to pad output if slots remain
    slots_remaining = max_results - len(selected_candidates)
    if slots_remaining > 0 and valid_b:
        print(f"[*] Evaluating Approach B candidates to pad remaining {slots_remaining} slots...")
        
        # Rank Approach B candidates semantically first
        for cand in valid_b:
            cid = cand["channel_id"]
            cand_item = qualified_map[cid]

            cand_uploads_id = cand_item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            cand_videos = []
            if cand_uploads_id:
                try:
                    cand_pl_res = youtube.playlistItems().list(part="snippet", playlistId=cand_uploads_id, maxResults=20).execute()
                    cand_videos = cand_pl_res.get("items", [])
                except Exception:
                    pass

            cand_profile = build_channel_profile(cand_item, cand_videos)
            cand_embedding = embed_service.generate_embeddings([cand_profile])[0]
            cand["semantic_similarity"] = float(embed_service.cosine_similarity(target_embedding, cand_embedding))

        valid_b.sort(key=lambda x: x.get("semantic_similarity", 0.0), reverse=True)

        for cand in valid_b:
            if len(selected_candidates) >= max_results:
                break

            cand_commenters = sample_candidate_commenters(youtube, cand["channel_id"])
            shared = target_commenters.intersection(cand_commenters)
            union = target_commenters.union(cand_commenters)

            jaccard = round(len(shared) / len(union), 4) if union else 0.0
            target_overlap_pct = round((len(shared) / len(target_commenters)) * 100, 2) if target_commenters else 0.0
            cand_overlap_pct = round((len(shared) / len(cand_commenters)) * 100, 2) if cand_commenters else 0.0

            if (target_overlap_pct + cand_overlap_pct) > 0:
                audience_affinity = (2 * target_overlap_pct * cand_overlap_pct) / (target_overlap_pct + cand_overlap_pct)
            else:
                audience_affinity = 0.0

            cand["jaccard_similarity"] = jaccard
            cand["shared_commenters_count"] = len(shared)
            cand["target_audience_overlap_pct"] = target_overlap_pct
            cand["candidate_audience_overlap_pct"] = cand_overlap_pct
            cand["audience_affinity"] = round(audience_affinity, 4)
            cand["peer_score"] = calculate_peer_score(cand)

            # Strict Guardrail: Ignore Approach B candidates if they do not pass audience affinity threshold
            if len(shared) == 0 and audience_affinity < MIN_AUDIENCE_AFFINITY_THRESHOLD:
                continue

            cand["selection_rationale"] = (
                f"Selected via Approach B (Video Padding). Verified audience affinity: {audience_affinity:.4f} "
                f"({len(shared)} shared users)."
            )
            selected_candidates.append(cand)

    # Final ordering by composite peer score
    selected_candidates.sort(key=lambda x: x.get("peer_score", 0.0), reverse=True)

    # 7. Update Scorecard JSON
    scorecard_path = Path(f"data/processed/{clean_handle}_scorecard.json")
    if scorecard_path.exists():
        with open(scorecard_path, "r", encoding="utf-8") as f:
            scorecard_data = json.load(f)
    else:
        scorecard_data = generate_channel_scorecard(clean_handle)

    scorecard_data["discovery"] = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "target_sampled_commenters": len(target_commenters),
        "min_subscribers_filter": MIN_SUBSCRIBERS,
        "candidates": selected_candidates
    }

    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard_data, f, indent=2, ensure_ascii=False)

    # 8. Render PNG cards into output_cards/<channel_id>/
    generate_cards_for_channels(youtube, target_id, selected_candidates, clean_handle, output_dir_base="./output_cards")

    print(f"[+] Discovery metadata successfully appended to {scorecard_path}\n")
    return scorecard_data["discovery"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover control group channels using YouTube Channel Graph & Audience Sampling.")
    parser.add_argument("handle", type=str, help="Target YouTube channel handle (e.g. donflurgundy)")
    parser.add_argument("-r", "--results", type=int, default=8, help="Maximum candidates to store (default: 8)")
    args = parser.parse_args()

    try:
        disc = run_discovery(args.handle, max_results=args.results)
        print("==================================================")
        print(f"      DISCOVERY RESULTS FOR @{args.handle.lstrip('@').upper()}")
        print("==================================================")
        print(f"Target Unique Commenters Sampled: {disc['target_sampled_commenters']}")
        print(f"Subscriber Floor Enforced: >= {disc['min_subscribers_filter']:,} subscribers\n")

        for i, c in enumerate(disc["candidates"], 1):
            print(f"{i}. {c['channel_title']} (@{c['handle']})")
            print(f"   • Source: {c.get('discovery_source', 'N/A')}")
            print(f"   • Peer Score: {c.get('peer_score', 0.0):.2f}")
            print(f"   • Semantic Similarity: {c.get('semantic_similarity', 0.0):.2f} | Audience Affinity: {c.get('audience_affinity', 0.0):.4f}")
            print(f"   • Overlap: {c['shared_commenters_count']} shared users ({c['target_audience_overlap_pct']}%) | Jaccard: {c['jaccard_similarity']}")
            print(f"   • Rationale: {c['selection_rationale']}\n")
        print("==================================================")
    except Exception as e:
        print(f"[!] Discovery Error: {e}")