"""Posts and top comments for the subreddits in a study config.

Uses the official API when REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set.
Otherwise it tries reddit.com's public JSON, then Arctic Shift. Arctic Shift
returns comments by time, so each thread keeps the highest scores from the
newest 100. A subreddit with an empty queries list is read as a whole.
Other subreddits are searched.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from sociomancy.sources.cache import cached_fetch, clean_text, http_get_json
from sociomancy.sources.windows import iso_day, parse_day
from sociomancy.studies.config import Study, Subreddit

ARCTIC_ROOT = "https://arctic-shift.photon-reddit.com/api"
LISTING_PAUSE = 1.2
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _headers(study: Study, browser: bool = False) -> dict:
    agent = BROWSER_UA if browser else study.reddit_user_agent
    return {"User-Agent": agent, "Accept": "application/json"}


def _oauth_session(study: Study) -> requests.Session | None:
    """App-only token from REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."""
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    session = requests.Session()
    session.headers.update(_headers(study))
    response = session.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        timeout=20,
    )
    if response.status_code != 200:
        detail = ""
        try:
            detail = str(response.json().get("message") or response.json().get("error") or "")
        except ValueError:
            detail = response.reason
        print(f"  [warn] Reddit token request failed ({response.status_code} {detail}).")
        return None
    token = response.json().get("access_token")
    if not token:
        print("  [warn] Reddit token response had no access_token.")
        return None
    session.headers["Authorization"] = f"bearer {token}"
    return session


def _uses_oauth(session: requests.Session) -> bool:
    return "Authorization" in session.headers


def _listing_url(session: requests.Session, path: str) -> str:
    if _uses_oauth(session):
        return f"https://oauth.reddit.com{path}"
    return f"https://www.reddit.com{path}.json"


def _post_row(study: Study, data: dict) -> dict | None:
    created = data.get("created_utc")
    if not study.in_window(created):
        return None
    title = clean_text(data.get("title"))
    body = clean_text(data.get("selftext"))
    text = title if not body else f"{title}\n\n{body}"
    if not text:
        return None
    post_id = str(data.get("id") or "")
    return {
        "item_id": f"reddit:t3_{post_id}",
        "source": "reddit",
        "kind": "post",
        "text": text,
        "engagement": int(data.get("score") or data.get("ups") or 0),
        "created_at": str(created),
        "day": iso_day(created),
        "window": study.window_for(created),
        "group": str(data.get("subreddit") or ""),
        "thread_id": post_id,
        "post_title": title,
        "permalink": data.get("permalink") or "",
    }


def _comment_row(study: Study, data: dict, post_id: str, post_title: str) -> dict | None:
    if data.get("kind") and data.get("kind") != "t1":
        return None
    created = data.get("created_utc")
    text = clean_text(data.get("body"))
    if not text or not study.in_window(created):
        return None
    comment_id = str(data.get("id") or "")
    return {
        "item_id": f"reddit:t1_{comment_id}",
        "source": "reddit",
        "kind": "comment",
        "text": text,
        "engagement": int(data.get("score") or data.get("ups") or 0),
        "created_at": str(created),
        "day": iso_day(created),
        "window": study.window_for(created),
        "group": str(data.get("subreddit") or ""),
        "thread_id": post_id,
        "post_title": post_title,
        "permalink": data.get("permalink") or "",
    }


def _listing_children(payload: dict) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return []
    return list(data.get("children") or [])


def _fetch_subreddit_posts(
    study: Study,
    session: requests.Session,
    subreddit: Subreddit,
    limit: int,
) -> list[dict]:
    found: list[dict] = []
    seen: set[str] = set()
    after = None
    headers = _headers(study, browser=not _uses_oauth(session))

    for _ in range(6):
        if subreddit.whole_subreddit:
            url = _listing_url(session, f"/r/{subreddit.name}/new")
            params: dict = {"limit": 100, "raw_json": 1}
        else:
            url = _listing_url(session, f"/r/{subreddit.name}/search")
            params = {
                "q": study.reddit_search_query or " OR ".join(subreddit.queries),
                "restrict_sr": "on",
                "sort": "new",
                "t": "month",
                "limit": 100,
                "type": "link",
                "raw_json": 1,
            }
        if after:
            params["after"] = after
        payload = http_get_json(session, url, params=params, headers=headers, pause=LISTING_PAUSE)
        children = _listing_children(payload)
        if not children:
            break

        oldest = None
        for child in children:
            data = child.get("data") or {}
            post_id = str(data.get("id") or "")
            day = parse_day(data.get("created_utc"))
            if day is not None and (oldest is None or day < oldest):
                oldest = day
            if not post_id or post_id in seen:
                continue
            row = _post_row(study, data)
            if row is None:
                continue
            seen.add(post_id)
            found.append(row)
            if len(found) >= limit:
                return found

        after = (payload.get("data") or {}).get("after")
        if not after or (oldest is not None and oldest < study.span_start):
            break
    return found


def _fetch_top_comments(
    study: Study,
    session: requests.Session,
    subreddit: str,
    post: dict,
    limit: int,
) -> list[dict]:
    url = _listing_url(session, f"/r/{subreddit}/comments/{post['thread_id']}")
    params = {"sort": "top", "limit": limit, "depth": 1, "raw_json": 1}
    payload = http_get_json(
        session,
        url,
        params=params,
        headers=_headers(study, browser=not _uses_oauth(session)),
        pause=LISTING_PAUSE,
    )
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    rows = []
    for child in _listing_children(payload[1]):
        data = child.get("data") or {}
        if child.get("kind") != "t1":
            continue
        row = _comment_row(study, data, post["thread_id"], post["post_title"])
        if row:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _official_available(study: Study, session: requests.Session) -> bool:
    probe = study.subreddits[0].name if study.subreddits else "popular"
    try:
        response = session.get(
            f"https://www.reddit.com/r/{probe}/new.json",
            params={"limit": 1, "raw_json": 1},
            headers=_headers(study, browser=True),
            timeout=15,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200 and response.text.lstrip().startswith("{")


def _arctic_error(response: requests.Response) -> str:
    try:
        return str(response.json().get("error") or response.status_code)
    except ValueError:
        return str(response.status_code)


def _arctic_get(study: Study, session: requests.Session, path: str, params: dict) -> tuple[list[dict], bool]:
    """Return (rows, timed_out). A 422 here means Arctic Shift gave up on a heavy search."""
    for attempt in range(3):
        response = session.get(
            f"{ARCTIC_ROOT}/{path}",
            params=params,
            headers=_headers(study),
            timeout=45,
        )
        if response.status_code in {422, 429}:
            message = _arctic_error(response)
            if attempt < 2:
                wait = 4 * (attempt + 1)
                print(f"  [wait] Arctic Shift {response.status_code} ({message}); sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"  [skip] Arctic Shift {message}")
            return [], True
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        time.sleep(LISTING_PAUSE)
        return list(payload.get("data") or []), False
    return [], True


def _matches_terms(data: dict, terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    haystack = f"{data.get('title') or ''} {data.get('selftext') or ''}".lower()
    return any(term.lower() in haystack for term in terms)


def _arctic_pages(
    study: Study,
    session: requests.Session,
    subreddit: Subreddit,
    limit: int,
    found: list[dict],
    seen: set[str],
    title: str | None,
    text_filter: tuple[str, ...],
    max_pages: int,
) -> bool:
    """Append matching posts. Returns True when Arctic Shift times out."""
    window_after, window_before = study.epoch_bounds()
    before = window_before
    for _ in range(max_pages):
        if len(found) >= limit:
            return False
        params = {
            "subreddit": subreddit.name,
            "limit": min(100, max(limit, 25)),
            "after": window_after,
            "before": before,
            "sort": "desc",
        }
        if title:
            params["title"] = title
        page, timed_out = _arctic_get(study, session, "posts/search", params)
        if timed_out:
            return True
        if not page:
            return False
        oldest = None
        for data in page:
            created = data.get("created_utc")
            if created is not None and (oldest is None or created < oldest):
                oldest = created
            post_id = str(data.get("id") or "")
            if not post_id or post_id in seen or not _matches_terms(data, text_filter):
                continue
            row = _post_row(study, data)
            if row is None:
                continue
            seen.add(post_id)
            found.append(row)
            if len(found) >= limit:
                return False
        if oldest is None or len(page) < params["limit"] or int(oldest) >= before or int(oldest) <= window_after:
            return False
        before = int(oldest)
    return False


def _arctic_posts(study: Study, session: requests.Session, subreddit: Subreddit, limit: int) -> list[dict]:
    found: list[dict] = []
    seen: set[str] = set()
    if not subreddit.queries:
        _arctic_pages(study, session, subreddit, limit, found, seen, None, (), max_pages=4)
        return found[:limit]

    # `title` is indexed. `query` searches the whole subreddit and Arctic Shift
    # answers that with 422 Timeout on busy subs like r/GTA.
    for title in subreddit.queries:
        timed_out = _arctic_pages(
            study, session, subreddit, limit, found, seen, title, (), max_pages=2
        )
        if timed_out:
            break
        if len(found) >= limit:
            return found[:limit]
    if len(found) < limit:
        print(f"  [reddit] r/{subreddit.name} title search was short; scanning recent posts")
        _arctic_pages(
            study, session, subreddit, limit, found, seen, None, subreddit.queries, max_pages=4
        )
    return found[:limit]


def _arctic_comments(study: Study, session: requests.Session, post: dict, limit: int) -> list[dict]:
    page, _timed_out = _arctic_get(
        study,
        session,
        "comments/search",
        {"link_id": post["thread_id"], "limit": min(100, max(limit * 8, limit)), "sort": "desc"},
    )
    rows = []
    for data in page:
        row = _comment_row(study, data, post["thread_id"], post["post_title"])
        if row:
            rows.append(row)
    rows.sort(key=lambda row: row["engagement"], reverse=True)
    return rows[:limit]


def _fetch(study: Study, posts_per_sub: int, comments_per_post: int) -> dict:
    session = _oauth_session(study)
    if session is not None:
        print("  [reddit] using the official API")
        backend = "reddit-oauth"
        official = True
    else:
        session = requests.Session()
        official = _official_available(study, session)
        backend = "reddit" if official else "arctic-shift"
        if official:
            print("  [reddit] using reddit.com listings")
        else:
            print("  [reddit] reddit.com blocked this network; using Arctic Shift")

    posts: list[dict] = []
    comments: list[dict] = []
    for subreddit in study.subreddits:
        print(f"  [reddit] r/{subreddit.name}")
        try:
            if official:
                sub_posts = _fetch_subreddit_posts(study, session, subreddit, posts_per_sub)
            else:
                sub_posts = _arctic_posts(study, session, subreddit, posts_per_sub)
        except (requests.HTTPError, RuntimeError) as exc:
            print(f"  [skip] r/{subreddit.name}: {exc}")
            continue
        posts.extend(sub_posts)
        for post in sub_posts:
            try:
                if official:
                    comments.extend(
                        _fetch_top_comments(study, session, subreddit.name, post, comments_per_post)
                    )
                else:
                    comments.extend(_arctic_comments(study, session, post, comments_per_post))
            except (requests.HTTPError, RuntimeError) as exc:
                print(f"  [skip] comments for {post['thread_id']}: {exc}")
    return {"backend": backend, "posts": posts, "comments": comments}


def collect_reddit(
    study: Study,
    refresh: bool = False,
    posts_per_sub: int | None = None,
    comments_per_post: int | None = None,
) -> pd.DataFrame:
    if not study.subreddits:
        print("  [skip] No reddit.subreddits in the study config.")
        return pd.DataFrame()

    posts_per_sub = study.posts_per_sub if posts_per_sub is None else posts_per_sub
    comments_per_post = study.comments_per_post if comments_per_post is None else comments_per_post
    names = ", ".join(f"r/{sub.name}" for sub in study.subreddits)
    print(f"Collecting Reddit posts and top comments ({names})...")
    study.ensure_dirs()
    raw = cached_fetch(
        study.raw_dir / "reddit_raw.json",
        refresh,
        lambda: _fetch(study, posts_per_sub, comments_per_post),
    )
    frame = pd.DataFrame([*(raw.get("posts") or []), *(raw.get("comments") or [])])
    if frame.empty:
        print("  [warn] Reddit returned no posts or comments in the study window.")
        return frame

    frame = frame.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    out = study.processed_dir / "reddit_items.csv"
    frame.to_csv(out, index=False)
    counts = frame.groupby(["group", "kind"]).size().to_dict()
    print(f"  [saved] {out} ({len(frame)} rows; {counts})")
    return frame
