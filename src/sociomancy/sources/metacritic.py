"""Metacritic user reviews for the slug in a study config.

Uses the keyless JSON endpoint the site itself calls. Reviews are sorted
newest-first and paging stops at the first window's start or at max_reviews.
Score-only ratings stay in the raw cache and in metacritic_scores.csv.
Rows with written text go to metacritic_reviews.csv for the ONNX pass.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import requests

from sociomancy.sources.cache import cached_fetch, clean_text, http_get_json
from sociomancy.sources.windows import iso_day
from sociomancy.studies.config import Study

# Public key shipped in Metacritic's frontend. Used only if the keyless call is rejected.
PUBLIC_API_KEY = "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
PAGE_SIZE = 50


def _reviews_url(slug: str) -> str:
    return f"https://backend.metacritic.com/reviews/metacritic/user/games/{slug}/web"


def _author_name(author) -> str:
    if isinstance(author, dict):
        return str(author.get("name") or author.get("username") or "")
    return str(author or "")


def _items(payload: dict) -> tuple[list[dict], int]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        data = payload if isinstance(payload, dict) else {}
    items = data.get("items") or data.get("reviews") or []
    total = int(data.get("totalResults") or data.get("total") or 0)
    return list(items), total


def _fetch_pages(study: Study, max_reviews: int) -> dict:
    session = requests.Session()
    collected: list[dict] = []
    offset = 0
    total = None
    use_key = False
    url = _reviews_url(study.metacritic_slug)
    stop_before = study.span_start.isoformat()

    while len(collected) < max_reviews:
        params = {
            "offset": offset,
            "limit": PAGE_SIZE,
            "filterBySentiment": "all",
            "sort": "date",
        }
        if use_key:
            params["apiKey"] = PUBLIC_API_KEY
        try:
            payload = http_get_json(session, url, params=params, headers=HEADERS, pause=0.4)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403} and not use_key:
                print("  [warn] Keyless Metacritic call was rejected; retrying with the public site key.")
                use_key = True
                continue
            raise

        items, reported_total = _items(payload)
        if total is None:
            total = reported_total
        if not items:
            break

        collected.extend(items)
        dates = [iso_day(item.get("date") or item.get("createdAt")) for item in items]
        dates = [day for day in dates if day]
        if dates and min(dates) < stop_before:
            break
        offset += PAGE_SIZE
        if total and offset >= total:
            break

    return {
        "slug": study.metacritic_slug,
        "total_reported": total or 0,
        "items": collected[:max_reviews],
    }


def _review_id(item: dict, text: str, created: str) -> str:
    raw_id = item.get("id") or item.get("reviewId")
    if raw_id:
        return f"metacritic:{raw_id}"
    digest = hashlib.sha1(f"{created}|{item.get('score')}|{text[:160]}".encode()).hexdigest()[:16]
    return f"metacritic:{digest}"


def _review_row(study: Study, item: dict) -> dict:
    created = str(item.get("date") or item.get("createdAt") or "")
    text = clean_text(item.get("quote") or item.get("review") or item.get("text"))
    return {
        "item_id": _review_id(item, text, created),
        "source": "metacritic",
        "kind": "review",
        "text": text,
        "engagement": int(item.get("thumbsUp") or item.get("thumbs_up") or 0),
        "created_at": created,
        "day": iso_day(created),
        "window": study.window_for(created, slack_days=study.date_slack_days),
        "group": str(item.get("platform") or "all"),
        "thread_id": str(item.get("platform") or "metacritic"),
        "user_score": item.get("score"),
        "author": _author_name(item.get("author")),
    }


def collect_metacritic(
    study: Study,
    refresh: bool = False,
    max_reviews: int | None = None,
) -> pd.DataFrame:
    if not study.metacritic_slug:
        print("  [skip] No metacritic.slug in the study config.")
        return pd.DataFrame()

    max_reviews = study.max_reviews if max_reviews is None else max_reviews
    print(f"Collecting Metacritic user reviews for {study.metacritic_slug} (up to {max_reviews})...")
    study.ensure_dirs()
    raw = cached_fetch(
        study.raw_dir / "metacritic_reviews_raw.json",
        refresh,
        lambda: _fetch_pages(study, max_reviews),
    )
    rows = [_review_row(study, item) for item in raw.get("items", [])]
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  [warn] Metacritic returned no reviews.")
        return frame

    frame = frame[frame["window"] != "outside"].reset_index(drop=True)
    scores = (
        frame.assign(user_score=pd.to_numeric(frame["user_score"], errors="coerce"))
        .dropna(subset=["user_score"])
        .groupby(["window", "user_score"], dropna=False)
        .size()
        .reset_index(name="reviews")
    )
    scores_path = study.processed_dir / "metacritic_scores.csv"
    scores.to_csv(scores_path, index=False)

    written = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    out = study.processed_dir / "metacritic_reviews.csv"
    written.to_csv(out, index=False)
    print(
        f"  [saved] {out} ({len(written)} with text, "
        f"{len(frame) - len(written)} score-only; reported total {raw.get('total_reported', 0)})"
    )
    return written
