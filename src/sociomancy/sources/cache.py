"""Raw-response cache shared by the study collectors."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import requests


def cached_fetch(cache_file: Path, refresh: bool, fetch_fn: Callable[[], Any]) -> Any:
    """Return cached JSON, or call fetch_fn and write it. Failures are not cached."""
    if cache_file.exists() and not refresh:
        print(f"  [cache] Loaded {cache_file}")
        return json.loads(cache_file.read_text(encoding="utf-8"))
    payload = fetch_fn()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  [saved] Cached raw response to {cache_file}")
    return payload


def clean_text(value: Any) -> str:
    text = " ".join(str(value or "").replace("\r", " ").split())
    if text.lower() in {"", "nan", "[removed]", "[deleted]", "none"}:
        return ""
    return text


def http_get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    pause: float = 0.0,
    timeout: float = 30,
) -> Any:
    """GET JSON, backing off on HTTP 429. Other errors raise."""
    response = None
    for attempt in range(3):
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 429 and attempt < 2:
            wait = 5 * (attempt + 1)
            print(f"  [wait] 429 from {url}; sleeping {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        if pause:
            time.sleep(pause)
        return response.json()
    if response is not None:
        response.raise_for_status()
    raise RuntimeError(f"No response from {url}")
