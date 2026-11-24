"""Google Trends interest for the keywords in a study config.

Web search is requested with the keywords together so they share one 0–100
scale. `interest` is the higher web value on each day. The same query is also
run for YouTube search, related queries, and country interest. Those calls are
cached separately so a quota miss on one of them does not drop the web series.
"""

from __future__ import annotations

import time

import pandas as pd
from trendspy import Trends

from sociomancy.sources.cache import cached_fetch
from sociomancy.studies.config import Study


def _records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.drop(columns=["isPartial"], errors="ignore")
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    value_cols = [col for col in frame.columns if col != "date"]
    frame[value_cols] = frame[value_cols].apply(pd.to_numeric, errors="coerce")
    return frame.groupby("date", as_index=False)[value_cols].max().to_dict(orient="records")


def _interest_frame(records: list[dict], suffix: str = "") -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    value_cols = [col for col in frame.columns if col != "date"]
    frame[value_cols] = frame[value_cols].apply(pd.to_numeric, errors="coerce")
    if suffix:
        frame = frame.rename(columns={col: f"{col}{suffix}" for col in value_cols})
    return frame


def _related_records(payload: dict, window_name: str) -> list[dict]:
    rows = []
    for kind in ("top", "rising"):
        for item in payload.get(kind) or []:
            raw = item.get("value")
            number = pd.to_numeric(raw, errors="coerce")
            rows.append(
                {
                    "window": window_name,
                    "kind": kind,
                    "query": item.get("query") or item.get("title") or "",
                    "value": "" if raw is None else str(raw),
                    "value_num": None if pd.isna(number) else float(number),
                }
            )
    return rows


def _region_records(frame: pd.DataFrame, keyword: str) -> list[dict]:
    if frame.empty:
        return []
    value_col = keyword if keyword in frame.columns else ""
    if not value_col:
        skip = {"geoName", "geoCode", "lat", "lng"}
        numeric = [col for col in frame.columns if col not in skip]
        value_col = numeric[0] if numeric else ""
    if not value_col or "geoName" not in frame.columns:
        return []
    columns = ["geoName", value_col] + (["geoCode"] if "geoCode" in frame.columns else [])
    keep = frame[columns].copy()
    keep = keep.rename(columns={value_col: "interest", "geoName": "country"})
    keep["interest"] = pd.to_numeric(keep["interest"], errors="coerce")
    keep = keep.dropna(subset=["interest"]).sort_values("interest", ascending=False)
    keep["keyword"] = keyword
    return keep.head(15).to_dict(orient="records")


def collect_trends(study: Study, refresh: bool = False) -> pd.DataFrame:
    if not study.trend_keywords:
        print("  [skip] No trends.keywords in the study config.")
        return pd.DataFrame()

    print(f"Collecting Google Trends for {study.title}...")
    study.ensure_dirs()
    keywords = list(study.trend_keywords)
    primary = keywords[0]
    client = Trends(request_delay=4, max_retries=1)

    def _web() -> list[dict]:
        return _records(client.interest_over_time(keywords, timeframe=study.trend_timeframe))

    web = _interest_frame(cached_fetch(study.raw_dir / "google_trends_raw.json", refresh, _web))
    if web.empty:
        print("  [warn] Google Trends returned no rows.")
        return web

    def _youtube() -> list[dict]:
        time.sleep(20)
        return _records(
            client.interest_over_time(keywords, timeframe=study.trend_timeframe, gprop="youtube")
        )

    try:
        youtube = _interest_frame(
            cached_fetch(study.raw_dir / "google_trends_youtube_raw.json", refresh, _youtube),
            suffix=" (YouTube)",
        )
    except Exception as exc:
        print(f"  [warn] YouTube-search Trends failed ({exc}). Web series still saved.")
        youtube = pd.DataFrame()

    related_rows: list[dict] = []
    for window in study.windows:
        def _one_window(window=window) -> list[dict]:
            time.sleep(20)
            timeframe = f"{window.start.isoformat()} {window.end.isoformat()}"
            payload = client.related_queries(
                primary,
                timeframe=timeframe,
                headers={"referer": "https://www.google.com/"},
            )
            packed = {
                kind: frame.to_dict(orient="records") if isinstance(frame, pd.DataFrame) else []
                for kind, frame in (payload or {}).items()
            }
            return _related_records(packed, window.name)

        try:
            related_rows.extend(
                cached_fetch(
                    study.raw_dir / f"google_trends_related_{window.name}.json",
                    refresh,
                    _one_window,
                )
            )
        except Exception as exc:
            print(f"  [warn] Related queries for {window.name} failed ({exc}).")

    def _region() -> list[dict]:
        time.sleep(20)
        frame = client.interest_by_region(
            primary,
            timeframe=study.trend_timeframe,
            resolution="COUNTRY",
        )
        return _region_records(frame, primary)

    try:
        region_rows = cached_fetch(study.raw_dir / "google_trends_region_raw.json", refresh, _region)
    except Exception as exc:
        print(f"  [warn] Country interest failed ({exc}).")
        region_rows = []

    value_cols = [col for col in web.columns if col != "date"]
    web["interest"] = web[value_cols].max(axis=1)
    web["window"] = web["date"].map(study.window_for)
    if not youtube.empty:
        web = web.merge(youtube, on="date", how="left")

    out = study.processed_dir / "google_trends.csv"
    web.to_csv(out, index=False)
    print(f"  [saved] {out} ({len(web)} days)")

    related = pd.DataFrame(
        related_rows,
        columns=["window", "kind", "query", "value", "value_num"],
    )
    related_path = study.processed_dir / "google_trends_related.csv"
    related.to_csv(related_path, index=False)
    print(f"  [saved] {related_path} ({len(related)} queries)")

    region = pd.DataFrame(region_rows)
    region_path = study.processed_dir / "google_trends_region.csv"
    region.to_csv(region_path, index=False)
    print(f"  [saved] {region_path} ({len(region)} countries)")
    return web
