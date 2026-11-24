"""Load a study JSON file into the settings the collectors read."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sociomancy.sources.windows import parse_day


@dataclass(frozen=True)
class Window:
    name: str
    start: date
    end: date

    @property
    def published_after(self) -> str:
        return f"{self.start.isoformat()}T00:00:00Z"

    @property
    def published_before(self) -> str:
        # YouTube search.list treats publishedBefore as exclusive.
        return f"{(self.end + timedelta(days=1)).isoformat()}T00:00:00Z"


@dataclass(frozen=True)
class Subreddit:
    name: str
    queries: tuple[str, ...]

    @property
    def whole_subreddit(self) -> bool:
        return not self.queries


@dataclass(frozen=True)
class Study:
    path: Path
    id: str
    title: str
    focus_window: str
    windows: tuple[Window, ...]
    trend_keywords: tuple[str, ...]
    trend_timeframe: str
    metacritic_slug: str
    max_reviews: int
    negative_score_at_most: float
    date_slack_days: int
    reddit_user_agent: str
    reddit_search_query: str
    posts_per_sub: int
    comments_per_post: int
    subreddits: tuple[Subreddit, ...]
    youtube_query: str
    videos_per_window: int
    comments_per_video: int
    min_duration_sec: int
    min_comments: int
    youtube_order: str
    tiers: tuple[tuple[str, int], ...]
    complaint_labels: dict[str, str]
    top_per_thread: int
    metacritic_top_fraction: float

    @property
    def raw_dir(self) -> Path:
        return Path("data/raw") / self.id

    @property
    def processed_dir(self) -> Path:
        return Path("data/processed") / self.id

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def span_start(self) -> date:
        return min(window.start for window in self.windows)

    @property
    def span_end(self) -> date:
        return max(window.end for window in self.windows)

    def epoch_bounds(self) -> tuple[int, int]:
        """Inclusive start and exclusive end, as UTC unix seconds."""
        start = datetime.combine(self.span_start, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(self.span_end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        return int(start.timestamp()), int(end.timestamp())

    def window_for(self, value, slack_days: int = 0) -> str:
        day = parse_day(value)
        if day is None:
            return "unknown"
        for window in self.windows:
            if window.start <= day <= window.end:
                return window.name
        if slack_days and self.windows:
            last = self.windows[-1]
            if last.end < day <= last.end + timedelta(days=slack_days):
                return last.name
        return "outside"

    def in_window(self, value) -> bool:
        return self.window_for(value) not in {"unknown", "outside"}

    def channel_tier(self, subs: int, hidden: bool) -> str:
        if hidden:
            return "Unknown"
        for name, minimum in self.tiers:
            if subs >= minimum:
                return name
        return self.tiers[-1][0] if self.tiers else "Unknown"


def _require(data: dict, key: str, where: str):
    if key not in data:
        raise ValueError(f"{where} is missing '{key}'")
    return data[key]


def _parse_date(value: str, where: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{where} must be YYYY-MM-DD, got {value!r}") from exc


def load_study(path: str | Path) -> Study:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    study_id = str(_require(data, "id", config_path.name))
    if not study_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Study id {study_id!r} must be letters, numbers, '_' or '-'.")

    windows = []
    for index, raw in enumerate(_require(data, "windows", config_path.name)):
        where = f"windows[{index}]"
        windows.append(
            Window(
                name=str(_require(raw, "name", where)),
                start=_parse_date(_require(raw, "start", where), f"{where}.start"),
                end=_parse_date(_require(raw, "end", where), f"{where}.end"),
            )
        )
    if not windows:
        raise ValueError(f"{config_path} needs at least one window.")

    trends = data.get("trends") or {}
    keywords = tuple(trends.get("keywords") or [])
    timeframe = trends.get("timeframe") or f"{windows[0].start.isoformat()} {windows[-1].end.isoformat()}"

    metacritic = data.get("metacritic") or {}
    reddit = data.get("reddit") or {}
    youtube = data.get("youtube") or {}
    scoring = data.get("scoring") or {}

    subreddits = []
    for index, raw in enumerate(reddit.get("subreddits") or []):
        subreddits.append(
            Subreddit(
                name=str(_require(raw, "name", f"reddit.subreddits[{index}]")),
                queries=tuple(raw.get("queries") or []),
            )
        )

    youtube_order = str(youtube.get("order") or "viewCount")
    allowed_orders = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
    if youtube_order not in allowed_orders:
        raise ValueError(
            f"youtube.order must be one of {', '.join(sorted(allowed_orders))}, got {youtube_order!r}"
        )

    tiers = []
    for index, raw in enumerate(youtube.get("tiers") or []):
        where = f"youtube.tiers[{index}]"
        tiers.append((str(_require(raw, "name", where)), int(_require(raw, "min_subscribers", where))))
    tiers.sort(key=lambda item: item[1], reverse=True)

    complaints: dict[str, str] = {}
    for index, raw in enumerate(data.get("complaints") or []):
        where = f"complaints[{index}]"
        complaints[str(_require(raw, "hypothesis", where))] = str(_require(raw, "name", where))

    names = {window.name for window in windows}
    focus = str(data.get("focus_window") or windows[-1].name)
    if focus not in names:
        raise ValueError(f"focus_window {focus!r} is not one of {sorted(names)}")

    return Study(
        path=config_path,
        id=study_id,
        title=str(data.get("title") or study_id),
        focus_window=focus,
        windows=tuple(windows),
        trend_keywords=tuple(str(keyword) for keyword in keywords),
        trend_timeframe=str(timeframe),
        metacritic_slug=str(metacritic.get("slug") or ""),
        max_reviews=int(metacritic.get("max_reviews", 200)),
        negative_score_at_most=float(metacritic.get("negative_score_at_most", 4)),
        date_slack_days=int(metacritic.get("date_slack_days", 0)),
        reddit_user_agent=str(reddit.get("user_agent") or "sociomancy/0.1 (launch study; local research)"),
        reddit_search_query=str(reddit.get("search_query") or ""),
        posts_per_sub=int(reddit.get("posts_per_sub", 20)),
        comments_per_post=int(reddit.get("comments_per_post", 8)),
        subreddits=tuple(subreddits),
        youtube_query=str(youtube.get("query") or ""),
        videos_per_window=int(youtube.get("videos_per_window", 12)),
        comments_per_video=int(youtube.get("comments_per_video", 40)),
        min_duration_sec=int(youtube.get("min_duration_sec", 65)),
        min_comments=int(youtube.get("min_comments", 0)),
        youtube_order=youtube_order,
        tiers=tuple(tiers),
        complaint_labels=complaints,
        top_per_thread=int(scoring.get("top_per_thread", 3)),
        metacritic_top_fraction=float(scoring.get("metacritic_top_fraction", 0.2)),
    )
