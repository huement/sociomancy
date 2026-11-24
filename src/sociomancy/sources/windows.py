"""Date parsing shared by the study collectors."""

from __future__ import annotations

from datetime import date, datetime, timezone


def parse_day(value) -> date | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return None
        stamp = float(value)
        if stamp > 10_000_000_000:
            stamp = stamp / 1000.0
        if stamp > 1_000_000_000:
            return datetime.fromtimestamp(int(stamp), tz=timezone.utc).date()
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if text.replace(".", "", 1).isdigit() and len(text) >= 10:
        return parse_day(float(text))
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def iso_day(value) -> str:
    day = parse_day(value)
    return day.isoformat() if day else ""
