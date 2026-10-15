"""Dukascopy datafeed: one compressed file per instrument and UTC day, public, no key."""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def url(inst: str, day: date) -> str:
    return URL.format(inst=inst, y=day.year, m=day.month - 1, d=day.day)     # months are 0-based


def last_published_day() -> date:
    """A day is published after it ends. Asking for today returns 404, and the old script saved that
    404 as an empty file forever (2026-09-10)."""
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def _http_get(u: str) -> tuple[int, bytes]:
    import requests
    r = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    return r.status_code, r.content


def fetch(inst: str, day: date, get=_http_get, pause_s: float = 1.5) -> bytes | None:
    """
    The raw .bi5 bytes for one day, or None if Dukascopy has no file (404).
    Raises after 3 failed attempts, so a network problem is never mistaken for "no data".
    """
    u = url(inst, day)
    for attempt in range(3):
        try:
            status, body = get(u)
            if status == 200 and body:
                return body
            if status in (200, 404):
                return None
        except Exception:  # noqa: BLE001 - retried below
            pass
        time.sleep(pause_s * (attempt + 1))
    raise ConnectionError(f"could not download {u}")
