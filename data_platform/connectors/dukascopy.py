"""Dukascopy datafeed -> Bronze. One compressed file per instrument and UTC day, public, no key."""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

from .. import catalog

log = logging.getLogger("data_platform.connector.dukascopy")
URL = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def url(inst: str, day: date) -> str:
    return URL.format(inst=inst, y=day.year, m=day.month - 1, d=day.day)     # months are 0-based


def bronze_file(inst: str, day: date):
    return catalog.bronze_dir("dukascopy", inst) / f"{day.isoformat()}.bi5"


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
        except Exception as e :  # noqa: BLE001 - retried below
            log.error("Attempt %d for %s failed: %s", attempt + 1, u, e)

        time.sleep(pause_s * (attempt + 1))
    raise ConnectionError(f"could not download {u}")


def run(inst: str, start: date | None = None, end: date | None = None, fetch=fetch) -> list[str]:
    """
    Bring the days missing from Bronze. Three rules, each from a problem that already happened:
      * today is never asked for: a day is published after it ends (2026-09-10 was saved empty forever);
      * a 404 saves nothing, and a 0-byte file left by the old script counts as missing;
      * a network failure is returned as a problem, never skipped in silence (28 days were lost that way).
    """
    start = start or date(2018, 1, 1)
    end = min(end or date.max, datetime.now(timezone.utc).date() - timedelta(days=1))
    problems, saved, day = [], 0, start
    while day <= end:
        #idempotency check
        f = bronze_file(inst, day)
        if not (f.exists() and f.stat().st_size > 0):
            try:
                raw = fetch(inst, day)
                if raw:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(raw)
                    catalog.bronze_metadata("dukascopy", inst, f, day=day.isoformat())
                    saved += 1
            except ConnectionError as e:
                problems.append(f"{day}: {e}")
        day += timedelta(days=1)
    print(f"extract dukascopy/{inst}: {saved} new days, {len(problems)} failed")
    return problems
