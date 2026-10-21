"""
Bronze: what each source sent, unchanged, plus one line of metadata per saved file.

    data/bronze/dukascopy/dukascopy_cache/<INSTRUMENT>/<YYYY-MM-DD>.bi5     bytes as downloaded
    data/bronze/ftmo/<INSTRUMENT>/<YYYY-MM-DDTHHMMSSZ>.parquet              one MT5 extraction
    data/bronze/<source>/_ingest_log.jsonl                                  who, what, when

The metadata is written here, at ingestion, because this is the only moment it is known.
"""
from __future__ import annotations

import logging
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze"
FOLDER = {"dukascopy": "dukascopy/dukascopy_cache", "mt5": "ftmo"}
LOG_FOLDER = {"dukascopy": "dukascopy", "mt5": "ftmo"}

log = logging.getLogger("data_pipeline.bronze")

def _folder(source: str, inst: str) -> Path:
    return BRONZE / FOLDER[source] / inst


def _log(source: str, inst: str, path: Path, extra: dict) -> None:
    rec = {"source": source, "instrument": inst, "file": path.relative_to(BRONZE).as_posix(),
           "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "bytes": path.stat().st_size, **extra}
    log = BRONZE / LOG_FOLDER[source] / "_ingest_log.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


#dukascopy downloads finished days of data (with year-month-day format)
def save_dukascopy(inst: str, day: date, raw: bytes) -> Path:
    p = _folder("dukascopy", inst) / f"{day.isoformat()}.bi5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    _log("dukascopy", inst, p, {"day": day.isoformat()})
    return p


#mt5 downloads snapshots/extracts of the current day with the time of download (with year-month-dayTHHMMSSZ format)
def save_mt5(inst: str, rates, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    p = _folder("mt5", inst) / f"{now:%Y-%m-%dT%H%M%SZ}.parquet"        # names sort by time
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rates).to_parquet(p, index=False)
    _log("mt5", inst, p, {"rows": len(rates)})
    return p


def ingested_at(source: str) -> dict[str, str]:
    """file -> when it was ingested, from the log. Files saved before the log existed are missing."""
    log = BRONZE / LOG_FOLDER[source] / "_ingest_log.jsonl"
    out = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["file"]] = rec["ingested_at"]
                # log.info(f"[INGESTED] {out}")
            except (ValueError, KeyError):
                continue          # a half-written last line after a crash must not stop the pipeline
    return out
