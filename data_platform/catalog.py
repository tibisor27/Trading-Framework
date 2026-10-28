"""
SSOT - Single Source of Truth) for the data platform.
The catalog: which datasets exist, where they live, and which instruments feed them.
It is the only file that knows paths. Kedro calls this the Data Catalog; dbt keeps it in sources.yml.

    data/bronze/dukascopy/<INSTRUMENT>/<YYYY-MM-DD>.bi5   what Dukascopy sent
    data/bronze/ftmo/<INSTRUMENT>/<YYYY-MM-DDTHHMMSSZ>.parquet           one MT5 extraction
    data/bronze/<source>/_ingest_log.jsonl                               who, what, when
    data/silver/<source>/<symbol>/<YYYY-MM-DD>.parquet                   the contract (CONTRACT.md)
    data/silver/<source>/<symbol>/_status.json                           ok / empty / rejected per day
    data/_runs/<dag>/<run_id>.json                                       what the orchestrator did
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
RUNS = ROOT / "data" / "_runs"

# decimals: Dukascopy prices are integers scaled by 10**decimals. DEUIDXEUR is 3: the old download
#   script used 5, so data/raw/DEUIDXEUR_1m.parquet is 100x too small.
# server_tz: the MT5 server clock. FTMO runs GMT+2 with EU summer time, the same rule as Europe/Bucharest.
INSTRUMENTS = {
    "USATECHIDXUSD": {"source": "dukascopy", "symbol": "US100", "decimals": 3},
    "USA500IDXUSD": {"source": "dukascopy", "symbol": "US500", "decimals": 3},
    "DEUIDXEUR": {"source": "dukascopy", "symbol": "GER40", "decimals": 3},
    "US100.cash": {"source": "mt5", "symbol": "US100", "server_tz": "Europe/Bucharest"},
    "XAUUSD": {"source": "dukascopy", "symbol": "XAUUSD", "decimals": 3},
}
_BRONZE_DIR = {"dukascopy": "dukascopy", "mt5": "ftmo"}
_LOG_DIR = {"dukascopy": "dukascopy", "mt5": "ftmo"}


def instrument(name: str) -> dict:
    if name not in INSTRUMENTS:
        raise KeyError(f"unknown instrument {name!r}; known: {sorted(INSTRUMENTS)}")
    return {"name": name, **INSTRUMENTS[name]}


# ------------------------------------------------------------------ Bronze
def bronze_dir(source: str, inst: str) -> Path:
    return BRONZE / _BRONZE_DIR[source] / inst


def bronze_metadata_path(source: str) -> Path:
    return BRONZE / _LOG_DIR[source] / "_ingest_metadata.jsonl"


def bronze_metadata(source: str, inst: str, path: Path, **extra) -> None:
    """One line per file saved in Bronze. Written at ingestion: the only moment we know where and when."""
    metadata = {"source": source, "instrument": inst, "file": path.relative_to(BRONZE).as_posix(),
           "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "bytes": path.stat().st_size, **extra}
    log = bronze_metadata_path(source)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")


#this ingested_At metadata is used further by the silver pipeline 
def ingested_at(source: str) -> dict[str, str]:
    """Bronze file -> when it was ingested. Files saved before the log existed are missing."""
    out, log = {}, bronze_metadata_path(source)
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["file"]] = rec["ingested_at"]
            except (ValueError, KeyError):
                continue          # a half-written last line after a crash must not stop the pipeline
    return out


#transform the absolute path to a relative path in the catalog.py file (data/bronze/source/instrument/file.ext)
#this is used in the lineage to track the files
#also handle the mac -> windows path conversion
#e.g: FROM /Users/tybyboiciuc/Downloads/trading_framework/data/bronze/dukascopy/US100/2026-10-23.bi5
#.    TO  dukascopy/US100/2026-10-23.bi5
def bronze_name(path: Path) -> str:
    return path.relative_to(BRONZE).as_posix()


# ------------------------------------------------------------------ Silver
def silver_file(source: str, symbol: str, day: date) -> Path:
    return SILVER / source / symbol / f"{day.isoformat()}.parquet"


def status_file(source: str, symbol: str) -> Path:
    return SILVER / source / symbol / "_status.json"
