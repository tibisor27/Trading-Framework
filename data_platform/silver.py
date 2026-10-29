"""
Silver: Bronze -> the contract (contract.py for what code can check, CONTRACT.md for what it means).
The same code for every source.

For each day Bronze has:   staging.bars  ->  conform  ->  check  ->  write  (or refuse)
Two outputs:               the day's parquet file, and one line in _status.json

Days whose Silver file is newer than all their Bronze inputs are skipped, so a second run does nothing.
"""
from __future__ import annotations

import logging
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalog, staging
from .contract import SILVER_BARS

log = logging.getLogger("data_platform.silver")

#list with all the columns(str)
COLUMNS = SILVER_BARS.column_names


# ------------------------------------------------------------------ the three common steps
#standardize function that ensures the columns names and types and index match the contract
#it renames or selects columns if needed to match the contract
def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Types and names only, never values or rows. The target shape comes from the contract."""
    out = df[COLUMNS].astype({c.name: c.dtype for c in SILVER_BARS.columns})
    # pandas 3 creates microsecond timestamps; the reference data is nanoseconds and they do not compare equal
    out.index = pd.DatetimeIndex(df.index).tz_convert("UTC").astype(SILVER_BARS.index.dtype)
    out.index.name = SILVER_BARS.index.name
    return out


def check(df: pd.DataFrame, day: date) -> list[str]:
    """
    The contract, enforced on the producer side (contract.py). Reports, never fixes:
    fixing in silence hides the cause (a repeated minute can be the symptom of a wrong clock).
    """
    return SILVER_BARS.validate(df, day)


def write(source: str, symbol: str, day: date, df: pd.DataFrame, lineage: dict) -> Path:
    """Lineage goes inside the file as metadata (pd.read_parquet(p).attrs), not as columns."""
    p = catalog.silver_file(source, symbol, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.attrs = lineage
    df.to_parquet(p)
    return p


def process_day(source: str, symbol: str, day: date, bars: pd.DataFrame, lineage: dict) -> list[str]:
    """conform -> check -> write, only if the checks pass. Returns the problems."""
    bars = conform(bars)
    problems = check(bars, day)
    if not problems and len(bars):  #len(bars) checks if df is empty or not
        write(source, symbol, day, bars, lineage)
    return problems


# ------------------------------------------------------------------ the job
def build(inst: str, rebuild_all: bool = False) -> list[str]:
    """Bronze -> Silver for one instrument. One loop, whatever the source."""
    spec = catalog.instrument(inst)
    source, symbol, stage = spec["source"], spec["symbol"], staging.for_source(spec["source"])
    when = catalog.ingested_at(source)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    problems, statuses, done = [], {}, 0

    for day in stage.available_days(inst):
        files = stage.bronze_files(inst, day)
        out = catalog.silver_file(source, symbol, day)

        # A day is skipped when its Silver file is newer than every Bronze file it was built from.
        # It is (re)built when:
        #   * it is new in Bronze, so there is no Silver file yet
        #   * a Bronze input changed after the last build (e.g. a new MT5 snapshot covering it)
        #   * it was rejected or empty last time, so no file was written
        #   * rebuild_all is set (--all)
        # What mtime cannot see (run --all after any of these):
        #   * code changes (conform, contract, staging): Bronze is unchanged, so nothing looks new
        #   * skipped days keep their old status and lineage
        #   * a parquet left half-written by a crash is newer than Bronze, so it is skipped too
        if not rebuild_all and out.exists() and out.stat().st_mtime >= max(f.stat().st_mtime for f in files):
            continue                                        # up to date with its Bronze inputs

        #df from raw data
        bars = stage.load_bars(inst, day)

        #clean file names for lineage
        clean_file_names = [catalog.bronze_name(f) for f in files]
        lineage = {"source": source, "instrument": inst, "bronze_files": clean_file_names,
                   "ingested_at": [when.get(n, "before the ingest log") for n in clean_file_names], "built_at": now}
        
        #conform -> check -> write df to silver and creates status for that day
        found = process_day(source, symbol, day, bars, lineage)
        statuses[day.isoformat()] = (
            {"status": "rejected", "problems": found, "checked_at": now, "serving_previous": out.exists()}
            if found else {"status": "empty", "checked_at": now} if len(bars) == 0
            else {"status": "ok", "rows": int(len(bars)), "checked_at": now})
        problems += [f"{day}: {p}" for p in found]
        done += 1

    _update_status(source, symbol, statuses)
    log.info(f"silver {source}/{symbol}: {done} days processed, {len(problems)} rejected")
    for p in problems[:20]:
        log.info(f"  {p}")
    return problems



# ------------------------------------------------------------------ the report
#Data Quality and Logs - when the contract is not respected, or a sample of data is missing in the GOLD layer
# this audit file stores the status of every day
def read_status(source: str, symbol: str) -> dict[str, dict]:
    """
    {"2026-09-08": {"status": "ok", "rows": 1440, ...}, ...}
      ok        in Silver, passed every check
      empty     the source had no bars for that day
      rejected  broke the contract; `problems` says why; not in Silver (or an older good version is
                still served: `serving_previous`)
    Gold reads this to tell "no data" from "data refused" instead of guessing from a missing file.
    """
    p = catalog.status_file(source, symbol)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _update_status(source: str, symbol: str, updates: dict[str, dict]) -> None:
    if updates:
        p = catalog.status_file(source, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        merged = {**read_status(source, symbol), **updates}
        p.write_text(json.dumps(dict(sorted(merged.items())), indent=1), encoding="utf-8")


# ------------------------------------------------------------------ for consumers
def read(source: str, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    folder = catalog.SILVER / source / symbol
    files = [p for p in sorted(folder.glob("*.parquet"))
             if (not start or date.fromisoformat(p.stem) >= start) and (not end or date.fromisoformat(p.stem) <= end)]
    if not files:
        raise FileNotFoundError(f"nothing in {folder}")
    df = pd.concat(pd.read_parquet(p) for p in files)
    df.index = df.index.astype(SILVER_BARS.index.dtype)
    return df


def parity(reference: Path, source: str = "dukascopy", symbol: str = "US100") -> tuple[bool, str]:
    """Silver vs the file the edge was measured on, on the days both have. Must be identical."""
    s = read(source, symbol)
    ref = pd.read_parquet(reference)
    ref.index = pd.DatetimeIndex(ref.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    common = s.index.normalize().unique().intersection(ref.index.normalize().unique())
    s, ref = s[s.index.normalize().isin(common)], ref[ref.index.normalize().isin(common)]
    same = s.index.equals(ref.index)
    diff = float(np.abs(s[COLUMNS].to_numpy() - ref[COLUMNS].to_numpy(dtype="float64")).max()) if same else float("nan")
    ok = same and diff == 0.0
    return ok, (f"{'PASS' if ok else 'FAIL'}: {len(common)} common days, {len(s)} silver bars, "
                f"{len(ref)} reference bars, max |diff| = {diff:g}")
