"""
The orchestrator: decides what to run and in which order. It is the only module that touches both
the connectors and the disk.

    ingest:  connector -> Bronze (+ metadata)
    silver:  Bronze -> parse -> clean -> standardize -> check -> write Silver (+ lineage)
    run:     ingest, then silver

Silver always reads from Bronze, even right after ingesting, so there is one path from source to
Silver and every Silver day can be rebuilt from Bronze alone.
"""

from __future__ import annotations
from pathlib import Path

from datetime import date, datetime, timedelta, timezone

import logging
import pandas as pd

from . import bronze, instruments, quality, silver
from .standardize import standardize
from .connectors import dukascopy as duka_conn
from .connectors import mt5 as mt5_conn
from .parsers import dukascopy as duka_parse
from .parsers import mt5 as mt5_parse

log = logging.getLogger('data_pipeline.orchestrator')

#in momentul de fata orchestratorul poate face 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 1. ingestare (broker + salvare in bronze layer)
# 2. construire silver layer (bronze -> parse -> clean -> standardize -> check -> write Silver (+ lineage))

# functiile nu au coeziune mica, si nu sunt OCP, pe viitor ramane de modualrizat astfel:
# - cream pipeline modular de ingestare si silver agnostic la broker(adica merge indiferent de broker-daca adaugam altul, il adaugam si gata, fara sa modificam orchestratorul)
# - o facem prin interfata comuna de tipul:class DataSourcePipeline(Protocol):
#                                                     def ingest(): ...
#                                                     def build_silver()
# iar fiecare broker va aveam implementarile corespunzatoare, iar orchestratorul face parte de ingest sau silver indiferent de broker
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ------------------------------------------------------------------ ingest
#handles the connection to brokers and saving raw data into bronze layer
def ingest(inst: str, start: date | None = None, end: date | None = None, fetch=None, spec: dict | None = None) -> list[str]:
    """Bring new data into Bronze. Returns the problems (empty list = all good)."""
    spec = spec or instruments.get(inst)
    if spec["source"] == "mt5":
        rates = (fetch or mt5_conn.fetch)(inst, spec["server_tz"])
        p = bronze.save_mt5(inst, rates)
        print(f"bronze: {p}")
        return []

    # Dukascopy: one file per day. Ask only for days not in Bronze yet. A 0-byte file left by the old
    # script counts as missing. A 404 saves nothing, so nothing gets stuck as "empty".
    fetch = fetch or duka_conn.fetch
    start = start or date(2018, 1, 1)
    end = min(end or date.max, duka_conn.last_published_day())
    failed, saved, day = [], 0, start
    while day <= end:

        #idempotenta: daca fisierul exista pe disc(si are date), sar peste
        f = bronze.dukascopy_file(inst, day)
        if not (f.exists() and f.stat().st_size > 0):
            try:
                #download 
                raw = fetch(inst, day)
                if raw:
                    #ingest into bronze layer
                    bronze.save_dukascopy(inst, day, raw, duka_conn.url(inst, day))
                    saved += 1
            except ConnectionError as e:
                failed.append(f"{day}: {e}")
        day += timedelta(days=1)
    log.info(f"bronze: {saved} new days for {inst}, {len(failed)} failed")
    
    return failed


# ------------------------------------------------------------------ silver
def to_silver(source: str, symbol: str, day: date, bars: pd.DataFrame, lineage: dict) -> list[str]:
    """The common half of every source: standardize, check, write only if the checks pass."""
    bars = standardize(bars)
    problems = quality.check(bars, day)
    if not problems and len(bars):
        silver.write(source, symbol, day, bars, lineage)
    return problems


def build_silver(inst: str, rebuild_all: bool = False, start: date | None = None, end: date | None = None) -> list[str]:
    """Bronze -> Silver. Returns the problems found (days with problems are not written)."""
    spec = instruments.get(inst)
    source, symbol = spec["source"], spec["symbol"]
    when = bronze.ingested_at(source)
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    problems, written = [], 0

    #fucntia transforma fiecare cale absoluta a fiserului in cale relativa(pt orice sistem de operare)
    def lineage(files: list[Path]) -> dict:
        rel = [f.relative_to(bronze.BRONZE).as_posix() for f in files]
        return {"source": source, "instrument": inst, "bronze_files": rel,
                "ingested_at": [when.get(r, "before the ingest log") for r in rel], "built_at": built_at}

    if source == "dukascopy":
        for f in sorted(bronze.folder(source, inst).glob("*.bi5")):
            if f.stat().st_size == 0:
                continue                                     # skip empty / corrupted files
            day = date.fromisoformat(f.stem)
            if (start and day < start) or (end and day > end):
                continue
            out = silver.path(source, symbol, day)
            if not rebuild_all and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
                continue                                     # already built from this exact file
            bars = duka_parse.parse(f.read_bytes(), day, spec["decimals"])      # no cleaning needed
            if bars.empty:
                continue
            log.info(f"[BRONZE -> SILVER] {f.relative_to(bronze.BRONZE).as_posix()} -> {out.relative_to(silver.SILVER).as_posix()}")
            problems += [f"{day}: {p}" for p in to_silver(source, symbol, day, bars, lineage([f]))]
            written += 1
    else:
        files = sorted(bronze.folder(source, inst).glob("*.parquet"))        # oldest first
        parsed = [mt5_parse.parse(pd.read_parquet(f), spec["server_tz"]).assign(_file=i)
                  for i, f in enumerate(files)]
        bars = mt5_parse.clean(parsed)
        for day, df in bars.groupby(bars.index.date):        # every day, every time: there are few
            if (start and day < start) or (end and day > end):
                continue
            used = [files[i] for i in sorted(df["_file"].unique())]
            problems += [f"{day}: {p}" for p in
                         to_silver(source, symbol, day, df.drop(columns="_file"), lineage(used))]
            written += 1

    print(f"silver: {written} days processed for {inst}, {len(problems)} problems")
    for p in problems[:20]:
        print(f"  {p}")
    return problems
