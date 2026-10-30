"""
Run one step by hand. For the whole chain on a schedule, use the orchestrator (python -m data_platform.orchestration).

    python -m data_platform extract USATECHIDXUSD --start 2026-09-01    # source -> Bronze
    python -m data_platform extract US100.cash                          # MT5 -> Bronze (Windows)
    python -m data_platform silver  USATECHIDXUSD                       # Bronze -> Silver
    python -m data_platform status  USATECHIDXUSD                       # ok / empty / rejected per day
    python -m data_platform parity                                      # Silver vs data/raw reference
"""
import argparse
import sys
from datetime import date

import logging
from . import catalog, silver
from .connectors import dukascopy as extract_dukascopy
from .connectors import mt5 as extract_mt5
from src.logging_config import setup_log
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("data_platform_entry")

def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m data_platform")
    ap.add_argument("cmd", choices=["extract", "silver", "status", "parity"])
    ap.add_argument("instrument", nargs="?", default="USATECHIDXUSD")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None, help="Dukascopy: last day to download (inclusive, YYYY-MM-DD)")
    ap.add_argument("--bars", type=int, default=None, help="MT5: how many bars to extract")
    ap.add_argument("--all", action="store_true", help="rebuild every Silver day")
    ap.add_argument("--reference", default="data/raw/USATECHIDXUSD_1m.parquet")
    a = ap.parse_args()

    setup_log(ROOT / "logs/data_platform.log")

    if a.cmd == "parity":
        ok, msg = silver.parity(catalog.ROOT / a.reference)
        print(msg)
        return 0 if ok else 1

    spec = catalog.instrument(a.instrument)
    if a.cmd == "extract":
        if spec["source"] == "mt5":
            return 1 if extract_mt5.run(a.instrument, a.bars) else 0
        start = date.fromisoformat(a.start) if a.start else None
        end   = date.fromisoformat(a.end)   if a.end   else None
        return 1 if extract_dukascopy.run(a.instrument, start, end) else 0

    if a.cmd == "silver":
        return 1 if silver.build(a.instrument, a.all) else 0

    st = silver.read_status(spec["source"], spec["symbol"])           # status
    counts = {}
    for v in st.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(f"{spec['source']}/{spec['symbol']}: {counts}")
    for day, v in st.items():
        if v["status"] == "rejected":
            print(f"  {day}: {'; '.join(v['problems'])}"
                  f"{'  (older version still served)' if v.get('serving_previous') else ''}")
    return 1 if counts.get("rejected") else 0


sys.exit(main())
