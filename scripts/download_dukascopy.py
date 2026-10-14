"""
Download 1-minute candles from Dukascopy's public datafeed (no API key) into a parquet file.

    python scripts/download_dukascopy.py --instrument USATECHIDXUSD --start 2018-01-01 --end 2026-09-01
    python scripts/download_dukascopy.py --instrument USA500IDXUSD  --start 2018-01-01

Instruments of interest: USATECHIDXUSD (Nasdaq 100 CFD), USA500IDXUSD (S&P 500 CFD),
EURUSD, XAUUSD (gold). Prices are BID candles; timestamps are UTC.

Files are fetched per day (BID_candles_min_1.bi5, LZMA-compressed, 24 bytes per bar:
int32 seconds-from-midnight, int32 open, int32 close, int32 low, int32 high, float32 volume; prices
are integers scaled by 10^decimals — 3 for indices, 5 for most FX pairs, 3 for XAUUSD).
Already-downloaded days are cached in data/raw/dukascopy_cache/<instrument>/ so re-runs are incremental.
Run this from a normal terminal (needs internet). Requires: pandas, pyarrow, requests.
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
DECIMALS = {"USATECHIDXUSD": 3, "USA500IDXUSD": 3, "XAUUSD": 3, "EURUSD": 5, "GBPUSD": 5, "USDJPY": 3}


#this fucntion get the data from dukascopy broker(as bytes), stores it into cache folder, and decompresses it into a pandas dataframe

def fetch_day(inst: str, day: date, cache: Path, session: requests.Session, retries: int = 3) -> pd.DataFrame | None:
    f = cache / f"{day.isoformat()}.bi5"
    if not f.exists():
        url = BASE.format(inst=inst, y=day.year, m=day.month - 1, d=day.day)  # Dukascopy months are 0-based
        for k in range(retries):
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 404 or len(r.content) == 0:
                    f.write_bytes(b"")  # remember empty days (weekends/holidays)
                    break
                r.raise_for_status()
                f.write_bytes(r.content)
                break
            except Exception as e:  # noqa: BLE001
                if k == retries - 1:
                    print(f"  ! {day}: {e}", file=sys.stderr)
                    return None
                time.sleep(1.5 * (k + 1))
    raw = f.read_bytes()
    if not raw:
        return None

    #decompress the raw bytes payload from cache folder and transform it into a dataframe
    data = lzma.decompress(raw)
    n = len(data) // 24
    rows = struct.iter_unpack(">5if", data[: n * 24])
    scale = 10 ** DECIMALS.get(inst, 5)
    base = pd.Timestamp(day, tz="UTC")
    rec = [(base + pd.Timedelta(seconds=t), o / scale, h / scale, l / scale, c / scale, v) for t, o, c, l, h, v in rows]
    return pd.DataFrame(rec, columns=["ts", "open", "high", "low", "close", "volume"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=None, help="default: data/raw/<instrument>_1m.parquet")
    args = ap.parse_args()

    inst = args.instrument.upper()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    cache = PROJECT_ROOT / "data" / "raw" / "dukascopy_cache" / inst
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / "raw" / f"{inst}_1m.parquet"

    frames, day, n_days = [], start, (end - start).days + 1
    with requests.Session() as s:
        s.headers["User-Agent"] = "Mozilla/5.0"
        for i in range(n_days):
            d = start + timedelta(days=i)
            if d.weekday() < 6 or True:  # Sunday evening has bars for indices; just try every day
                df = fetch_day(inst, d, cache, s)
                if df is not None and len(df):
                    frames.append(df)
            if i % 50 == 0:
                print(f"{d}  days={i}/{n_days}  bars={sum(len(x) for x in frames):,}")
    if not frames:
        sys.exit("no data downloaded")
    all_ = pd.concat(frames).drop_duplicates("ts").sort_values("ts").set_index("ts")
    all_.to_parquet(out)
    print(f"saved {len(all_):,} bars {all_.index.min()} -> {all_.index.max()} to {out}")
    print(all_.head(3))
    print("sanity: first close should look like a real index level (e.g. 6,000-30,000 for Nasdaq); if it is 1000x off, fix DECIMALS")


if __name__ == "__main__":
    main()
