"""
Measure the REAL historical spread instead of assuming a cost.

Dukascopy serves BID and ASK candles as separate files. You already have BID. This downloads the
matching ASK days and reports spread = (ask_close - bid_close) / mid, in basis points, broken down
by the only thing that matters: the minutes at which this strategy actually trades.

    python scripts/measure_spread.py --instrument USATECHIDXUSD --start 2022-01-01

Needs internet, so run it from your own terminal. Reuses the BID cache already on disk; ASK days are
cached separately, so re-runs are incremental.

Why this matters: the backtest fills every entry and exit at the BID. A long entry really pays the
ASK, and a short exit really pays the ASK, so each round trip is short by exactly ONE full spread -
whatever the direction. `cost_bps` therefore has to cover the FULL spread plus commission, not half.
The whole cost ladder in the results dossier hinges on this number being right.
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DECIMALS = {"USATECHIDXUSD": 3, "USA500IDXUSD": 3, "XAUUSD": 3, "EURUSD": 5, "GBPUSD": 5, "USDJPY": 3}


def fetch_day(inst: str, day: date, side: str, cache: Path, ses: requests.Session) -> pd.DataFrame | None:
    f = cache / f"{day.isoformat()}.bi5"
    if not f.exists():
        url = BASE.format(inst=inst, y=day.year, m=day.month - 1, d=day.day, side=side)
        for _ in range(3):
            try:
                r = ses.get(url, timeout=30)
                if r.status_code == 404 or len(r.content) == 0:
                    f.write_bytes(b"")
                    break
                r.raise_for_status()
                f.write_bytes(r.content)
                break
            except Exception:
                time.sleep(1.5)
        else:
            return None
    raw = f.read_bytes()
    if not raw:
        return None
    try:
        buf = lzma.decompress(raw)
    except lzma.LZMAError:
        return None
    n = len(buf) // 24
    rows = struct.unpack(">" + "5if" * n, buf[: n * 24])
    a = np.array(rows, dtype=float).reshape(n, 6)
    scale = 10 ** DECIMALS.get(inst, 5)
    ts = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(a[:, 0], unit="s")
    return pd.DataFrame({"close": a[:, 2] / scale}, index=ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="USATECHIDXUSD")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--tz", default="America/New_York")
    args = ap.parse_args()

    inst = args.instrument.upper()
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today() - timedelta(days=1)

    caches = {}
    for side in ("BID", "ASK"):
        c = ROOT / "data" / "raw" / ("dukascopy_cache" if side == "BID" else "dukascopy_cache_ask") / inst
        c.mkdir(parents=True, exist_ok=True)
        caches[side] = c

    ses = requests.Session()
    ses.headers.update({"User-Agent": "Mozilla/5.0"})
    bid, ask, day, k = [], [], start, 0
    while day <= end:
        if day.weekday() < 5:
            b = fetch_day(inst, day, "BID", caches["BID"], ses)
            a = fetch_day(inst, day, "ASK", caches["ASK"], ses)
            if b is not None and a is not None:
                bid.append(b)
                ask.append(a)
            k += 1
            if k % 100 == 0:
                print(f"  {day} ... {k} zile", flush=True)
        day += timedelta(days=1)

    if not bid:
        raise SystemExit("nimic descarcat")
    B = pd.concat(bid).rename(columns={"close": "bid"})
    A = pd.concat(ask).rename(columns={"close": "ask"})
    df = B.join(A, how="inner")
    df = df[(df["ask"] > 0) & (df["bid"] > 0)]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 1e4
    df["spread_pts"] = df["ask"] - df["bid"]

    loc = df.index.tz_convert(args.tz)
    df["hm"] = loc.strftime("%H:%M")
    df["year"] = loc.year
    rth = (loc.hour * 60 + loc.minute >= 570) & (loc.hour * 60 + loc.minute < 960)

    print(f"\n{inst}: {len(df):,} minute comune BID/ASK, {df.index[0].date()} -> {df.index[-1].date()}")
    print("\n=== SPREAD in sesiune (09:30-16:00 New York) ===")
    r = df[rth]
    print(r["spread_bps"].describe(percentiles=[.25, .5, .75, .9, .99]).round(3).to_string())
    print(f"\nmediana in puncte de index: {r['spread_pts'].median():.2f}")

    dec = r[r["hm"].isin([f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)])]
    print("\n=== SPREAD la minutele de DECIZIE (astea conteaza) ===")
    print(dec.groupby("hm")["spread_bps"].agg(n="size", median="median", p90=lambda s: s.quantile(.9)).round(3).to_string())

    print("\n=== SPREAD an cu an, in sesiune ===")
    print(r.groupby("year")["spread_bps"].agg(median="median", p90=lambda s: s.quantile(.9)).round(3).to_string())

    med = float(dec["spread_bps"].median())
    print("\n=== CE INSEAMNA PENTRU BACKTEST ===")
    print(f"  spread median la intrare/iesire : {med:.2f} bps")
    print(f"  cost dus-intors = UN spread plin: {med:.2f} bps")
    print(f"  + comision (0 pe indici la prop firms; ~0.5 bps pe MNQ)")
    print(f"  => cost_bps de folosit: {med:.2f} pana la {med + 0.5:.2f}")
    out = ROOT / "results" / "spread_measured.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    r[["bid", "ask", "spread_bps", "spread_pts", "hm", "year"]].to_csv(out)
    print(f"\nscris in {out}")


if __name__ == "__main__":
    main()
