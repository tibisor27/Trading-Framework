"""
Why a 14-day MEAN of |Close/Open-1|? Test the choice instead of arguing about it.

Two questions, deliberately kept apart:

  1. FORECAST  - which estimator predicts tomorrow's |move| best? (corr, MAE)
  2. TRADING   - which estimator produces the best strategy? (Sharpe, t, max DD)

They are not the same question, and the gap between them is the point.

Every candidate is rescaled so its AVERAGE level matches the paper's baseline. Otherwise a wider
estimator is just a smaller `vm`, and we would be re-running the vm sweep under a new name.
All candidates are evaluated on the SAME dates (after the longest warm-up), so the comparison is
paired, not confounded by sample period.

    python scripts/sweep_vol_estimator.py --data data/raw/USATECHIDXUSD_1m.parquet --cost-bps 0.65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import backtest            # noqa: E402
from src.data.intraday import load_intraday, rth_mask  # noqa: E402

CANDIDATES = [
    ("mean", 5,   "medie 5 zile"),
    ("mean", 10,  "medie 10 zile"),
    ("mean", 14,  "medie 14 zile  <- PAPER"),
    ("mean", 20,  "medie 20 zile"),
    ("mean", 30,  "medie 30 zile"),
    ("mean", 60,  "medie 60 zile"),
    ("mean", 120, "medie 120 zile"),
    ("mean", 250, "medie 250 zile"),
    ("median", 14, "mediana 14 zile"),
    ("median", 60, "mediana 60 zile"),
    ("std", 14,   "deviatie standard 14"),
    ("std", 60,   "deviatie standard 60"),
    ("ewma", 5,   "EWMA halflife 5"),
    ("ewma", 10,  "EWMA halflife 10"),
    ("ewma", 20,  "EWMA halflife 20"),
    ("ewma", 40,  "EWMA halflife 40"),
    ("expanding", 0, "medie de la inceput (expanding)"),
]


def build_pivot(df, tz, rth_start, rth_end):
    mask = rth_mask(df.index, tz, rth_start, rth_end)
    d = df.loc[mask.values, ["open", "close"]].copy()
    loc = d.index.tz_convert(tz)
    d["date"] = loc.date
    s0 = int(rth_start[:2]) * 60 + int(rth_start[3:])
    d["mod"] = (loc.hour * 60 + loc.minute) - s0
    day_open = d.groupby("date")["open"].first()
    day_close = d.groupby("date")["close"].last()
    d["move"] = (d["close"] / d["date"].map(day_open) - 1.0).abs()
    piv = d.pivot_table(index="date", columns="mod", values="move")
    return d, piv, day_open, day_close


def estimate(piv, kind, param):
    p = piv.shift(1)                       # only days strictly before -> no look-ahead
    if kind == "mean":
        return p.rolling(param, min_periods=param).mean()
    if kind == "median":
        return p.rolling(param, min_periods=param).median()
    if kind == "std":
        return p.rolling(param, min_periods=param).std(ddof=1)
    if kind == "ewma":
        return p.ewm(halflife=param, min_periods=3 * param).mean()
    if kind == "expanding":
        return p.expanding(min_periods=60).mean()
    raise ValueError(kind)


def bands_and_signal(df, sigma, d, day_open, day_close, vm, tz, decision=(0, 30), first="10:00"):
    ref_hi = np.maximum(day_open, day_close.shift(1))
    ref_lo = np.minimum(day_open, day_close.shift(1))
    ub = sigma.mul(vm).add(1.0).mul(ref_hi, axis=0)
    lb = (1.0 - sigma.mul(vm)).mul(ref_lo, axis=0)

    ri = ub.index.get_indexer(d["date"].to_numpy())
    ci = ub.columns.get_indexer(d["mod"].to_numpy())
    ok = (ri >= 0) & (ci >= 0)
    ris, cis = np.clip(ri, 0, None), np.clip(ci, 0, None)
    pick = lambda f: np.where(ok, f.to_numpy()[ris, cis], np.nan)  # noqa: E731

    bands = pd.DataFrame({"ub": pick(ub), "lb": pick(lb)}, index=d.index)
    loc = d.index.tz_convert(tz)
    fd = int(first[:2]) * 60 + int(first[3:])
    is_dec = loc.minute.isin(decision) & ((loc.hour * 60 + loc.minute) >= fd)
    cl = d["close"].to_numpy()
    v = np.where(cl > bands["ub"], 1.0, np.where(cl < bands["lb"], -1.0, 0.0))
    v = np.where(bands["ub"].isna().to_numpy(), np.nan, v)
    sig = pd.Series(np.nan, index=df.index, name="signal")
    sig.loc[d.index[is_dec]] = v[is_dec]
    return bands, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cost-bps", type=float, default=0.65)
    ap.add_argument("--vm", type=float, default=1.3)
    ap.add_argument("--exit-rule", default="opposite_band")
    ap.add_argument("--tz", default="America/New_York")
    args = ap.parse_args()

    df = load_intraday(args.data)
    d, piv, day_open, day_close = build_pivot(df, args.tz, "09:30", "16:00")

    sigmas = {}
    for kind, param, label in CANDIDATES:
        sigmas[label] = estimate(piv, kind, param)

    base = sigmas["medie 14 zile  <- PAPER"]
    # common evaluation window: every candidate must be warm
    first_ok = max(s.dropna(how="all").index[0] for s in sigmas.values())
    print(f"fereastra comuna de evaluare incepe la {first_ok} "
          f"({(piv.index >= first_ok).sum()} zile din {len(piv)})")

    rows = []
    for kind, param, label in CANDIDATES:
        s = sigmas[label]
        w = s.index >= first_ok
        # rescale so the average LEVEL matches the paper's, isolating the dynamics
        scale = float(np.nanmean(base[w].to_numpy())) / float(np.nanmean(s[w].to_numpy()))
        s = s * scale

        # 1. forecast quality: predicted sigma vs the move actually realised that day
        pred = s[w].to_numpy().ravel()
        act = piv[w].to_numpy().ravel()
        m = np.isfinite(pred) & np.isfinite(act)
        corr = float(np.corrcoef(pred[m], act[m])[0, 1])
        mae = float(np.mean(np.abs(pred[m] - act[m])) * 1e4)
        stab = float(np.nanmean(np.abs(np.diff(s[w].to_numpy(), axis=0))) /
                     np.nanmean(s[w].to_numpy()) * 100)  # % day-to-day churn

        # 2. trading quality
        bands, sig = bands_and_signal(df, s, d, day_open, day_close, args.vm, args.tz)
        start = pd.Timestamp(first_ok).tz_localize(args.tz).tz_convert("UTC")
        sl = df.index >= start
        r = backtest(df.loc[sl], sig.loc[sl], bands.reindex(df.index).loc[sl],
                     exit_rule=args.exit_rule, cost_bps=args.cost_bps)
        st = r.stats
        rows.append({
            "estimator": label, "scale": round(scale, 3),
            "corr": round(corr, 3), "MAE_bps": round(mae, 1), "churn_%": round(stab, 1),
            "trades": st["trades"], "sharpe": st["sharpe"], "t": st["t_stat"],
            "CAGR_%": st["cagr_pct"], "maxDD_%": st["max_dd_pct"],
            "worst_day_%": st["worst_day_pct"], "skew": st["skew_daily"],
        })
        print(f"  {label:34s} corr={corr:.3f}  Sharpe={st['sharpe']:.2f}  trades={st['trades']}")

    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "backtest" / "sweep_vol_estimator.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print()
    print(out.to_markdown(index=False))
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
