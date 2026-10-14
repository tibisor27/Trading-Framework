"""
Two-stage evaluation with a STATIC max loss (FTMO-style), simulated on the strategy's own returns.

Rules modelled per stage, counters reset between stages:
    pass   : cumulative P&L from the START of the stage >= target
    fail   : cumulative P&L from the start of the stage <= -max_loss   (STATIC, not trailing)
    fail   : any single day <= -daily_loss

Stage 1 target 10%, stage 2 target 5%. Draws are i.i.d. resamples, so P(both) = P(1) x P(2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import backtest                                # noqa: E402
from src.data.intraday import load_intraday                             # noqa: E402
from src.events.noise_area import noise_area_bands, noise_area_signal   # noqa: E402

CONFIGS = [("band_vwap", 1.0), ("opposite_band", 1.3), ("opposite_band", 1.5), ("band_vwap", 1.3)]
SIZES = [0.004, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]


def stage(x, target, daily_loss, max_loss, max_days, n_sims, block, rng):
    n = len(x)
    nb = int(np.ceil(max_days / block))
    st = rng.integers(0, max(1, n - block + 1), size=(n_sims, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :max_days]
    p = x[np.clip(idx, 0, n - 1)]
    eq = np.cumsum(p, axis=1)
    fail = (eq <= -max_loss) | (p <= -daily_loss)
    win = eq >= target
    big = max_days + 1
    ff = np.where(fail.any(1), fail.argmax(1), big)
    fw = np.where(win.any(1), win.argmax(1), big)
    ok = fw < ff
    dd = (ff < fw) & (ff < big)
    return (float(ok.mean()), float(np.median(fw[ok] + 1)) if ok.any() else np.nan,
            float(dd.mean()), float(((fw >= big) & (ff >= big)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cost-bps", type=float, default=0.65)
    ap.add_argument("--daily-loss", type=float, default=0.05)
    ap.add_argument("--max-loss", type=float, default=0.10)
    ap.add_argument("--t1", type=float, default=0.10)
    ap.add_argument("--t2", type=float, default=0.05)
    ap.add_argument("--max-days", type=int, default=250)
    ap.add_argument("--n-sims", type=int, default=20000)
    args = ap.parse_args()

    df = load_intraday(args.data)
    rows = []
    cache = {}
    for ex, vm in CONFIGS:
        if vm not in cache:
            cache[vm] = (noise_area_signal(df, lookback=14, vm=vm), noise_area_bands(df, lookback=14, vm=vm))
        sig, bnd = cache[vm]
        r = backtest(df, sig, bnd, exit_rule=ex, cost_bps=args.cost_bps)
        d = r.daily["ret"].to_numpy(dtype=float)
        sd0 = d.std(ddof=1)
        for s in SIZES:
            x = d * (s / sd0)
            rng = np.random.default_rng(0)
            p1, dd1, f1, to1 = stage(x, args.t1, args.daily_loss, args.max_loss, args.max_days, args.n_sims, 20, rng)
            p2, dd2, f2, to2 = stage(x, args.t2, args.daily_loss, args.max_loss, args.max_days, args.n_sims, 20, rng)
            rows.append({"exit": ex, "vm": vm, "vol_zi_%": round(s * 100, 2),
                         "vol_an_%": round(s * np.sqrt(252) * 100, 1),
                         "P_faza1": round(p1, 3), "zile_f1": None if np.isnan(dd1) else int(dd1),
                         "P_faza2": round(p2, 3), "zile_f2": None if np.isnan(dd2) else int(dd2),
                         "P_ambele": round(p1 * p2, 3),
                         "P_pica_DD_f1": round(f1, 3), "P_expira_f1": round(to1, 3)})
    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "backtest" / "sim_ftmo.csv"
    out.to_csv(dest, index=False)
    print(f"reguli: faza1 +{args.t1*100:.0f}%  faza2 +{args.t2*100:.0f}%  "
          f"zi -{args.daily_loss*100:.0f}%  max -{args.max_loss*100:.0f}% STATIC  "
          f"limita {args.max_days} zile\n")
    print(out.sort_values("P_ambele", ascending=False).head(14).to_markdown(index=False))
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
