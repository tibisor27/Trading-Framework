"""
Optimise the entry threshold for an EVALUATION, not for Sharpe.

A prop-firm evaluation is a barrier-crossing problem: reach +target before you hit the trailing
drawdown or a daily loss limit. That objective rewards a completely different shape of return
distribution than Sharpe does - it cares about the worst day and the skew, and it is not invariant
to position size. So the search is over BOTH the entry threshold (vm) and the size.

For each (exit rule, vm) the strategy is backtested at 1x, then its own daily returns are resampled
in blocks through the evaluation rules at several sizings, and we keep the sizing that maximises the
pass probability. Two rule sets are simulated, because firm selection is itself a lever.

    python scripts/sweep_prop_firm.py --data data/raw/USATECHIDXUSD_1m.parquet --cost-bps 0.65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import backtest                  # noqa: E402
from src.data.intraday import load_intraday               # noqa: E402
from src.events.noise_area import noise_area_bands, noise_area_signal  # noqa: E402


def _fast_pass(r: np.ndarray, target: float, trailing_dd: float, daily_loss: float,
               scale_to_daily_vol: float, max_days: int = 250, n_sims: int = 20000,
               block: int = 20, seed: int = 0) -> dict:
    """Block-bootstrap the evaluation. Builds only `max_days` columns, unlike the generic helper."""
    sd = r.std(ddof=1)
    x = r * (scale_to_daily_vol / sd) if sd > 0 else r
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(max_days / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :max_days]
    paths = x[np.clip(idx, 0, n - 1)]
    eq = np.cumsum(paths, axis=1)
    hw = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
    fail = (hw - eq >= trailing_dd) | (paths <= -daily_loss)
    win = eq >= target
    big = max_days + 1
    ff = np.where(fail.any(1), fail.argmax(1), big)
    fw = np.where(win.any(1), win.argmax(1), big)
    ok = fw < ff
    return {"p_pass": round(float(ok.mean()), 3),
            "median_days": int(np.median(fw[ok] + 1)) if ok.any() else None,
            "p_fail_dd": round(float(((ff < fw) & (ff < big)).mean()), 3)}


VMS = [0.8, 1.0, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
EXITS = ["opposite_band", "band_vwap"]
SIZES = [0.0025, 0.004, 0.005, 0.0075, 0.01, 0.015]       # daily volatility to size to
FIRMS = {
    "strict (+6% / -4% trailing / -2% zi)":  dict(target=0.06, trailing_dd=0.04, daily_loss=0.02),
    "relaxat (+8% / -10% / -5% zi)":         dict(target=0.08, trailing_dd=0.10, daily_loss=0.05),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cost-bps", type=float, default=0.65)
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--n-sims", type=int, default=20000)
    args = ap.parse_args()

    df = load_intraday(args.data)
    rows = []
    for vm in VMS:
        sig = noise_area_signal(df, lookback=args.lookback, vm=vm)
        bnd = noise_area_bands(df, lookback=args.lookback, vm=vm)
        for ex in EXITS:
            r = backtest(df, sig, bnd, exit_rule=ex, cost_bps=args.cost_bps)
            st, daily = r.stats, r.daily["ret"]
            row = {
                "exit": ex, "vm": vm, "trades": st["trades"],
                "trades_zi": st["trades_per_day"], "sharpe": st["sharpe"],
                "worst_day_%": st["worst_day_pct"], "skew": st["skew_daily"],
                "maxDD_%": st["max_dd_pct"], "CAGR_%": st["cagr_pct"],
            }
            for fname, rules in FIRMS.items():
                best = max(
                    (_fast_pass(daily.to_numpy(dtype=float), **rules, scale_to_daily_vol=s,
                               n_sims=args.n_sims) | {"size": s} for s in SIZES),
                    key=lambda p: p["p_pass"],
                )
                tag = "strict" if fname.startswith("strict") else "relax"
                row[f"p_pass_{tag}"] = best["p_pass"]
                row[f"size_{tag}_%"] = round(best["size"] * 100, 2)
                row[f"zile_{tag}"] = best["median_days"]
                row[f"p_DD_{tag}"] = best["p_fail_dd"]
            rows.append(row)
            print(f"  {ex:14s} vm={vm:<5} Sharpe={st['sharpe']:.2f} "
                  f"worst={st['worst_day_pct']:.2f}% skew={st['skew_daily']:.2f} "
                  f"-> p_pass strict={row['p_pass_strict']:.3f} relax={row['p_pass_relax']:.3f}",
                  flush=True)

    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "backtest" / "sweep_prop_firm.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print("\n=== ordonat dupa probabilitatea de a trece evaluarea stricta ===")
    print(out.sort_values("p_pass_strict", ascending=False).to_markdown(index=False))
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
