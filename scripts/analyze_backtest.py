"""
Deep-dive on a backtest folder: bootstrap intervals, paired comparisons, year table, drawdowns,
volatility targeting on the strategy's own volatility, and prop-firm pass probabilities.

    python scripts/analyze_backtest.py --dir results/backtest/noise_area_USATECHIDXUSD_1m
    python scripts/analyze_backtest.py --dir results/backtest/noise_area_USATECHIDXUSD_1m \
        --baseline close --prop-vols 0.004,0.005,0.0075,0.01

Reads every daily_*.csv in the folder and writes analysis.md + rolling_sharpe.png next to them.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.analysis import (  # noqa: E402
    bootstrap_sharpe, compare_variants, drawdown_table, prop_firm_pass,
    rolling_sharpe, vol_target_strategy, yearly_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analyze")


def load_daily(folder: Path) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(folder.glob("daily_*.csv")):
        d = pd.read_csv(f)
        date_col = d.columns[0]
        d[date_col] = pd.to_datetime(d[date_col])
        out[f.stem.replace("daily_", "")] = d.set_index(date_col)["ret"].astype(float)
    if not out:
        raise SystemExit(f"No daily_*.csv found in {folder}")
    return out


def plot_rolling(series: dict[str, pd.Series], title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#111827", "#1baf7a", "#2a78d6", "#eda100", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
    ax.axhline(0, color="#b7bfc9", lw=1)
    ax.axhline(1, color="#d03b3b", lw=1, ls="--", label="Sharpe = 1")
    for (name, r), col in zip(series.items(), colors):
        rs = rolling_sharpe(r, 252)
        ax.plot(pd.to_datetime(rs.index), rs.to_numpy(), lw=1.5, color=col, label=name)
    ax.set_ylabel("Sharpe pe 12 luni")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default=None, help="variant tag to compare the others against")
    ap.add_argument("--prop-vols", default="0.004,0.005,0.0075,0.01",
                    help="daily volatilities to size to for the prop-firm simulation")
    ap.add_argument("--prop-target", type=float, default=0.06)
    ap.add_argument("--prop-dd", type=float, default=0.04)
    ap.add_argument("--prop-daily-loss", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    folder = (PROJECT_ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    daily = load_daily(folder)
    log.info("variants: %s", list(daily))
    base = args.baseline if args.baseline in daily else list(daily)[0]

    L = [f"# Analiză backtest · {folder.name}", ""]

    # 1. bootstrap Sharpe
    L += ["## 1. Sharpe cu interval de încredere (block bootstrap, blocuri de 20 de zile)", "",
          "| variantă | Sharpe | 95% jos | 95% sus | P(Sharpe ≤ 0) | zile |", "|---|---|---|---|---|---|"]
    boots = {}
    for name, r in daily.items():
        b = bootstrap_sharpe(r, n_boot=args.n_boot)
        boots[name] = b
        L.append(f"| {name} | {b['sharpe']} | {b['lo95']} | {b['hi95']} | {b['p_sharpe_le_0']} | {len(r)} |")

    # 2. paired comparison
    L += ["", f"## 2. Diferență de Sharpe față de `{base}` (bootstrap pereche, aceleași zile)", "",
          "| variantă | ΔSharpe | 95% jos | 95% sus | P(Δ ≤ 0) |", "|---|---|---|---|---|"]
    for name, r in daily.items():
        if name == base:
            continue
        c = compare_variants(r, daily[base], n_boot=args.n_boot)
        L.append(f"| {name} − {base} | {c['diff']} | {c['lo95']} | {c['hi95']} | {c['p_diff_le_0']} |")

    # 3. year by year
    L += ["", "## 3. An cu an", ""]
    for name, r in daily.items():
        L += [f"### {name}", "", yearly_table(r).to_markdown(), ""]

    # 4. drawdowns
    L += ["## 4. Cele mai mari 5 scăderi", ""]
    for name, r in daily.items():
        t = drawdown_table(r, 5)
        L += [f"### {name}", "", t.to_markdown(index=False) if not t.empty else "_fără scăderi_", ""]

    # 5. vol targeting on the strategy's own volatility
    L += ["## 5. Vol targeting pe volatilitatea PROPRIE a strategiei (nu a indicelui)", "",
          "| variantă | țintă zilnică | Sharpe 1x | Sharpe scalat | vol anual % | MDD % |", "|---|---|---|---|---|---|"]
    for name, r in daily.items():
        for tgt in (0.005, 0.01):
            s = vol_target_strategy(r, target_daily=tgt, lookback=20, max_leverage=4.0)
            s = s[s != 0]
            eq = (1 + s).cumprod()
            mdd = (eq / eq.cummax() - 1).min() * 100
            L.append(f"| {name} | {tgt*100:.1f}% | {bootstrap_sharpe(r, n_boot=500)['sharpe']} | "
                     f"{round(float(s.mean()/s.std(ddof=1)*np.sqrt(252)), 2)} | "
                     f"{round(float(s.std(ddof=1)*np.sqrt(252)*100), 1)} | {round(float(mdd), 1)} |")

    # 6. prop firm
    vols = [float(v) for v in args.prop_vols.split(",")]
    L += ["", f"## 6. Evaluare prop firm (țintă +{args.prop_target*100:.0f}%, trailing DD "
          f"{args.prop_dd*100:.0f}%, limită zilnică {args.prop_daily_loss*100:.0f}%)", "",
          "Randamentele proprii ale strategiei, reeșantionate în blocuri, scalate la volatilitatea zilnică din coloana 2.", "",
          "| variantă | vol zilnică | P(trece) | zile mediane | P(pică pe DD) | P(expiră în 250 zile) |", "|---|---|---|---|---|---|"]
    for name, r in daily.items():
        for v in vols:
            p = prop_firm_pass(r, target=args.prop_target, trailing_dd=args.prop_dd,
                               daily_loss=args.prop_daily_loss, scale_to_daily_vol=v, n_sims=20000)
            L.append(f"| {name} | {v*100:.2f}% | **{p['p_pass']}** | {p['median_days']} | {p['p_fail_dd']} | {p['p_timeout']} |")

    plot_rolling(daily, f"Sharpe pe 12 luni · {folder.name}", str(folder / "rolling_sharpe.png"))
    md = "\n".join(L)
    (folder / "analysis.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten to {folder / 'analysis.md'} and rolling_sharpe.png")


if __name__ == "__main__":
    main()
