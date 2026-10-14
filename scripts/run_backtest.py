"""
Backtest the ladder of exit/sizing variants for a strategy, on one intraday file.

    python scripts/run_backtest.py --data data/raw/USATECHIDXUSD_1m.parquet --cost-bps 0.65
    python scripts/run_backtest.py --data data/raw/USA500IDXUSD_1m.parquet  --cost-bps 0.2 \
        --set vm=1.3 lookback=14

Writes results/backtest/<strategy>_<data-stem>/ : table.md (Table-3 style), stats.csv, equity.png,
trades_<variant>.csv, daily_<variant>.csv and manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import alpha_beta, backtest, buy_hold  # noqa: E402
from src.data.intraday import load_intraday  # noqa: E402
from src.events.noise_area import noise_area_bands, noise_area_signal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_backtest")

SIGNALS = {"noise_area": (noise_area_signal, noise_area_bands)}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def plot_equity(curves: dict, bh: pd.Series | None, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#111827", "#1baf7a", "#2a78d6", "#eda100", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    for (name, eq), col in zip(curves.items(), colors):
        ax.plot(pd.to_datetime(eq.index), eq.to_numpy() * 100, lw=1.6, color=col, label=name)
    if bh is not None:
        ax.plot(pd.to_datetime(bh.index), bh.to_numpy() * 100, lw=1.4, color="#d03b3b", alpha=0.9, label="buy & hold")
    ax.set_yscale("log")
    ax.set_ylabel("capital (start = 100, scară log)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="noise_area", help="name of configs/backtest/<name>.yaml")
    ap.add_argument("--events", default=None, help="events config for the signal params (default: same name)")
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override signal params, e.g. vm=1.3 lookback=20")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bt_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "backtest" / f"{args.config}.yaml"))
    strategy = bt_cfg["strategy"]
    ev_name = args.events or strategy
    sig_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "events" / f"{ev_name}.yaml"))
    sig_cfg.pop("name", None)
    for kv in args.set:
        k, v = kv.split("=", 1)
        sig_cfg[k] = yaml.safe_load(v)
    if args.cost_bps is not None:
        bt_cfg["cost_bps"] = args.cost_bps

    data_path = (PROJECT_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "backtest" / f"{strategy}_{data_path.stem}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_intraday(data_path)
    make_signal, make_bands = SIGNALS[strategy]
    signal = make_signal(df, **sig_cfg)
    bands = make_bands(df, **sig_cfg)
    log.info("signal: %d decision bars, %d non-zero", int(np.isfinite(signal).sum()), int((signal.abs() > 0).sum()))

    common = {k: bt_cfg[k] for k in ("cost_bps", "stop_every_bar", "vol_lookback", "max_leverage", "tz", "rth_start", "rth_end")}
    rows, curves = [], {}
    for v in bt_cfg["variants"]:
        name = v["name"]
        res = backtest(df, signal, bands, exit_rule=v["exit_rule"], vol_target=v.get("vol_target"), **common)
        ab = alpha_beta(res.daily["ret"], df, common["tz"], common["rth_start"], common["rth_end"])
        rows.append({"variant": name, "exit": v["exit_rule"], "vol_target": v.get("vol_target"), **res.stats, **ab})
        curves[name] = res.equity
        tag = v["exit_rule"] + ("_vt" if v.get("vol_target") else "")
        res.trades.to_csv(out / f"trades_{tag}.csv", index=False)
        res.daily.to_csv(out / f"daily_{tag}.csv")
        log.info("%-32s Sharpe %.2f  CAGR %.1f%%  MDD %.1f%%  trades %d", name, res.stats["sharpe"],
                 res.stats["cagr_pct"], res.stats["max_dd_pct"], res.stats["trades"])

    bh = buy_hold(df, common["tz"], common["rth_start"], common["rth_end"])
    rows.append({"variant": "buy & hold", "exit": "-", "vol_target": None, **bh})
    table = pd.DataFrame(rows)
    table.to_csv(out / "stats.csv", index=False)

    show = ["variant", "trades", "trades_per_day", "total_return_pct", "cagr_pct", "vol_pct", "sharpe",
            "t_stat", "max_dd_pct", "hit_days_pct", "skew_daily", "worst_day_pct", "alpha_ann_pct", "beta"]
    show = [c for c in show if c in table.columns]
    title = f"{strategy} · {data_path.stem} · cost {bt_cfg['cost_bps']} bps · params {sig_cfg}"
    md = f"# Backtest {strategy} · {data_path.stem}\n\n{title}\n\n" + table[show].to_markdown(index=False)
    (out / "table.md").write_text(md, encoding="utf-8")

    mkt = None
    try:
        mask_close = df["close"]
        dates = np.asarray(df.index.tz_convert(common["tz"]).date)
        dc = pd.Series(mask_close.to_numpy(), index=dates).groupby(level=0).last()
        first = min(c.index.min() for c in curves.values())
        mkt = (1 + dc.pct_change().fillna(0)).cumprod()
        mkt = mkt[mkt.index >= first]
        mkt = mkt / mkt.iloc[0]
    except Exception as e:  # noqa: BLE001
        log.warning("buy&hold curve skipped: %s", e)
    plot_equity(curves, mkt, title, str(out / "equity.png"))

    manifest = {"run_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
                "data": str(data_path), "data_sha256": sha256(data_path), "bars": int(len(df)),
                "signal_config": sig_cfg, "backtest_config": bt_cfg}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
