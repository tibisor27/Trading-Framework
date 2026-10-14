"""
Parameter surface for an event generator: runs the event study over a small grid and prints one row
per setting (drift, t, N at a chosen horizon). Use it to see whether the result is a plateau or a peak.

    python scripts/sweep_event_study.py --data data/raw/USATECHIDXUSD_1m.parquet --events noise_area \
        --grid vm=0.7,1.0,1.3,1.5,2.0 lookback=10,14,20,30,60,90 --horizon close --cost-bps 0.65

    python scripts/sweep_event_study.py --data data/raw/USATECHIDXUSD_1m.parquet --events macro_sweep \
        --grid swing_n=5,15,30 sweep_min=2,5,10 --horizon 120 --cost-bps 0.65

Grid keys are generator params (see configs/events/<name>.yaml); the full cartesian product is run.
Writes results/event_study/sweep_<events>_<data-stem>.csv. Fewer settings = fewer chances to fool
yourself: decide the grid before looking, and judge the plateau, not the best cell.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.intraday import load_intraday  # noqa: E402
from src.events.registry import build_events  # noqa: E402
from src.event_study import run_event_study  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--grid", nargs="+", required=True, help="param=v1,v2,... (cartesian product)")
    ap.add_argument("--horizon", default="close")
    ap.add_argument("--cost-bps", type=float, default=0.65)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    ev_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "events" / f"{args.events}.yaml"))
    st_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "event_study.yaml"))
    st_cfg["cost_bps"] = args.cost_bps
    st_cfg["n_perm"] = args.n_perm

    grid = {}
    for kv in args.grid:
        k, vs = kv.split("=", 1)
        grid[k] = [yaml.safe_load(v) for v in vs.split(",")]
    data_path = (PROJECT_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    df = load_intraday(data_path)

    rows = []
    for combo in itertools.product(*grid.values()):
        cfg = dict(ev_cfg, **dict(zip(grid.keys(), combo)))
        try:
            events, eligible = build_events(df, cfg)
            res = run_event_study(df, events, eligible, **st_cfg)
            s = res.summary.loc[str(args.horizon)]
            row = dict(zip(grid.keys(), combo))
            row.update({"N": int(s["N"]), "events_per_day": round(res.meta["events_per_day"], 2),
                        "drift_bps": round(s["mean_bps"], 2), "t": round(s["t"], 2), "hit": round(s["hit"], 3),
                        "IC": round(s["IC"], 3), "excess_bps": round(s["excess_bps"], 2), "p_perm": round(s["p_perm"], 3),
                        "edge_over_cost": round(s["edge_over_cost"], 1)})
        except Exception as e:  # noqa: BLE001
            row = dict(zip(grid.keys(), combo)); row["error"] = str(e)[:80]
        rows.append(row)
        print(row, flush=True)

    out = PROJECT_ROOT / "results" / "event_study" / f"sweep_{args.events}_{data_path.stem}_{args.horizon}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten to {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
