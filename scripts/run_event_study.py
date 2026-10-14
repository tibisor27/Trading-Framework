"""
Run an event study for one generator on one intraday file.

    python scripts/run_event_study.py --data index_NAS100_USD_1m.parquet --events noise_area
    python scripts/run_event_study.py --data data/raw/USATECHIDXUSD_1m.parquet --events macro_sweep_ifvg --cost-bps 0.65

Writes to results/event_study/<events>_<data-stem>/ : summary.csv, events.csv, returns.csv,
report.md, curve.png and manifest.json (config + data hash + row counts) so every result is traceable.
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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.intraday import load_intraday  # noqa: E402
from src.events.registry import build_events  # noqa: E402
from src.event_study import plot_curve, report_markdown, run_event_study  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_event_study")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet/csv with 1-minute OHLCV")
    ap.add_argument("--events", required=True, help="name of configs/events/<name>.yaml")
    ap.add_argument("--study", default="event_study", help="name of configs/<name>.yaml")
    ap.add_argument("--cost-bps", type=float, default=None, help="override round-trip cost in bps")
    ap.add_argument("--set", nargs="*", default=[], help="override generator params, e.g. vm=1.5 lookback=20")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ev_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "events" / f"{args.events}.yaml"))
    st_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / f"{args.study}.yaml"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        ev_cfg[k] = yaml.safe_load(v)
    if args.cost_bps is not None:
        st_cfg["cost_bps"] = args.cost_bps

    data_path = (PROJECT_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / "event_study" / f"{args.events}_{data_path.stem}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_intraday(data_path)
    events, eligible = build_events(df, ev_cfg)
    log.info("%d raw events, %d eligible times", len(events), len(eligible))
    res = run_event_study(df, events, eligible, **st_cfg)

    title = f"{args.events} · {data_path.stem}"
    res.summary.to_csv(out / "summary.csv")
    res.events.to_csv(out / "events.csv", index=False)
    res.returns.to_csv(out / "returns.csv", index=False)
    (out / "report.md").write_text(report_markdown(res, title), encoding="utf-8")
    plot_curve(res, title, str(out / "curve.png"))
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "data": str(data_path), "data_sha256": sha256(data_path), "bars": int(len(df)),
        "period": res.meta["period"], "event_config": ev_cfg, "study_config": st_cfg, "meta": res.meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(report_markdown(res, title))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
