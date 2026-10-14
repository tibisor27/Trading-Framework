"""
Does the strategy survive a change of data feed?

You discovered the edge on Dukascopy. You would execute on FTMO. Those are different quote streams
for the same underlying, and the question that matters is not "are the prices close" - they always
are - but "does the STRATEGY make the same decisions". A band is a threshold, and a threshold turns
a tiny price difference into a completely different trade whenever the close sits near it.

    python scripts/compare_feeds.py --a data/raw/USATECHIDXUSD_1m.parquet \\
                                    --b data/raw/FTMO_US100_1m.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import backtest                                 # noqa: E402
from src.data.intraday import load_intraday                              # noqa: E402
from src.events.noise_area import noise_area_bands, noise_area_signal     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="feed-ul de referinta (pe care ai facut cercetarea)")
    ap.add_argument("--b", required=True, help="feed-ul locului real de executie")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--vm", type=float, default=1.5)
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--exit-rule", default="opposite_band")
    ap.add_argument("--cost-bps", type=float, default=1.5)
    args = ap.parse_args()

    A = load_intraday(args.a)
    B = load_intraday(args.b)
    la = args.label_a or Path(args.a).stem
    lb = args.label_b or Path(args.b).stem

    lo = max(A.index[0], B.index[0])
    hi = min(A.index[-1], B.index[-1])
    if lo >= hi:
        raise SystemExit("cele doua fisiere nu se suprapun in timp")
    A, B = A.loc[lo:hi], B.loc[lo:hi]
    print(f"suprapunere: {lo.date()} -> {hi.date()}")
    print(f"  {la}: {len(A):,} bare | {lb}: {len(B):,} bare")

    j = A[["close"]].join(B[["close"]], how="inner", rsuffix="_b").dropna()
    print(f"  minute comune: {len(j):,} "
          f"({len(j)/max(1,len(A))*100:.1f}% din {la}, {len(j)/max(1,len(B))*100:.1f}% din {lb})")
    if j.empty:
        raise SystemExit("niciun minut comun - verifica fusul orar la import")

    diff_bps = (j["close_b"] / j["close"] - 1.0) * 1e4
    print("\n=== NIVELURI ===")
    print(f"  diferenta mediana : {diff_bps.median():+.2f} bps")
    print(f"  abatere absoluta  : {diff_bps.abs().median():.2f} bps (mediana), "
          f"{diff_bps.abs().quantile(.99):.2f} bps (p99)")
    ra = np.log(j["close"]).diff()
    rb = np.log(j["close_b"]).diff()
    m = ra.notna() & rb.notna()
    print(f"  corelatia randamentelor de 1 minut: {float(np.corrcoef(ra[m], rb[m])[0,1]):.5f}")

    print("\n=== STRATEGIA, PE FIECARE FEED ===")
    res = {}
    for label, frame in ((la, A), (lb, B)):
        s = noise_area_signal(frame, lookback=args.lookback, vm=args.vm)
        bd = noise_area_bands(frame, lookback=args.lookback, vm=args.vm)
        r = backtest(frame, s, bd, exit_rule=args.exit_rule, cost_bps=args.cost_bps)
        res[label] = r
        st = r.stats
        print(f"  {label:28s} Sharpe {st['sharpe']:5.2f} | CAGR {st['cagr_pct']:6.2f}% | "
              f"trades {st['trades']:4d} | MDD {st['max_dd_pct']:6.1f}%")

    ta = res[la].trades.copy()
    tb = res[lb].trades.copy()
    for t in (ta, tb):
        t["t_in"] = pd.to_datetime(t["t_in"], utc=True)
    ka = set(zip(ta["t_in"], ta["side"]))
    kb = set(zip(tb["t_in"], tb["side"]))
    both, only_a, only_b = ka & kb, ka - kb, kb - ka
    denom = max(1, len(ka | kb))
    print("\n=== ACORD LA NIVEL DE TRANZACTIE ===")
    print(f"  identice (acelasi minut, aceeasi directie): {len(both)}")
    print(f"  doar in {la}: {len(only_a)}")
    print(f"  doar in {lb}: {len(only_b)}")
    print(f"  acord: {len(both)/denom*100:.1f}%")

    if both:
        pa = ta.set_index(["t_in", "side"])["ret_net"]
        pb = tb.set_index(["t_in", "side"])["ret_net"]
        common = sorted(both)
        d = (pb.loc[common] - pa.loc[common]) * 1e4
        print(f"  pe tranzactiile comune, diferenta de randament: "
              f"mediana {d.median():+.2f} bps, medie {d.mean():+.2f} bps")

    print("\n=== CUM CITESTI ===")
    print("  acord peste ~90% = strategia se transfera; diferentele sunt zgomot de cotatie.")
    print("  acord sub ~75%  = pragul cade prea des exact pe diferenta dintre feed-uri.")
    print("                    Nu inseamna ca edge-ul e fals, ci ca REZULTATELE de pe un feed")
    print("                    nu prezic rezultatele de pe celalalt. Reia cercetarea pe feed-ul real.")


if __name__ == "__main__":
    main()
