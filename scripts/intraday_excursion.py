"""Excursia intraday: cat de adanc coboara echitatea in timpul zilei fata de inchidere.

sim_ftmo.py verifica barierele FTMO pe randamentul ZILNIC. Contul real le verifica
pe echitate, continuu, inclusiv P&L flotant. Diferenta e exact seria de aici.

Iesire: date, ret, min_intra, max_intra, n_trades  -- gata de dat lui stage().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os
D = ROOT / os.environ.get("EXC_DIR", "results/backtest/noise_area_USATECHIDXUSD_1m")
BARS = ROOT / "data/raw/USATECHIDXUSD_1m.parquet"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "opposite_band"

# ---------------------------------------------------------------- incarca
bars = pd.read_parquet(BARS, columns=["high", "low"]).sort_index()
if bars.index.tz is None:
    bars.index = bars.index.tz_localize("UTC")
idx = bars.index
hi_a = bars["high"].to_numpy()
lo_a = bars["low"].to_numpy()

tr = pd.read_csv(D / f"trades_{VARIANT}.csv")
tr["t_in"] = pd.to_datetime(tr["t_in"], utc=True)
tr["t_out"] = pd.to_datetime(tr["t_out"], utc=True)

print(f"varianta        : {VARIANT}")
print(f"bare            : {len(bars):,}   {idx[0]} -> {idx[-1]}")
print(f"trade-uri       : {len(tr):,}   {tr['date'].min()} -> {tr['date'].max()}")
print("\nminute de intrare (top 6):")
print(tr["t_in"].dt.strftime("%H:%M").value_counts().head(6).to_string())

# ------------------------------------------------- MAE / MFE in unitati de pnl
a = idx.searchsorted(tr["t_in"].to_numpy(), side="left")
b = idx.searchsorted(tr["t_out"].to_numpy(), side="right")
side = tr["side"].to_numpy(float)
pin = tr["price_in"].to_numpy(float)
lev = tr["leverage"].to_numpy(float)
cost = (tr["ret_net"] - tr["ret_gross"]).to_numpy(float)   # negativ

mae = np.zeros(len(tr)); mfe = np.zeros(len(tr)); nbar = np.zeros(len(tr), int)
for i in range(len(tr)):
    s, e = a[i], b[i]
    nbar[i] = e - s
    if e <= s:
        continue
    lo, hi = lo_a[s:e].min(), hi_a[s:e].max()
    if side[i] > 0:
        g_lo, g_hi = lo / pin[i] - 1.0, hi / pin[i] - 1.0
    else:
        g_lo, g_hi = 1.0 - hi / pin[i], 1.0 - lo / pin[i]
    mae[i] = (g_lo + cost[i]) * lev[i]
    mfe[i] = (g_hi + cost[i]) * lev[i]

tr["mae"], tr["mfe"], tr["nbar"] = mae, mfe, nbar
print(f"\ntrade-uri fara bare acoperite: {(nbar == 0).sum()}")

bad = (tr["pnl"] < tr["mae"] - 1e-12) | (tr["pnl"] > tr["mfe"] + 1e-12)
if bad.any():
    print(tr.loc[bad, ["t_in", "t_out", "side", "price_in", "price_out",
                       "pnl", "mae", "mfe", "nbar"]].head(10).to_string())
    sys.exit(f"OPRESTE-TE: {bad.sum()} trade-uri cu pnl in afara [mae, mfe]")
print("test pnl in [mae, mfe] : OK")

# ------------------------------------------------------------- traseul zilei
rows = []
for day, g in tr.groupby("date", sort=True):
    realizat = lo = hi = 0.0
    for t in g.sort_values("t_in").itertuples():
        lo = min(lo, realizat + t.mae)
        hi = max(hi, realizat + t.mfe)
        realizat += t.pnl
    rows.append((day, realizat, lo, hi))
zi = pd.DataFrame(rows, columns=["date", "close", "min_intra", "max_intra"])
zi["date"] = pd.to_datetime(zi["date"]).dt.date

# -------------------------------------------------------------------- POARTA
d = pd.read_csv(D / f"daily_{VARIANT}.csv")
d["date"] = pd.to_datetime(d["date"]).dt.date
f = d.merge(zi, on="date", how="left", validate="1:1")
f[["min_intra", "max_intra", "close"]] = f[["min_intra", "max_intra", "close"]].fillna(0.0)

orfane = int((~zi["date"].isin(set(d["date"]))).sum())
dif = float((f["close"] - f["ret"]).abs().max())
print(f"reconciliere           : {len(f)} zile, orfane {orfane}, dif max {dif:.3e}")
if orfane or not (dif < 1e-10):
    sys.exit("OPRESTE-TE: traseele zilnice nu reproduc daily.csv")

# ---------------------------------------------------------------- rezultatul
f["gap"] = f["ret"] - f["min_intra"]
sig = float(f["ret"].std(ddof=1))
prez = 0.5826 * sig
cu = f[f["n_trades"] > 0]

print("\n--- gap = inchidere - minim intraday, in %, doar zile cu trade ---")
print((cu["gap"] * 100).describe(percentiles=[.5, .75, .9, .95, .99]).round(4).to_string())
print(f"\nsigma zilnica (toate zilele)   : {sig:.4%}")
print(f"sigma zilnica (zile cu trade)  : {cu['ret'].std(ddof=1):.4%}")
print(f"gap median  (zile cu trade)    : {cu['gap'].median():.4%}")
print(f"gap mediu   (zile cu trade)    : {cu['gap'].mean():.4%}")
print(f"gap p95     (zile cu trade)    : {cu['gap'].quantile(.95):.4%}")
print(f"prezis de 0.5826*sigma         : {prez:.4%}")
print(f"raport median / prezis         : {cu['gap'].median()/prez:.2f}x")
print(f"raport p95 / prezis            : {cu['gap'].quantile(.95)/prez:.2f}x")
print(f"\ncea mai proasta inchidere      : {f['ret'].min():.3%}")
print(f"cel mai adanc minim intraday   : {f['min_intra'].min():.3%}")
print(f"cel mai bun maxim intraday     : {f['max_intra'].max():.3%}")

out = D / f"excursion_{VARIANT}.csv"
f[["date", "ret", "min_intra", "max_intra", "n_trades"]].to_csv(out, index=False)
print(f"\nscris: {out.relative_to(ROOT)}")
