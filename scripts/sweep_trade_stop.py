"""Stop pe trade: taie excursia LA SURSA, nu la nivel de cont.

Pentru fiecare nivel S, un trade care atinge -S iese la -S. Recalculam totul:
momente, plafonul impus de regula zilnica de 5%, si frontiera viteza/taxe.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

D = "results/backtest/noise_area_USATECHIDXUSD_1m"
bars = pd.read_parquet("data/raw/USATECHIDXUSD_1m.parquet", columns=["high", "low"]).sort_index()
if bars.index.tz is None: bars.index = bars.index.tz_localize("UTC")
ix_, hi_a, lo_a = bars.index, bars["high"].to_numpy(), bars["low"].to_numpy()

tr = pd.read_csv(f"{D}/trades_opposite_band.csv")
tr["t_in"] = pd.to_datetime(tr["t_in"], utc=True); tr["t_out"] = pd.to_datetime(tr["t_out"], utc=True)
a = ix_.searchsorted(tr["t_in"].to_numpy(), "left"); b = ix_.searchsorted(tr["t_out"].to_numpy(), "right")
mae = np.zeros(len(tr))
for i in range(len(tr)):
    w_lo, w_hi = lo_a[a[i]:b[i]], hi_a[a[i]:b[i]]
    if len(w_lo) == 0: continue
    p, s = tr.price_in.iloc[i], tr.side.iloc[i]
    g = (w_lo.min()/p - 1.0) if s > 0 else (1.0 - w_hi.max()/p)
    mae[i] = g + (tr.ret_net.iloc[i] - tr.ret_gross.iloc[i])
tr["mae"] = mae
cal = pd.read_csv(f"{D}/daily_opposite_band.csv"); cal["date"] = pd.to_datetime(cal["date"]).dt.date

print(f"{'stop':>6} {'trade-uri':>10} {'CAGR':>7} {'vol':>6} {'sharpe':>7} {'skew':>6} {'kurt':>6} "
      f"{'worst intra':>12} {'plafon vol':>11} {'castig vs 0.40%':>16}")
for S in (None, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030):
    t = tr.copy()
    if S is not None:
        h = t["mae"] <= -S
        t.loc[h, "pnl"] = -S * t.loc[h, "leverage"]
        t["mae_eff"] = np.maximum(t["mae"], -S)
        n_hit = int(h.sum())
    else:
        t["mae_eff"] = t["mae"]; n_hit = 0
    rows = []
    for day, g in t.groupby("date", sort=True):
        rz = lo = 0.0
        for r in g.sort_values("t_in").itertuples():
            lo = min(lo, rz + r.mae_eff); rz += r.pnl
        rows.append((day, rz, lo))
    z = pd.DataFrame(rows, columns=["date", "ret2", "min2"])
    z["date"] = pd.to_datetime(z["date"]).dt.date
    f = cal.merge(z, on="date", how="left").fillna({"ret2": 0.0, "min2": 0.0})
    R, M = f["ret2"].to_numpy(), f["min2"].to_numpy()
    sig = R.std(ddof=1); sh = R.mean()/sig*np.sqrt(252); cagr = R.mean()*252
    ceil_ = 0.05*sig/abs(M.min())*100
    # castig in probabilitate la plafon vs planul de 0.40%
    lab = "fara" if S is None else f"{S:.2%}"
    print(f"{lab:>6} {n_hit:10d} {cagr:6.1%} {sig*np.sqrt(252):5.1%} {sh:7.2f} "
          f"{stats.skew(R):+6.2f} {stats.kurtosis(R):6.1f} {M.min():11.2%} "
          f"{ceil_:10.3f}% {ceil_/0.507:15.2f}x")
    if S is not None:
        z2 = f[["date", "ret2", "min2"]].rename(columns={"ret2": "ret", "min2": "min_intra"})
        z2["max_intra"] = 0.0; z2["n_trades"] = (f["ret2"] != 0).astype(int)
        z2.to_csv(f"{D}/excursion_stop{int(S*10000)}.csv", index=False)
