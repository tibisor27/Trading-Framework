"""FTMO 2-Step simulat pe ECHITATE INTRADAY, nu pe inchiderea zilnica.

Diferenta fata de sim_ftmo.py: barierele (max loss statica 10%, pierdere zilnica 5%)
se verifica pe cel mai prost moment al zilei, nu pe randamentul de la inchidere.
Contul real verifica echitatea continuu, inclusiv P&L flotant.

Ruleaza ambele moduri pe ACELEASI trasee (indici partajati) -> comparatie pereche.
Fara plafon artificial de timp: FTMO 2-Step nu are limita de timp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/backtest/noise_area_USATECHIDXUSD_1m/excursion_opposite_band.csv"

SHARPE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
VOLS = [0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]
N_SIMS, MAX_DAYS, BLOCK, CHUNK = 10_000, 1500, 20, 1000
TARGET1, TARGET2, DAILY_LOSS, MAX_LOSS = 0.10, 0.05, 0.05, 0.10

d = pd.read_csv(SRC)
R = d["ret"].to_numpy(float)
DLO = (d["min_intra"] - d["ret"]).to_numpy(float)   # <= 0, excursie sub inchidere
DHI = (d["max_intra"] - d["ret"]).to_numpy(float)   # >= 0
SIG, MU = R.std(ddof=1), R.mean()
N = len(R)
print(f"serie: {N} zile   sigma {SIG:.4%}/zi   sharpe brut "
      f"{MU/SIG*np.sqrt(252):.2f}   -> scalat la sharpe {SHARPE:.2f}")
print(f"{N_SIMS:,} trasee x {MAX_DAYS} zile, bloc {BLOCK}, fara plafon de timp\n")


def paths(k, mu_t, rng, m):
    nb = int(np.ceil(MAX_DAYS / BLOCK))
    st = rng.integers(0, max(1, N - BLOCK + 1), size=(m, nb))
    ix = (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(m, -1)[:, :MAX_DAYS]
    ix = np.clip(ix, 0, N - 1)
    r = (R[ix] - MU) * k + mu_t
    return r, r + DLO[ix] * k, r + DHI[ix] * k


def stage(r, lo, target):
    """-> (a trecut, ziua rezolvarii). Bariere pe lo (intraday) sau pe r (close)."""
    eq = np.cumsum(r, axis=1)
    eq_lo = eq - r + lo                       # echitate la cel mai prost moment al zilei
    fail = (eq_lo <= -MAX_LOSS) | (lo <= -DAILY_LOSS)
    win = eq >= target
    big = MAX_DAYS + 1
    tw = np.where(win.any(1), win.argmax(1), big)
    tf = np.where(fail.any(1), fail.argmax(1), big)
    return tw < tf, np.minimum(tw, tf)


rng = np.random.default_rng(7)
print(f"{'vol/zi':>7} | {'--- pe INCHIDERE (vechi) ---':^30} | {'--- pe INTRADAY (corect) ---':^30} |")
print(f"{'':>7} | {'P(f1)':>6} {'P(f2)':>6} {'P(amb)':>7} {'zile':>7} | "
      f"{'P(f1)':>6} {'P(f2)':>6} {'P(amb)':>7} {'zile':>7} | {'%/an':>6}")
print("-" * 92)

tabel = []
for v in VOLS:
    k = (v / 100.0) / SIG
    mu_t = SHARPE * (v / 100.0) / np.sqrt(252.0)
    acc = {m: {1: [0, []], 2: [0, []]} for m in ("close", "intra")}
    done = 0
    while done < N_SIMS:
        m = min(CHUNK, N_SIMS - done)
        r, lo, hi = paths(k, mu_t, rng, m)
        for mod, L in (("close", r), ("intra", lo)):
            for s, tg in ((1, TARGET1), (2, TARGET2)):
                w, t = stage(r, L, tg)
                acc[mod][s][0] += int(w.sum())
                acc[mod][s][1].append(t[w])
        done += m
    row = [v]
    for mod in ("close", "intra"):
        p1 = acc[mod][1][0] / N_SIMS
        p2 = acc[mod][2][0] / N_SIMS
        z1 = np.concatenate(acc[mod][1][1]); z2 = np.concatenate(acc[mod][2][1])
        zile = (np.median(z1) + np.median(z2)) if len(z1) and len(z2) else np.nan
        row += [p1, p2, p1 * p2, zile]
    row.append(SHARPE * v / 100.0 * np.sqrt(252.0) * 100.0)
    tabel.append(row)
    print(f"{v:6.2f}% | {row[1]:6.3f} {row[2]:6.3f} {row[3]:7.3f} {row[4]:7.0f} | "
          f"{row[5]:6.3f} {row[6]:6.3f} {row[7]:7.3f} {row[8]:7.0f} | {row[9]:5.1f}%")

pd.DataFrame(tabel, columns=["vol", "c_p1", "c_p2", "c_both", "c_days",
                             "i_p1", "i_p2", "i_both", "i_days", "pct_an"]
             ).to_csv(ROOT / "results/backtest/ftmo_intraday_sweep.csv", index=False)

# ---- unde incepe sa muste regula zilnica de 5% --------------------------
print("\n--- regula zilnica de 5%: cate zile din istoric o incalca ---")
print(f"{'vol/zi':>7} | {'pe inchidere':>13} | {'pe intraday':>12} | "
      f"{'cea mai rea zi (close / intra)':>32}")
for v in VOLS:
    k = (v / 100.0) / SIG
    rc = (R - MU) * k + SHARPE * (v / 100.0) / np.sqrt(252.0)
    ri = rc + DLO * k
    print(f"{v:6.2f}% | {int((rc <= -DAILY_LOSS).sum()):13d} | "
          f"{int((ri <= -DAILY_LOSS).sum()):12d} | {rc.min():14.2%} / {ri.min():13.2%}")
