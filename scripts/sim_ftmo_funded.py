"""Contul FINANTAT: supravietuire si venit, cu bariere verificate pe echitate intraday.

Reguli persistente dupa finantare: pierdere zilnica 5%, pierdere maxima 10% STATICA
de la balanta initiala. Fara tinta. Doua politici de retragere:
  A) nu retragi nimic  -> profitul construieste tampon, contul devine mai sigur
  B) retragi lunar tot ce e peste balanta initiala -> tamponul nu creste niciodata
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "results/backtest/noise_area_USATECHIDXUSD_1m/excursion_opposite_band.csv")
R = d["ret"].to_numpy(float)
DLO = (d["min_intra"] - d["ret"]).to_numpy(float)
SIG, MU, N = R.std(ddof=1), R.mean(), len(R)

VOLS = [0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50]
N_SIMS, DAYS, BLOCK, CHUNK = 10_000, 504, 20, 1000
DAILY_LOSS, MAX_LOSS, MONTH = 0.05, 0.10, 21
SHARPE = 0.95
rng = np.random.default_rng(11)

print(f"cont finantat, sharpe {SHARPE:.2f}, {N_SIMS:,} trasee x {DAYS} zile (24 luni)\n")
print(f"{'vol/zi':>7} | {'A: fara retrageri':^23} | {'B: retrageri lunare':^31} | {'%/an':>6}")
print(f"{'':>7} | {'viu 12l':>8} {'viu 24l':>8} | {'viu 12l':>8} {'viu 24l':>8} {'retras 12l':>11} | ")
print("-" * 84)

for v in VOLS:
    k = (v / 100.0) / SIG
    mu_t = SHARPE * (v / 100.0) / np.sqrt(252.0)
    surv = {"A": np.zeros(2), "B": np.zeros(2)}
    paid = 0.0
    done = 0
    while done < N_SIMS:
        m = min(CHUNK, N_SIMS - done)
        nb = int(np.ceil(DAYS / BLOCK))
        st = rng.integers(0, max(1, N - BLOCK + 1), size=(m, nb))
        ix = np.clip((st[:, :, None] + np.arange(BLOCK)[None, None, :]
                      ).reshape(m, -1)[:, :DAYS], 0, N - 1)
        r = (R[ix] - MU) * k + mu_t
        lo = r + DLO[ix] * k

        # A) fara retrageri
        eq = np.cumsum(r, axis=1)
        bad = ((eq - r + lo) <= -MAX_LOSS) | (lo <= -DAILY_LOSS)
        tA = np.where(bad.any(1), bad.argmax(1), DAYS + 1)
        surv["A"] += [(tA >= 252).sum(), (tA >= DAYS).sum()]

        # B) retrageri lunare: la fiecare 21 zile, tot ce e peste 0 pleaca
        eqB = np.zeros(m); alive = np.ones(m, bool)
        tB = np.full(m, DAYS + 1); tot = np.zeros(m)
        for t in range(DAYS):
            low = eqB + lo[:, t]
            hit = alive & ((low <= -MAX_LOSS) | (lo[:, t] <= -DAILY_LOSS))
            tB[hit] = t; alive &= ~hit
            eqB = np.where(alive, eqB + r[:, t], eqB)
            if (t + 1) % MONTH == 0:
                w = np.where(alive & (eqB > 0), eqB, 0.0)
                tot += w; eqB -= w
        surv["B"] += [(tB >= 252).sum(), (tB >= DAYS).sum()]
        paid += float(tot[tB >= 252].sum()) if (tB >= 252).any() else 0.0
        # venit contabilizat doar pe traseele vii la 12 luni
        done += m

    a, b = surv["A"] / N_SIMS, surv["B"] / N_SIMS
    inc = paid / max(1, int(b[0] * N_SIMS))
    print(f"{v:6.2f}% | {a[0]:8.3f} {a[1]:8.3f} | {b[0]:8.3f} {b[1]:8.3f} "
          f"{inc:10.2%} | {SHARPE*v/100*np.sqrt(252)*100:5.1f}%")

# ---- cat de fragil e planul daca edge-ul e mai mic decat crezi -----------
print("\n--- P(treci ambele faze) la vol 0.50%/zi, in functie de sharpe-ul REAL ---")
VOL, MAXD = 0.50, 1500
k = (VOL / 100.0) / SIG
for sh in [1.35, 1.19, 0.95, 0.70, 0.50, 0.30, 0.0]:
    mu_t = sh * (VOL / 100.0) / np.sqrt(252.0)
    ps = []
    for tg in (0.10, 0.05):
        w = 0; done = 0
        while done < 6000:
            m = min(1000, 6000 - done)
            nb = int(np.ceil(MAXD / BLOCK))
            st = rng.integers(0, max(1, N - BLOCK + 1), size=(m, nb))
            ix = np.clip((st[:, :, None] + np.arange(BLOCK)[None, None, :]
                          ).reshape(m, -1)[:, :MAXD], 0, N - 1)
            r = (R[ix] - MU) * k + mu_t
            lo = r + DLO[ix] * k
            eq = np.cumsum(r, axis=1)
            f = ((eq - r + lo) <= -MAX_LOSS) | (lo <= -DAILY_LOSS)
            g = eq >= tg
            big = MAXD + 1
            tw = np.where(g.any(1), g.argmax(1), big)
            tf = np.where(f.any(1), f.argmax(1), big)
            w += int((tw < tf).sum()); done += m
        ps.append(w / 6000)
    print(f"  sharpe {sh:4.2f}  ->  P(f1) {ps[0]:.3f}   P(f2) {ps[1]:.3f}   "
          f"P(ambele) {ps[0]*ps[1]:.3f}")
