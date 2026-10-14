"""Frontiera VITEZA vs TAXE: cat de repede ajungi finantat daca taxele nu constrang.

Obiectiv: NU P(trecere), ci E[timp pana la cont finantat] cu incercari repetate.
Esecurile se rezolva rapid, deci sigma mare poate fi mai RAPIDA chiar daca trece mai rar.

Testam si un intrerupator zilnic (flatten daca ziua e sub -D%): taie coada stanga
a excursiei intraday -> scade kurtoza -> muta plafonul impus de regula zilnica de 5%.
Tinta se evalueaza pe INCHIDERE (conservator), barierele pe minimul INTRADAY.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(".").resolve()
d = pd.read_csv("results/backtest/noise_area_USATECHIDXUSD_1m/excursion_opposite_band.csv")
R0 = d["ret"].to_numpy(float); DLO = (d["min_intra"] - d["ret"]).to_numpy(float)
SIG, N = R0.std(ddof=1), len(R0)
SHARPE = 0.95
BASE = (R0 - R0.mean()) + SHARPE * SIG / np.sqrt(252.0)   # serie cu sharpe 0.95, aceeasi vol

VOLS = [0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00, 3.00]
BREAKS = [None, 0.04, 0.03, 0.02]
DAILY, MAXL, BLOCK, NSIM, CHUNK = 0.05, 0.10, 20, 6000, 1000
rng = np.random.default_rng(101)


def series(v, D):
    k = (v / 100.0) / SIG
    r, lo = BASE * k, (BASE + DLO) * k
    if D is not None:
        hit = lo <= -D
        r = np.where(hit, -D, r)
        lo = np.where(hit, -D, lo)
        return r, lo, float(hit.mean())
    return r, lo, 0.0


def run(r, lo, target, md):
    won = 0; tw_all = []; tf_all = []
    done = 0
    while done < NSIM:
        m = min(CHUNK, NSIM - done)
        nb = int(np.ceil(md / BLOCK))
        st = rng.integers(0, max(1, N - BLOCK + 1), size=(m, nb))
        ix = np.clip((st[:, :, None] + np.arange(BLOCK)[None, None, :]
                      ).reshape(m, -1)[:, :md], 0, N - 1)
        rr, ll = r[ix], lo[ix]
        eq = np.cumsum(rr, axis=1)
        fail = ((eq - rr + ll) <= -MAXL) | (ll <= -DAILY)
        win = eq >= target
        big = md + 1
        tw = np.where(win.any(1), win.argmax(1), big)
        tf = np.where(fail.any(1), fail.argmax(1), big)
        w = tw < tf
        won += int(w.sum()); tw_all.append(tw[w]); tf_all.append(tf[~w])
        done += m
    tw_all = np.concatenate(tw_all); tf_all = np.concatenate(tf_all)
    return (won / NSIM,
            float(tw_all.mean()) if len(tw_all) else np.nan,
            float(np.clip(tf_all, 0, md).mean()) if len(tf_all) else np.nan)


for D in BREAKS:
    lab = "fara intrerupator" if D is None else f"intrerupator zilnic la -{D:.0%}"
    print(f"\n===== {lab} =====")
    print(f"{'vol/zi':>7} {'sharpe':>7} {'zile op':>8} | {'P(amb)':>7} | "
          f"{'E[zile->finantat]':>18} {'ani':>5} | {'taxe risipite':>14}")
    for v in VOLS:
        r, lo, frq = series(v, D)
        sh = r.mean() / r.std(ddof=1) * np.sqrt(252.0)
        md = int(np.clip(600.0 / v, 300, 1500))
        p1, w1, f1 = run(r, lo, 0.10, md)
        p2, w2, f2 = run(r, lo, 0.05, md)
        p = p1 * p2
        # timp pana la primul succes: esecuri repetate + o reusita, pe ambele faze
        t1 = w1 + f1 * (1 - p1) / max(p1, 1e-9)
        t2 = w2 + f2 * (1 - p2) / max(p2, 1e-9)
        tot = t1 + (t2 + t1 * (1 - p2) / max(p2, 1e-9)) if False else t1 / max(p1, 1e-9) * p1 + t1 * (1 - p1) / max(p1, 1e-9) * 0
        # explicit: E[timp] = (1/p)*E[durata unei incercari complete]
        dur = p1 * (w1 + p2 * w2 + (1 - p2) * f2) + (1 - p1) * f1
        etime = dur / max(p, 1e-9)
        print(f"{v:6.2f}% {sh:7.2f} {frq:7.1%} | {p:7.3f} | {etime:18.0f} "
              f"{etime/252:5.2f} | {1/max(p,1e-9)-1:13.2f}")
