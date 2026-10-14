"""FTMO 1-Step vs 2-Step: comparatie completa sub REGULILE REALE (ftmo.com, 25 sep 2026).

2-Step: tinta 10% -> 5% | zi 5% din capital initial | max loss 10% STATIC
        min 4 zile tranzactionate/faza | timp nelimitat | split 80%
        taxa 540 EUR RAMBURSATA la prima retragere | fara Best Day Rule
1-Step: tinta 10% | zi 3% din capital initial | max loss 10% TRAILING EOD pe BALANTA
        (creste, nu scade; fara lock; reset complet doar la retragere)
        fara min zile | timp nelimitat | split 90%
        taxa 499 EUR NERAMBURSATA | Best Day Rule 50% la challenge SI la finantat

Bariera de zi si de max loss se verifica pe EQUITY intraday (deci pe min_intra).
Trailing-ul se actualizeaza pe BALANTA de la 00:00 (deci pe inchiderea zilei).
"""
from __future__ import annotations
import numpy as np, pandas as pd, sys, time

D = "results/backtest/noise_area_USATECHIDXUSD_1m"
d = pd.read_csv(f"{D}/excursion_opposite_band.csv")
R0 = d["ret"].to_numpy(float)
LOD = (d["min_intra"] - d["ret"]).to_numpy(float)     # <=0, adancimea sub inchidere
SIG = R0.std(ddof=1); N = len(R0)
SH = 0.95                                              # Sharpe folosit peste tot in sesiune
BASE = (R0 - R0.mean()) + SH*SIG/np.sqrt(252)          # kernel empiric, drift setat la SH
TRADED = (R0 != 0.0)                                   # zi cu cel putin o pozitie

EURUSD = 1.08
FEE1 = 499*EURUSD/100000.0      # 1-Step, nerambursabila
FEE2 = 540*EURUSD/100000.0      # 2-Step, rambursabila la prima retragere
T1, T2 = 0.10, 0.05
MAXL = 0.10
DLY1, DLY2 = 0.03, 0.05
SPLIT1, SPLIT2 = 0.90, 0.80
MINDAYS2 = 4

def hdr(s): print("\n" + "="*78 + f"\n{s}\n" + "="*78)

# ============================ 0. DIAGNOSTIC GEOMETRIC =======================
hdr("0. CE INSEAMNA FIECARE BARIERA IN MARIMEA POZITIEI")
mi = d["min_intra"].to_numpy(float)
print(f"sigma zilnica la levier 1 : {SIG*100:.4f}%/zi     ({N} zile reale)")
print(f"cea mai proasta excursie  : {mi.min()*100:.3f}% intraday (2020-03-12 inchide +1.80%)")
print(f"\n{'regula de zi':>16} {'levier max':>11} {'vol/zi':>9}   interpretare")
for dl, lab in ((DLY2, "5% (2-Step)"), (DLY1, "3% (1-Step)")):
    k = dl/abs(mi.min())
    print(f"{lab:>16} {k:10.3f}x {k*SIG*100:8.3f}%   zero spargeri pe tot istoricul")
print(f"\n  -> regula de 3% taie plafonul 'niciodata nu sparg' cu {(1-DLY1/DLY2)*100:.0f}%: "
      f"{DLY1/abs(mi.min())*SIG*100:.3f}% vs {DLY2/abs(mi.min())*SIG*100:.3f}%/zi")

print("\n--- max loss: 10% STATIC vs 10% TRAILING EOD, pe istoricul real ---")
print(f"{'vol/zi':>8} {'levier':>7} {'DD max (static)':>17} {'DD max pe fereastra':>21} {'moare trailing?':>17}")
for v in (0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50):
    k = (v/100.0)/SIG; x = BASE*k
    eq = np.cumsum(x); dd = eq - np.maximum.accumulate(eq)
    w = 10
    roll = np.array([ (np.cumsum(x[i:i+w]) - np.maximum.accumulate(np.r_[0.0, np.cumsum(x[i:i+w])])[1:]).min()
                      for i in range(0, N-w) ])
    print(f"{v:7.2f}% {k:6.2f}x {dd.min():16.2%} {roll.min():20.2%} "
          f"{'DA' if dd.min()<=-MAXL else 'nu':>17}")
print("\n  'DD max pe fereastra' = cea mai adanca cadere varf-la-vale in 10 zile de tranzactionare")
print("  (= ciclul dintre doua retrageri, care RESETEAZA trailing-ul)")

# ============================ 1. BEST DAY RULE ==============================
hdr("1. BEST DAY RULE 50% - cat de mult doare skew-ul de 2.21")
def bestday_stats(vol, target, ns=6000, horizon=400, seed=7):
    """cate zile in plus ceri ca sa diluezi ziua cea mai buna sub 50%"""
    rng = np.random.default_rng(seed); k = (vol/100.0)/SIG
    blk = 20; nb = int(np.ceil(horizon/blk))
    st = rng.integers(0, max(1, N-blk+1), size=(ns, nb))
    IX = np.clip((st[:,:,None]+np.arange(blk)[None,None,:]).reshape(ns,-1)[:,:horizon], 0, N-1)
    r = BASE[IX]*k
    eq = np.zeros(ns); pos = np.zeros(ns); best = np.zeros(ns)
    d_hit = np.full(ns, -1); d_ok = np.full(ns, -1)
    for t in range(horizon):
        eq += r[:,t]
        p = np.maximum(r[:,t], 0.0); pos += p; best = np.maximum(best, p)
        hit = (d_hit<0) & (eq>=target); d_hit[hit] = t
        ok = (d_hit>=0) & (d_ok<0) & (eq>=target) & (best <= 0.5*pos+1e-12); d_ok[ok] = t
    m = (d_hit>=0) & (d_ok>=0)
    return d_hit, d_ok, m
print(f"{'vol/zi':>8} {'ating 10%':>11} {'si Best Day ok':>16} {'zile pana la 10%':>18} "
      f"{'zile pana la VALID':>20} {'intarziere':>12}")
for v in (0.30, 0.60, 1.00, 1.60, 2.50):
    dh, do, m = bestday_stats(v, T1)
    a = dh>=0
    print(f"{v:7.2f}% {a.mean():10.1%} {m.mean():15.1%} "
          f"{np.median(dh[a]) if a.any() else np.nan:17.0f} "
          f"{np.median(do[m]) if m.any() else np.nan:19.0f} "
          f"{(np.median(do[m])-np.median(dh[m])) if m.any() else np.nan:11.0f} zile")
print("\n  (fara bariere - doar cat dureaza sa faci profitul ACCEPTABIL, nu doar sa-l faci)")

# ============================ 2. CONTUL FINANTAT ============================
hdr("2. CONTUL FINANTAT: 10% static (2-Step) vs 10% trailing EOD (1-Step)")
H_FUND = 504                       # 2 ani de zile de tranzactionare
NSF, BLK = 20000, 20
rngF = np.random.default_rng(2026)
nbF = int(np.ceil(H_FUND/BLK))
stF = rngF.integers(0, max(1, N-BLK+1), size=(NSF, nbF))
IXF = np.clip((stF[:,:,None]+np.arange(BLK)[None,None,:]).reshape(NSF,-1)[:,:H_FUND], 0, N-1)

def funded(vol, trailing, daily, split, buffer_, cycle, bestday, horizon=H_FUND):
    """intoarce (cash net catre el in unitati de capital initial, zile supravietuite)"""
    k = (vol/100.0)/SIG
    r = BASE[IXF]*k; lo = (BASE+LOD)[IXF]*k
    eq = np.full(NSF, buffer_); hwm = np.zeros(NSF)
    alive = np.ones(NSF, bool); cash = np.zeros(NSF); life = np.full(NSF, horizon)
    pos = np.zeros(NSF); best = np.zeros(NSF); npay = np.zeros(NSF)
    for t in range(horizon):
        if not alive.any(): break
        floor_ = (hwm - MAXL) if trailing else np.full(NSF, -MAXL)
        bad = alive & (((eq + lo[:,t]) <= floor_) | (lo[:,t] <= -daily))
        alive[bad] = False; life[bad] = t
        ad = alive.copy()
        eq[ad] += r[ad,t]
        p = np.maximum(r[:,t], 0.0)
        pos[ad] += p[ad]; best[ad] = np.maximum(best[ad], p[ad])
        if trailing: hwm[ad] = np.maximum(hwm[ad], eq[ad])
        if (t+1) % cycle == 0:
            can = ad & (eq > buffer_)
            if bestday: can &= (best <= 0.5*pos + 1e-12)
            w = np.where(can, eq - buffer_, 0.0)
            cash += split*w; eq -= w; npay += can
            if trailing: hwm[can] = 0.0            # reset la retragere
            pos[can] = 0.0; best[can] = 0.0
    return cash, life, npay

print(f"{'produs':>10} {'vol/zi':>7} {'buffer':>7} {'ciclu':>6} {'supravietuire 2 ani':>20} "
      f"{'zile mediane':>13} {'cash net 2 ani':>15} {'retrageri':>10}")
ROWS=[]
for lab, tr, dl, sp, bd in (("2-Step", False, DLY2, SPLIT2, False),
                            ("1-Step", True,  DLY1, SPLIT1, True)):
    for v in (0.25, 0.40, 0.60, 0.90, 1.30):
        for buf in (0.00, 0.02, 0.04):
            for cyc in (10, 21):
                c, l, np_ = funded(v, tr, dl, sp, buf, cyc, bd)
                surv = (l >= H_FUND).mean()
                ROWS.append((lab, v, buf, cyc, surv, np.median(l), c.mean(), np_.mean()))
best_by = {}
for lab, v, buf, cyc, surv, med, cm, npm in ROWS:
    if lab not in best_by or cm > best_by[lab][6]: best_by[lab] = (lab, v, buf, cyc, surv, med, cm, npm)
for lab, v, buf, cyc, surv, med, cm, npm in sorted(ROWS, key=lambda z:(z[0], -z[6]))[:0]: pass
for lab in ("2-Step", "1-Step"):
    sub = sorted([z for z in ROWS if z[0]==lab], key=lambda z:-z[6])[:6]
    print(f"--- {lab} ---")
    for lab_, v, buf, cyc, surv, med, cm, npm in sub:
        print(f"{'':>10} {v:6.2f}% {buf:6.1%} {cyc:5d}z {surv:19.1%} {med:13.0f} "
              f"{cm*100000:14,.0f}$ {npm:10.2f}")
b1 = best_by["1-Step"]; b2 = best_by["2-Step"]
RATE1 = b1[6]/2.0; RATE2 = b2[6]/2.0     # cash net pe AN, unitati de capital initial
print(f"\n  optim 2-Step : {b2[1]:.2f}%/zi, buffer {b2[2]:.0%}, ciclu {b2[3]}z "
      f"-> {b2[6]*100000:,.0f}$ / 2 ani  ({RATE2:.2%}/an)")
print(f"  optim 1-Step : {b1[1]:.2f}%/zi, buffer {b1[2]:.0%}, ciclu {b1[3]}z "
      f"-> {b1[6]*100000:,.0f}$ / 2 ani  ({RATE1:.2%}/an)")
print(f"  raport       : 1-Step aduce {b1[6]/b2[6]:.2f}x cat 2-Step pe contul finantat")
np.save("results/backtest/_rates.npy", np.array([RATE1, RATE2]))
