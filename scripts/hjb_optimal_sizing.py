"""Politica de marime OPTIMA pentru un challenge cu termen fix.

Problema: optiune digitala cu bariera dubla si knock-out, in care VOLATILITATEA
E CONTROLUL. Se rezolva prin programare dinamica (HJB discretizat):

    V_p(e, t) = max_sigma  E_d[ V dupa o zi ]

Nucleul de tranzitie NU e gaussian: e distributia empirica a celor 2240 de zile
reale, scalate. Deci cozile, asimetria si excursia intraday intra exact asa cum
sunt masurate. Rezultatul e probabilitatea MAXIMA atinsa de orice politica.
"""
from __future__ import annotations
import numpy as np, pandas as pd, sys

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R.std(ddof=1)
SH=float(sys.argv[1]) if len(sys.argv)>1 else 0.95
T=int(sys.argv[2]) if len(sys.argv)>2 else 63
BASE=(R-R.mean())+SH*SIG/np.sqrt(252)
DAILY,MAXL,T1,T2=0.05,0.10,0.10,0.05
E=np.linspace(-MAXL,T1,141)                       # grila de echitate
VOLS=np.array([0.3,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0])

def solve(T):
    V1=np.zeros((T+1,len(E))); V2=np.zeros((T+1,len(E)))
    A1=np.zeros((T+1,len(E))); A2=np.zeros((T+1,len(E)))
    for t in range(1,T+1):
        for ph,(tgt,Vc,Vn,Ac) in ((1,(T1,V1,V2,A1)),(2,(T2,V2,None,A2))):
            best=np.full(len(E),-1.0); arg=np.zeros(len(E))
            for v in VOLS:
                k=(v/100.0)/SIG; rk=BASE*k; lok=(BASE+L)*k
                dead_day=lok<=-DAILY                                  # regula zilnica
                en=E[:,None]+rk[None,:]                               # (nE, nDays)
                brk=dead_day[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
                if ph==1:
                    nxt=np.interp(en.ravel(),E,V1[t-1]).reshape(en.shape)
                    win=en>=tgt
                    nxt=np.where(win,V2[t-1][np.searchsorted(E,0.0)],nxt)
                else:
                    nxt=np.interp(en.ravel(),E,V2[t-1]).reshape(en.shape)
                    nxt=np.where(en>=tgt,1.0,nxt)
                val=np.where(brk,0.0,nxt).mean(axis=1)
                m=val>best; best[m]=val[m]; arg[m]=v
            Vc[t]=best; Ac[t]=arg
    return V1,V2,A1,A2

V1,V2,A1,A2=solve(T)
i0=int(np.searchsorted(E,0.0))
print(f"Sharpe asumat {SH:.2f} | orizont {T} zile ({T/21:.0f} luni) | "
      f"{len(VOLS)} marimi candidate | grila {len(E)} puncte\n")
print("=== P(treci AMBELE faze) sub politica OPTIMA, o singura incercare ===")
for h in (21,42,63,84,126,189,252):
    if h>T: continue
    print(f"  {h:3d} zile ({h/21:.0f} luni): {V1[h][i0]:.3f}")
print(f"\n=== marimea optima la start (echitate 0, faza 1) ===")
for h in (21,42,63,126,252):
    if h>T: continue
    print(f"  cu {h:3d} zile ramase: {A1[h][i0]:.2f}%/zi")
print(f"\n=== politica optima, faza 1, {T} zile ramase ===")
print(f"{'echitate':>9} {'sigma optim':>12} {'P(finantat)':>12}")
for e in (-0.06,-0.03,0.0,0.03,0.06,0.09):
    j=int(np.searchsorted(E,e)); print(f"{e:+8.0%} {A1[T][j]:11.2f}% {V1[T][j]:12.3f}")
print(f"\n=== politica optima, faza 2, {T} zile ramase ===")
for e in (-0.05,0.0,0.02,0.04):
    j=int(np.searchsorted(E,e)); print(f"{e:+8.0%} {A2[T][j]:11.2f}% {V2[T][j]:12.3f}")
np.save("/tmp/hjb_A1.npy",A1); np.save("/tmp/hjb_A2.npy",A2); np.save("/tmp/hjb_E.npy",E)
