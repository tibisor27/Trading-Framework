"""HJB cu RELUARE: la esec deschizi cont nou a doua zi, ceasul anului merge inainte.

  W_p(e,t) = max_sigma E[ ... , iar la spargere -> W_1(0, t-1) ]

Urmarim si F = E[taxe] sub politica optima (evaluare, nu optimizare).
"""
from __future__ import annotations
import numpy as np, pandas as pd, sys

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R.std(ddof=1); SH=float(sys.argv[1]) if len(sys.argv)>1 else 0.95
T=252; BASE=(R-R.mean())+SH*SIG/np.sqrt(252)
DAILY,MAXL,T1,T2=0.05,0.10,0.10,0.05
E=np.linspace(-MAXL,T1,141); i0=int(np.searchsorted(E,0.0))
VOLS=np.array([0.3,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])

W1=np.zeros((T+1,len(E))); W2=np.zeros((T+1,len(E)))
F1=np.zeros((T+1,len(E))); F2=np.zeros((T+1,len(E)))
A1=np.zeros((T+1,len(E))); A2=np.zeros((T+1,len(E)))
for t in range(1,T+1):
    rst=W1[t-1][i0]; rstF=F1[t-1][i0]+1.0           # valoarea unui cont nou + o taxa
    for ph in (2,1):
        tgt=T1 if ph==1 else T2
        best=np.full(len(E),-1.0); arg=np.zeros(len(E)); fee=np.zeros(len(E))
        for v in VOLS:
            k=(v/100.0)/SIG; rk=BASE*k; lok=(BASE+L)*k
            brk=(lok<=-DAILY)[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
            en=E[:,None]+rk[None,:]
            if ph==1:
                nx=np.interp(en.ravel(),E,W1[t-1]).reshape(en.shape)
                fx=np.interp(en.ravel(),E,F1[t-1]).reshape(en.shape)
                nx=np.where(en>=tgt,W2[t-1][i0],nx); fx=np.where(en>=tgt,F2[t-1][i0],fx)
            else:
                nx=np.interp(en.ravel(),E,W2[t-1]).reshape(en.shape)
                fx=np.interp(en.ravel(),E,F2[t-1]).reshape(en.shape)
                nx=np.where(en>=tgt,1.0,nx); fx=np.where(en>=tgt,0.0,fx)
            val=np.where(brk,rst,nx).mean(axis=1)
            fv =np.where(brk,rstF,fx).mean(axis=1)
            m=val>best; best[m]=val[m]; arg[m]=v; fee[m]=fv[m]
        if ph==1: W1[t],A1[t],F1[t]=best,arg,fee
        else:     W2[t],A2[t],F2[t]=best,arg,fee

print(f"Sharpe asumat {SH:.2f} | HJB cu reluare | {len(VOLS)} marimi | grila {len(E)}\n")
print("=== P(finantat) MAXIMA sub orice politica, cu reluari ===")
print(f"{'orizont':>18} {'P(finantat)':>12} {'E[taxe]':>9}")
for h,lab in ((42,"2 luni"),(63,"3 luni"),(84,"4 luni"),(126,"6 luni"),
              (189,"9 luni"),(252,"12 luni")):
    print(f"{lab:>18} {W1[h][i0]:12.3f} {F1[h][i0]:9.2f}")
print("\n=== marimea optima la start de cont, dupa cate zile mai ai ===")
for h in (21,42,63,84,126,189,252):
    print(f"  {h:3d} zile ramase: {A1[h][i0]:5.2f}%/zi")
print(f"\n=== politica optima completa, faza 1 ===")
print(f"{'zile ram.':>10} " + "".join(f"{f'eq {e:+.0%}':>9}" for e in (-0.06,-0.03,0.0,0.03,0.06,0.09)))
for h in (252,189,126,63,42,21):
    row=f"{h:10d} "
    for e in (-0.06,-0.03,0.0,0.03,0.06,0.09):
        row+=f"{A1[h][int(np.searchsorted(E,e))]:8.2f}%"
    print(row)
print(f"\n=== politica optima, faza 2 ===")
print(f"{'zile ram.':>10} " + "".join(f"{f'eq {e:+.0%}':>9}" for e in (-0.05,0.0,0.02,0.04)))
for h in (252,126,63,42,21):
    row=f"{h:10d} "
    for e in (-0.05,0.0,0.02,0.04):
        row+=f"{A2[h][int(np.searchsorted(E,e))]:8.2f}%"
    print(row)
np.save("/tmp/A1r.npy",A1); np.save("/tmp/A2r.npy",A2); np.save("/tmp/Er.npy",E)
