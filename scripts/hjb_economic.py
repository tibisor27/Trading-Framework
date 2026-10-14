"""HJB cu obiectiv ECONOMIC: nu P(finantat), ci valoarea neta.

    W_p(e,t) = max_sigma E[ ... ;  la spargere -> W_1(0,t-1) - c ]
    la finantare -> VAL (valoarea prezenta a contului finantat, ca % din cont)

c = taxa ca fractie din cont. Cu c = 0 obiectivul degenereaza in "cumpara bilete
de loterie": maximizezi P fara sa platesti nimic pentru incercari.
"""
from __future__ import annotations
import numpy as np, pandas as pd, sys

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R.std(ddof=1); T=252
DAILY,MAXL,T1,T2=0.05,0.10,0.10,0.05
E=np.linspace(-MAXL,T1,81); i0=int(np.searchsorted(E,0.0))
VOLS=np.array([0.3,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
VAL=0.137*0.9      # venit net asteptat pe 24 luni pe cont finantat la 0.40%/zi, split 90%

def run(SH,c):
    BASE=(R-R.mean())+SH*SIG/np.sqrt(252)
    W1=np.zeros((T+1,len(E))); W2=np.zeros((T+1,len(E)))
    P1=np.zeros((T+1,len(E))); P2=np.zeros((T+1,len(E)))   # probabilitatea, sub aceeasi politica
    F1=np.zeros((T+1,len(E))); F2=np.zeros((T+1,len(E)))   # taxele
    A1=np.zeros((T+1,len(E)))
    for t in range(1,T+1):
        rW=W1[t-1][i0]-c; rP=P1[t-1][i0]; rF=F1[t-1][i0]+1.0
        for ph in (2,1):
            tgt=T1 if ph==1 else T2
            best=np.full(len(E),-1e9); arg=np.zeros(len(E))
            bp=np.zeros(len(E)); bf=np.zeros(len(E))
            for v in VOLS:
                k=(v/100.0)/SIG; rk=BASE*k; lok=(BASE+L)*k
                brk=(lok<=-DAILY)[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
                en=E[:,None]+rk[None,:]; fl=en.ravel()
                if ph==1:
                    nw=np.interp(fl,E,W1[t-1]).reshape(en.shape)
                    npb=np.interp(fl,E,P1[t-1]).reshape(en.shape)
                    nf=np.interp(fl,E,F1[t-1]).reshape(en.shape)
                    w=en>=tgt
                    nw=np.where(w,W2[t-1][i0],nw); npb=np.where(w,P2[t-1][i0],npb)
                    nf=np.where(w,F2[t-1][i0],nf)
                else:
                    nw=np.interp(fl,E,W2[t-1]).reshape(en.shape)
                    npb=np.interp(fl,E,P2[t-1]).reshape(en.shape)
                    nf=np.interp(fl,E,F2[t-1]).reshape(en.shape)
                    w=en>=tgt
                    nw=np.where(w,VAL,nw); npb=np.where(w,1.0,npb); nf=np.where(w,0.0,nf)
                val=np.where(brk,rW,nw).mean(axis=1)
                m=val>best
                best[m]=val[m]; arg[m]=v
                bp[m]=np.where(brk,rP,npb).mean(axis=1)[m]
                bf[m]=np.where(brk,rF,nf).mean(axis=1)[m]
            if ph==1: W1[t],A1[t],P1[t],F1[t]=best,arg,bp,bf
            else:     W2[t],P2[t],F2[t]=best,bp,bf
    return W1,P1,F1,A1

print(f"valoarea unui cont finantat = {VAL:.1%} din cont (24 luni la 0.40%/zi, split 90%)\n")
print("=== TESTUL DECISIV: aceeasi politica, cu si fara edge (taxa = 0, doar probabilitate) ===")
print(f"{'orizont':>10} {'Sharpe 0.95':>13} {'Sharpe 0.00':>13} {'cat aduce edge-ul':>19}")
a=run(0.95,0.0); b=run(0.00,0.0)
for h,lab in ((63,"3 luni"),(126,"6 luni"),(252,"12 luni")):
    print(f"{lab:>10} {a[1][h][i0]:13.3f} {b[1][h][i0]:13.3f} {a[1][h][i0]-b[1][h][i0]:+19.3f}")

print("\n=== OBIECTIV ECONOMIC: cum se schimba totul cand taxa costa ===")
print(f"{'taxa (% din cont)':>18} {'sigma optim':>12} {'P(3 luni)':>11} {'P(12 luni)':>12} "
      f"{'E[taxe]':>9} {'valoare neta':>13}")
for c in (0.000,0.002,0.006,0.012,0.025):
    W,P,F,A=run(0.95,c)
    print(f"{c:18.1%} {A[252][i0]:11.2f}% {P[63][i0]:11.3f} {P[252][i0]:12.3f} "
          f"{F[252][i0]:9.2f} {W[252][i0]:13.2%}")
