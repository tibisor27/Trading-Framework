"""1 faza, tinta 10%, trailing DD 3% -- comparat cap la cap cu FTMO 2-Step.

Motor verificat (acelasi care a reprodus ruina jucatorului). Reluare imediata.
"""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); LO=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG,N=R.std(ddof=1),len(R); SH=0.95
BASE=(R-R.mean())+SH*SIG/np.sqrt(252)
BLOCK,T,NS=20,252,12000
rng=np.random.default_rng(5150)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,N-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
B0,L0=BASE[IX],(BASE+LO)[IX]

def sim(vol,stages,trail=None,lock=None,static=None,daily=None):
    k=(vol/100.0)/SIG; r=B0*k; lo=L0*k
    eq=np.zeros(NS); hwm=np.zeros(NS); ph=np.zeros(NS,int)
    fd=np.full(NS,T+1); fees=np.ones(NS,int); tg=np.array(stages)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        if static is not None: floor_=np.full(NS,-static)
        elif lock is not None: floor_=np.where(hwm>=lock,0.0,hwm-trail)
        else: floor_=hwm-trail
        bad=live&((eq+lo[:,t])<=floor_)
        if daily is not None: bad|=live&(lo[:,t]<=-daily)
        eq[bad]=0.0; hwm[bad]=0.0; ph[bad]=0; fees[bad]+=1
        ad=live&~bad
        eq[ad]+=r[ad,t]; hwm[ad]=np.maximum(hwm[ad],eq[ad])
        cur=tg[np.minimum(ph,len(tg)-1)]
        hit=ad&(eq>=cur)
        last=hit&(ph>=len(tg)-1); nxt=hit&(ph<len(tg)-1)
        fd[last]=t; eq[nxt]=0.0; hwm[nxt]=0.0; ph[nxt]+=1
    return fd,fees

CFG=[("FTMO 2-Step (10+5, static 10%, zi 5%)",dict(stages=[0.10,0.05],static=0.10,daily=0.05),
      (0.40,0.75,1.00,1.50,2.50)),
     ("1 faza 10%, trailing 3% pur",dict(stages=[0.10],trail=0.03),
      (0.10,0.15,0.20,0.25,0.30,0.40)),
     ("1 faza 10%, trailing 3% + lock la +3%",dict(stages=[0.10],trail=0.03,lock=0.03),
      (0.15,0.25,0.40,0.60,1.00,1.50)),
     ("1 faza 10%, trailing 3% + lock, zi 3%",dict(stages=[0.10],trail=0.03,lock=0.03,daily=0.03),
      (0.15,0.25,0.40,0.60,1.00))]
for lab,kw,vols in CFG:
    print(f"\n=== {lab} ===")
    print(f"{'vol/zi':>7} {'lev':>6} {'3 luni':>8} {'4 luni':>8} {'6 luni':>8} {'12 luni':>9} "
          f"{'taxe':>7} {'med zile':>9}")
    for v in vols:
        fd,fe=sim(v,**kw); ok=fd<=T
        print(f"{v:6.2f}% {(v/100)/SIG:5.2f}x {(fd<=63).mean():8.1%} {(fd<=84).mean():8.1%} "
              f"{(fd<=126).mean():8.1%} {ok.mean():9.1%} {fe.mean():7.2f} "
              f"{np.median(fd[ok]) if ok.any() else np.nan:9.0f}")

print("\n=== de ce 3% e atat de strimt: drawdown-ul la fiecare levier ===")
b=(R-R.mean())+SH*SIG/np.sqrt(252)
print(f"{'vol/zi':>7} {'lev':>6} {'max DD istoric':>15} {'zile sub -3% de la varf':>25}")
for v in (0.15,0.25,0.40,0.60,1.00):
    k=(v/100)/SIG; x=b*k; eq=np.cumsum(x); dd=eq-np.maximum.accumulate(eq)
    print(f"{v:6.2f}% {k:5.2f}x {dd.min():15.2%} {(dd<=-0.03).mean():24.1%}")
