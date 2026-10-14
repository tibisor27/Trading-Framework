"""Sizing ADAPTIV in interiorul challenge-ului.

Pozitia ta fata de bariere se schimba pe parcurs. La +8% cu 2% de facut si 18% de
spatiu dedesubt, esti in alta problema decat in ziua 1. Politica fixa ignora asta.

  FIX(v)              : mereu v
  TREPTE(hi,lo,theta) : hi pana bancarizezi +theta in faza curenta, apoi lo
  RITM(cap)           : sigma = cat iti trebuie ca driftul singur sa acopere
                        distanta ramasa in timpul ramas, plafonat la cap
"""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG,N=R.std(ddof=1),len(R)
SH=0.95; BASE=(R-R.mean())+SH*SIG/np.sqrt(252); SHD=SH/np.sqrt(252)
DAILY,MAXL,BLOCK,T,NS=0.05,0.10,252,12000,None
T=252; NS=12000
rng=np.random.default_rng(4242)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,N-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
B0,L0=BASE[IX],(BASE+L)[IX]          # nescalate: se inmultesc cu k_t pe zi

def run(policy):
    eq=np.zeros(NS); ph=np.ones(NS,int); fd=np.full(NS,T+1); fees=np.ones(NS,int)
    start=np.zeros(NS,int)            # ziua in care a inceput incercarea curenta
    for t in range(T):
        live=fd>T
        if not live.any(): break
        tgt=np.where(ph==1,0.10,0.05)
        v=policy(eq,ph,tgt,t,start)               # %/zi, per sim
        k=(v/100.0)/SIG
        rt=B0[:,t]*k; lt=L0[:,t]*k
        br=live&((eq+lt<=-MAXL)|(lt<=-DAILY))
        eq[br]=0.0; ph[br]=1; fees[br]+=1; start[br]=t+1
        ad=live&~br
        eq[ad]+=rt[ad]
        hit=ad&(eq>=tgt)
        p2=hit&(ph==2); p1=hit&(ph==1)
        fd[p2]=t
        eq[p1]=0.0; ph[p1]=2
    return fd,fees

def FIX(v):  return lambda eq,ph,tgt,t,s: np.full(NS,v)
def TREPTE(hi,lo,th): return lambda eq,ph,tgt,t,s: np.where(eq>=th,lo,hi)
def RITM(cap,floor=0.30,horizon=252):
    def f(eq,ph,tgt,t,s):
        left=np.maximum(horizon-(t-s),20)         # zile ramase din bugetul incercarii
        need=np.maximum(tgt-eq,1e-4)/left         # drift zilnic necesar
        return np.clip(need/SHD*100.0,floor,cap)  # sigma care da acel drift
    return f

POL=[("FIX 0.60",FIX(0.60)),("FIX 0.80",FIX(0.80)),("FIX 1.00",FIX(1.00)),
     ("FIX 1.20",FIX(1.20)),("FIX 1.50",FIX(1.50)),("FIX 2.00",FIX(2.00)),
     ("TREPTE 1.20->0.50 la +4%",TREPTE(1.20,0.50,0.04)),
     ("TREPTE 1.50->0.50 la +4%",TREPTE(1.50,0.50,0.04)),
     ("TREPTE 2.00->0.50 la +4%",TREPTE(2.00,0.50,0.04)),
     ("TREPTE 2.00->0.40 la +6%",TREPTE(2.00,0.40,0.06)),
     ("RITM cap 1.00",RITM(1.00)),("RITM cap 1.50",RITM(1.50)),
     ("RITM cap 2.00",RITM(2.00)),("RITM cap 2.50",RITM(2.50)),
     ("RITM cap 2.00, buget 180z",RITM(2.00,horizon=180)),
     ("RITM cap 2.50, buget 150z",RITM(2.50,horizon=150))]

print(f"{'politica':>28} {'6 luni':>8} {'12 luni':>9} {'E[taxe]':>9} "
      f"{'P/taxa':>8} {'taxe|reusit':>12}")
res=[]
for nm,p in POL:
    fd,fe=run(p); ok=fd<=T
    pp=ok.mean(); ef=fe.mean()
    res.append((nm,(fd<=126).mean(),pp,ef,pp/ef))
    print(f"{nm:>28} {(fd<=126).mean():8.1%} {pp:9.1%} {ef:9.2f} {pp/ef:8.1%} "
          f"{fe[ok].mean() if ok.any() else np.nan:12.2f}")

print("\n=== frontiera eficienta (nedominate: nimic nu are si P mai mare si taxe mai putine) ===")
res.sort(key=lambda x:x[3])
best=-1
for nm,p6,p12,ef,pe in res:
    if p12>best:
        best=p12; print(f"  {nm:>28}  taxe {ef:5.2f}  ->  12 luni {p12:6.1%}  (6 luni {p6:5.1%})")
