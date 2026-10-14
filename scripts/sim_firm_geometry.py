"""Compara geometriile: FTMO 2-Step, FTMO 1-Step (trailing EOD), si o firma ipotetica
fara limita zilnica. Acelasi traseu de piata pentru toate -> comparatie pereche.
"""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG,N=R.std(ddof=1),len(R); BASE=(R-R.mean())+0.95*SIG/np.sqrt(252)
BLOCK,T,NS=20,1000,6000
rng=np.random.default_rng(909)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,N-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
B,LL=BASE[IX],L[IX]

def phase(r,lo,target,daily,maxl,trailing,start_day):
    """-> (trecut, ziua). start_day: vector de zile de pornire per sim."""
    n=r.shape[0]
    eq=np.zeros(n); hwm=np.zeros(n); alive=np.ones(n,bool); done=np.zeros(n,bool)
    day=np.full(n,T+1); t=start_day.copy()
    for step in range(T):
        act=alive&~done&(t<T)
        if not act.any(): break
        i=np.where(act)[0]; rr=r[i,t[i]]; ll=lo[i,t[i]]
        floor_=(hwm[i]-maxl) if trailing else -maxl
        bad=((eq[i]+ll)<=floor_)|(ll<=-daily)
        alive[i[bad]]=False
        ok=i[~bad]
        eq[ok]+=r[ok,t[ok]]
        w=ok[eq[ok]>=target]; done[w]=True; day[w]=t[w]
        if trailing: hwm[ok]=np.maximum(hwm[ok],eq[ok])
        t[i]+=1
    return done, day

def firm(vol,name,stages,daily,maxl,trailing):
    k=(vol/100.0)/SIG; r=B*k; lo=r+LL*k
    start=np.zeros(NS,int); alive=np.ones(NS,bool); fin=np.full(NS,T+1)
    for tg in stages:
        ok,day=phase(r,lo,tg,daily,maxl,trailing,start)
        ok&=alive; alive=ok; start=np.where(ok,np.minimum(day+1,T),T)
        fin=np.where(ok,day,T+1)
    return fin

FIRMS=[("FTMO 2-Step", [0.10,0.05], 0.05, 0.10, False),
       ("FTMO 1-Step", [0.10],      0.03, 0.10, True),
       ("ipotetic: fara limita zilnica", [0.10,0.05], 9.99, 0.10, False)]
VOLS=[0.25,0.30,0.40,0.50,0.60,0.80,1.00,1.40]
for name,stg,dl,ml,tr in FIRMS:
    print(f"\n=== {name} ===")
    print(f"{'vol/zi':>7} {'P(trece)':>9} {'6 luni':>8} {'12 luni':>8} {'24 luni':>8} "
          f"{'median zile':>12} {'ani->finantat':>14}")
    for v in VOLS:
        f=firm(v,name,stg,dl,ml,tr); ok=f<=T; p=ok.mean()
        med=np.median(f[ok]) if ok.any() else np.nan
        dur=np.where(ok,f,T).mean()
        print(f"{v:6.2f}% {p:9.3f} {(f<=126).mean():8.1%} {(f<=252).mean():8.1%} "
              f"{(f<=504).mean():8.1%} {med:12.0f} {dur/max(p,1e-9)/252:14.2f}")
