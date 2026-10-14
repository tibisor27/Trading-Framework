"""Program cu buget mare pe FTMO 2-Step: mai multe conturi in paralel, la marimi
diferite, cu RELUARE imediata dupa fiecare esec. Orizont 12 luni.

Toate conturile vad ACELASI traseu de piata (corelatie reala). Conturi de aceeasi
marime pornite in aceeasi zi sunt identice -> bugetul cumpara marimi DISTINCTE
si reluari, nu duplicate.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from itertools import combinations

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG,N=R.std(ddof=1),len(R); BASE=(R-R.mean())+0.95*SIG/np.sqrt(252)
DAILY,MAXL,BLOCK,T,NS=0.05,0.10,20,252,12000
rng=np.random.default_rng(2024)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,N-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
B,LL=BASE[IX],L[IX]

def slot(vol):
    """-> (ziua finantarii sau T+1, nr de taxe consumate in 12 luni)"""
    k=(vol/100.0)/SIG; r=B*k; lo=r+LL*k
    eq=np.zeros(NS); ph=np.ones(NS,int); fd=np.full(NS,T+1); fees=np.ones(NS,int)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        rt,lt=r[:,t],lo[:,t]
        br=live&((eq+lt<=-MAXL)|(lt<=-DAILY))
        eq[br]=0.0; ph[br]=1; fees[br]+=1
        ad=live&~br
        eq[ad]+=rt[ad]
        tg=np.where(ph==1,0.10,0.05)
        hit=ad&(eq>=tg)
        # p2 SE CALCULEAZA INAINTE de a muta ph, altfel cei care tocmai au trecut
        # faza 1 apar imediat ca ph==2 si "trec" si faza 2 in aceeasi zi.
        p2=hit&(ph==2); p1=hit&(ph==1)
        fd[p2]=t
        eq[p1]=0.0; ph[p1]=2
    return fd,fees

VOLS=[0.40,0.60,0.80,1.00,1.20,1.50,2.00]
FD={};FE={}
print("=== un singur slot (cu reluare dupa esec), orizont 12 luni ===")
print(f"{'vol/zi':>7} {'6 luni':>8} {'9 luni':>8} {'12 luni':>9} {'taxe medii':>11} {'taxe|reusit':>12}")
for v in VOLS:
    fd,fe=slot(v); FD[v],FE[v]=fd,fe; ok=fd<=T
    print(f"{v:6.2f}% {(fd<=126).mean():8.1%} {(fd<=189).mean():8.1%} {ok.mean():9.1%} "
          f"{fe.mean():11.2f} {fe[ok].mean() if ok.any() else np.nan:12.2f}")

print("\n=== program: mai multe marimi in paralel ===")
print(f"{'marimi':>34} {'slot':>5} {'6 luni':>8} {'12 luni':>9} {'E[conturi]':>11} {'E[taxe]':>9}")
PROGS=[[1.00],[0.60,1.00],[0.60,1.00,1.50],[0.40,0.80,1.20],[0.40,0.80,1.20,2.00],
       [0.40,0.60,0.80,1.00,1.20,1.50],[0.40,0.60,0.80,1.00,1.20,1.50,2.00]]
for P in PROGS:
    mn=np.minimum.reduce([FD[v] for v in P])
    cnt=np.sum([ (FD[v]<=T).astype(int) for v in P],axis=0)
    fees=np.sum([FE[v] for v in P],axis=0)
    lab="+".join(f"{v:.2f}" for v in P)
    print(f"{lab:>34} {len(P):5d} {(mn<=126).mean():8.1%} {(mn<=T).mean():9.1%} "
          f"{cnt.mean():11.2f} {fees.mean():9.1f}")

print("\n=== cea mai buna combinatie de k marimi (P(cel putin unul in 12 luni)) ===")
for k in (1,2,3,4,5):
    best=max(combinations(VOLS,k),
             key=lambda c: (np.minimum.reduce([FD[v] for v in c])<=T).mean())
    mn=np.minimum.reduce([FD[v] for v in best])
    cnt=np.sum([(FD[v]<=T).astype(int) for v in best],axis=0)
    fees=np.sum([FE[v] for v in best],axis=0)
    print(f"  k={k}: {'+'.join(f'{v:.2f}' for v in best):<28} "
          f"12 luni {(mn<=T).mean():6.1%} | 6 luni {(mn<=126).mean():6.1%} | "
          f"E[conturi] {cnt.mean():.2f} | E[taxe] {fees.mean():.1f}")
