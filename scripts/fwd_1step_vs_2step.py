"""Monte Carlo inainte sub politicile HJB, cu TOATE regulile reale.
Validare independenta a numerelor din backward induction + regulile pe care HJB le ignora
(min 4 zile/faza la 2-Step, Best Day Rule 50% la 1-Step)."""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R0=d["ret"].to_numpy(float); LOD=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R0.std(ddof=1); N=len(R0); SH=0.95
BASE=(R0-R0.mean())+SH*SIG/np.sqrt(252); LOW=BASE+LOD; TRADED=(R0!=0.0)
EURUSD=1.08; FEE1=499*EURUSD; FEE2=540*EURUSD
T1,T2,MAXL=0.10,0.05,0.10; DLY1,DLY2=0.03,0.05; MIND=4
z=np.load("results/backtest/_hjb12.npz")
E,A1,A2,H,Dg,A=z["E"],z["A1"],z["A2"],z["H"],z["Dg"],z["A"]
T=252; NS,BLK=40000,20
rng=np.random.default_rng(99)
nb=int(np.ceil(T/BLK))
st=rng.integers(0,max(1,N-BLK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)

def fwd2(fixed=None, mindays=MIND):
    eq=np.zeros(NS); ph=np.ones(NS,int); fd=np.full(NS,T+1); fee=np.ones(NS,int)
    td=np.zeros(NS,int)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        k=np.clip(np.searchsorted(E,eq),0,len(E)-1)
        v=np.full(NS,fixed) if fixed else np.where(ph==1,A1[T-t][k],A2[T-t][k])
        s=(v/100.0)/SIG
        i=IX[:,t]; r=BASE[i]*s; lo=LOW[i]*s
        bad=live&((eq+lo<=-MAXL)|(lo<=-DLY2))
        eq[bad]=0;ph[bad]=1;fee[bad]+=1;td[bad]=0
        ad=live&~bad; eq[ad]+=r[ad]; td[ad]+=TRADED[i][ad]
        tg=np.where(ph==1,T1,T2)
        hit=ad&(eq>=tg)&(td>=mindays)
        p2=hit&(ph==2); p1=hit&(ph==1)
        fd[p2]=t; eq[p1]=0; ph[p1]=2; td[p1]=0
    return fd,fee

def fwd1(fixed=None, bestday=True):
    h=np.zeros(NS); dd=np.zeros(NS); fd=np.full(NS,T+1); fee=np.ones(NS,int)
    pos=np.zeros(NS); best=np.zeros(NS)
    jh=lambda x: np.clip(np.searchsorted(H,x),0,len(H)-1)
    jd=lambda x: np.clip(np.searchsorted(Dg,x),0,len(Dg)-1)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        v=np.full(NS,fixed) if fixed else A[T-t][jh(h),jd(dd)]
        s=(v/100.0)/SIG
        i=IX[:,t]; r=BASE[i]*s; lo=LOW[i]*s
        bad=live&((dd+lo<=-MAXL)|(lo<=-DLY1))
        h[bad]=0;dd[bad]=0;fee[bad]+=1;pos[bad]=0;best[bad]=0
        ad=live&~bad
        eq=h+dd+np.where(ad,r,0.0)
        p=np.maximum(r,0.0); pos[ad]+=p[ad]; best[ad]=np.maximum(best[ad],p[ad])
        nh=ad&(eq>h); h[nh]=eq[nh]; dd[nh]=0.0
        no=ad&~nh; dd[no]=eq[no]-h[no]
        ok=(best<=0.5*pos+1e-12) if bestday else np.ones(NS,bool)
        hit=ad&(eq>=T1)&ok
        fd[hit]=t
    return fd,fee

def line(lab,fd,fee,f):
    ok=fd<=T
    print(f"{lab:>34} {(fd<=63).mean():8.1%} {(fd<=126).mean():8.1%} {ok.mean():8.1%} "
          f"{np.median(fd[ok]) if ok.any() else np.nan:8.0f} "
          f"{np.percentile(fd[ok],25) if ok.any() else np.nan:6.0f} {fee.mean():7.2f} "
          f"{fee.mean()*f:9,.0f}EUR")

print(f"{'':>34} {'3 luni':>8} {'6 luni':>8} {'12 luni':>8} {'median':>8} {'p25':>6} "
      f"{'taxe':>7} {'cost mediu':>12}")
fd,fe=fwd2();                 line("2-Step, HJB, toate regulile",fd,fe,FEE2); r2=(fd,fe)
fd,fe=fwd2(mindays=0);        line("2-Step, HJB, fara min-zile",fd,fe,FEE2)
fd,fe=fwd1();                 line("1-Step, HJB, toate regulile",fd,fe,FEE1); r1=(fd,fe)
fd,fe=fwd1(bestday=False);    line("1-Step, HJB, fara Best Day",fd,fe,FEE1)
print()
for v in (0.20,0.30,0.50,0.75,1.00,1.50,2.50):
    fd,fe=fwd2(fixed=v); line(f"2-Step fix {v:.2f}%/zi",fd,fe,FEE2)
print()
for v in (0.15,0.20,0.30,0.50,0.75,1.00,1.50,2.50):
    fd,fe=fwd1(fixed=v); line(f"1-Step fix {v:.2f}%/zi",fd,fe,FEE1)

print("\n=== costul in bani pana la finantare (doar cei care reusesc in 12 luni) ===")
for lab,(fd,fe),f,refund in (("2-Step",r2,FEE2,True),("1-Step",r1,FEE1,False)):
    ok=fd<=T; c=fe[ok]*f - (f if refund else 0.0)
    print(f"{lab:>8}: taxe mediane {np.median(fe[ok]):.0f} | cost net mediu "
          f"{c.mean():,.0f}EUR | p90 {np.percentile(c,90):,.0f}EUR "
          f"| {'ultima taxa se ramburseaza' if refund else 'nimic nu se ramburseaza'}")
