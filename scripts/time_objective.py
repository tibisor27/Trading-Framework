"""Obiectiv CONSTIENT DE TIMP + toate masuratorile pe axa timpului.

Vechi : V constant  -> finantat in ziua 30 si in ziua 250 valoreaza la fel. Gresit.
Nou   : V(t) = rata_venit x timp_ramas_de_castigat, pe un orizont total de 3 ani.
        Finantat cu t zile ramase din an => mai ai (504+t)/252 ani de castigat.
"""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R0=d["ret"].to_numpy(float); LOD=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG0=R0.std(ddof=1); T1,T2,MAXL,DAILY=0.10,0.05,0.10,0.05
C,T=0.0055,252; RATE=0.06*0.9                      # 6%/an pe cont, split 90%
E=np.linspace(-MAXL,T1,41); i0=int(np.searchsorted(E,0.0))
VOLS=np.array([0.3,0.6,1.0,1.6,2.5])
Vconst=0.137*0.9
def Vtime(t): return RATE*(504+t)/252.0            # 16.2% daca esti finantat imediat, 10.8% la limita

def solve(R,LO,timeaware):
    sig=R.std(ddof=1); K={v:((R*(v/100)/sig),((R+LO)*(v/100)/sig)) for v in VOLS}
    W1=np.zeros(len(E));W2=np.zeros(len(E));F1=np.zeros(len(E));F2=np.zeros(len(E))
    A1=np.zeros((T+1,len(E)));A2=np.zeros((T+1,len(E)))
    for t in range(1,T+1):
        Vt=Vtime(t) if timeaware else Vconst
        rW,rF=W1[i0]-C,F1[i0]+1.0; nW1=nW2=None
        for ph in (2,1):
            tgt=T1 if ph==1 else T2
            Wp,Fp=(W1,F1) if ph==1 else (W2,F2)
            wW=(nW2[i0] if ph==1 else Vt); wF=(nF2[i0] if ph==1 else 0.0)
            best=np.full(len(E),-1e9);bf=np.zeros(len(E));ba=np.zeros(len(E))
            for v in VOLS:
                rk,lok=K[v]
                brk=(lok<=-DAILY)[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
                en=E[:,None]+rk[None,:]; w=en>=tgt; fl=en.ravel()
                nw=np.where(w,wW,np.interp(fl,E,Wp).reshape(en.shape))
                nf=np.where(w,wF,np.interp(fl,E,Fp).reshape(en.shape))
                val=np.where(brk,rW,nw).mean(axis=1); m=val>best
                best[m]=val[m]; ba[m]=v; bf[m]=np.where(brk,rF,nf).mean(axis=1)[m]
            if ph==1: nW1,nF1,A1[t]=best,bf,ba
            else:     nW2,nF2,A2[t]=best,bf,ba
        W1,F1,W2,F2=nW1,nF1,nW2,nF2
    return A1,A2,W1[i0]

NS,BLOCK=15000,20
rng=np.random.default_rng(808)
def forward(R,LO,A1,A2):
    """distributia ZILEI finantarii sub politica data"""
    sig=R.std(ddof=1); N=len(R)
    nb=int(np.ceil(T/BLOCK))
    st=rng.integers(0,N-BLOCK+1,size=(NS,nb))
    IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
    eq=np.zeros(NS);ph=np.ones(NS,int);fd=np.full(NS,T+1);fee=np.ones(NS,int)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        rem=T-t
        j=np.clip(np.searchsorted(E,eq),0,len(E)-1)
        v=np.where(ph==1,A1[rem][j],A2[rem][j])
        k=(v/100.0)/sig
        r=R[IX[:,t]]*k; lo=(R+LO)[IX[:,t]]*k
        bad=live&((eq+lo<=-MAXL)|(lo<=-DAILY))
        eq[bad]=0;ph[bad]=1;fee[bad]+=1
        ad=live&~bad; eq[ad]+=r[ad]
        tg=np.where(ph==1,T1,T2); hit=ad&(eq>=tg)
        p2=hit&(ph==2);p1=hit&(ph==1); fd[p2]=t; eq[p1]=0;ph[p1]=2
    return fd,fee

def dem(R,sh): return (R-R.mean())+sh*R.std(ddof=1)/np.sqrt(252)
def thin(R,cap): return np.minimum(0.0,np.maximum(R+LOD,-cap)-R)

print("=== 1. obiectivul vechi (orb la timp) vs cel nou (plateste viteza) ===")
print(f"{'obiectiv':>22} {'ziua1':>7} {'3 luni':>8} {'6 luni':>8} {'9 luni':>8} {'12 luni':>8} "
      f"{'median':>8} {'p25':>6} {'taxe':>6}")
base=dem(R0,0.95)
for ta,lab in ((False,"vechi: V constant"),(True,"nou: V x timp ramas")):
    A1,A2,_=solve(base,LOD,ta); fd,fe=forward(base,LOD,A1,A2); ok=fd<=T
    print(f"{lab:>22} {A1[T][i0]:6.2f}% {(fd<=63).mean():8.1%} {(fd<=126).mean():8.1%} "
          f"{(fd<=189).mean():8.1%} {ok.mean():8.1%} "
          f"{np.median(fd[ok]) if ok.any() else np.nan:8.0f} "
          f"{np.percentile(fd[ok],25) if ok.any() else np.nan:6.0f} {fe.mean():6.2f}")

print("\n=== 2. SENSIBILITATE PE AXA TIMPULUI (obiectiv nou) ===")
print(f"{'varianta':>28} {'ziua1':>7} {'3 luni':>8} {'6 luni':>8} {'12 luni':>8} "
      f"{'median':>8} {'p25':>6} {'p75':>6} {'taxe':>6}")
def run(lab,R,LO):
    A1,A2,_=solve(R,LO,True); fd,fe=forward(R,LO,A1,A2); ok=fd<=T
    q=lambda p: np.percentile(fd[ok],p) if ok.any() else np.nan
    print(f"{lab:>28} {A1[T][i0]:6.2f}% {(fd<=63).mean():8.1%} {(fd<=126).mean():8.1%} "
          f"{ok.mean():8.1%} {np.median(fd[ok]):8.0f} {q(25):6.0f} {q(75):6.0f} {fe.mean():6.2f}")
for sh in (0.60,0.95,1.35,1.90):
    run(f"Sharpe {sh:.2f}, coada reala", dem(R0,sh), LOD)
print()
for sh in (0.60,0.95,1.35):
    b=dem(R0,sh); run(f"Sharpe {sh:.2f}, coada la -2%", b, thin(b,0.02))
