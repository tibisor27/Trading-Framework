"""Ce conteaza de fapt: Sharpe, coada, asimetria sau kurtoza?

Perturb fiecare proprietate pe rand, re-rezolv programarea dinamica si masor
valoarea neta. Obiectiv: V=12.33% pe cont finantat, taxa c=0.55%, plafon 2.5%/zi.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R0=d["ret"].to_numpy(float); LOD=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG0=R0.std(ddof=1); T1,T2,MAXL,DAILY=0.10,0.05,0.10,0.05
VAL,C,T=0.137*0.9,0.0055,252
E=np.linspace(-MAXL,T1,41); i0=int(np.searchsorted(E,0.0))
VOLS=np.array([0.3,0.6,1.0,1.6,2.5])

def solve(R,LO):
    """R: randamente zilnice. LO: excursia sub inchidere (<=0). Grila de vol fixa."""
    sig=R.std(ddof=1)
    W1=np.zeros(len(E));W2=np.zeros(len(E));P1=np.zeros(len(E));P2=np.zeros(len(E))
    F1=np.zeros(len(E));F2=np.zeros(len(E));A1=0
    K={}
    for v in VOLS:
        k=(v/100.0)/sig; K[v]=(R*k,(R+LO)*k)
    for t in range(1,T+1):
        rW,rP,rF=W1[i0]-C,P1[i0],F1[i0]+1.0
        nW1=nW2=None
        for ph in (2,1):
            tgt=T1 if ph==1 else T2
            Wp,Pp,Fp=(W1,P1,F1) if ph==1 else (W2,P2,F2)
            wW=(nW2[i0] if ph==1 else VAL); wP=(nP2[i0] if ph==1 else 1.0)
            wF=(nF2[i0] if ph==1 else 0.0)
            best=np.full(len(E),-1e9);bp=np.zeros(len(E));bf=np.zeros(len(E));ba=np.zeros(len(E))
            for v in VOLS:
                rk,lok=K[v]
                brk=(lok<=-DAILY)[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
                en=E[:,None]+rk[None,:]; w=en>=tgt; fl=en.ravel()
                nw=np.where(w,wW,np.interp(fl,E,Wp).reshape(en.shape))
                np_=np.where(w,wP,np.interp(fl,E,Pp).reshape(en.shape))
                nf=np.where(w,wF,np.interp(fl,E,Fp).reshape(en.shape))
                val=np.where(brk,rW,nw).mean(axis=1); m=val>best
                best[m]=val[m]; ba[m]=v
                bp[m]=np.where(brk,rP,np_).mean(axis=1)[m]
                bf[m]=np.where(brk,rF,nf).mean(axis=1)[m]
            if ph==1: nW1,nP1,nF1,A1=best,bp,bf,ba[i0]
            else:     nW2,nP2,nF2=best,bp,bf
        W1,P1,F1,W2,P2,F2=nW1,nP1,nF1,nW2,nP2,nF2
    return P1[i0],F1[i0],W1[i0],A1

def stats_of(R,LO):
    s=R.std(ddof=1)
    return dict(sh=R.mean()/s*np.sqrt(252), sig=s, sk=stats.skew(R), ku=stats.kurtosis(R),
                worst=(R+LO).min(), ceil=0.05*s/abs((R+LO).min())*100)

def row(lab,R,LO):
    st=stats_of(R,LO); p,f,w,a=solve(R,LO)
    print(f"{lab:>30} {st['sh']:6.2f} {st['sk']:+6.2f} {st['ku']:6.1f} {st['worst']:8.2%} "
          f"{st['ceil']:7.3f}% {p:8.1%} {f:6.2f} {w:8.2%} {a:6.2f}%")
    return w

HDR=(f"{'varianta':>30} {'Sharpe':>6} {'skew':>6} {'kurt':>6} {'cea mai':>8} {'plafon':>8} "
     f"{'P(12l)':>8} {'taxe':>6} {'valoare':>8} {'ziua1':>7}")
def demean(R,sh): 
    return (R-R.mean())+sh*R.std(ddof=1)/np.sqrt(252)

print("=== A. doar SHARPE se schimba (forma identica) ===")
print(HDR)
base=demean(R0,0.95)
for sh in (0.60,0.95,1.35,1.90,2.70): row(f"Sharpe {sh:.2f}", demean(R0,sh), LOD)

print("\n=== B. doar COADA EXCURSIEI se schimba (randamente si Sharpe identice) ===")
print(HDR)
for cap in (None,0.04,0.03,0.02,0.015):
    LO2=LOD if cap is None else np.maximum(LOD, -cap-base)  # excursie limitata la -cap fata de 0
    LO2=LOD if cap is None else np.minimum(0.0,np.maximum(base+LOD,-cap)-base)
    row("fara limita" if cap is None else f"excursie oprita la -{cap:.1%}", base, LO2)

print("\n=== C. interventii REALE pe randamente (schimba si Sharpe) ===")
print(HDR)
for lab,f_ in (("winsorizare pierderi la -1.5%", lambda r: np.maximum(r,-0.015)),
               ("winsorizare pierderi la -1.0%", lambda r: np.maximum(r,-0.010)),
               ("winsorizare castiguri la +2%",  lambda r: np.minimum(r, 0.020)),
               ("ambele coase taiate la 2%",     lambda r: np.clip(r,-0.02,0.02))):
    Rn=f_(R0); Rn=(Rn-Rn.mean())+ (R0.mean()-0.0)  # pastram media reala pierduta/castigata
    Rn=f_(R0)   # fara re-centrare: interventia isi plateste costul
    Rn=(Rn-R0.mean())*0+Rn
    row(lab, Rn, LOD)

print("\n=== D. intrebarea ta: Sharpe mic + coada subtire  VS  Sharpe mare + coada groasa ===")
print(HDR)
LO_thin=np.minimum(0.0,np.maximum(base+LOD,-0.02)-base)
row("Sharpe 1.35, coada reala", demean(R0,1.35), LOD)
row("Sharpe 0.95, coada reala", demean(R0,0.95), LOD)
row("Sharpe 0.95, coada la -2%", demean(R0,0.95), LO_thin)
row("Sharpe 0.70, coada la -2%", demean(R0,0.70), LO_thin)
row("Sharpe 0.60, coada la -2%", demean(R0,0.60), LO_thin)
