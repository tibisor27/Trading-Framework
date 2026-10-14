"""HJB pe ambele geometrii FTMO, cu regulile reale si obiectiv constient de timp.

2-Step : stare 1D (equity), 2 faze, prag STATIC -10%, zi 5%
1-Step : stare 2D (hwm, drawdown sub hwm), prag TRAILING EOD, zi 3%
         h = cel mai mare sold de inchidere (>=0), d = equity - h  (<=0)
         spargere: d + lo <= -10%   SAU   lo <= -3%
         trecere : h + d + r >= 10%
         tranzitie: daca d+r>0 -> (h'=h+d+r, d'=0)  altfel  (h'=h, d'=d+r)
"""
from __future__ import annotations
import numpy as np, pandas as pd, time

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R0=d["ret"].to_numpy(float); LOD=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R0.std(ddof=1); N=len(R0); SH=0.95
BASE=(R0-R0.mean())+SH*SIG/np.sqrt(252); LOW=BASE+LOD
TRADED=(R0!=0.0)
EURUSD=1.08
FEE1=499*EURUSD/1e5; FEE2=540*EURUSD/1e5
T1,T2,MAXL=0.10,0.05,0.10; DLY1,DLY2=0.03,0.05
RATE1,RATE2=np.load("results/backtest/_rates.npy")
VOLS=np.array([0.20,0.30,0.50,0.75,1.00,1.50,2.00,2.50])
HTOT=504
def V1(t): return RATE1*(HTOT+t)/252.0
def V2(t): return RATE2*(HTOT+t)/252.0 + FEE2      # taxa 2-Step se intoarce la prima retragere

# --------------------------- 2-Step: HJB 1D --------------------------------
def solve2(T):
    E=np.linspace(-MAXL,T1,81); i0=int(np.searchsorted(E,0.0))
    K={v:(BASE*((v/100)/SIG), LOW*((v/100)/SIG)) for v in VOLS}
    BRK={v:((K[v][1]<=-DLY2)[None,:]|((E[:,None]+K[v][1][None,:])<=-MAXL)) for v in VOLS}
    W1=np.zeros(len(E));W2=np.zeros(len(E));P1=np.zeros(len(E));P2=np.zeros(len(E))
    F1=np.zeros(len(E));F2=np.zeros(len(E))
    A1=np.zeros((T+1,len(E)));A2=np.zeros((T+1,len(E)))
    for t in range(1,T+1):
        rW,rP,rF=W1[i0]-FEE2,P1[i0],F1[i0]+1.0; n1=n2=None
        for ph in (2,1):
            tgt=T1 if ph==1 else T2
            Wp,Pp,Fp=(W1,P1,F1) if ph==1 else (W2,P2,F2)
            wW=(n2[0][i0] if ph==1 else V2(t)); wP=(n2[1][i0] if ph==1 else 1.0)
            wF=(n2[2][i0] if ph==1 else 0.0)
            best=np.full(len(E),-1e9);ba=np.zeros(len(E));bp=np.zeros(len(E));bf=np.zeros(len(E))
            for v in VOLS:
                rk,lok=K[v]; brk=BRK[v]
                en=E[:,None]+rk[None,:]; fl=en.ravel(); w=en>=tgt
                nw=np.where(w,wW,np.interp(fl,E,Wp).reshape(en.shape))
                val=np.where(brk,rW,nw).mean(axis=1); m=val>best
                if m.any():
                    best[m]=val[m]; ba[m]=v
                    npb=np.where(w,wP,np.interp(fl,E,Pp).reshape(en.shape))
                    nfb=np.where(w,wF,np.interp(fl,E,Fp).reshape(en.shape))
                    bp[m]=np.where(brk,rP,npb).mean(axis=1)[m]
                    bf[m]=np.where(brk,rF,nfb).mean(axis=1)[m]
            if ph==1: n1=(best,bp,bf); A1[t]=ba
            else:     n2=(best,bp,bf); A2[t]=ba
        W1,P1,F1=n1; W2,P2,F2=n2
    return E,i0,A1,A2,W1[i0],P1[i0],F1[i0]

# --------------------------- 1-Step: HJB 2D --------------------------------
def solve1(T,NH=16,ND=41):
    H=np.linspace(0.0,T1,NH); Dg=np.linspace(-MAXL,0.0,ND); id0=ND-1
    K={v:(BASE*((v/100)/SIG), LOW*((v/100)/SIG)) for v in VOLS}
    PRE={}
    for v in VOLS:
        rk,lok=K[v]
        dr=Dg[:,None]+rk[None,:]                       # (ND,N)
        nh=dr>0.0
        brk=(lok<=-DLY1)[None,:]|((Dg[:,None]+lok[None,:])<=-MAXL)
        ep=H[:,None,None]+dr[None,:,:]                 # (NH,ND,N) equity noua
        PRE[v]=(dr,nh,brk,ep,ep>=T1)
    W=np.zeros((NH,ND));P=np.zeros((NH,ND));F=np.zeros((NH,ND))
    A=np.zeros((T+1,NH,ND),dtype=np.float32)
    for t in range(1,T+1):
        rW,rP,rF=W[0,id0]-FEE1,P[0,id0],F[0,id0]+1.0
        wW,wP,wF=V1(t),1.0,0.0
        best=np.full((NH,ND),-1e9);ba=np.zeros((NH,ND));bp=np.zeros((NH,ND));bf=np.zeros((NH,ND))
        Wd0,Pd0,Fd0=W[:,id0],P[:,id0],F[:,id0]
        for v in VOLS:
            dr,nh,brk,ep,pas=PRE[v]
            fl=ep.ravel()
            nwB=np.interp(fl,H,Wd0).reshape(ep.shape)           # varf nou -> d'=0
            nwA=np.empty_like(ep)
            drf=dr.ravel()
            for ih in range(NH): nwA[ih]=np.interp(drf,Dg,W[ih]).reshape(dr.shape)
            nw=np.where(nh[None,:,:],np.where(pas,wW,nwB),nwA)
            val=np.where(brk[None,:,:],rW,nw).mean(axis=2); m=val>best
            if m.any():
                best[m]=val[m]; ba[m]=v
                npB=np.interp(fl,H,Pd0).reshape(ep.shape); nfB=np.interp(fl,H,Fd0).reshape(ep.shape)
                npA=np.empty_like(ep); nfA=np.empty_like(ep)
                for ih in range(NH):
                    npA[ih]=np.interp(drf,Dg,P[ih]).reshape(dr.shape)
                    nfA[ih]=np.interp(drf,Dg,F[ih]).reshape(dr.shape)
                npx=np.where(nh[None,:,:],np.where(pas,wP,npB),npA)
                nfx=np.where(nh[None,:,:],np.where(pas,wF,nfB),nfA)
                bp[m]=np.where(brk[None,:,:],rP,npx).mean(axis=2)[m]
                bf[m]=np.where(brk[None,:,:],rF,nfx).mean(axis=2)[m]
        W,P,F,A[t]=best,bp,bf,ba
    return H,Dg,A,W[0,id0],P[0,id0],F[0,id0]

if __name__=="__main__":
    T=252; t0=time.time()
    E,i0,A1,A2,w2,p2,f2=solve2(T); print(f"2-Step rezolvat in {time.time()-t0:.0f}s",flush=True)
    t0=time.time()
    H,Dg,A,w1,p1,f1=solve1(T); print(f"1-Step rezolvat in {time.time()-t0:.0f}s",flush=True)
    np.savez("results/backtest/_hjb12.npz",E=E,A1=A1,A2=A2,H=H,Dg=Dg,A=A,
             w1=w1,p1=p1,f1=f1,w2=w2,p2=p2,f2=f2)
    print(f"\n{'':>10} {'P(finantat 12l)':>16} {'E[taxe]':>9} {'valoare':>9}")
    print(f"{'2-Step':>10} {p2:16.1%} {f2:9.2f} {w2:9.4f}")
    print(f"{'1-Step':>10} {p1:16.1%} {f1:9.2f} {w1:9.4f}")
    print(f"\nmarime optima la start de cont (faza 1 / unica), dupa zile ramase:")
    jd0=len(Dg)-1
    for h in (252,189,126,84,63,42,21):
        print(f"  {h:3d} zile: 2-Step {A1[h][i0]:5.2f}%/zi | 1-Step {A[h][0,jd0]:5.2f}%/zi")
