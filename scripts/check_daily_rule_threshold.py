"""Venitul real pe 12 luni + de unde vine pragul de volatilitate (regula zilnica de 5%)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/claude" if False else ".").resolve()
d = pd.read_csv("results/backtest/noise_area_USATECHIDXUSD_1m/excursion_opposite_band.csv")
R = d["ret"].to_numpy(float); DLO = (d["min_intra"]-d["ret"]).to_numpy(float)
SIG, MU, N = R.std(ddof=1), R.mean(), len(R)
DAILY, MAXL, BLOCK, SH = 0.05, 0.10, 20, 0.95
rng = np.random.default_rng(23)

def draw(k, mu, m, days):
    nb = int(np.ceil(days/BLOCK))
    st = rng.integers(0, max(1, N-BLOCK+1), size=(m, nb))
    ix = np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(m,-1)[:,:days],0,N-1)
    r = (R[ix]-MU)*k + mu
    return r, r + DLO[ix]*k

print("=== 1. venit real pe 12 luni, retrageri lunare (corectat) ===")
print(f"{'vol/zi':>7} {'viu 12l':>9} {'retras 12l':>12} {'retras|viu':>12} {'CAGR':>7}")
for v in (0.30,0.40,0.50,0.60,0.75):
    k, mu = (v/100)/SIG, SH*(v/100)/np.sqrt(252)
    alive_n=0; paid=0.0; done=0
    while done < 8000:
        m = min(1000, 8000-done); r, lo = draw(k, mu, m, 252)
        eq=np.zeros(m); alive=np.ones(m,bool); tot=np.zeros(m)
        for t in range(252):
            hit = alive & (((eq+lo[:,t])<=-MAXL) | (lo[:,t]<=-DAILY))
            alive &= ~hit
            eq = np.where(alive, eq+r[:,t], eq)
            if (t+1)%21==0:
                w=np.where(alive&(eq>0),eq,0.0); tot+=w; eq-=w
        alive_n+=int(alive.sum()); paid+=float(tot.sum()); done+=m
    print(f"{v:6.2f}% {alive_n/8000:9.3f} {paid/8000:11.2%} "
          f"{paid/max(1,alive_n):11.2%} {SH*v/100*np.sqrt(252):6.1%}")

print("\n=== 2. de unde vine pragul: cu vs fara regula zilnica de 5% ===")
print(f"{'vol/zi':>7} {'P(amb) cu regula':>18} {'P(amb) fara':>13} {'cat costa regula':>18}")
for v in (0.40,0.45,0.50,0.55,0.60,0.65,0.75,1.00):
    k, mu = (v/100)/SIG, SH*(v/100)/np.sqrt(252)
    res={}
    for use in (True, False):
        pr=[]
        for tg in (0.10,0.05):
            w=0; done=0
            while done<6000:
                m=min(1000,6000-done); r,lo=draw(k,mu,m,1500)
                eq=np.cumsum(r,axis=1); eqlo=eq-r+lo
                f = (eqlo<=-MAXL)
                if use: f = f | (lo<=-DAILY)
                g = eq>=tg; big=1501
                tw=np.where(g.any(1),g.argmax(1),big); tf=np.where(f.any(1),f.argmax(1),big)
                w+=int((tw<tf).sum()); done+=m
            pr.append(w/6000)
        res[use]=pr[0]*pr[1]
    print(f"{v:6.2f}% {res[True]:18.3f} {res[False]:13.3f} "
          f"{res[False]-res[True]:17.3f}")
