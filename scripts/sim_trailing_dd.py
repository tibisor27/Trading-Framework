"""Geometrie cu DRAWDOWN TRAILING (stil futures prop / instant funding).

Diferenta fundamentala fata de FTMO 2-Step:
  static 10%  -> spatiul de sub tine CRESTE pe masura ce faci profit
  trailing 4% -> spatiul e MEREU 4%, nu construiesti niciodata tampon

Deci nu mai conteaza ziua cea mai proasta, conteaza ADANCIMEA RETRAGERILOR.
Metrica devine Calmar (CAGR / max drawdown), nu Sharpe.

Variante modelate: trail pe inchidere (EOD) vs pe maximul intraday, cu si fara
"lock" (multe firme opresc trailul cand ajungi la +X% si il fixeaza la breakeven).
"""
from __future__ import annotations
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float)
LO=(d["min_intra"]-d["ret"]).to_numpy(float)
HIx=(d["max_intra"]-d["ret"]).to_numpy(float)
SIG,N=R.std(ddof=1),len(R); SH=0.95
BASE=(R-R.mean())+SH*SIG/np.sqrt(252)
BLOCK,T,NS=20,378,12000          # 18 luni orizont
rng=np.random.default_rng(606)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,N-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,N-1)
B0,L0,H0=BASE[IX],(BASE+LO)[IX],(BASE+HIx)[IX]

def sim(vol,target,trail,intraday_trail,lock_at=None,daily=None,static=None):
    k=(vol/100.0)/SIG; r=B0*k; lo=L0*k; hi=H0*k
    eq=np.zeros(NS); hwm=np.zeros(NS); fd=np.full(NS,T+1); fees=np.ones(NS,int)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        floor_=np.where((lock_at is not None)&(hwm>=(lock_at if lock_at else 9)),0.0,hwm-trail)
        if static is not None: floor_=np.maximum(floor_,-static)
        low=eq+lo[:,t]
        bad=live&(low<=floor_)
        if daily is not None: bad|=live&(lo[:,t]<=-daily)
        eq[bad]=0.0; hwm[bad]=0.0; fees[bad]+=1
        ad=live&~bad
        if intraday_trail:
            hwm[ad]=np.maximum(hwm[ad],eq[ad]+hi[ad,t])
        eq[ad]+=r[ad,t]
        if not intraday_trail:
            hwm[ad]=np.maximum(hwm[ad],eq[ad])
        w=ad&(eq>=target); fd[w]=t
    return fd,fees

print(f"Calmar la levier 1 = {R.mean()*252/0.1018:.2f}   (CAGR {R.mean()*252:.1%} / maxDD 10.18%)\n")
print("=== trailing 4% pe INCHIDERE, tinta 8%, fara limita zilnica ===")
print(f"{'vol/zi':>7} {'levier':>7} {'P(trece)':>9} {'6 luni':>8} {'12 luni':>9} {'E[taxe]':>9} {'med zile':>9}")
for v in (0.15,0.20,0.25,0.30,0.40,0.50):
    fd,fe=sim(v,0.08,0.04,False); ok=fd<=T
    print(f"{v:6.2f}% {(v/100)/SIG:6.2f}x {ok.mean():9.1%} {(fd<=126).mean():8.1%} "
          f"{(fd<=252).mean():9.1%} {fe.mean():9.2f} {np.median(fd[ok]) if ok.any() else np.nan:9.0f}")

print("\n=== acelasi, dar trailul se BLOCHEAZA la breakeven cand ajungi la +4% ===")
print(f"{'vol/zi':>7} {'levier':>7} {'P(trece)':>9} {'6 luni':>8} {'12 luni':>9} {'E[taxe]':>9} {'med zile':>9}")
for v in (0.20,0.30,0.40,0.50,0.70,1.00):
    fd,fe=sim(v,0.08,0.04,False,lock_at=0.04); ok=fd<=T
    print(f"{v:6.2f}% {(v/100)/SIG:6.2f}x {ok.mean():9.1%} {(fd<=126).mean():8.1%} "
          f"{(fd<=252).mean():9.1%} {fe.mean():9.2f} {np.median(fd[ok]) if ok.any() else np.nan:9.0f}")

print("\n=== trailing pe MAXIMUL INTRADAY (varianta dura) vs pe inchidere, tinta 8% ===")
print(f"{'vol/zi':>7} {'EOD: P(trece)':>14} {'med zile':>9} | {'INTRADAY: P(trece)':>19} {'med zile':>9}")
for v in (0.15,0.20,0.25,0.30):
    a,_=sim(v,0.08,0.04,False); b,_=sim(v,0.08,0.04,True)
    print(f"{v:6.2f}% {(a<=T).mean():14.1%} {np.median(a[a<=T]) if (a<=T).any() else np.nan:9.0f} | "
          f"{(b<=T).mean():19.1%} {np.median(b[b<=T]) if (b<=T).any() else np.nan:9.0f}")

print("\n=== referinta: FTMO 2-Step (static 10%, zilnic 5%, tinta 10%+5%) ===")
for v in (0.40,0.60,1.00):
    fd,fe=sim(v,0.10,99.0,False,daily=0.05,static=0.10); ok=fd<=T
    print(f"{v:6.2f}% {(v/100)/SIG:6.2f}x  faza1: P {ok.mean():.1%}  med {np.median(fd[ok]) if ok.any() else np.nan:.0f} zile  "
          f"taxe {fe.mean():.2f}")
