"""Deflated Sharpe Ratio, Bailey & Lopez de Prado (2014), formula COMPLETA.

    SR0 = sqrt(V[SR]) * [ (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N*e)) ]
    DSR = Phi[ (SR - SR0)*sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) ]

V[SR] = varianta Sharpe-urilor PESTE incercari. Fara ea, SR0 e in unitati gresite
si DSR iese 0 pentru orice. (Asta a fost bug-ul.)
"""
import numpy as np, pandas as pd
from itertools import product
from scipy.stats import norm

VMS=["0.7","1.0","1.3","1.5","2.0"]; EXITS=["opposite_band","band_vwap","band_vwap_vt","close"]
S={}
for v,e in product(VMS,EXITS):
    d=pd.read_csv(f"results/backtest/wf_vm{v}/daily_{e}.csv")
    r=d["ret"].to_numpy(float); S[(v,e)]=r.mean()/r.std(ddof=1)   # Sharpe ZILNIC
srs=np.array(list(S.values())); V=srs.var(ddof=1)
print(f"{len(srs)} configuratii rulate | Sharpe zilnic: medie {srs.mean():.4f} "
      f"std {np.sqrt(V):.4f}  (anualizat: {srs.mean()*np.sqrt(252):.2f} +/- {np.sqrt(V)*np.sqrt(252):.2f})")

d=pd.read_csv("results/backtest/wf_vm1.5/daily_opposite_band.csv")
r=d["ret"].to_numpy(float); T=len(r)
SR=r.mean()/r.std(ddof=1); g3=pd.Series(r).skew(); g4=pd.Series(r).kurtosis()+3.0
print(f"candidat: vm1.5/opposite_band | SR zilnic {SR:.4f} (anual {SR*np.sqrt(252):.2f}) "
      f"| T {T} | skew {g3:.2f} | kurtoza {g4:.1f}\n")

g=0.5772156649
den=np.sqrt(1 - g3*SR + (g4-1)/4*SR**2)
print(f"{'N incercari':>12} {'SR0 anual':>11} {'z':>7} {'DSR':>8}")
for N in (20,30,50,100,200,500):
    emax=(1-g)*norm.ppf(1-1/N)+g*norm.ppf(1-1/(N*np.e))
    SR0=np.sqrt(V)*emax
    z=(SR-SR0)*np.sqrt(T-1)/den
    print(f"{N:12d} {SR0*np.sqrt(252):11.2f} {z:7.2f} {norm.cdf(z):8.4f}")

print("\nla cate incercari DSR scade sub 0.95?")
for N in range(20, 20001, 20):
    emax=(1-g)*norm.ppf(1-1/N)+g*norm.ppf(1-1/(N*np.e))
    if norm.cdf((SR-np.sqrt(V)*emax)*np.sqrt(T-1)/den) < 0.95:
        print(f"  N = {N}"); break
else: print("  peste 20.000")
