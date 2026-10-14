"""Walk-forward COMUN peste vm x regula de iesire (20 configuratii).

Selectia pe TRAIN (Sharpe in-sample), evaluarea pe blocul OOS imediat urmator.
Comparam cu alegerile fixe, ca sa vedem ce aduce de fapt selectia.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from itertools import product

VMS=["0.7","1.0","1.3","1.5","2.0"]
EXITS=["opposite_band","band_vwap","band_vwap_vt","close"]
S={}
for v,e in product(VMS,EXITS):
    p=f"results/backtest/wf_vm{v}/daily_{e}.csv"
    d=pd.read_csv(p); d["date"]=pd.to_datetime(d["date"])
    S[(v,e)]=d.set_index("date")["ret"]
idx=S[("1.5","opposite_band")].index
M=pd.DataFrame({k:s.reindex(idx).fillna(0.0) for k,s in S.items()})
n=len(M); A=M.to_numpy()
def sh(x): 
    s=x.std(ddof=1); return x.mean()/s*np.sqrt(252) if s>0 else -9.0

print("=== Sharpe pe toata perioada (in-sample complet), 20 configuratii ===")
print(f"{'vm':>5} " + " ".join(f"{e:>15}" for e in EXITS))
for v in VMS:
    print(f"{v:>5} " + " ".join(f"{sh(M[(v,e)].to_numpy()):>15.2f}" for e in EXITS))

T0,NW=1100,10; STEP=(n-T0)//NW
cols=list(M.columns)
print(f"\n=== WALK-FORWARD: {NW} ferestre a cate {STEP} zile OOS ===")
print(f"{'#':>2} {'perioada OOS':>23} | {'ales':>28} {'IS sh':>6} | {'OOS sh':>7} | "
      f"{'OOS 1.5/opp':>12} {'OOS 1.5/close':>14}")
sel=[];oos=[];fix_o=[];fix_c=[];fix_13o=[]
for w in range(NW):
    e=T0+w*STEP; o=slice(e,min(e+STEP,n))
    isv=[sh(A[:e,i]) for i in range(len(cols))]
    j=int(np.argmax(isv)); k=cols[j]
    so=sh(A[o,j]); po=sh(M[("1.5","opposite_band")].to_numpy()[o])
    pc=sh(M[("1.5","close")].to_numpy()[o]); p13=sh(M[("1.3","opposite_band")].to_numpy()[o])
    per=f"{idx[e].date()}..{idx[min(e+STEP,n)-1].date()}"
    sel.append(k);oos.append(so);fix_o.append(po);fix_c.append(pc);fix_13o.append(p13)
    print(f"{w+1:2d} {per:>23} | {f'vm{k[0]}/{k[1]}':>28} {isv[j]:6.2f} | {so:7.2f} | "
          f"{po:12.2f} {pc:14.2f}")

oos=np.array(oos);fix_o=np.array(fix_o);fix_c=np.array(fix_c);fix_13o=np.array(fix_13o)
print(f"\nconfiguratii alese: {pd.Series([f'vm{a}/{b}' for a,b in sel]).value_counts().to_dict()}")
print(f"\n{'strategie':>34} {'OOS mediu':>10} {'median':>8} {'pozitive':>9} {'min':>7}")
for nm,x in (("selectie walk-forward",oos),("fix vm1.5 / opposite_band",fix_o),
             ("fix vm1.5 / close",fix_c),("fix vm1.3 / opposite_band",fix_13o)):
    print(f"{nm:>34} {x.mean():10.2f} {np.median(x):8.2f} {int((x>0).sum()):6d}/{len(x)} {x.min():7.2f}")

# --- DSR cu N-ul real ---
from scipy import stats as sstats
def dsr(shr, N, T, sk, ku):
    e=0.5772156649
    emax=(1-e)*sstats.norm.ppf(1-1/N)+e*sstats.norm.ppf(1-1/(N*np.e))
    num=(shr-emax)*np.sqrt(T-1)
    den=np.sqrt(1-sk*shr+(ku-1)/4*shr**2)
    return sstats.norm.cdf(num/den)
r=M[("1.5","opposite_band")].to_numpy()
shd=r.mean()/r.std(ddof=1)   # pe zi
sk=pd.Series(r).skew(); ku=pd.Series(r).kurtosis()+3
print(f"\n=== Deflated Sharpe Ratio (vm1.5 / opposite_band) ===")
for N in (20,30,100,200):
    print(f"  N = {N:3d} incercari -> DSR = {dsr(shd,N,len(r),sk,ku):.3f}")
