"""Walk-forward pe criteriul FIRMEI: plafonul de levier se estimeaza pe TRAIN,
randamentul si siguranta se masoara pe OOS. Nimic din OOS nu atinge alegerea.
"""
from __future__ import annotations
import numpy as np, pandas as pd, os

bars=pd.read_parquet("data/raw/USATECHIDXUSD_1m.parquet",columns=["high","low"]).sort_index()
if bars.index.tz is None: bars.index=bars.index.tz_localize("UTC")
IXB,HI,LO=bars.index,bars["high"].to_numpy(),bars["low"].to_numpy()

def series(dirp,ex):
    t=pd.read_csv(f"{dirp}/trades_{ex}.csv")
    ti=pd.to_datetime(t["t_in"],utc=True).to_numpy(); to=pd.to_datetime(t["t_out"],utc=True).to_numpy()
    a=IXB.searchsorted(ti,"left"); b=IXB.searchsorted(to,"right")
    sd=t["side"].to_numpy(float); pin=t["price_in"].to_numpy(float)
    lev=t["leverage"].to_numpy(float); cst=(t["ret_net"]-t["ret_gross"]).to_numpy(float)
    mae=np.zeros(len(t))
    for i in range(len(t)):
        s,e=a[i],b[i]
        if e<=s: continue
        g=(LO[s:e].min()/pin[i]-1.0) if sd[i]>0 else (1.0-HI[s:e].max()/pin[i])
        mae[i]=(g+cst[i])*lev[i]
    t["mae"]=mae
    rows=[]
    for day,g in t.groupby("date",sort=True):
        rz=lo=0.0
        for r in g.sort_values("t_in").itertuples(): lo=min(lo,rz+r.mae); rz+=r.pnl
        rows.append((day,rz,lo))
    z=pd.DataFrame(rows,columns=["date","r","m"]); z["date"]=pd.to_datetime(z["date"]).dt.date
    cal=pd.read_csv(f"{dirp}/daily_{ex}.csv"); cal["date"]=pd.to_datetime(cal["date"]).dt.date
    f=cal.merge(z,on="date",how="left").fillna({"r":0.0,"m":0.0})
    return f["r"].to_numpy(),f["m"].to_numpy(),pd.to_datetime(f["date"]).to_numpy()

CAND={"vm1.0/band_vwap":("results/backtest/wf_vm1.0","band_vwap"),
      "vm1.5/opposite_band":("results/backtest/wf_vm1.5","opposite_band"),
      "vm1.3/opposite_band":("results/backtest/wf_vm1.3","opposite_band")}
S={k:series(*v) for k,v in CAND.items()}
n=len(S["vm1.5/opposite_band"][0]); dts=S["vm1.5/opposite_band"][2]
T0,NW=1100,10; STEP=(n-T0)//NW
print(f"{n} zile, {NW} ferestre a cate {STEP} zile OOS\n")
print(f"{'#':>2} {'perioada OOS':>23} | " + " | ".join(f"{k:>30}" for k in CAND))
print(f"{'':>2} {'':>23} | " + " | ".join(f"{'lev(IS)':>8}{'%/an OOS':>11}{'spart?':>9}" for _ in CAND))
acc={k:[] for k in CAND}; brk={k:0 for k in CAND}
for w in range(NW):
    e=T0+w*STEP; o=slice(e,min(e+STEP,n))
    line=f"{w+1:2d} {str(dts[e])[:10]}..{str(dts[min(e+STEP,n)-1])[:10]} | "
    parts=[]
    for k in CAND:
        R,M,_=S[k]
        lev=0.7*0.05/abs(M[:e].min())
        oos=R[o].mean()*252*lev
        worst=M[o].min()*lev
        sp = worst<=-0.05
        acc[k].append(oos); brk[k]+=int(sp)
        parts.append(f"{lev:7.2f}x{oos:10.1%}{'DA' if sp else 'nu':>9}")
    print(line+" | ".join(parts))
print("\n" + "="*96)
print(f"{'varianta':>22} {'%/an OOS mediu':>15} {'median':>9} {'pozitive':>9} {'min':>9} {'ferestre sparte':>16}")
for k in CAND:
    x=np.array(acc[k])
    print(f"{k:>22} {x.mean():15.1%} {np.median(x):9.1%} {int((x>0).sum()):6d}/{NW} "
          f"{x.min():9.1%} {brk[k]:12d}/{NW}")
print("\ncate ferestre castiga vm1.0/band_vwap fata de vm1.5/opposite_band:")
a=np.array(acc["vm1.0/band_vwap"]); b=np.array(acc["vm1.5/opposite_band"])
print(f"  {int((a>b).sum())}/{NW}   (diferenta medie {np.mean(a-b):+.1%}/an)")
