"""Clasament dupa criteriul FIRMEI, nu dupa Sharpe.

Plafonul de levier vine din regula zilnica de 5% aplicata pe EXCURSIA INTRADAY.
  levier_max   = 5% / |cea mai proasta excursie intraday la levier 1|
  randament    = CAGR@1x x levier   (la 70% din plafon, ca sa ai marja)
Castiga varianta cu cel mai mult drift pe unitate de coada, nu cu cel mai bun Sharpe.
"""
from __future__ import annotations
import numpy as np, pandas as pd, itertools, os

bars=pd.read_parquet("data/raw/USATECHIDXUSD_1m.parquet",columns=["high","low"]).sort_index()
if bars.index.tz is None: bars.index=bars.index.tz_localize("UTC")
IXB,HI,LO=bars.index,bars["high"].to_numpy(),bars["low"].to_numpy()

def excursion(dirp, ex):
    tf=f"{dirp}/trades_{ex}.csv"; df=f"{dirp}/daily_{ex}.csv"
    if not os.path.exists(tf): return None
    t=pd.read_csv(tf)
    if len(t)==0: return None
    ti=pd.to_datetime(t["t_in"],utc=True).to_numpy(); to=pd.to_datetime(t["t_out"],utc=True).to_numpy()
    a=IXB.searchsorted(ti,"left"); b=IXB.searchsorted(to,"right")
    side=t["side"].to_numpy(float); pin=t["price_in"].to_numpy(float)
    lev=t["leverage"].to_numpy(float); cst=(t["ret_net"]-t["ret_gross"]).to_numpy(float)
    mae=np.zeros(len(t))
    for i in range(len(t)):
        s,e=a[i],b[i]
        if e<=s: continue
        g=(LO[s:e].min()/pin[i]-1.0) if side[i]>0 else (1.0-HI[s:e].max()/pin[i])
        mae[i]=(g+cst[i])*lev[i]
    t["mae"]=mae
    rows=[]
    for day,g in t.groupby("date",sort=True):
        rz=lo=0.0
        for r in g.sort_values("t_in").itertuples(): lo=min(lo,rz+r.mae); rz+=r.pnl
        rows.append((day,rz,lo))
    z=pd.DataFrame(rows,columns=["date","r","m"]); z["date"]=pd.to_datetime(z["date"]).dt.date
    cal=pd.read_csv(df); cal["date"]=pd.to_datetime(cal["date"]).dt.date
    f=cal.merge(z,on="date",how="left").fillna({"r":0.0,"m":0.0})
    return f["r"].to_numpy(), f["m"].to_numpy(), f["date"].to_numpy()

rows=[]
for tag,root in (("30min","results/backtest/wf_vm"),("1min","results/backtest/tt_vm")):
    for vm in ("1.0","1.3","1.5"):
        for ex in ("opposite_band","band_vwap","close"):
            out=excursion(f"{root}{vm}",ex)
            if out is None: continue
            R,M,dts=out; sig=R.std(ddof=1)
            if sig<=0 or abs(M.min())<1e-9: continue
            cagr=R.mean()*252; sh=cagr/(sig*np.sqrt(252))
            lmax=0.05/abs(M.min()); lsafe=0.7*lmax
            rows.append(dict(stop=tag,vm=vm,exit=ex,sharpe=sh,cagr=cagr,
                             vol=sig*np.sqrt(252),skew=pd.Series(R).skew(),
                             worst=M.min(),lmax=lmax,volcont=lsafe*sig*100,
                             randcont=lsafe*cagr))
            np.save(f"/tmp/ser_{tag}_{vm}_{ex}.npy",np.vstack([R,M]))
d=pd.DataFrame(rows).sort_values("randcont",ascending=False)
pd.set_option("display.width",200)
print("clasament dupa RANDAMENT PE CONT la 70% din plafonul impus de regula zilnica:\n")
print(f"{'stop':>6} {'vm':>4} {'iesire':>14} {'Sharpe':>7} {'CAGR@1x':>8} {'skew':>6} "
      f"{'worst intra':>12} {'lev max':>8} {'vol cont':>9} {'%/an cont':>10}")
for r in d.itertuples():
    print(f"{r.stop:>6} {r.vm:>4} {getattr(r,'exit'):>14} {r.sharpe:7.2f} {r.cagr:7.1%} "
          f"{r.skew:6.2f} {r.worst:11.2%} {r.lmax:7.2f}x {r.volcont:8.2f}% {r.randcont:9.1%}")
