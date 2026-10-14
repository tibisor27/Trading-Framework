"""Walk-forward pe nivelul stopului, cu obiectivul REAL: E[ani pana la cont finantat].

Pentru fiecare fereastra:
  1. pe TRAIN: pentru fiecare stop S, plafonul de volatilitate = 5% * sigma / |worst_intra|,
     rulam la 70% din plafon si masuram E[ani -> finantat]. Alegem S* si vol*.
  2. pe TEST (nevazut): aceleasi (S*, vol*) -> P(ambele), E[ani], si testul curat
     fara simulare: a existat in TEST o zi care ar fi spart limita de 5%?
Nimic din TEST nu atinge alegerea.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "results/backtest/noise_area_USATECHIDXUSD_1m"
BARSF = sys.argv[2] if len(sys.argv) > 2 else "data/raw/USATECHIDXUSD_1m.parquet"
STOPS = [None, 0.0035, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
MARGIN, HC = 0.70, 0.95/1.35
DAILY, MAXL, BLOCK, NSIM = 0.05, 0.10, 20, 2500
rng = np.random.default_rng(1234)

bars = pd.read_parquet(BARSF, columns=["high","low"]).sort_index()
if bars.index.tz is None: bars.index = bars.index.tz_localize("UTC")
ixb, hia, loa = bars.index, bars["high"].to_numpy(), bars["low"].to_numpy()
tr = pd.read_csv(f"{D}/trades_opposite_band.csv")
tr["t_in"]=pd.to_datetime(tr["t_in"],utc=True); tr["t_out"]=pd.to_datetime(tr["t_out"],utc=True)
a=ixb.searchsorted(tr["t_in"].to_numpy(),"left"); b=ixb.searchsorted(tr["t_out"].to_numpy(),"right")
mae=np.zeros(len(tr))
for i in range(len(tr)):
    wl,wh=loa[a[i]:b[i]],hia[a[i]:b[i]]
    if len(wl)==0: continue
    p,s=tr.price_in.iloc[i],tr.side.iloc[i]
    mae[i]=((wl.min()/p-1.0) if s>0 else (1.0-wh.max()/p))+(tr.ret_net.iloc[i]-tr.ret_gross.iloc[i])
tr["mae"]=mae
cal=pd.read_csv(f"{D}/daily_opposite_band.csv"); cal["date"]=pd.to_datetime(cal["date"]).dt.date

def build(S):
    t=tr.copy()
    if S is None: t["m2"]=t["mae"]
    else:
        h=t["mae"]<=-S; t.loc[h,"pnl"]=-S*t.loc[h,"leverage"]; t["m2"]=np.maximum(t["mae"],-S)
    rows=[]
    for day,g in t.groupby("date",sort=True):
        rz=lo=0.0
        for r in g.sort_values("t_in").itertuples():
            lo=min(lo,rz+r.m2); rz+=r.pnl
        rows.append((day,rz,lo))
    z=pd.DataFrame(rows,columns=["date","r","m"]); z["date"]=pd.to_datetime(z["date"]).dt.date
    f=cal.merge(z,on="date",how="left").fillna({"r":0.0,"m":0.0})
    return f["r"].to_numpy(), f["m"].to_numpy()
SER={s:build(s) for s in STOPS}
NDAYS=len(SER[None][0])

def evaluate(R,M,vol,nsim=NSIM):
    sig=R.std(ddof=1)
    if sig<=0: return 0.0,np.nan
    k=(vol/100.0)/sig; sh=R.mean()/sig*np.sqrt(252)*HC
    base=(R-R.mean())+sh*sig/np.sqrt(252)
    r=base*k; lo=(base+(M-R))*k; n=len(r)
    md=int(np.clip(600.0/vol,250,1200))
    out=[]
    for tg in (0.10,0.05):
        nb=int(np.ceil(md/BLOCK))
        st=rng.integers(0,max(1,n-BLOCK+1),size=(nsim,nb))
        ix=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(nsim,-1)[:,:md],0,n-1)
        x,y=r[ix],lo[ix]; eq=np.cumsum(x,axis=1)
        f=((eq-x+y)<=-MAXL)|(y<=-DAILY); g=eq>=tg; big=md+1
        tw=np.where(g.any(1),g.argmax(1),big); tf=np.where(f.any(1),f.argmax(1),big)
        w=tw<tf
        out.append((w.mean(), tw[w].mean() if w.any() else md,
                    np.clip(tf[~w],0,md).mean() if (~w).any() else md))
    p=out[0][0]*out[1][0]
    dur=out[0][0]*(out[0][1]+out[1][0]*out[1][1]+(1-out[1][0])*out[1][2])+(1-out[0][0])*out[0][2]
    return p, (dur/max(p,1e-9))/252.0

T0, NW = 1100, 10
STEP=(NDAYS-T0)//NW
print(f"{NDAYS} zile, antrenare initiala {T0}, {NW} ferestre de cate {STEP} zile OOS\n")
print(f"{'#':>2} {'train->':>8} {'S*':>7} {'vol*':>7} | {'IS ani':>7} | "
      f"{'OOS P(amb)':>10} {'OOS ani':>8} | {'zi OOS cea mai rea':>19} {'spart?':>7}")
res=[]
for w in range(NW):
    e=T0+w*STEP; o=slice(e,min(e+STEP,NDAYS))
    best=(None,1e9,None)
    for S in STOPS:
        R,M=SER[S]; Rt,Mt=R[:e],M[:e]
        if Rt.std(ddof=1)<=0 or Mt.min()>=0: continue
        ceil_=0.05*Rt.std(ddof=1)/abs(Mt.min())*100
        vol=MARGIN*ceil_
        p,yr=evaluate(Rt,Mt,vol)
        if yr<best[1]: best=(S,yr,vol)
    S,isyr,vol=best
    R,M=SER[S]; Ro,Mo=R[o],M[o]
    p,yr=evaluate(Ro,Mo,vol,nsim=2000) if Ro.std(ddof=1)>0 else (np.nan,np.nan)
    k=(vol/100.0)/R[:e].std(ddof=1)
    worst=Mo.min()*k
    res.append((S,vol,isyr,p,yr,worst))
    lab="fara" if S is None else f"{S:.2%}"
    print(f"{w+1:2d} {e:8d} {lab:>7} {vol:6.2f}% | {isyr:7.2f} | {p:10.3f} {yr:8.2f} | "
          f"{worst:18.2%} {'DA' if worst<=-DAILY else 'nu':>7}")

r=pd.DataFrame(res,columns=["S","vol","is_yr","oos_p","oos_yr","oos_worst"])
print(f"\nstopuri alese: {r['S'].value_counts(dropna=False).to_dict()}")
print(f"vol* median {r['vol'].median():.2f}%   OOS P(amb) median {r['oos_p'].median():.3f}   "
      f"OOS ani median {r['oos_yr'].median():.2f}")
print(f"ferestre cu OOS mai bun decat 2.69 ani (planul prudent): "
      f"{int((r['oos_yr']<2.69).sum())}/{NW}")
print(f"ferestre in care limita zilnica de 5% a fost sparta OOS: "
      f"{int((r['oos_worst']<=-DAILY).sum())}/{NW}")
