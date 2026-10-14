"""
Turn an MT5 export into a parquet the framework can read - and prove the timezone is right.

MT5 stamps bars in SERVER time. FTMO's servers run GMT+2 with EU daylight saving, which is the
EET/EEST zone (Europe/Bucharest and friends). EU and US daylight saving switch on DIFFERENT dates,
so a naive "add two hours" is wrong for several weeks every year - and those weeks would silently
shift the whole session by an hour, which shifts every band, every decision and every result.

So this does not trust the conversion: after converting, it CHECKS that the regular session really
lands on 09:30-16:00 New York, and refuses to write the file if it does not.

    python scripts/import_mt5_csv.py --csv ~/Downloads/US100_M1.csv --out data/raw/FTMO_US100_1m.parquet
    python scripts/import_mt5_csv.py --csv US500_M1.csv --server-tz Europe/Bucharest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NY = "America/New_York"


def load_export(csv: Path, server_tz: str) -> tuple[pd.DataFrame, int | None]:
    d = pd.read_csv(csv)
    d.columns = [c.strip().lower() for c in d.columns]
    if "server_time" not in d.columns:
        raise SystemExit(f"coloana 'server_time' lipseste. Coloane gasite: {list(d.columns)}")

    naive = pd.to_datetime(d["server_time"], format="mixed", dayfirst=False)
    idx = (naive.dt.tz_localize(server_tz, ambiguous="NaT", nonexistent="NaT")
           .dt.tz_convert("UTC"))
    bad = int(idx.isna().sum())
    if bad:
        print(f"  {bad} bare in ora ambigua/inexistenta de DST - le scot")

    vol = d["real_volume"] if "real_volume" in d.columns else pd.Series(0.0, index=d.index)
    if float(pd.to_numeric(vol, errors="coerce").fillna(0).sum()) <= 0 and "tick_volume" in d.columns:
        vol = d["tick_volume"]

    # .to_numpy() is not cosmetic: handing pandas Series that still carry the original RangeIndex
    # together with a DatetimeIndex makes it REINDEX them, and every value silently becomes NaN.
    out = pd.DataFrame({
        "open": pd.to_numeric(d["open"], errors="coerce").to_numpy(),
        "high": pd.to_numeric(d["high"], errors="coerce").to_numpy(),
        "low": pd.to_numeric(d["low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(d["close"], errors="coerce").to_numpy(),
        "volume": pd.to_numeric(vol, errors="coerce").fillna(0.0).to_numpy(),
    }, index=pd.DatetimeIndex(idx.to_numpy(), tz="UTC", name="ts"))

    offset = None
    if "gmt_offset_sec_at_export" in d.columns:
        vals = pd.to_numeric(d["gmt_offset_sec_at_export"], errors="coerce").dropna()
        if len(vals):
            offset = int(vals.iloc[-1])

    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, offset


def validate(df: pd.DataFrame, server_tz: str, offset: int | None) -> list[str]:
    """
    Catch a wrong timezone, non-circularly.

    Checking that the 09:30-16:00 slice starts at 09:30 proves nothing - the slice is defined that
    way. For an index CFD that trades nearly around the clock, the honest test is where the ACTIVITY
    is: the cash open at 09:30 New York is by far the loudest minute of the day. On this instrument
    its median high-low range is 30.5 points against 9.5 for the session as a whole, so if the
    busiest minute is not sitting on 09:30, the conversion is wrong.
    """
    problems: list[str] = []
    loc = df.index.tz_convert(NY)
    mod = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
    wd = np.asarray(loc.dayofweek) < 5
    if wd.sum() == 0:
        return ["nicio bara in zilele lucratoare"]

    rng = pd.Series((df["high"] - df["low"]).to_numpy()[wd]).groupby(mod[wd]).median()
    if rng.empty or float(rng.max()) <= 0:
        problems.append("toate barele au range zero - export gol sau corupt")
    else:
        peak = int(rng.idxmax())
        if not (565 <= peak <= 580):
            problems.append(
                f"minutul cel mai agitat al zilei cade la {peak//60:02d}:{peak%60:02d} New York, "
                f"nu la ~09:30 (deschiderea cash). Fusul '{server_tz}' e aproape sigur gresit")
        in_s = [m for m in rng.index if 570 <= m < 960]
        out_s = [m for m in rng.index if not (570 <= m < 960)]
        if in_s and out_s:
            ratio = float(rng.loc[in_s].median()) / max(1e-12, float(rng.loc[out_s].median()))
            if ratio < 1.5:
                problems.append(
                    f"volatilitatea din sesiune e doar {ratio:.1f}x fata de restul zilei "
                    f"(asteptat ~2.7x) - fus gresit sau instrument gresit")

    rth = (mod >= 570) & (mod < 960) & wd
    if rth.sum() == 0:
        problems.append("ZERO bare intre 09:30 si 16:00 New York")
    else:
        rloc = df[rth].index.tz_convert(NY)
        per_day = pd.Series(1, index=rloc).groupby(rloc.date).sum()
        med = float(per_day.median())
        if not (380 <= med <= 391):
            problems.append(f"mediana barelor pe sesiune este {med:.0f}, nu ~390 - export incomplet")

    if offset is not None:
        expect = pd.Timestamp.now(tz=server_tz).utcoffset().total_seconds()
        if abs(expect - offset) > 60:
            problems.append(f"offsetul raportat de terminal ({offset/3600:.1f}h) nu se potriveste cu "
                            f"'{server_tz}' ({expect/3600:.1f}h)")

    bad = int(((df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) |
               (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)).sum())
    if bad:
        problems.append(f"{bad} bare cu OHLC imposibil")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--server-tz", default="Europe/Bucharest",
                    help="fusul serverului MT5; FTMO = GMT+2 cu DST european")
    ap.add_argument("--force", action="store_true", help="scrie chiar daca validarea pica")
    args = ap.parse_args()

    csv = Path(args.csv).expanduser()
    if not csv.exists():
        raise SystemExit(f"nu gasesc {csv}")

    df, offset = load_export(csv, args.server_tz)
    print(f"{csv.name}: {len(df):,} bare, {df.index[0]} -> {df.index[-1]}")
    if offset is not None:
        print(f"  offset server-GMT raportat de terminal: {offset/3600:.1f} ore")

    problems = validate(df, args.server_tz, offset)
    loc = df.index.tz_convert(NY)
    mins = loc.hour * 60 + loc.minute
    rth = (mins >= 570) & (mins < 960) & (loc.dayofweek < 5)
    rloc = df[rth].index.tz_convert(NY)
    print(f"  bare in sesiune: {int(rth.sum()):,} pe {rloc.normalize().nunique()} zile")

    if problems:
        print("\n*** VALIDAREA A PICAT ***")
        for p in problems:
            print("  -", p)
        print("\nIncearca alt fus: --server-tz Europe/Bucharest | Europe/Athens | Etc/GMT-2 | Etc/GMT-3")
        if not args.force:
            sys.exit(1)
        print("\n--force: scriu oricum, dar rezultatele nu sunt de incredere.")
    else:
        print("  validare OK: varful de activitate cade pe deschiderea cash de la 09:30 New York")

    out = Path(args.out) if args.out else ROOT / "data" / "raw" / f"{csv.stem.replace('_M1','')}_1m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nscris in {out}")


if __name__ == "__main__":
    main()
