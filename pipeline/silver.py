"""
Silver storage: one parquet per source, symbol and UTC day, all 24 hours:

    data/silver/<source>/<symbol>/<YYYY-MM-DD>.parquet

Writing, reading back, and parity with the file the edge was measured on.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .standardize import COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"


def path(source: str, symbol: str, day: date) -> Path:
    return SILVER / source / symbol / f"{day.isoformat()}.parquet"


def write(source: str, symbol: str, day: date, df: pd.DataFrame, lineage: dict) -> Path:
    """
    Save one day that already passed the checks. `lineage` (source, Bronze files, when they were
    ingested) is stored inside the parquet file as metadata, not as columns, so readers that expect
    only OHLCV are unaffected. See it with pd.read_parquet(path).attrs
    """
    p = path(source, symbol, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.attrs = lineage
    df.to_parquet(p)
    return p


def read(source: str, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    files = [p for p in sorted((SILVER / source / symbol).glob("*.parquet"))
             if (not start or date.fromisoformat(p.stem) >= start) and (not end or date.fromisoformat(p.stem) <= end)]
    if not files:
        raise FileNotFoundError(f"nothing in {SILVER / source / symbol}")
    df = pd.concat(pd.read_parquet(p) for p in files)
    df.index = df.index.astype("datetime64[ns, UTC]")
    return df



#functia parity compara rezultatele din nou pipeline cu rezultate de dinainte de pipleine pentru a vedea daca exista 
# nepotriviri sau buguri silentioase
def parity(reference: Path, source: str = "dukascopy", symbol: str = "US100") -> tuple[bool, str]:
    """Silver vs the file the edge was measured on, on the days both have. Must be identical."""
    s = read(source, symbol)
    ref = pd.read_parquet(reference)
    ref.index = pd.DatetimeIndex(ref.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    days = s.index.normalize().unique().intersection(ref.index.normalize().unique())
    s, ref = s[s.index.normalize().isin(days)], ref[ref.index.normalize().isin(days)]
    same_index = s.index.equals(ref.index)
    diff = float(np.abs(s[COLUMNS].to_numpy() - ref[COLUMNS].to_numpy(dtype="float64")).max()) if same_index else float("nan")
    ok = same_index and diff == 0.0
    msg = (f"{'PASS' if ok else 'FAIL'}: {len(days)} common days, {len(s)} silver bars, {len(ref)} reference bars, "
           f"max |diff| = {diff:g}")
    return ok, msg
