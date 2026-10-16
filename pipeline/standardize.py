"""
Standardize: bring parsed bars from any source to the one Silver schema. Common to every source.

    index  "ts", datetime64[ns, UTC]
    columns open, high, low, close, volume, float64, in that order

It changes types and names only, never values or rows. That is the same format
src/data/intraday.load_intraday returns, so research code reads Silver as it is.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLUMNS].astype("float64")
    # pandas 3 creates microsecond timestamps; the reference data is nanoseconds and they do not compare equal
    out.index = pd.DatetimeIndex(df.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    out.index.name = "ts"
    return out
