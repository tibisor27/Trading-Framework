"""
Quality checks: rules every Silver day must satisfy, whatever the source. Common to every source.

A check only reports; it never fixes. Fixing a problem silently hides it, and the next time the
same problem means something else (a wrong clock, a broken source) nobody finds out.
A day with any problem is not written to Silver.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def check(df: pd.DataFrame, day: date) -> list[str]:
    """What is wrong with one standardized day of bars. Empty list = OK."""
    out = []
    if df.index.has_duplicates:
        out.append(f"{int(df.index.duplicated().sum())} duplicated minutes")
    if not df.index.is_monotonic_increasing:
        out.append("minutes out of order")
    if (df.index.normalize() != pd.Timestamp(day, tz="UTC")).any():
        out.append("bars outside the UTC day")
    px = df[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(px).all() or (px <= 0).any():
        out.append("missing, infinite or non-positive prices")
    elif (df["high"] < df[["open", "close"]].max(axis=1)).any() or (df["low"] > df[["open", "close"]].min(axis=1)).any():
        out.append("high/low outside open/close")
    return out
