"""
.bi5 bytes -> bars. The file is LZMA-compressed records of 24 bytes, big-endian:
    int32 seconds-from-midnight, int32 open, int32 CLOSE, int32 low, int32 high, float32 volume
Times are already UTC.
"""
from __future__ import annotations

import lzma
from datetime import date

import numpy as np
import pandas as pd

RECORD = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])


def parse(raw: bytes, day: date, decimals: int) -> pd.DataFrame:
    data = lzma.decompress(raw) if raw else b""
    a = np.frombuffer(data[: len(data) // RECORD.itemsize * RECORD.itemsize], dtype=RECORD)
    scale = 10.0 ** decimals
    idx = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(a["t"].astype("int64"), unit="s")
    return pd.DataFrame({"open": a["o"] / scale, "high": a["h"] / scale, "low": a["l"] / scale,
                         "close": a["c"] / scale, "volume": a["v"].astype("float64")},
                        index=pd.DatetimeIndex(idx, name="ts"))
