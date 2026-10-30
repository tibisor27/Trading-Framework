"""
Dukascopy staging. A .bi5 file is LZMA-compressed records of 24 bytes, big-endian:
    int32 seconds-from-midnight, int32 open, int32 CLOSE, int32 low, int32 high, float32 volume
Times are already UTC. No known fixes are needed: a day's file is complete and never revised.
!! TO BE CONSIDERED!! :
    minutes without ticks are filled by Dukascopy itselfL 
    open = high = close = low = the last close, volume 0.
    Nothing here fills or drops rows; Gold decides what to do with them (volume > 0, session cut).
"""
from __future__ import annotations

import lzma
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import catalog

RECORD = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])


def parse(raw: bytes, day: date, decimals: int) -> pd.DataFrame:
    """Pure: bytes of one .bi5 file -> bars for that UTC day."""
    data = lzma.decompress(raw) if raw else b""
    a = np.frombuffer(data[: len(data) // RECORD.itemsize * RECORD.itemsize], dtype=RECORD)
    scale = 10.0 ** decimals
    idx = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(a["t"].astype("int64"), unit="s")
    return pd.DataFrame({"open": a["o"] / scale, "high": a["h"] / scale, "low": a["l"] / scale,
                         "close": a["c"] / scale, "volume": a["v"].astype("float64")},
                        index=pd.DatetimeIndex(idx, name="ts"))


#returns a list of bronze layer files for an instrument sorted by day
# p.stem - extract only the file name ( without extension)
def available_days(inst: str) -> list[date]:
    return sorted(date.fromisoformat(p.stem) for p in catalog.bronze_dir("dukascopy", inst).glob("*.bi5"))

#returns the file(s)path from bronze layer for an instrument and a day(1 single file per day from dukascopy broker)
def bronze_files(inst: str, day: date) -> list[Path]:
    return [catalog.bronze_dir("dukascopy", inst) / f"{day.isoformat()}.bi5"]


def load_bars(inst: str, day: date) -> pd.DataFrame:
    return parse(bronze_files(inst, day)[0].read_bytes(), day, catalog.instrument(inst)["decimals"])
