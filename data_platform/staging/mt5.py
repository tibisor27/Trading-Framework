"""
MT5 staging. Bronze holds snapshots of "the last N bars", stamped on the server clock, overlapping
on purpose. For one UTC day: take the snapshots that cover it, convert, apply the two known fixes.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .. import catalog


def server_to_utc(time_s, server_tz: str) -> pd.DatetimeIndex:
    """
    Server epoch -> UTC with the time-zone rule, bar by bar.
    The bot subtracts one measured offset, which is right for the last few hours. A 3-day snapshot
    can cross the weekend Europe changes its clocks (25 Oct 2026), where one offset shifts Friday by an hour.
    """
    naive = pd.to_datetime(pd.Series(time_s).astype("int64"), unit="s")
    return pd.DatetimeIndex(naive.dt.tz_localize(server_tz).dt.tz_convert("UTC"), name="ts").as_unit("ns")


def parse(raw: pd.DataFrame, server_tz: str) -> pd.DataFrame:
    """Pure: one snapshot -> bars in UTC. Every row is kept; fixes come in clean()."""
    vol = "tick_volume" if "tick_volume" in raw.columns else "real_volume"
    return pd.DataFrame({"open": raw["open"].to_numpy(), "high": raw["high"].to_numpy(),
                         "low": raw["low"].to_numpy(), "close": raw["close"].to_numpy(),
                         "volume": raw[vol].to_numpy()}, index=server_to_utc(raw["time"], server_tz))


def clean(parsed: list[pd.DataFrame]) -> pd.DataFrame:
    """
    The two fixes MT5 data needs, both with a known cause (pass snapshots oldest first):
      * the last bar of each snapshot was still forming when it was taken, so it is dropped;
      * snapshots overlap on purpose, so a minute present in several is taken from the newest one.
    Only duplicates BETWEEN snapshots are resolved. A minute repeated INSIDE one snapshot has no known
    cause, so it is left in place for silver.check to stop the day.
    """
    df = pd.concat([p.iloc[:-1].assign(_snapshot=i) for i, p in enumerate(parsed)])
    newest = df.groupby(level=0)["_snapshot"].transform("max")
    return df[df["_snapshot"] == newest].drop(columns="_snapshot").sort_index(kind="stable")


def _snapshots(inst: str) -> list[Path]:
    return sorted(catalog.bronze_dir("mt5", inst).glob("*.parquet"))       # names sort by time


@lru_cache(maxsize=256)
def _span(path: Path, mtime: float, server_tz: str) -> tuple[date, date]:
    t = pd.read_parquet(path, columns=["time"])["time"]
    utc = server_to_utc([t.min(), t.max()], server_tz)
    return utc[0].date(), utc[1].date()


def _covers(inst: str) -> list[tuple[Path, date, date]]:
    tz = catalog.instrument(inst)["server_tz"]
    return [(p, *_span(p, p.stat().st_mtime, tz)) for p in _snapshots(inst)]


def available_days(inst: str) -> list[date]:
    out = set()
    for _, lo, hi in _covers(inst):
        d = lo
        while d <= hi:
            out.add(d)
            d += timedelta(days=1)
    return sorted(out)


def bronze_files(inst: str, day: date) -> list[Path]:
    return [p for p, lo, hi in _covers(inst) if lo <= day <= hi]


def load_bars(inst: str, day: date) -> pd.DataFrame:
    tz = catalog.instrument(inst)["server_tz"]
    df = clean([parse(pd.read_parquet(p), tz) for p in bronze_files(inst, day)])
    start = pd.Timestamp(day, tz="UTC")
    return df[(df.index >= start) & (df.index < start + pd.Timedelta(days=1))]
