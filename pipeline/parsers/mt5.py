"""MT5 extractions -> bars in UTC (parse), then the two MT5-specific fixes (clean)."""
from __future__ import annotations

import logging
import pandas as pd

log = logging.getLogger('data_pipeline.parser.mt5')

def server_to_utc(time_s, server_tz: str) -> pd.DatetimeIndex:
    """
    Server epoch -> UTC with the time-zone rule, bar by bar.
    The bot subtracts one measured offset, which is right for the last few hours. A 3-day extraction
    can cross the weekend Europe changes its clocks (25 Oct 2026), where one offset shifts Friday by an hour.
    """
    naive = pd.to_datetime(pd.Series(time_s).astype("int64"), unit="s")
    return pd.DatetimeIndex(naive.dt.tz_localize(server_tz).dt.tz_convert("UTC"), name="ts").as_unit("ns")


def parse(raw: pd.DataFrame, server_tz: str) -> pd.DataFrame:
    """One extraction -> bars in UTC. Every row is kept; cleaning is the next step."""
    vol = "tick_volume" if "tick_volume" in raw.columns else "real_volume"
    return pd.DataFrame({"open": raw["open"].to_numpy(), "high": raw["high"].to_numpy(),
                         "low": raw["low"].to_numpy(), "close": raw["close"].to_numpy(),
                         "volume": raw[vol].to_numpy()}, index=server_to_utc(raw["time"], server_tz))


def clean(parsed: list[pd.DataFrame]) -> pd.DataFrame:
    """
    The two fixes MT5 data needs, both with a known cause:
      * the last bar of each extraction was still forming when it was taken, so it is dropped;
      * extractions overlap on purpose (3 days each night), so a minute present in several
        extractions is taken from the newest one (pass them oldest first).

    Only duplicates BETWEEN extractions are resolved here. A minute repeated INSIDE one extraction
    has no known cause, so it is left in place for quality.check to stop the day.
    """
    df = pd.concat([p.iloc[:-1].assign(_extraction=i) for i, p in enumerate(parsed)])
    newest = df.groupby(level=0)["_extraction"].transform("max")
    return df[df["_extraction"] == newest].drop(columns="_extraction").sort_index(kind="stable")
