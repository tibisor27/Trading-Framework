"""
MetaTrader 5 terminal (FTMO) -> Bronze. Windows only.
MT5 cannot give "the bars of 10 September", only "the last N bars", so every run saves one file:
a snapshot of the last N bars, exactly as the terminal returned them (`time` still on the SERVER clock).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .. import catalog

NIGHTLY_BARS = 3 * 1440        # three days: a missed night is covered by the next one
FIRST_RUN_BARS = 30_000        # about four weeks: enough warm-up for the 14-day bands from day one


def fetch(inst: str, server_tz: str, count: int):
    """The array copy_rates_from_pos returns, unchanged."""
    from src.live.broker import MT5Broker      # imported here: MetaTrader5 does not exist on macOS

    b = MT5Broker()
    b.connect()
    try:
        # The offset the terminal shows now must match the rule staging will use to convert.
        measured = b.measure_server_offset(inst)
        rule = int(pd.Timestamp.now(tz=server_tz).utcoffset().total_seconds())
        if measured != rule:
            raise RuntimeError(f"server offset {measured/3600:+.1f}h, but {server_tz} says {rule/3600:+.1f}h")
        rates = b.mt5.copy_rates_from_pos(inst, b.mt5.TIMEFRAME_M1, 0, int(count))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no bars for {inst}: {b.mt5.last_error()}")
        return rates
    finally:
        b.mt5.shutdown()


def save(inst: str, rates, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    f = catalog.bronze_dir("mt5", inst) / f"{now:%Y-%m-%dT%H%M%SZ}.parquet"      # names sort by time
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rates).to_parquet(f, index=False)
    catalog.bronze_metadata("mt5", inst, f, rows=len(rates))
    return f


def run(inst: str, count: int | None = None, fetch=fetch) -> list[str]:
    spec = catalog.instrument(inst)
    first = not any(catalog.bronze_dir("mt5", inst).glob("*.parquet"))
    count = count or (FIRST_RUN_BARS if first else NIGHTLY_BARS)
    f = save(inst, fetch(inst, spec["server_tz"], count))
    print(f"extract mt5/{inst}: {count} bars -> {f.name}")
    return []
