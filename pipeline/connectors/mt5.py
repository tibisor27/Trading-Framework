"""
MetaTrader 5 terminal (FTMO). Windows only. MT5 cannot give "the bars of 10 September", only
"the last N bars", so each call returns the last 3 days.
"""
from __future__ import annotations

import pandas as pd


def fetch(inst: str, server_tz: str, count: int = 3 * 1440):
    """The array copy_rates_from_pos returns, unchanged: `time` is still the SERVER clock."""
    from src.live.broker import MT5Broker      # imported here: MetaTrader5 does not exist on macOS

    b = MT5Broker()
    b.connect()
    try:
        # masoara diferenta dintre ora serverului si UTC
        # daca serverul e in Europe/London, atunci diferenta este de 0 secunde (ora serverului este egala cu UTC)
        # cand le compar cu Europe/Bucuresti, atunci diferenta este de 3 ore (ora serverului este cu 3 ore inaintea UTC)
        # daca exista diferenta intre ora pe care cred eu ca o are serverul si ora pe care o are serverul, atunci 
        # dau CRASH imediat(am confgurat gresit server_tz)
        #measured = difference in seconds between the server hour and UTC
        measured = b.measure_server_offset(inst)

        #diferente in secunde dintre ora UTC si ora serverului
        rule = int(pd.Timestamp.now(tz=server_tz).utcoffset().total_seconds())
        if measured != rule:
            raise RuntimeError(f"server offset {measured/3600:+.1f}h, but {server_tz} says {rule/3600:+.1f}h")
        rates = b.mt5.copy_rates_from_pos(inst, b.mt5.TIMEFRAME_M1, 0, int(count))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no bars for {inst}: {b.mt5.last_error()}")
        return rates
    finally:
        b.mt5.shutdown()
