"""
Staging: turn what one source left in Bronze into bars with a UTC index. One module per source,
like dbt's stg_<source> models. Each module answers the same three questions:

    days(inst)          -> list[date]     which UTC days Bronze has something for
    inputs(inst, day)   -> list[Path]     the Bronze files that day is built from
    bars(inst, day)     -> DataFrame      parse + that source's known fixes, still in the source's shape

Because every source answers the same questions, silver.py has one loop for all of them.
"""
from . import dukascopy, mt5

STAGING = {"dukascopy": dukascopy, "mt5": mt5}


def for_source(source: str):
    return STAGING[source]
