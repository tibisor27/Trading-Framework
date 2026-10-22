"""
The Silver contract, executable. One definition, used on both sides of the boundary:

    producer   silver.py validates every day before writing it      (nothing broken gets in)
    consumer   Gold calls SILVER_BARS.validate(df) after reading     (trust, but verify, cheaply)

Hand-made on purpose, in the shape of a pandera DataFrameSchema, so the move to pandera later is a
translation, not a redesign. What code cannot check (what a value MEANS: start-of-minute timestamps,
volume units per source, never mixing sources) stays in CONTRACT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

# A row check returns a boolean Series, True where the row is BAD.
RowCheck = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str = "float64"
    checks: tuple[tuple[str, RowCheck], ...] = ()


@dataclass(frozen=True)
class Index:
    name: str
    dtype: str
    unique: bool = True
    increasing: bool = True
    on_minute: bool = True


@dataclass(frozen=True)
class Schema:
    name: str
    version: str
    index: Index
    columns: tuple[Column, ...]
    checks: tuple[tuple[str, RowCheck], ...] = field(default=())     # checks across columns

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def validate(self, df: pd.DataFrame, day: date | None = None) -> list[str]:
        """
        Every violation, each with how many rows break it. Empty list = the frame honours the contract.
        With `day`, also checks that every bar belongs to that UTC day (one Silver file = one day).
        Reports everything at once instead of stopping at the first problem (pandera: lazy=True).
        """
        out: list[str] = []
        # --- structure: if this is wrong, the row checks below would be meaningless
        if list(df.columns) != self.column_names:
            return [f"columns: expected {self.column_names}, got {list(df.columns)}"]
        wrong = [f"{c.name}={df[c.name].dtype}" for c in self.columns if str(df[c.name].dtype) != c.dtype]
        if wrong:
            return [f"dtypes: expected {self.columns[0].dtype}, got {', '.join(wrong)}"]
        ix = df.index
        if ix.name != self.index.name or str(ix.dtype) != self.index.dtype:
            return [f"index: expected {self.index.name!r} {self.index.dtype}, got {ix.name!r} {ix.dtype}"]
        if df.empty:
            return []

        # --- index
        if self.index.unique and ix.has_duplicates:
            out.append(f"index unique: {int(ix.duplicated().sum())} duplicated minutes")
        if self.index.increasing and not ix.is_monotonic_increasing:
            out.append("index increasing: minutes out of order")
        if self.index.on_minute and (ix.asi8 % 60_000_000_000 != 0).any():
            out.append(f"index on_minute: {int((ix.asi8 % 60_000_000_000 != 0).sum())} timestamps not on a whole minute")
        if day is not None:
            bad = int((ix.normalize() != pd.Timestamp(day, tz="UTC")).sum())
            if bad:
                out.append(f"index in_day: {bad} bars outside {day} UTC")

        # --- values
        for c in self.columns:
            for name, fn in c.checks:
                n = int(fn(df[[c.name]].rename(columns={c.name: "x"})).sum())
                if n:
                    out.append(f"{c.name} {name}: {n} rows")
        for name, fn in self.checks:
            n = int(fn(df).sum())
            if n:
                out.append(f"{name}: {n} rows")
        return out

    def describe(self) -> str:
        """The contract as text, generated from the code, so the two can never disagree."""
        lines = [f"{self.name} v{self.version}",
                 f"  index   {self.index.name} · {self.index.dtype}"
                 + "".join(f" · {k}" for k in ("unique", "increasing", "on_minute") if getattr(self.index, k))]
        for c in self.columns:
            lines.append(f"  {c.name:<7} {c.dtype}" + "".join(f" · {n}" for n, _ in c.checks))
        lines += [f"  check   {n}" for n, _ in self.checks]
        return "\n".join(lines)


def _bad_price(d: pd.DataFrame) -> pd.Series:
    return ~np.isfinite(d["x"]) | (d["x"] <= 0)


def _bad_volume(d: pd.DataFrame) -> pd.Series:
    return ~np.isfinite(d["x"]) | (d["x"] < 0)


SILVER_BARS = Schema(
    name="silver.bars",
    version="1",
    index=Index(name="ts", dtype="datetime64[ns, UTC]"),
    columns=(
        Column("open", checks=(("finite and > 0", _bad_price),)),
        Column("high", checks=(("finite and > 0", _bad_price),)),
        Column("low", checks=(("finite and > 0", _bad_price),)),
        Column("close", checks=(("finite and > 0", _bad_price),)),
        Column("volume", checks=(("finite and >= 0", _bad_volume),)),
    ),
    checks=(
        ("high >= open, close", lambda d: d["high"] < d[["open", "close"]].max(axis=1)),
        ("low <= open, close", lambda d: d["low"] > d[["open", "close"]].min(axis=1)),
    ),
)
