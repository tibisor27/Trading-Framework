from typing import List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    raw_file: str
    pair: str
    interim_file: str
    features_file: str
    processed_file: str
    timeframe: str
    timestamp_col: str
    tz_source: str
    tz_target: str
    required_columns: List[str]
    drop_zero_volume: bool
    fill_missing_timestamps: bool

    
