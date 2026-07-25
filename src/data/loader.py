from pathlib import Path
import logging
from src.data.config import DataConfig
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_raw_data(cfg: DataConfig) -> pd.DataFrame:

    raw_path = PROJECT_ROOT / cfg.raw_file
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    logger.info(f"Loading {cfg.pair} {cfg.timeframe} data from {raw_path}")

    df = pd.read_csv(raw_path, index_col=cfg.timestamp_col, parse_dates=True)

    #setting the timezone from 'UTC' to 'America/New_York' 
    #it handles the summer/winter time automatically
    df.index = df.index.tz_localize(cfg.tz_source).tz_convert(cfg.tz_target)

    df.sort_index(inplace=True)

    n_rows = len(df)
    date_min = df.index.min()
    date_max = df.index.max()

    logger.info(f"Loaded {n_rows:,} rows from {date_min} to {date_max}")

    return df

    
    