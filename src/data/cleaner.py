import logging
import pandas as pd
from src.config import DataConfig

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:

    #validation
    missing_cols = [col for col in cfg.required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Critical error: missing columns in dataset: {missing_cols}")
        
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Critical error: index must be DatetimeIndex!")

    initial_rows = len(df)
    
    #drop duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Reconstrucția secvențialității temporale (pentru modele Recurente gen LSTM)
    if getattr(cfg, 'fill_missing_timestamps', False):
        df = df.resample('1min').asfreq()
        
        # Salvăm o mască a rândurilor care au fost inventate acum de resample (au NaN la preț)
        is_synthetic = df['close'].isna()
        
        # Forward Fill corect: O piață fără tranzacții NU are volatilitate!
        last_close = df['close'].ffill(limit=5)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].fillna(last_close)
        
        # Minutele lipsă umplute capătă volum 0 (piață plată)
        mask_valid = df['close'].notna()
        df.loc[mask_valid, 'volume'] = df.loc[mask_valid, 'volume'].fillna(0)
        
        # Adăugăm un flag pentru modelele de ML și indicatorii tehnici (ex: Garman-Klass)
        df['is_imputed'] = 0
        # Setăm 1 doar acolo unde rândul a fost sintetic ȘI a fost astupat cu succes de limita de 5
        df.loc[is_synthetic & mask_valid, 'is_imputed'] = 1
    
    #drop NaNs
    df = df.dropna(subset=cfg.required_columns)
    
    # drop zero volume
    if cfg.drop_zero_volume:
        df = df[df['volume'] > 0]
    
    n_nans = df[cfg.required_columns].isna().sum().sum()
    logger.info(f"NaN values in OHLCV before cleaning: {n_nans}")
    
    final_rows = len(df)    
    logger.info(f"Cleaned {initial_rows - final_rows} rows from {cfg.pair} {cfg.timeframe}")
    return df
