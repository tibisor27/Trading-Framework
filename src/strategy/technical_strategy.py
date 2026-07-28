import pandas as pd

def mean_reversion(df: pd.DataFrame, friday_cutoff_hour: int, 
    rsi_period: int, bb_period: int, bb_std: float, 
    rsi_oversold: int, rsi_overbought: int) -> pd.Series:
    
    
    rsi_col = f"rsi_{rsi_period}"
    
    bb_lower_col = f"bb_lower_{bb_period}_{bb_std}"
    bb_upper_col = f"bb_upper_{bb_period}_{bb_std}"
    
    # DATA CONTRACT VALIDATION - checking if the required columns exist in the dataframe
    required_cols = [rsi_col, bb_lower_col, bb_upper_col, "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Contract Violated! Strategia 'mean_reversion' are nevoie de coloanele: {missing_cols}. Ele nu există în Dataset! Asigură-te că le-ai adăugat în fișierul yaml pentru Feature Engineering.")

    is_friday_late = (
        (df.index.dayofweek == 4) & (df.index.hour >= friday_cutoff_hour)
    )

    long_setup = (
        (df[rsi_col] < rsi_oversold)
        & (df["close"] < df[bb_lower_col])
        & (~is_friday_late)
    )

    short_setup = (
        (df[rsi_col] > rsi_overbought)
        & (df["close"] > df[bb_upper_col])
        & (~is_friday_late)
    )

    # Creăm o serie plină cu 0 (No Trade)
    signals = pd.Series(0, index=df.index, dtype=int, name = "signal")
    
    # Aplicăm semnalele
    signals.loc[long_setup] = 1
    signals.loc[short_setup] = -1

    return signals


