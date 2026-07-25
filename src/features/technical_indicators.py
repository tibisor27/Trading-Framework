import pandas as pd
import numpy as np


def calc_sma(df: pd.DataFrame, periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    for period in periods:
        new_cols[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return new_cols


def calc_ema(df: pd.DataFrame, periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    for period in periods:
        new_cols[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return new_cols


def calc_macd(df: pd.DataFrame, fasts: list, slows: list, signals: list) -> dict[str, pd.Series]:
    new_cols = {}
    for fast in fasts:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        
        for slow in slows:
            if fast >= slow:
                continue  # Fast EMA period must be smaller than Slow EMA period
                
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            
            for signal in signals:
                macd_signal = macd.ewm(span=signal, adjust=False).mean()
                macd_hist = macd - macd_signal
                
                new_cols[f'macd_{fast}_{slow}_{signal}'] = macd
                new_cols[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
                new_cols[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
                
    return new_cols


def calc_rsi(df: pd.DataFrame, periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    for period in periods:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        new_cols[f'rsi_{period}'] = rsi
        
    return new_cols


def calc_stochastic(df: pd.DataFrame, k_periods: list, d_periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    for k_period in k_periods:

        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)

        for d_period in d_periods:

            stoch_d = stoch_k.rolling(window=d_period).mean()
            new_cols[f'stoch_k_{k_period}_{d_period}'] = stoch_k
            new_cols[f'stoch_d_{k_period}_{d_period}'] = stoch_d
    
    return new_cols


def calc_bollinger_bands(df: pd.DataFrame, periods: list, std_devs: list[float]) -> dict[str, pd.Series]:

    new_cols = {}
    for period in periods:
        bb_middle = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
    
        for std_dev in std_devs:
            bb_upper = bb_middle + (std_dev * rolling_std)
            bb_lower = bb_middle - (std_dev * rolling_std)

            new_cols[f'bb_middle_{period}_{std_dev}'] = bb_middle
            new_cols[f'bb_upper_{period}_{std_dev}'] = bb_upper
            new_cols[f'bb_lower_{period}_{std_dev}'] = bb_lower
        
            bb_width = bb_upper - bb_lower
            bb_pct = (df['close'] - bb_lower) / bb_width

            new_cols[f'bb_width_{period}_{std_dev}'] = bb_width
            new_cols[f'bb_pct_{period}_{std_dev}'] = bb_pct

    return new_cols


def calc_atr(df: pd.DataFrame, periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    prev_close = df['close'].shift(1)
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in periods:
        atr = tr.ewm(span=period, adjust=False).mean()
        new_cols[f'atr_{period}'] = atr
    
    return new_cols


def calc_obv(df: pd.DataFrame) -> dict[str, pd.Series]:
    price_change = df['close'].diff()
    volume_direction = np.where(price_change > 0, df['volume'],
                                np.where(price_change < 0, -df['volume'], 0))
    obv = pd.Series(volume_direction, index=df.index).cumsum()
    
    return {'obv': obv}


def calc_volume_sma(df: pd.DataFrame, periods: list) -> dict[str, pd.Series]:
    new_cols = {}
    for period in periods:
        new_cols[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
    return new_cols
