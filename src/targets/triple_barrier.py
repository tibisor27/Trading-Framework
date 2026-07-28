import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)

def triple_barrier_target(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    horizon: int,
    pip_size: float,
    use_atr: bool,
    atr_col: str = None,
    atr_tp_multiplier: float = None,
    atr_sl_multiplier: float = None,
) -> pd.Series:
    """
    Adds a direction-aware trade target.

    trade_target = 1 if the technical setup reaches TP before SL.
    trade_target = 0 if it reaches SL before TP or expires.
    trade_target = NaN if there is no technical setup.
    """

    if use_atr:
        # when the yaml doesnt have the required parameters for atr it will raise an error
        if atr_col is None or atr_tp_multiplier is None or atr_sl_multiplier is None:
            raise ValueError(
                "use_atr=True but ATR parameters are missing from config! "
                "Set 'atr_col', 'atr_tp_multiplier', 'atr_sl_multiplier' in your YAML."
            )
        
        if atr_col not in df.columns:
            raise ValueError(f"Column '{atr_col}' not found in the DataFrame. Add '{atr_col}' to the feature config.")
        
        logger.info(f"Using ATR-based target distances: TP={atr_tp_multiplier}x ATR, SL={atr_sl_multiplier}x ATR")
        atr_values = df[atr_col].to_numpy()

        tp_distance = atr_values * atr_tp_multiplier + tp_pips * pip_size
        sl_distance = atr_values * atr_sl_multiplier + sl_pips * pip_size

        logger.info(f"Tp pips: {np.mean(tp_distance):.4f}, Sl pips {np.mean(sl_distance):.4f}")
    else:
        tp_distance = tp_pips * pip_size
        sl_distance = sl_pips * pip_size
        logger.info(f"Using fixed TP/SL: TP={tp_distance} pips, SL={sl_distance} pips")

    df = df.copy()

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    direction = df["signal"].to_numpy()

    trade_target = np.full(len(df), np.nan)

    for i in range(len(df)):
        if direction[i] == 0:
            continue

        entry_price = close[i]

        if use_atr:
            current_tp_dist = tp_distance[i]
            current_sl_dist = sl_distance[i]
        else:
            current_tp_dist = tp_distance
            current_sl_dist = sl_distance

        if direction[i] == 1:
            tp_price = entry_price + current_tp_dist
            sl_price = entry_price - current_sl_dist
        elif direction[i] == -1:
            tp_price = entry_price - current_tp_dist
            sl_price = entry_price + current_sl_dist
        else:
            continue

        end_idx = min(i + horizon, len(df) - 1)

        label = 0

        for j in range(i + 1, end_idx + 1):
            if direction[i] == 1:
                tp_hit = high[j] >= tp_price
                sl_hit = low[j] <= sl_price
            else:
                tp_hit = low[j] <= tp_price
                sl_hit = high[j] >= sl_price

            if tp_hit and sl_hit:
                # Conservative assumption for ambiguous same-candle hits.
                label = 0
                break

            if tp_hit:
                label = 1
                break

            if sl_hit:
                label = 0
                break

        trade_target[i] = label

    return trade_target
