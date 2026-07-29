import pandas as pd
import logging

from src.features.time_features import add_time_features
from src.features.technical_indicators import (
    calc_sma, calc_ema, calc_macd, calc_rsi, 
    calc_stochastic, calc_bollinger_bands, 
    calc_atr, calc_obv, calc_volume_sma
)

logger = logging.getLogger(__name__)

FEATURE_REGISTRY = {
    # Temporal
    "time_features": add_time_features,
    
    # Technical
    "sma": calc_sma,
    "ema": calc_ema,
    "macd": calc_macd,
    "rsi": calc_rsi,
    "stochastic": calc_stochastic,
    "bollinger_bands": calc_bollinger_bands,
    "atr": calc_atr,
    "obv": calc_obv,
    "volume_sma": calc_volume_sma
}

def build_all_features(df: pd.DataFrame, feature_config: dict) -> tuple[pd.DataFrame, list[str]]:

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Schema Contract violated! Expected DatetimeIndex, but got {type(df.index)}")
    
    if df.empty:
        raise ValueError("Schema Contract violated! DataFrame is empty!")
    
    all_new_series_dict = {}
    categories = {
        "temporal": feature_config.get("temporal", []),
        "technical": feature_config.get("technical", [])
    }
    
    for category_name, indicators_list in categories.items():
        if not indicators_list:
            continue
            
        logger.info(f"Building features for category: {category_name}")
            
        for ind_cfg in indicators_list:
            params = ind_cfg.copy()
            if "name" not in params:
                logger.warning(f"Missing key 'name' from {params}. Skipping!")
                continue
                
            name = params.pop("name")
            
            logger.info(f"   -> Calculating feature: {name}: {params}")
            
            if name not in FEATURE_REGISTRY:
                logger.warning(f"Warning: Indicator '{name}' not found in Feature Registry. Skipping!")
                continue

            #it takes the function from the dictionary  
            func = FEATURE_REGISTRY[name]

            #**params unpacks the parameters in the dictionary
            result_dict = func(df, **params)
            all_new_series_dict.update(result_dict)
            
    if all_new_series_dict:
        new_features_df = pd.DataFrame(all_new_series_dict, index=df.index)

        # Concatenate 1 single time, all the series features (good performance)
        # axis = 1 -> concatenate along columns (axis = 0 -> concatenate along rows)
        df = pd.concat([df, new_features_df], axis=1)
        
        # Drop "Warm-up bars" (NaNs) caused by the features that uses rolling windows and dont have values for the first bars
        logger.info(f"Dropping warm-up bars (NaNs)")
        initial_rows = len(df)
        df = df.dropna(subset=new_features_df.columns)
        logger.info(f"Dropped {initial_rows - len(df)} warm-up rows.")

        feature_names = list(new_features_df.columns)
        
        return df, feature_names
        
    return df, []
