import pandas as pd
import logging 
from src.strategy.technical_strategy import mean_reversion

logger = logging.getLogger(__name__)

STRATEGY_REGISTRY = {
    "mean_reversion": mean_reversion,
}

def build_strategy(df: pd.DataFrame, strategy_config: dict) -> pd.Series:
    """
    Factory function pentru a obține obiectul strategiei cerute.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Schema Contract violated! Expected DatetimeIndex, but got {type(df.index)}")
    
    if df.empty:
        raise ValueError("Schema Contract violated! DataFrame is empty!")
    
    name_strategy = strategy_config.pop("name")
    
    if name_strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategia '{name_strategy}' nu există în STRATEGY_REGISTRY! Opțiuni valide: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy = STRATEGY_REGISTRY[name_strategy]
    logger.info(f"Loading strategy: {strategy}, type: {strategy.__name__} with parameters: {strategy_config}")
    return strategy(df, strategy_config)
