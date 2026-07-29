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

    #local_config prevents the original config from being modified 
    #because it share the same memory if it's not copied
    local_config = strategy_config.copy()

    name_strategy = local_config.pop("name")
    
    if name_strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategia '{name_strategy}' nu există în STRATEGY_REGISTRY! Opțiuni valide: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy = STRATEGY_REGISTRY[name_strategy]
    logger.info(f"Loading strategy: {strategy}, type: {strategy.__name__} with parameters: {strategy_config}")
    return strategy(df, **local_config)
