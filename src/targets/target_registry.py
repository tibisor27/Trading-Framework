import pandas as pd
from src.targets.triple_barrier import triple_barrier_target
import logging
logger = logging.getLogger(__name__)

LABELER_REGISTRY = {
    "triple_barrier": triple_barrier_target
}

def build_target(df: pd.DataFrame, target_config: dict) -> pd.Series:
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Schema Contract violated! Expected DatetimeIndex, but got {type(df.index)}")
    
    if df.empty:
        raise ValueError("Schema Contract violated! DataFrame is empty!")
    
    if "signal" not in df.columns:
        raise ValueError("Schema Contract violated! DataFrame is missing 'signal' column!")
    
    local_config = target_config.copy()
    name_target = local_config.pop("type")

    if name_target not in LABELER_REGISTRY:
        raise ValueError(f"Labeler-ul '{name_target}' nu există! Opțiuni: {list(LABELER_REGISTRY.keys())}")
    
    labeler = LABELER_REGISTRY[name_target]
    logger.info(f"Loading labeler: {labeler.__name__}")
    trade_target = labeler(df, **local_config)
    return pd.Series(trade_target, index=df.index, name="target_y")
