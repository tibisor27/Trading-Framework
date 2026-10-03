import pandas as pd
from src.splitting.cronological_splitting import chronological_fixed_split
from src.splitting.walk_forward_splitting import walk_forward_split
from src.config import DataSplit
from src.contracts import Dataset
import logging
logger = logging.getLogger(__name__)

SPLITTER_REGISTRY = {
    "chronological_fixed_split": chronological_fixed_split,
    "walk_forward_split": walk_forward_split
}

def get_splitter(dataset: Dataset,split_config: dict, purging: int) -> DataSplit | list[DataSplit]:
    
    
    local_config = split_config.copy()
    name_splitter = local_config.pop("type")

    if name_splitter not in SPLITTER_REGISTRY:
        raise ValueError(f"Splitter-ul '{name_splitter}' nu există! Opțiuni: {list(SPLITTER_REGISTRY.keys())}")
    
    splitter = SPLITTER_REGISTRY[name_splitter]
    logger.info(f"Loading splitter: {splitter.__name__}")
    data_splits = splitter(dataset, **local_config, purging=purging)
    
    return data_splits
