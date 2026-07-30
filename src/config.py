from typing import List
from dataclasses import dataclass
import pandas as pd
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    raw_file: str
    pair: str
    interim_file: str
    features_file: str
    processed_file: str
    timeframe: str
    timestamp_col: str
    tz_source: str
    tz_target: str
    required_columns: List[str]
    drop_zero_volume: bool
    fill_missing_timestamps: bool

    

@dataclass
class PipelineConfig:
    data: DataConfig
    features: dict
    strategy: dict
    target: dict



def _find_configs_dir() -> Path:

    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"
    
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"'configs/' directory not found at '{configs_dir}'. "
        )
    
    return configs_dir


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, raising an error if it is not found."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.absolute()}")
    with open(path) as f:
        return yaml.safe_load(f) or {}



def load_pipeline_config(
    data: str="eurusd_1m",        # "eurusd_1m"     → configs/data/eurusd_1m.yaml
    features: str = "pipeline",    # "pipeline"      → configs/features/pipeline.yaml
    strategy: str = "mean_reversion",    # "mean_reversion"→ configs/strategy/mean_reversion.yaml
    target: str = "triple_barrier",      # "triple_barrier"→ configs/targets/triple_barrier.yaml
) -> PipelineConfig:
    """
    Composition Root: Primește DOAR identificatori (nume simple).
    Rezolvă path-urile, citește, validează, și returnează config imutabil.
    """
    
    configs_dir = _find_configs_dir()

    raw_data     = _load_yaml(configs_dir / "data"     / f"{data}.yaml")
    raw_features = _load_yaml(configs_dir / "features" / f"{features}.yaml")
    raw_strategy = _load_yaml(configs_dir / "strategy" / f"{strategy}.yaml")
    raw_target   = _load_yaml(configs_dir / "targets"  / f"{target}.yaml")
    
    # Validare + Construcție Dataclass (Fail-Fast)
    try:
        data_cfg = DataConfig(**raw_data)
    except TypeError as e:
        raise TypeError(f"DataConfig ('{data}.yaml'): {e}") from e

    return PipelineConfig(
        data=data_cfg,
        features=raw_features,
        strategy=raw_strategy,
        target=raw_target,
    )