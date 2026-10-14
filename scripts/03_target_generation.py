"""
Step 3: Target Generation & Cleanup
Reads math features, applies strategy targets, drops warm-up bars, saves final ML dataset.
"""
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.target_registry import get_labeler
from src.targets.strategy_registry import get_strategy
from src.data.config import DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, default="eurusd_5m.yaml")
    parser.add_argument("--strategy", type=str, default="mean_reversion.yaml")
    parser.add_argument("--targets", type=str, default="triple_barrier.yaml")
    args = parser.parse_args()

    raw_data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data" / args.data_config)
    data_cfg = DataConfig(**raw_data_cfg)
    
    strategy_cfg = load_yaml(PROJECT_ROOT / "configs" / "strategy" / args.strategy)
    target_cfg = load_yaml(PROJECT_ROOT / "configs" / "targets" / args.targets)

    input_path = PROJECT_ROOT / data_cfg.features_file
    output_path = PROJECT_ROOT / data_cfg.processed_file

    logger.info(f"Loading Features dataset from {input_path}")
    df = pd.read_parquet(input_path)
    
    initial_rows = len(df)
    
    logger.info(f"Loading Strategy: {strategy_cfg['name']}...")
    strategy = get_strategy(strategy_cfg["name"], **strategy_cfg)
    
    logger.info("Evaluating Trading Setups (Signals)...")
    signals = strategy.generate_signals(df)
    df["setup_direction"] = signals
    
    logger.info(f"Applying Target Labeler: {target_cfg['name']}...")
    labeler = get_labeler(target_cfg["name"], **target_cfg)
    df = labeler.generate_targets(df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Successfully saved FINAL Model-Ready dataset to {output_path}")

if __name__ == "__main__":
    main()
