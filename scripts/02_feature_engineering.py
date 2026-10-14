"""
Step 2: Feature Engineering (Master Pipeline)
Reads clean data, applies all features from pipeline.yaml (Temporal + Technical),
and saves intermediate features file.
"""
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.registry import build_all_features
from src.features.config import FeaturesConfig
from src.data.config import DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, default="eurusd_5m.yaml")
    parser.add_argument("--pipeline-config", type=str, default="pipeline.yaml")
    args = parser.parse_args()

    # Încărcăm YAML brut și îl trecem imediat prin Data Contract
    raw_data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data" / args.data_config)
    data_cfg = DataConfig(**raw_data_cfg)
    
    # Încărcăm YAML brut, dar îl trecem prin Contract (simplu)
    raw_pipeline_cfg = load_yaml(PROJECT_ROOT / "configs" / "features" / args.pipeline_config)
    pipeline_cfg = FeaturesConfig(**raw_pipeline_cfg)

    input_path = PROJECT_ROOT / data_cfg.interim_file
    output_path = PROJECT_ROOT / data_cfg.features_file

    logger.info(f"Loading cleaned dataset from {input_path}")
    # Parquet păstrează automat tipul DatetimeIndex. Nu mai e nevoie de parse_dates!
    df = pd.read_parquet(input_path)
    
    logger.info("Building ALL Features dynamically from Master Registry...")
    df = build_all_features(df, pipeline_cfg)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Successfully saved all Features to {output_path}")

if __name__ == "__main__":
    main()
