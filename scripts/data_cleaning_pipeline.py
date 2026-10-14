"""
Step 1: Data Cleaning Orchestrator
Loads raw data, applies data quality rules, and saves to interim format.
"""
import sys
from pathlib import Path
import yaml
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DataConfig
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import argparse

def load_config(config_name: str) -> dict:
    config_path = PROJECT_ROOT / "configs" / "data" / config_name
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--config", type=str, default="eurusd_5m.yaml", help="Name of the yaml config file in configs/data/")
    args = parser.parse_args()
    
    data_cfg = load_config(args.config)
    
    # Mapează dicționarul din yaml la clasa noastră strictă DataConfig
    cfg = DataConfig(**data_cfg)
    
    logger.info("Starting Data Cleaning Pipeline...")
    
    # 1. Extract (Loader)
    df = load_raw_data(cfg)
    
    # 2. Transform (Cleaner)
    df = clean_data(df, cfg)
    
    # 3. Load (Save Interim)
    output_path = PROJECT_ROOT / cfg.interim_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvăm în format Parquet (Păstrează schema și tipurile de date)
    df.to_parquet(output_path)
    logger.info(f"Successfully saved clean dataset to {output_path}")

if __name__ == "__main__":
    main()
