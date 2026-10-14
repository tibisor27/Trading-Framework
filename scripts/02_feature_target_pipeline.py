"""
Step 2: Feature & Target Engineering Pipeline
Applies pure mathematical features, then evaluates business rules for targets.
"""
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_features import add_time_features
from src.features.technical_indicators import build_technical_features
from src.targets.trade_target import add_setup_direction, add_trade_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, default="eurusd_5m.yaml")
    parser.add_argument("--feature-config", type=str, default="technical.yaml")
    args = parser.parse_args()

    # Încărcăm Configurațiile Hydra-style
    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data" / args.data_config)
    feat_cfg = load_yaml(PROJECT_ROOT / "configs" / "features" / args.feature_config)
    
    # Target-urile sunt momentan în config-ul global (strategy_v2)
    global_cfg = load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    strategy_cfg = global_cfg["strategy_v2"]

    input_path = PROJECT_ROOT / data_cfg["interim_file"]
    output_path = PROJECT_ROOT / data_cfg["processed_file"]

    logger.info(f"Loading cleaned dataset from {input_path}")
    df = pd.read_csv(input_path, index_col=data_cfg.get("timestamp_col", "timestamp"), parse_dates=True)
    
    initial_rows = len(df)
    
    # ==========================================
    # 1. FEATURE ENGINEERING (Matematică Pură)
    # ==========================================
    logger.info("Adding Time Features...")
    df = add_time_features(df)
    
    logger.info("Building Technical Features dynamically from Registry...")
    df = build_technical_features(df, feat_cfg["indicators"])
    
    # ==========================================
    # 2. TARGET GENERATION (Reguli de Business)
    # ==========================================
    logger.info("Evaluating Trading Setups (Targets)...")
    df = add_setup_direction(df, strategy_config=strategy_cfg)
    
    logger.info("Simulating Trades for Trade Targets...")
    df = add_trade_target(
        df=df,
        tp_pips=strategy_cfg["tp_pips"],
        sl_pips=strategy_cfg["sl_pips"],
        time_limit_bars=strategy_cfg["time_exit_minutes"]
    )
    
    # ==========================================
    # 3. FINAL CLEANUP
    # ==========================================
    logger.info("Dropping NaNs created by rolling indicators (warm-up bars)...")
    # Ștergem NaNs doar de pe coloanele de Features.
    # NU ștergem de pe 'trade_target' deoarece acela este intenționat NaN în 99% din timp (când nu există setup)
    indicator_columns = [col for col in df.columns if col not in ['setup_direction', 'trade_target']]
    df = df.dropna(subset=indicator_columns)
    
    final_rows = len(df)
    logger.info(f"Dropped {initial_rows - final_rows} warm-up rows.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Successfully saved Model-Ready dataset to {output_path}")

if __name__ == "__main__":
    main()
