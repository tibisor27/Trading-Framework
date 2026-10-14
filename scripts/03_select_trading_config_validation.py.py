#!/usr/bin/env python3
"""
Select trading configurations on the validation set.

This script loads the ML models selected in 02_train_models.py and runs a
trading grid search on VALIDATION only. TEST is not used here.
"""

import sys
from pathlib import Path
from ast import literal_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from src.utils.split import temporal_train_validation_test_split
from src.backtesting.trading_grid import evaluate_trading_grid, select_validation_candidates


MODEL_FILES = {
    "random_forest": "random_forest_selected_train_only.pkl",
    "xgboost": "xgboost_selected_train_only.pkl",
    "lightgbm": "lightgbm_selected_train_only.pkl",
}


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(config):
    path = PROJECT_ROOT / config["data"]["processed_file"]
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    features = config["strategy_v2"]["features"]

    required_cols = features + [
        "open",
        "high",
        "low",
        "close",
        "rsi",
        "bb_lower",
        "bb_upper",
        "setup_direction",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")


    df = df.dropna(subset=required_cols).copy()
    df["setup_direction"] = df["setup_direction"].astype(int)

    return df, features


def load_models(models_dir):
    models = {}

    for model_name, file_name in MODEL_FILES.items():
        path = models_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Missing model for {model_name}: {path}")

        models[model_name] = {
            "model": joblib.load(path),
            "path": path,
        }

    return models


def candidate_to_dict(model_name, model_path, row, rank):
    return {
        "model": model_name,
        "model_path": str(model_path),
        "validation_rank": int(rank),
        "signal_mode": "ml_regime_filter",
        "ml_threshold": float(row["ml_threshold"]),
        "allowed_hours": literal_eval(row["allowed_hours"]),
        "direction_mode": row["direction_mode"],
        "validation_metrics": {
            "trades": int(row["trades"]),
            "win_rate": float(row["win_rate"]),
            "profit_factor": float(row["profit_factor"]),
            "total_pnl": float(row["total_pnl"]),
            "avg_trade_pnl": float(row["avg_trade_pnl"]),
            "max_drawdown": float(row["max_drawdown"]),
        },
    }


def main():
    print("Selecting trading configurations on VALIDATION")

    config = load_config()
    df, features = load_dataset(config)

    _, validation_df, _ = temporal_train_validation_test_split(
        df=df,
        train_ratio=config["split"]["train_ratio"],
        validation_ratio=config["split"]["validation_ratio"],
        purge_bars=config["split"]["purge_bars"],
    )

    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    reports_dir = PROJECT_ROOT / config["output"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(models_dir)

    selected_config = {
        "selection_split": "validation",
        "test_used": False,
        "models": {},
    }

    all_results = []

    for model_name, model_info in models.items():
        print(f"\nModel: {model_name}")

        validation_with_prob = validation_df.copy()
        validation_with_prob["ml_prob"] = model_info["model"].predict_proba(
            validation_with_prob[features]
        )[:, 1]

        results = evaluate_trading_grid(
            df=validation_with_prob,
            base_strategy_config=config["strategy_v2"],
            trading_config=config["trading"]
                    )

        results["model"] = model_name

        candidates = select_validation_candidates(results, top_n=5)

        results.to_csv(
            reports_dir / f"{model_name}_trading_grid_validation.csv",
            index=False,
        )

        candidates.to_csv(
            reports_dir / f"{model_name}_selected_trading_candidates_validation.csv",
            index=False,
        )

        selected_config["models"][model_name] = {
            "model_path": str(model_info["path"]),
            "candidates": [
                candidate_to_dict(
                    model_name=model_name,
                    model_path=model_info["path"],
                    row=row,
                    rank=rank,
                )
                for rank, (_, row) in enumerate(candidates.iterrows(), start=1)
            ],
        }

        all_results.append(results)

        print(candidates.to_string(index=False))

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(
        reports_dir / "trading_grid_validation_all_models.csv",
        index=False,
    )

    output_path = reports_dir / "selected_trading_configs_validation.yaml"

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(selected_config, file, sort_keys=False)

    print(f"\nSaved selected configs: {output_path}")
    print("Next: run scripts/04_final_backtest_test.py")


if __name__ == "__main__":
    main()