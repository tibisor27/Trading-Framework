'''
Run the final strict backtest on TEST.
    
    - Uses trading configurations selected on VALIDATION by script 03.
    - No grid search and no model training are performed here
'''

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from src.utils.split import temporal_train_validation_test_split
from src.backtesting.trading_grid import run_trading_backtest


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



def load_selected_configs(reports_dir):
    path = reports_dir / "selected_trading_configs_validation.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected config: {path}. Run script 03 first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_strategy_config(base_config, candidate):
    strategy_config = base_config.copy()

    strategy_config["signal_mode"] = candidate["signal_mode"]
    strategy_config["ml_threshold"] = candidate["ml_threshold"]
    strategy_config["allowed_hours"] = candidate["allowed_hours"]
    strategy_config["direction_mode"] = candidate["direction_mode"]

    return strategy_config


def evaluate_candidate(model_name, candidate, model, test_df, features, config):
    test_df = test_df.copy()
    test_df["ml_prob"] = model.predict_proba(test_df[features])[:, 1]

    strategy_config = make_strategy_config(
        base_config=config["strategy_v2"],
        candidate=candidate,
    )

    metrics = run_trading_backtest(
        df=test_df,
        strategy_config=strategy_config,
        trading_config=config["trading"],
    )

    validation_metrics = candidate["validation_metrics"]

    return {
        "model": model_name,
        "validation_rank": candidate["validation_rank"],
        "ml_threshold": candidate["ml_threshold"],
        "allowed_hours": str(candidate["allowed_hours"]),
        "direction_mode": candidate["direction_mode"],

        "validation_trades": validation_metrics["trades"],
        "validation_profit_factor": validation_metrics["profit_factor"],
        "validation_total_pnl": validation_metrics["total_pnl"],

        "test_trades": metrics["trades"],
        "test_win_rate": metrics["win_rate"],
        "test_profit_factor": metrics["profit_factor"],
        "test_total_pnl": metrics["total_pnl"],
        "test_avg_trade_pnl": metrics["avg_trade_pnl"],
        "test_max_drawdown": metrics["max_drawdown"],
    }


def main():
    print("Running final strict backtest on TEST")

    config = load_config()
    df, features = load_dataset(config)

    _, _, test_df = temporal_train_validation_test_split(
        df=df,
        train_ratio=config["split"]["train_ratio"],
        validation_ratio=config["split"]["validation_ratio"],
        purge_bars=config["split"]["purge_bars"],
    )

    reports_dir = PROJECT_ROOT / config["output"]["reports_dir"]
    selected_configs = load_selected_configs(reports_dir)

    rows = []

    for model_name, model_block in selected_configs["models"].items():
        model_path = Path(model_block["model_path"])
        model = joblib.load(model_path)

        print(f"\nModel: {model_name}")
        print(f"Loaded: {model_path}")

        for candidate in model_block["candidates"]:
            result = evaluate_candidate(
                model_name=model_name,
                candidate=candidate,
                model=model,
                test_df=test_df,
                features=features,
                config=config,
            )

            rows.append(result)

            print(
                f"rank={result['validation_rank']} | "
                f"thr={result['ml_threshold']} | "
                f"hours={result['allowed_hours']} | "
                f"dir={result['direction_mode']} | "
                f"PF={result['test_profit_factor']:.2f} | "
                f"PnL={result['test_total_pnl']:.2f}"
            )

    results_df = pd.DataFrame(rows)

    all_path = reports_dir / "final_backtest_test_all_candidates.csv"
    summary_path = reports_dir / "final_backtest_test_summary.csv"

    results_df.to_csv(all_path, index=False)

    # report the best configuration selected on VALIDATION for each model.
    summary_df = (
        results_df[results_df["validation_rank"] == 1]
        .sort_values(by=["model"])
        .copy()
    )

    summary_df.to_csv(summary_path, index=False)

    print("\nFinal TEST summary")
    print("-" * 80)

    display_cols = [
        "model",
        "ml_threshold",
        "allowed_hours",
        "direction_mode",
        "test_trades",
        "test_win_rate",
        "test_profit_factor",
        "test_total_pnl",
        "test_max_drawdown",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print("\nSaved files:")
    print(f"All candidates: {all_path}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    main()