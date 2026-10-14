'''
Prepare the processed dataset used by the EUR/USD M1 strategy.

- load and clean raw OHLCV data
- add technical indicators.
- add strategy-specific features
- creates direction-aware trading labels used by MLs
- save the processed dataset
'''

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.trade_target import add_setup_direction, add_trade_target
from src.features.technical_indicators import (
    add_all_indicators,
    add_strategy_v2_features,
)
from src.utils.data_loader import load_and_prepare_data


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    raw_file = PROJECT_ROOT / config["data"]["raw_file"]
    output_file = PROJECT_ROOT / config["data"]["processed_file"]

    strategy_config = config["strategy_v2"]

    print("Preparing processed dataset")
    print("-" * 60)

    df = load_and_prepare_data(filepath=raw_file)
    print(f"Loaded rows: {len(df):,}")

    df = add_all_indicators(df)
    df = add_strategy_v2_features(df)

    df = add_setup_direction(df, strategy_config=strategy_config)

    df = add_trade_target(
        df=df,
        tp_pips=strategy_config["tp_pips"],
        sl_pips=strategy_config["sl_pips"],
        time_limit_bars=strategy_config["time_exit_minutes"],
    )

    rows_before_dropna = len(df)

    required_non_target_cols = [
        "open",
        "high",
        "low",
        "close",
        "rsi",
        "bb_lower",
        "bb_upper",
    ]

    feature_cols = strategy_config["features"]

    df = df.dropna(subset=required_non_target_cols + feature_cols).copy()

    rows_after_dropna = len(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file)

    print(f"Saved dataset: {output_file}")
    print(f"Final rows: {rows_after_dropna:,}")
    print(f"Dropped rows: {rows_before_dropna - rows_after_dropna:,}")

    print("Setup direction distribution:")
    print(df["setup_direction"].value_counts(dropna=False).sort_index())

    target_rows = df.dropna(subset=["trade_target"]).copy()

    if not target_rows.empty:
        class_balance = target_rows["trade_target"].value_counts(normalize=True).sort_index()

        print("Trade target distribution:")
        for target_class, ratio in class_balance.items():
            print(f"  class {int(target_class)}: {ratio * 100:.2f}%")

        print("Trade target count:")
        print(target_rows["trade_target"].value_counts().sort_index())
    else:
        print("Warning: no trade_target rows found.")


if __name__ == "__main__":
    main()