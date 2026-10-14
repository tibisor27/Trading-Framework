#!/usr/bin/env python3
"""
Train and select ML models for the EUR/USD M1 strategy.

Models are trained on TRAIN, selected on VALIDATION using PR-AUC,
and evaluated on TEST. No trading backtest is performed here.

Models are trained to predict direction-aware trade_target:
1 = technical setup reaches TP before SL, 0 = SL/time expiry.

Outputs:
- selected ML models saved in results/models
- grid search results saved in results/reports
- selected model summary saved as CSV
- selected model configuration saved as YAML
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from src.utils.split import temporal_train_validation_test_split
from src.utils.ml_metrics import evaluate_classifier
from src.models.model_grids import get_model_grids
from src.models.model_factory import build_model


REPORTING_THRESHOLD = 0.20


def load_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(config):
    dataset_path = PROJECT_ROOT / config["data"]["processed_file"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. "
            "Run scripts/01_prepare_dataset.py first."
        )

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

    features = config["strategy_v2"]["features"]

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")

    required_columns = features + ["trade_target", "setup_direction"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    # Keep only rows with a valid technical setup target.
    # Rows without setup have trade_target = NaN and are excluded from ML training.
    df = df.dropna(subset=features + ["trade_target"]).copy()
    df["trade_target"] = df["trade_target"].astype(int)
    df["setup_direction"] = df["setup_direction"].astype(int)

    return df, features


def print_split_info(train_df, val_df, test_df):
    print("\n" + "=" * 90)
    print("TEMPORAL SPLIT")
    print("=" * 90)

    print(f"TRAIN:      {train_df.index[0]} -> {train_df.index[-1]} | rows: {len(train_df):,}")
    print(f"VALIDATION: {val_df.index[0]} -> {val_df.index[-1]} | rows: {len(val_df):,}")
    print(f"TEST:       {test_df.index[0]} -> {test_df.index[-1]} | rows: {len(test_df):,}")

    print("=" * 90)


def flatten_selected_params(summary, selected_params):
    for key, value in selected_params.items():
        if key not in ["model_type", "model_name"]:
            summary[f"param_{key}"] = value

    return summary


def train_model_family(
    model_type,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    random_state,
    n_jobs,
):
    rows = []
    best_model = None
    best_params = None
    best_pr_auc = -1.0

    print("\n" + "#" * 90)
    print(f"GRID SEARCH: {model_type}")
    print("#" * 90)
    print(f"Candidate configurations: {len(param_grid)}")

    for index, params in enumerate(param_grid, start=1):
        model_name = params["model_name"]

        print(f"\n[{index}/{len(param_grid)}] Training {model_name}")

        model = build_model(params, random_state=random_state, n_jobs=n_jobs)
        model.fit(X_train, y_train)

        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]

        train_metrics = evaluate_classifier(
            y_true=y_train,
            y_prob=train_prob,
            threshold=REPORTING_THRESHOLD,
        )

        val_metrics = evaluate_classifier(
            y_true=y_val,
            y_prob=val_prob,
            threshold=REPORTING_THRESHOLD,
        )

        row = {
            "model_type": model_type,
            "model_name": model_name,
            **params,
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["pr_auc"],
            "validation_precision_at_0_20": val_metrics["precision"],
            "validation_recall_at_0_20": val_metrics["recall"],
            "validation_f1_at_0_20": val_metrics["f1"],
        }

        rows.append(row)

        print(
            "Validation metrics | "
            f"ROC-AUC: {val_metrics['roc_auc']:.6f} | "
            f"PR-AUC: {val_metrics['pr_auc']:.6f} | "
            f"F1@0.20: {val_metrics['f1']:.6f}"
        )

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_model = model
            best_params = params

            print(f"New best {model_type}: {model_name} | validation PR-AUC: {best_pr_auc:.6f}")

    test_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_classifier(
        y_true=y_test,
        y_prob=test_prob,
        threshold=REPORTING_THRESHOLD,
    )

    summary = {
        "model_type": model_type,
        "selected_model_name": best_params["model_name"],
        "selection_metric": "validation_pr_auc",
        "validation_pr_auc": best_pr_auc,
        "test_threshold": REPORTING_THRESHOLD,
        "test_precision_at_0_20": test_metrics["precision"],
        "test_recall_at_0_20": test_metrics["recall"],
        "test_f1_at_0_20": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    }

    summary = flatten_selected_params(summary, best_params)

    print("\n" + "-" * 90)
    print(f"SELECTED MODEL FOR {model_type}")
    print("-" * 90)
    print(f"Selected model:      {best_params['model_name']}")
    print(f"Validation PR-AUC:   {best_pr_auc:.6f}")
    print(f"Test Precision@0.20: {test_metrics['precision']:.6f}")
    print(f"Test Recall@0.20:    {test_metrics['recall']:.6f}")
    print(f"Test F1@0.20:        {test_metrics['f1']:.6f}")
    print(f"Test ROC-AUC:        {test_metrics['roc_auc']:.6f}")
    print(f"Test PR-AUC:         {test_metrics['pr_auc']:.6f}")

    return best_model, best_params, pd.DataFrame(rows), summary


def build_selected_config(summary_df):
    selected_config = {}

    for _, row in summary_df.iterrows():
        model_type = row["model_type"]

        hyperparameters = {}

        for col in summary_df.columns:
            if not col.startswith("param_"):
                continue

            value = row[col]

            if pd.isna(value):
                continue

            clean_name = col.replace("param_", "")
            hyperparameters[clean_name] = value

        selected_config[model_type] = {
            "selected_model_name": row["selected_model_name"],
            "selection_metric": row["selection_metric"],
            "validation_pr_auc": float(row["validation_pr_auc"]),
            "test_threshold": float(row["test_threshold"]),
            "test_precision_at_0_20": float(row["test_precision_at_0_20"]),
            "test_recall_at_0_20": float(row["test_recall_at_0_20"]),
            "test_f1_at_0_20": float(row["test_f1_at_0_20"]),
            "test_roc_auc": float(row["test_roc_auc"]),
            "test_pr_auc": float(row["test_pr_auc"]),
            "model_path": row["model_path"],
            "hyperparameters": hyperparameters,
        }

    return selected_config


def print_final_report(summary_df, output_files):
    print("\n" + "=" * 90)
    print("FINAL SELECTED MODELS SUMMARY")
    print("=" * 90)

    display_cols = [
        "model_type",
        "selected_model_name",
        "validation_pr_auc",
        "test_precision_at_0_20",
        "test_recall_at_0_20",
        "test_f1_at_0_20",
        "test_roc_auc",
        "test_pr_auc",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("SELECTED HYPERPARAMETERS")
    print("=" * 90)

    param_cols = [col for col in summary_df.columns if col.startswith("param_")]

    for _, row in summary_df.iterrows():
        print(f"\n{row['model_type']} -> {row['selected_model_name']}")

        for col in param_cols:
            value = row[col]

            if pd.isna(value):
                continue

            clean_name = col.replace("param_", "")
            print(f"  {clean_name}: {value}")

    print("\n" + "=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    for label, path in output_files.items():
        print(f"{label}: {path}")

    print("=" * 90)


def main():
    print("\n" + "=" * 90)
    print("ML MODEL TRAINING AND SELECTION")
    print("=" * 90)
    print("Train split:       model fitting")
    print("Validation split:  hyperparameter selection using PR-AUC")
    print("Test split:        final ML evaluation")
    print("Trading backtest:  not performed in this script")
    print("=" * 90)

    config = load_config()
    df, features = load_dataset(config)

    train_df, val_df, test_df = temporal_train_validation_test_split(
        df=df,
        train_ratio=config["split"]["train_ratio"],
        validation_ratio=config["split"]["validation_ratio"],
        purge_bars=config["split"]["purge_bars"],
    )

    print_split_info(train_df, val_df, test_df)

    target_col = "trade_target"

    X_train, y_train = train_df[features], train_df[target_col]
    X_val, y_val = val_df[features], val_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    print("\nTraining target:", target_col)
    print("Target distribution:")
    print("TRAIN:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("VALIDATION:")
    print(y_val.value_counts(normalize=True).sort_index())
    print("TEST:")
    print(y_test.value_counts(normalize=True).sort_index())

    print("\nRows used for ML:")
    print(f"TRAIN:      {len(train_df):,}")
    print(f"VALIDATION: {len(val_df):,}")
    print(f"TEST:       {len(test_df):,}")

    random_state = config["model"].get("random_state", 42)
    n_jobs = config["model"]["random_forest"].get("n_jobs", -1)

    models_dir = PROJECT_ROOT / "results" / "models"
    reports_dir = PROJECT_ROOT / "results" / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    all_results = []
    summaries = []
    output_files = {}

    for model_type, grid in get_model_grids(random_state).items():
        model, params, results_df, summary = train_model_family(
            model_type=model_type,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        model_path = models_dir / f"{model_type}_selected_train_only.pkl"
        joblib.dump(model, model_path)

        summary["model_path"] = str(model_path)

        grid_results_path = reports_dir / f"{model_type}_grid_search_results_{timestamp}.csv"
        results_df.to_csv(grid_results_path, index=False)

        output_files[f"{model_type} selected model"] = model_path
        output_files[f"{model_type} grid search CSV"] = grid_results_path

        all_results.append(results_df)
        summaries.append(summary)

        print(f"\nSaved selected model: {model_path}")
        print(f"Saved grid search results: {grid_results_path}")

    all_grid_results_path = reports_dir / f"ml_grid_search_all_models_{timestamp}.csv"
    pd.concat(all_results, ignore_index=True).to_csv(all_grid_results_path, index=False)

    summary_df = pd.DataFrame(summaries)

    summary_path = reports_dir / "selected_models_ml_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    selected_config = build_selected_config(summary_df)

    config_path = reports_dir / "selected_models_ml_config.yaml"

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(selected_config, file, sort_keys=False)

    output_files["all grid search CSV"] = all_grid_results_path
    output_files["selected models summary CSV"] = summary_path
    output_files["selected models config YAML"] = config_path

    print_final_report(summary_df, output_files)


if __name__ == "__main__":
    main()