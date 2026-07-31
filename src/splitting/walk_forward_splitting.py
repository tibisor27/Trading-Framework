import pandas as pd
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from src.contracts import Dataset
from src.config import DataSplit

logger = logging.getLogger(__name__)


def walk_forward_split(dataset: Dataset, purging: int, n_folds: int) -> list[DataSplit]:
    """
    Walk forward split for time series data.

    Contract In:  Dataset (from Data Engineering Pipeline)
    Contract Out: DataSplit (for Trainer)
    """


    # 1. Tai Test set-ul final (cronologic fix)
    X_temp, X_test, y_temp, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, shuffle=False)
    folds = []


    # 2. Faci Walk-Forward doar pe X_temp (pe care îl împarți în Train / Val)
    tscv = TimeSeriesSplit(n_splits=n_folds, gap=purging)
    for k, (train_idx, val_idx) in enumerate(tscv.split(X_temp)):
    
        X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
        y_train = y_temp.iloc[train_idx]
        y_val = y_temp.iloc[val_idx]

        logger.info(
            f"Fold {k + 1}: Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        folds.append(
            DataSplit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                feature_names=dataset.feature_names,
            )
        )
        
        

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=dataset.feature_names,
    )
