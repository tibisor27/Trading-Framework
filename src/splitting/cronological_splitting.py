import pandas as pd
import logging

from src.contracts import Dataset
from src.config import DataSplit

logger = logging.getLogger(__name__)


def chronological_fixed_split(dataset: Dataset, train_end: str, validation_end: str, purging: int) -> DataSplit:
    """
    Chronological split for time series data.

    Contract In:  Dataset (from Data Engineering Pipeline)
    Contract Out: DataSplit (for ML Pipeline - Trainer)
    """
    X = dataset.X
    y = dataset.y

    train_end_raw = pd.Timestamp(train_end)
    val_end_raw = pd.Timestamp(validation_end)

    train_end_raw = train_end_raw.tz_localize(X.index.tz)
    val_end_raw = val_end_raw.tz_localize(X.index.tz)
    
    train_end_purging = train_end_raw - pd.Timedelta(minutes=purging)
    val_end_purging = val_end_raw - pd.Timedelta(minutes=purging)

    if train_end_purging >= val_end_purging:
        raise ValueError(
            f"train_end ({train_end_purging}) must be before validation_end ({val_end_purging})!"
        )

    # CHRONOLOGICAL SPLIT (using .loc on DatetimeIndex)
    X_train = X.loc[X.index <= train_end_purging]
    y_train = y.loc[y.index <= train_end_purging]

    X_val = X.loc[(X.index > train_end_raw) & (X.index <= val_end_purging)]
    y_val = y.loc[(y.index > train_end_raw) & (y.index <= val_end_purging)]

    X_test = X.loc[X.index > val_end_raw]
    y_test = y.loc[y.index > val_end_raw]

    logger.info(
        f"Chronological fixed split complete:\n"
        f"  Train: {len(X_train):,} samples ({X_train.index.min()} -> {X_train.index.max()})\n"
        f"  Val:   {len(X_val):,} samples ({X_val.index.min()} -> {X_val.index.max()})\n"
        f"  Test:  {len(X_test):,} samples ({X_test.index.min()} -> {X_test.index.max()})"
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
