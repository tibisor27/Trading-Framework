import pandas as pd
import logging

from src.schemas import Dataset

logger = logging.getLogger(__name__)
    

def assemble(X: pd.DataFrame,
            y: pd.Series, 
            feature_cols: list[str],
            ) -> Dataset:

    for name, obj in [("X", X), ("y", y)]:
        if not isinstance(obj.index, pd.DatetimeIndex):
            raise TypeError(f"Alignment Contract Violated! '{name}' does not have DatetimeIndex.")
    
    
    y_clean = y.dropna()
    logger.info(
        f"Target: {len(y)} total rows -> {len(y_clean)} tradeable rows "
        f"({len(y_clean)/len(y)*100:.1f}%)"
    )
    
    idx = X.index.intersection(y_clean.index)
   
    allowed_cols = feature_cols

    missing = [c for c in allowed_cols if c not in X.columns]
    if missing:
        raise KeyError(
            f"Whitelist columns missing from DataFrame: {missing}. "
            f"Check your feature pipeline.yaml or signal name."
        )
    
    X_ml = X.loc[idx, allowed_cols]
    y_ml = y_clean.loc[idx]

    assert X_ml.index.equals(y_ml.index), "CRITICAL: X and y misaligned after assembly!"

    nan_cols = X_ml.columns[X_ml.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"NaN values found in feature columns after assembly: {nan_cols}. "
            f"This means a feature was not properly calculated."
        )
    
    logger.info(
        f"Assembly complete: {X_ml.shape[0]} samples, "
        f"{X_ml.shape[1]} features, "
        f"period: {X_ml.index.min()} -> {X_ml.index.max()}"
    )

    return Dataset(
        X=X_ml,
        y=y_ml,
        feature_names=tuple(X_ml.columns.tolist()),
        target_name=y.name,
    )