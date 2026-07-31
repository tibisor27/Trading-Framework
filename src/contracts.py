from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: tuple[str, ...]
