# training/baseline_stats.py
from __future__ import annotations
import pandas as pd

def compute_baseline_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-feature stats used as a simple drift baseline.
    Returns a DataFrame indexed by ["mean", "std", "min", "max"].
    """
    return X.describe(percentiles=[]).loc[["mean", "std", "min", "max"]]

def save_baseline_json(df: pd.DataFrame, path: str = "baseline.json") -> None:
    df.to_json(path)
