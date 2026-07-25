"""
dataset.py
-----------
Sliding-window sequence construction + WALK-FORWARD (expanding window)
splitting, framework-agnostic (plain NumPy arrays -- fed directly into
TensorFlow/Keras via `model.fit(X, y)`).

We deliberately do NOT use a random train/test split on time series -- that
leaks future information into training via overlapping windows and gives
falsely optimistic metrics. Walk-forward validation trains on the past and
always evaluates strictly on a later, unseen block of time.
"""
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, HORIZONS

SEQ_LEN = 30  # 30 trading days of lookback


def build_sequences(df: pd.DataFrame, feature_cols=FEATURE_COLUMNS,
                     seq_len: int = SEQ_LEN, scaler=None):
    """Turns a (possibly multi-ticker) feature table into sliding-window
    arrays ready for Keras:

        X: (n_samples, seq_len, n_features)
        y: (n_samples, n_horizons)

    Windows are built independently per ticker (never crossing ticker
    boundaries) and rows with any NaN in the window or target are dropped.
    """
    target_cols = [f"target_return_{h}d" for h in HORIZONS]
    X_list, y_list = [], []

    for _, g in df.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        feats = g[feature_cols].values.astype(np.float32)
        targets = g[target_cols].values.astype(np.float32)

        for i in range(seq_len, len(g)):
            window = feats[i - seq_len:i]
            target = targets[i - 1]  # already forward-looking at row i-1
            if np.isnan(window).any() or np.isnan(target).any():
                continue
            X_list.append(window)
            y_list.append(target)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    if scaler is not None:
        n_samples, seq, n_feat = X.shape
        X = scaler.transform(X.reshape(-1, n_feat)).reshape(n_samples, seq, n_feat)

    return X, y


class StandardScalerNP:
    """Lightweight standard scaler fit on training features only (no
    scikit-learn dependency needed for something this simple)."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split by DATE (not randomly) so val/test are strictly later in time
    than anything the model trained on -- prevents lookahead leakage."""
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac)]
    val_end = dates[int(n * (train_frac + val_frac))]

    train_df = df[df["Date"] <= train_end]
    val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)]
    test_df = df[df["Date"] > val_end]
    return train_df, val_df, test_df
