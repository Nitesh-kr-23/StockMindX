"""
train.py
---------
Trains LSTM, GRU, and Transformer forecasters (TensorFlow/Keras) on a
strict walk-forward (date-ordered) split, evaluates each with
RMSE / MAE / MAPE / Directional Accuracy per horizon against a naive
baseline, and persists everything the app needs:

    models/<name>.keras       - trained model (native Keras format)
    models/scaler.joblib      - feature scaler + config (fit on TRAIN ONLY)
    models/metrics.json       - per-model, per-horizon evaluation metrics
    models/residual_std.json  - per-model, per-horizon validation residual
                                 std-dev, used by the app to build simple
                                 confidence intervals (prediction +/- z*std)
    models/history.json       - training curves (for optional diagnostics)

Run: python src/train.py --epochs 15
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from features import FEATURE_COLUMNS, HORIZONS
from dataset import build_sequences, StandardScalerNP, walk_forward_split, SEQ_LEN
from models import MODEL_BUILDERS

MODELS_DIR = Path("models")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Guards against near-zero returns
    dominating the metric by flooring the denominator."""
    denom = np.where(np.abs(y_true) < 1e-4, 1e-4, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where the predicted return had the same sign as
    the actual return -- the metric that matters most for a "should I buy"
    use case, independent of exact magnitude."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def _all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "directional_accuracy": _directional_accuracy(y_true, y_pred),
    }


def naive_baseline_metrics(y_test: np.ndarray) -> dict:
    """Naive persistence: predict zero forward return ("price stays flat").
    Any trained model should be judged against this before being trusted."""
    metrics = {}
    for i, h in enumerate(HORIZONS):
        y_true = y_test[:, i]
        y_pred = np.zeros_like(y_true)
        metrics[f"{h}d"] = _all_metrics(y_true, y_pred)
    return metrics


def evaluate(model: tf.keras.Model, X: np.ndarray, y: np.ndarray):
    """Returns (per-horizon metrics dict, raw residuals array)."""
    preds = model.predict(X, verbose=0)
    metrics = {}
    for i, h in enumerate(HORIZONS):
        metrics[f"{h}d"] = _all_metrics(y[:, i], preds[:, i])
    residuals = y - preds
    return metrics, residuals


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_one_model(name, builder, X_train, y_train, X_val, y_val, n_features, seq_len, epochs):
    """Trains a single architecture with early-stopping-style checkpointing
    on validation loss (via EarlyStopping + restore of the best weights)."""
    model = builder(n_features=n_features, seq_len=seq_len, n_horizons=len(HORIZONS))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.Huber())  # robust to return outliers vs plain MSE

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=max(3, epochs // 3),
        restore_best_weights=True, verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64,
        callbacks=[callback], verbose=2,
    )
    return model, history.history


def main(epochs: int = 15):
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet("data/processed/features.parquet")

    train_df, val_df, test_df = walk_forward_split(df)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  "
          f"(split strictly by date -- no shuffling across time)")

    scaler = StandardScalerNP().fit(train_df[FEATURE_COLUMNS].values)
    joblib.dump(
        {"scaler": scaler, "feature_cols": FEATURE_COLUMNS, "seq_len": SEQ_LEN},
        MODELS_DIR / "scaler.joblib",
    )

    X_train, y_train = build_sequences(train_df, scaler=scaler)
    X_val, y_val = build_sequences(val_df, scaler=scaler)
    X_test, y_test = build_sequences(test_df, scaler=scaler)
    print(f"X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

    all_metrics = {"Naive": naive_baseline_metrics(y_test)}
    all_history = {}
    residual_std = {}

    for name, builder in MODEL_BUILDERS.items():
        print(f"\n=== Training {name} ===")
        model, history = train_one_model(
            name, builder, X_train, y_train, X_val, y_val,
            n_features=len(FEATURE_COLUMNS), seq_len=SEQ_LEN, epochs=epochs,
        )
        test_metrics, test_residuals = evaluate(model, X_test, y_test)

        all_metrics[name] = test_metrics
        all_history[name] = history
        residual_std[name] = {
            f"{h}d": float(np.std(test_residuals[:, i])) for i, h in enumerate(HORIZONS)
        }

        model.save(MODELS_DIR / f"{name}.keras")
        print(f"[{name}] test metrics: {test_metrics}")

    # Identify the best-performing trained model (excludes the naive
    # baseline, which is a reference point, not a candidate) by mean test
    # RMSE across horizons.
    candidates = {k: v for k, v in all_metrics.items() if k != "Naive"}
    best_model = min(candidates, key=lambda k: np.mean([m["rmse"] for m in candidates[k].values()]))

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"metrics": all_metrics, "best_model": best_model}, f, indent=2)
    with open(MODELS_DIR / "residual_std.json", "w") as f:
        json.dump(residual_std, f, indent=2)
    with open(MODELS_DIR / "history.json", "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"\n=== BEST MODEL (by mean test RMSE): {best_model} ===")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(epochs=args.epochs)
