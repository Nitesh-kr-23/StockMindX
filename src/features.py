"""
features.py
------------
Feature engineering for NeuralHorizon.

Computes a focused set of ~13 technical indicators per ticker (chosen for
proven predictive/diagnostic value, not exhaustiveness — more indicators is
not automatically better and adds noise/collinearity) plus the forward
targets used for training.

TARGET DESIGN: we predict forward LOG RETURNS at each horizon, not raw
price. Predicting raw price is a common beginner mistake because a model
can look deceptively accurate by simply echoing "tomorrow's price = today's
price". Log returns force the model to actually learn direction and
magnitude of change. Future PRICES are then derived deterministically from
predicted returns at inference time (see app/app.py) — no separate price
model is trained.
"""
import numpy as np
import pandas as pd

# Forecast horizons, in trading days.
HORIZONS = [1, 5, 20]


# --------------------------------------------------------------------------- #
# Indicator building blocks
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index — momentum oscillator (0-100), flags
    overbought (>70) / oversold (<30) conditions."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram — trend/momentum indicator."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """%B: where price sits within its Bollinger Bands (0 = lower band,
    1 = upper band). More informative for ML than the raw band values."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = ma + n_std * std, ma - n_std * std
    return (close - lower) / (upper - lower + 1e-9)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — volatility indicator that accounts for gaps."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume flow signed by price direction,
    used as a leading indicator of buying/selling pressure."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# --------------------------------------------------------------------------- #
# Main feature builder
# --------------------------------------------------------------------------- #
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features + targets for a SINGLE ticker's rows (must be
    pre-sorted or will be sorted here by Date)."""
    df = df.sort_values("Date").copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # -- Returns & volatility --
    df["log_return_1d"] = np.log(close / close.shift(1))
    df["volatility_20d"] = df["log_return_1d"].rolling(20).std() * np.sqrt(252)

    # -- Trend: moving averages & price-to-MA ratios --
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["price_to_sma_20"] = close / df["sma_20"] - 1
    df["price_to_sma_50"] = close / df["sma_50"] - 1

    # -- Momentum --
    df["rsi_14"] = _rsi(close)
    df["macd"], _, df["macd_hist"] = _macd(close)
    df["momentum_10"] = close / close.shift(10) - 1

    # -- Volatility bands --
    df["bb_pct_b"] = _bollinger_pct_b(close)
    df["atr_14"] = _atr(high, low, close)

    # -- Volume --
    df["volume_z"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    df["obv_z"] = (
        (_obv(close, volume) - _obv(close, volume).rolling(20).mean())
        / (_obv(close, volume).rolling(20).std() + 1e-9)
    )

    # -- Forward targets: log return at each horizon --
    for h in HORIZONS:
        df[f"target_return_{h}d"] = np.log(close.shift(-h) / close)

    return df


# The ~13 features actually fed to the models. Kept deliberately short:
# each one earns its place (trend, momentum, volatility, volume — no
# redundant duplicates of the same signal).
FEATURE_COLUMNS = [
    "log_return_1d",
    "volatility_20d",
    "price_to_sma_20",
    "price_to_sma_50",
    "rsi_14",
    "macd",
    "macd_hist",
    "momentum_10",
    "bb_pct_b",
    "atr_14",
    "volume_z",
    "obv_z",
]


def build_feature_table(raw_path: str = "data/raw/stocks.csv",
                         out_path: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Read raw OHLCV, engineer features per ticker, drop warm-up NaN rows,
    and persist the combined feature table."""
    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    processed = [add_technical_features(g) for _, g in raw.groupby("Ticker")]
    full = pd.concat(processed, ignore_index=True)

    required = FEATURE_COLUMNS + [f"target_return_{h}d" for h in HORIZONS]
    full = full.dropna(subset=required)

    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"Feature table: {full.shape} -> {out_path}")
    return full


if __name__ == "__main__":
    build_feature_table()
