from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from features import add_technical_features, FEATURE_COLUMNS
from dataset import SEQ_LEN


INDICATOR_WARMUP_DAYS = 70
# Calendar-day buffer to comfortably cover weekends/holidays for the
# trading-day lookback above.
CALENDAR_DAY_BUFFER = 1.6


def fetch_live_history(ticker: str, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Downloads recent daily OHLCV for one ticker from Yahoo Finance and
    returns it with all technical indicators computed, ready for charting
    or windowing. Raises a clear error if the network call fails or the
    ticker returns no data, rather than silently falling back to anything
    stale or synthetic."""
    lookback_days = int((seq_len + INDICATOR_WARMUP_DAYS) * CALENDAR_DAY_BUFFER)
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    try:
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        raise ConnectionError(
            f"Live data fetch for '{ticker}' failed: {e}. "
            "This dashboard requires an internet connection to Yahoo Finance "
            "at inference time -- no offline/synthetic fallback is used."
        ) from e

    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for ticker '{ticker}'.")

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw["Ticker"] = ticker

    return add_technical_features(raw)


def get_latest_window(ticker: str, seq_len: int = SEQ_LEN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full_history_with_indicators, latest_seq_len_window) for a
    ticker, fetched live. `full_history` is used for charting (price,
    indicators); `window` is exactly what gets fed to the model."""
    history = fetch_live_history(ticker, seq_len=seq_len)
    history = history.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if len(history) < seq_len:
        raise ValueError(
            f"Only {len(history)} valid rows of live data are available for "
            f"'{ticker}' after indicator warmup -- need at least {seq_len}. "
            "This can happen for a very recently listed ticker."
        )

    window = history.tail(seq_len).reset_index(drop=True)
    return history, window
