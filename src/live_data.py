import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

from features import add_technical_features, FEATURE_COLUMNS
from dataset import SEQ_LEN

INDICATOR_WARMUP_DAYS = 70

CALENDAR_DAY_BUFFER = 1.6

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
TRAINING_SNAPSHOT_PATH = ROOT / "data" / "raw" / "stocks.csv"

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.5


class LiveDataUnavailable(Exception):
    """Raised only when live, cached, AND training-snapshot data are all
    unavailable for a ticker -- i.e. there's truly nothing to show."""


# live fetch

def _make_session():
    """A curl_cffi session impersonating a real Chrome browser's TLS
    fingerprint. Yahoo's anti-bot detection fingerprints the HTTP client,
    and plain `requests`/urllib traffic is increasingly likely to be
    blocked outright -- this is the workaround yfinance's own maintainers
    currently recommend."""
    return curl_requests.Session(impersonate="chrome")


def _fetch_live(ticker: str, seq_len: int) -> pd.DataFrame:
    """Single attempt at a live fetch. Returns a DataFrame with indicators
    computed, or raises if Yahoo returns nothing."""
    lookback_days = int((seq_len + INDICATOR_WARMUP_DAYS) * CALENDAR_DAY_BUFFER)
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    session = _make_session()
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True, session=session)

    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for ticker '{ticker}'.")

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw["Ticker"] = ticker

    return add_technical_features(raw)


def _fetch_live_with_retry(ticker: str, seq_len: int, attempts: int = RETRY_ATTEMPTS) -> pd.DataFrame:
    """Retries transient failures (rate limits, momentary network blips)
    with exponential backoff before giving up on the live tier entirely."""
    last_error = None
    for attempt in range(attempts):
        try:
            return _fetch_live(ticker, seq_len)
        except Exception as e: 
            last_error = e
            if attempt < attempts - 1:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    raise ConnectionError(f"Live fetch for '{ticker}' failed after {attempts} attempts: {last_error}")



# on-disk cache of the last successful live fetch

def _cache_paths(ticker: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (CACHE_DIR / f"{ticker}.parquet", CACHE_DIR / f"{ticker}.meta.json")


def _save_to_cache(ticker: str, history: pd.DataFrame) -> None:
    data_path, meta_path = _cache_paths(ticker)
    try:
        history.to_parquet(data_path, index=False)
        meta_path.write_text(json.dumps({"fetched_at": datetime.utcnow().isoformat() + "Z"}))
    except Exception:
        pass


def _load_from_cache(ticker: str):
    data_path, meta_path = _cache_paths(ticker)
    if not data_path.exists():
        return None, None
    try:
        history = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return history, meta.get("fetched_at")
    except Exception:
        return None, None


# static training snapshot

def _load_from_training_snapshot(ticker: str):
    if not TRAINING_SNAPSHOT_PATH.exists():
        return None, None
    try:
        raw = pd.read_csv(TRAINING_SNAPSHOT_PATH, parse_dates=["Date"])
        raw = raw[raw["Ticker"] == ticker]
        if raw.empty:
            return None, None
        history = add_technical_features(raw)
        as_of = history["Date"].max()
        return history, as_of.date().isoformat() if pd.notna(as_of) else None
    except Exception:
        return None, None


# Public entry point

def get_latest_window(ticker: str, seq_len: int = SEQ_LEN):
    # Tier 1: live
    try:
        history = _fetch_live_with_retry(ticker, seq_len)
        _save_to_cache(ticker, history)
        meta = {"source": "live", "as_of": datetime.utcnow().isoformat() + "Z"}
    except Exception as live_error:
        # Tier 2: cache
        history, cached_at = _load_from_cache(ticker)
        if history is not None:
            meta = {"source": "cache", "as_of": cached_at, "error": str(live_error)}
        else:
            # Tier 3: training snapshot
            history, snapshot_as_of = _load_from_training_snapshot(ticker)
            if history is not None:
                meta = {"source": "snapshot", "as_of": snapshot_as_of, "error": str(live_error)}
            else:
                raise LiveDataUnavailable(
                    f"No live, cached, or training-snapshot data available for '{ticker}'. "
                    f"Live fetch error: {live_error}"
                ) from live_error

    history = history.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(history) < seq_len:
        raise LiveDataUnavailable(
            f"Only {len(history)} valid rows of data are available for '{ticker}' "
            f"after indicator warmup (source: {meta['source']}) -- need at least {seq_len}."
        )

    window = history.tail(seq_len).reset_index(drop=True)
    return history, window, meta