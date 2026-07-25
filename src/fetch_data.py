"""
fetch_data.py
--------------
Downloads real daily OHLCV history for 20 popular, liquid US stocks from
2015-01-01 to the present day via `yfinance`, and saves it in the schema
used throughout the rest of the pipeline:

    Date, Open, High, Low, Close, Volume, Ticker

Run: python src/fetch_data.py

NOTE ON THIS SANDBOX: this development environment has no outbound network
access to Yahoo Finance (only a small allow-list of package registries is
reachable), so this script cannot be executed here to produce live data.
It has been written and structured to run correctly in any normal internet-
connected environment (your laptop, Colab, a CI runner, etc.) --
`pip install -r requirements.txt` and then `python src/fetch_data.py` is
all that's needed there.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# 20 popular, liquid large-cap tickers spanning multiple sectors, so the
# model sees varied volatility/trend regimes rather than one sector's
# idiosyncrasies.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",       # Tech / mega-cap growth
    "META", "TSLA", "NFLX", "AMD", "CRM",           # Tech / consumer internet
    "JPM", "BAC", "GS",                              # Financials
    "WMT", "TGT", "COST",                            # Retail / consumer staples
    "XOM", "CVX",                                    # Energy
    "JNJ", "PFE",                                    # Healthcare
    "DIS",                                            # Media
]

START_DATE = "2015-01-01"


def fetch_all(tickers=TICKERS, start=START_DATE, out_path="data/raw/stocks.csv",
              retries=3, pause=2.0) -> pd.DataFrame:
    """Downloads each ticker individually (rather than one bulk multi-ticker
    call) so a single failed/delisted ticker doesn't abort the whole run,
    and retries transient failures a few times."""
    frames = []
    for ticker in tickers:
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
                if df.empty:
                    raise ValueError("empty response")
                df = df.reset_index()
                # yfinance sometimes returns a MultiIndex column header even
                # for a single ticker -- flatten it defensively.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df["Ticker"] = ticker
                frames.append(df)
                print(f"[{ticker}] {len(df)} rows fetched")
                break
            except Exception as e:
                print(f"[{ticker}] attempt {attempt}/{retries} failed: {e}")
                if attempt == retries:
                    print(f"[{ticker}] SKIPPED after {retries} failed attempts")
                else:
                    time.sleep(pause)

    if not frames:
        raise RuntimeError(
            "No tickers were fetched successfully. Check your internet "
            "connection -- this script requires access to Yahoo Finance."
        )

    full = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"\nSaved {len(full)} rows across {full['Ticker'].nunique()} tickers -> {out_path}")
    return full


if __name__ == "__main__":
    fetch_all()
