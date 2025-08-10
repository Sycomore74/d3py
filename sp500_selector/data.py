from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


def fetch_sp500_constituents(session: requests.Session | None = None) -> List[str]:
    """Fetch the current list of S&P 500 tickers from Wikipedia.

    Returns yfinance-compatible tickers (e.g., "BRK-B" instead of "BRK.B").
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    session = session or requests.Session()
    response = session.get(url, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(response.text)
    if not tables:
        raise RuntimeError("Failed to parse S&P 500 table from Wikipedia")

    table = tables[0]
    if "Symbol" not in table.columns:
        raise RuntimeError("Unexpected Wikipedia table format: 'Symbol' column missing")

    raw = table["Symbol"].astype(str).str.strip().tolist()

    # Wikipedia uses dots for class shares; yfinance expects dashes.
    tickers = [ticker.replace(".", "-") for ticker in raw]

    # Remove known non-standard entries if any stray values leak in
    tickers = [t for t in tickers if t.upper() == t and len(t) > 0]
    return tickers


def _chunked(iterable: List[str], chunk_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def download_price_history(
    tickers: List[str],
    start: dt.date | str,
    end: dt.date | str | None = None,
    min_history_days: int = 252,
    batch_size: int = 50,
    max_retries: int = 2,
    progress: bool = True,
) -> pd.DataFrame:
    """Download auto-adjusted daily close prices for many tickers.

    Returns a DataFrame indexed by date with columns as tickers. Drops tickers
    with less than `min_history_days` of available data in the requested window.
    """
    if isinstance(start, dt.date):
        start_str = start.isoformat()
    else:
        start_str = start

    if end is None:
        end_dt = dt.date.today()
    elif isinstance(end, dt.date):
        end_dt = end
    else:
        end_dt = dt.date.fromisoformat(end)

    all_prices: Dict[str, pd.Series] = {}

    batches = list(_chunked(tickers, batch_size))
    iterator = tqdm(batches, disable=not progress, desc="Downloading price batches")

    for batch in iterator:
        attempt = 0
        success = False
        last_error: Exception | None = None
        while attempt <= max_retries and not success:
            try:
                data = yf.download(
                    tickers=batch,
                    start=start_str,
                    end=end_dt.isoformat(),
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    actions=False,
                )
                success = True
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempt += 1
                if attempt > max_retries:
                    raise

        # Normalize the returned structure. For multiple tickers yfinance returns
        # a column MultiIndex (Ticker, Field). We want 'Close' or 'Adj Close'.
        if isinstance(data.columns, pd.MultiIndex):
            # Prefer 'Close' because auto_adjust=True already adjusted it.
            if ("Close" in data.columns.get_level_values(1)):
                field = "Close"
            elif ("Adj Close" in data.columns.get_level_values(1)):
                field = "Adj Close"
            else:
                field = None

            if field is not None:
                for ticker in batch:
                    try:
                        series = data[(ticker, field)].rename(ticker).dropna()
                        all_prices[ticker] = series
                    except KeyError:
                        continue
        else:
            # Single ticker case; 'Close' or 'Adj Close' direct columns
            col = "Close" if "Close" in data.columns else ("Adj Close" if "Adj Close" in data.columns else None)
            if col is not None:
                # If only one ticker in batch, yfinance uses that ticker name
                ticker = batch[0]
                series = data[col].rename(ticker).dropna()
                all_prices[ticker] = series

    if not all_prices:
        raise RuntimeError("No price data was downloaded")

    prices = pd.concat(all_prices.values(), axis=1)
    prices = prices.sort_index()

    # Filter tickers with insufficient data
    sufficient = prices.notna().sum(axis=0) >= min_history_days
    prices = prices.loc[:, sufficient]

    # Forward-fill occasional missing days per ticker to align panel, then drop any
    # leading rows that are still NaN for some tickers
    prices = prices.ffill().dropna(how="all")

    return prices