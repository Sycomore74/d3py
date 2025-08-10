from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class FactorConfig:
    momentum_lookback_days: int = 252  # 12 months
    momentum_skip_recent_days: int = 21  # exclude most recent month
    volatility_lookback_days: int = 126  # 6 months
    sharpe_lookback_days: int = 252  # 12 months
    drawdown_lookback_days: int = 252  # 12 months


def _window(prices: pd.Series, end_date: pd.Timestamp, lookback_days: int) -> pd.Series:
    start_date = end_date - pd.tseries.offsets.BDay(lookback_days)
    return prices.loc[start_date:end_date]


def _safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def compute_factors_for_date(
    prices: pd.DataFrame,
    as_of: dt.date | str | pd.Timestamp | None = None,
    config: Optional[FactorConfig] = None,
) -> pd.DataFrame:
    """Compute price-based factors for each ticker as of a given date.

    Returns a DataFrame indexed by ticker with columns:
    - momentum_12_1
    - sharpe_1y
    - volatility_6m
    - max_drawdown_1y
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    if as_of is None:
        end_ts = pd.Timestamp(prices.index.max())
    elif isinstance(as_of, pd.Timestamp):
        end_ts = as_of
    elif isinstance(as_of, str):
        end_ts = pd.Timestamp(as_of)
    else:
        end_ts = pd.Timestamp(as_of)

    cfg = config or FactorConfig()

    factors: Dict[str, Dict[str, float]] = {}

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue

        if series.index.max() < end_ts:
            # If latest data precedes end_ts, try to include as much as possible
            end_use = series.index.max()
        else:
            end_use = end_ts

        # Momentum 12-1
        mom_end = end_use - pd.tseries.offsets.BDay(cfg.momentum_skip_recent_days)
        mom_start = mom_end - pd.tseries.offsets.BDay(cfg.momentum_lookback_days)
        mom_window = series.loc[mom_start:mom_end]
        if len(mom_window) >= int(0.7 * cfg.momentum_lookback_days):
            momentum = float(mom_window.iloc[-1] / mom_window.iloc[0] - 1.0)
        else:
            momentum = np.nan

        # Volatility 6m
        vol_window = _window(series, end_use, cfg.volatility_lookback_days)
        daily_returns_6m = _safe_pct_change(vol_window)
        volatility = float(daily_returns_6m.std(ddof=0)) if len(daily_returns_6m) > 2 else np.nan

        # Sharpe 1y (simple, using daily mean/std)
        sharpe_window = _window(series, end_use, cfg.sharpe_lookback_days)
        daily_returns_1y = _safe_pct_change(sharpe_window)
        if len(daily_returns_1y) > 2 and daily_returns_1y.std(ddof=0) > 0:
            sharpe = float(
                (daily_returns_1y.mean() / daily_returns_1y.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
        else:
            sharpe = np.nan

        # Max drawdown 1y
        dd_window = _window(series, end_use, cfg.drawdown_lookback_days)
        if len(dd_window) > 10:
            running_max = dd_window.cummax()
            drawdowns = dd_window / running_max - 1.0
            max_drawdown = float(drawdowns.min())
        else:
            max_drawdown = np.nan

        factors[ticker] = {
            "momentum_12_1": momentum,
            "sharpe_1y": sharpe,
            "volatility_6m": volatility,
            "max_drawdown_1y": max_drawdown,
        }

    result = pd.DataFrame.from_dict(factors, orient="index")
    return result