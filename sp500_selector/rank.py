from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Weights:
    momentum: float = 0.4
    sharpe: float = 0.3
    low_vol: float = 0.15
    low_dd: float = 0.15


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype=float)

    ranks = valid.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks.reindex(series.index)


def rank_stocks(factors: pd.DataFrame, weights: Weights | Dict[str, float] | None = None) -> pd.DataFrame:
    """Compute percentile ranks and a weighted composite score.

    Returns a DataFrame with per-factor percentile ranks, composite score, and sorted descending by score.
    """
    w = weights if isinstance(weights, Weights) else Weights(**(weights or {}))

    df = factors.copy()

    df["rank_momentum"] = _percentile_rank(df["momentum_12_1"], higher_is_better=True)
    df["rank_sharpe"] = _percentile_rank(df["sharpe_1y"], higher_is_better=True)
    df["rank_low_vol"] = _percentile_rank(df["volatility_6m"], higher_is_better=False)
    df["rank_low_dd"] = _percentile_rank(df["max_drawdown_1y"], higher_is_better=False)

    df["composite_score"] = (
        df["rank_momentum"].fillna(0) * w.momentum
        + df["rank_sharpe"].fillna(0) * w.sharpe
        + df["rank_low_vol"].fillna(0) * w.low_vol
        + df["rank_low_dd"].fillna(0) * w.low_dd
    )

    df = df.sort_values("composite_score", ascending=False)
    return df