from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict

import pandas as pd

from sp500_selector.data import fetch_sp500_constituents, download_price_history
from sp500_selector.factors import FactorConfig, compute_factors_for_date
from sp500_selector.rank import Weights, rank_stocks


def parse_weights(arg: str | None) -> Dict[str, float]:
    if not arg:
        return {}
    result: Dict[str, float] = {}
    for item in arg.split(","):
        if not item:
            continue
        key, val = item.split("=")
        result[key.strip()] = float(val)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top S&P 500 stocks by composite factor ranking")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top stocks to output")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD). Default: ~3 years before end")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--min-history-days", type=int, default=252, help="Minimum required history per ticker")
    parser.add_argument("--output", type=str, default="sp500_ranked.csv", help="CSV path for ranked output")
    parser.add_argument("--save-prices", type=str, default=None, help="Optional CSV path to save price panel")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Custom weights like momentum=0.4,sharpe=0.3,low_vol=0.15,low_dd=0.15",
    )
    parser.add_argument("--limit-tickers", type=int, default=None, help="Optional cap on number of tickers for testing")

    args = parser.parse_args()

    end_dt = dt.date.today() if not args.end else dt.date.fromisoformat(args.end)
    start_dt = end_dt - dt.timedelta(days=3 * 365) if not args.start else dt.date.fromisoformat(args.start)

    print("Fetching S&P 500 constituents…")
    tickers = fetch_sp500_constituents()
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"Using {len(tickers)} tickers")

    print("Downloading price history…")
    prices = download_price_history(
        tickers=tickers,
        start=start_dt,
        end=end_dt,
        min_history_days=args.min_history_days,
        batch_size=50,
        max_retries=2,
        progress=True,
    )

    if args.save_prices:
        print(f"Saving price panel to {args.save_prices}")
        prices.to_csv(args.save_prices, index=True)

    print("Computing factors…")
    factors = compute_factors_for_date(prices=prices, as_of=end_dt, config=FactorConfig())

    print("Ranking stocks…")
    weights = Weights(**parse_weights(args.weights)) if args.weights else Weights()
    ranked = rank_stocks(factors, weights=weights)

    print(f"Saving ranked results to {args.output}")
    ranked.to_csv(args.output, index=True)

    print("Top selections:")
    display_cols = [
        "momentum_12_1",
        "sharpe_1y",
        "volatility_6m",
        "max_drawdown_1y",
        "composite_score",
    ]
    print(ranked.head(args.top_n)[display_cols].round(4).to_string())


if __name__ == "__main__":
    main()