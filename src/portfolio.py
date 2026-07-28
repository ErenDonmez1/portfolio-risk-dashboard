"""Portfolio weighting and aggregation helpers for the Portfolio Risk Dashboard."""

import math

import pandas as pd


def validate_weights(weights: dict, tickers: list) -> None:
    """Validate that weights are complete, finite, non-negative and total 1.

    Portfolio weights describe how much of the portfolio is invested in each
    asset. For example, a weight of 0.60 means 60% of the money is invested in
    that ticker.
    """
    missing_tickers = [ticker for ticker in tickers if ticker not in weights]
    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise ValueError(f"Missing weight(s) for ticker(s): {missing}")

    selected_weights = [weights[ticker] for ticker in tickers]
    try:
        all_finite = all(math.isfinite(weight) for weight in selected_weights)
    except TypeError as error:
        raise ValueError("Portfolio weights must be finite numbers") from error

    if not all_finite:
        raise ValueError("Portfolio weights must be finite numbers")
    if any(weight < 0 for weight in selected_weights):
        raise ValueError("Portfolio weights must not be negative")

    total_weight = sum(selected_weights)
    if abs(total_weight - 1.0) > 0.000001:
        raise ValueError(f"Portfolio weights must sum to 1. Current sum: {total_weight}")


def calculate_portfolio_returns(
    returns_df: pd.DataFrame, weights: dict
) -> pd.Series:
    """Calculate portfolio returns as the weighted sum of asset returns.

    Diversification means spreading money across different assets instead of
    relying on one ticker. The portfolio return combines each asset's return
    based on how much money was allocated to it.
    """
    tickers = list(returns_df.columns)
    validate_weights(weights, tickers)

    ordered_weights = pd.Series(
        [weights[ticker] for ticker in tickers],
        index=tickers,
    )
    portfolio_returns = returns_df.mul(ordered_weights, axis="columns").sum(axis=1)
    portfolio_returns.name = "Portfolio"

    return portfolio_returns


def calculate_portfolio_cumulative_returns(
    portfolio_returns: pd.Series,
) -> pd.Series:
    """Calculate compounded portfolio returns over time.

    Cumulative portfolio return shows how much the whole portfolio has gained
    or lost since the start, after each period's return builds on the previous
    portfolio value.
    """
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    return cumulative_returns
