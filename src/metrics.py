"""Return and performance metric calculations for the Portfolio Risk Dashboard."""

import numpy as np
import pandas as pd


def calculate_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each ticker's daily percentage return from closing prices.

    A daily return measures how much a price changed from one trading day to the
    next. Missing prices are not filled, and the final table keeps only dates
    where every displayed asset has a valid return.
    """
    price_table = price_df.pivot(index="Date", columns="Ticker", values="Close")
    # Use complete rows only: a missing price invalidates that asset's return,
    # and the dashboard keeps dates where every displayed asset has a return.
    daily_returns = price_table.pct_change(fill_method=None).dropna()

    return daily_returns


def calculate_cumulative_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total compounded return over time from daily returns.

    Cumulative return shows how much an investment has gained or lost since the
    start of the period, assuming each day's return builds on previous days.
    """
    cumulative_returns = (1 + returns_df).cumprod() - 1

    return cumulative_returns


def calculate_annualised_volatility(
    returns_df: pd.DataFrame, trading_days: int = 252
) -> pd.Series:
    """Calculate annualised volatility for each ticker.

    Volatility measures how much daily returns move around. Higher volatility
    means the investment has had larger ups and downs, which is one way to
    describe risk.
    """
    annualised_volatility = returns_df.std() * np.sqrt(trading_days)

    return annualised_volatility


def calculate_drawdown(cumulative_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each return series' percentage fall from its previous peak.

    Drawdown shows how far an asset or portfolio has dropped from the best
    value it had reached so far. A drawdown of -0.20 means it is 20% below its
    previous peak.
    """
    portfolio_value = 1 + cumulative_returns_df
    previous_peak = portfolio_value.cummax()
    drawdown = (portfolio_value / previous_peak) - 1

    return drawdown


def calculate_max_drawdown(cumulative_returns_df: pd.DataFrame) -> pd.Series:
    """Calculate the worst drawdown for each return series.

    Max drawdown is the largest percentage fall from a previous peak during the
    period. It helps show the worst peak-to-trough decline an investor would
    have seen before a recovery.
    """
    drawdown = calculate_drawdown(cumulative_returns_df)
    max_drawdown = drawdown.min()

    return max_drawdown
