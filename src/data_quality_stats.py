"""Statistical diagnostics used by the Data Quality Lab."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DISTRIBUTION_SHIFT_NOTE = (
    "This is an automated diagnostic warning, not proof of a structural market "
    "change or data problem."
)


def calculate_daily_percentage_returns(
    price_df: pd.DataFrame,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    price_column: str = "Close",
) -> pd.DataFrame:
    """Calculate sequential percentage returns for each asset without filling gaps."""
    _require_columns(price_df, {date_column, asset_column, price_column})
    working_data = price_df.loc[
        :, [date_column, asset_column, price_column]
    ].copy()
    working_data[date_column] = pd.to_datetime(
        working_data[date_column], errors="coerce"
    )
    working_data[price_column] = pd.to_numeric(
        working_data[price_column], errors="coerce"
    )
    working_data = working_data.dropna(
        subset=[date_column, asset_column, price_column]
    )
    working_data = working_data.drop_duplicates(
        subset=[date_column, asset_column], keep="last"
    ).sort_values([asset_column, date_column])

    working_data["Return"] = working_data.groupby(asset_column)[
        price_column
    ].pct_change(fill_method=None)
    working_data["Return"] = working_data["Return"].replace(
        [np.inf, -np.inf], np.nan
    )

    return (
        working_data.dropna(subset=["Return"])
        .loc[:, [date_column, asset_column, "Return"]]
        .sort_values([date_column, asset_column])
        .reset_index(drop=True)
    )


def detect_extreme_returns(
    returns_df: pd.DataFrame,
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    asset_column: str = "Ticker",
    return_column: str = "Return",
) -> pd.DataFrame:
    """Flag returns that cross configurable z-score or IQR thresholds."""
    if z_score_threshold <= 0:
        raise ValueError("z_score_threshold must be greater than 0")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than 0")
    _require_columns(returns_df, {asset_column, return_column})

    flagged_groups: list[pd.DataFrame] = []
    for _, ticker_returns in returns_df.groupby(asset_column, sort=True):
        ticker_returns = ticker_returns.copy()
        values = pd.to_numeric(ticker_returns[return_column], errors="coerce")
        mean_return = values.mean()
        standard_deviation = values.std(ddof=0)

        if pd.isna(standard_deviation) or standard_deviation == 0:
            z_scores = pd.Series(0.0, index=ticker_returns.index)
        else:
            z_scores = (values - mean_return) / standard_deviation

        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        iqr = third_quartile - first_quartile
        iqr_lower = first_quartile - iqr_multiplier * iqr
        iqr_upper = third_quartile + iqr_multiplier * iqr

        ticker_returns["Z-Score"] = z_scores
        ticker_returns["IQR Lower Bound"] = iqr_lower
        ticker_returns["IQR Upper Bound"] = iqr_upper
        ticker_returns["Z-Score Flag"] = z_scores.abs() > z_score_threshold
        ticker_returns["IQR Flag"] = (values < iqr_lower) | (values > iqr_upper)
        ticker_returns["Detection Method"] = ticker_returns.apply(
            _describe_extreme_return_method, axis=1
        )
        flagged_groups.append(
            ticker_returns[
                ticker_returns["Z-Score Flag"] | ticker_returns["IQR Flag"]
            ]
        )

    if not flagged_groups:
        return _empty_extreme_returns_frame(returns_df)
    return pd.concat(flagged_groups, ignore_index=True).reset_index(drop=True)


def identify_insufficient_observations(
    price_df: pd.DataFrame,
    minimum_observations: int = 30,
    asset_column: str = "Ticker",
    price_column: str = "Close",
) -> pd.DataFrame:
    """Identify assets with fewer valid prices than the chosen minimum."""
    if minimum_observations <= 1:
        raise ValueError("minimum_observations must be greater than 1")
    _require_columns(price_df, {asset_column, price_column})

    numeric_prices = pd.to_numeric(price_df[price_column], errors="coerce")
    observation_counts = (
        price_df.assign(_NumericPrice=numeric_prices)
        .groupby(asset_column)["_NumericPrice"]
        .count()
        .rename("Valid Observations")
        .reset_index()
    )
    observation_counts["Minimum Required"] = minimum_observations
    return observation_counts[
        observation_counts["Valid Observations"] < minimum_observations
    ].reset_index(drop=True)


def split_return_series_halves(
    returns_df: pd.DataFrame,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    return_column: str = "Return",
) -> pd.DataFrame:
    """Label each asset return as belonging to its first or second half."""
    _require_columns(returns_df, {date_column, asset_column, return_column})
    labelled_groups: list[pd.DataFrame] = []

    for _, ticker_returns in returns_df.groupby(asset_column, sort=True):
        ticker_returns = ticker_returns.sort_values(date_column).copy()
        midpoint = len(ticker_returns) // 2
        ticker_returns["Period"] = "Second half"
        ticker_returns.iloc[
            :midpoint, ticker_returns.columns.get_loc("Period")
        ] = "First half"
        labelled_groups.append(ticker_returns)

    if not labelled_groups:
        return pd.DataFrame(
            columns=[date_column, asset_column, return_column, "Period"]
        )
    return pd.concat(labelled_groups, ignore_index=True)


def compare_return_distributions(
    returns_df: pd.DataFrame,
    significance_level: float = 0.05,
    minimum_half_observations: int = 3,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    return_column: str = "Return",
) -> pd.DataFrame:
    """Compare the first and second halves of each asset's return history.

    The KS test is a diagnostic prompt for further research, not proof of a
    structural change.
    """
    if significance_level <= 0 or significance_level >= 1:
        raise ValueError("significance_level must be between 0 and 1")
    if minimum_half_observations < 2:
        raise ValueError("minimum_half_observations must be at least 2")
    _require_columns(returns_df, {date_column, asset_column, return_column})

    comparisons: list[dict[str, Any]] = []
    for ticker, ticker_returns in returns_df.groupby(asset_column, sort=True):
        values = (
            ticker_returns.sort_values(date_column)[return_column]
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]
        enough_data = (
            len(first_half) >= minimum_half_observations
            and len(second_half) >= minimum_half_observations
        )

        if enough_data:
            result = ks_2samp(first_half, second_half, alternative="two-sided")
            ks_statistic = float(result.statistic)
            ks_p_value = float(result.pvalue)
            first_mean = float(np.mean(first_half))
            second_mean = float(np.mean(second_half))
            first_std = float(np.std(first_half, ddof=1))
            second_std = float(np.std(second_half, ddof=1))
            warning = bool(ks_p_value < significance_level)
            interpretation = (
                "The distributions differ at the selected diagnostic threshold. "
                + DISTRIBUTION_SHIFT_NOTE
                if warning
                else "No distribution difference was flagged at the selected "
                "threshold. This does not prove the series is stable."
            )
        else:
            ks_statistic = np.nan
            ks_p_value = np.nan
            first_mean = float(np.mean(first_half)) if len(first_half) else np.nan
            second_mean = (
                float(np.mean(second_half)) if len(second_half) else np.nan
            )
            first_std = (
                float(np.std(first_half, ddof=1)) if len(first_half) > 1 else np.nan
            )
            second_std = (
                float(np.std(second_half, ddof=1))
                if len(second_half) > 1
                else np.nan
            )
            warning = False
            interpretation = (
                "Too few observations for a reliable two-sample diagnostic. "
                + DISTRIBUTION_SHIFT_NOTE
            )

        comparisons.append(
            {
                "Ticker": ticker,
                "First Half Observations": int(len(first_half)),
                "Second Half Observations": int(len(second_half)),
                "First Half Mean": first_mean,
                "Second Half Mean": second_mean,
                "First Half Std": first_std,
                "Second Half Std": second_std,
                "KS Statistic": ks_statistic,
                "KS P-Value": ks_p_value,
                "Diagnostic Warning": warning,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(comparisons)


def calculate_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 20,
    trading_days: int = 252,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    return_column: str = "Return",
) -> pd.DataFrame:
    """Calculate annualised rolling volatility for each asset return series."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if trading_days <= 0:
        raise ValueError("trading_days must be greater than 0")
    _require_columns(returns_df, {date_column, asset_column, return_column})

    rolling_groups: list[pd.DataFrame] = []
    for _, ticker_returns in returns_df.groupby(asset_column, sort=True):
        ticker_returns = ticker_returns.sort_values(date_column).copy()
        ticker_returns["Rolling Volatility"] = (
            ticker_returns[return_column].rolling(window=window).std()
            * np.sqrt(trading_days)
        )
        rolling_groups.append(ticker_returns)

    if not rolling_groups:
        return pd.DataFrame(
            columns=[date_column, asset_column, "Rolling Volatility"]
        )
    return (
        pd.concat(rolling_groups, ignore_index=True)
        .dropna(subset=["Rolling Volatility"])
        .loc[:, [date_column, asset_column, "Rolling Volatility"]]
    )


def _require_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def _describe_extreme_return_method(row: pd.Series) -> str:
    if row["Z-Score Flag"] and row["IQR Flag"]:
        return "Z-score and IQR"
    if row["Z-Score Flag"]:
        return "Z-score"
    return "IQR"


def _empty_extreme_returns_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(returns_df.columns) + [
        "Z-Score",
        "IQR Lower Bound",
        "IQR Upper Bound",
        "Z-Score Flag",
        "IQR Flag",
        "Detection Method",
    ]
    return pd.DataFrame(columns=columns)
