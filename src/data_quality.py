"""Data validation checks and orchestration for the Data Quality Lab."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_quality_report import (
    build_asset_quality_table,
    build_validation_report,
    calculate_data_quality_score,
    generate_data_quality_warnings,
)
from src.data_quality_stats import (
    DISTRIBUTION_SHIFT_NOTE,
    calculate_daily_percentage_returns,
    calculate_rolling_volatility,
    compare_return_distributions,
    detect_extreme_returns,
    identify_insufficient_observations,
    split_return_series_halves,
)


REQUIRED_PRICE_COLUMNS = {"Date", "Ticker", "Close"}

__all__ = [
    "DISTRIBUTION_SHIFT_NOTE",
    "build_asset_quality_table",
    "build_validation_report",
    "calculate_daily_percentage_returns",
    "calculate_data_quality_score",
    "calculate_rolling_volatility",
    "check_missing_values_by_column",
    "compare_return_distributions",
    "detect_duplicate_dates",
    "detect_extreme_returns",
    "detect_irregular_date_gaps",
    "detect_non_numeric_prices",
    "detect_stale_prices",
    "generate_data_quality_warnings",
    "identify_insufficient_observations",
    "run_data_quality_analysis",
    "split_return_series_halves",
    "validate_price_dataset",
]


def validate_price_dataset(price_df: pd.DataFrame) -> None:
    """Validate the minimum structure needed for price-data diagnostics."""
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("Price data must be provided as a pandas DataFrame.")
    if price_df.empty:
        raise ValueError("The uploaded dataset is empty.")

    missing_columns = REQUIRED_PRICE_COLUMNS - set(price_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        if "Date" in missing_columns:
            if len(missing_columns) > 1:
                others = ", ".join(sorted(missing_columns - {"Date"}))
                raise ValueError(
                    f"Missing required date column: Date. Also missing: {others}"
                )
            raise ValueError("Missing required date column: Date.")
        raise ValueError(f"Missing required column(s): {missing}")

    converted_dates = pd.to_datetime(price_df["Date"], errors="coerce")
    invalid_dates = int((price_df["Date"].notna() & converted_dates.isna()).sum())
    if invalid_dates:
        raise ValueError(
            f"Date contains {invalid_dates} value(s) that could not be parsed."
        )


def check_missing_values_by_column(price_df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values and their percentage for every dataset column."""
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("Price data must be provided as a pandas DataFrame.")

    row_count = len(price_df)
    missing_counts = price_df.isna().sum()
    missing_percent = (
        missing_counts / row_count
        if row_count
        else pd.Series(0.0, index=price_df.columns)
    )
    return pd.DataFrame(
        {
            "Column": price_df.columns,
            "Missing Values": missing_counts.to_numpy(dtype=int),
            "Missing Percent": missing_percent.to_numpy(dtype=float),
        }
    )


def detect_duplicate_dates(
    price_df: pd.DataFrame,
    date_column: str = "Date",
    asset_column: str = "Ticker",
) -> pd.DataFrame:
    """Find repeated asset-date pairs in long-format price data."""
    _require_columns(price_df, {date_column, asset_column})
    working_data = price_df.loc[:, [date_column, asset_column]].copy()
    working_data[date_column] = pd.to_datetime(
        working_data[date_column], errors="coerce"
    )
    duplicate_counts = (
        working_data.dropna(subset=[date_column, asset_column])
        .groupby([asset_column, date_column], as_index=False)
        .size()
        .rename(columns={"size": "Duplicate Count"})
    )
    duplicate_counts = duplicate_counts[duplicate_counts["Duplicate Count"] > 1]
    return duplicate_counts.loc[
        :, [date_column, asset_column, "Duplicate Count"]
    ].sort_values([date_column, asset_column]).reset_index(drop=True)


def detect_non_numeric_prices(
    price_df: pd.DataFrame,
    price_column: str = "Close",
) -> pd.DataFrame:
    """Return rows containing non-empty prices that cannot be read as numbers."""
    _require_columns(price_df, {price_column})
    numeric_prices = pd.to_numeric(price_df[price_column], errors="coerce")
    invalid_mask = price_df[price_column].notna() & numeric_prices.isna()
    invalid_rows = price_df.loc[invalid_mask].copy()
    invalid_rows.insert(0, "Row", invalid_rows.index)
    invalid_rows = invalid_rows.rename(columns={price_column: "Invalid Price"})
    preferred_columns = [
        column
        for column in ["Row", "Date", "Ticker", "Invalid Price"]
        if column in invalid_rows.columns
    ]
    return invalid_rows.loc[:, preferred_columns].reset_index(drop=True)


def detect_stale_prices(
    price_df: pd.DataFrame,
    consecutive_observations: int = 3,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    price_column: str = "Close",
) -> pd.DataFrame:
    """Find runs where an asset has the same price for several observations."""
    if consecutive_observations < 2:
        raise ValueError("consecutive_observations must be at least 2")
    _require_columns(price_df, {date_column, asset_column, price_column})

    working_data = _prepare_numeric_prices(
        price_df, date_column, asset_column, price_column
    )
    stale_runs: list[dict[str, Any]] = []

    for ticker, ticker_data in working_data.groupby(asset_column, sort=True):
        ticker_data = ticker_data.sort_values(date_column).reset_index(drop=True)
        price_changed = ticker_data[price_column].ne(
            ticker_data[price_column].shift()
        )
        missing_price = ticker_data[price_column].isna()
        run_ids = (price_changed | missing_price).cumsum()

        for _, run in ticker_data.groupby(run_ids):
            if run[price_column].notna().all() and len(run) >= consecutive_observations:
                stale_runs.append(
                    {
                        "Ticker": ticker,
                        "Start Date": run[date_column].iloc[0],
                        "End Date": run[date_column].iloc[-1],
                        "Price": float(run[price_column].iloc[0]),
                        "Stale Observations": int(len(run)),
                    }
                )

    return pd.DataFrame(
        stale_runs,
        columns=[
            "Ticker",
            "Start Date",
            "End Date",
            "Price",
            "Stale Observations",
        ],
    )


def detect_irregular_date_gaps(
    price_df: pd.DataFrame,
    gap_multiplier: float = 3.0,
    date_column: str = "Date",
    asset_column: str = "Ticker",
) -> pd.DataFrame:
    """Find date gaps that are unusually long for each asset's own history."""
    if gap_multiplier <= 1:
        raise ValueError("gap_multiplier must be greater than 1")
    _require_columns(price_df, {date_column, asset_column})

    working_data = price_df.loc[:, [date_column, asset_column]].copy()
    working_data[date_column] = pd.to_datetime(
        working_data[date_column], errors="coerce"
    )
    working_data = working_data.dropna().drop_duplicates()
    irregular_gaps: list[dict[str, Any]] = []

    for ticker, ticker_data in working_data.groupby(asset_column, sort=True):
        dates = ticker_data[date_column].sort_values().drop_duplicates()
        gaps = dates.diff().dt.days.dropna()
        positive_gaps = gaps[gaps > 0]
        if positive_gaps.empty:
            continue

        expected_gap = float(positive_gaps.median())
        threshold = expected_gap * gap_multiplier
        previous_dates = dates.shift()
        for index, gap_days in gaps[gaps > threshold].items():
            irregular_gaps.append(
                {
                    "Ticker": ticker,
                    "Previous Date": previous_dates.loc[index],
                    "Current Date": dates.loc[index],
                    "Gap Days": int(gap_days),
                    "Expected Gap Days": expected_gap,
                }
            )

    return pd.DataFrame(
        irregular_gaps,
        columns=[
            "Ticker",
            "Previous Date",
            "Current Date",
            "Gap Days",
            "Expected Gap Days",
        ],
    )


def run_data_quality_analysis(
    price_df: pd.DataFrame,
    stale_observations: int = 3,
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    minimum_observations: int = 30,
    gap_multiplier: float = 3.0,
    rolling_window: int = 20,
) -> dict[str, Any]:
    """Run all diagnostics and return tables and summaries for the UI."""
    validate_price_dataset(price_df)
    analysed_data = price_df.copy()
    analysed_data["Date"] = pd.to_datetime(analysed_data["Date"], errors="coerce")

    missing_values = check_missing_values_by_column(analysed_data)
    duplicate_dates = detect_duplicate_dates(analysed_data)
    non_numeric_prices = detect_non_numeric_prices(analysed_data)
    stale_prices = detect_stale_prices(
        analysed_data, consecutive_observations=stale_observations
    )
    returns = calculate_daily_percentage_returns(analysed_data)
    extreme_returns = detect_extreme_returns(
        returns,
        z_score_threshold=z_score_threshold,
        iqr_multiplier=iqr_multiplier,
    )
    insufficient_observations = identify_insufficient_observations(
        analysed_data, minimum_observations=minimum_observations
    )
    irregular_gaps = detect_irregular_date_gaps(
        analysed_data, gap_multiplier=gap_multiplier
    )
    distribution_shift = compare_return_distributions(returns)
    return_halves = split_return_series_halves(returns)
    rolling_volatility = calculate_rolling_volatility(
        returns, window=rolling_window
    )

    diagnostics = {
        "missing_values": missing_values,
        "duplicate_dates": duplicate_dates,
        "non_numeric_prices": non_numeric_prices,
        "stale_prices": stale_prices,
        "extreme_returns": extreme_returns,
        "insufficient_observations": insufficient_observations,
        "irregular_gaps": irregular_gaps,
        "distribution_shift": distribution_shift,
    }
    asset_quality = build_asset_quality_table(
        analysed_data, diagnostics, minimum_observations
    )
    score = calculate_data_quality_score(analysed_data, returns, diagnostics)
    warnings = generate_data_quality_warnings(analysed_data, returns, diagnostics)

    return {
        **diagnostics,
        "returns": returns,
        "return_halves": return_halves,
        "rolling_volatility": rolling_volatility,
        "asset_quality": asset_quality,
        "score": score,
        "warnings": warnings,
        "validation_report": build_validation_report(
            score, warnings, asset_quality
        ),
        "row_count": len(analysed_data),
        "column_count": len(analysed_data.columns),
        "asset_count": analysed_data["Ticker"].nunique(dropna=True),
        "start_date": analysed_data["Date"].min(),
        "end_date": analysed_data["Date"].max(),
    }


def _require_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def _prepare_numeric_prices(
    price_df: pd.DataFrame,
    date_column: str,
    asset_column: str,
    price_column: str,
) -> pd.DataFrame:
    working_data = price_df.loc[
        :, [date_column, asset_column, price_column]
    ].copy()
    working_data[date_column] = pd.to_datetime(
        working_data[date_column], errors="coerce"
    )
    working_data[price_column] = pd.to_numeric(
        working_data[price_column], errors="coerce"
    )
    return working_data
