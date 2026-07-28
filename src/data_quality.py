"""Reusable data-quality diagnostics for long-format market price data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {"Date", "Ticker", "Close"}
DISTRIBUTION_SHIFT_NOTE = (
    "This is an automated diagnostic warning, not proof of a structural market "
    "change or data problem."
)


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
            raise ValueError(
                "Missing required date column: Date. "
                f"Also missing: {', '.join(sorted(missing_columns - {'Date'}))}"
                if len(missing_columns) > 1
                else "Missing required date column: Date."
            )
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
    if row_count == 0:
        missing_percent = pd.Series(0.0, index=price_df.columns)
    else:
        missing_percent = missing_counts / row_count

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
    """Find repeated asset-date pairs in long-format price data.

    A date appearing once for each ticker is expected. A duplicate exists when
    the same ticker has more than one observation for the same date.
    """
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


def calculate_daily_percentage_returns(
    price_df: pd.DataFrame,
    date_column: str = "Date",
    asset_column: str = "Ticker",
    price_column: str = "Close",
) -> pd.DataFrame:
    """Calculate sequential daily percentage returns for each asset.

    The function keeps returns in long format so each observation retains its
    date and ticker. Invalid or missing prices are not silently filled.
    """
    _require_columns(price_df, {date_column, asset_column, price_column})
    working_data = _prepare_numeric_prices(
        price_df, date_column, asset_column, price_column
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
    """Flag unusually large returns using z-score and IQR diagnostics.

    These rules identify observations for review. Large returns can be genuine
    market moves, corporate actions, or data errors, so a flag is not a finding
    of corruption.
    """
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

    flagged_returns = pd.concat(flagged_groups, ignore_index=True)
    return flagged_returns.reset_index(drop=True)


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


def detect_irregular_date_gaps(
    price_df: pd.DataFrame,
    gap_multiplier: float = 3.0,
    date_column: str = "Date",
    asset_column: str = "Ticker",
) -> pd.DataFrame:
    """Find date gaps that are unusually long for each asset's own history.

    The expected spacing is the median positive calendar-day gap for an asset.
    A gap is flagged only when it is greater than the configured multiple, so
    normal weekend gaps in daily market data are not automatically treated as
    errors.
    """
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

    Means and standard deviations describe location and dispersion. The
    two-sample Kolmogorov-Smirnov test compares the full empirical
    distributions. A low p-value is a diagnostic prompt for further research,
    not proof of a regime change.
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
            ks_statistic, ks_p_value = _kolmogorov_smirnov_test(
                first_half, second_half
            )
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


def build_asset_quality_table(
    price_df: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    minimum_observations: int,
) -> pd.DataFrame:
    """Summarise data-quality issue counts for every asset."""
    _require_columns(price_df, {"Ticker", "Close"})
    tickers = sorted(price_df["Ticker"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []

    numeric_prices = pd.to_numeric(price_df["Close"], errors="coerce")
    for ticker in tickers:
        ticker_mask = price_df["Ticker"].astype(str) == ticker
        ticker_prices = price_df.loc[ticker_mask, "Close"]
        ticker_numeric = numeric_prices.loc[ticker_mask]
        duplicate_rows = _sum_for_ticker(
            diagnostics["duplicate_dates"],
            ticker,
            "Duplicate Count",
            subtract_one=True,
        )
        issue_count = (
            int(ticker_prices.isna().sum())
            + int((ticker_prices.notna() & ticker_numeric.isna()).sum())
            + duplicate_rows
            + _count_for_ticker(diagnostics["stale_prices"], ticker)
            + _count_for_ticker(diagnostics["extreme_returns"], ticker)
            + _count_for_ticker(diagnostics["irregular_gaps"], ticker)
        )
        valid_observations = int(ticker_numeric.notna().sum())

        rows.append(
            {
                "Ticker": ticker,
                "Rows": int(ticker_mask.sum()),
                "Valid Prices": valid_observations,
                "Missing Prices": int(ticker_prices.isna().sum()),
                "Non-Numeric Prices": int(
                    (ticker_prices.notna() & ticker_numeric.isna()).sum()
                ),
                "Duplicate Rows": duplicate_rows,
                "Stale Runs": _count_for_ticker(
                    diagnostics["stale_prices"], ticker
                ),
                "Extreme Returns": _count_for_ticker(
                    diagnostics["extreme_returns"], ticker
                ),
                "Irregular Gaps": _count_for_ticker(
                    diagnostics["irregular_gaps"], ticker
                ),
                "Observation Status": (
                    "Insufficient"
                    if valid_observations < minimum_observations
                    else "Sufficient"
                ),
                "Review Status": "Review" if issue_count else "Pass",
            }
        )

    return pd.DataFrame(rows)


def calculate_data_quality_score(
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
) -> float:
    """Calculate a transparent weighted data-quality score from 0 to 100.

    The score applies capped deductions for missing data, duplicate rows,
    non-numeric prices, stale runs, extreme returns, insufficient histories,
    irregular date gaps and distribution-shift warnings. It is a screening
    summary rather than a guarantee that a dataset is correct.
    """
    validate_price_dataset(price_df)
    row_count = max(len(price_df), 1)
    asset_count = max(price_df["Ticker"].nunique(dropna=True), 1)
    return_count = max(len(returns_df), 1)

    missing_count = int(price_df.isna().sum().sum())
    possible_cells = max(price_df.size, 1)
    duplicate_rows = int(
        (diagnostics["duplicate_dates"]["Duplicate Count"] - 1).clip(lower=0).sum()
    )
    non_numeric_count = len(diagnostics["non_numeric_prices"])
    stale_observations = int(
        diagnostics["stale_prices"].get(
            "Stale Observations", pd.Series(dtype=float)
        ).sum()
    )
    extreme_count = len(diagnostics["extreme_returns"])
    insufficient_count = len(diagnostics["insufficient_observations"])
    gap_count = len(diagnostics["irregular_gaps"])
    shift_count = int(
        diagnostics["distribution_shift"]
        .get("Diagnostic Warning", pd.Series(dtype=bool))
        .sum()
    )

    penalties = [
        20 * min(missing_count / possible_cells, 1.0),
        15 * min(duplicate_rows / row_count, 1.0),
        15 * min(non_numeric_count / row_count, 1.0),
        15 * min(stale_observations / row_count, 1.0),
        10 * min(extreme_count / return_count, 1.0),
        10 * min(insufficient_count / asset_count, 1.0),
        10 * min(gap_count / row_count, 1.0),
        5 * min(shift_count / asset_count, 1.0),
    ]
    return round(max(0.0, min(100.0, 100.0 - sum(penalties))), 1)


def generate_data_quality_warnings(
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create plain-English warnings and recommended research actions."""
    warnings: list[dict[str, str]] = []

    missing_count = int(price_df.isna().sum().sum())
    if missing_count:
        warnings.append(
            _warning(
                "High",
                "Missing values",
                f"{missing_count} missing cell(s) were found.",
                "Gaps can remove return observations or bias comparisons.",
                "Inspect the source, then document whether to remove, replace, "
                "or leave each missing value.",
            )
        )

    _append_count_warning(
        warnings,
        diagnostics["duplicate_dates"],
        "High",
        "Duplicate asset dates",
        "Repeated ticker-date observations can double-count a market period.",
        "Check the source key and retain one verified observation per ticker and date.",
    )
    _append_count_warning(
        warnings,
        diagnostics["non_numeric_prices"],
        "High",
        "Non-numeric prices",
        "Text in a price column prevents reliable return calculations.",
        "Correct the source values or exclude the affected rows with an audit note.",
    )
    _append_count_warning(
        warnings,
        diagnostics["stale_prices"],
        "Medium",
        "Stale price runs",
        "Repeated prices may reflect illiquidity, market closures, or stale feeds.",
        "Compare the flagged period with the original source and trading calendar.",
    )
    _append_count_warning(
        warnings,
        diagnostics["extreme_returns"],
        "Medium",
        "Extreme returns",
        "Outliers can dominate volatility, correlation, and tail-risk estimates.",
        "Check corporate actions, unit changes, and source prices before deciding "
        "whether an observation is valid.",
    )
    _append_count_warning(
        warnings,
        diagnostics["insufficient_observations"],
        "Medium",
        "Insufficient observations",
        "Short histories make estimated risk statistics unstable.",
        "Use a longer comparable history or clearly limit the interpretation.",
    )
    _append_count_warning(
        warnings,
        diagnostics["irregular_gaps"],
        "Medium",
        "Irregular date gaps",
        "Unexpected gaps can make asset histories misaligned.",
        "Check the exchange calendar and source coverage before filling or removing gaps.",
    )

    shift_warnings = diagnostics["distribution_shift"]
    flagged_shift_count = (
        int(shift_warnings["Diagnostic Warning"].sum())
        if "Diagnostic Warning" in shift_warnings
        else 0
    )
    if flagged_shift_count:
        warnings.append(
            _warning(
                "Research",
                "Potential distribution shift",
                f"{flagged_shift_count} asset(s) crossed the KS diagnostic threshold.",
                "Changing return behaviour can make full-sample estimates less representative.",
                "Review sub-periods, market events, and alternative tests. "
                + DISTRIBUTION_SHIFT_NOTE,
            )
        )

    asset_count = price_df["Ticker"].nunique(dropna=True)
    if asset_count == 1:
        warnings.append(
            _warning(
                "Information",
                "Single-asset dataset",
                "Only one asset is present.",
                "Cross-asset correlation and diversification checks are unavailable.",
                "Continue with single-asset diagnostics or add comparable assets.",
            )
        )

    if returns_df.empty:
        warnings.append(
            _warning(
                "High",
                "Too few valid returns",
                "No valid daily returns could be calculated.",
                "Return-based diagnostics need at least two valid prices per asset.",
                "Add valid chronological price observations before interpreting risk.",
            )
        )

    return pd.DataFrame(
        warnings,
        columns=[
            "Severity",
            "Issue",
            "Evidence",
            "Why It Matters",
            "Recommended Action",
        ],
    )


def build_validation_report(
    score: float,
    warnings: pd.DataFrame,
    asset_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Build a flat validation report suitable for CSV download."""
    records: list[dict[str, Any]] = [
        {
            "Record Type": "Overall",
            "Ticker": "",
            "Severity": "Diagnostic",
            "Issue": "Data quality score",
            "Value": score,
            "Why It Matters": (
                "A weighted screening score summarises automated checks."
            ),
            "Recommended Action": (
                "Review the detailed flags before drawing research conclusions."
            ),
        }
    ]

    for _, warning_row in warnings.iterrows():
        records.append(
            {
                "Record Type": "Warning",
                "Ticker": "",
                "Severity": warning_row["Severity"],
                "Issue": warning_row["Issue"],
                "Value": warning_row["Evidence"],
                "Why It Matters": warning_row["Why It Matters"],
                "Recommended Action": warning_row["Recommended Action"],
            }
        )

    for _, asset_row in asset_quality.iterrows():
        records.append(
            {
                "Record Type": "Asset summary",
                "Ticker": asset_row["Ticker"],
                "Severity": asset_row["Review Status"],
                "Issue": "Asset-level diagnostics",
                "Value": (
                    f"valid={asset_row['Valid Prices']}; "
                    f"missing={asset_row['Missing Prices']}; "
                    f"non_numeric={asset_row['Non-Numeric Prices']}; "
                    f"duplicates={asset_row['Duplicate Rows']}; "
                    f"stale_runs={asset_row['Stale Runs']}; "
                    f"extreme_returns={asset_row['Extreme Returns']}; "
                    f"irregular_gaps={asset_row['Irregular Gaps']}"
                ),
                "Why It Matters": "Issue counts help prioritise manual validation.",
                "Recommended Action": (
                    "Inspect flagged observations and record any cleaning decision."
                ),
            }
        )

    return pd.DataFrame(records)


def run_data_quality_analysis(
    price_df: pd.DataFrame,
    stale_observations: int = 3,
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    minimum_observations: int = 30,
    gap_multiplier: float = 3.0,
    rolling_window: int = 20,
) -> dict[str, Any]:
    """Run all diagnostics and return UI-ready tables and summary values."""
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
    warnings = generate_data_quality_warnings(
        analysed_data, returns, diagnostics
    )
    validation_report = build_validation_report(score, warnings, asset_quality)

    return {
        **diagnostics,
        "returns": returns,
        "return_halves": return_halves,
        "rolling_volatility": rolling_volatility,
        "asset_quality": asset_quality,
        "score": score,
        "warnings": warnings,
        "validation_report": validation_report,
        "row_count": len(analysed_data),
        "column_count": len(analysed_data.columns),
        "asset_count": analysed_data["Ticker"].nunique(dropna=True),
        "start_date": analysed_data["Date"].min(),
        "end_date": analysed_data["Date"].max(),
    }


def _require_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error when a diagnostic input is missing columns."""
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
    """Return selected price columns with parsed dates and numeric prices."""
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


def _describe_extreme_return_method(row: pd.Series) -> str:
    """Describe which outlier rule flagged a return observation."""
    if row["Z-Score Flag"] and row["IQR Flag"]:
        return "Z-score and IQR"
    if row["Z-Score Flag"]:
        return "Z-score"
    return "IQR"


def _empty_extreme_returns_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Create an empty result with the same schema as extreme-return output."""
    columns = list(returns_df.columns) + [
        "Z-Score",
        "IQR Lower Bound",
        "IQR Upper Bound",
        "Z-Score Flag",
        "IQR Flag",
        "Detection Method",
    ]
    return pd.DataFrame(columns=columns)


def _kolmogorov_smirnov_test(
    first_sample: np.ndarray,
    second_sample: np.ndarray,
) -> tuple[float, float]:
    """Run a two-sample KS test, with an asymptotic fallback if needed."""
    try:
        from scipy.stats import ks_2samp
    except ModuleNotFoundError:
        return _asymptotic_ks_2sample(first_sample, second_sample)

    result = ks_2samp(first_sample, second_sample, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _asymptotic_ks_2sample(
    first_sample: np.ndarray,
    second_sample: np.ndarray,
) -> tuple[float, float]:
    """Approximate a two-sample KS statistic and p-value without SciPy."""
    first_sorted = np.sort(first_sample)
    second_sorted = np.sort(second_sample)
    combined = np.sort(np.concatenate([first_sorted, second_sorted]))
    first_cdf = np.searchsorted(first_sorted, combined, side="right") / len(
        first_sorted
    )
    second_cdf = np.searchsorted(second_sorted, combined, side="right") / len(
        second_sorted
    )
    statistic = float(np.max(np.abs(first_cdf - second_cdf)))

    effective_n = np.sqrt(
        len(first_sorted)
        * len(second_sorted)
        / (len(first_sorted) + len(second_sorted))
    )
    adjusted = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    terms = [
        (-1) ** (term - 1) * np.exp(-2 * (term**2) * (adjusted**2))
        for term in range(1, 101)
    ]
    p_value = float(np.clip(2 * sum(terms), 0.0, 1.0))
    return statistic, p_value


def _count_for_ticker(data: pd.DataFrame, ticker: str) -> int:
    """Count diagnostic rows belonging to one ticker."""
    if data.empty or "Ticker" not in data.columns:
        return 0
    return int((data["Ticker"].astype(str) == ticker).sum())


def _sum_for_ticker(
    data: pd.DataFrame,
    ticker: str,
    value_column: str,
    subtract_one: bool = False,
) -> int:
    """Sum a numeric diagnostic field for one ticker."""
    if data.empty or "Ticker" not in data.columns or value_column not in data.columns:
        return 0
    values = data.loc[
        data["Ticker"].astype(str) == ticker, value_column
    ].astype(float)
    if subtract_one:
        values = (values - 1).clip(lower=0)
    return int(values.sum())


def _warning(
    severity: str,
    issue: str,
    evidence: str,
    why_it_matters: str,
    recommended_action: str,
) -> dict[str, str]:
    """Build one warning record with a consistent schema."""
    return {
        "Severity": severity,
        "Issue": issue,
        "Evidence": evidence,
        "Why It Matters": why_it_matters,
        "Recommended Action": recommended_action,
    }


def _append_count_warning(
    warnings: list[dict[str, str]],
    issue_data: pd.DataFrame,
    severity: str,
    issue: str,
    why_it_matters: str,
    recommended_action: str,
) -> None:
    """Append a warning when a diagnostic table contains flagged rows."""
    if issue_data.empty:
        return
    warnings.append(
        _warning(
            severity,
            issue,
            f"{len(issue_data)} flagged record(s) were found.",
            why_it_matters,
            recommended_action,
        )
    )
