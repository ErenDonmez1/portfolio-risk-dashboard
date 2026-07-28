"""Scoring and reporting helpers for Data Quality Lab diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_quality_stats import DISTRIBUTION_SHIFT_NOTE


def build_asset_quality_table(
    price_df: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    minimum_observations: int,
) -> pd.DataFrame:
    """Summarise issue counts and review status for every asset."""
    _require_columns(price_df, {"Ticker", "Close"})
    tickers = sorted(price_df["Ticker"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    numeric_prices = pd.to_numeric(price_df["Close"], errors="coerce")

    for ticker in tickers:
        ticker_mask = price_df["Ticker"].astype(str) == ticker
        ticker_prices = price_df.loc[ticker_mask, "Close"]
        ticker_numeric = numeric_prices.loc[ticker_mask]
        valid_observations = int(ticker_numeric.notna().sum())
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
            + int(valid_observations < minimum_observations)
        )

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
    """Calculate the existing weighted screening score from 0 to 100."""
    _require_columns(price_df, {"Ticker", "Close"})
    if price_df.empty:
        raise ValueError("The uploaded dataset is empty.")

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

    distribution_shift = diagnostics["distribution_shift"]
    shift_count = (
        int(distribution_shift["Diagnostic Warning"].sum())
        if "Diagnostic Warning" in distribution_shift
        else 0
    )
    if shift_count:
        warnings.append(
            _warning(
                "Research",
                "Potential distribution shift",
                f"{shift_count} asset(s) crossed the KS diagnostic threshold.",
                "Changing return behaviour can make full-sample estimates less representative.",
                "Review sub-periods, market events, and alternative tests. "
                + DISTRIBUTION_SHIFT_NOTE,
            )
        )

    if price_df["Ticker"].nunique(dropna=True) == 1:
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


def _require_columns(data: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def _count_for_ticker(data: pd.DataFrame, ticker: str) -> int:
    if data.empty or "Ticker" not in data.columns:
        return 0
    return int((data["Ticker"].astype(str) == ticker).sum())


def _sum_for_ticker(
    data: pd.DataFrame,
    ticker: str,
    value_column: str,
    subtract_one: bool = False,
) -> int:
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
