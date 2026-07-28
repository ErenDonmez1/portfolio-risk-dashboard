"""Tests for modular market-data quality diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_quality import (
    build_asset_quality_table,
    build_validation_report,
    calculate_daily_percentage_returns,
    calculate_rolling_volatility,
    check_missing_values_by_column,
    compare_return_distributions,
    detect_duplicate_dates,
    detect_extreme_returns,
    detect_irregular_date_gaps,
    detect_non_numeric_prices,
    detect_stale_prices,
    generate_data_quality_warnings,
    identify_insufficient_observations,
    run_data_quality_analysis,
    split_return_series_halves,
    validate_price_dataset,
)


def make_quality_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-10",
                    "2026-01-11",
                    "2026-01-12",
                ]
            ),
            "Ticker": ["AAA"] * 6 + ["BBB"] * 6,
            "Close": [
                100.0,
                100.0,
                100.0,
                150.0,
                151.0,
                152.0,
                50.0,
                "bad",
                np.nan,
                53.0,
                54.0,
                55.0,
            ],
        }
    )


def make_clean_dataset() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Ticker": ["AAA"] * 10 + ["BBB"] * 10,
            "Close": (
                [100.0 + index for index in range(10)]
                + [200.0 + index * 2 for index in range(10)]
            ),
        }
    )


def test_validate_price_dataset_rejects_empty_and_missing_date_data():
    with pytest.raises(ValueError, match="dataset is empty"):
        validate_price_dataset(pd.DataFrame(columns=["Date", "Ticker", "Close"]))

    with pytest.raises(ValueError, match="Missing required date column"):
        validate_price_dataset(
            pd.DataFrame({"Ticker": ["AAA"], "Close": [100.0]})
        )


def test_check_missing_values_by_column_counts_missing_cells():
    data = make_quality_dataset()

    result = check_missing_values_by_column(data)
    close_row = result.set_index("Column").loc["Close"]

    assert close_row["Missing Values"] == 1
    assert close_row["Missing Percent"] == pytest.approx(1 / len(data))


def test_detect_duplicate_dates_checks_asset_date_pairs():
    result = detect_duplicate_dates(make_quality_dataset())

    assert len(result) == 1
    assert result.iloc[0]["Ticker"] == "AAA"
    assert result.iloc[0]["Date"] == pd.Timestamp("2026-01-05")
    assert result.iloc[0]["Duplicate Count"] == 2


def test_detect_non_numeric_prices_excludes_missing_values():
    result = detect_non_numeric_prices(make_quality_dataset())

    assert len(result) == 1
    assert result.iloc[0]["Ticker"] == "BBB"
    assert result.iloc[0]["Invalid Price"] == "bad"


def test_detect_stale_prices_finds_consecutive_unchanged_run():
    result = detect_stale_prices(
        make_quality_dataset(), consecutive_observations=3
    )

    assert len(result) == 1
    assert result.iloc[0]["Ticker"] == "AAA"
    assert result.iloc[0]["Price"] == 100.0
    assert result.iloc[0]["Stale Observations"] == 3


def test_calculate_daily_percentage_returns_uses_each_asset_history():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]
            ),
            "Ticker": ["AAA", "AAA", "BBB", "BBB"],
            "Close": [100.0, 110.0, 200.0, 190.0],
        }
    )

    result = calculate_daily_percentage_returns(data).set_index("Ticker")

    assert result.loc["AAA", "Return"] == pytest.approx(0.10)
    assert result.loc["BBB", "Return"] == pytest.approx(-0.05)


def test_detect_extreme_returns_supports_z_score_and_iqr_methods():
    returns = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=9, freq="D"),
            "Ticker": ["AAA"] * 9,
            "Return": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )

    result = detect_extreme_returns(
        returns, z_score_threshold=2.0, iqr_multiplier=1.5
    )

    assert len(result) == 1
    assert result.iloc[0]["Return"] == 1.0
    assert result.iloc[0]["Z-Score Flag"]
    assert result.iloc[0]["IQR Flag"]
    assert result.iloc[0]["Detection Method"] == "Z-score and IQR"


def test_identify_insufficient_observations_counts_valid_numeric_prices():
    result = identify_insufficient_observations(
        make_quality_dataset(), minimum_observations=6
    ).set_index("Ticker")

    assert "AAA" not in result.index
    assert result.loc["BBB", "Valid Observations"] == 4


def test_detect_irregular_date_gaps_uses_asset_specific_spacing():
    result = detect_irregular_date_gaps(
        make_quality_dataset(), gap_multiplier=3.0
    )

    assert len(result) == 1
    assert result.iloc[0]["Ticker"] == "BBB"
    assert result.iloc[0]["Gap Days"] == 7
    assert result.iloc[0]["Expected Gap Days"] == pytest.approx(1.0)


def test_compare_return_distributions_returns_ks_diagnostics():
    returns = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "Ticker": ["AAA"] * 12,
            "Return": [0.0] * 6 + [0.10] * 6,
        }
    )

    result = compare_return_distributions(returns)
    comparison = result.iloc[0]

    assert comparison["First Half Mean"] == pytest.approx(0.0)
    assert comparison["Second Half Mean"] == pytest.approx(0.10)
    assert comparison["KS Statistic"] == pytest.approx(1.0)
    assert 0.0 <= comparison["KS P-Value"] <= 1.0
    assert "not proof" in comparison["Interpretation"]


def test_split_returns_and_rolling_volatility_prepare_chart_data():
    returns = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "Ticker": ["AAA"] * 6,
            "Return": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        }
    )

    halves = split_return_series_halves(returns)
    rolling = calculate_rolling_volatility(returns, window=3)

    assert list(halves["Period"].value_counts().sort_index()) == [3, 3]
    assert len(rolling) == 4
    assert (rolling["Rolling Volatility"] >= 0).all()


def test_asset_table_score_and_warnings_distinguish_clean_from_dirty_data():
    dirty_report = run_data_quality_analysis(
        make_quality_dataset(),
        minimum_observations=6,
        rolling_window=3,
        z_score_threshold=2.0,
    )
    clean_report = run_data_quality_analysis(
        make_clean_dataset(),
        minimum_observations=5,
        rolling_window=3,
    )

    assert set(dirty_report["asset_quality"]["Ticker"]) == {"AAA", "BBB"}
    assert dirty_report["score"] < clean_report["score"]
    assert 0 <= dirty_report["score"] <= 100
    assert not dirty_report["warnings"].empty
    assert dirty_report["warnings"]["Recommended Action"].str.len().min() > 0


def test_insufficient_observations_require_asset_review():
    report = run_data_quality_analysis(
        make_clean_dataset(),
        minimum_observations=20,
        rolling_window=3,
    )
    asset = report["asset_quality"].set_index("Ticker").loc["AAA"]

    assert asset["Observation Status"] == "Insufficient"
    assert asset["Review Status"] == "Review"


def test_build_helpers_create_downloadable_validation_rows():
    data = make_clean_dataset()
    returns = calculate_daily_percentage_returns(data)
    empty_diagnostics = {
        "duplicate_dates": detect_duplicate_dates(data),
        "non_numeric_prices": detect_non_numeric_prices(data),
        "stale_prices": detect_stale_prices(data),
        "extreme_returns": detect_extreme_returns(returns),
        "insufficient_observations": identify_insufficient_observations(
            data, minimum_observations=5
        ),
        "irregular_gaps": detect_irregular_date_gaps(data),
        "distribution_shift": compare_return_distributions(returns),
    }

    asset_quality = build_asset_quality_table(
        data, empty_diagnostics, minimum_observations=5
    )
    warnings = generate_data_quality_warnings(data, returns, empty_diagnostics)
    validation_report = build_validation_report(100.0, warnings, asset_quality)

    assert validation_report.iloc[0]["Record Type"] == "Overall"
    assert (validation_report["Record Type"] == "Asset summary").sum() == 2


def test_single_asset_and_short_histories_return_diagnostics_without_crashing():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Ticker": ["ONLY", "ONLY"],
            "Close": [100.0, 101.0],
        }
    )

    report = run_data_quality_analysis(
        data, minimum_observations=5, rolling_window=2
    )

    assert report["asset_count"] == 1
    assert report["distribution_shift"].iloc[0]["KS P-Value"] != report[
        "distribution_shift"
    ].iloc[0]["KS P-Value"]
    assert "Single-asset dataset" in set(report["warnings"]["Issue"])
