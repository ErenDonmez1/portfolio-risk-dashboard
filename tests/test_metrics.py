"""Tests for return and performance metric calculations."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    calculate_annualised_volatility,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_drawdown,
    calculate_max_drawdown,
)


def test_calculate_daily_returns_from_simple_prices():
    """Calculate easy-to-check daily percentage returns for each ticker."""
    price_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                ]
            ),
            "Ticker": ["ALPHA", "BETA", "ALPHA", "BETA", "ALPHA", "BETA"],
            "Close": [100.0, 200.0, 110.0, 180.0, 121.0, 198.0],
        }
    )

    result = calculate_daily_returns(price_data)

    expected = pd.DataFrame(
        {
            "ALPHA": [0.10, 0.10],
            "BETA": [-0.10, 0.10],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    expected.index.name = "Date"
    expected.columns.name = "Ticker"

    assert_frame_equal(result, expected)


def test_calculate_daily_returns_from_hand_calculated_prices():
    """Use 100 -> 110 -> 121 to prove two 10% daily returns by hand."""
    price_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Ticker": ["ALPHA", "ALPHA", "ALPHA"],
            "Close": [100.0, 110.0, 121.0],
        }
    )

    result = calculate_daily_returns(price_data)

    # 110 / 100 - 1 = 0.10, then 121 / 110 - 1 = 0.10.
    expected_daily_returns = [0.10, 0.10]

    assert result["ALPHA"].tolist() == pytest.approx(expected_daily_returns)


def test_calculate_daily_returns_does_not_fill_missing_prices():
    price_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "Ticker": ["ALPHA", "ALPHA", "ALPHA"],
            "Close": [100.0, None, 110.0],
        }
    )

    result = calculate_daily_returns(price_data)

    assert result.empty


def test_calculate_cumulative_returns_from_simple_returns():
    """Compound daily returns into cumulative returns over time."""
    returns_data = pd.DataFrame(
        {
            "ALPHA": [0.10, 0.10],
            "BETA": [-0.10, 0.10],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    returns_data.index.name = "Date"

    result = calculate_cumulative_returns(returns_data)

    expected = pd.DataFrame(
        {
            "ALPHA": [0.10, 0.21],
            "BETA": [-0.10, -0.01],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    expected.index.name = "Date"

    assert_frame_equal(result, expected)


def test_calculate_cumulative_returns_final_value_from_hand_calculated_returns():
    """Compound two 10% returns to prove the final cumulative return is 21%."""
    daily_returns = pd.DataFrame({"ALPHA": [0.10, 0.10]})

    result = calculate_cumulative_returns(daily_returns)

    # (1.10 * 1.10) - 1 = 0.21.
    expected_final_cumulative_return = 0.21

    assert result["ALPHA"].iloc[-1] == pytest.approx(
        expected_final_cumulative_return
    )


def test_calculate_annualised_volatility_from_simple_returns():
    """Calculate volatility from easy-to-check daily returns."""
    returns_data = pd.DataFrame(
        {
            "ALPHA": [0.10, 0.20, 0.30],
            "BETA": [-0.10, 0.00, 0.10],
        }
    )

    result = calculate_annualised_volatility(returns_data, trading_days=1)

    expected = pd.Series({"ALPHA": 0.10, "BETA": 0.10})

    assert_series_equal(result, expected)


def test_calculate_drawdown_from_simple_cumulative_returns():
    """Calculate percentage falls from each ticker's previous peak."""
    cumulative_returns = pd.DataFrame(
        {
            "ALPHA": [0.00, 0.20, 0.10, 0.30],
            "BETA": [0.00, 0.50, 0.20, 0.60],
        }
    )

    result = calculate_drawdown(cumulative_returns)

    expected = pd.DataFrame(
        {
            "ALPHA": [0.00, 0.00, -0.08333333333333337, 0.00],
            "BETA": [0.00, 0.00, -0.19999999999999996, 0.00],
        }
    )

    assert_frame_equal(result, expected)


def test_calculate_drawdown_from_hand_calculated_wealth_path():
    """Use wealth 1.0 -> 1.1 -> 0.99 -> 1.188 to prove a 10% drawdown."""
    cumulative_returns = pd.DataFrame({"ALPHA": [0.00, 0.10, -0.01, 0.188]})

    result = calculate_drawdown(cumulative_returns)

    # The running peak is 1.10, so 0.99 / 1.10 - 1 = -0.10.
    expected_drawdown_at_trough = -0.10

    assert result["ALPHA"].iloc[2] == pytest.approx(expected_drawdown_at_trough)


def test_calculate_max_drawdown_from_simple_cumulative_returns():
    """Return the worst drawdown for each ticker."""
    cumulative_returns = pd.DataFrame(
        {
            "ALPHA": [0.00, 0.20, 0.10, 0.30],
            "BETA": [0.00, 0.50, 0.20, 0.60],
        }
    )

    result = calculate_max_drawdown(cumulative_returns)

    expected = pd.Series(
        {
            "ALPHA": -0.08333333333333337,
            "BETA": -0.19999999999999996,
        }
    )

    assert_series_equal(result, expected)
