"""Tests for portfolio weighting and aggregation calculations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import (
    calculate_portfolio_cumulative_returns,
    calculate_portfolio_returns,
    validate_weights,
)


def test_validate_weights_accepts_complete_weights_that_sum_to_one():
    """Accept weights when every ticker is included and the total is 100%."""
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    result = validate_weights(weights, ["ALPHA", "BETA"])

    assert result is None


def test_validate_weights_raises_value_error_for_missing_weight():
    """Raise a clear error when a ticker has no portfolio weight."""
    weights = {"ALPHA": 1.00}

    with pytest.raises(ValueError, match="Missing weight\\(s\\) for ticker\\(s\\): BETA"):
        validate_weights(weights, ["ALPHA", "BETA"])


def test_validate_weights_raises_value_error_when_weights_do_not_sum_to_one():
    """Raise a clear error when weights do not add up to 100% of the portfolio."""
    weights = {"ALPHA": 0.60, "BETA": 0.30}

    with pytest.raises(ValueError, match="Portfolio weights must sum to 1"):
        validate_weights(weights, ["ALPHA", "BETA"])


def test_validate_weights_rejects_negative_values():
    with pytest.raises(ValueError, match="must not be negative"):
        validate_weights({"ALPHA": 1.10, "BETA": -0.10}, ["ALPHA", "BETA"])


@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, -np.inf])
def test_validate_weights_rejects_non_finite_values(invalid_weight):
    with pytest.raises(ValueError, match="must be finite numbers"):
        validate_weights(
            {"ALPHA": invalid_weight, "BETA": 0.0},
            ["ALPHA", "BETA"],
        )


def test_calculate_portfolio_returns_from_simple_weights():
    """Calculate weighted portfolio returns from simple asset returns."""
    returns_data = pd.DataFrame(
        {
            "ALPHA": [0.10, 0.00],
            "BETA": [0.00, 0.20],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    result = calculate_portfolio_returns(returns_data, weights)

    expected = pd.Series(
        [0.06, 0.08],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        name="Portfolio",
    )

    assert_series_equal(result, expected)


def test_calculate_portfolio_return_from_hand_calculated_weighted_assets():
    """Use 0.6 * 10% + 0.4 * -5% to prove a 4% portfolio return."""
    asset_returns = pd.DataFrame({"ALPHA": [0.10], "BETA": [-0.05]})
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    result = calculate_portfolio_returns(asset_returns, weights)

    # 0.60 * 0.10 + 0.40 * -0.05 = 0.04.
    expected_portfolio_return = 0.04

    assert result.iloc[0] == pytest.approx(expected_portfolio_return)


def test_calculate_portfolio_cumulative_returns_from_simple_returns():
    """Compound portfolio returns over time."""
    portfolio_returns = pd.Series(
        [0.06, 0.08],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        name="Portfolio",
    )

    result = calculate_portfolio_cumulative_returns(portfolio_returns)

    expected = pd.Series(
        [0.06, 0.1448],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        name="Portfolio",
    )

    assert_series_equal(result, expected)


def test_calculate_portfolio_cumulative_return_from_hand_calculated_returns():
    """Compound two 4% returns to prove the final cumulative return is 8.16%."""
    portfolio_returns = pd.Series([0.04, 0.04], name="Portfolio")

    result = calculate_portfolio_cumulative_returns(portfolio_returns)

    # (1.04 * 1.04) - 1 = 0.0816.
    expected_final_cumulative_return = 0.0816

    assert result.iloc[-1] == pytest.approx(expected_final_cumulative_return)
