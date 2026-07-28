"""Tests for risk calculation functions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pandas.testing import assert_frame_equal

from src.risk import (
    calculate_historical_var,
    calculate_simulation_summary,
    calculate_stress_test_loss,
    run_correlated_monte_carlo_simulation,
    run_monte_carlo_simulation,
)


def make_two_asset_returns() -> pd.DataFrame:
    """Create a tiny deterministic return history for simulation tests."""
    return pd.DataFrame(
        {
            "ALPHA": [0.01, -0.02, 0.03, 0.00],
            "BETA": [0.00, 0.01, -0.01, 0.02],
        }
    )


def test_calculate_historical_var_uses_lower_tail_percentile():
    """Calculate 95% historical VaR from simple known portfolio returns."""
    portfolio_returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    result = calculate_historical_var(portfolio_returns, confidence_level=0.95)

    assert result == pytest.approx(0.09)


def test_calculate_historical_var_returns_positive_hand_calculated_loss():
    """Use NumPy's percentile convention to prove VaR is a positive loss."""
    portfolio_returns = pd.Series([-0.20, -0.10, 0.00, 0.10, 0.20])

    result = calculate_historical_var(portfolio_returns, confidence_level=0.95)

    # NumPy's 5th percentile is -0.18, and VaR is reported as a positive loss.
    expected_var_loss = 0.18

    assert result == pytest.approx(expected_var_loss)


def test_calculate_historical_var_accepts_different_confidence_level():
    """Calculate historical VaR using the percentile implied by confidence."""
    portfolio_returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    result = calculate_historical_var(portfolio_returns, confidence_level=0.80)

    assert result == pytest.approx(0.06)


def test_calculate_historical_var_does_not_return_negative_loss():
    """Return zero VaR when the selected percentile is still a gain."""
    portfolio_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

    result = calculate_historical_var(portfolio_returns, confidence_level=0.95)

    assert result == 0.0


def test_calculate_historical_var_raises_value_error_for_invalid_confidence():
    """Raise a clear error when confidence level is outside the valid range."""
    portfolio_returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        calculate_historical_var(portfolio_returns, confidence_level=1.00)


def test_calculate_historical_var_rejects_empty_returns():
    with pytest.raises(ValueError, match="at least one finite value"):
        calculate_historical_var(pd.Series(dtype=float))


def test_calculate_historical_var_rejects_returns_without_finite_values():
    with pytest.raises(ValueError, match="at least one finite value"):
        calculate_historical_var(pd.Series([float("nan"), float("inf")]))


def test_calculate_historical_var_drops_isolated_non_finite_values():
    result = calculate_historical_var(
        pd.Series([-0.10, -0.05, 0.05, float("nan"), float("inf")])
    )

    assert result == pytest.approx(0.095)


def test_calculate_stress_test_loss_from_simple_weighted_shock():
    """Calculate known weighted stress impact and stressed portfolio value."""
    weights = {"ALPHA": 0.60, "BETA": 0.40}
    shocks = {"ALPHA": -0.10, "BETA": -0.20}

    result = calculate_stress_test_loss(weights, shocks, portfolio_value=10000)

    assert result["portfolio_impact_percent"] == pytest.approx(-0.14)
    assert result["portfolio_impact_value"] == pytest.approx(-1400.0)
    assert result["stressed_portfolio_value"] == pytest.approx(8600.0)


def test_calculate_stress_test_loss_from_hand_calculated_shocks():
    """Use 0.6 * -10% + 0.4 * -5% to prove an 8% portfolio stress loss."""
    weights = {"ALPHA": 0.60, "BETA": 0.40}
    shocks = {"ALPHA": -0.10, "BETA": -0.05}

    result = calculate_stress_test_loss(weights, shocks, portfolio_value=10000)

    # 0.60 * -0.10 + 0.40 * -0.05 = -0.08.
    expected_impact_percent = -0.08
    expected_value_impact = -800.0
    expected_stressed_value = 9200.0

    assert result["portfolio_impact_percent"] == pytest.approx(
        expected_impact_percent
    )
    assert result["portfolio_impact_value"] == pytest.approx(expected_value_impact)
    assert result["stressed_portfolio_value"] == pytest.approx(expected_stressed_value)


def test_calculate_stress_test_loss_raises_value_error_for_missing_shock():
    """Raise a clear error when any ticker is missing a shock assumption."""
    weights = {"ALPHA": 0.60, "BETA": 0.40}
    shocks = {"ALPHA": -0.10}

    with pytest.raises(ValueError, match="Missing shock\\(s\\) for ticker\\(s\\): BETA"):
        calculate_stress_test_loss(weights, shocks)


def test_calculate_stress_test_loss_raises_value_error_for_invalid_value():
    """Raise a clear error when the portfolio value is not positive."""
    weights = {"ALPHA": 1.00}
    shocks = {"ALPHA": -0.10}

    with pytest.raises(ValueError, match="portfolio_value must be greater than 0"):
        calculate_stress_test_loss(weights, shocks, portfolio_value=0)


def test_run_monte_carlo_simulation_returns_expected_shape():
    """Return one column per simulation and one extra row for the start value."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    result = run_monte_carlo_simulation(
        portfolio_returns,
        num_simulations=3,
        num_days=5,
        initial_value=1000,
    )

    assert result.shape == (6, 3)


def test_run_monte_carlo_simulation_first_row_equals_initial_value():
    """Start every simulation path at the same initial portfolio value."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    result = run_monte_carlo_simulation(
        portfolio_returns,
        num_simulations=3,
        num_days=5,
        initial_value=1000,
    )

    assert result.iloc[0].tolist() == [1000, 1000, 1000]


def test_run_monte_carlo_simulation_same_seed_gives_same_result():
    """Use the random seed to make simulations reproducible in tests."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    first_result = run_monte_carlo_simulation(
        portfolio_returns,
        num_simulations=3,
        num_days=5,
        random_seed=7,
    )
    second_result = run_monte_carlo_simulation(
        portfolio_returns,
        num_simulations=3,
        num_days=5,
        random_seed=7,
    )

    assert_frame_equal(first_result, second_result)


def test_run_monte_carlo_simulation_raises_value_error_for_empty_returns():
    """Raise a clear error when there are no historical returns."""
    with pytest.raises(ValueError, match="portfolio_returns must not be empty"):
        run_monte_carlo_simulation(pd.Series(dtype=float))


def test_run_monte_carlo_simulation_raises_value_error_for_bad_simulation_count():
    """Raise a clear error when the requested simulation count is invalid."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
        run_monte_carlo_simulation(portfolio_returns, num_simulations=0)


def test_run_monte_carlo_simulation_raises_value_error_for_bad_day_count():
    """Raise a clear error when the requested number of future days is invalid."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    with pytest.raises(ValueError, match="num_days must be greater than 0"):
        run_monte_carlo_simulation(portfolio_returns, num_days=0)


def test_run_monte_carlo_simulation_raises_value_error_for_bad_initial_value():
    """Raise a clear error when the starting portfolio value is invalid."""
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    with pytest.raises(ValueError, match="initial_value must be greater than 0"):
        run_monte_carlo_simulation(portfolio_returns, initial_value=0)


def test_run_correlated_monte_carlo_simulation_returns_expected_shape():
    """Return one simulated path per column and include the starting row."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    result = run_correlated_monte_carlo_simulation(
        returns_df,
        weights,
        num_simulations=4,
        num_days=6,
        initial_value=1000,
    )

    assert result.shape == (7, 4)


def test_run_correlated_monte_carlo_simulation_first_row_equals_initial_value():
    """Start every correlated simulation at the same portfolio value."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    result = run_correlated_monte_carlo_simulation(
        returns_df,
        weights,
        num_simulations=3,
        num_days=5,
        initial_value=1000,
    )

    assert result.iloc[0].tolist() == [1000, 1000, 1000]


def test_run_correlated_monte_carlo_simulation_same_seed_gives_same_result():
    """Use the random seed to make correlated simulations reproducible."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    first_result = run_correlated_monte_carlo_simulation(
        returns_df,
        weights,
        num_simulations=3,
        num_days=5,
        random_seed=7,
    )
    second_result = run_correlated_monte_carlo_simulation(
        returns_df,
        weights,
        num_simulations=3,
        num_days=5,
        random_seed=7,
    )

    assert_frame_equal(first_result, second_result)


def test_run_correlated_monte_carlo_simulation_raises_for_missing_weight_ticker():
    """Raise a clear error when a weighted ticker is absent from returns data."""
    returns_df = pd.DataFrame({"ALPHA": [0.01, -0.02, 0.03, 0.00]})
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    with pytest.raises(ValueError, match="Missing return data for ticker\\(s\\): BETA"):
        run_correlated_monte_carlo_simulation(returns_df, weights)


def test_run_correlated_monte_carlo_simulation_raises_for_empty_weights():
    """Raise a clear error when no portfolio weights are supplied."""
    returns_df = make_two_asset_returns()

    with pytest.raises(ValueError, match="weights must not be empty"):
        run_correlated_monte_carlo_simulation(returns_df, {})


def test_run_correlated_monte_carlo_simulation_raises_for_bad_weight_total():
    """Raise a clear error when correlated simulation weights do not sum to 1."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.30}

    with pytest.raises(ValueError, match="Portfolio weights must sum to 1"):
        run_correlated_monte_carlo_simulation(returns_df, weights)


def test_run_correlated_monte_carlo_simulation_raises_for_bad_simulation_count():
    """Raise a clear error when correlated simulation count is invalid."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
        run_correlated_monte_carlo_simulation(
            returns_df,
            weights,
            num_simulations=0,
        )


def test_run_correlated_monte_carlo_simulation_raises_for_bad_day_count():
    """Raise a clear error when correlated simulation horizon is invalid."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    with pytest.raises(ValueError, match="num_days must be greater than 0"):
        run_correlated_monte_carlo_simulation(returns_df, weights, num_days=0)


def test_run_correlated_monte_carlo_simulation_raises_for_bad_initial_value():
    """Raise a clear error when correlated simulation starting value is invalid."""
    returns_df = make_two_asset_returns()
    weights = {"ALPHA": 0.60, "BETA": 0.40}

    with pytest.raises(ValueError, match="initial_value must be greater than 0"):
        run_correlated_monte_carlo_simulation(returns_df, weights, initial_value=0)


def test_calculate_simulation_summary_returns_expected_keys():
    """Return the dashboard summary fields for final simulated values."""
    simulation_df = pd.DataFrame(
        {
            "Simulation 1": [1000, 1050, 1100],
            "Simulation 2": [1000, 950, 900],
            "Simulation 3": [1000, 1020, 1030],
        }
    )

    result = calculate_simulation_summary(simulation_df, initial_value=1000)

    assert set(result) == {
        "median_final_value",
        "lower_percentile_final_value",
        "upper_percentile_final_value",
        "probability_of_loss",
        "simulated_var",
        "expected_shortfall",
    }


def test_calculate_simulation_summary_probability_of_loss_is_between_zero_and_one():
    """Return probability of loss as a valid share of simulations."""
    simulation_df = pd.DataFrame(
        {
            "Simulation 1": [1000, 1050, 1100],
            "Simulation 2": [1000, 950, 900],
            "Simulation 3": [1000, 1020, 1030],
        }
    )

    result = calculate_simulation_summary(simulation_df, initial_value=1000)

    assert 0 <= result["probability_of_loss"] <= 1


def test_calculate_simulation_summary_returns_non_negative_risk_losses():
    """Return simulated VaR and Expected Shortfall as non-negative losses."""
    simulation_df = pd.DataFrame(
        {
            "Simulation 1": [1000, 1100],
            "Simulation 2": [1000, 900],
            "Simulation 3": [1000, 1030],
            "Simulation 4": [1000, 800],
        }
    )

    result = calculate_simulation_summary(simulation_df, initial_value=1000)

    simulated_var = result["simulated_var"]
    expected_shortfall = result["expected_shortfall"]

    assert simulated_var >= 0
    assert expected_shortfall >= 0


def test_calculate_simulation_summary_from_hand_calculated_final_values():
    """Prove horizon VaR, Expected Shortfall, and loss probability by hand."""
    simulation_df = pd.DataFrame(
        {
            "Simulation 1": [1000, 1100],
            "Simulation 2": [1000, 900],
            "Simulation 3": [1000, 1030],
            "Simulation 4": [1000, 800],
            "Simulation 5": [1000, 1000],
        }
    )

    result = calculate_simulation_summary(
        simulation_df,
        initial_value=1000,
        confidence_level=0.80,
    )

    # Final values are 1100, 900, 1030, 800, 1000.
    # Two of five simulations finish below 1000, so probability of loss is 40%.
    expected_probability_of_loss = 0.40

    # At 80% confidence, the lower 20th percentile is 880, so VaR is 120.
    expected_simulated_var = 120.0

    # Only the worst-tail final value, 800, is averaged, so ES is 200.
    expected_shortfall = 200.0

    assert result["probability_of_loss"] == pytest.approx(
        expected_probability_of_loss
    )
    assert result["simulated_var"] == pytest.approx(expected_simulated_var)
    assert result["expected_shortfall"] == pytest.approx(expected_shortfall)
