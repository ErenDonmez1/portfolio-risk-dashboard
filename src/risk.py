"""Risk calculation helpers for the Portfolio Risk Dashboard."""

import numpy as np
import pandas as pd

from src.portfolio import validate_weights


def calculate_historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical Value at Risk from daily portfolio returns.

    Value at Risk estimates a loss threshold from past returns. At 95%
    confidence, this historical method uses the 5th percentile daily return and
    reports the loss as a positive number. Isolated NaN and infinite values are
    excluded; the calculation fails if no finite observations remain.
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Ignore isolated unusable observations, but never calculate a percentile
    # when no finite return data remains.
    usable_returns = pd.to_numeric(portfolio_returns, errors="coerce")
    usable_returns = usable_returns[np.isfinite(usable_returns)]
    if usable_returns.empty:
        raise ValueError("portfolio_returns must contain at least one finite value")

    percentile = (1 - confidence_level) * 100
    var_return = np.percentile(usable_returns, percentile)
    historical_var = max(0.0, -var_return)

    return float(historical_var)


def calculate_stress_test_loss(
    weights: dict,
    shocks: dict,
    portfolio_value: float = 10000,
) -> dict:
    """Calculate portfolio impact from hypothetical ticker-level shocks.

    Stress testing applies assumed market shocks to each holding to estimate how
    the total portfolio would respond. It is a scenario analysis tool, not a
    prediction of what markets will do.
    """
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("portfolio_value must be greater than 0")
    if not weights:
        raise ValueError("weights must not be empty")
    validate_weights(weights, list(weights))

    missing_shocks = [ticker for ticker in weights if ticker not in shocks]
    if missing_shocks:
        missing = ", ".join(missing_shocks)
        raise ValueError(f"Missing shock(s) for ticker(s): {missing}")

    numeric_shocks = {}
    for ticker in weights:
        try:
            numeric_shock = float(shocks[ticker])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Shock for {ticker} must be a finite number"
            ) from error
        if not np.isfinite(numeric_shock):
            raise ValueError(f"Shock for {ticker} must be a finite number")
        numeric_shocks[ticker] = numeric_shock

    portfolio_impact_percent = sum(
        float(weights[ticker]) * numeric_shocks[ticker] for ticker in weights
    )
    portfolio_impact_value = portfolio_value * portfolio_impact_percent
    stressed_portfolio_value = portfolio_value + portfolio_impact_value

    return {
        "portfolio_impact_percent": portfolio_impact_percent,
        "portfolio_impact_value": portfolio_impact_value,
        "stressed_portfolio_value": stressed_portfolio_value,
    }


def run_monte_carlo_simulation(
    portfolio_returns: pd.Series,
    num_simulations: int = 1000,
    num_days: int = 252,
    initial_value: float = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate illustrative portfolio value paths using historical returns.

    A Monte Carlo simulation creates many scenario paths by drawing
    random daily returns. This simple version uses the portfolio's historical
    average return and volatility, so it is a risk illustration rather than a
    market forecast.
    """
    if num_simulations <= 0:
        raise ValueError("num_simulations must be greater than 0")
    if num_days <= 0:
        raise ValueError("num_days must be greater than 0")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be greater than 0")

    usable_returns = pd.to_numeric(portfolio_returns, errors="coerce")
    usable_returns = usable_returns[np.isfinite(usable_returns)]
    if len(usable_returns) < 2:
        raise ValueError(
            "At least two finite portfolio returns are required for simulation."
        )

    mean_return = float(usable_returns.mean())
    return_volatility = float(usable_returns.std())
    if not np.isfinite(mean_return):
        raise ValueError("Historical mean return must be finite")
    if not np.isfinite(return_volatility) or return_volatility < 0:
        raise ValueError(
            "Historical return standard deviation must be finite and non-negative"
        )

    random_generator = np.random.default_rng(random_seed)
    simulated_returns = random_generator.normal(
        loc=mean_return,
        scale=return_volatility,
        size=(num_days, num_simulations),
    )

    starting_values = np.full((1, num_simulations), initial_value)
    simulated_values = initial_value * (1 + simulated_returns).cumprod(axis=0)
    simulation_paths = np.vstack([starting_values, simulated_values])
    if not np.isfinite(simulation_paths).all():
        raise ValueError("Simulation produced non-finite portfolio paths")

    columns = [f"Simulation {number}" for number in range(1, num_simulations + 1)]

    return pd.DataFrame(simulation_paths, columns=columns)


def run_correlated_monte_carlo_simulation(
    returns_df: pd.DataFrame,
    weights: dict,
    num_simulations: int = 1000,
    num_days: int = 252,
    initial_value: float = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate portfolio paths from correlated asset-level return scenarios.

    This simulation draws asset returns jointly from their historical mean
    returns and covariance matrix. The covariance matrix is the finance concept
    that captures how holdings have tended to move together, so the simulated
    portfolio paths reflect diversification more directly than a one-series
    portfolio simulation.
    """
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be greater than 0")
    if num_days <= 0:
        raise ValueError("num_days must be greater than 0")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be greater than 0")
    if not weights:
        raise ValueError("weights must not be empty")

    missing_tickers = [ticker for ticker in weights if ticker not in returns_df.columns]
    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise ValueError(f"Missing return data for ticker(s): {missing}")

    validate_weights(weights, list(returns_df.columns))
    tickers = list(weights.keys())
    selected_returns = (
        returns_df.loc[:, tickers]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    if len(selected_returns) < 2:
        raise ValueError(
            "At least two complete finite return rows are required for "
            "correlated simulation."
        )

    mean_returns = selected_returns.mean().to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        covariance_matrix = selected_returns.cov().to_numpy()
    if not np.isfinite(mean_returns).all():
        raise ValueError("Historical asset mean returns must be finite")
    if not np.isfinite(covariance_matrix).all():
        raise ValueError("Historical covariance matrix must be finite")
    weight_array = np.array([weights[ticker] for ticker in tickers], dtype=float)

    random_generator = np.random.default_rng(random_seed)
    simulated_asset_returns = random_generator.multivariate_normal(
        mean=mean_returns,
        cov=covariance_matrix,
        size=(num_days, num_simulations),
    )

    # Weighted asset returns turn each simulated market day into one portfolio return.
    simulated_portfolio_returns = simulated_asset_returns @ weight_array
    starting_values = np.full((1, num_simulations), initial_value)
    simulated_values = initial_value * (1 + simulated_portfolio_returns).cumprod(
        axis=0
    )
    simulation_paths = np.vstack([starting_values, simulated_values])
    if not np.isfinite(simulation_paths).all():
        raise ValueError("Simulation produced non-finite portfolio paths")

    columns = [f"Simulation {number}" for number in range(1, num_simulations + 1)]

    return pd.DataFrame(simulation_paths, columns=columns)


def calculate_simulation_summary(
    simulation_df: pd.DataFrame,
    initial_value: float = 10000,
    confidence_level: float = 0.95,
) -> dict:
    """Summarise final-value risk from simulated portfolio paths.

    The final row represents the value at the simulation horizon. VaR and
    Expected Shortfall are shown as positive currency losses from the starting
    value, which makes downside risk easier to read in the dashboard.
    """
    if simulation_df.empty:
        raise ValueError("simulation_df must not be empty")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be greater than 0")
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1")

    final_values = pd.to_numeric(simulation_df.iloc[-1], errors="coerce")
    final_values = final_values[np.isfinite(final_values)]
    if final_values.empty:
        raise ValueError(
            "simulation_df must contain at least one finite final value"
        )
    lower_percentile = (1 - confidence_level) * 100
    upper_percentile = confidence_level * 100
    lower_percentile_final_value = float(np.percentile(final_values, lower_percentile))
    upper_percentile_final_value = float(np.percentile(final_values, upper_percentile))
    worst_tail_values = final_values[final_values <= lower_percentile_final_value]
    worst_tail_losses = (initial_value - worst_tail_values).clip(lower=0)

    if worst_tail_losses.empty:
        expected_shortfall = 0.0
    else:
        expected_shortfall = float(worst_tail_losses.mean())

    summary = {
        "median_final_value": float(final_values.median()),
        "lower_percentile_final_value": lower_percentile_final_value,
        "upper_percentile_final_value": upper_percentile_final_value,
        "probability_of_loss": float((final_values < initial_value).mean()),
        "simulated_var": max(0.0, initial_value - lower_percentile_final_value),
        "expected_shortfall": expected_shortfall,
    }
    if not all(np.isfinite(value) for value in summary.values()):
        raise ValueError("Simulation summary produced a non-finite result")
    return summary
