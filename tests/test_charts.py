"""Tests for matplotlib chart helper functions."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.charts import (
    MUTED_TEXT_COLOR,
    TEXT_COLOR,
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_monte_carlo_paths,
)


def test_plot_cumulative_returns_returns_figure():
    cumulative_returns = pd.DataFrame(
        {"ALPHA": [0.00, 0.10], "BETA": [0.00, -0.05]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    figure = plot_cumulative_returns(cumulative_returns)

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_drawdowns_returns_figure():
    drawdowns = pd.DataFrame(
        {"ALPHA": [0.00, -0.10], "BETA": [0.00, -0.05]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    figure = plot_drawdowns(drawdowns)

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_correlation_matrix_returns_figure():
    returns = pd.DataFrame(
        {"ALPHA": [0.01, 0.02, -0.01], "BETA": [0.00, 0.03, -0.02]}
    )

    figure = plot_correlation_matrix(returns)

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_monte_carlo_paths_returns_figure():
    simulations = pd.DataFrame(
        {
            "Simulation 1": [1000, 1010, 1020],
            "Simulation 2": [1000, 990, 995],
        }
    )

    figure = plot_monte_carlo_paths(simulations)

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_monte_carlo_paths_uses_compact_summary_legend():
    simulations = pd.DataFrame(
        {
            f"Simulation {number}": [1000, 1000 + number, 1010 + number]
            for number in range(1, 101)
        }
    )

    figure = plot_monte_carlo_paths(simulations)
    legend = figure.axes[0].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]

    assert labels == [
        "5th-95th percentile range",
        "Median simulated path",
    ]
    plt.close(figure)


def test_plot_monte_carlo_paths_limits_background_lines_to_10():
    simulations = pd.DataFrame(
        {
            f"Simulation {number}": [1000, 1000 + number, 1010 + number]
            for number in range(1, 101)
        }
    )

    figure = plot_monte_carlo_paths(simulations)
    plotted_lines = figure.axes[0].lines

    assert len(plotted_lines) == 11
    plt.close(figure)


def test_monte_carlo_axis_uses_sterling_currency():
    simulation_data = pd.DataFrame(
        {
            "Simulation 1": [10000.0, 10100.0],
            "Simulation 2": [10000.0, 9900.0],
        }
    )

    figure = plot_monte_carlo_paths(simulation_data)
    formatter = figure.axes[0].yaxis.get_major_formatter()

    assert formatter(10000, 0) == "£10,000.00"
    plt.close(figure)


def test_chart_axis_text_uses_light_theme_colours():
    cumulative_returns = pd.DataFrame(
        {"ALPHA": [0.00, 0.10], "BETA": [0.00, -0.05]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    figure = plot_cumulative_returns(cumulative_returns)
    axis = figure.axes[0]

    assert axis.title.get_color() == TEXT_COLOR
    assert axis.xaxis.label.get_color() == MUTED_TEXT_COLOR
    assert axis.yaxis.label.get_color() == MUTED_TEXT_COLOR
    assert all(
        label.get_color() == MUTED_TEXT_COLOR
        for label in axis.get_xticklabels() + axis.get_yticklabels()
    )
    plt.close(figure)
