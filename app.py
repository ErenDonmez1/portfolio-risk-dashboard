"""Streamlit user interface for the Portfolio Risk Dashboard."""

import pandas as pd
import streamlit as st

from src.charts import (
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_monte_carlo_paths,
)
from src.data_loader import (
    MAX_YFINANCE_TICKERS,
    fetch_yfinance_price_data,
    load_price_data,
    parse_ticker_list,
)
from src.data_quality import run_data_quality_analysis
from src.data_quality_ui import render_data_quality_lab
from src.formatting import format_currency
from src.metrics import (
    calculate_annualised_volatility,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_drawdown,
    calculate_max_drawdown,
)
from src.portfolio import (
    calculate_portfolio_cumulative_returns,
    calculate_portfolio_returns,
    validate_weights,
)
import src.risk as risk
import src.ui_components as ui


YFINANCE_PRESETS = {
    "US Mega-Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "US Banks": ["JPM", "BAC", "GS", "MS", "C"],
    "UK Large Caps": ["SHEL.L", "BP.L", "HSBA.L", "RR.L", "TSCO.L"],
    "Global ETFs": ["SPY", "QQQ", "VTI", "EFA", "EEM"],
    "Mixed Demo Portfolio": ["AAPL", "MSFT", "JPM", "SHEL.L", "RR.L"],
}
YFINANCE_PERIODS = ["6mo", "1y", "2y", "5y"]
STRESS_SCENARIOS = {
    "Mild sell-off: all assets -5%": -0.05,
    "Market correction: all assets -10%": -0.10,
    "Severe shock: all assets -20%": -0.20,
    "Custom ticker shocks": None,
}
YFINANCE_DISCLAIMER = (
    "Market data is fetched using yfinance for educational and research "
    "purposes. yfinance is not affiliated with, endorsed by, or vetted by "
    "Yahoo. Data may be delayed, unavailable, or subject to Yahoo's terms of "
    "use. This dashboard does not provide investment advice."
)


st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide",
)


def format_percent(value: float) -> str:
    """Format decimal returns and risk values as percentages for the UI."""
    return f"{value:.2%}"


def format_date(value) -> str:
    """Format a pandas timestamp for compact dashboard display."""
    return value.strftime("%d %b %Y")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_yfinance_price_data(
    tickers: tuple[str, ...],
    period: str,
    interval: str,
):
    """Fetch yfinance data through Streamlit's cache to reduce repeat calls."""
    return fetch_yfinance_price_data(list(tickers), period=period, interval=interval)


ui.inject_global_styles()
st.sidebar.title("Portfolio Setup")
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose price data source",
    ["Demo data", "Upload CSV", "Fetch market data with yfinance"],
)
data_source_badge = {
    "Demo data": "Synthetic demo data",
    "Upload CSV": "Uploaded CSV",
    "Fetch market data with yfinance": "yfinance market data",
}[data_source]
ui.page_header(
    "Portfolio Risk Dashboard",
    (
        "Analyse portfolio risk from price data using returns, volatility, "
        "drawdowns, correlations, historical Value at Risk, and simulation."
    ),
    [
        ("Input source", data_source_badge),
        ("Risk view", "Returns, VaR and simulations"),
    ],
)

uploaded_file = None
yfinance_tickers = []
yfinance_period = "2y"

if data_source == "Demo data":
    st.sidebar.caption("Uses synthetic sample data included in the project.")
elif data_source == "Upload CSV":
    st.sidebar.caption("CSV upload is the most reliable option.")
    uploaded_file = st.sidebar.file_uploader(
        "Upload price data CSV",
        type=["csv"],
        help="CSV must contain Date, Ticker, and Close columns.",
    )
else:
    st.sidebar.info(
        "Market data is fetched using yfinance for educational use. It may be "
        "delayed or unavailable for some tickers."
    )
    preset_name = st.sidebar.selectbox(
        "Preset ticker group",
        list(YFINANCE_PRESETS.keys()),
        index=4,
    )
    preset_tickers = ", ".join(YFINANCE_PRESETS[preset_name])
    ticker_text = st.sidebar.text_area(
        "Tickers",
        value=preset_tickers,
        key=f"yfinance_tickers_{preset_name}",
        help="Enter ticker symbols separated by commas.",
    )
    yfinance_period = st.sidebar.selectbox(
        "History period",
        YFINANCE_PERIODS,
        index=2,
    )

    try:
        entered_tickers = parse_ticker_list(ticker_text)
        if len(entered_tickers) > MAX_YFINANCE_TICKERS:
            st.sidebar.warning(
                f"Only the first {MAX_YFINANCE_TICKERS} tickers will be used. "
                f"You entered {len(entered_tickers)}."
            )
        yfinance_tickers = entered_tickers[:MAX_YFINANCE_TICKERS]
    except ValueError as error:
        st.sidebar.error(str(error))

st.sidebar.divider()
st.sidebar.subheader("Data Quality Settings")
quality_z_score = st.sidebar.number_input(
    "Extreme-return z-score",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
)
quality_iqr_multiplier = st.sidebar.number_input(
    "IQR multiplier",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.25,
)
quality_stale_observations = st.sidebar.number_input(
    "Stale-price observations",
    min_value=2,
    max_value=20,
    value=3,
    step=1,
)
quality_minimum_observations = st.sidebar.number_input(
    "Minimum observations",
    min_value=2,
    max_value=1000,
    value=30,
    step=10,
)
quality_rolling_window = st.sidebar.selectbox(
    "Rolling volatility window",
    [5, 20, 60],
    index=1,
    format_func=lambda days: f"{days} observations",
)

with st.container(border=True):
    st.markdown('<div class="panel-title">Data Source</div>', unsafe_allow_html=True)
    if data_source == "Demo data":
        st.markdown(
            '<div class="panel-note">Using synthetic demo data from '
            "<code>data/sample_prices.csv</code>.</div>",
            unsafe_allow_html=True,
        )
    elif data_source == "Upload CSV":
        st.markdown(
            '<div class="panel-note">Upload a CSV with Date, Ticker and Close '
            "columns. This is the most reliable option.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel-note">Fetching adjusted daily close prices where '
            "available using yfinance.</div>",
            unsafe_allow_html=True,
        )
        st.caption(YFINANCE_DISCLAIMER)

try:
    if data_source == "Demo data":
        st.info(
            "Demo mode: using synthetic sample data from "
            "`data/sample_prices.csv`. Upload your own CSV to replace it."
        )
        price_data = load_price_data("data/sample_prices.csv")
    elif data_source == "Upload CSV":
        if uploaded_file is None:
            st.warning("Upload a CSV file to continue, or select demo data.")
            st.stop()
        st.success(f"Loaded uploaded file: `{uploaded_file.name}`.")
        price_data = load_price_data(uploaded_file)
    else:
        if not yfinance_tickers:
            st.warning("Enter at least one ticker to fetch market data.")
            st.stop()
        with st.spinner("Fetching market data with yfinance..."):
            price_data = cached_fetch_yfinance_price_data(
                tuple(yfinance_tickers),
                yfinance_period,
                "1d",
            )
        returned_tickers = set(price_data["Ticker"].unique())
        missing_tickers = [
            ticker for ticker in yfinance_tickers if ticker not in returned_tickers
        ]
        if missing_tickers:
            missing = ", ".join(missing_tickers)
            st.warning(f"No usable price data was returned for: {missing}")
        st.success(
            "Loaded yfinance market data for: "
            f"{', '.join(sorted(returned_tickers))}"
        )
except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

try:
    data_quality_report = run_data_quality_analysis(
        price_data,
        stale_observations=int(quality_stale_observations),
        z_score_threshold=float(quality_z_score),
        iqr_multiplier=float(quality_iqr_multiplier),
        minimum_observations=int(quality_minimum_observations),
        rolling_window=int(quality_rolling_window),
    )
except ValueError as error:
    st.error(f"Data Quality Lab could not analyse this dataset: {error}")
    st.stop()

if not data_quality_report["non_numeric_prices"].empty:
    st.error(
        "Risk calculations are paused because the Close column contains "
        "non-numeric values. Review the flagged rows below and correct the "
        "source file."
    )
    render_data_quality_lab(data_quality_report)
    st.stop()

if not data_quality_report["non_positive_prices"].empty:
    st.error(
        "Risk calculations are paused because the Close column contains zero "
        "or negative prices. These values are unsupported by standard "
        "percentage-return calculations. Review the flagged rows below and "
        "correct the source file."
    )
    render_data_quality_lab(data_quality_report)
    st.stop()

try:
    daily_returns = calculate_daily_returns(price_data)
    cumulative_returns = calculate_cumulative_returns(daily_returns)
except (TypeError, ValueError) as error:
    st.error(str(error))
    render_data_quality_lab(data_quality_report)
    st.stop()

if daily_returns.empty:
    st.error("At least two price dates are needed to calculate returns.")
    render_data_quality_lab(data_quality_report)
    st.stop()

tickers = list(daily_returns.columns)
equal_weight = 1 / len(tickers)

st.sidebar.divider()
st.sidebar.header("Portfolio Weights")
st.sidebar.caption("Set weights as decimals. The default is an equal-weight portfolio.")

weights = {}
with st.sidebar.expander("Asset weights", expanded=True):
    for ticker in tickers:
        weights[ticker] = st.number_input(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=float(equal_weight),
            step=0.01,
            format="%.4f",
        )

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) <= 0.000001:
        st.success(f"Total weight: {total_weight:.4f}")
    else:
        st.error(f"Total weight: {total_weight:.4f}. It must equal 1.0000.")

st.sidebar.divider()
st.sidebar.subheader("Monte Carlo Settings")
show_monte_carlo = st.sidebar.checkbox(
    "Show Monte Carlo simulation",
    value=True,
    help="Turn this off if you only want historical portfolio risk metrics.",
)
if not show_monte_carlo:
    st.sidebar.caption("Monte Carlo simulation is disabled.")

ui.ticker_strip(price_data)

st.subheader("Dataset Overview")
ui.metric_card_grid(
    [
        ("Tickers", str(len(tickers)), "Assets detected", "blue"),
        ("Price Rows", f"{len(price_data):,}", "Raw observations", "slate"),
        ("Return Days", f"{len(daily_returns):,}", "Calculated periods", "green"),
        ("Start Date", format_date(price_data["Date"].min()), "First price", "amber"),
        ("End Date", format_date(price_data["Date"].max()), "Latest price", "red"),
    ]
)

with st.expander("Raw price data preview", expanded=False):
    preview_data = price_data.head(20).copy()
    preview_data["Date"] = preview_data["Date"].dt.strftime("%Y-%m-%d")
    ui.styled_table(preview_data)

asset_tab, portfolio_tab, quality_tab = st.tabs(
    ["Asset Risk", "Portfolio Analysis", "Data Quality Lab"]
)

with asset_tab:
    with st.container(border=True):
        st.subheader("Asset Return Performance")
        st.markdown(
            '<p class="section-note">Compounded returns show how each ticker has '
            "grown or fallen since the first return date.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_cumulative_returns(cumulative_returns), clear_figure=True)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        with st.container(border=True):
            st.subheader("Annualised Volatility")
            st.markdown(
                '<p class="risk-table-note">Higher volatility means larger historical '
                "ups and downs in daily returns.</p>",
                unsafe_allow_html=True,
            )
            annualised_volatility = calculate_annualised_volatility(daily_returns)
            volatility_table = annualised_volatility.rename_axis("Ticker").rename("Value")
            ui.styled_table(volatility_table.map(format_percent).to_frame())

    with right_col:
        with st.container(border=True):
            st.subheader("Maximum Drawdown")
            st.markdown(
                '<p class="risk-table-note">Worst peak-to-trough fall in the selected '
                "period.</p>",
                unsafe_allow_html=True,
            )
            max_drawdown = calculate_max_drawdown(cumulative_returns)
            drawdown_table = max_drawdown.rename_axis("Ticker").rename("Value")
            ui.styled_table(drawdown_table.map(format_percent).to_frame())

    drawdowns = calculate_drawdown(cumulative_returns)

    with st.container(border=True):
        st.subheader("Drawdown History")
        st.markdown(
            '<p class="section-note">This chart shows when each ticker was below '
            "its previous high and how deep the decline became.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_drawdowns(drawdowns), clear_figure=True)

    with st.container(border=True):
        st.subheader("Return Correlations")
        st.markdown(
            '<p class="section-note">Correlation helps show whether tickers tend '
            "to move together or provide diversification.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_correlation_matrix(daily_returns), clear_figure=True)

with portfolio_tab:
    st.subheader("Portfolio Results")
    st.markdown(
        '<p class="section-note">Portfolio results use the sidebar weights and '
        "combine each ticker's daily return into one weighted portfolio.</p>",
        unsafe_allow_html=True,
    )

    try:
        validate_weights(weights, tickers)
        portfolio_returns = calculate_portfolio_returns(daily_returns, weights)
        portfolio_cumulative_returns = calculate_portfolio_cumulative_returns(
            portfolio_returns
        )
        historical_var = risk.calculate_historical_var(portfolio_returns)
    except ValueError as error:
        st.error(str(error))
    else:
        portfolio_volatility = calculate_annualised_volatility(
            portfolio_returns.to_frame()
        ).iloc[0]
        portfolio_max_drawdown = calculate_max_drawdown(
            portfolio_cumulative_returns.to_frame()
        ).iloc[0]
        final_cumulative_return = portfolio_cumulative_returns.iloc[-1]

        ui.metric_card_grid(
            [
                (
                    "Cumulative Return",
                    format_percent(final_cumulative_return),
                    "Weighted portfolio result",
                    "green",
                ),
                (
                    "Historical VaR (95%, 1-day)",
                    format_percent(historical_var),
                    "5th percentile daily portfolio loss",
                    "red",
                ),
                (
                    "Annualised Volatility",
                    format_percent(portfolio_volatility),
                    "Annualised from daily returns using 252 trading days",
                    "amber",
                ),
                (
                    "Max Drawdown",
                    format_percent(portfolio_max_drawdown),
                    "Worst peak-to-trough fall",
                    "blue",
                ),
            ]
        )

        with st.container(border=True):
            st.subheader("Portfolio Cumulative Return")
            st.markdown(
                '<p class="section-note">This chart shows the compounded result of '
                "the weighted portfolio over the available history.</p>",
                unsafe_allow_html=True,
            )
            st.pyplot(
                plot_cumulative_returns(portfolio_cumulative_returns.to_frame()),
                clear_figure=True,
            )

        with st.container(border=True):
            st.subheader("Stress testing")
            st.markdown(
                '<p class="section-note">Stress testing shows how the portfolio '
                "would respond to hypothetical market shocks. It is not a "
                "prediction.</p>",
                unsafe_allow_html=True,
            )

            stress_col1, stress_col2 = st.columns([1, 1])
            with stress_col1:
                stress_scenario = st.selectbox(
                    "Stress scenario",
                    list(STRESS_SCENARIOS.keys()),
                )
            with stress_col2:
                stress_portfolio_value = st.number_input(
                    "Stress test portfolio value (£)",
                    min_value=100.0,
                    value=10000.0,
                    step=500.0,
                    format="%.2f",
                )

            if STRESS_SCENARIOS[stress_scenario] is None:
                st.caption("Enter shocks as percentages. For example, -10 means -10%.")
                shocks = {}
                shock_columns = st.columns(min(4, len(tickers)))
                for index, ticker in enumerate(tickers):
                    with shock_columns[index % len(shock_columns)]:
                        shock_percent = st.number_input(
                            f"{ticker} shock (%)",
                            value=-10.0,
                            step=1.0,
                            format="%.2f",
                        )
                    shocks[ticker] = shock_percent / 100
            else:
                scenario_shock = STRESS_SCENARIOS[stress_scenario]
                shocks = {ticker: scenario_shock for ticker in tickers}

            try:
                stress_result = risk.calculate_stress_test_loss(
                    weights,
                    shocks,
                    portfolio_value=float(stress_portfolio_value),
                )
            except ValueError as error:
                st.error(str(error))
            else:
                ui.metric_card_grid(
                    [
                        (
                            "Portfolio impact",
                            format_percent(stress_result["portfolio_impact_percent"]),
                            "Weighted shock effect",
                            "red",
                        ),
                        (
                            "Estimated value impact",
                            format_currency(
                                stress_result["portfolio_impact_value"]
                            ),
                            "Change in portfolio value",
                            "amber",
                        ),
                        (
                            "Stressed portfolio value",
                            format_currency(
                                stress_result["stressed_portfolio_value"]
                            ),
                            "Value after scenario shock",
                            "blue",
                        ),
                    ]
                )

        with st.expander("Monte Carlo risk simulation", expanded=False):
            if show_monte_carlo:
                st.markdown(
                    '<p class="section-note">This simulation uses historical mean '
                    "return and volatility to generate possible future paths. It "
                    "is a simplified uncertainty model, not a prediction of future "
                    "prices.</p>",
                    unsafe_allow_html=True,
                )

                control_col1, control_col2, control_col3 = st.columns(3)
                with control_col1:
                    num_simulations = st.number_input(
                        "Number of simulations",
                        min_value=100,
                        max_value=2000,
                        value=500,
                        step=100,
                    )
                with control_col2:
                    num_days = st.selectbox(
                        "Simulation horizon",
                        [30, 90, 252, 1260],
                        index=2,
                        format_func=lambda days: f"{days} trading days",
                    )
                with control_col3:
                    initial_value = st.number_input(
                        "Initial portfolio value (£)",
                        min_value=100.0,
                        value=10000.0,
                        step=500.0,
                        format="%.2f",
                    )
                st.caption(f"Starting value: {format_currency(initial_value)}")

                simulation_paths = risk.run_monte_carlo_simulation(
                    portfolio_returns,
                    num_simulations=int(num_simulations),
                    num_days=int(num_days),
                    initial_value=float(initial_value),
                )
                st.pyplot(plot_monte_carlo_paths(simulation_paths), clear_figure=True)
            else:
                st.info("Monte Carlo simulation is disabled in the sidebar.")

        with st.expander("Advanced correlated Monte Carlo", expanded=False):
            if show_monte_carlo:
                st.markdown(
                    '<p class="section-note">This model simulates assets jointly '
                    "using their historical covariance matrix, so it captures how "
                    "portfolio holdings may move together.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="section-note">This is a simplified educational '
                    "model, not a forecast of future portfolio values.</p>",
                    unsafe_allow_html=True,
                )

                advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
                with advanced_col1:
                    advanced_num_simulations = st.number_input(
                        "Advanced simulations",
                        min_value=100,
                        max_value=5000,
                        value=1000,
                        step=100,
                    )
                with advanced_col2:
                    advanced_num_days = st.selectbox(
                        "Advanced horizon",
                        [30, 90, 252, 1260],
                        index=2,
                        format_func=lambda days: f"{days} trading days",
                    )
                with advanced_col3:
                    advanced_initial_value = st.number_input(
                        "Advanced initial portfolio value (£)",
                        min_value=100.0,
                        value=10000.0,
                        step=500.0,
                        format="%.2f",
                    )
                st.caption(
                    "Starting value: "
                    f"{format_currency(advanced_initial_value)}"
                )

                try:
                    correlated_simulation_paths = risk.run_correlated_monte_carlo_simulation(
                        daily_returns,
                        weights,
                        num_simulations=int(advanced_num_simulations),
                        num_days=int(advanced_num_days),
                        initial_value=float(advanced_initial_value),
                    )
                    simulation_summary = risk.calculate_simulation_summary(
                        correlated_simulation_paths,
                        initial_value=float(advanced_initial_value),
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    ui.metric_card_grid(
                        [
                            (
                                "Median final value",
                                format_currency(
                                    simulation_summary["median_final_value"]
                                ),
                                "Middle simulated ending value",
                                "blue",
                            ),
                            (
                                "5th percentile final value",
                                format_currency(
                                    simulation_summary[
                                        "lower_percentile_final_value"
                                    ]
                                ),
                                "Downside final-value threshold",
                                "red",
                            ),
                            (
                                "95th percentile final value",
                                format_currency(
                                    simulation_summary[
                                        "upper_percentile_final_value"
                                    ]
                                ),
                                "Upside final-value threshold",
                                "green",
                            ),
                        ]
                    )

                    ui.metric_card_grid(
                        [
                            (
                                "Probability of loss",
                                format_percent(
                                    simulation_summary["probability_of_loss"]
                                ),
                                "Share ending below initial value",
                                "amber",
                            ),
                            (
                                "Simulated VaR",
                                format_currency(
                                    simulation_summary["simulated_var"]
                                ),
                                "Loss at the lower-tail final value",
                                "red",
                            ),
                            (
                                "Expected Shortfall",
                                format_currency(
                                    simulation_summary["expected_shortfall"]
                                ),
                                "Average loss in the worst 5% of outcomes",
                                "slate",
                            ),
                        ]
                    )

                    st.pyplot(
                        plot_monte_carlo_paths(correlated_simulation_paths),
                        clear_figure=True,
                    )
            else:
                st.info("Monte Carlo simulation is disabled in the sidebar.")

        with st.expander("Assumptions and limitations"):
            st.write(
                "This dashboard is for educational analysis only and is not "
                "investment advice. Historical VaR is based on daily portfolio "
                "returns. Monte Carlo paths assume normally distributed returns, "
                "constant mean and volatility, no transaction costs, and fixed "
                "portfolio weights over the simulated period."
            )

with quality_tab:
    render_data_quality_lab(data_quality_report)
