"""Streamlit rendering for the Data Quality Lab."""

import pandas as pd
import plotly.express as px
import streamlit as st

import src.ui_components as ui


def render_data_quality_lab(report: dict) -> None:
    """Render pre-calculated data-quality diagnostics."""
    st.subheader("Data Quality Lab")
    st.caption(
        "Automated checks help prioritise manual validation before modelling. "
        "Flags are diagnostic evidence, not proof of data corruption or a "
        "structural market change."
    )

    warning_count = len(report["warnings"])
    score_tone = (
        "green"
        if report["score"] >= 90
        else "amber"
        if report["score"] >= 70
        else "red"
    )
    date_range = (
        f"{_format_date(report['start_date'])} to {_format_date(report['end_date'])}"
        if pd.notna(report["start_date"]) and pd.notna(report["end_date"])
        else "Unavailable"
    )
    ui.metric_card_grid(
        [
            (
                "Quality Score",
                f"{report['score']:.1f}/100",
                "Weighted screening score, not a certification",
                score_tone,
            ),
            (
                "Dataset Size",
                f"{report['row_count']:,} x {report['column_count']}",
                "Rows x columns",
                "blue",
            ),
            ("Date Range", date_range, "Parsed observation period", "slate"),
            (
                "Warnings",
                str(warning_count),
                "Items recommended for review",
                "red" if warning_count else "green",
            ),
        ]
    )
    st.caption(
        "A high score does not guarantee that the dataset contains enough "
        "history for reliable modelling. Check the asset-level observation status."
    )

    with st.container(border=True):
        st.subheader("Validation summary")
        st.caption(
            "Each warning explains why the issue can affect research and suggests "
            "a validation step."
        )
        if report["warnings"].empty:
            st.success("No automated warning thresholds were triggered.")
        else:
            ui.styled_table(report["warnings"], max_rows=50)

    missing_summary = _format_quality_table(
        report["missing_values"], percent_columns=["Missing Percent"]
    )
    duplicate_summary = _format_quality_table(
        report["duplicate_dates"], date_columns=["Date"]
    )
    first_column, second_column = st.columns(2)
    with first_column:
        _render_diagnostic_table(
            "Missing values",
            "Missing prices or identifiers can remove observations and make "
            "asset histories difficult to compare.",
            missing_summary,
            "No missing values were detected.",
        )
    with second_column:
        _render_diagnostic_table(
            "Duplicate asset dates",
            "More than one price for the same ticker and date can double-count "
            "a market period.",
            duplicate_summary,
            "No duplicate ticker-date pairs were detected.",
        )

    stale_summary = _format_quality_table(
        report["stale_prices"], date_columns=["Start Date", "End Date"]
    )
    extreme_summary = _format_quality_table(
        report["extreme_returns"],
        percent_columns=["Return", "IQR Lower Bound", "IQR Upper Bound"],
        date_columns=["Date"],
    )
    stale_column, extreme_column = st.columns(2)
    with stale_column:
        _render_diagnostic_table(
            "Stale prices",
            "Unchanged runs may be valid, but can also indicate illiquidity, "
            "market closures, or an outdated feed.",
            stale_summary,
            "No stale-price runs crossed the selected threshold.",
        )
    with extreme_column:
        _render_diagnostic_table(
            "Extreme returns",
            "Z-score and IQR flags highlight observations that may strongly "
            "influence volatility and tail-risk estimates.",
            extreme_summary,
            "No return outliers crossed either selected threshold.",
        )

    with st.container(border=True):
        st.subheader("Asset-level quality")
        st.caption(
            "Issue counts and observation status identify the histories that "
            "need manual review first."
        )
        ui.styled_table(report["asset_quality"], max_rows=100)

    shift_display = _format_quality_table(
        report["distribution_shift"],
        percent_columns=[
            "First Half Mean",
            "Second Half Mean",
            "First Half Std",
            "Second Half Std",
        ],
    )
    with st.container(border=True):
        st.subheader("Return-distribution comparison")
        st.caption(
            "The first and second halves are compared using means, standard "
            "deviations, and a two-sample Kolmogorov-Smirnov test. A low "
            "p-value is a research prompt, not proof of a regime change."
        )
        ui.styled_table(shift_display, max_rows=100)
        _render_distribution_chart(report["return_halves"])

    with st.container(border=True):
        st.subheader("Rolling volatility")
        st.caption(
            "Rolling annualised volatility shows how return variability changes "
            "through time. It describes the sample and does not predict future "
            "volatility."
        )
        _render_rolling_volatility(report["rolling_volatility"])

    st.download_button(
        "Download CSV validation report",
        data=report["validation_report"].to_csv(index=False).encode("utf-8"),
        file_name="data_quality_validation_report.csv",
        mime="text/csv",
        help="Download the score, warnings, recommendations, and asset summaries.",
    )
    st.caption(
        "The report records automated diagnostics only. Research conclusions "
        "still require source checks and human judgement."
    )


def _format_quality_table(
    data: pd.DataFrame,
    percent_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    display_data = data.copy()
    for column in percent_columns or []:
        if column in display_data.columns:
            display_data[column] = display_data[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    for column in date_columns or []:
        if column in display_data.columns:
            display_data[column] = pd.to_datetime(
                display_data[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    return display_data


def _render_diagnostic_table(
    title: str,
    explanation: str,
    data: pd.DataFrame,
    empty_message: str,
) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.caption(explanation)
        if data.empty:
            st.success(empty_message)
        else:
            ui.styled_table(data, max_rows=100)


def _render_distribution_chart(return_halves: pd.DataFrame) -> None:
    if return_halves.empty:
        return

    ticker = st.selectbox(
        "Distribution ticker",
        sorted(return_halves["Ticker"].unique()),
        key="quality_distribution_ticker",
    )
    selected_returns = return_halves[return_halves["Ticker"] == ticker]
    figure = px.histogram(
        selected_returns,
        x="Return",
        color="Period",
        barmode="overlay",
        opacity=0.62,
        histnorm="probability density",
        color_discrete_map={
            "First half": "#2563eb",
            "Second half": "#d97706",
        },
        title=f"{ticker}: first vs second half returns",
    )
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="Sample period",
        xaxis_title="Daily return",
        yaxis_title="Density",
    )
    figure.update_xaxes(tickformat=".1%")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _render_rolling_volatility(rolling_volatility: pd.DataFrame) -> None:
    if rolling_volatility.empty:
        st.info("There are too few valid returns for the selected rolling window.")
        return

    figure = px.line(
        rolling_volatility,
        x="Date",
        y="Rolling Volatility",
        color="Ticker",
        title="Rolling annualised return volatility",
    )
    figure.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="Ticker",
        xaxis_title="Date",
        yaxis_title="Annualised volatility",
    )
    figure.update_yaxes(tickformat=".1%")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _format_date(value) -> str:
    return value.strftime("%d %b %Y")
