"""Reusable Streamlit UI components for the Portfolio Risk Dashboard."""

from html import escape

import streamlit as st


def inject_global_styles() -> None:
    """Inject light-mode dashboard styling for a polished finance UI."""
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f3f6fb;
            --card-bg: #ffffff;
            --card-muted: #f8fafc;
            --border: #dbe4f0;
            --border-strong: #c6d3e1;
            --text: #111827;
            --muted: #64748b;
            --subtle: #94a3b8;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #059669;
            --amber: #d97706;
            --red: #dc2626;
            --shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        }

        html,
        body,
        #root,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--app-bg) !important;
            color: var(--text) !important;
            color-scheme: light !important;
        }

        header[data-testid="stHeader"],
        .stApp > header,
        .stAppHeader,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] {
            background: rgba(243, 246, 251, 0.94) !important;
            border-bottom: 1px solid var(--border) !important;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span {
            color: var(--text) !important;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: 10px 0 30px rgba(15, 23, 42, 0.04);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
            color: var(--text) !important;
        }

        .stApp input,
        .stApp textarea,
        .stApp div[data-baseweb="input"],
        .stApp div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background: #ffffff !important;
            border-color: var(--border-strong) !important;
            color: var(--text) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stExpander"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow) !important;
        }

        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"],
        div[data-testid="column"] > div {
            min-width: 0 !important;
        }

        div[data-testid="stExpander"] summary {
            background: var(--card-muted) !important;
            border-radius: 14px 14px 0 0 !important;
        }

        div[data-testid="stExpander"] summary p {
            color: var(--text) !important;
            font-weight: 700;
        }

        button[data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 10px 10px 0 0 !important;
            padding-top: 0.55rem !important;
            padding-bottom: 0.55rem !important;
        }

        button[data-baseweb="tab"] p {
            color: var(--muted) !important;
            font-weight: 700;
        }

        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--blue) !important;
        }

        .dashboard-header {
            background:
                radial-gradient(circle at 92% 12%, rgba(37, 99, 235, 0.16), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f8fbff 58%, #edf5ff 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            display: grid;
            gap: 1.25rem;
            grid-template-columns: minmax(0, 1.7fr) minmax(260px, 0.8fr);
            margin-bottom: 1.05rem;
            padding: 1.45rem 1.55rem;
        }

        .dashboard-eyebrow,
        .sidebar-eyebrow {
            color: var(--blue) !important;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
        }

        .dashboard-header h1 {
            color: var(--text) !important;
            font-size: 2.05rem;
            line-height: 1.12;
            margin: 0 0 0.45rem 0;
        }

        .dashboard-header p {
            color: var(--muted) !important;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0;
            max-width: 780px;
        }

        .header-meta {
            align-content: center;
            display: grid;
            gap: 0.65rem;
        }

        .header-pill {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.72rem 0.85rem;
        }

        .header-pill-label {
            color: var(--subtle) !important;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.16rem;
            text-transform: uppercase;
        }

        .header-pill-value {
            color: var(--text) !important;
            font-size: 1rem;
            font-weight: 800;
        }

        .panel-title {
            color: var(--text) !important;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
        }

        .panel-note,
        .section-note,
        .risk-table-note {
            color: var(--muted) !important;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .metric-card {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            box-sizing: border-box;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06) !important;
            display: block;
            margin-bottom: 0.65rem;
            min-height: 118px;
            overflow: hidden;
            padding: 1rem 1rem 0.9rem 1rem;
            position: relative;
            width: 100%;
        }

        .metric-card-grid {
            box-sizing: border-box;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            margin: 0.15rem 0 0.75rem 0;
            max-width: 100%;
            overflow: hidden;
            width: 100%;
        }

        .metric-label {
            color: var(--muted) !important;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.42rem;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--text) !important;
            font-size: 1.7rem;
            font-weight: 850;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .metric-detail {
            color: var(--muted) !important;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }

        .metric-card-blue::before { background: var(--blue) !important; }
        .metric-card-green::before { background: var(--green) !important; }
        .metric-card-amber::before { background: var(--amber) !important; }
        .metric-card-red::before { background: var(--red) !important; }
        .metric-card-slate::before { background: #475569 !important; }

        .ticker-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin: 0.3rem 0 1.05rem 0;
        }

        .ticker-chip {
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 999px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            display: inline-flex;
            gap: 0.55rem;
            padding: 0.48rem 0.72rem;
        }

        .ticker-symbol {
            color: var(--text) !important;
            font-size: 0.82rem;
            font-weight: 850;
        }

        .ticker-price {
            color: var(--muted) !important;
            font-size: 0.8rem;
        }

        .styled-table-wrap {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            margin-top: 0.35rem;
            max-height: 380px;
            overflow: auto;
        }

        table.styled-table {
            border-collapse: collapse;
            color: var(--text) !important;
            font-size: 0.86rem;
            width: 100%;
        }

        table.styled-table thead th {
            background: #f1f5f9 !important;
            border-bottom: 1px solid var(--border-strong) !important;
            border-right: 0 !important;
            color: #475569 !important;
            font-size: 0.72rem;
            font-weight: 850;
            padding: 0.68rem 0.76rem;
            text-align: left;
            text-transform: uppercase;
        }

        table.styled-table tbody th,
        table.styled-table tbody td {
            background: transparent !important;
            border-bottom: 1px solid #e5eaf2 !important;
            border-right: 0 !important;
            color: var(--text) !important;
            padding: 0.62rem 0.76rem;
            text-align: left;
        }

        table.styled-table tbody tr:nth-child(even) {
            background: #f8fafc !important;
        }

        table.styled-table tbody tr:nth-child(odd) {
            background: #ffffff !important;
        }

        table.styled-table tbody tr:hover {
            background: #eef6ff !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px !important;
        }

        @media (max-width: 900px) {
            .dashboard-header {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, badges: list[tuple[str, str]]) -> None:
    """Render the compact dashboard header used at the top of the page."""
    badge_html = "".join(
        (
            '<div class="header-pill">'
            f'<div class="header-pill-label">{escape(label)}</div>'
            f'<div class="header-pill-value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in badges
    )

    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <div class="dashboard-eyebrow">Risk analytics dashboard</div>
                <h1>{escape(title)}</h1>
                <p>{escape(subtitle)}</p>
            </div>
            <div class="header-meta">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card_html(label: str, value: str, detail: str = "", tone: str = "blue") -> str:
    """Build metric card HTML for responsive metric grids."""
    return (
        f'<div class="metric-card metric-card-{escape(tone)}" '
        'style="box-sizing: border-box; display: block; max-width: 100%; '
        'overflow: hidden; width: 100%;">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-detail">{escape(detail)}</div>'
        "</div>"
    )


def metric_card_grid(cards: list[tuple[str, str, str, str]]) -> None:
    """Render metric cards in a responsive grid that stays inside containers."""
    card_html = "".join(
        metric_card_html(label, value, detail, tone)
        for label, value, detail, tone in cards
    )
    st.markdown(
        f'<div class="metric-card-grid">{card_html}</div>',
        unsafe_allow_html=True,
    )


def styled_table(
    data,
    max_rows: int | None = None,
    show_index: bool = False,
) -> None:
    """Render a compact table that matches the dashboard theme."""
    table_data = data.copy()
    if max_rows is not None:
        table_data = table_data.head(max_rows)

    html_table = table_data.to_html(
        classes="styled-table",
        border=0,
        escape=True,
        index=show_index,
    )
    st.markdown(
        f'<div class="styled-table-wrap">{html_table}</div>',
        unsafe_allow_html=True,
    )


def ticker_strip(price_data, max_tickers: int = 8) -> None:
    """Render latest close prices as compact ticker chips."""
    latest_prices = (
        price_data.sort_values("Date")
        .groupby("Ticker", as_index=False)
        .tail(1)
        .sort_values("Ticker")
        .head(max_tickers)
    )
    chips = []
    for _, row in latest_prices.iterrows():
        chips.append(
            '<div class="ticker-chip">'
            f'<span class="ticker-symbol">{escape(str(row["Ticker"]))}</span>'
            f'<span class="ticker-price">Close {row["Close"]:,.2f}</span>'
            "</div>"
        )

    st.markdown(
        f'<div class="ticker-strip">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )
