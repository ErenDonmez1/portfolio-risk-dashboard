"""Data loading helpers for the Portfolio Risk Dashboard."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Date", "Ticker", "Close"}
MAX_YFINANCE_TICKERS = 8
YFINANCE_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / ".cache" / "yfinance"
)


def parse_ticker_list(ticker_text: str) -> list[str]:
    """Parse comma-separated ticker text into a clean list of ticker symbols."""
    tickers = []
    seen_tickers = set()

    for raw_ticker in ticker_text.split(","):
        ticker = raw_ticker.strip().upper()
        if ticker and ticker not in seen_tickers:
            tickers.append(ticker)
            seen_tickers.add(ticker)

    if not tickers:
        raise ValueError("Enter at least one ticker symbol.")

    return tickers


def load_price_data(file_path: str) -> pd.DataFrame:
    """Load stock price data from a CSV file and return a clean DataFrame."""
    data = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    clean_data = data.loc[:, ["Date", "Ticker", "Close"]].copy()
    clean_data["Date"] = pd.to_datetime(clean_data["Date"])

    # Close is the end-of-day price used later to calculate returns and risk.
    clean_data = clean_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return clean_data


def _extract_close_prices(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw_data.empty:
        raise ValueError("No market data was returned by yfinance.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        first_level = raw_data.columns.get_level_values(0)
        second_level = raw_data.columns.get_level_values(1)

        if "Close" in first_level:
            close_prices = raw_data["Close"]
        elif "Adj Close" in first_level:
            close_prices = raw_data["Adj Close"]
        elif "Close" in second_level:
            close_prices = raw_data.xs("Close", axis=1, level=1)
        elif "Adj Close" in second_level:
            close_prices = raw_data.xs("Adj Close", axis=1, level=1)
        else:
            raise ValueError("No Close price data was returned by yfinance.")
    else:
        if "Close" in raw_data.columns:
            close_prices = raw_data["Close"]
        elif "Adj Close" in raw_data.columns:
            close_prices = raw_data["Adj Close"]
        else:
            raise ValueError("No Close price data was returned by yfinance.")

    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers[0])
    else:
        close_prices = close_prices.copy()
        close_prices.columns = [str(column).upper() for column in close_prices.columns]

    return close_prices


def _format_yfinance_price_data(
    raw_data: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    close_prices = _extract_close_prices(raw_data, tickers)
    close_prices.index.name = "Date"

    long_data = close_prices.reset_index().melt(
        id_vars="Date",
        var_name="Ticker",
        value_name="Close",
    )
    long_data["Date"] = pd.to_datetime(long_data["Date"])
    long_data["Ticker"] = long_data["Ticker"].astype(str).str.upper()
    long_data["Close"] = pd.to_numeric(long_data["Close"], errors="coerce")
    long_data = long_data.dropna(subset=["Close"])

    if long_data.empty:
        raise ValueError("No usable Close price data was returned by yfinance.")

    return long_data.loc[:, ["Date", "Ticker", "Close"]].sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


def fetch_yfinance_price_data(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch daily adjusted close prices from yfinance in app-ready format."""
    clean_tickers = parse_ticker_list(",".join(tickers))

    try:
        import yfinance as yf
    except ModuleNotFoundError as error:
        raise ValueError(
            "yfinance is not installed. Run pip install -r requirements.txt first."
        ) from error

    # yfinance stores timezone and cookie metadata in SQLite files. Keeping the
    # cache inside the project avoids permission errors from system cache paths.
    YFINANCE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(YFINANCE_CACHE_DIRECTORY))

    raw_data = yf.download(
        tickers=clean_tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    return _format_yfinance_price_data(raw_data, clean_tickers)
