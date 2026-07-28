"""Tests for CSV price data loading."""

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    fetch_yfinance_price_data,
    load_price_data,
    parse_ticker_list,
)


def test_load_price_data_loads_sample_csv():
    data = load_price_data("data/sample_prices.csv")

    assert list(data.columns) == ["Date", "Ticker", "Close"]
    assert len(data) == 18
    assert pd.api.types.is_datetime64_any_dtype(data["Date"])
    assert data.iloc[0]["Ticker"] == "ALPHA"
    assert data.iloc[-1]["Ticker"] == "GAMMA"
    assert data.equals(data.sort_values(["Date", "Ticker"]).reset_index(drop=True))


def test_load_price_data_raises_value_error_for_missing_columns(tmp_path):
    bad_csv = tmp_path / "missing_close.csv"
    bad_csv.write_text("Date,Ticker\n2026-01-02,ALPHA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required column\\(s\\): Close"):
        load_price_data(str(bad_csv))


def test_parse_ticker_list_cleans_and_deduplicates_tickers():
    result = parse_ticker_list(" aapl, MSFT, aapl, shel.l ")

    assert result == ["AAPL", "MSFT", "SHEL.L"]


def test_parse_ticker_list_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="Enter at least one ticker symbol"):
        parse_ticker_list(" , ")


def test_fetch_yfinance_price_data_formats_multiple_tickers(monkeypatch):
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "MSFT"),
            ("Close", "AAPL"),
        ]
    )
    raw_data = pd.DataFrame(
        [[200.0, 100.0], [202.0, 101.0]],
        index=dates,
        columns=columns,
    )

    def fake_download(**kwargs):
        assert kwargs["tickers"] == ["AAPL", "MSFT"]
        assert kwargs["period"] == "1y"
        assert kwargs["interval"] == "1d"
        assert kwargs["auto_adjust"] is True
        return raw_data

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    result = fetch_yfinance_price_data(["AAPL", "MSFT"], period="1y")

    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "Ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "Close": [100.0, 200.0, 101.0, 202.0],
        }
    )

    assert result.equals(expected)


def test_fetch_yfinance_price_data_formats_single_ticker(monkeypatch):
    raw_data = pd.DataFrame(
        {"Close": [100.0, None, 102.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: raw_data),
    )

    result = fetch_yfinance_price_data(["AAPL"])

    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "Ticker": ["AAPL", "AAPL"],
            "Close": [100.0, 102.0],
        }
    )

    assert result.equals(expected)


def test_fetch_yfinance_price_data_raises_value_error_for_empty_download(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: pd.DataFrame()),
    )

    with pytest.raises(ValueError, match="No market data was returned by yfinance"):
        fetch_yfinance_price_data(["AAPL"])
