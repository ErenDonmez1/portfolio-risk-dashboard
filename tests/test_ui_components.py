"""Tests for reusable Streamlit display helpers."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ui_components as ui


def test_styled_table_hides_dataframe_index_by_default(monkeypatch):
    rendered_html = {}

    def capture_markdown(content, **_kwargs):
        rendered_html["content"] = content

    monkeypatch.setattr(ui.st, "markdown", capture_markdown)

    ui.styled_table(pd.DataFrame({"Ticker": ["ALPHA"]}))

    assert "<th>0</th>" not in rendered_html["content"]
