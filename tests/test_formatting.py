from src.formatting import format_currency


def test_format_currency_uses_pounds_without_changing_value():
    assert format_currency(12345.6) == "£12,345.60"
