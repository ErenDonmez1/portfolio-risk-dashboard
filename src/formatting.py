"""Display formatting shared by the dashboard and charts."""


def format_currency(value: float) -> str:
    """Format a portfolio value in pounds sterling without converting it."""
    return f"£{value:,.2f}"
