"""Formatting utilities for market cap, prices, and percentages."""

from stock_analyzer.models import Market


def format_market_cap(value: float, market: Market) -> str:
    """Format market cap with appropriate unit for the market.

    Args:
        value: Raw market cap value (in the currency's base unit).
        market: Market type to determine currency suffix.
    """
    if value <= 0:
        return "N/A"

    suffix_map = {
        Market.US: "美元",
        Market.HK: "港元",
        Market.A: "",
        Market.ETF: "",
    }
    suffix = suffix_map.get(market, "")

    yi = 1e8  # 亿
    wan_yi = 1e12  # 万亿

    if value >= wan_yi:
        formatted = f"{value / wan_yi:.3f}万亿"
    elif value >= yi:
        formatted = f"{value / yi:.0f}亿"
    elif value >= 1e4:
        formatted = f"{value / 1e4:.2f}万"
    else:
        formatted = f"{value:.0f}"

    if suffix:
        formatted += suffix
    return formatted


def format_price(price: float) -> str:
    """Format price to appropriate decimal places."""
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.2f}"
    else:
        return f"{price:.3f}"


def format_change_pct(pct: float) -> str:
    """Format percentage change with sign and color class."""
    if pct > 0:
        return f"+{pct:.2f}%"
    elif pct < 0:
        return f"{pct:.2f}%"
    else:
        return "0.00%"


def change_css_class(pct: float) -> str:
    """Return CSS class for positive/negative change."""
    if pct > 0:
        return "positive"
    elif pct < 0:
        return "negative"
    return ""
