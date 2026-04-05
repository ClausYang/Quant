"""Add/reduce position price level calculation based on EMA support/resistance."""

from __future__ import annotations

from stock_analyzer.models import EMAResult, PriceLevels


def compute_price_levels(
    current_price: float,
    ema: EMAResult,
    score: float,
    action: str,
) -> PriceLevels:
    """Compute add-position and reduce-position price levels.

    Add price: based on nearest EMA support (buy on pullback to support).
    Reduce price: based on nearest EMA resistance or recent swing high.
    """
    add_price = None
    reduce_price = None

    # For bullish actions, calculate add/reduce levels
    if "多头" in action or "观察" in action:
        # Add price: nearest support EMA below current price
        if ema.nearest_support is not None:
            add_price = round(ema.nearest_support, 2)

        # Reduce price: if price is above resistance, use a trailing stop
        # Otherwise use the nearest resistance above
        if ema.nearest_resistance is not None:
            reduce_price = round(ema.nearest_resistance, 2)
        elif ema.price_above_all:
            # Price above all EMAs: use EMA5 as trailing stop
            reduce_price = round(ema.ema5, 2)
    else:
        # For "不交易" stocks
        # Reduce price: nearest support breakdown level
        if ema.nearest_support is not None:
            reduce_price = round(ema.nearest_support * 0.97, 2)  # 3% below support

    # Format strings
    add_str = f"{add_price:.2f}" if add_price is not None else "不适用"
    reduce_str = f"{reduce_price:.2f}" if reduce_price is not None else "不适用"

    return PriceLevels(
        add_price=add_price,
        reduce_price=reduce_price,
        add_price_str=add_str,
        reduce_price_str=reduce_str,
    )
