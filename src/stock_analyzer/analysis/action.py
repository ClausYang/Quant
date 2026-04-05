"""Trading action determination based on score and indicator state."""

from __future__ import annotations

from stock_analyzer.models import (
    Alignment,
    CrossState,
    EMAResult,
    KDJResult,
    KDJZone,
    MACDResult,
    MomentumDirection,
)


def determine_action(
    score: float,
    ema: EMAResult,
    macd: MACDResult,
    kdj: KDJResult,
) -> tuple[str, str]:
    """Determine trading action and CSS class.

    Returns:
        (action_text, css_class) where css_class is "long" or "hold".
    """
    # Score >= 4.5: strong bullish
    if score >= 4.5:
        if macd.cross_state == CrossState.GOLDEN and macd.days_since_cross <= 5:
            return "多头", "long"
        if ema.alignment == Alignment.BULLISH and macd.above_zero:
            return "多头", "long"
        return "多头（观察/轻仓试探）", "long"

    # Score 4.0
    if score >= 4.0:
        if macd.cross_state == CrossState.GOLDEN and macd.above_zero:
            return "多头", "long"
        if ema.short_term_alignment == Alignment.BULLISH:
            if macd.histogram_positive and macd.momentum == MomentumDirection.EXPANDING:
                return "多头", "long"
            return "多头（观察/轻仓试探）", "long"
        return "多头（观察/轻仓试探）", "long"

    # Score 3.5
    if score >= 3.5:
        if (
            macd.cross_state == CrossState.GOLDEN
            and ema.short_term_alignment == Alignment.BULLISH
        ):
            return "观察/轻仓试探", "hold"
        if kdj.bullish_divergence:
            return "观察/轻仓试探", "hold"
        return "观察", "hold"

    # Score 3.0
    if score >= 3.0:
        if kdj.zone == KDJZone.OVERSOLD and kdj.cross_state == CrossState.GOLDEN:
            return "观察/轻仓试探", "hold"
        return "不交易/观察中", "hold"

    # Score 2.0-2.5
    if score >= 2.0:
        if macd.momentum == MomentumDirection.CONTRACTING and not macd.histogram_positive:
            return "不交易（观察中）", "hold"
        return "不交易", "hold"

    # Score 1.0-1.5
    return "不交易", "hold"
