"""Scoring algorithm: weighted composite score from technical indicators."""

from __future__ import annotations

import math

from stock_analyzer.models import (
    Alignment,
    CrossState,
    EMAResult,
    KDJResult,
    KDJZone,
    MACDResult,
    MomentumDirection,
)


def score_trend(ema: EMAResult) -> float:
    """Score trend structure (0-5) based on EMA alignment and price position."""
    score = 2.5  # neutral baseline

    # Overall alignment
    if ema.alignment == Alignment.BULLISH:
        score += 1.5
    elif ema.alignment == Alignment.BEARISH:
        score -= 1.5

    # Short-term alignment
    if ema.short_term_alignment == Alignment.BULLISH:
        score += 0.5
    elif ema.short_term_alignment == Alignment.BEARISH:
        score -= 0.5

    # Medium-term alignment
    if ema.medium_term_alignment == Alignment.BULLISH:
        score += 0.3
    elif ema.medium_term_alignment == Alignment.BEARISH:
        score -= 0.3

    # Price position
    if ema.price_above_all:
        score += 0.5
    elif ema.price_below_all:
        score -= 0.5

    return max(0.0, min(5.0, score))


def score_macd(macd: MACDResult) -> float:
    """Score MACD (0-5) based on momentum and cross state."""
    score = 2.5

    # Zero-axis position
    if macd.above_zero and macd.dea_above_zero:
        score += 0.8
    elif not macd.above_zero and not macd.dea_above_zero:
        score -= 0.8
    elif macd.above_zero:
        score += 0.3

    # Cross state
    if macd.cross_state == CrossState.GOLDEN:
        bonus = 1.0 if macd.days_since_cross <= 5 else 0.5
        score += bonus
    elif macd.cross_state == CrossState.DEATH:
        penalty = 1.0 if macd.days_since_cross <= 5 else 0.5
        score -= penalty

    # Momentum direction
    if macd.histogram_positive:
        if macd.momentum == MomentumDirection.EXPANDING:
            score += 0.5
        elif macd.momentum == MomentumDirection.CONTRACTING:
            score += 0.1
    else:
        if macd.momentum == MomentumDirection.EXPANDING:
            score -= 0.5
        elif macd.momentum == MomentumDirection.CONTRACTING:
            score -= 0.1

    return max(0.0, min(5.0, score))


def score_kdj(kdj: KDJResult) -> float:
    """Score KDJ (0-5) based on oscillator state and divergence."""
    score = 2.5

    # Zone
    if kdj.zone == KDJZone.OVERSOLD:
        # Oversold can be bullish opportunity if golden cross
        if kdj.cross_state == CrossState.GOLDEN:
            score += 1.0
        else:
            score -= 0.3
    elif kdj.zone == KDJZone.OVERBOUGHT:
        if kdj.cross_state == CrossState.DEATH:
            score -= 1.0
        else:
            score += 0.3

    # Cross state in neutral zone
    if kdj.zone == KDJZone.NEUTRAL:
        if kdj.cross_state == CrossState.GOLDEN:
            score += 0.8
        elif kdj.cross_state == CrossState.DEATH:
            score -= 0.8

    # Blunting
    if kdj.is_blunting:
        if kdj.zone == KDJZone.OVERBOUGHT:
            score += 0.3  # strong uptrend, blunting is bullish
        elif kdj.zone == KDJZone.OVERSOLD:
            score -= 0.3  # weak, blunting is bearish

    # Divergence
    if kdj.bullish_divergence:
        score += 1.0
    if kdj.bearish_divergence:
        score -= 1.0

    return max(0.0, min(5.0, score))


def score_context(ema: EMAResult, macd: MACDResult, kdj: KDJResult) -> float:
    """Score context/stability (0-5) based on indicator agreement."""
    score = 2.5
    bullish_signals = 0
    bearish_signals = 0

    # Count agreement
    if ema.alignment == Alignment.BULLISH:
        bullish_signals += 1
    elif ema.alignment == Alignment.BEARISH:
        bearish_signals += 1

    if macd.above_zero and macd.histogram_positive:
        bullish_signals += 1
    elif not macd.above_zero and not macd.histogram_positive:
        bearish_signals += 1

    if kdj.cross_state == CrossState.GOLDEN:
        bullish_signals += 1
    elif kdj.cross_state == CrossState.DEATH:
        bearish_signals += 1

    # Agreement bonus
    if bullish_signals == 3:
        score += 2.0
    elif bullish_signals == 2:
        score += 1.0
    elif bearish_signals == 3:
        score -= 2.0
    elif bearish_signals == 2:
        score -= 1.0

    return max(0.0, min(5.0, score))


def compute_score(
    ema: EMAResult,
    macd: MACDResult,
    kdj: KDJResult,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute composite score (1.0 - 5.0, step 0.5).

    Default weights: trend=0.4, macd=0.3, kdj=0.2, context=0.1
    """
    if weights is None:
        weights = {"trend": 0.4, "macd": 0.3, "kdj": 0.2, "context": 0.1}

    trend_s = score_trend(ema)
    macd_s = score_macd(macd)
    kdj_s = score_kdj(kdj)
    context_s = score_context(ema, macd, kdj)

    raw = (
        trend_s * weights["trend"]
        + macd_s * weights["macd"]
        + kdj_s * weights["kdj"]
        + context_s * weights["context"]
    )

    # Clamp to [1, 5] and round to nearest 0.5
    clamped = max(1.0, min(5.0, raw))
    rounded = round(clamped * 2) / 2
    return rounded
