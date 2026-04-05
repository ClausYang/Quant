"""MACD (Moving Average Convergence Divergence) calculation."""

from __future__ import annotations

import pandas as pd

from stock_analyzer.models import CrossState, MACDResult, MomentumDirection


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """Compute MACD indicators.

    DIF = EMA(close, fast) - EMA(close, slow)
    DEA = EMA(DIF, signal)
    Histogram = 2 * (DIF - DEA)
    """
    close = df["Close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    histogram = 2 * (dif - dea)

    curr_dif = float(dif.iloc[-1])
    curr_dea = float(dea.iloc[-1])
    curr_hist = float(histogram.iloc[-1])

    # Zero-axis position
    above_zero = curr_dif > 0
    dea_above_zero = curr_dea > 0

    # Cross state detection (look back for recent cross)
    cross_state = CrossState.NONE
    days_since_cross = 0

    for i in range(len(dif) - 1, max(len(dif) - 30, 0), -1):
        prev_diff = float(dif.iloc[i - 1] - dea.iloc[i - 1])
        curr_diff = float(dif.iloc[i] - dea.iloc[i])

        if prev_diff <= 0 < curr_diff:
            cross_state = CrossState.GOLDEN
            days_since_cross = len(dif) - 1 - i
            break
        elif prev_diff >= 0 > curr_diff:
            cross_state = CrossState.DEATH
            days_since_cross = len(dif) - 1 - i
            break

    # Momentum direction (histogram trend)
    if len(histogram) >= 3:
        recent = [float(histogram.iloc[j]) for j in range(-3, 0)]
        abs_recent = [abs(v) for v in recent]
        if abs_recent[-1] > abs_recent[-2] > abs_recent[-3]:
            momentum = MomentumDirection.EXPANDING
        elif abs_recent[-1] < abs_recent[-2] < abs_recent[-3]:
            momentum = MomentumDirection.CONTRACTING
        else:
            momentum = MomentumDirection.FLAT
    else:
        momentum = MomentumDirection.FLAT

    return MACDResult(
        dif=round(curr_dif, 4),
        dea=round(curr_dea, 4),
        histogram=round(curr_hist, 4),
        above_zero=above_zero,
        dea_above_zero=dea_above_zero,
        cross_state=cross_state,
        days_since_cross=days_since_cross,
        momentum=momentum,
        histogram_positive=curr_hist > 0,
    )
