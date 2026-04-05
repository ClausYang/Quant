"""KDJ oscillator calculation using the Chinese SMA formula."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analyzer.models import CrossState, KDJResult, KDJZone


def chinese_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """Chinese-style SMA used in KDJ calculation.

    SMA(X, N, M) = (M * X + (N - M) * prev_SMA) / N

    This differs from standard Western EMA/SMA. It is the formula used
    by Chinese trading platforms (通达信, 同花顺, etc.).
    """
    result = np.zeros(len(series))
    result[0] = float(series.iloc[0])
    for i in range(1, len(series)):
        result[i] = (m * float(series.iloc[i]) + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


def compute_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> KDJResult:
    """Compute KDJ oscillator.

    RSV = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100
    K = SMA(RSV, M1, 1)
    D = SMA(K, M2, 1)
    J = 3K - 2D
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # RSV calculation
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()

    hl_range = highest_high - lowest_low
    # Avoid division by zero
    hl_range = hl_range.replace(0, np.nan).ffill().fillna(1)
    rsv = (close - lowest_low) / hl_range * 100

    # K, D, J using Chinese SMA formula
    k = chinese_sma(rsv, m1, 1)
    d = chinese_sma(k, m2, 1)
    j = 3 * k - 2 * d

    curr_k = float(k.iloc[-1])
    curr_d = float(d.iloc[-1])
    curr_j = float(j.iloc[-1])

    # Zone classification
    if curr_k > 80 and curr_d > 80:
        zone = KDJZone.OVERBOUGHT
    elif curr_k < 20 and curr_d < 20:
        zone = KDJZone.OVERSOLD
    else:
        zone = KDJZone.NEUTRAL

    # Cross state (look back 30 bars)
    cross_state = CrossState.NONE
    for i in range(len(k) - 1, max(len(k) - 30, 0), -1):
        prev_diff = float(k.iloc[i - 1] - d.iloc[i - 1])
        curr_diff = float(k.iloc[i] - d.iloc[i])

        if prev_diff <= 0 < curr_diff:
            cross_state = CrossState.GOLDEN
            break
        elif prev_diff >= 0 > curr_diff:
            cross_state = CrossState.DEATH
            break

    # Blunting detection: K/D stuck in extreme zone for 5+ days
    is_blunting = False
    lookback = min(5, len(k))
    if lookback >= 5:
        recent_k = [float(k.iloc[-i]) for i in range(1, lookback + 1)]
        if all(v > 80 for v in recent_k) or all(v < 20 for v in recent_k):
            is_blunting = True

    # Divergence detection (look back ~30 bars for swing lows/highs)
    bullish_divergence = _detect_bullish_divergence(close, k, lookback=60)
    bearish_divergence = _detect_bearish_divergence(close, k, lookback=60)

    return KDJResult(
        k=round(curr_k, 2),
        d=round(curr_d, 2),
        j=round(curr_j, 2),
        zone=zone,
        cross_state=cross_state,
        is_blunting=is_blunting,
        bullish_divergence=bullish_divergence,
        bearish_divergence=bearish_divergence,
    )


def _find_swing_lows(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """Find local minima in a series."""
    lows = []
    for i in range(window, len(series) - window):
        val = float(series.iloc[i])
        if all(val <= float(series.iloc[j]) for j in range(i - window, i + window + 1)):
            lows.append((i, val))
    return lows


def _find_swing_highs(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """Find local maxima in a series."""
    highs = []
    for i in range(window, len(series) - window):
        val = float(series.iloc[i])
        if all(val >= float(series.iloc[j]) for j in range(i - window, i + window + 1)):
            highs.append((i, val))
    return highs


def _detect_bullish_divergence(
    close: pd.Series, k: pd.Series, lookback: int = 60
) -> bool:
    """Detect bullish divergence: price makes lower low, K makes higher low."""
    recent_close = close.iloc[-lookback:]
    recent_k = k.iloc[-lookback:]

    price_lows = _find_swing_lows(recent_close)
    k_lows = _find_swing_lows(recent_k)

    if len(price_lows) >= 2 and len(k_lows) >= 2:
        # Price: lower low
        if price_lows[-1][1] < price_lows[-2][1]:
            # K: higher low
            if k_lows[-1][1] > k_lows[-2][1]:
                return True
    return False


def _detect_bearish_divergence(
    close: pd.Series, k: pd.Series, lookback: int = 60
) -> bool:
    """Detect bearish divergence: price makes higher high, K makes lower high."""
    recent_close = close.iloc[-lookback:]
    recent_k = k.iloc[-lookback:]

    price_highs = _find_swing_highs(recent_close)
    k_highs = _find_swing_highs(recent_k)

    if len(price_highs) >= 2 and len(k_highs) >= 2:
        # Price: higher high
        if price_highs[-1][1] > price_highs[-2][1]:
            # K: lower high
            if k_highs[-1][1] < k_highs[-2][1]:
                return True
    return False
