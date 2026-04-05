"""EMA (Exponential Moving Average) system calculation."""

from __future__ import annotations

import pandas as pd

from stock_analyzer.models import Alignment, EMAResult

# Period -> label mapping
EMA_LABELS = {
    5: "EMA5", 10: "EMA10", 20: "EMA20", 30: "EMA30",
    55: "EMA55", 120: "EMA120", 144: "EMA144", 169: "EMA169", 200: "EMA200",
}

DEFAULT_PERIODS = [5, 10, 20, 30, 55, 120, 144, 169, 200]


def calculate_emas(df: pd.DataFrame, periods: list[int] | None = None) -> dict[int, pd.Series]:
    """Calculate EMA for all given periods on the Close column."""
    if periods is None:
        periods = DEFAULT_PERIODS
    close = df["Close"]
    return {p: close.ewm(span=p, adjust=False).mean() for p in periods}


def classify_alignment(values: list[float]) -> Alignment:
    """Classify alignment of a list of EMA values (ordered short to long).

    Bullish: each shorter EMA is above the longer one.
    Bearish: each shorter EMA is below the longer one.
    """
    if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
        return Alignment.BULLISH
    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
        return Alignment.BEARISH
    return Alignment.MIXED


def compute_ema(df: pd.DataFrame, periods: list[int] | None = None) -> EMAResult:
    """Compute the full EMA system result for a stock's OHLCV data."""
    if periods is None:
        periods = DEFAULT_PERIODS

    emas = calculate_emas(df, periods)
    current_price = float(df["Close"].iloc[-1])

    # Get latest EMA values
    latest = {p: float(emas[p].iloc[-1]) for p in periods}

    # Alignment classification
    all_vals = [latest[p] for p in sorted(periods)]
    short_vals = [latest[p] for p in [5, 10, 20] if p in latest]
    medium_vals = [latest[p] for p in [20, 55] if p in latest]
    long_vals = [latest[p] for p in [120, 144, 169, 200] if p in latest]

    alignment = classify_alignment(all_vals)
    short_alignment = classify_alignment(short_vals) if len(short_vals) >= 2 else Alignment.MIXED
    medium_alignment = classify_alignment(medium_vals) if len(medium_vals) >= 2 else Alignment.MIXED
    long_alignment = classify_alignment(long_vals) if len(long_vals) >= 2 else Alignment.MIXED

    # Price position relative to EMAs
    emas_below = [(p, v) for p, v in latest.items() if v < current_price]
    emas_above = [(p, v) for p, v in latest.items() if v > current_price]

    price_above_all = len(emas_above) == 0 and len(emas_below) > 0
    price_below_all = len(emas_below) == 0 and len(emas_above) > 0

    # Nearest support (highest EMA below price)
    nearest_support = None
    support_label = ""
    if emas_below:
        best = max(emas_below, key=lambda x: x[1])
        nearest_support = best[1]
        support_label = EMA_LABELS.get(best[0], f"EMA{best[0]}")

    # Nearest resistance (lowest EMA above price)
    nearest_resistance = None
    resistance_label = ""
    if emas_above:
        best = min(emas_above, key=lambda x: x[1])
        nearest_resistance = best[1]
        resistance_label = EMA_LABELS.get(best[0], f"EMA{best[0]}")

    return EMAResult(
        ema5=round(latest.get(5, 0), 4),
        ema10=round(latest.get(10, 0), 4),
        ema20=round(latest.get(20, 0), 4),
        ema30=round(latest.get(30, 0), 4),
        ema55=round(latest.get(55, 0), 4),
        ema120=round(latest.get(120, 0), 4),
        ema144=round(latest.get(144, 0), 4),
        ema169=round(latest.get(169, 0), 4),
        ema200=round(latest.get(200, 0), 4),
        alignment=alignment,
        short_term_alignment=short_alignment,
        medium_term_alignment=medium_alignment,
        long_term_alignment=long_alignment,
        price_above_all=price_above_all,
        price_below_all=price_below_all,
        nearest_support=round(nearest_support, 4) if nearest_support else None,
        nearest_resistance=round(nearest_resistance, 4) if nearest_resistance else None,
        support_ema_label=support_label,
        resistance_ema_label=resistance_label,
    )
