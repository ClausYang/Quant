"""Core data models for the stock analysis pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Market(str, Enum):
    US = "美股"
    HK = "港股"
    A = "A股"
    ETF = "ETF"


class Alignment(str, Enum):
    BULLISH = "多头排列"
    BEARISH = "空头排列"
    MIXED = "混合排列"


class CrossState(str, Enum):
    GOLDEN = "金叉"
    DEATH = "死叉"
    NONE = "无交叉"


class KDJZone(str, Enum):
    OVERBOUGHT = "超买区"
    OVERSOLD = "超卖区"
    NEUTRAL = "中性区"


class MomentumDirection(str, Enum):
    EXPANDING = "放大"
    CONTRACTING = "收缩"
    FLAT = "持平"


# --- Stock Data ---

class StockConfig(BaseModel):
    """Stock configuration from stocks.yaml."""
    code: str
    name: str
    market: Market
    sector: str = ""
    sector_override: str = ""


class StockData(BaseModel):
    """Raw stock data fetched from market APIs."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    name: str
    market: Market
    sector: str
    price: float
    change_pct: float
    market_cap: float  # raw value in local currency
    market_cap_str: str = ""  # formatted string like "2342亿"
    ohlcv: pd.DataFrame  # columns: Open, High, Low, Close, Volume


# --- Indicator Results ---

class EMAResult(BaseModel):
    """Result of EMA system calculation."""
    ema5: float
    ema10: float
    ema20: float
    ema30: float
    ema55: float
    ema120: float
    ema144: float
    ema169: float
    ema200: float

    alignment: Alignment
    short_term_alignment: Alignment  # EMA5/10/20
    medium_term_alignment: Alignment  # EMA20/55
    long_term_alignment: Alignment  # EMA120+

    price_above_all: bool
    price_below_all: bool
    nearest_support: Optional[float] = None  # nearest EMA below price
    nearest_resistance: Optional[float] = None  # nearest EMA above price
    support_ema_label: str = ""
    resistance_ema_label: str = ""


class MACDResult(BaseModel):
    """Result of MACD calculation."""
    dif: float
    dea: float
    histogram: float

    above_zero: bool  # DIF above zero axis
    dea_above_zero: bool
    cross_state: CrossState
    days_since_cross: int = 0
    momentum: MomentumDirection  # histogram expanding or contracting
    histogram_positive: bool


class KDJResult(BaseModel):
    """Result of KDJ calculation."""
    k: float
    d: float
    j: float

    zone: KDJZone
    cross_state: CrossState
    is_blunting: bool  # stuck in extreme zone
    bullish_divergence: bool  # price lower low, KDJ higher low
    bearish_divergence: bool  # price higher high, KDJ lower high


# --- Analysis Results ---

class AnalysisTexts(BaseModel):
    """LLM-generated analysis texts for the 4 sections."""
    trend_structure: str = ""  # 趋势结构
    macd_status: str = ""  # MACD状态
    kdj_status: str = ""  # KDJ状态
    analysis_reason: str = ""  # 分析原因


class PriceLevels(BaseModel):
    """Add/reduce position price levels."""
    add_price: Optional[float] = None  # 加仓价格
    reduce_price: Optional[float] = None  # 减仓价格
    add_price_str: str = "不适用"
    reduce_price_str: str = "不适用"


class AnalysisResult(BaseModel):
    """Complete analysis result for a single stock."""
    stock: StockData
    ema: EMAResult
    macd: MACDResult
    kdj: KDJResult
    score: float  # 1.0 - 5.0, step 0.5
    action: str  # e.g. "多头", "不交易"
    action_css_class: str  # "long" or "hold"
    texts: AnalysisTexts
    price_levels: PriceLevels
