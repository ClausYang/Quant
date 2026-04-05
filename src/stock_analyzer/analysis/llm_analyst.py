"""LLM-based analysis text generation with rule-based fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from stock_analyzer.models import (
    Alignment,
    AnalysisTexts,
    CrossState,
    EMAResult,
    KDJResult,
    KDJZone,
    MACDResult,
    MomentumDirection,
    StockData,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一位专业的右侧交易技术分析师。你的分析风格特点：
1. 严格遵循右侧交易原则，顺势而为，不做左侧摸底
2. 使用专业中文金融术语（多头排列、空头排列、金叉、死叉、零轴、钝化、背离等）
3. 分析简洁有力，用分号分隔要点
4. 重视赔率和胜率的评估
5. 对趋势不明确的标的保持观望态度

请严格按照以下格式输出4段分析，每段一行，不要加标题或编号：
第1行：趋势结构分析（基于EMA排列和价格位置）
第2行：MACD状态分析（基于DIF/DEA/柱状图）
第3行：KDJ状态分析（基于K/D/J值和交叉）
第4行：综合分析原因（结合所有指标给出交易建议理由，提及赔率和风险收益比）

每行30-80个汉字，用分号分隔要点。不要使用换行符或空行。"""


def _build_indicator_prompt(
    stock: StockData,
    ema: EMAResult,
    macd: MACDResult,
    kdj: KDJResult,
    score: float,
    action: str,
) -> str:
    """Build the user prompt with all indicator data."""
    return f"""\
股票: {stock.name} ({stock.code})
市场: {stock.market.value}
当前价格: {stock.price}
涨跌幅: {stock.change_pct:+.2f}%
评分: {score}/5
交易动作: {action}

【EMA系统】
EMA5={ema.ema5}, EMA10={ema.ema10}, EMA20={ema.ema20}, EMA30={ema.ema30}
EMA55={ema.ema55}, EMA120={ema.ema120}, EMA144={ema.ema144}, EMA169={ema.ema169}, EMA200={ema.ema200}
整体排列: {ema.alignment.value}
短期排列(5/10/20): {ema.short_term_alignment.value}
中期排列(20/55): {ema.medium_term_alignment.value}
长期排列(120+): {ema.long_term_alignment.value}
价格在所有均线之上: {ema.price_above_all}
价格在所有均线之下: {ema.price_below_all}
最近支撑: {ema.support_ema_label}={ema.nearest_support}
最近阻力: {ema.resistance_ema_label}={ema.nearest_resistance}

【MACD】
DIF={macd.dif}, DEA={macd.dea}, 柱状图={macd.histogram}
DIF在零轴上方: {macd.above_zero}
交叉状态: {macd.cross_state.value}（{macd.days_since_cross}天前）
动量方向: {macd.momentum.value}
柱状图为正: {macd.histogram_positive}

【KDJ】
K={kdj.k}, D={kdj.d}, J={kdj.j}
区域: {kdj.zone.value}
交叉状态: {kdj.cross_state.value}
钝化: {kdj.is_blunting}
底背离: {kdj.bullish_divergence}
顶背离: {kdj.bearish_divergence}

请生成4行分析文本。"""


async def _call_openai_compatible(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Call any OpenAI-compatible API (MiniMax, DeepSeek, Qwen, etc.)."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


async def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def generate_analysis_llm(
    stock: StockData,
    ema: EMAResult,
    macd: MACDResult,
    kdj: KDJResult,
    score: float,
    action: str,
    model: str = "claude-sonnet-4-20250514",
    semaphore: Optional[asyncio.Semaphore] = None,
    provider: str = "anthropic",
    base_url: str = "",
) -> AnalysisTexts:
    """Generate analysis texts using LLM API.

    Supports multiple providers via `provider` parameter:
    - "anthropic": Claude API (requires ANTHROPIC_API_KEY)
    - "minimax":   MiniMax API (requires MINIMAX_API_KEY)
    - "openai":    OpenAI API (requires OPENAI_API_KEY)
    - "deepseek":  DeepSeek API (requires DEEPSEEK_API_KEY)
    - "custom":    Any OpenAI-compatible API (requires LLM_API_KEY + LLM_BASE_URL)
    """
    # --- Resolve API key and base_url by provider ---
    provider_configs = {
        "anthropic": {"key_env": "ANTHROPIC_API_KEY", "base_url": ""},
        "minimax":   {"key_env": "MINIMAX_API_KEY",   "base_url": "https://api.minimax.chat/v1"},
        "openai":    {"key_env": "OPENAI_API_KEY",     "base_url": "https://api.openai.com/v1"},
        "deepseek":  {"key_env": "DEEPSEEK_API_KEY",   "base_url": "https://api.deepseek.com/v1"},
        "qwen":      {"key_env": "QWEN_API_KEY",       "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        "custom":    {"key_env": "LLM_API_KEY",        "base_url": base_url},
    }

    cfg = provider_configs.get(provider, provider_configs["anthropic"])
    api_key = os.environ.get(cfg["key_env"], "")
    resolved_base_url = base_url or cfg["base_url"]

    if not api_key:
        logger.warning(f"{cfg['key_env']} not set, falling back to template mode")
        return generate_analysis_template(ema, macd, kdj, score, action)

    prompt = _build_indicator_prompt(stock, ema, macd, kdj, score, action)

    async def _call():
        if provider == "anthropic":
            return await _call_anthropic(prompt, model, api_key)
        else:
            return await _call_openai_compatible(prompt, model, api_key, resolved_base_url)

    try:
        if semaphore:
            async with semaphore:
                text = await _call()
        else:
            text = await _call()

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        while len(lines) < 4:
            lines.append("")

        return AnalysisTexts(
            trend_structure=lines[0],
            macd_status=lines[1],
            kdj_status=lines[2],
            analysis_reason=lines[3],
        )

    except Exception as e:
        logger.error(f"LLM analysis failed for {stock.code}: {e}")
        return generate_analysis_template(ema, macd, kdj, score, action)


def generate_analysis_template(
    ema: EMAResult,
    macd: MACDResult,
    kdj: KDJResult,
    score: float,
    action: str,
) -> AnalysisTexts:
    """Generate analysis texts using rule-based templates (fallback)."""

    # --- Trend Structure ---
    trend_parts = []
    if ema.alignment == Alignment.BULLISH:
        trend_parts.append("均线系统呈标准多头排列")
        if ema.price_above_all:
            trend_parts.append("价格站稳所有均线之上，趋势强劲")
        else:
            trend_parts.append(f"价格回踩至{ema.resistance_ema_label}附近")
    elif ema.alignment == Alignment.BEARISH:
        trend_parts.append("均线系统呈标准空头排列")
        if ema.price_below_all:
            trend_parts.append("价格处于所有均线之下，空头动能强劲")
        else:
            trend_parts.append(f"价格反弹至{ema.resistance_ema_label}附近受阻")
    else:
        trend_parts.append("均线系统呈混合排列")
        if ema.short_term_alignment == Alignment.BULLISH:
            trend_parts.append("短期均线多头排列，短线偏强")
        elif ema.short_term_alignment == Alignment.BEARISH:
            trend_parts.append("短期均线空头排列，短线偏弱")
        else:
            trend_parts.append("短期方向不明，多空交织")

    if ema.nearest_support:
        trend_parts.append(f"下方{ema.support_ema_label}提供支撑")

    trend_text = "；".join(trend_parts)

    # --- MACD Status ---
    macd_parts = []
    axis = "零轴上方" if macd.above_zero else "零轴下方"

    if macd.cross_state == CrossState.GOLDEN:
        if macd.above_zero:
            macd_parts.append(f"{axis}金叉运行，属于强势信号")
        else:
            macd_parts.append(f"{axis}金叉形成，关注能否突破零轴")
    elif macd.cross_state == CrossState.DEATH:
        if not macd.above_zero:
            macd_parts.append(f"{axis}死叉运行，空头动能延续")
        else:
            macd_parts.append(f"{axis}死叉形成，多头动能减弱")
    else:
        macd_parts.append(f"MACD在{axis}运行")

    if macd.histogram_positive:
        if macd.momentum == MomentumDirection.EXPANDING:
            macd_parts.append("红柱持续放大，多头动能增强")
        elif macd.momentum == MomentumDirection.CONTRACTING:
            macd_parts.append("红柱逐渐缩短，上涨动能衰减")
        else:
            macd_parts.append("红柱维持，动能平稳")
    else:
        if macd.momentum == MomentumDirection.EXPANDING:
            macd_parts.append("绿柱持续放大，空头动能增强")
        elif macd.momentum == MomentumDirection.CONTRACTING:
            macd_parts.append("绿柱逐渐缩短，下跌动能衰减")
        else:
            macd_parts.append("绿柱维持，空头动能平稳")

    macd_text = "；".join(macd_parts)

    # --- KDJ Status ---
    kdj_parts = []

    if kdj.zone == KDJZone.OVERBOUGHT:
        kdj_parts.append(f"K={kdj.k:.0f}，D={kdj.d:.0f}，J={kdj.j:.0f}，处于超买区域")
        if kdj.is_blunting:
            kdj_parts.append("高位钝化中，强势特征明显")
    elif kdj.zone == KDJZone.OVERSOLD:
        kdj_parts.append(f"K={kdj.k:.0f}，D={kdj.d:.0f}，J={kdj.j:.0f}，处于超卖区域")
        if kdj.is_blunting:
            kdj_parts.append("低位钝化中，弱势特征明显")
    else:
        kdj_parts.append(f"K={kdj.k:.0f}，D={kdj.d:.0f}，J={kdj.j:.0f}，处于中性区域")

    if kdj.cross_state == CrossState.GOLDEN:
        kdj_parts.append("金叉信号形成")
    elif kdj.cross_state == CrossState.DEATH:
        kdj_parts.append("死叉信号形成")

    if kdj.bullish_divergence:
        kdj_parts.append("出现底背离信号，关注反转机会")
    if kdj.bearish_divergence:
        kdj_parts.append("出现顶背离信号，警惕回调风险")

    kdj_text = "；".join(kdj_parts)

    # --- Analysis Reason ---
    reason_parts = []

    if score >= 4:
        reason_parts.append("多项技术指标共振向好，右侧趋势确认")
        if ema.alignment == Alignment.BULLISH and macd.above_zero:
            reason_parts.append("均线多头排列叠加MACD零轴上方运行，多头格局稳固")
        reason_parts.append("赔率较高，风险收益比合理，可以积极参与")
    elif score >= 3:
        reason_parts.append("技术面信号混合，趋势尚未完全确认")
        if ema.short_term_alignment == Alignment.BULLISH:
            reason_parts.append("短期有企稳迹象，但中长期方向待确认")
        reason_parts.append("赔率一般，建议观望等待右侧信号明确")
    elif score >= 2:
        reason_parts.append("技术面偏弱，多项指标显示下行压力")
        reason_parts.append("目前处于左侧区域，赔率较低，不建议盲目入场")
    else:
        reason_parts.append("技术面全面走弱，均线空头排列，MACD空头运行")
        reason_parts.append("处于右侧下行趋势中，赔率极低，严禁盲目摸底")

    reason_text = "；".join(reason_parts)

    return AnalysisTexts(
        trend_structure=trend_text,
        macd_status=macd_text,
        kdj_status=kdj_text,
        analysis_reason=reason_text,
    )
