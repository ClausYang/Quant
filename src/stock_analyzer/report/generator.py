"""HTML report generator using Jinja2 templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from stock_analyzer.models import AnalysisResult, Market
from stock_analyzer.utils.formatting import change_css_class, format_change_pct, format_price

TEMPLATE_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output"

# Market prefix for output filenames
MARKET_PREFIX = {
    Market.US: "US",
    Market.HK: "HK",
    Market.A: "A",
    Market.ETF: "ETF",
}


def _score_display(score: float) -> str:
    """Format score for display: 4.0 -> '4', 4.5 -> '4.5'."""
    if score == int(score):
        return str(int(score))
    return str(score)


def _prepare_template_data(results: list[AnalysisResult]) -> dict:
    """Prepare template context data from analysis results."""
    now = datetime.now()
    report_id = f"analysis_results_{now.strftime('%Y%m%d_%H%M%S')}"
    generated_time = now.strftime("%Y-%m-%d %H:%M:%S")

    stocks = []
    for r in results:
        stocks.append({
            "name": r.stock.name,
            "code": r.stock.code,
            "score": _score_display(r.score),
            "score_display": _score_display(r.score),
            "action": r.action,
            "action_css": r.action_css_class,
            "sector": r.stock.sector or "N/A",
            "price": format_price(r.stock.price),
            "change_pct": format_change_pct(r.stock.change_pct),
            "change_css": change_css_class(r.stock.change_pct),
            "market_cap": r.stock.market_cap_str or "N/A",
            "market": r.stock.market.value,
            "add_price": r.price_levels.add_price_str,
            "reduce_price": r.price_levels.reduce_price_str,
            "trend_structure": r.texts.trend_structure,
            "macd_status": r.texts.macd_status,
            "kdj_status": r.texts.kdj_status,
            "analysis_reason": r.texts.analysis_reason,
        })

    # Collect distinct values for filters
    distinct_scores = sorted(set(s["score"] for s in stocks), key=lambda x: -float(x))
    distinct_actions = sorted(set(s["action"] for s in stocks))
    distinct_markets = sorted(set(s["market"] for s in stocks))

    return {
        "report_id": report_id,
        "generated_time": generated_time,
        "total_count": len(stocks),
        "stocks": stocks,
        "distinct_scores": distinct_scores,
        "distinct_actions": distinct_actions,
        "distinct_markets": distinct_markets,
    }


def generate_report(
    results: list[AnalysisResult],
    market: Market | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate an HTML report from analysis results.

    Args:
        results: List of analysis results to include.
        market: Market type for filename prefix. If None, uses "ALL".
        output_dir: Output directory. Defaults to project output/ dir.

    Returns:
        Path to the generated HTML file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("template.html")

    data = _prepare_template_data(results)
    html = template.render(**data)

    # Determine filename
    prefix = MARKET_PREFIX.get(market, "ALL") if market else "ALL"
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{prefix}_{date_str}_分析报告.html"
    output_path = output_dir / filename

    output_path.write_text(html, encoding="utf-8")
    return output_path
