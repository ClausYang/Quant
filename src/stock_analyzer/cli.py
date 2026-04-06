"""CLI interface for the stock analyzer."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import defaultdict

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from stock_analyzer.analysis.action import determine_action
from stock_analyzer.analysis.llm_analyst import (
    generate_analysis_llm,
    generate_analysis_template,
)
from stock_analyzer.analysis.price_levels import compute_price_levels
from stock_analyzer.analysis.scorer import compute_score
from stock_analyzer.config import Settings
from stock_analyzer.data.fetcher import FetcherFactory
from stock_analyzer.indicators.ema import compute_ema
from stock_analyzer.indicators.kdj import compute_kdj
from stock_analyzer.indicators.macd import compute_macd
from stock_analyzer.models import AnalysisResult, AnalysisTexts, Market, StockConfig, StockData
from stock_analyzer.report.generator import generate_report

console = Console()

MARKET_CHOICES = ["us", "hk", "a", "etf", "all"]
MARKET_MAP = {"us": Market.US, "hk": Market.HK, "a": Market.A, "etf": Market.ETF}


def _lookup_stock_name(code: str, market: Market) -> str:
    """Try to resolve stock name from code via data provider."""
    try:
        if market == Market.A:
            import baostock as bs
            from stock_analyzer.data.a_fetcher import _to_bs_code
            bs.login()
            rs = bs.query_stock_basic(code=_to_bs_code(code))
            name = code
            if rs.error_code == "0" and rs.next():
                name = rs.get_row_data()[1] or code
            bs.logout()
            return name
        if market == Market.HK:
            import yfinance as yf
            ticker = yf.Ticker(f"{code.zfill(5)}.HK")
            info = ticker.info
            return info.get("longName") or info.get("shortName") or code
        if market == Market.US:
            import yfinance as yf
            ticker = yf.Ticker(code)
            info = ticker.info
            return info.get("shortName") or code
    except Exception:
        pass
    return code


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
def cli():
    """Stock Analyzer - Multi-market technical analysis tool."""
    pass


@cli.command()
@click.option(
    "--market", "-m",
    type=click.Choice(MARKET_CHOICES, case_sensitive=False),
    default="all",
    help="Market to analyze (us/hk/a/etf/all).",
)
@click.option("--portfolio-only", is_flag=True, help="Only analyze portfolio stocks.")
@click.option("--code", "-c", default=None, help="Single stock code to analyze (e.g. 600519). Requires --market.")
@click.option("--name", "-n", default=None, help="Stock name for --code (optional, defaults to code).")
@click.option(
    "--mode",
    type=click.Choice(["llm", "template"], case_sensitive=False),
    default=None,
    help="Analysis text generation mode. Overrides config.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
def run(market: str, portfolio_only: bool, code: str | None, name: str | None, mode: str | None, verbose: bool):
    """Run the full analysis pipeline and generate HTML reports."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    settings = Settings()
    analysis_mode = mode or settings.analysis_mode

    # Determine which markets to process
    if market == "all":
        markets = [Market.US, Market.HK, Market.A, Market.ETF]
    else:
        markets = [MARKET_MAP[market]]

    # Get stock configs grouped by market
    stocks_by_market: dict[Market, list[StockConfig]] = defaultdict(list)
    for mkt in markets:
        mkt_key = {Market.US: "us", Market.HK: "hk", Market.A: "a", Market.ETF: "etf"}[mkt]
        if code:
            resolved_name = name or _lookup_stock_name(code, mkt)
            configs = [StockConfig(code=code, name=resolved_name, market=mkt)]
        else:
            configs = settings.get_stocks(mkt_key)
        if portfolio_only and not code:
            # Filter to portfolio only (exclude watchlist)
            portfolio_codes = set()
            for item in (settings.stocks_config.get("portfolio", {}).get(mkt_key, []) or []):
                portfolio_codes.add(str(item["code"]))
            configs = [c for c in configs if c.code in portfolio_codes]
        stocks_by_market[mkt] = configs

    # Apply sector overrides
    overrides = settings.get_sector_overrides()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        all_results_by_market: dict[Market, list[AnalysisResult]] = defaultdict(list)

        for mkt, configs in stocks_by_market.items():
            if not configs:
                continue

            # Apply sector overrides
            for cfg in configs:
                if cfg.code in overrides and not cfg.sector:
                    cfg.sector = overrides[cfg.code]

            # --- Step 1: Fetch data ---
            task = progress.add_task(f"[cyan]获取{mkt.value}数据 ({len(configs)}只)...", total=None)
            fetcher = FetcherFactory.get_fetcher(mkt)
            stock_data_list = fetcher.fetch(configs)
            progress.update(task, completed=True, description=f"[green]✓ {mkt.value}数据获取完成 ({len(stock_data_list)}只)")

            if not stock_data_list:
                logger.warning(f"No data fetched for {mkt.value}, skipping")
                continue

            # --- Step 2: Calculate indicators ---
            task2 = progress.add_task(f"[cyan]计算{mkt.value}技术指标...", total=None)
            indicator_results = []
            for sd in stock_data_list:
                try:
                    if len(sd.ohlcv) < 30:
                        logger.warning(f"Insufficient data for {sd.code} ({len(sd.ohlcv)} bars), skipping")
                        continue
                    ema_result = compute_ema(sd.ohlcv, settings.ema_periods)
                    macd_result = compute_macd(sd.ohlcv, **settings.macd_params)
                    kdj_result = compute_kdj(sd.ohlcv, **settings.kdj_params)
                    indicator_results.append((sd, ema_result, macd_result, kdj_result))
                except Exception as e:
                    logger.error(f"Indicator calculation failed for {sd.code}: {e}")
            progress.update(task2, completed=True, description=f"[green]✓ {mkt.value}指标计算完成")

            # --- Step 3: Score and determine actions ---
            task3 = progress.add_task(f"[cyan]评分与动作判定...", total=None)
            scored = []
            for sd, ema, macd, kdj in indicator_results:
                score = compute_score(ema, macd, kdj, settings.scoring_weights)
                action, action_css = determine_action(score, ema, macd, kdj)
                price_levels = compute_price_levels(sd.price, ema, score, action)
                scored.append((sd, ema, macd, kdj, score, action, action_css, price_levels))
            progress.update(task3, completed=True, description=f"[green]✓ 评分完成")

            # --- Step 4: Generate analysis texts ---
            task4 = progress.add_task(f"[cyan]生成{mkt.value}分析文本 ({analysis_mode})...", total=None)

            if analysis_mode == "llm":
                texts_list = _run_llm_analysis(scored, settings)
            else:
                texts_list = []
                for sd, ema, macd, kdj, score, action, _, _ in scored:
                    texts = generate_analysis_template(ema, macd, kdj, score, action)
                    texts_list.append(texts)

            progress.update(task4, completed=True, description=f"[green]✓ {mkt.value}分析文本生成完成")

            # --- Step 5: Assemble results ---
            for i, (sd, ema, macd, kdj, score, action, action_css, price_levels) in enumerate(scored):
                texts = texts_list[i] if i < len(texts_list) else AnalysisTexts()
                result = AnalysisResult(
                    stock=sd,
                    ema=ema,
                    macd=macd,
                    kdj=kdj,
                    score=score,
                    action=action,
                    action_css_class=action_css,
                    texts=texts,
                    price_levels=price_levels,
                )
                all_results_by_market[mkt].append(result)

        # --- Step 6: Generate reports ---
        for mkt, results in all_results_by_market.items():
            if not results:
                continue
            task5 = progress.add_task(f"[cyan]生成{mkt.value}报告...", total=None)
            output_path = generate_report(results, market=mkt)
            progress.update(task5, completed=True, description=f"[green]✓ {mkt.value}报告已生成")
            console.print(f"  [bold]{output_path}[/bold]")

    console.print("\n[bold green]分析完成！[/bold green]")


def _run_llm_analysis(scored, settings) -> list[AnalysisTexts]:
    """Run LLM analysis for all scored stocks, with async concurrency."""
    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def _generate_all():
        tasks = []
        for sd, ema, macd, kdj, score, action, _, _ in scored:
            tasks.append(
                generate_analysis_llm(
                    sd, ema, macd, kdj, score, action,
                    model=settings.llm_model,
                    semaphore=semaphore,
                    provider=settings.llm_provider,
                    base_url=settings.llm_base_url,
                )
            )
        return await asyncio.gather(*tasks)

    return asyncio.run(_generate_all())


@cli.command("list")
@click.option(
    "--market", "-m",
    type=click.Choice(MARKET_CHOICES, case_sensitive=False),
    default="all",
    help="Market to list.",
)
def list_stocks(market: str):
    """List configured stocks."""
    settings = Settings()

    if market == "all":
        market_keys = ["us", "hk", "a", "etf"]
    else:
        market_keys = [market]

    for mkt_key in market_keys:
        stocks = settings.get_stocks(mkt_key)
        if not stocks:
            continue
        console.print(f"\n[bold]{MARKET_MAP[mkt_key].value}[/bold] ({len(stocks)}只)")
        for s in stocks:
            console.print(f"  {s.code:>8s}  {s.name}")


if __name__ == "__main__":
    cli()
