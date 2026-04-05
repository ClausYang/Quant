"""Hong Kong stock data fetcher using yfinance."""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400


def hk_code_to_yf(code: str) -> str:
    """Convert 5-digit HK code to yfinance format: '00883' -> '0883.HK'."""
    numeric = code.lstrip("0") or "0"
    # yfinance HK tickers use 4-digit codes
    return f"{int(numeric):04d}.HK"


class HKFetcher(BaseFetcher):
    """Fetch HK stock data via yfinance."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []
        config_map = {s.code: s for s in stocks}

        yf_symbols = {s.code: hk_code_to_yf(s.code) for s in stocks}
        symbol_list = list(yf_symbols.values())

        logger.info(f"Downloading HK data for {len(symbol_list)} symbols...")
        raw = yf.download(
            symbol_list,
            period=f"{HISTORY_DAYS}d",
            group_by="ticker",
            threads=True,
            progress=False,
        )

        for orig_code, yf_code in yf_symbols.items():
            try:
                if len(symbol_list) == 1:
                    df = raw.copy()
                else:
                    df = raw[yf_code].copy()

                df = df.dropna(subset=["Close"])
                if df.empty:
                    logger.warning(f"No data for {orig_code} ({yf_code}), skipping")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                df = df.rename(columns={
                    "Open": "Open", "High": "High", "Low": "Low",
                    "Close": "Close", "Volume": "Volume",
                })
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                cfg = config_map[orig_code]
                ticker = yf.Ticker(yf_code)
                info = ticker.info or {}
                name = cfg.name or info.get("shortName", orig_code)
                sector = cfg.sector or info.get("sector", "") or info.get("industry", "")
                market_cap = info.get("marketCap", 0) or 0
                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                results.append(StockData(
                    code=orig_code,
                    name=name,
                    market=Market.HK,
                    sector=sector,
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    market_cap=market_cap,
                    market_cap_str=format_market_cap(market_cap, Market.HK),
                    ohlcv=df,
                ))
                logger.info(f"  {orig_code} ({name}): HK${price:.2f} ({change_pct:+.2f}%)")

            except Exception as e:
                logger.error(f"Failed to process {orig_code}: {e}")
                continue

        return results
