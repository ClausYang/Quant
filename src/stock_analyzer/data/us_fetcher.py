"""US stock data fetcher using yfinance with curl_cffi session."""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400
MAX_RETRIES = 5
INITIAL_DELAY = 2


def _get_session():
    """Return a curl_cffi session impersonating Chrome to bypass Yahoo rate limits."""
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except ImportError:
        logger.warning("curl_cffi not installed, falling back to default session. "
                       "Install with: pip install curl_cffi")
        return None


def _is_rate_limit_error(e: Exception) -> bool:
    err_str = str(e)
    return any(kw in err_str for kw in (
        "Rate limit", "429", "Too Many Requests", "YFRateLimitError", "YFDownloadError"
    ))


class USFetcher(BaseFetcher):
    """Fetch US stock data via yfinance (requires proxy for access from China)."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []
        session = _get_session()

        for i, cfg in enumerate(stocks):
            symbol = cfg.code.upper()
            logger.info(f"  [{i+1}/{len(stocks)}] Fetching {symbol} ({cfg.name})...")

            df = pd.DataFrame()
            for attempt in range(MAX_RETRIES):
                try:
                    ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
                    df = ticker.history(period=f"{HISTORY_DAYS}d", auto_adjust=False)
                    if not df.empty:
                        break
                except Exception as e:
                    wait = INITIAL_DELAY * (2 ** attempt)
                    if _is_rate_limit_error(e):
                        logger.warning(f"  [{symbol}] rate limited, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                    else:
                        logger.warning(f"  [{symbol}] error: {e}, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)

            if df.empty:
                logger.warning(f"  No data for {symbol}, skipping")
                continue

            try:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.sort_index().dropna(subset=["Close"])

                if len(df) < 30:
                    logger.warning(f"  Insufficient data for {symbol}, skipping")
                    continue

                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                info = {}
                try:
                    ticker = yf.Ticker(symbol, session=session) if session else yf.Ticker(symbol)
                    info = ticker.info or {}
                except Exception:
                    pass

                name = cfg.name or info.get("shortName", symbol)
                sector = cfg.sector or info.get("sector", "") or info.get("industry", "")
                market_cap = info.get("marketCap", 0) or 0

                logger.info(f"    ${price:.2f} ({change_pct:+.2f}%)")
                results.append(StockData(
                    code=symbol,
                    name=name,
                    market=Market.US,
                    sector=sector,
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    market_cap=market_cap,
                    market_cap_str=format_market_cap(market_cap, Market.US),
                    ohlcv=df,
                ))

            except Exception as e:
                logger.error(f"  Failed to process {symbol}: {e}")

        return results
