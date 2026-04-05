"""US stock data fetcher using yfinance."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400
MAX_RETRIES = 5
INITIAL_DELAY = 2  # 初始重试等待秒数
MAX_WORKERS = 2     # info 并发数，避免同时触发限流
BATCH_DELAY = 3    # 每批下载后等待秒数，降低 burst 风险


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a yfinance rate limit error."""
    err_str = str(e)
    return any(kw in err_str for kw in ("Rate limit", "429", "Too Many Requests", "YFRateLimitError", "YFDownloadError"))


def _download_single(symbol: str) -> pd.DataFrame:
    """Download single symbol history with retry and backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{HISTORY_DAYS}d", auto_adjust=False)
            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    wait = INITIAL_DELAY * (2 ** attempt)
                    logger.warning(f"  [{symbol}] empty data, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                    continue
                return df
            return df
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = INITIAL_DELAY * (2 ** attempt)
                logger.warning(f"  [{symbol}] rate limited, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"  [{symbol}] error: {e}, retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(INITIAL_DELAY * (2 ** attempt))
    logger.error(f"  [{symbol}] all {MAX_RETRIES} attempts failed")
    return pd.DataFrame()


def _fetch_ticker_info(symbol: str) -> dict:
    """Fetch ticker info with retry and rate limit handling."""
    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            return info
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = INITIAL_DELAY * (2 ** attempt)
                logger.warning(f"  [{symbol}] info rate limited, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"  [{symbol}] info error: {e}, retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(INITIAL_DELAY * (2 ** attempt))
    logger.error(f"  [{symbol}] info all attempts failed, using empty info")
    return {}


class USFetcher(BaseFetcher):
    """Fetch US stock data via yfinance."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []
        symbols = [s.code for s in stocks]
        config_map = {s.code: s for s in stocks}

        # 逐个下载 OHLCV（yfinance batch download 并发太高，极易触发限流）
        logger.info(f"Downloading US data for {len(symbols)} symbols (single-threaded with backoff)...")
        all_raw: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"  [{i+1}/{len(symbols)}] Downloading {symbol}...")
            df = _download_single(symbol)
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
            else:
                all_raw[symbol] = df
            # 每下载完一个稍作等待，最后一个不需要等
            if i < len(symbols) - 1:
                time.sleep(0.5)

        # 并发获取 ticker info，限制并发数
        logger.info(f"Fetching ticker info for {len(symbols)} symbols (max {MAX_WORKERS} concurrent)...")
        info_map: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {executor.submit(_fetch_ticker_info, s): s for s in symbols}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    info_map[symbol] = future.result()
                except Exception as e:
                    logger.error(f"  [{symbol}] unexpected error: {e}")
                    info_map[symbol] = {}

        for symbol in symbols:
            try:
                df = all_raw.get(symbol)
                if df is None:
                    continue

                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df.dropna(subset=["Close"])

                if df.empty:
                    logger.warning(f"No valid data for {symbol}, skipping")
                    continue

                info = info_map.get(symbol, {})
                name = config_map[symbol].name or info.get("shortName", symbol)
                sector = (
                    config_map[symbol].sector
                    or info.get("sector", "")
                    or info.get("industry", "")
                )
                market_cap = info.get("marketCap", 0) or 0
                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

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
                logger.info(f"  {symbol}: ${price:.2f} ({change_pct:+.2f}%)")

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue

        return results
