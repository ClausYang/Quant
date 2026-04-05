"""A-share stock data fetcher using akshare."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 2


def _is_connection_error(e: Exception) -> bool:
    """Check if exception is a connection/reset error that warrants retry."""
    err_str = str(e)
    return any(kw in err_str for kw in (
        "RemoteDisconnected",
        "ConnectionReset",
        "Connection aborted",
        "Connection refused",
        "Timeout",
        "HTTPSConnectionPool",
        "Max retries exceeded",
    ))


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _patch_requests_headers() -> None:
    """Patch requests.Session.request to always inject browser headers.

    akshare creates sessions internally, so we must patch at the requests level
    to intercept all HTTP calls.
    """
    import requests

    _original_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("User-Agent", _HEADERS["User-Agent"])
        headers.setdefault("Referer", _HEADERS["Referer"])
        headers.setdefault("Accept", _HEADERS["Accept"])
        headers.setdefault("Accept-Language", _HEADERS["Accept-Language"])
        kwargs["headers"] = headers
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request


def _bypass_proxy() -> dict:
    """Bypass all proxies for domestic Chinese API access.

    On macOS, requests/urllib reads system proxy from Network Preferences
    even when env vars are cleared. We must monkey-patch urllib.request.getproxies
    to return an empty dict.
    """
    import urllib.request

    saved: dict = {"env": {}, "getproxies": urllib.request.getproxies}

    # 1. Clear all proxy env vars
    proxy_keys = [
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY",
    ]
    for key in proxy_keys:
        val = os.environ.get(key)
        if val is not None:
            saved["env"][key] = val
        os.environ.pop(key, None)

    # 2. Force no_proxy for all hosts
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"

    # 3. Monkey-patch urllib.request.getproxies to return empty
    #    This blocks requests from reading macOS System Preferences proxy
    urllib.request.getproxies = lambda: {}

    return saved


def _restore_proxy(saved: dict) -> None:
    """Restore previously saved proxy settings."""
    import urllib.request

    # Restore env vars
    # First remove no_proxy we added
    os.environ.pop("no_proxy", None)
    os.environ.pop("NO_PROXY", None)

    for key, val in saved["env"].items():
        os.environ[key] = val

    # Restore original getproxies function
    urllib.request.getproxies = saved["getproxies"]


class AFetcher(BaseFetcher):
    """Fetch A-share stock data via akshare."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")

        # Bypass proxy for domestic Chinese data sources (eastmoney.com etc.)
        saved_proxy = _bypass_proxy()
        logger.debug("Proxy bypassed for A-share data access")

        # Patch requests to inject browser headers on all calls (akshare creates sessions internally)
        _patch_requests_headers()

        try:
            results = self._fetch_all(stocks, start_date, end_date)
        finally:
            _restore_proxy(saved_proxy)
            logger.debug("Proxy settings restored")

        return results

    def _fetch_all(self, stocks, start_date, end_date):
        results: list[StockData] = []
        for i, cfg in enumerate(stocks):
            code = cfg.code.split(".")[0]
            df = None
            info_dict = {}

            # Retry loop for OHLCV fetch
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"  [{i+1}/{len(stocks)}] Fetching {code} ({cfg.name})...")
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )
                    break
                except Exception as e:
                    if _is_connection_error(e) and attempt < MAX_RETRIES - 1:
                        wait = RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"  [{code}] connection error, retry {attempt+1}/{MAX_RETRIES} in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        logger.error(f"  Failed to process {code}: {e}")
                        df = None
                        break

            if df is None or df.empty:
                logger.warning(f"No data for {code}, skipping")
                time.sleep(REQUEST_DELAY)
                continue

            # Normalize columns
            col_map = {
                "日期": "Date", "开盘": "Open", "最高": "High",
                "最低": "Low", "收盘": "Close", "成交量": "Volume",
            }
            df = df.rename(columns=col_map)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.sort_index()

            # Current price and change
            price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

            # Fetch metadata (with retry)
            sector = cfg.sector
            market_cap = 0.0
            for attempt in range(MAX_RETRIES):
                try:
                    info_df = ak.stock_individual_info_em(symbol=code)
                    if info_df is not None and not info_df.empty:
                        info_dict = dict(zip(info_df["item"], info_df["value"]))
                        market_cap = float(info_dict.get("总市值", 0) or 0)
                        if not sector:
                            sector = info_dict.get("行业", "")
                    break
                except Exception as e:
                    if _is_connection_error(e) and attempt < MAX_RETRIES - 1:
                        wait = RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"  [{code}] metadata connection error, retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.debug(f"Could not fetch metadata for {code}: {e}")
                        break

            results.append(StockData(
                code=code,
                name=cfg.name,
                market=Market.A,
                sector=sector,
                price=round(price, 2),
                change_pct=round(change_pct, 2),
                market_cap=market_cap,
                market_cap_str=format_market_cap(market_cap, Market.A),
                ohlcv=df,
            ))
            logger.info(f"    ¥{price:.2f} ({change_pct:+.2f}%)")

            # Rate limiting between stocks (not after the last one)
            if i < len(stocks) - 1:
                time.sleep(REQUEST_DELAY)

        return results
