"""ETF data fetcher using akshare."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from stock_analyzer.data.a_fetcher import _bypass_proxy, _restore_proxy
from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400
REQUEST_DELAY = 0.4


class ETFFetcher(BaseFetcher):
    """Fetch CN-listed ETF data via akshare."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")

        # Bypass proxy for domestic Chinese data sources
        saved_proxy = _bypass_proxy()
        try:
            return self._fetch_all(stocks, start_date, end_date)
        finally:
            _restore_proxy(saved_proxy)

    def _fetch_all(self, stocks, start_date, end_date):
        results: list[StockData] = []
        for i, cfg in enumerate(stocks):
            code = cfg.code
            try:
                logger.info(f"  [{i+1}/{len(stocks)}] Fetching ETF {code} ({cfg.name})...")

                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )

                if df is None or df.empty:
                    logger.warning(f"No data for ETF {code}, skipping")
                    continue

                col_map = {
                    "日期": "Date", "开盘": "Open", "最高": "High",
                    "最低": "Low", "收盘": "Close", "成交量": "Volume",
                }
                df = df.rename(columns=col_map)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.sort_index()

                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                market_cap = 0.0
                sector = cfg.sector

                results.append(StockData(
                    code=code,
                    name=cfg.name,
                    market=Market.ETF,
                    sector=sector,
                    price=round(price, 4),
                    change_pct=round(change_pct, 2),
                    market_cap=market_cap,
                    market_cap_str="N/A",
                    ohlcv=df,
                ))
                logger.info(f"    ¥{price:.4f} ({change_pct:+.2f}%)")

            except Exception as e:
                logger.error(f"Failed to process ETF {code}: {e}")
                continue

            if i < len(stocks) - 1:
                time.sleep(REQUEST_DELAY)

        return results
