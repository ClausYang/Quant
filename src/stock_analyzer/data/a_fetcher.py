"""A-share stock data fetcher using baostock."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400
REQUEST_DELAY = 0.1  # baostock is local TCP, no need for long delays


def _to_bs_code(code: str) -> str:
    """Convert bare stock code to baostock format (sh.600519 / sz.000001)."""
    c = code.split(".")[0]
    if c.startswith(("6", "5")):
        return f"sh.{c}"
    return f"sz.{c}"


class AFetcher(BaseFetcher):
    """Fetch A-share stock data via baostock."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock login failed: {lg.error_msg}")
            return []
        try:
            return self._fetch_all(stocks)
        finally:
            bs.logout()

    def _fetch_all(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

        for i, cfg in enumerate(stocks):
            bs_code = _to_bs_code(cfg.code)
            logger.info(f"  [{i+1}/{len(stocks)}] Fetching {cfg.code} ({cfg.name})...")

            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",  # 前复权
                )
                if rs.error_code != "0":
                    logger.warning(f"  [{cfg.code}] query failed: {rs.error_msg}")
                    continue

                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if not rows:
                    logger.warning(f"  No data for {cfg.code}, skipping")
                    continue

                df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna().sort_index()

                if len(df) < 30:
                    logger.warning(f"  Insufficient data for {cfg.code} ({len(df)} bars), skipping")
                    continue

                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                logger.info(f"    ¥{price:.2f} ({change_pct:+.2f}%)")

                results.append(StockData(
                    code=cfg.code.split(".")[0],
                    name=cfg.name,
                    market=Market.A,
                    sector=cfg.sector or "",
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    market_cap=0.0,
                    market_cap_str="",
                    ohlcv=df,
                ))

            except Exception as e:
                logger.error(f"  Failed to process {cfg.code}: {e}")

            if i < len(stocks) - 1:
                time.sleep(REQUEST_DELAY)

        return results
