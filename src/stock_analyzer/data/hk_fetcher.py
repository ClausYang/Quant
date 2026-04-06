"""Hong Kong stock data fetcher using Tushare Pro."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from stock_analyzer.data.fetcher import BaseFetcher
from stock_analyzer.models import Market, StockConfig, StockData
from stock_analyzer.utils.formatting import format_market_cap

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400


def _to_ts_code(code: str) -> str:
    """Convert HK code to Tushare format: '00700' -> '00700.HK'."""
    bare = code.split(".")[0].zfill(5)
    return f"{bare}.HK"


def _get_pro():
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set")
    ts.set_token(token)
    return ts.pro_api()


class HKFetcher(BaseFetcher):
    """Fetch HK stock data via Tushare Pro hk_daily."""

    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        results: list[StockData] = []

        try:
            pro = _get_pro()
        except RuntimeError as e:
            logger.error(str(e))
            return results

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")

        for i, cfg in enumerate(stocks):
            ts_code = _to_ts_code(cfg.code)
            logger.info(f"  [{i+1}/{len(stocks)}] Fetching {cfg.code} ({cfg.name})...")

            try:
                df = pro.hk_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None or df.empty:
                    logger.warning(f"  No data for {cfg.code}, skipping")
                    continue

                df = df.rename(columns={
                    "trade_date": "Date",
                    "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "vol": "Volume",
                })
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
                df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.sort_index()

                if len(df) < 30:
                    logger.warning(f"  Insufficient data for {cfg.code} ({len(df)} bars), skipping")
                    continue

                price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                logger.info(f"    HK${price:.2f} ({change_pct:+.2f}%)")
                results.append(StockData(
                    code=cfg.code,
                    name=cfg.name,
                    market=Market.HK,
                    sector=cfg.sector or "",
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    market_cap=0.0,
                    market_cap_str="",
                    ohlcv=df,
                ))

            except Exception as e:
                logger.error(f"  Failed to process {cfg.code}: {e}")

        return results
