"""Abstract base fetcher and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

from stock_analyzer.models import Market, StockConfig, StockData


class BaseFetcher(ABC):
    """Abstract base class for market data fetchers."""

    @abstractmethod
    def fetch(self, stocks: list[StockConfig]) -> list[StockData]:
        """Fetch OHLCV data and metadata for a list of stocks."""
        ...


class FetcherFactory:
    """Factory to get the appropriate fetcher for a market."""

    @staticmethod
    def get_fetcher(market: Market) -> BaseFetcher:
        if market == Market.US:
            from stock_analyzer.data.us_fetcher import USFetcher
            return USFetcher()
        elif market == Market.HK:
            from stock_analyzer.data.hk_fetcher import HKFetcher
            return HKFetcher()
        elif market == Market.A:
            from stock_analyzer.data.a_fetcher import AFetcher
            return AFetcher()
        elif market == Market.ETF:
            from stock_analyzer.data.etf_fetcher import ETFFetcher
            return ETFFetcher()
        else:
            raise ValueError(f"Unknown market: {market}")
