"""Configuration loading and management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from stock_analyzer.models import Market, StockConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings:
    """Application settings loaded from config files."""

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        self.settings = load_yaml(CONFIG_DIR / "settings.yaml")
        self.stocks_config = load_yaml(CONFIG_DIR / "stocks.yaml")

        OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Indicator parameters ---

    @property
    def ema_periods(self) -> list[int]:
        return self.settings.get("indicators", {}).get(
            "ema_periods", [5, 10, 20, 30, 55, 120, 144, 169, 200]
        )

    @property
    def macd_params(self) -> dict[str, int]:
        defaults = {"fast": 12, "slow": 26, "signal": 9}
        return self.settings.get("indicators", {}).get("macd", defaults)

    @property
    def kdj_params(self) -> dict[str, int]:
        defaults = {"n": 9, "m1": 3, "m2": 3}
        return self.settings.get("indicators", {}).get("kdj", defaults)

    # --- Scoring weights ---

    @property
    def scoring_weights(self) -> dict[str, float]:
        defaults = {"trend": 0.4, "macd": 0.3, "kdj": 0.2, "context": 0.1}
        return self.settings.get("scoring", {}).get("weights", defaults)

    # --- Analysis mode ---

    @property
    def analysis_mode(self) -> str:
        return self.settings.get("analysis", {}).get("mode", "llm")

    @property
    def llm_provider(self) -> str:
        return self.settings.get("analysis", {}).get("provider", "anthropic")

    @property
    def llm_model(self) -> str:
        return self.settings.get("analysis", {}).get(
            "model", "claude-sonnet-4-20250514"
        )

    @property
    def llm_base_url(self) -> str:
        return self.settings.get("analysis", {}).get("base_url", "")

    @property
    def llm_concurrency(self) -> int:
        return self.settings.get("analysis", {}).get("concurrency", 10)

    # --- Stock lists ---

    def get_stocks(self, market: str | None = None) -> list[StockConfig]:
        """Get stock configs, optionally filtered by market."""
        stocks: list[StockConfig] = []
        market_map = {"us": Market.US, "hk": Market.HK, "a": Market.A, "etf": Market.ETF}

        for section in ["portfolio", "watchlist"]:
            section_data = self.stocks_config.get(section, {})
            for mkt_key, items in section_data.items():
                mkt = market_map.get(mkt_key)
                if mkt is None:
                    continue
                if market and mkt_key != market:
                    continue
                for item in (items or []):
                    stocks.append(StockConfig(
                        code=str(item["code"]),
                        name=item["name"],
                        market=mkt,
                        sector=item.get("sector", ""),
                    ))
        return stocks

    def get_sector_overrides(self) -> dict[str, str]:
        """Get manual sector overrides for stocks with missing data."""
        return self.stocks_config.get("sector_overrides", {})
