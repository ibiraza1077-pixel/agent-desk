"""Broker interface. Long-only, whole shares, stocks/ETFs."""
from __future__ import annotations

from typing import Protocol

from ..models import Account, Bar, NewsItem, PendingOrder, Position


class Broker(Protocol):
    def begin_cycle(self) -> None:
        """Hook called at the start of every cycle (the simulator advances one day here)."""

    def trading_date(self) -> str:
        """Current trading date, YYYY-MM-DD (used for the daily-loss halt)."""

    def is_market_open(self) -> bool: ...

    def clock_info(self) -> tuple[bool, str]:
        """(market open, trading date) from a single clock lookup."""

    def get_open_buy_orders(self) -> list[PendingOrder]:
        """Unfilled BUY orders (count toward exposure/positions in risk checks)."""

    def is_price_stale(self, symbol: str) -> bool:
        """True if the last get_latest_price(symbol) fell back to an old daily close."""

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def get_daily_bars(self, symbol: str, n: int = 120) -> list[Bar]: ...

    def get_latest_price(self, symbol: str) -> float: ...

    def get_news(self, symbol: str, limit: int = 5) -> list[NewsItem]: ...

    def submit_bracket_buy(self, symbol: str, qty: int, stop: float, take_profit: float) -> str:
        """Place a market buy with attached stop-loss and take-profit. Returns an order id."""

    def close_position(self, symbol: str) -> str: ...

    def close_all(self) -> int:
        """Close every position; returns how many were closed."""
