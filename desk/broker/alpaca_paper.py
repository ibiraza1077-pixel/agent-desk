"""Alpaca PAPER trading broker. Live trading is deliberately unsupported: there is no switch for it."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from alpaca.common.enums import BaseURL
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, MarketOrderRequest, StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

from ..models import Account, Bar, NewsItem, PendingOrder, Position

log = logging.getLogger("desk.alpaca")
ET = ZoneInfo("America/New_York")
PAPER_URL = BaseURL.TRADING_PAPER.value  # https://paper-api.alpaca.markets
LIVE_URL = BaseURL.TRADING_LIVE.value

# Bracket child orders inherit the parent's time-in-force. With DAY the protective stop/target would expire at
# the close and the position would sit unprotected overnight, so GTC is used (Alpaca allows day|gtc for brackets).
BRACKET_TIF = TimeInForce.GTC


class LiveTradingRefused(RuntimeError):
    """Raised when anything suggests the trading client is not pointed at the paper endpoint."""


def assert_paper(client: Any) -> None:
    """Refuse to run unless `client` is demonstrably a paper-trading client."""
    base = getattr(client, "_base_url", None)
    base = str(getattr(base, "value", base) or "")
    if base != PAPER_URL or base == LIVE_URL:
        raise LiveTradingRefused(
            f"Refusing to start: trading client base URL is {base!r}, expected {PAPER_URL!r}. "
            "Agent Desk supports paper trading only."
        )


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class AlpacaPaperBroker:
    intraday_last_bar = True  # during market hours the latest daily bar is today's partial bar

    def __init__(self, api_key: str, secret_key: str, *, trading_client: Any = None, data_client: Any = None,
                 news_client: Any = None):
        # paper=True is hard-coded; there is no configuration path to a live endpoint.
        self.trading = trading_client or TradingClient(api_key, secret_key, paper=True)
        assert_paper(self.trading)
        self.on_alert: Optional[Callable[[str], None]] = None  # Desk hooks this to post loud feed alerts
        self.cancel_timeout_s = 10.0
        self.cancel_poll_s = 0.5
        self._stale: set[str] = set()
        self.data = data_client or StockHistoricalDataClient(api_key, secret_key)
        try:
            self.news = news_client or NewsClient(api_key, secret_key)
        except Exception:  # noqa: BLE001 - news is optional
            self.news = None

    # --- clock ---
    def begin_cycle(self) -> None:
        return None

    def trading_date(self) -> str:
        try:
            return self.trading.get_clock().timestamp.astimezone(ET).date().isoformat()
        except Exception:  # noqa: BLE001
            return datetime.now(ET).date().isoformat()

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def clock_info(self) -> tuple[bool, str]:
        c = self.trading.get_clock()  # one API call for both values
        return bool(c.is_open), c.timestamp.astimezone(ET).date().isoformat()

    def _alert(self, msg: str) -> None:
        log.error(msg)
        if self.on_alert:
            self.on_alert(msg)

    def is_price_stale(self, symbol: str) -> bool:
        return symbol in self._stale

    def get_open_buy_orders(self) -> list[PendingOrder]:
        out = []
        for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)):
            if str(getattr(o.side, "value", o.side)) != "buy":
                continue
            qty = _f(o.qty) - _f(o.filled_qty)
            price = _f(o.limit_price) or _f(o.stop_price)
            if not price:
                try:
                    price = self.get_latest_price(o.symbol)
                except Exception:  # noqa: BLE001
                    price = 0.0
            if qty > 0:
                out.append(PendingOrder(o.symbol, qty, price))
        return out

    # --- account ---
    def get_account(self) -> Account:
        a = self.trading.get_account()
        eq = _f(a.equity)
        return Account(equity=eq, cash=_f(a.cash), last_equity=_f(a.last_equity, eq) or eq)

    def _bracket_levels(self) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Best-effort (stop, target) per symbol from open bracket legs."""
        out: dict[str, list[Optional[float]]] = {}
        try:
            orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500))
            for o in orders:
                for leg in [o, *(o.legs or [])]:
                    lv = out.setdefault(leg.symbol, [None, None])
                    if leg.stop_price:
                        lv[0] = _f(leg.stop_price)
                    elif leg.limit_price and str(getattr(leg.side, "value", leg.side)) == "sell":
                        lv[1] = _f(leg.limit_price)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read bracket levels: %s", e)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def get_positions(self) -> list[Position]:
        levels = self._bracket_levels()
        out = []
        for p in self.trading.get_all_positions():
            stop, tp = levels.get(p.symbol, (None, None))
            out.append(Position(p.symbol, _f(p.qty), _f(p.avg_entry_price), _f(p.current_price),
                                _f(p.unrealized_pl), stop, tp))
        return out

    # --- market data (IEX feed = free tier) ---
    def get_daily_bars(self, symbol: str, n: int = 120) -> list[Bar]:
        start = datetime.now(timezone.utc) - timedelta(days=int(n * 1.7) + 10)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, feed=DataFeed.IEX)
        bars = self.data.get_stock_bars(req).data.get(symbol, [])
        out = [Bar(b.timestamp.astimezone(ET).date().isoformat(), _f(b.open), _f(b.high), _f(b.low), _f(b.close),
                   _f(b.volume)) for b in bars]
        return out[-n:]

    def get_latest_price(self, symbol: str) -> float:
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            px = _f(self.data.get_stock_latest_trade(req)[symbol].price)
            self._stale.discard(symbol)
            return px
        except Exception as e:  # noqa: BLE001
            log.warning("latest trade failed for %s (%s); using last daily close (stale)", symbol, e)
            bars = self.get_daily_bars(symbol, 5)
            if not bars:
                raise
            self._stale.add(symbol)
            return bars[-1].close

    def get_news(self, symbol: str, limit: int = 5) -> list[NewsItem]:
        if self.news is None:
            return []
        try:
            req = NewsRequest(symbols=symbol, limit=limit, start=datetime.now(timezone.utc) - timedelta(days=7))
            items = self.news.get_news(req).data.get("news", [])
            return [NewsItem(n.headline or "", (n.summary or "")[:300], n.created_at.isoformat()) for n in items][:limit]
        except Exception as e:  # noqa: BLE001
            log.warning("news unavailable for %s: %s", symbol, e)
            return []

    # --- orders ---
    def submit_bracket_buy(self, symbol: str, qty: int, stop: float, take_profit: float) -> str:
        req = MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=OrderSide.BUY, time_in_force=BRACKET_TIF,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop, 2)),
        )
        return str(self.trading.submit_order(req).id)

    def _open_orders(self, symbol: str) -> list:
        return list(self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])))

    def _cancel_and_wait(self, symbol: str) -> None:
        """Cancel the symbol's open orders (bracket legs reserve the shares) and wait until none remain.
        Raises if any cancel fails or the orders are still open after the timeout."""
        for o in self._open_orders(symbol):
            self.trading.cancel_order_by_id(o.id)  # a failure propagates: abort the close
        deadline = time.monotonic() + self.cancel_timeout_s
        while self._open_orders(symbol):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"open orders for {symbol} still not cancelled after {self.cancel_timeout_s:.0f}s")
            time.sleep(self.cancel_poll_s)

    def _reprotect(self, levels: dict[str, tuple[Optional[float], Optional[float]]], only: Optional[str] = None) -> None:
        """Put a GTC protective stop back on any position that has none (after a failed close)."""
        try:
            for p in self.trading.get_all_positions():
                if only and p.symbol != only:
                    continue
                if any(getattr(o, "stop_price", None) for o in self._open_orders(p.symbol)):
                    continue
                qty = int(_f(p.qty))
                prev = levels.get(p.symbol, (None, None))[0]
                stop = round(prev if prev else _f(p.current_price) * 0.92, 2)
                if qty < 1:
                    continue
                self.trading.submit_order(StopOrderRequest(symbol=p.symbol, qty=qty, side=OrderSide.SELL,
                                                           time_in_force=TimeInForce.GTC, stop_price=stop))
                self._alert(f"ALERT: close of {p.symbol} failed; re-submitted protective stop for {qty} shares at ${stop:.2f}.")
        except Exception as e:  # noqa: BLE001
            self._alert(f"ALERT: could not restore protective stops after a failed close ({e}). "
                        "CHECK THE ALPACA DASHBOARD NOW: positions may be unprotected.")

    def close_position(self, symbol: str) -> str:
        levels = self._bracket_levels()
        try:
            self._cancel_and_wait(symbol)
            return str(self.trading.close_position(symbol).id)
        except Exception:
            self._reprotect(levels, only=symbol)
            raise

    def close_all(self) -> int:
        levels = self._bracket_levels()
        n = len(self.trading.get_all_positions())
        try:
            self.trading.close_all_positions(cancel_orders=True)
        except Exception:
            self._reprotect(levels)
            raise
        return n
