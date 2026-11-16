"""SimBroker: an in-memory synthetic market for demo mode. No network."""
from __future__ import annotations

import math
import random
import threading
import zlib
from datetime import date, timedelta
from typing import Optional

from ..models import Account, Bar, NewsItem, PendingOrder, Position

SLIPPAGE = 0.0005
START_CASH = 100_000.0
HISTORY_DAYS = 120

# symbol -> (start price, annual drift, annual vol, base volume)
PROFILES: dict[str, tuple[float, float, float, float]] = {
    "SPY": (520, 0.10, 0.15, 60e6), "QQQ": (450, 0.14, 0.20, 40e6), "AAPL": (190, 0.12, 0.25, 55e6),
    "MSFT": (420, 0.14, 0.24, 22e6), "NVDA": (880, 0.35, 0.50, 45e6), "AMZN": (180, 0.15, 0.30, 40e6),
    "GOOGL": (160, 0.12, 0.28, 25e6), "META": (480, 0.18, 0.35, 15e6), "TSLA": (175, 0.05, 0.55, 90e6),
    "AMD": (160, 0.20, 0.48, 55e6), "JPM": (195, 0.08, 0.22, 9e6), "XOM": (115, 0.04, 0.24, 14e6),
}

POSITIVE = [
    "{s} beats earnings estimates and raises full-year guidance",
    "Analysts upgrade {s} citing accelerating demand",
    "{s} announces major partnership, shares in focus",
    "Record quarter for {s} as margins expand",
]
NEGATIVE = [
    "{s} cut to underperform as growth concerns mount",
    "{s} faces regulatory probe, investors cautious",
    "{s} misses revenue estimates, guides lower",
    "Supply chain disruption weighs on {s} outlook",
]
NEUTRAL = [
    "{s} to present at upcoming industry conference",
    "{s} announces routine dividend and buyback update",
    "What to watch in {s} ahead of next week's macro data",
    "{s} executives reiterate long-term strategy",
]


def _symbol_seed(seed: int, symbol: str) -> int:
    return seed * 1_000_003 + zlib.crc32(symbol.encode())


def _next_bday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


class _Sym:
    def __init__(self, symbol: str, seed: int):
        base = PROFILES.get(symbol)
        rng = random.Random(_symbol_seed(seed, symbol))
        if base is None:  # unknown ticker: plausible synthetic profile
            base = (rng.uniform(30, 400), rng.uniform(0.0, 0.2), rng.uniform(0.2, 0.45), rng.uniform(5e6, 40e6))
        self.price0, self.mu, self.sigma, self.base_vol = base
        self.rng = rng
        self.bars: list[Bar] = []
        self.price = self.price0
        self.regime = 0.0  # slowly varying drift shock, gives trends

    def step(self, day: date) -> Bar:
        r = self.rng
        self.regime = 0.95 * self.regime + r.gauss(0, 0.0006)
        ret = self.mu / 252 - 0.5 * (self.sigma ** 2) / 252 + self.regime + self.sigma / math.sqrt(252) * r.gauss(0, 1)
        o = self.price
        c = max(0.5, o * math.exp(ret))
        wick = abs(r.gauss(0, self.sigma / math.sqrt(252) * 0.5))
        h = max(o, c) * (1 + wick)
        lo = min(o, c) * (1 - abs(r.gauss(0, self.sigma / math.sqrt(252) * 0.5)))
        vol = self.base_vol * math.exp(r.gauss(0, 0.25)) * (1 + 12 * abs(ret))
        bar = Bar(day.isoformat(), round(o, 2), round(h, 2), round(lo, 2), round(c, 2), round(vol))
        self.price = c
        self.bars.append(bar)
        return bar


class _Held:
    def __init__(self, qty: int, avg: float, stop: float, tp: float):
        self.qty, self.avg, self.stop, self.tp = qty, avg, stop, tp


class SimBroker:
    def __init__(self, symbols: list[str], seed: int = 42, start_cash: float = START_CASH):
        self.seed = seed
        self.symbols = list(dict.fromkeys([*symbols, "SPY"]))
        self._lock = threading.RLock()  # guards cash/held (API threads read while the cycle thread writes)
        self.cash = start_cash
        self.syms = {s: _Sym(s, seed) for s in self.symbols}
        self.held: dict[str, _Held] = {}
        self.realized_pl = 0.0
        self._order_n = 0
        self._news_rng = random.Random(seed + 7)
        self.today = date(2026, 1, 2)
        start = self.today
        for _ in range(HISTORY_DAYS):
            start = _next_bday(start)
            for s in self.syms.values():
                s.step(start)
        self.today = start
        self.last_equity = self._equity()

    # --- internals ---
    def _px(self, symbol: str) -> float:
        return self.syms[symbol].price

    def _equity(self) -> float:
        return self.cash + sum(h.qty * self._px(s) for s, h in self.held.items())

    def _oid(self) -> str:
        self._order_n += 1
        return f"sim-{self._order_n:05d}"

    def _sell(self, symbol: str, price: float) -> None:
        h = self.held.pop(symbol)
        self.cash += h.qty * price
        self.realized_pl += h.qty * (price - h.avg)

    # --- Broker API ---
    def begin_cycle(self) -> None:
        """Advance the market one day. Exits are checked on the new bar: a gap through the stop or target fills at
        the open; otherwise the stop wins if both are touched intrabar."""
        with self._lock:
            self.last_equity = self._equity()
            self.today = _next_bday(self.today)
            for s in self.syms.values():
                s.step(self.today)
            for symbol in list(self.held):
                h, bar = self.held[symbol], self.syms[symbol].bars[-1]
                if bar.open <= h.stop:
                    self._sell(symbol, bar.open * (1 - SLIPPAGE))
                elif bar.open >= h.tp:
                    self._sell(symbol, bar.open)
                elif bar.low <= h.stop:
                    self._sell(symbol, h.stop * (1 - SLIPPAGE))
                elif bar.high >= h.tp:
                    self._sell(symbol, h.tp)

    def trading_date(self) -> str:
        return self.today.isoformat()

    def is_market_open(self) -> bool:
        return True

    def clock_info(self) -> tuple[bool, str]:
        return True, self.trading_date()

    def get_open_buy_orders(self) -> list[PendingOrder]:
        return []

    def is_price_stale(self, symbol: str) -> bool:
        return False

    def get_account(self) -> Account:
        with self._lock:
            eq = self._equity()
            return Account(equity=eq, cash=self.cash, last_equity=self.last_equity)

    def get_positions(self) -> list[Position]:
        with self._lock:
            out = []
            for s, h in self.held.items():
                px = self._px(s)
                out.append(Position(s, h.qty, h.avg, px, (px - h.avg) * h.qty, h.stop, h.tp))
            return out

    def get_daily_bars(self, symbol: str, n: int = 120) -> list[Bar]:
        return list(self.syms[symbol].bars[-n:])

    def get_latest_price(self, symbol: str) -> float:
        return round(self._px(symbol), 2)

    def get_news(self, symbol: str, limit: int = 5) -> list[NewsItem]:
        sym = self.syms[symbol]
        rng = random.Random(_symbol_seed(self.seed, symbol) + len(sym.bars))
        closes = [b.close for b in sym.bars]
        recent = closes[-1] / closes[-6] - 1 if len(closes) > 6 else 0.0
        bias = max(-0.4, min(0.4, recent * 6))  # recent strength nudges headline sentiment
        items = []
        for i in range(min(limit, 3)):
            u = rng.random()
            pool = POSITIVE if u < 0.33 + bias else NEGATIVE if u > 0.66 + bias else NEUTRAL
            head = rng.choice(pool).format(s=symbol)
            when = (self.today - timedelta(days=i)).isoformat()
            items.append(NewsItem(head, f"Simulated headline for {symbol} (demo data).", when))
        return items

    def submit_bracket_buy(self, symbol: str, qty: int, stop: float, take_profit: float) -> str:
        with self._lock:
            if symbol in self.held:
                raise ValueError(f"already holding {symbol}")
            price = self._px(symbol) * (1 + SLIPPAGE)
            qty = int(min(qty, self.cash // price))
            if qty < 1:
                raise ValueError("insufficient cash")
            self.cash -= qty * price
            self.held[symbol] = _Held(qty, price, round(stop, 2), round(take_profit, 2))
            return self._oid()

    def close_position(self, symbol: str) -> str:
        with self._lock:
            if symbol not in self.held:
                raise ValueError(f"no position in {symbol}")
            self._sell(symbol, self._px(symbol) * (1 - SLIPPAGE))
            return self._oid()

    def close_all(self) -> int:
        with self._lock:
            n = 0
            for s in list(self.held):
                self.close_position(s)
                n += 1
            return n
