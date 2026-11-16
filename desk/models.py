"""Shared data models."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

Stance = Literal["bullish", "bearish", "neutral"]


def clamp(x, lo: float, hi: float, default: float = 0.0) -> float:
    """Coerce an untrusted number into [lo, hi]; junk (None/NaN/str) -> default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return max(lo, min(hi, v))


def opt_price(x) -> Optional[float]:
    """Untrusted price -> positive float or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class NewsItem:
    headline: str
    summary: str
    time: str


@dataclass
class Account:
    equity: float
    cash: float
    last_equity: float  # equity at the start of the trading day


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    current_price: float
    unrealized_pl: float
    stop: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price


@dataclass
class PendingOrder:
    """An open (unfilled) BUY order: it will become exposure and a position if it fills."""
    symbol: str
    qty: float
    price: float

    @property
    def notional(self) -> float:
        return self.qty * self.price


class AgentNote(BaseModel):
    agent: str
    stance: Stance = "neutral"
    confidence: float = 0.0
    text: str = ""
    # Charts: suggested levels. Skeptic: key risk.
    stop: Optional[float] = None
    target: Optional[float] = None
    key_risk: str = ""
    degraded: bool = False  # True when the LLM call failed/refused and this is a neutral stand-in


class Decision(BaseModel):
    symbol: str
    action: Literal["BUY", "WAIT", "SKIP", "HOLD", "SELL"] = "SKIP"
    size_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = Field(default=0.0)
    rationale: str = ""


@dataclass
class RiskResult:
    approved: bool
    qty: int = 0
    reasons: list[str] = field(default_factory=list)
    halt: bool = False  # daily loss limit newly breached: caller should persist the halt

    @property
    def status(self) -> str:
        return "approved" if self.approved else "rejected"
