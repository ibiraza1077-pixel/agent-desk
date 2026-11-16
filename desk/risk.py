"""RiskGuard: pure, deterministic limits. The only path from a Decision to an order.

No I/O, no clock, no randomness. Everything the LLM produced is clamped here.
"""
from __future__ import annotations

import math
from typing import Sequence

from .config import INDEX_SYMBOL, Settings
from .models import Account, Decision, PendingOrder, Position, RiskResult, clamp, opt_price

# Headroom so a fill with slippage can never push a position past the percentage caps.
SIZING_BUFFER = 1.002


def daily_loss_breached(day_pnl: float, equity: float, settings: Settings) -> bool:
    start_equity = equity - day_pnl
    return start_equity > 0 and day_pnl <= -settings.daily_loss_limit_pct * start_equity


def index_window_applies(symbol: str, settings: Settings) -> bool:
    """The VIX rule governs buys of the index symbol in real-market modes (the simulator has no VIX)."""
    return symbol == INDEX_SYMBOL and settings.index_buy_min_vix > 0 and not settings.is_demo


def index_window_open(vix: float | None, settings: Settings) -> bool:
    return vix is not None and settings.index_buy_min_vix > 0 and vix >= settings.index_buy_min_vix


def evaluate(
    decision: Decision,
    account: Account,
    positions: Sequence[Position],
    day_pnl: float,
    last_price: float,
    settings: Settings,
    kill_switch: bool,
    halted_today: bool = False,
    pending: Sequence[PendingOrder] = (),
    price_stale: bool = False,
    vix: float | None = None,
) -> RiskResult:
    if decision.action == "SELL":
        held = any(p.symbol == decision.symbol and p.qty > 0 for p in positions)
        if not held:
            return RiskResult(False, 0, [f"no open position in {decision.symbol} to sell"])
        qty = int(sum(p.qty for p in positions if p.symbol == decision.symbol))
        return RiskResult(True, qty, ["sells are always allowed"])
    if decision.action != "BUY":
        return RiskResult(False, 0, [f"{decision.action} is not an order"])

    reasons: list[str] = []
    s = settings
    equity = account.equity
    start_equity = equity - day_pnl
    halt = False

    if kill_switch:
        reasons.append("kill switch is on: new buys are blocked")
    if halted_today:
        reasons.append("daily loss limit hit earlier today: new buys are halted until tomorrow")
    elif daily_loss_breached(day_pnl, equity, s):
        halt = True
        reasons.append(f"daily loss limit reached ({day_pnl / start_equity:.2%} <= -{s.daily_loss_limit_pct:.2%}): "
                       "new buys halted for the rest of the day")

    if price_stale:
        reasons.append("latest price is stale (live quote unavailable, only an old daily close): buys refused")
    if index_window_applies(decision.symbol, s):
        if vix is None:
            reasons.append(f"{INDEX_SYMBOL} buys need the VIX window, but the latest VIX is unavailable")
        elif vix < s.index_buy_min_vix:
            reasons.append(f"{INDEX_SYMBOL} buys only when VIX >= {s.index_buy_min_vix:g} (latest close {vix:.1f}); "
                           "the 10-year study found the index edge only after fear spikes")

    price = opt_price(last_price)
    if price is None or equity <= 0:
        reasons.append("no valid price or equity available")
        return RiskResult(False, 0, reasons, halt)

    conf = clamp(decision.confidence, 0.0, 1.0)
    if conf < s.min_confidence:
        reasons.append(f"confidence {conf:.2f} below minimum {s.min_confidence:.2f}")

    stop = opt_price(decision.stop_loss)
    target = opt_price(decision.take_profit)
    if stop is None or stop >= price:
        reasons.append("stop-loss must be set below the current price")
    elif (price - stop) / price > s.max_stop_distance_pct:
        reasons.append(f"stop-loss {(price - stop) / price:.1%} away exceeds max {s.max_stop_distance_pct:.1%}")
    if target is None or target <= price:
        reasons.append("take-profit must be set above the current price")
    if stop is not None and stop < price and target is not None and target > price:
        rr = (target - price) / (price - stop)
        if rr < s.min_reward_risk:
            reasons.append(f"reward:risk {rr:.2f} below minimum {s.min_reward_risk:.2f}")

    held = {p.symbol for p in positions if p.qty > 0}
    pend = {o.symbol for o in pending}
    if decision.symbol in held:
        reasons.append(f"already holding {decision.symbol} (no pyramiding)")
    elif decision.symbol in pend:
        reasons.append(f"a buy order for {decision.symbol} is already pending (no pyramiding)")
    if len(held | pend) >= s.max_open_positions:
        reasons.append(f"max open positions ({s.max_open_positions}) reached")

    if reasons:
        return RiskResult(False, 0, reasons, halt)

    size_pct = min(clamp(decision.size_pct, 0.0, 1.0), s.max_position_pct)
    gross = sum(p.market_value for p in positions) + sum(o.notional for o in pending)
    remaining_gross = max(0.0, s.max_gross_exposure_pct * equity - gross)
    dollars = min(size_pct * equity, remaining_gross, max(0.0, account.cash))
    qty = math.floor(dollars / (price * SIZING_BUFFER))
    if qty < 1:
        if remaining_gross < price:
            why = "gross exposure cap leaves no room"
        elif account.cash < price:
            why = "not enough cash"
        else:
            why = "position size rounds to less than 1 share"
        return RiskResult(False, 0, [f"{why} (quantity would be 0)"], halt)
    capped = []
    if size_pct * equity > remaining_gross:
        capped.append("gross exposure cap")
    if size_pct * equity > account.cash:
        capped.append("cash")
    note = f"{qty} shares (~${qty * price:,.0f}, {qty * price / equity:.1%} of equity)"
    if capped:
        note += "; size reduced by " + " and ".join(capped)
    return RiskResult(True, qty, [note], halt)
