"""Orchestrator. One cycle: screen -> debate -> decide -> risk -> execute -> snapshot."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from . import risk
from .agents.base import AgentTeam, AgentUnavailable
from .broker.base import Broker
from .config import INDEX_SYMBOL, Settings
from .indicators import compute_indicators, session_volume_fraction
from .models import Decision, Position
from .store import Store

ET = ZoneInfo("America/New_York")
log = logging.getLogger("desk")

SYSTEM = "System"


def screen_score(ind: dict, price: float) -> float:
    """Trend + momentum + volume score used to rank candidates (code, no LLM)."""
    sma20, sma50 = ind.get("sma20"), ind.get("sma50")
    score = 0.0
    if sma20 and price > sma20:
        score += 1.0
    if sma20 and sma50 and sma20 > sma50:
        score += 1.0
    score += max(-1.5, min(1.5, (ind.get("ret_20d") or 0.0) * 10))
    rsi = ind.get("rsi14")
    if rsi is not None:
        score += 0.5 if 45 <= rsi <= 70 else (-1.0 if rsi > 78 else 0.0)
    if (ind.get("volume_ratio20") or 0) > 1.2:
        score += 0.5
    if (ind.get("dist_from_20d_high") or -1) > -0.03:
        score += 0.5
    return round(score, 2)


def scout_pitch(symbol: str, ind: dict, price: float, score: float) -> str:
    bits = []
    sma20, sma50 = ind.get("sma20"), ind.get("sma50")
    if sma20 and sma50 and price > sma20 > sma50:
        bits.append("trading above a rising 20/50-day average")
    elif sma20 and price > sma20:
        bits.append("back above its 20-day average")
    if ind.get("ret_20d") is not None:
        r20 = ind["ret_20d"]
        bits.append(f"{'up' if r20 >= 0 else 'down'} {abs(r20):.1%} over 20 days")
    if (ind.get("volume_ratio20") or 0) > 1.2:
        bits.append(f"volume {ind['volume_ratio20']:.1f}x normal")
    if (ind.get("dist_from_20d_high") or -1) > -0.03:
        bits.append("pressing its 20-day high")
    return f"{symbol} surfaced with screen score {score:+.1f}: " + (", ".join(bits) or "ranked best of what is left") + "."


class Desk:
    def __init__(self, settings: Settings, store: Store, broker: Broker, team: AgentTeam, history=None):
        self.s, self.store, self.broker, self.team = settings, store, broker, team
        self.history = history  # desk.history.History (real-market modes only): base rates + VIX regime
        self.lock = threading.Lock()
        self.running = False  # scheduler loop alive
        self.next_cycle_at: Optional[str] = None
        self.cycle_active = False
        self._order_lock = threading.Lock()  # serialises "check kill switch + place order" against flatten
        self._cache: dict[str, tuple[float, object]] = {}
        if hasattr(broker, "on_alert"):
            broker.on_alert = lambda msg: self.feed(None, None, SYSTEM, "rejected", msg)
        self._closed_notice = False
        self._last_closed_snapshot = 0.0

    # --- helpers ---
    CACHE_TTL_S = 5.0

    def _read(self, key: str, fn):
        """Short-TTL cache for dashboard/API read paths only. Trading and risk checks call the broker directly."""
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.CACHE_TTL_S:
            return hit[1]
        val = fn()
        self._cache[key] = (time.monotonic(), val)
        return val

    def feed(self, cycle_id, symbol, agent, stance, text) -> None:
        self.store.add_feed(cycle_id, symbol, agent, stance, text)

    def system(self, text: str, cycle_id: Optional[int] = None, symbol: Optional[str] = None) -> None:
        self.feed(cycle_id, symbol, SYSTEM, "neutral", text)

    def halted_today(self) -> bool:
        return self.store.halted_date == self.broker.trading_date()

    def llm_budget_exhausted(self) -> bool:
        # Free providers are never budget-gated (a $0 budget or earlier paid spend must not starve them).
        return (self.team.uses_llm and not self.s.llm_is_free
                and self.store.spend_today() >= self.s.daily_llm_budget_usd)

    def snapshot_equity(self) -> None:
        acct = self.broker.get_account()
        try:
            spy = self.broker.get_latest_price("SPY")
        except Exception:  # noqa: BLE001 - benchmark price is optional
            spy = None
        self.store.add_equity(acct.equity, acct.cash, spy)

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def last_bar_fraction(self, bars) -> Optional[float]:
        """How much of a normal day's volume today's bar should hold so far (1.0 if the last bar is a full day).

        Only brokers whose daily bars include today's still-forming bar (Alpaca) set `intraday_last_bar`."""
        if not bars or not getattr(self.broker, "intraday_last_bar", False):
            return 1.0
        now = self._now_et()
        if bars[-1].date != now.date().isoformat():
            return 1.0
        opened = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return session_volume_fraction((now - opened).total_seconds() / 60.0)

    def current_vix(self) -> Optional[float]:
        """Latest VIX close (FRED, cached), or None in demo mode / when unavailable."""
        if self.history is None:
            return None
        try:
            return self.history.vix()[0]
        except Exception:  # noqa: BLE001
            return None

    def fact_sheet(self, symbol: str, account, positions: list[Position]) -> dict:
        bars = self.broker.get_daily_bars(symbol, 120)
        price = self.broker.get_latest_price(symbol)
        try:
            news = [{"headline": n.headline, "summary": n.summary[:240], "time": n.time}
                    for n in self.broker.get_news(symbol, 5)]
        except Exception:  # noqa: BLE001 - news is best-effort
            news = []
        gross = sum(p.market_value for p in positions)
        ind = compute_indicators(bars, self.last_bar_fraction(bars))
        sheet = {
            "symbol": symbol,
            "as_of": bars[-1].date if bars else "",
            "price": round(price, 2),
            "indicators": ind,
            "recent_closes": [round(b.close, 2) for b in bars[-10:]],
            "news": news,
            "portfolio": {
                "equity": round(account.equity, 2), "cash": round(account.cash, 2),
                "open_positions": len(positions), "held_symbols": [p.symbol for p in positions],
                "gross_exposure_pct": round(gross / account.equity, 3) if account.equity else 0.0,
            },
        }
        if self.history is not None:
            try:
                h = self.history.context(symbol, screen_score(ind, price) if len(bars) >= 55 else None,
                                         index_buy_min_vix=self.s.index_buy_min_vix)
                if h:
                    sheet["history"] = h
            except Exception as e:  # noqa: BLE001 - context only
                log.warning("history context failed for %s: %s", symbol, e)
        return sheet

    def screen(self, account, positions: list[Position], pending=()) -> list[tuple[float, str, dict, float]]:
        held = {p.symbol for p in positions} | {o.symbol for o in pending}
        scored = []
        for sym in self.s.watchlist:
            if sym in held:
                continue
            try:
                bars = self.broker.get_daily_bars(sym, 120)
                if len(bars) < 55:
                    continue
                price = self.broker.get_latest_price(sym)
                ind = compute_indicators(bars, self.last_bar_fraction(bars))
                scored.append((screen_score(ind, price), sym, ind, price))
            except Exception as e:  # noqa: BLE001
                log.warning("screen failed for %s: %s", sym, e)
        # The 10-year study found the screen score has no predictive value (IC ~0), so it is only used to order
        # candidates, and names debated in the last two cycles go to the back of the queue: every watchlist name
        # gets debated in turn instead of the same three each cycle.
        recent = {d["symbol"] for d in self.store.get_decisions(2 * max(1, self.s.max_candidates_per_cycle))}
        scored.sort(key=lambda t: (t[1] in recent, -t[0], t[1]))
        # VIX window: when fear is elevated, the index goes to the front of the queue (the one robust finding).
        if risk.index_window_open(self.current_vix(), self.s) and not self.s.is_demo:
            idx = next((t for t in scored if t[1] == INDEX_SYMBOL), None)
            if idx:
                scored.remove(idx)
                scored.insert(0, idx)
        return scored[: self.s.max_candidates_per_cycle]

    # --- cycle ---
    def run_cycle(self) -> Optional[int]:
        """Run one cycle. Returns the cycle id, or None if one is already running."""
        if not self.lock.acquire(blocking=False):
            return None
        cycle_id = None
        try:
            self.cycle_active = True
            cycle_id = self.store.start_cycle(self.s.mode)
            status = "done"
            try:
                self._cycle(cycle_id)
            except Exception as e:  # noqa: BLE001 - the loop must survive anything
                log.exception("cycle failed")
                self.system(f"Cycle failed: {type(e).__name__}: {e}", cycle_id)
                status = "error"
            self.store.finish_cycle(cycle_id, status)
            return cycle_id
        finally:
            self._cache.clear()
            self.cycle_active = False
            self.lock.release()

    def _cycle(self, cid: int) -> None:
        self.broker.begin_cycle()
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        today = self.broker.trading_date()
        day_pnl = account.equity - account.last_equity
        if risk.daily_loss_breached(day_pnl, account.equity, self.s) and not self.halted_today():
            self.store.set_halted(today)
            self.feed(cid, None, "Risk", "rejected",
                      f"Daily loss limit hit ({day_pnl:+,.0f}). New buys are halted for the rest of {today}.")

        budget_out = self.llm_budget_exhausted()
        if budget_out:
            self.system(f"LLM budget reached (${self.store.spend_today():.2f} of ${self.s.daily_llm_budget_usd:.2f} "
                        "today). Skipping debates until tomorrow; broker-side stops still protect open positions.",
                        cid)

        # 1) review held positions (Chief alone)
        if not budget_out:
            for pos in list(positions):
                try:
                    self._review_position(cid, pos, account)
                except AgentUnavailable as e:
                    self.system(str(e), cid, pos.symbol)
                except Exception as e:  # noqa: BLE001
                    log.exception("review failed")
                    self.system(f"Review of {pos.symbol} failed: {e}", cid, pos.symbol)
            account = self.broker.get_account()
            positions = self.broker.get_positions()

        # 2) screen and debate new candidates
        pending = self.broker.get_open_buy_orders()
        n_exposed = len({p.symbol for p in positions} | {o.symbol for o in pending})
        blocked = self._entry_block_reason(n_exposed)
        if budget_out:
            pass
        elif blocked:
            self.feed(cid, None, "Risk", "rejected", f"No new entries this cycle: {blocked}.")
        else:
            for score, sym, ind, price in self.screen(account, positions, pending):
                if self.llm_budget_exhausted():
                    self.system("LLM budget reached mid-cycle; skipping remaining candidates.", cid)
                    break
                try:
                    self._debate(cid, sym, ind, price, score)
                except AgentUnavailable as e:
                    self.system(str(e), cid, sym)
                except Exception as e:  # noqa: BLE001
                    log.exception("debate failed")
                    self.system(f"Debate on {sym} failed: {type(e).__name__}: {e}", cid, sym)
        self.snapshot_equity()

    def _entry_block_reason(self, n_positions: int) -> str:
        if self.store.kill_switch:
            return "kill switch is on"
        if self.halted_today():
            return "daily loss limit was hit today"
        if n_positions >= self.s.max_open_positions:
            return f"already at max open positions ({self.s.max_open_positions})"
        return ""

    def _review_position(self, cid: int, pos: Position, account) -> None:
        facts = self.fact_sheet(pos.symbol, account, self.broker.get_positions())
        facts["position"] = {
            "qty": pos.qty, "avg_entry": round(pos.avg_entry, 2),
            "unrealized_pct": round(pos.current_price / pos.avg_entry - 1, 4) if pos.avg_entry else 0.0,
            "stop": pos.stop, "take_profit": pos.take_profit,
        }
        d = self.team.review(facts)
        d.symbol = pos.symbol
        self.feed(cid, pos.symbol, "Chief", "sell" if d.action == "SELL" else "hold",
                  f"{d.action} (review): {d.rationale}")
        if d.action != "SELL":
            return
        res = risk.evaluate(d, account, self.broker.get_positions(), 0.0, pos.current_price, self.s,
                            self.store.kill_switch, self.halted_today())
        if not res.approved:
            self.feed(cid, pos.symbol, "Risk", "rejected", "Rejected: " + "; ".join(res.reasons))
            self.store.add_decision(cycle_id=cid, symbol=pos.symbol, action="SELL", confidence=d.confidence,
                                    size_pct=0, stop=None, target=None, rationale=d.rationale,
                                    risk_status="rejected", risk_reasons=res.reasons, qty=0, order_id=None)
            return
        oid = self.broker.close_position(pos.symbol)
        self.feed(cid, pos.symbol, "Risk", "approved", f"Approved: closing {res.qty} shares of {pos.symbol} (order {oid}).")
        self.store.add_decision(cycle_id=cid, symbol=pos.symbol, action="SELL", confidence=d.confidence, size_pct=0,
                                stop=None, target=None, rationale=d.rationale, risk_status="approved",
                                risk_reasons=res.reasons, qty=res.qty, order_id=str(oid))

    def _debate(self, cid: int, sym: str, ind: dict, price: float, score: float) -> None:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        facts = self.fact_sheet(sym, account, positions)
        self.feed(cid, sym, "Scout", "neutral", scout_pitch(sym, ind, price, score))

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa, fc = pool.submit(self.team.analyst, facts), pool.submit(self.team.charts, facts)
            analyst, charts = fa.result(), fc.result()
        self.feed(cid, sym, "Analyst", analyst.stance, analyst.text)
        levels = (f" Suggested stop {charts.stop}, target {charts.target}."
                  if charts.stop and charts.target and not charts.degraded else "")
        self.feed(cid, sym, "Charts", charts.stance, charts.text + levels)
        skeptic = self.team.skeptic(facts, analyst, charts)
        self.feed(cid, sym, "Skeptic", skeptic.stance,
                  skeptic.text + (f" Key risk: {skeptic.key_risk}."
                                  if skeptic.key_risk and skeptic.key_risk not in skeptic.text and not skeptic.degraded
                                  else ""))

        d = self.team.chief(facts, [analyst, charts, skeptic], self.s.risk_dict())
        d.symbol = sym
        # Round to cents BEFORE risk validation so the validated values are exactly the ones submitted.
        if d.stop_loss is not None:
            d.stop_loss = round(d.stop_loss, 2)
        if d.take_profit is not None:
            d.take_profit = round(d.take_profit, 2)
        self.feed(cid, sym, "Chief", d.action.lower(), self._chief_text(d))

        if d.action != "BUY":
            self.store.add_decision(cycle_id=cid, symbol=sym, action=d.action, confidence=d.confidence,
                                    size_pct=d.size_pct, stop=d.stop_loss, target=d.take_profit,
                                    rationale=d.rationale, risk_status="n/a", risk_reasons=[], qty=0, order_id=None)
            return

        # Re-read account/positions: nothing may be stale when sizing an order.
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        pending = self.broker.get_open_buy_orders()
        last = self.broker.get_latest_price(sym)
        res = risk.evaluate(d, account, positions, account.equity - account.last_equity, last, self.s,
                            self.store.kill_switch, self.halted_today(), pending=pending,
                            price_stale=self.broker.is_price_stale(sym), vix=self.current_vix())
        if res.halt:
            self.store.set_halted(self.broker.trading_date())
        order_id = None
        if res.approved:
            try:
                with self._order_lock:  # last-moment kill-switch check (flatten sets it, then closes everything)
                    if self.store.kill_switch:
                        raise RuntimeError("kill switch was set before the order was placed; order aborted")
                    order_id = str(self.broker.submit_bracket_buy(sym, res.qty, d.stop_loss, d.take_profit))
                self.feed(cid, sym, "Risk", "approved",
                          f"Approved {res.qty} shares of {sym} at ~${last:,.2f}, stop ${d.stop_loss:,.2f}, "
                          f"target ${d.take_profit:,.2f}. {res.reasons[0]}. Order {order_id}.")
            except Exception as e:  # noqa: BLE001
                res.approved = False
                res.reasons = [f"order failed at the broker: {e}"]
                self.feed(cid, sym, "Risk", "rejected", "Rejected: " + res.reasons[0])
        else:
            self.feed(cid, sym, "Risk", "rejected", "Rejected: " + "; ".join(res.reasons))
        self.store.add_decision(cycle_id=cid, symbol=sym, action="BUY", confidence=d.confidence, size_pct=d.size_pct,
                                stop=d.stop_loss, target=d.take_profit, rationale=d.rationale,
                                risk_status=res.status, risk_reasons=res.reasons,
                                qty=res.qty if res.approved else 0, order_id=order_id)

    @staticmethod
    def _chief_text(d: Decision) -> str:
        if d.action == "BUY":
            return (f"BUY {d.size_pct:.1%} of equity, stop {d.stop_loss}, target {d.take_profit}, "
                    f"confidence {d.confidence:.0%}. {d.rationale}")
        return f"{d.action} (confidence {d.confidence:.0%}). {d.rationale}"

    # --- scheduler entry point ---
    def tick(self, force: bool = False) -> str:
        """One scheduler beat. Returns 'cycle', 'busy', or 'closed'."""
        if not self.s.is_demo and not self.broker.is_market_open():
            if not self._closed_notice:
                self.system("Market closed, desk sleeping.")
                self._closed_notice = True
            if time.time() - self._last_closed_snapshot >= 3600:
                if self.lock.acquire(blocking=False):
                    try:
                        self.snapshot_equity()
                        self._last_closed_snapshot = time.time()
                    finally:
                        self.lock.release()
            return "closed"
        self._closed_notice = False
        return "cycle" if self.run_cycle() is not None else "busy"

    # --- API helpers ---
    def state(self) -> dict:
        acct = self._read("account", self.broker.get_account)
        day_pnl = acct.equity - acct.last_equity
        try:
            market_open, trading_date = self._read("clock", self.broker.clock_info)
        except Exception:  # noqa: BLE001
            market_open, trading_date = False, ""
        return {
            "mode": self.s.mode,
            "running": self.running,
            "cycle_active": self.cycle_active,
            "market_open": market_open,
            "kill_switch": self.store.kill_switch,
            "halted_today": bool(trading_date) and self.store.halted_date == trading_date,
            "equity": round(acct.equity, 2),
            "cash": round(acct.cash, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl / acct.last_equity, 5) if acct.last_equity else 0.0,
            "open_positions": len(self._read("positions", self.broker.get_positions)),
            "llm_spend_today": round(self.store.spend_today(), 4),
            "llm_budget": self.s.daily_llm_budget_usd,
            "llm_label": (f"{self.s.model_chief} · {self.s.llm_provider}" if self.s.uses_llm else "scripted"),
            "llm_free": self.s.llm_is_free,
            "llm_plan": self.s.uses_llm and self.s.llm_provider == "claude_code",
            "llm_calls_today": self.store.calls_today(),
            "llm_paused": (self.team.caller.paused() if hasattr(getattr(self.team, "caller", None), "paused")
                           else ""),
            "last_cycle_at": self.store.last_cycle_at(),
            "cycle_minutes": round(self.s.cycle_seconds / 60, 2),
            "provider": self.s.llm_provider if self.s.uses_llm else "scripted",
            "model_chief": self.s.model_chief if self.s.uses_llm else "",
            "vix": self.current_vix(),
            "index_buy_min_vix": self.s.index_buy_min_vix,
            "index_symbol": INDEX_SYMBOL,
            "gross_exposure": round(sum(p.market_value for p in self._read("positions", self.broker.get_positions)), 2),
            "next_cycle_at": self.next_cycle_at,
            "risk": self.s.risk_dict(),
        }

    WATCHLIST_TTL_S = 120.0  # dashboard watchlist: 1 bars request per symbol at most every 2 minutes

    def watchlist_view(self) -> list[dict]:
        """Every watchlist symbol with last price, day change, a 30-day sparkline and the Scout's screen inputs.
        Dashboard only (cached); trading never reads this."""
        hit = self._cache.get("watchlist")
        if hit and time.monotonic() - hit[0] < self.WATCHLIST_TTL_S:
            return hit[1]
        out = []
        for sym in self.s.watchlist:
            try:
                bars = self.broker.get_daily_bars(sym, 120)
                if len(bars) < 2:
                    continue
                last, prev = bars[-1].close, bars[-2].close
                ind = compute_indicators(bars, self.last_bar_fraction(bars))
                out.append({
                    "symbol": sym, "price": round(last, 2), "date": bars[-1].date,
                    "change_pct": round(last / prev - 1, 5) if prev else 0.0,
                    "spark": [round(b.close, 2) for b in bars[-30:]],
                    "score": screen_score(ind, last) if len(bars) >= 55 else None,
                    "rsi": ind.get("rsi14"), "ret_20d": ind.get("ret_20d"),
                    "volume_ratio": ind.get("volume_ratio20"),
                })
            except Exception as e:  # noqa: BLE001 - one bad symbol must not blank the panel
                log.warning("watchlist view failed for %s: %s", sym, e)
        self._cache["watchlist"] = (time.monotonic(), out)
        return out

    def positions_view(self) -> list[dict]:
        levels = self.store.latest_buy_levels()
        out = []
        for p in self._read("positions", self.broker.get_positions):
            stop, tp = p.stop, p.take_profit
            if stop is None and p.symbol in levels:
                stop = levels[p.symbol][0]
            if tp is None and p.symbol in levels:
                tp = levels[p.symbol][1]
            out.append({"symbol": p.symbol, "qty": p.qty, "avg_entry": round(p.avg_entry, 2),
                        "last": round(p.current_price, 2), "unrealized_pl": round(p.unrealized_pl, 2),
                        "unrealized_pct": round(p.current_price / p.avg_entry - 1, 4) if p.avg_entry else 0.0,
                        "stop": stop, "take_profit": tp})
        return out

    def equity_series(self) -> list[dict]:
        rows = self.store.get_equity()
        if not rows:
            return []
        e0 = rows[0]["equity"]
        spy0 = next((r["spy_price"] for r in rows if r["spy_price"]), None)
        return [{"ts": r["ts"], "equity": round(r["equity"], 2),
                 "spy": round(e0 * r["spy_price"] / spy0, 2) if spy0 and r["spy_price"] else None} for r in rows]

    def flatten(self) -> int:
        """Kill switch first (persisted), so no cycle can open new positions while/after we close everything."""
        with self._order_lock:
            self.store.set_kill_switch(True)
            n = self.broker.close_all()
        self._cache.clear()
        self.system(f"FLATTEN requested: kill switch ON, closed {n} position(s). Resume to allow buying again.")
        return n


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
