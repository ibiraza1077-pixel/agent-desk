"""Claude-backed agents. Any bad model output degrades to a neutral note / SKIP, never an exception."""
from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from ..models import AgentNote, Decision, clamp, opt_price
from .base import LLMCaller

DESK = (
    "You are part of a small research desk that paper-trades US stocks and ETFs (long only, simulated money). "
    "Be honest about uncertainty. Markets are noisy and 'no trade' is a good outcome whenever the edge is unclear. "
    "Use only the fact sheet provided; do not invent prices, news or numbers. "
)

STANCE = {"type": "string", "enum": ["bullish", "bearish", "neutral"]}
NUM = {"type": "number"}
STR = {"type": "string"}


def _schema(**props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


ANALYST_SCHEMA = _schema(stance=STANCE, confidence=NUM, note=STR)
CHARTS_SCHEMA = _schema(stance=STANCE, confidence=NUM, note=STR, stop_price=NUM, target_price=NUM)
SKEPTIC_SCHEMA = _schema(stance=STANCE, note=STR, key_risk=STR)
CHIEF_SCHEMA = _schema(action={"type": "string", "enum": ["BUY", "WAIT", "SKIP"]}, size_pct=NUM, stop_loss=NUM,
                       take_profit=NUM, confidence=NUM, rationale=STR)
REVIEW_SCHEMA = _schema(action={"type": "string", "enum": ["HOLD", "SELL"]}, confidence=NUM, rationale=STR)

ANALYST_SYS = DESK + (
    "You are the ANALYST. Read the news, catalysts and context in the fact sheet and give a stance "
    "(bullish/bearish/neutral), a confidence 0-1 and a 2-4 sentence note. Weak or absent news means neutral. "
    "Note: in 10 years of data, headline sentiment did not predict the next two weeks' returns versus SPY "
    "(news is priced in within hours), so weigh news mainly for risks and genuinely new, material events. "
    "Check that each headline is actually about this company."
)
CHARTS_SYS = DESK + (
    "You are CHARTS, the technical analyst. Read the indicators (trend vs SMAs, RSI, ATR, volume, returns) and give "
    "a stance, confidence 0-1, a 2-4 sentence note, and suggested stop_price (below price) and target_price "
    "(above price) if you were to buy now. Use ATR to place the stop sensibly."
)
SKEPTIC_SYS = DESK + (
    "You are the SKEPTIC. You see the Analyst's and Charts' notes. Argue the strongest honest case AGAINST entering "
    "this trade now: give a stance (bearish if the case against is strong, neutral if weak), a short note, and the "
    "single key_risk."
)
CHIEF_SYS = DESK + (
    "You are the CHIEF, who makes the final call after hearing everyone. Choose BUY, WAIT (revisit later) or SKIP "
    "(not interesting). For BUY give size_pct (fraction of equity, at most the max position size in the risk limits), "
    "stop_loss below price, take_profit above price (respect the reward:risk and stop-distance limits), confidence "
    "0-1 and a 2-3 sentence rationale. A hard-coded Risk Guard will reject anything outside the limits. "
    "Confidence means your probability that buying now beats simply holding SPY over the next ~2 weeks. "
    "Calibrate it against the fact sheet's `history` block when present: the base rate is about 50%, and no "
    "signal tested over 10 years (technicals, the Scout score, news sentiment, earnings) reliably beat it. So a "
    "confidence of 0.6 or more needs specific, strong evidence beyond those, and your rationale must name it. "
    "Exception, SPY (the index): there confidence means your probability that SPY is higher in ~2 weeks. Calibrate "
    "it to history.market_regime.spy_next_10d_when_vix_similar.up_share. SPY buys are only possible when "
    "history.market_regime.index_buy_window_open is true (VIX elevated after a selloff), the one regime where 10 "
    "years of data showed an index edge; otherwise choose WAIT for SPY. "
    "For WAIT or SKIP use size_pct 0 and stop_loss/take_profit 0."
)
REVIEW_SYS = DESK + (
    "You are the CHIEF reviewing an OPEN position. Decide HOLD or SELL given the thesis, price versus entry, "
    "stop and target, and current technicals. Do not sell merely from noise. Give confidence 0-1 and a 2-3 "
    "sentence rationale."
)


def _dump(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def _neutral(agent: str, reason: str) -> AgentNote:
    return AgentNote(agent=agent, stance="neutral", confidence=0.0, degraded=True,
                     text=f"No usable read: {reason}.")


def _stance(v: Any) -> str:
    return v if v in ("bullish", "bearish", "neutral") else "neutral"


class LLMTeam:
    uses_llm = True

    def __init__(self, settings: Settings, caller: LLMCaller):
        self.s = settings
        self.caller = caller

    def _ask(self, agent, model, effort, system, user, schema):
        return self.caller.call(agent, model, effort, system, user, schema)

    def analyst(self, facts: dict) -> AgentNote:
        r = self._ask("Analyst", self.s.model_analysts, self.s.effort_analysts, ANALYST_SYS, _dump(facts),
                      ANALYST_SCHEMA)
        if r.data is None:
            return _neutral("Analyst", r.reason)
        d = r.data
        return AgentNote(agent="Analyst", stance=_stance(d.get("stance")), confidence=clamp(d.get("confidence"), 0, 1),
                         text=str(d.get("note", "")).strip())

    def charts(self, facts: dict) -> AgentNote:
        r = self._ask("Charts", self.s.model_analysts, self.s.effort_analysts, CHARTS_SYS, _dump(facts), CHARTS_SCHEMA)
        if r.data is None:
            return _neutral("Charts", r.reason)
        d = r.data
        return AgentNote(agent="Charts", stance=_stance(d.get("stance")), confidence=clamp(d.get("confidence"), 0, 1),
                         text=str(d.get("note", "")).strip(), stop=opt_price(d.get("stop_price")),
                         target=opt_price(d.get("target_price")))

    def skeptic(self, facts: dict, analyst: AgentNote, charts: AgentNote) -> AgentNote:
        user = _dump({"facts": facts, "analyst": {"stance": analyst.stance, "confidence": analyst.confidence,
                                                  "note": analyst.text},
                      "charts": {"stance": charts.stance, "confidence": charts.confidence, "note": charts.text,
                                 "stop": charts.stop, "target": charts.target}})
        r = self._ask("Skeptic", self.s.model_analysts, self.s.effort_analysts, SKEPTIC_SYS, user, SKEPTIC_SCHEMA)
        if r.data is None:
            return _neutral("Skeptic", r.reason)
        d = r.data
        return AgentNote(agent="Skeptic", stance=_stance(d.get("stance")), text=str(d.get("note", "")).strip(),
                         key_risk=str(d.get("key_risk", "")).strip())

    def chief(self, facts: dict, notes: list[AgentNote], limits: dict) -> Decision:
        sym = facts.get("symbol", "?")
        user = _dump({"facts": facts, "risk_limits": limits,
                      "notes": [n.model_dump(exclude={"degraded"}) for n in notes]})
        r = self._ask("Chief", self.s.model_chief, self.s.effort_chief, CHIEF_SYS, user, CHIEF_SCHEMA)
        if r.data is None:
            return Decision(symbol=sym, action="SKIP", rationale=f"Skipping: {r.reason}.")
        d = r.data
        action = d.get("action") if d.get("action") in ("BUY", "WAIT", "SKIP") else "SKIP"
        return Decision(symbol=sym, action=action, size_pct=clamp(d.get("size_pct"), 0, self.s.max_position_pct),
                        stop_loss=opt_price(d.get("stop_loss")), take_profit=opt_price(d.get("take_profit")),
                        confidence=clamp(d.get("confidence"), 0, 1), rationale=str(d.get("rationale", "")).strip())

    def review(self, facts: dict) -> Decision:
        sym = facts.get("symbol", "?")
        r = self._ask("Chief", self.s.model_chief, self.s.effort_chief, REVIEW_SYS, _dump(facts), REVIEW_SCHEMA)
        if r.data is None:
            return Decision(symbol=sym, action="HOLD", rationale=f"Holding by default: {r.reason}.")
        d = r.data
        return Decision(symbol=sym, action="SELL" if d.get("action") == "SELL" else "HOLD",
                        confidence=clamp(d.get("confidence"), 0, 1), rationale=str(d.get("rationale", "")).strip())
