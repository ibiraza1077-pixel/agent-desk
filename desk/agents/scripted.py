"""Deterministic rule-based agents for demo mode. Same interface as LLMTeam, zero network."""
from __future__ import annotations

import zlib

from ..config import Settings
from ..models import AgentNote, Decision, clamp

POS_WORDS = ("beats", "raises", "upgrade", "record", "partnership", "expand", "accelerating")
NEG_WORDS = ("cut", "probe", "misses", "lower", "disruption", "concerns", "cautious", "underperform")


def pick(key: str, options: list[str]) -> str:
    """Stable pseudo-random template choice (crc32, unlike str hash(), is the same on every run)."""
    return options[zlib.crc32(key.encode()) % len(options)]


def news_score(facts: dict) -> int:
    score = 0
    for item in facts.get("news", []):
        t = (item["headline"] if isinstance(item, dict) else str(item)).lower()
        score += sum(w in t for w in POS_WORDS) - sum(w in t for w in NEG_WORDS)
    return score


class ScriptedTeam:
    uses_llm = False

    def __init__(self, settings: Settings):
        self.s = settings

    def analyst(self, facts: dict) -> AgentNote:
        sym, ind = facts["symbol"], facts["indicators"]
        sc = news_score(facts)
        mom = ind.get("ret_20d") or 0.0
        key = f"{sym}{facts.get('as_of')}A"
        if sc > 0:
            stance, conf = "bullish", clamp(0.62 + 0.08 * sc + max(0, mom) * 1.5, 0, 0.9)
            if mom > 0:
                text = pick(key, [
                    f"Headlines for {sym} lean positive, and the tape agrees with a {mom:+.1%} move over 20 days. "
                    "Sentiment looks supportive, though one good headline is not a thesis.",
                    f"News flow is constructive for {sym}: the catalyst reads as real rather than noise. "
                    "I would not fade it, but I would want price to confirm.",
                    f"{sym} has a fresh positive catalyst and decent follow-through in price. Context favours buyers for now.",
                ])
            else:
                text = pick(key, [
                    f"Headlines for {sym} lean positive even though price is {mom:+.1%} over 20 days, so the market "
                    "has not bought the story yet. A good headline is not a thesis.",
                    f"News flow is constructive for {sym}, but price has not followed ({mom:+.1%} over 20 days). "
                    "I would want the chart to confirm before leaning on it.",
                ])
        elif sc < 0:
            stance, conf = "bearish", clamp(0.6 + 0.08 * -sc, 0, 0.85)
            text = pick(key, [
                f"The news around {sym} skews negative, and that is the sort of overhang that tends to cap rallies.",
                f"Headlines on {sym} are unhelpful: downgrades or guidance worries usually take time to digest.",
                f"Negative flow on {sym}. Even if price holds, I would not want to be the buyer into this.",
            ])
        else:
            stance = "bullish" if mom > 0.04 else "neutral"
            conf = 0.5 + min(abs(mom), 0.1)
            text = pick(key, [
                f"Nothing decisive in the news for {sym}; the story is mostly price-driven ({mom:+.1%} over 20 days).",
                f"No clear catalyst for {sym}. Routine headlines only, so I lean on the chart for direction.",
                f"Quiet news tape for {sym}. Neither a reason to buy nor a reason to avoid.",
            ])
        return AgentNote(agent="Analyst", stance=stance, confidence=round(conf, 2), text=text)

    def charts(self, facts: dict) -> AgentNote:
        sym, px, ind = facts["symbol"], facts["price"], facts["indicators"]
        sma20, sma50 = ind.get("sma20") or px, ind.get("sma50") or px
        rsi, atr = ind.get("rsi14") or 50.0, ind.get("atr14") or px * 0.02
        vr, hi = ind.get("volume_ratio20") or 1.0, ind.get("dist_from_20d_high") or 0.0
        uptrend, downtrend = px > sma20 > sma50, px < sma20 < sma50
        key = f"{sym}{facts.get('as_of')}C"
        stop = round(max(px - 1.5 * atr, px * (1 - self.s.max_stop_distance_pct + 0.005)), 2)
        target = round(px + max(2.4 * atr, 1.6 * (px - stop)), 2)
        if uptrend and rsi < 75:
            stance, conf = "bullish", clamp(0.6 + (0.1 if vr > 1.1 else 0) + (0.08 if hi > -0.02 else 0), 0, 0.9)
            text = pick(key, [
                f"{sym} sits above a rising 20/50-day stack with RSI {rsi:.0f}, so the trend is intact without being stretched. "
                f"Volume is {vr:.1f}x normal. A stop near {stop} would cover about 1.5 ATR.",
                f"Constructive chart: higher than both moving averages, RSI {rsi:.0f}, {abs(hi):.1%} off the 20-day high. "
                f"Buyers are in control; invalidation is below {stop}.",
            ])
        elif downtrend or rsi > 80:
            stance, conf = "bearish", clamp(0.6 + (0.1 if rsi > 80 else 0), 0, 0.9)
            text = pick(key, [
                f"{sym} is {'below both averages in a downtrend' if downtrend else f'overbought (RSI {rsi:.0f})'}; "
                "the risk-reward for a fresh long is poor.",
                f"Technicals argue for patience on {sym}: {'trend is down' if downtrend else 'momentum is overextended'}.",
            ])
        else:
            stance, conf = "neutral", 0.5
            text = pick(key, [
                f"{sym} is chopping around its 20-day average with RSI {rsi:.0f}; no clean setup yet.",
                f"Mixed picture for {sym}: trend signals disagree, volume {vr:.1f}x. I would wait for a decisive move.",
            ])
        return AgentNote(agent="Charts", stance=stance, confidence=round(conf, 2), text=text, stop=stop, target=target)

    def skeptic(self, facts: dict, analyst: AgentNote, charts: AgentNote) -> AgentNote:
        sym, px, ind = facts["symbol"], facts["price"], facts["indicators"]
        rsi, sma20 = ind.get("rsi14") or 50.0, ind.get("sma20") or px
        vr, ret5 = ind.get("volume_ratio20") or 1.0, ind.get("ret_5d") or 0.0
        key = f"{sym}{facts.get('as_of')}S"
        if rsi > 68:
            risk = f"RSI {rsi:.0f} means a lot of the move may already be priced in"
        elif px > sma20 * 1.06:
            risk = f"price is {(px / sma20 - 1):.1%} above its 20-day average, so a pullback is likely"
        elif vr < 0.8:
            risk = f"volume is only {vr:.1f}x average, so the move lacks conviction"
        elif analyst.stance == "bearish":
            risk = "the news flow is negative and could drag the stock lower"
        elif ret5 > 0.05:
            risk = f"a {ret5:.1%} jump in five days invites profit-taking"
        else:
            risk = "broad-market weakness could take the stock down regardless of its own setup"
        strong = rsi > 68 or analyst.stance == "bearish" or charts.stance == "bearish"
        text = pick(key, [
            f"Playing devil's advocate: {risk}. Entering now means paying up for a setup that can fail quickly.",
            f"My concern is simple: {risk}. I would want a better price or more proof.",
            f"The bull case is real but not free. {risk[0].upper() + risk[1:]}.",
        ])
        return AgentNote(agent="Skeptic", stance="bearish" if strong else "neutral", text=text, key_risk=risk)

    def chief(self, facts: dict, notes: list[AgentNote], limits: dict) -> Decision:
        sym, px = facts["symbol"], facts["price"]
        a = next(n for n in notes if n.agent == "Analyst")
        c = next(n for n in notes if n.agent == "Charts")
        k = next(n for n in notes if n.agent == "Skeptic")
        key = f"{sym}{facts.get('as_of')}X"
        bulls = (a.stance == "bullish") + (c.stance == "bullish")
        bears = (a.stance == "bearish") + (c.stance == "bearish")
        conf = round(clamp(0.45 + 0.15 * bulls - 0.12 * bears + 0.1 * (a.confidence + c.confidence) / 2
                           - (0.04 if k.stance == "bearish" else 0), 0, 0.95), 2)
        if c.stance == "bullish" and bulls >= 1 and bears == 0 and conf >= 0.6:
            size = round(clamp(0.04 + (conf - 0.6) * 0.2, 0.03, limits["max_position_pct"]), 3)
            return Decision(symbol=sym, action="BUY", size_pct=size, stop_loss=c.stop, take_profit=c.target,
                            confidence=conf, rationale=pick(key, [
                                f"Trend and flow line up for {sym}. The Skeptic's point ({k.key_risk}) is noted and "
                                f"the stop at {c.stop} caps the downside. Taking a {size:.0%} starter position.",
                                f"Enough agreement to act on {sym}: chart is constructive and the news does not fight it. "
                                f"Sizing at {size:.0%} with the stop at {c.stop} and target {c.target}.",
                            ]))
        if bears >= 1 and bulls == 0:
            return Decision(symbol=sym, action="SKIP", confidence=conf, rationale=pick(key, [
                f"No edge on {sym}: the read is negative or the chart is weak. Moving on.",
                f"Passing on {sym}. The desk leans bearish and I do not want to fight it.",
            ]))
        return Decision(symbol=sym, action="WAIT", confidence=conf, rationale=pick(key, [
            f"{sym} is interesting but not compelling yet. Waiting for cleaner confirmation.",
            f"Mixed signals on {sym}; {k.key_risk}. I will revisit next cycle.",
        ]))

    def review(self, facts: dict) -> Decision:
        sym, px, ind = facts["symbol"], facts["price"], facts["indicators"]
        pos = facts["position"]
        sma20, rsi = ind.get("sma20") or px, ind.get("rsi14") or 50.0
        pnl = pos["unrealized_pct"]
        if px < sma20 * 0.97 and rsi < 40:
            return Decision(symbol=sym, action="SELL", confidence=0.7,
                            rationale=f"{sym} broke below its 20-day average with weak momentum (RSI {rsi:.0f}). "
                                      f"Thesis is damaged at {pnl:+.1%}; closing rather than waiting for the stop.")
        if pnl > 0.05 and rsi > 78:
            return Decision(symbol=sym, action="SELL", confidence=0.65,
                            rationale=f"{sym} is up {pnl:+.1%} and overbought (RSI {rsi:.0f}). Taking profit.")
        return Decision(symbol=sym, action="HOLD", confidence=0.6,
                        rationale=f"{sym} thesis intact at {pnl:+.1%}; stop and target stand. Holding.")
