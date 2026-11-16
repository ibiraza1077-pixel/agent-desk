import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from desk.agents.base import AgentUnavailable, LLMCaller
from desk.agents.llm import LLMTeam
from desk.config import estimate_cost
from desk.models import AgentNote


def resp(payload=None, stop="end_turn", text=None, tin=1000, tout=500):
    body = text if text is not None else json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop, usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
        content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text=body)])


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def team(settings, store, *responses):
    client = FakeClient(*responses)
    return LLMTeam(settings, LLMCaller(settings, store, client=client, backoff_s=0)), client


FACTS = {"symbol": "AAPL", "price": 100.0}
GOOD_CHIEF = {"action": "BUY", "size_pct": 0.5, "stop_loss": 95, "take_profit": 112, "confidence": 1.7, "rationale": "ok"}


def test_valid_json_and_request_shape(settings, store):
    t, client = team(settings, store, resp({"stance": "bullish", "confidence": 0.7, "note": "Good."}))
    n = t.analyst(FACTS)
    assert (n.stance, n.confidence, n.text, n.degraded) == ("bullish", 0.7, "Good.", False)
    kw = client.calls[0]
    assert kw["max_tokens"] == 16000 and kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    sch = kw["output_config"]["format"]["schema"]
    assert sch["additionalProperties"] is False and set(sch["required"]) == set(sch["properties"])
    for forbidden in ("thinking", "temperature", "top_p", "top_k", "tool_choice"):
        assert forbidden not in kw
    assert kw["messages"][-1]["role"] == "user"  # no assistant prefill


def test_chief_values_are_clamped(settings, store):
    t, _ = team(settings, store, resp(GOOD_CHIEF))
    d = t.chief(FACTS, [], settings.risk_dict())
    assert d.action == "BUY" and d.size_pct == settings.max_position_pct and d.confidence == 1.0


@pytest.mark.parametrize("r", [
    resp(text="not json at all"),
    resp(text="[1,2,3]"),
    resp({"stance": "bullish"}, stop="refusal"),
    resp({"stance": "bullish"}, stop="max_tokens"),
    resp(text=""),
])
def test_bad_output_degrades_without_raising(settings, store, r):
    t, _ = team(settings, store, r, r, r, r, r)
    a = t.analyst(FACTS)
    assert a.stance == "neutral" and a.degraded and a.confidence == 0
    c = t.chief(FACTS, [AgentNote(agent="Analyst")], settings.risk_dict())
    assert c.action == "SKIP" and c.rationale
    assert t.review(FACTS).action == "HOLD"


def test_garbage_field_values_are_neutralised(settings, store):
    t, _ = team(settings, store, resp({"stance": "moonshot", "confidence": "high", "note": "x",
                                      "stop_price": -3, "target_price": "n/a"}))
    n = t.charts(FACTS)
    assert n.stance == "neutral" and n.confidence == 0 and n.stop is None and n.target is None


def test_cost_accounting(settings, store):
    assert estimate_cost("claude-opus-5-5", 1000, 500) == pytest.approx(0.014)
    assert estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost("some-future-model", 1_000_000, 0) == pytest.approx(4.0)  # unknown -> opus
    t, _ = team(settings, store, resp({"stance": "neutral", "confidence": 0.5, "note": "n"}, tin=2000, tout=1000))
    t.analyst(FACTS)
    assert store.spend_today() == pytest.approx(0.028)
    row = store._rows("SELECT * FROM llm_usage")[0]
    assert (row["agent"], row["input_tokens"], row["output_tokens"]) == ("Analyst", 2000, 1000)


def _status_error(cls, code):
    r = httpx.Response(code, request=httpx.Request("POST", "http://x"))
    return cls("boom", response=r, body=None)


@pytest.mark.parametrize("err", [
    _status_error(anthropic.RateLimitError, 429),
    _status_error(anthropic.InternalServerError, 500),
    _status_error(anthropic.AuthenticationError, 401),
    anthropic.APIConnectionError(request=httpx.Request("POST", "http://x")),
])
def test_api_errors_raise_agent_unavailable(settings, store, err):
    t, _ = team(settings, store, err)
    with pytest.raises(AgentUnavailable):
        t.analyst(FACTS)


def test_bad_request_on_beta_retries_plain_then_reports(settings, store):
    bad = _status_error(anthropic.BadRequestError, 400)
    t, client = team(settings, store, bad, bad)
    with pytest.raises(AgentUnavailable):
        t.analyst(FACTS)
    assert len(client.calls) == 2  # beta attempt, then plain attempt
