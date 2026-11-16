"""Free LLM backends: Ollama (local) and OpenAI-compatible endpoints, tested with a mocked HTTP transport."""
import json

import httpx
import pytest

from desk.agents.base import AgentUnavailable
from desk.agents.llm import LLMTeam
from desk.agents.providers import OllamaCaller, OpenAICompatCaller, parse_json_object
from desk.config import ConfigError, load_settings
from desk.store import Store

FACTS = {"symbol": "AAPL", "price": 100.0}


def mk_settings(tmp_path, **extra):
    # CLAUDE_CODE_CLI points nowhere so defaults behave as on a machine without Claude Code installed.
    env = {"DESK_MODE": "demo", "DESK_DB": str(tmp_path / "p.db"), "DESK_AUTOSTART": "0",
           "CLAUDE_CODE_CLI": "/nonexistent/claude", **extra}
    return load_settings(env, use_dotenv=False)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- parsing ----------

@pytest.mark.parametrize("text,ok", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),
    ('Sure! Here it is: {"a": 1} hope that helps', True),
    ('[{"a": 1}]', True),
    ("[1, 2]", False),
    ("not json", False),
    ("", False),
    (None, False),
])
def test_parse_json_object(text, ok):
    data, reason = parse_json_object(text)
    assert (data == {"a": 1}) if ok else (data is None and reason)


# ---------- config ----------

def test_default_provider_is_free_ollama(tmp_path):
    s = mk_settings(tmp_path)
    assert s.llm_provider == "ollama" and s.model_chief == "qwen3.5:9b" and not s.uses_llm


def test_default_prefers_claude_plan_when_cli_installed(tmp_path):
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    s = mk_settings(tmp_path, CLAUDE_CODE_CLI=str(cli))
    assert s.llm_provider == "claude_code" and s.model_chief == "claude-opus-5-5" and s.llm_is_free
    assert mk_settings(tmp_path, CLAUDE_CODE_CLI=str(cli), ANTHROPIC_API_KEY="k").llm_provider == "anthropic"


def test_demo_llm_with_ollama_needs_no_keys(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    assert s.uses_llm and s.llm_provider == "ollama"


def test_openai_compat_requires_url_and_model_but_key_is_optional(tmp_path):
    with pytest.raises(ConfigError) as e:
        mk_settings(tmp_path, DEMO_AGENTS="llm", LLM_PROVIDER="openai_compat")
    msg = str(e.value)
    assert "OPENAI_COMPAT_BASE_URL" in msg and "MODEL_CHIEF" in msg and "OPENAI_COMPAT_API_KEY" not in msg


def test_existing_claude_setup_stays_on_claude(tmp_path):
    s = mk_settings(tmp_path, DESK_MODE="paper", ANTHROPIC_API_KEY="sk-ant-x", ALPACA_API_KEY="a",
                    ALPACA_SECRET_KEY="b", DESK_DB=str(tmp_path / "paper.db"))
    assert s.llm_provider == "anthropic" and s.model_chief == "claude-opus-5-5" and not s.llm_is_free


def test_free_flags(tmp_path):
    assert mk_settings(tmp_path).llm_is_free  # scripted demo
    assert mk_settings(tmp_path, DEMO_AGENTS="llm").llm_is_free  # ollama
    base = dict(DEMO_AGENTS="llm", LLM_PROVIDER="openai_compat", OPENAI_COMPAT_BASE_URL="https://x/v1",
                MODEL_CHIEF="m", MODEL_ANALYSTS="m")
    assert mk_settings(tmp_path, **base).llm_is_free
    assert not mk_settings(tmp_path, **base, OPENAI_COMPAT_PRICE_OUT="0.6").llm_is_free
    with pytest.raises(ConfigError):
        mk_settings(tmp_path, **base, OPENAI_COMPAT_PRICE_IN="-1")


def test_zero_budget_does_not_starve_free_ollama(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm", DAILY_LLM_BUDGET_USD="0")
    store = Store(s.db_path)
    store.add_usage("Chief", "claude-opus-5-5", 1000, 1000, 5.0)  # earlier paid spend today
    c = OllamaCaller(s, store, http=client_for(lambda req: ollama_reply('{"x": 1}')))
    assert c.call("Analyst", "qwen3.5:9b", "low", "sys", "user", {"type": "object"}).data == {"x": 1}


def test_shutdown_stops_new_requests(tmp_path):
    from desk.agents.base import SHUTDOWN
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    c = OllamaCaller(s, Store(s.db_path), http=client_for(lambda req: ollama_reply('{"x": 1}')))
    SHUTDOWN.set()
    try:
        with pytest.raises(AgentUnavailable, match="shutting down"):
            c.call("Analyst", "qwen3.5:9b", "low", "sys", "user", {"type": "object"})
    finally:
        SHUTDOWN.clear()


def test_unknown_provider_and_demo_agents_rejected(tmp_path):
    with pytest.raises(ConfigError):
        mk_settings(tmp_path, LLM_PROVIDER="gpt")
    with pytest.raises(ConfigError):
        mk_settings(tmp_path, DEMO_AGENTS="maybe")


def test_openai_compat_key_not_in_repr(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm", LLM_PROVIDER="openai_compat",
                    OPENAI_COMPAT_BASE_URL="https://x.example/v1", OPENAI_COMPAT_API_KEY="sk-secret",
                    MODEL_CHIEF="m", MODEL_ANALYSTS="m")
    assert "sk-secret" not in repr(s)


# ---------- Ollama ----------

def ollama_reply(content, done_reason="stop"):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                     "done_reason": done_reason, "prompt_eval_count": 120, "eval_count": 40})


def test_ollama_request_shape_and_free_usage(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    store = Store(s.db_path)
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return ollama_reply(json.dumps({"stance": "bullish", "confidence": 0.7, "note": "Looks fine."}))

    t = LLMTeam(s, OllamaCaller(s, store, http=client_for(handler)))
    n = t.analyst(FACTS)
    assert (n.stance, n.confidence, n.degraded) == ("bullish", 0.7, False)
    body = seen[0]
    assert body["model"] == "qwen3.5:9b" and body["stream"] is False and body["think"] is False
    assert body["format"]["type"] == "object" and body["messages"][0]["role"] == "system"
    assert store.spend_today() == 0.0  # local model: free


@pytest.mark.parametrize("reply,reason", [
    (lambda: ollama_reply("garbage"), "invalid JSON"),
    (lambda: ollama_reply('{"stance": "bullish"', done_reason="length"), "cut off"),
    (lambda: ollama_reply(""), "no text"),
])
def test_ollama_bad_output_degrades(tmp_path, reply, reason):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    t = LLMTeam(s, OllamaCaller(s, Store(s.db_path), http=client_for(lambda req: reply())))
    n = t.analyst(FACTS)
    assert n.degraded and n.stance == "neutral"
    d = t.chief(FACTS, [n], {})
    assert d.action == "SKIP"


def test_ollama_retries_without_think_for_models_that_lack_it(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    bodies = []

    def handler(req):
        b = json.loads(req.content)
        bodies.append(b)
        if "think" in b:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return ollama_reply('{"stance": "neutral", "confidence": 0.5, "note": "Meh."}')

    n = LLMTeam(s, OllamaCaller(s, Store(s.db_path), http=client_for(handler))).analyst(FACTS)
    assert not n.degraded and len(bodies) == 2 and "think" not in bodies[1]


@pytest.mark.parametrize("handler,msg", [
    (lambda req: httpx.Response(404, json={"error": "model not found"}), "not downloaded"),
    (lambda req: httpx.Response(500, text="boom"), "Ollama error 500"),
])
def test_ollama_errors_raise_agent_unavailable(tmp_path, handler, msg):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    c = OllamaCaller(s, Store(s.db_path), http=client_for(handler))
    with pytest.raises(AgentUnavailable, match=msg):
        c.call("Analyst", "qwen3.5:9b", "low", "sys", "user", {"type": "object"})


def test_ollama_unreachable_raises_agent_unavailable(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")

    def handler(req):
        raise httpx.ConnectError("refused")

    c = OllamaCaller(s, Store(s.db_path), http=client_for(handler))
    with pytest.raises(AgentUnavailable, match="could not reach Ollama"):
        c.call("Analyst", "qwen3.5:9b", "low", "sys", "user", {"type": "object"})
    with pytest.raises(RuntimeError, match="not reachable"):
        c.check(["qwen3.5:9b"])


def test_ollama_check_reports_missing_model(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm")
    tags = lambda req: httpx.Response(200, json={"models": [{"name": "gemma4:12b"}, {"name": "llama3:latest"}]})
    c = OllamaCaller(s, Store(s.db_path), http=client_for(tags))
    c.check(["gemma4:12b", "llama3"])  # ":latest" is implied
    with pytest.raises(RuntimeError, match="ollama pull qwen3.5:9b"):
        c.check(["qwen3.5:9b"])


# ---------- OpenAI-compatible ----------

def compat_settings(tmp_path):
    return mk_settings(tmp_path, DEMO_AGENTS="llm", LLM_PROVIDER="openai_compat",
                       OPENAI_COMPAT_BASE_URL="https://free.example/v1/", OPENAI_COMPAT_API_KEY="k",
                       MODEL_CHIEF="free-model", MODEL_ANALYSTS="free-model")


def test_openai_compat_request_and_parse(tmp_path):
    s = compat_settings(tmp_path)
    store = Store(s.db_path)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": '```json\n{"stance": "bearish", '
                                                                         '"confidence": 0.8, "note": "No."}\n```'}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}})

    n = LLMTeam(s, OpenAICompatCaller(s, store, http=client_for(handler))).analyst(FACTS)
    assert (n.stance, n.confidence) == ("bearish", 0.8)
    req = seen[0]
    assert str(req.url) == "https://free.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["response_format"] == {"type": "json_object"} and "JSON schema" in body["messages"][0]["content"]
    assert store.spend_today() == 0.0


@pytest.mark.parametrize("status,msg", [(429, "rate limit"), (401, "rejected the key"), (503, "error 503")])
def test_openai_compat_http_errors(tmp_path, status, msg):
    s = compat_settings(tmp_path)
    c = OpenAICompatCaller(s, Store(s.db_path), backoff_s=0,
                           http=client_for(lambda req: httpx.Response(status, text="x")))
    with pytest.raises(AgentUnavailable, match=msg):
        c.call("Chief", "free-model", "low", "sys", "user", {"type": "object"})


def test_error_text_is_the_provider_message_not_a_raw_page(tmp_path):
    s = compat_settings(tmp_path)
    page = lambda req: httpx.Response(502, text="<html><body>proxy junk Bearer sk-123</body></html>")
    c = OpenAICompatCaller(s, Store(s.db_path), http=client_for(page))
    with pytest.raises(AgentUnavailable) as e:
        c.call("Chief", "free-model", "low", "sys", "user", {"type": "object"})
    assert "html" not in str(e.value) and "sk-123" not in str(e.value) and "HTTP 502" in str(e.value)
    msg = lambda req: httpx.Response(400, json={"error": {"message": "model not found"}})
    with pytest.raises(AgentUnavailable, match="model not found"):
        OpenAICompatCaller(s, Store(s.db_path), http=client_for(msg)).call(
            "Chief", "free-model", "low", "sys", "user", {"type": "object"})


def test_openai_compat_no_key_sends_no_auth_and_paid_endpoint_is_metered(tmp_path):
    s = mk_settings(tmp_path, DEMO_AGENTS="llm", LLM_PROVIDER="openai_compat",
                    OPENAI_COMPAT_BASE_URL="http://localhost:1234/v1", MODEL_CHIEF="m", MODEL_ANALYSTS="m",
                    OPENAI_COMPAT_PRICE_IN="1", OPENAI_COMPAT_PRICE_OUT="2")
    store = Store(s.db_path)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}],
                                         "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}})

    r = OpenAICompatCaller(s, store, http=client_for(handler)).call("Chief", "m", "low", "s", "u", {})
    assert "authorization" not in seen[0].headers
    assert r.cost == pytest.approx(2.0) and store.spend_today() == pytest.approx(2.0)


def test_openai_compat_length_degrades(tmp_path):
    s = compat_settings(tmp_path)
    reply = lambda req: httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                               "message": {"content": '{"a":'}}]})
    r = OpenAICompatCaller(s, Store(s.db_path), http=client_for(reply)).call(
        "Chief", "free-model", "low", "sys", "user", {"type": "object"})
    assert r.data is None and "cut off" in r.reason
