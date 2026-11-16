"""ClaudeCodeCaller with a fake `claude` CLI (no real calls, no plan usage)."""
import json
import subprocess

import pytest

from desk.agents.base import SHUTDOWN, AgentUnavailable
from desk.agents.providers import ClaudeCodeCaller, make_caller
from desk.config import load_settings

SCHEMA = {"type": "object", "properties": {"stance": {"type": "string"}}, "required": ["stance"]}


def _settings(tmp_path, **extra):
    env = {"DESK_MODE": "demo", "DEMO_AGENTS": "llm", "LLM_PROVIDER": "claude_code",
           "DESK_DB": str(tmp_path / "d.db"), "DESK_AUTOSTART": "0", **extra}
    return load_settings(env, use_dotenv=False)


def _ok(**over):
    d = {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
         "result": '{"stance": "bullish"}', "structured_output": {"stance": "bullish"},
         "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 200,
                   "output_tokens": 50}}
    d.update(over)
    return d


class FakeCLI:
    """Stands in for both subprocess.run (auth status) and subprocess.Popen (agent calls)."""

    def __init__(self, payload=None, rc=0, stderr="", auth=None, hang=False):
        self.payload, self.rc, self.stderr, self.hang = payload, rc, stderr, hang
        self.auth = auth if auth is not None else {"loggedIn": True, "authMethod": "claude.ai",
                                                   "subscriptionType": "max"}
        self.calls, self.killed = [], False

    def run(self, cmd, **kw):
        self.calls.append((cmd, kw))
        # the real `claude auth status` pretty-prints its JSON over several lines
        return subprocess.CompletedProcess(cmd, 0, json.dumps(self.auth, indent=2) + "\n", "")

    def popen(self, cmd, **kw):
        self.calls.append((cmd, kw))
        fake = self

        class Proc:
            returncode = None

            def communicate(self, input=None, timeout=None):
                if input is not None:
                    fake.stdin = input
                if fake.hang and not fake.killed:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                self.returncode = -9 if fake.killed else fake.rc
                out = json.dumps(fake.payload) if fake.payload is not None else "not json"
                return out + "\n", fake.stderr

            def kill(self):
                fake.killed = True
        return Proc()


def _caller(tmp_path, store, fake, clock=None, **extra):
    kw = {"clock": clock} if clock else {}
    c = ClaudeCodeCaller(_settings(tmp_path, **extra), store, runner=fake.run, popen=fake.popen, **kw)
    c.cli = "claude"
    return c


def test_settings_and_free_flag(tmp_path):
    s = _settings(tmp_path)
    assert s.llm_provider == "claude_code" and s.model_chief == "claude-opus-5-5" and s.llm_is_free


def test_success_uses_structured_output_records_zero_cost(tmp_path, store):
    fake = FakeCLI(_ok())
    r = _caller(tmp_path, store, fake).call("Analyst", "claude-opus-5-5", "low", "SYS", "USER", SCHEMA)
    assert r.data == {"stance": "bullish"} and r.cost == 0.0
    cmd, kw = fake.calls[-1]
    assert cmd[:2] == ["claude", "-p"] and json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5" and cmd[cmd.index("--effort") + 1] == "low"
    assert fake.stdin == "USER" and "SYS" in cmd
    assert store.calls_today() == 1 and store.spend_today() == 0.0


def test_only_a_minimal_environment_reaches_the_cli(tmp_path, store, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ALPACA_SECRET_KEY", "OPENAI_COMPAT_API_KEY"):
        monkeypatch.setenv(k, "secret-not-real")
    fake = FakeCLI(_ok())
    _caller(tmp_path, store, fake).call("Chief", "claude-opus-5-5", "medium", "S", "U", SCHEMA)
    env = fake.calls[-1][1]["env"]
    assert "PATH" in env and "HOME" in env
    assert not any(k.startswith(("ANTHROPIC", "ALPACA", "OPENAI")) for k in env)


def test_startup_requires_a_claude_plan_login(tmp_path, store, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
    ok = _caller(tmp_path, store, FakeCLI(_ok()))
    ok.check(["m"])
    assert ok.plan == "max"
    with pytest.raises(RuntimeError, match="not a Claude plan"):
        _caller(tmp_path, store, FakeCLI(_ok(), auth={"loggedIn": True, "authMethod": "api_key"})).check(["m"])
    with pytest.raises(RuntimeError, match="not logged in"):
        _caller(tmp_path, store, FakeCLI(_ok(), auth={"loggedIn": False})).check(["m"])


def test_falls_back_to_parsing_result_text(tmp_path, store):
    fake = FakeCLI(_ok(structured_output=None, result='```json\n{"stance": "neutral"}\n```'))
    assert _caller(tmp_path, store, fake).call("Charts", "m", "low", "S", "U", SCHEMA).data == {"stance": "neutral"}


def test_normal_answer_mentioning_limits_is_not_an_error(tmp_path, store):
    fake = FakeCLI(_ok(structured_output={"stance": "bearish"},
                       result="The usage limit of leverage... please run /login? no: just an answer"))
    c = _caller(tmp_path, store, fake)
    assert c.call("Skeptic", "m", "low", "S", "U", SCHEMA).data == {"stance": "bearish"} and not c.paused()


@pytest.mark.parametrize("text", ["Claude usage limit reached. Resets at 5pm.",
                                  "You've hit your session limit · resets 3pm",
                                  "You're out of extra usage", "You've hit your weekly limit"])
def test_plan_limit_pauses_for_the_cooldown(tmp_path, store, text):
    t = [1000.0]
    fake = FakeCLI(_ok(is_error=True, subtype="error", result=text), rc=1)
    c = _caller(tmp_path, store, fake, clock=lambda: t[0])
    with pytest.raises(AgentUnavailable, match="usage limit"):
        c.call("Analyst", "m", "low", "S", "U", SCHEMA)
    n = len(fake.calls)
    with pytest.raises(AgentUnavailable, match="paused"):
        c.call("Charts", "m", "low", "S", "U", SCHEMA)
    assert len(fake.calls) == n  # no CLI spawned while paused
    t[0] += 29 * 60
    assert c.paused()
    t[0] += 2 * 60
    fake.payload, fake.rc = _ok(), 0
    assert c.call("Chief", "m", "low", "S", "U", SCHEMA).data == {"stance": "bullish"}


def test_generic_rate_limit_text_is_not_a_plan_cooldown(tmp_path, store):
    t = [0.0]
    fake = FakeCLI(_ok(is_error=True, result="Rate limit reached for requests"), rc=1)
    c = _caller(tmp_path, store, fake, clock=lambda: t[0])
    with pytest.raises(AgentUnavailable, match="busy"):
        c.call("A", "m", "low", "S", "U", SCHEMA)
    t[0] += 121
    assert not c.paused()


def test_missing_user_is_filled_in_for_keychain_login(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    assert ClaudeCodeCaller._env().get("USER")


def test_overloaded_is_only_a_short_pause(tmp_path, store):
    t = [0.0]
    fake = FakeCLI(_ok(is_error=True, result="API Error: Overloaded", api_error_status=529), rc=1)
    c = _caller(tmp_path, store, fake, clock=lambda: t[0])
    with pytest.raises(AgentUnavailable, match="busy"):
        c.call("A", "m", "low", "S", "U", SCHEMA)
    t[0] += 121
    assert not c.paused()


def test_not_logged_in_and_generic_failures(tmp_path, store):
    fake = FakeCLI(_ok(is_error=True, result="Invalid API key · Please run /login"), rc=1)
    with pytest.raises(AgentUnavailable, match="not logged in"):
        _caller(tmp_path, store, fake).call("A", "m", "low", "S", "U", SCHEMA)
    with pytest.raises(AgentUnavailable, match="failed"):
        _caller(tmp_path, store, FakeCLI(None, rc=2, stderr="boom")).call("A", "m", "low", "S", "U", SCHEMA)


def test_timeout_and_shutdown_kill_the_child(tmp_path, store):
    t = [0.0]
    fake = FakeCLI(_ok(), hang=True)

    def clock():
        t[0] += 50.0
        return t[0]
    with pytest.raises(AgentUnavailable, match="too long"):
        _caller(tmp_path, store, fake, clock=clock).call("A", "m", "low", "S", "U", SCHEMA)
    assert fake.killed

    fake2 = FakeCLI(_ok(), hang=True)
    c = _caller(tmp_path, store, fake2)
    orig = fake2.popen

    def popen_then_shutdown(cmd, **kw):
        p = orig(cmd, **kw)
        SHUTDOWN.set()
        return p
    c.popen = popen_then_shutdown
    try:
        with pytest.raises(AgentUnavailable, match="shutting down"):
            c.call("A", "m", "low", "S", "U", SCHEMA)
        assert fake2.killed
    finally:
        SHUTDOWN.clear()


def test_missing_cli_fails_fast_with_help(tmp_path, store):
    s = _settings(tmp_path, CLAUDE_CODE_CLI="/nonexistent/claude")
    with pytest.raises(RuntimeError, match="not found"):
        make_caller(s, store)
