"""Free LLM backends with the same `.call()` contract as the Claude `LLMCaller`.

- OllamaCaller: a model running locally through Ollama (https://ollama.com). No key, no cost, data stays on
  this machine. Uses Ollama's native /api/chat with `format=<json schema>` for structured output.
- ClaudeCodeCaller: Claude (Opus 5.5 by default) through the Claude Code CLI (`claude -p`), using the logged-in
  Claude subscription (Pro/Max) instead of an API key. No per-token bill, but calls count against the plan's
  usage limits, so a usage-limit reply pauses the agents for a cooldown instead of failing every call.
- OpenAICompatCaller: any OpenAI-compatible /chat/completions endpoint (Groq, Google Gemini's OpenAI endpoint,
  OpenRouter, LM Studio, llama.cpp server...). Many have free tiers with rate limits.

Both degrade to an empty LLMResult (-> SKIP / neutral) on bad output and raise AgentUnavailable on transport
problems, exactly like LLMCaller, so the desk and RiskGuard behave identically whichever brain is plugged in.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config import Settings
from ..store import Store
from .base import SHUTDOWN, AgentUnavailable, LLMCaller, LLMResult, check_shutdown

log = logging.getLogger("desk.llm")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(text: Optional[str]) -> tuple[Optional[dict], str]:
    """Parse a JSON object from model text, tolerating ```json fences and chatter around the object."""
    if not text or not text.strip():
        return None, "the model returned no text"
    cleaned = _FENCE.sub("", text.strip())
    reason = "the model returned invalid JSON"
    for candidate in (cleaned, _outer_braces(cleaned)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data, ""
        reason = "the model returned JSON that is not an object"  # e.g. [{...}]: try the brace fallback next
    return None, reason


def _outer_braces(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else ""


def _error_detail(r: httpx.Response) -> str:
    """A short, safe description of a provider error: its JSON error message if any, never a raw page body."""
    try:
        err = r.json().get("error")
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict):
        err = err.get("message")
    if isinstance(err, str) and err.strip():
        return err.strip()[:200]
    return f"HTTP {r.status_code}"


def _budget_gate(settings: Settings, store: Store, agent: str) -> None:
    check_shutdown(agent)
    if settings.llm_is_free:
        return  # free providers are never budget-gated
    spent = store.spend_today()
    if spent >= settings.daily_llm_budget_usd:
        raise AgentUnavailable(f"{agent}: daily LLM budget reached (${spent:.2f} of "
                               f"${settings.daily_llm_budget_usd:.2f}); request not sent")


class OllamaCaller:
    """Local model via Ollama. Free: usage is recorded with cost 0."""

    def __init__(self, settings: Settings, store: Store, http: Optional[httpx.Client] = None):
        self.settings = settings
        self.store = store
        self.base = settings.ollama_url.rstrip("/")
        # Local generation on a laptop is slow (tens of seconds per answer, longer on first load). Non-streaming
        # replies send nothing until done, so this read timeout is the real per-call ceiling.
        self.http = http or httpx.Client(timeout=httpx.Timeout(240.0, connect=5.0))

    def check(self, models: list[str]) -> None:
        """Fail fast with a helpful message if Ollama isn't running or a model isn't pulled."""
        try:
            r = self.http.get(f"{self.base}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama is not reachable at {self.base}. Open the Ollama app (or run "
                               f"`ollama serve`) and try again.") from e
        have = {m.get("name") for m in r.json().get("models", [])}
        have |= {n.removesuffix(":latest") for n in have if n}
        missing = sorted({m for m in models if m not in have})
        if missing:
            raise RuntimeError("Ollama model(s) not downloaded: " + ", ".join(missing) +
                               ". Run: " + " && ".join(f"ollama pull {m}" for m in missing))

    def _post(self, body: dict) -> httpx.Response:
        return self.http.post(f"{self.base}/api/chat", json=body)

    def call(self, agent: str, model: str, effort: str, system: str, user: str, schema: dict) -> LLMResult:
        _budget_gate(self.settings, self.store, agent)
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "think": False,          # answer directly; thinking roughly triples latency on a laptop
            "keep_alive": "30m",     # keep the model loaded between cycles
            "format": schema,        # Ollama constrains decoding to this JSON schema
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            r = self._post(body)
            if r.status_code == 400 and "think" in r.text.lower():
                body.pop("think")  # model without a thinking toggle
                r = self._post(body)
        except httpx.TimeoutException as e:
            raise AgentUnavailable(f"{agent}: the local model took too long to answer; skipping") from e
        except httpx.HTTPError as e:
            raise AgentUnavailable(f"{agent}: could not reach Ollama at {self.base} (is the app running?)") from e
        if r.status_code == 404:
            raise AgentUnavailable(f"{agent}: Ollama model {model!r} is not downloaded (ollama pull {model})")
        if r.status_code >= 400:
            raise AgentUnavailable(f"{agent}: Ollama error {r.status_code}: {_error_detail(r)}")
        try:
            payload = r.json()
        except ValueError as e:
            raise AgentUnavailable(f"{agent}: Ollama returned a non-JSON reply") from e

        self.store.add_usage(agent, model, int(payload.get("prompt_eval_count") or 0),
                             int(payload.get("eval_count") or 0), 0.0)
        if payload.get("done_reason") == "length":
            return LLMResult(None, "the response was cut off (length limit)", 0.0)
        data, reason = parse_json_object((payload.get("message") or {}).get("content"))
        return LLMResult(data, reason, 0.0)


class OpenAICompatCaller:
    """Any OpenAI-compatible chat endpoint. Cost uses OPENAI_COMPAT_PRICE_IN/OUT ($/1M tokens; 0 = free tier)."""

    def __init__(self, settings: Settings, store: Store, http: Optional[httpx.Client] = None,
                 backoff_s: float = 5.0):
        self.settings = settings
        self.store = store
        self.backoff_s = backoff_s
        self.base = settings.openai_compat_base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))

    def call(self, agent: str, model: str, effort: str, system: str, user: str, schema: dict) -> LLMResult:
        _budget_gate(self.settings, self.store, agent)
        # json_object mode is the most widely supported structured-output option across free providers;
        # the schema goes in the prompt and pydantic validates/clamps the result afterwards.
        sys_prompt = (f"{system}\n\nRespond with ONLY a JSON object (no markdown) matching this JSON schema:\n"
                      f"{json.dumps(schema)}")
        body = {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        }
        key = self.settings.openai_compat_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}  # local servers need no key
        try:
            r = self.http.post(f"{self.base}/chat/completions", json=body, headers=headers)
        except httpx.TimeoutException as e:
            raise AgentUnavailable(f"{agent}: the LLM endpoint timed out; skipping") from e
        except httpx.HTTPError as e:
            raise AgentUnavailable(f"{agent}: could not reach the LLM endpoint; skipping") from e
        if r.status_code == 429:
            log.warning("rate limited by %s", self.base)
            time.sleep(self.backoff_s)
            raise AgentUnavailable(f"{agent}: free-tier rate limit hit; skipping this candidate")
        if r.status_code in (401, 403):
            raise AgentUnavailable(f"{agent}: the LLM endpoint rejected the key (check OPENAI_COMPAT_API_KEY)")
        if r.status_code >= 400:
            raise AgentUnavailable(f"{agent}: LLM endpoint error {r.status_code}: {_error_detail(r)}")
        try:
            payload = r.json()
            choice = payload["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise AgentUnavailable(f"{agent}: unexpected reply shape from the LLM endpoint") from e

        usage = payload.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cost = (tin * self.settings.openai_compat_price_in + tout * self.settings.openai_compat_price_out) / 1e6
        self.store.add_usage(agent, model, tin, tout, cost)
        if choice.get("finish_reason") == "length":
            return LLMResult(None, "the response was cut off (length limit)", cost)
        data, reason = parse_json_object((choice.get("message") or {}).get("content"))
        return LLMResult(data, reason, cost)


# Environment passed to the Claude Code CLI: only what it needs to find its login and run. Everything else (API keys
# that would switch it to API billing, the Alpaca keys loaded from .env, ...) is withheld.
_ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "TERM", "CLAUDE_CONFIG_DIR",
              # subscription token from `claude setup-token` (headless machines) - plan billing, not API
              "CLAUDE_CODE_OAUTH_TOKEN",
              # networks that need a proxy or a custom certificate authority
              "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
              "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE")
# Replies meaning the plan's usage window is used up (5-hour / weekly / extra-usage limits).
_PLAN_LIMIT = re.compile(r"usage limit|(usage|session|weekly|plan|spend(ing)?) limit reached|hit your|"
                         r"out of (extra )?usage|session limit|weekly limit|spend(ing)? limit|credit balance|quota",
                         re.IGNORECASE)
# Short-lived capacity errors: worth a brief pause, not a 30-minute one.
_TRANSIENT = re.compile(r"overloaded|rate.?limit|too many requests|try again", re.IGNORECASE)
_AUTH = re.compile(r"not logged in|please run /login|/login|authenticat|unauthori[sz]ed|invalid api key", re.IGNORECASE)
TRANSIENT_PAUSE_S = 120.0


class ClaudeCodeCaller:
    """Claude via the Claude Code CLI in headless mode, billed to the user's Claude subscription.

    Each call is an isolated, tool-less, one-shot `claude -p` run: no tools, no MCP servers, no settings/hooks/
    CLAUDE.md, no saved session, in an empty working directory, with a minimal environment. The JSON schema is
    enforced by the CLI (--json-schema), so the reply arrives as `structured_output`. At startup the CLI must report a
    claude.ai (subscription) login; an API-key login is refused so the desk can never run up an API bill this way.
    """

    def __init__(self, settings: Settings, store: Store, runner=subprocess.run, popen=subprocess.Popen,
                 clock=time.monotonic):
        self.settings = settings
        self.store = store
        self.run = runner          # short commands (--version, auth status)
        self.popen = popen         # agent calls: polled so a desk shutdown can stop them mid-call
        self.clock = clock
        self.cli = settings.claude_code_cli
        self.cwd = Path(settings.db_path).resolve().parent / ".claude-code-cwd"  # empty dir: nothing to discover
        self.cooldown_s = settings.claude_code_cooldown_minutes * 60
        self.call_timeout_s = 300.0
        self._paused_until = 0.0
        self._pause_reason = ""
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(2)  # Analyst + Charts run in parallel; never more

    def check(self, models: list[str]) -> None:
        exe = shutil.which(self.cli) or (self.cli if os.path.isfile(self.cli) else None)
        if not exe:
            raise RuntimeError(f"Claude Code CLI {self.cli!r} not found. Install Claude Code and log in with your "
                               "Claude plan (run `claude`, then /login), or set CLAUDE_CODE_CLI to its path.")
        self.cli = exe
        try:
            r = self.run([exe, "auth", "status"], capture_output=True, text=True, timeout=30, env=self._env(),
                         cwd=self._cwd())
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Could not run the Claude Code CLI at {exe}: {e}") from e
        status = _json_doc(r.stdout) or {}
        if not status.get("loggedIn"):
            raise RuntimeError("Claude Code is not logged in for this user (checked with USER/HOME set). Run "
                               "`claude auth login` in Terminal and sign in with your Claude plan.")
        if status.get("authMethod") != "claude.ai":
            raise RuntimeError(f"Claude Code is logged in via {status.get('authMethod')!r}, not a Claude plan, so "
                               "calls would be billed to an API account. Log in with your Claude plan (/login), or "
                               "use LLM_PROVIDER=anthropic if API billing is intended.")
        self.plan = status.get("subscriptionType") or "claude.ai"
        log.info("Claude Code ready: %s plan login", self.plan)

    def _cwd(self) -> Path:
        self.cwd.mkdir(parents=True, exist_ok=True)
        return self.cwd

    @staticmethod
    def _env() -> dict:
        env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOW or k.startswith("LC_")}
        if "USER" not in env:  # the CLI needs USER to find its keychain login (launchd/cron may not set it)
            try:
                import pwd
                env["USER"] = pwd.getpwuid(os.getuid()).pw_name
            except (ImportError, KeyError, OSError):
                pass
        return env

    def paused(self) -> str:
        """Why the agents are paused for plan limits ('' if not)."""
        with self._lock:
            return self._pause_reason if self.clock() < self._paused_until else ""

    def _pause(self, why: str, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, self.clock() + seconds)
            self._pause_reason = why
        log.warning("Claude: %s; pausing AI calls for %.0f min", why, seconds / 60)

    def _execute(self, agent: str, cmd: list[str], user: str) -> tuple[int, str, str]:
        """Run one CLI call, polling SHUTDOWN so stopping the desk kills an in-flight call."""
        proc = self.popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          cwd=self._cwd(), env=self._env())
        deadline = self.clock() + self.call_timeout_s
        first = True
        while True:
            try:
                out, err = proc.communicate(input=user if first else None, timeout=1.0)
                return proc.returncode, out or "", err or ""
            except subprocess.TimeoutExpired:
                first = False
                if SHUTDOWN.is_set() or self.clock() > deadline:
                    proc.kill()
                    proc.communicate()
                    if SHUTDOWN.is_set():
                        raise AgentUnavailable(f"{agent}: desk is shutting down; call cancelled")
                    raise AgentUnavailable(f"{agent}: Claude Code took too long to answer; skipping")

    def call(self, agent: str, model: str, effort: str, system: str, user: str, schema: dict) -> LLMResult:
        check_shutdown(agent)
        why = self.paused()
        if why:
            raise AgentUnavailable(f"{agent}: Claude paused ({why}); retrying after the pause")
        cmd = [self.cli, "-p", "--model", model, "--effort", effort, "--system-prompt", system,
               "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
               "--output-format", "json", "--json-schema", json.dumps(schema)]
        with self._slots:
            check_shutdown(agent)
            try:
                rc, out, err = self._execute(agent, cmd, user)
            except OSError as e:
                raise AgentUnavailable(f"{agent}: could not run the Claude Code CLI ({e})") from e
        payload = _last_json(out)
        if payload is None or payload.get("is_error") or rc != 0:
            text = (payload or {}).get("result")
            # classify on the CLI's own error text; stderr only when there is no JSON reply at all
            msg = text if isinstance(text, str) and text else err[-500:]  # stderr only when the reply has no text
            status = (payload or {}).get("api_error_status")
            if _PLAN_LIMIT.search(msg):
                self._pause("plan usage limit reached", self.cooldown_s)
                raise AgentUnavailable(f"{agent}: Claude plan usage limit reached; AI paused for "
                                       f"{self.cooldown_s / 60:.0f} min")
            if status in (429, 529) or _TRANSIENT.search(msg):
                self._pause("Claude is busy / rate limited", TRANSIENT_PAUSE_S)
                raise AgentUnavailable(f"{agent}: Claude is busy (rate limited); short pause")
            if status in (401, 403) or _AUTH.search(msg):
                raise AgentUnavailable(f"{agent}: Claude Code is not logged in (run `claude` and /login)")
            raise AgentUnavailable(f"{agent}: Claude Code call failed (exit {rc}"
                                   f"{f', HTTP {status}' if status else ''})")

        u = payload.get("usage") or {}
        tin = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0) + \
            int(u.get("cache_read_input_tokens") or 0)
        self.store.add_usage(agent, f"{model} (Claude plan)", tin, int(u.get("output_tokens") or 0), 0.0)
        if payload.get("stop_reason") == "refusal":
            return LLMResult(None, "the model declined", 0.0)
        data = payload.get("structured_output")
        if isinstance(data, dict):
            return LLMResult(data, "", 0.0)
        text = payload.get("result") if isinstance(payload.get("result"), str) else ""
        data, reason = parse_json_object(text)
        return LLMResult(data, reason, 0.0)


def _json_doc(stdout: Optional[str]) -> Optional[dict]:
    """A JSON object from CLI output, whether pretty-printed over many lines or one line among others."""
    text = (stdout or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return _last_json(text)


def _last_json(stdout: Optional[str]) -> Optional[dict]:
    """The CLI's JSON result object (the last JSON line printed)."""
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def make_caller(settings: Settings, store: Store):
    """The LLM backend chosen by LLM_PROVIDER."""
    if settings.llm_provider == "ollama":
        caller = OllamaCaller(settings, store)
        caller.check(sorted({settings.model_chief, settings.model_analysts}))
        return caller
    if settings.llm_provider == "openai_compat":
        return OpenAICompatCaller(settings, store)
    if settings.llm_provider == "claude_code":
        caller = ClaudeCodeCaller(settings, store)
        caller.check([settings.model_chief, settings.model_analysts])
        return caller
    return LLMCaller(settings, store)
