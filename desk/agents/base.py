"""Agent interface plus the shared Claude call helper (cost accounting, error handling)."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import anthropic

from ..config import Settings, estimate_cost
from ..models import AgentNote, Decision
from ..store import Store

log = logging.getLogger("desk.llm")

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Set when the server is shutting down: callers stop sending new LLM requests so a cycle running in a worker
# thread finishes quickly (at most the one request already in flight) instead of blocking Ctrl-C for minutes.
SHUTDOWN = threading.Event()


class AgentUnavailable(Exception):
    """The LLM could not be used for this candidate (rate limit, outage, auth...). Skip it."""


def check_shutdown(agent: str) -> None:
    if SHUTDOWN.is_set():
        raise AgentUnavailable(f"{agent}: desk is shutting down; request not sent")


class AgentTeam(Protocol):
    uses_llm: bool

    def analyst(self, facts: dict) -> AgentNote: ...
    def charts(self, facts: dict) -> AgentNote: ...
    def skeptic(self, facts: dict, analyst: AgentNote, charts: AgentNote) -> AgentNote: ...
    def chief(self, facts: dict, notes: list[AgentNote], limits: dict) -> Decision: ...
    def review(self, facts: dict) -> Decision: ...


@dataclass
class LLMResult:
    data: Optional[dict]  # parsed JSON, or None when degraded
    reason: str = ""  # why it degraded (refusal, max_tokens, invalid JSON)
    cost: float = 0.0


class LLMCaller:
    """Calls Claude for structured JSON, records usage/cost, and degrades instead of raising on bad output."""

    def __init__(self, settings: Settings, store: Store, client: Any = None, backoff_s: float = 5.0):
        self.settings = settings
        self.store = store
        self.backoff_s = backoff_s
        if client is None:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)  # key never logged
        self.client = client

    def _create(self, params: dict) -> Any:
        # Opt into server-side refusal fallbacks. beta.messages.create takes `fallbacks=` and `betas=` natively
        # in the installed SDK (2.6.0), so no extra_body is needed. If the beta call is rejected (signature or a
        # 400 on a model without fallback support) we retry once on the plain endpoint with identical params.
        try:
            return self.client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
        except (TypeError, AttributeError, anthropic.BadRequestError) as e:
            log.info("beta fallbacks call rejected (%s); using plain messages.create", type(e).__name__)
            return self.client.messages.create(**params)

    def call(self, agent: str, model: str, effort: str, system: str, user: str, schema: dict) -> LLMResult:
        check_shutdown(agent)
        # Hard budget gate: every LLM request (debates and position reviews alike) passes through here.
        spent = self.store.spend_today()
        if spent >= self.settings.daily_llm_budget_usd:
            raise AgentUnavailable(f"{agent}: daily LLM budget reached (${spent:.2f} of "
                                   f"${self.settings.daily_llm_budget_usd:.2f}); request not sent")
        params = dict(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}, "effort": effort},
        )
        try:
            resp = self._create(params)
        except anthropic.RateLimitError as e:
            log.warning("rate limited: %s", e)
            time.sleep(self.backoff_s)
            raise AgentUnavailable(f"{agent}: rate limited by the Claude API; skipping this candidate") from e
        except anthropic.AuthenticationError as e:
            raise AgentUnavailable(f"{agent}: Claude API authentication failed (check ANTHROPIC_API_KEY)") from e
        except anthropic.BadRequestError as e:
            raise AgentUnavailable(f"{agent}: Claude API rejected the request: {getattr(e, 'message', e)}") from e
        except anthropic.APIStatusError as e:
            raise AgentUnavailable(f"{agent}: Claude API error {getattr(e, 'status_code', '?')}; skipping") from e
        except anthropic.APIConnectionError as e:
            raise AgentUnavailable(f"{agent}: could not reach the Claude API; skipping") from e

        usage = getattr(resp, "usage", None)
        tin, tout = int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
        cost = estimate_cost(model, tin, tout)
        self.store.add_usage(agent, model, tin, tout, cost)

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            return LLMResult(None, "the model declined to answer (refusal)", cost)
        if stop == "max_tokens":
            return LLMResult(None, "the response was cut off (max_tokens)", cost)
        text = next((b.text for b in getattr(resp, "content", []) or [] if getattr(b, "type", "") == "text"), None)
        if not text:
            return LLMResult(None, "the model returned no text", cost)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return LLMResult(None, "the model returned invalid JSON", cost)
        if not isinstance(data, dict):
            return LLMResult(None, "the model returned JSON that is not an object", cost)
        return LLMResult(data, "", cost)
