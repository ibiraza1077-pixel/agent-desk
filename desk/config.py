"""Settings loaded from environment / .env, with validation."""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

# $ per 1M tokens: (input, output). Unknown models are priced as Opus.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_PRICE_KEY = "claude-opus-5-5"

EFFORTS = ("low", "medium", "high", "xhigh", "max")
PROVIDERS = ("claude_code", "ollama", "openai_compat", "anthropic")
DEFAULT_MODELS = {"claude_code": "claude-opus-5-5", "ollama": "qwen3.5:9b", "anthropic": "claude-opus-5-5",
                  "openai_compat": ""}
INDEX_SYMBOL = "SPY"  # the index the 10-year study tested for the VIX-regime effect
DEFAULT_WATCHLIST = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,JPM,XOM"


class ConfigError(ValueError):
    """Raised for invalid or missing configuration (message is user-facing)."""


def price_for(model: str) -> tuple[float, float]:
    return PRICING.get(model, PRICING[DEFAULT_PRICE_KEY])


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class Settings:
    mode: str = "demo"
    anthropic_api_key: str = field(default="", repr=False)
    alpaca_api_key: str = field(default="", repr=False)
    alpaca_secret_key: str = field(default="", repr=False)
    watchlist: list[str] = field(default_factory=lambda: DEFAULT_WATCHLIST.split(","))
    cycle_minutes: float = 15
    demo_cycle_seconds: float = 20
    max_candidates_per_cycle: int = 3
    demo_seed: int = 42
    # risk
    max_position_pct: float = 0.10
    max_open_positions: int = 5
    max_gross_exposure_pct: float = 0.60
    daily_loss_limit_pct: float = 0.02
    min_confidence: float = 0.6
    max_stop_distance_pct: float = 0.08
    min_reward_risk: float = 1.5
    # llm
    llm_provider: str = "anthropic"  # load_settings defaults to the free local "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = field(default="", repr=False)
    # $ per 1M tokens for openai_compat; leave 0 for free tiers / local servers. Set them for a paid endpoint
    # so spend is metered and DAILY_LLM_BUDGET_USD applies.
    openai_compat_price_in: float = 0.0
    openai_compat_price_out: float = 0.0
    demo_agents: str = "scripted"  # demo mode: "scripted" (offline) or "llm" (real agents on the sim market)
    model_chief: str = "claude-opus-5-5"
    model_analysts: str = "claude-opus-5-5"
    effort_chief: str = "medium"
    effort_analysts: str = "low"
    daily_llm_budget_usd: float = 10.0
    # runtime
    # SPY buys only when the latest VIX close is at least this (0 disables). 30: the 10-year study's edge sits above
    # 30 (SPY +2.4% avg over 10 days, up 76% of the time); at 25-30 the average was higher but the up-rate wasn't.
    index_buy_min_vix: float = 30.0
    claude_code_cli: str = "claude"         # Claude Code CLI used by LLM_PROVIDER=claude_code
    claude_code_cooldown_minutes: float = 30.0  # pause after a plan usage-limit reply
    db_path: str = ""
    autostart: bool = True
    host: str = "127.0.0.1"
    phone_access: bool = False   # DESK_PHONE_ACCESS=1: also serve on the home network, behind an access key
    port: int = 8000

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def uses_llm(self) -> bool:
        return not self.is_demo or self.demo_agents == "llm"

    @property
    def llm_is_free(self) -> bool:
        """True when no paid LLM calls can happen: scripted agents, local Ollama, or an unpriced openai_compat."""
        if not self.uses_llm or self.llm_provider in ("ollama", "claude_code"):
            return True  # claude_code: covered by the Claude subscription (plan usage limits, no per-token bill)
        if self.llm_provider == "openai_compat":
            return self.openai_compat_price_in == 0 and self.openai_compat_price_out == 0
        return False

    @property
    def cycle_seconds(self) -> float:
        return self.demo_cycle_seconds if self.is_demo else self.cycle_minutes * 60

    def risk_dict(self) -> dict:
        return {
            "max_position_pct": self.max_position_pct,
            "max_open_positions": self.max_open_positions,
            "max_gross_exposure_pct": self.max_gross_exposure_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "min_confidence": self.min_confidence,
            "max_stop_distance_pct": self.max_stop_distance_pct,
            "min_reward_risk": self.min_reward_risk,
        }

    def validate(self) -> None:
        if self.mode not in ("demo", "paper"):
            raise ConfigError(
                f"DESK_MODE must be 'demo' or 'paper' (got {self.mode!r}). "
                "Live trading is deliberately unsupported."
            )
        if self.llm_provider not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got {self.llm_provider!r}).")
        if self.demo_agents not in ("scripted", "llm"):
            raise ConfigError(f"DEMO_AGENTS must be 'scripted' or 'llm' (got {self.demo_agents!r}).")
        required: list[tuple[str, str]] = []
        if self.mode == "paper":
            required += [("ALPACA_API_KEY", self.alpaca_api_key), ("ALPACA_SECRET_KEY", self.alpaca_secret_key)]
        if self.uses_llm:
            if self.llm_provider == "anthropic":
                required.append(("ANTHROPIC_API_KEY", self.anthropic_api_key))
            elif self.llm_provider == "openai_compat":
                # The key is optional: local servers (LM Studio, llama.cpp) don't need one.
                required.append(("OPENAI_COMPAT_BASE_URL", self.openai_compat_base_url))
            required += [("MODEL_CHIEF", self.model_chief), ("MODEL_ANALYSTS", self.model_analysts)]
        if required:
            missing = [name for name, val in required if not val]
            if missing:
                raise ConfigError(
                    f"DESK_MODE={self.mode} with LLM_PROVIDER={self.llm_provider} requires these "
                    "environment variables / .env entries: " + ", ".join(missing)
                    + ". Copy .env.example to .env and fill them in, or use DESK_MODE=demo."
                )
        for name, val in vars(self).items():
            if isinstance(val, float) and not math.isfinite(val):
                raise ConfigError(f"{name.upper()} must be a finite number, got {val!r}.")
        if not self.watchlist:
            raise ConfigError("WATCHLIST is empty.")
        for s in self.watchlist:
            if "/" in s:
                raise ConfigError(
                    f"WATCHLIST symbol {s!r} contains '/': only US stocks/ETFs are supported for now."
                )
            if not s.replace(".", "").replace("-", "").isalnum():
                raise ConfigError(f"WATCHLIST symbol {s!r} is not a valid ticker.")
        for name in ("max_position_pct", "max_gross_exposure_pct", "daily_loss_limit_pct",
                     "min_confidence", "max_stop_distance_pct"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                raise ConfigError(f"{name.upper()} must be in (0, 1], got {v}.")
        if self.max_open_positions < 1 or self.max_candidates_per_cycle < 0:
            raise ConfigError("MAX_OPEN_POSITIONS must be >= 1 and MAX_CANDIDATES_PER_CYCLE >= 0.")
        if self.min_reward_risk <= 0 or self.daily_llm_budget_usd < 0:
            raise ConfigError("MIN_REWARD_RISK must be > 0 and DAILY_LLM_BUDGET_USD >= 0.")
        if self.openai_compat_price_in < 0 or self.openai_compat_price_out < 0:
            raise ConfigError("OPENAI_COMPAT_PRICE_IN/OUT must be >= 0.")
        for name in ("effort_chief", "effort_analysts"):
            if getattr(self, name) not in EFFORTS:
                raise ConfigError(f"{name.upper()} must be one of {', '.join(EFFORTS)}.")
        if self.cycle_seconds <= 0:
            raise ConfigError("Cycle interval must be > 0.")
        if self.index_buy_min_vix < 0:
            raise ConfigError("INDEX_BUY_MIN_VIX must be >= 0 (0 disables the VIX window).")
        if self.claude_code_cooldown_minutes <= 0:
            raise ConfigError("CLAUDE_CODE_COOLDOWN_MINUTES must be > 0.")


def _num(env: Mapping[str, str], key: str, default, cast):
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        val = cast(str(raw).strip())
    except ValueError:
        raise ConfigError(f"{key} must be a {cast.__name__}, got {raw!r}.") from None
    if isinstance(val, float) and not math.isfinite(val):
        raise ConfigError(f"{key} must be a finite number, got {raw!r}.")
    return val


def load_settings(env: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    """Build Settings from `env` (default: os.environ after loading .env) and validate."""
    if env is None:
        if use_dotenv:
            load_dotenv(PROJECT_DIR / ".env")
        env = os.environ

    def s(key: str, default: str = "") -> str:
        v = env.get(key)
        return default if v is None or str(v).strip() == "" else str(v).strip()

    mode = s("DESK_MODE", "demo").lower()
    # No LLM_PROVIDER line: keep existing API-key setups on the API; otherwise prefer Claude through the user's
    # Claude subscription (Claude Code CLI) when it is installed, else the free local model.
    if s("ANTHROPIC_API_KEY"):
        fallback = "anthropic"
    elif shutil.which(s("CLAUDE_CODE_CLI", "claude")):
        fallback = "claude_code"
    else:
        fallback = "ollama"
    provider = s("LLM_PROVIDER", fallback).lower()
    default_model = DEFAULT_MODELS.get(provider, "")
    wl = [x.strip().upper() for x in s("WATCHLIST", DEFAULT_WATCHLIST).split(",") if x.strip()]
    db = s("DESK_DB") or str(PROJECT_DIR / ("desk-demo.db" if mode == "demo" else "desk.db"))
    st = Settings(
        mode=mode,
        anthropic_api_key=s("ANTHROPIC_API_KEY"),
        alpaca_api_key=s("ALPACA_API_KEY"),
        alpaca_secret_key=s("ALPACA_SECRET_KEY"),
        watchlist=wl,
        cycle_minutes=_num(env, "CYCLE_MINUTES", 15, float),
        demo_cycle_seconds=_num(env, "DEMO_CYCLE_SECONDS", 20, float),
        max_candidates_per_cycle=_num(env, "MAX_CANDIDATES_PER_CYCLE", 3, int),
        demo_seed=_num(env, "DEMO_SEED", 42, int),
        max_position_pct=_num(env, "MAX_POSITION_PCT", 0.10, float),
        max_open_positions=_num(env, "MAX_OPEN_POSITIONS", 5, int),
        max_gross_exposure_pct=_num(env, "MAX_GROSS_EXPOSURE_PCT", 0.60, float),
        daily_loss_limit_pct=_num(env, "DAILY_LOSS_LIMIT_PCT", 0.02, float),
        min_confidence=_num(env, "MIN_CONFIDENCE", 0.6, float),
        max_stop_distance_pct=_num(env, "MAX_STOP_DISTANCE_PCT", 0.08, float),
        min_reward_risk=_num(env, "MIN_REWARD_RISK", 1.5, float),
        llm_provider=provider,
        ollama_url=s("OLLAMA_URL", "http://127.0.0.1:11434"),
        openai_compat_base_url=s("OPENAI_COMPAT_BASE_URL"),
        openai_compat_api_key=s("OPENAI_COMPAT_API_KEY"),
        openai_compat_price_in=_num(env, "OPENAI_COMPAT_PRICE_IN", 0.0, float),
        openai_compat_price_out=_num(env, "OPENAI_COMPAT_PRICE_OUT", 0.0, float),
        demo_agents=s("DEMO_AGENTS", "scripted").lower(),
        model_chief=s("MODEL_CHIEF", default_model),
        model_analysts=s("MODEL_ANALYSTS", default_model),
        effort_chief=s("EFFORT_CHIEF", "medium").lower(),
        effort_analysts=s("EFFORT_ANALYSTS", "low").lower(),
        daily_llm_budget_usd=_num(env, "DAILY_LLM_BUDGET_USD", 10.0, float),
        index_buy_min_vix=_num(env, "INDEX_BUY_MIN_VIX", 30.0, float),
        claude_code_cli=s("CLAUDE_CODE_CLI", "claude"),
        claude_code_cooldown_minutes=_num(env, "CLAUDE_CODE_COOLDOWN_MINUTES", 30.0, float),
        db_path=db,
        autostart=s("DESK_AUTOSTART", "1").lower() not in ("0", "false", "no"),
        port=_num(env, "DESK_PORT", 8000, int),
        phone_access=s("DESK_PHONE_ACCESS", "0").lower() in ("1", "true", "yes"),
        host="0.0.0.0" if s("DESK_PHONE_ACCESS", "0").lower() in ("1", "true", "yes") else "127.0.0.1",
    )
    st.validate()
    return st
