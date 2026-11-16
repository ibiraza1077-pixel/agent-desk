# Agent Desk

A multi-agent AI "trading desk" that **paper-trades** US stocks and ETFs. Several Claude-powered agents debate each
trade candidate, a hard-coded Risk Guard enforces limits the AI cannot override, and a live web dashboard shows the
agents' chat, decisions, positions and an equity curve versus SPY.

> **Paper trading only. Live (real-money) trading is deliberately unsupported.** There is no setting, flag or code
> path that points the broker at a live endpoint; the Alpaca broker hard-codes `paper=True`, asserts the paper base
> URL at construction, and raises `LiveTradingRefused` otherwise. This is not financial advice. LLM agents have no
> proven trading edge; this project exists to find out, with fake money, whether they have one.

## Two modes

| Mode | Market | Agents | Needs |
|------|--------|--------|-------|
| `demo` (default) | Synthetic seeded random-walk market, $100k simulated account | Scripted rule-based agents, or real AI agents with `DEMO_AGENTS=llm` | Nothing (scripted); the Ollama app for free local AI agents |
| `paper` | Alpaca paper account, real (IEX, free-tier) market data | AI agents from `LLM_PROVIDER`: local Ollama (free, default), a free hosted tier, or Claude | Free Alpaca paper keys (+ a key only for hosted providers) |

## The agents

| Agent | Kind | Job |
|-------|------|-----|
| **Scout** | code | Screens the watchlist (trend + momentum + volume score) and writes a one-line pitch for the top candidates. No LLM call. |
| **Analyst** | LLM | Reads news/catalyst context: stance, confidence, short note. |
| **Charts** | LLM | Reads technicals (SMA 20/50, RSI, ATR, volume, returns): stance, confidence, note, suggested stop and target. |
| **Skeptic** | LLM | Sees Analyst + Charts and argues the strongest case against entering; names the key risk. |
| **Chief** | LLM | Hears everyone, then decides BUY / WAIT / SKIP with size, stop, target, confidence, rationale. Also reviews open positions (HOLD / SELL) alone. |
| **Risk** | code | `RiskGuard`: pure deterministic code and the only path to an order. Posts "Approved 37 shares..." or "Rejected: ...". |

Each LLM agent only sees a compact JSON fact sheet (indicators, last price, ten recent closes, headlines, portfolio
context), never raw bar dumps.

## Architecture

```
                  every cycle (15 min in paper mode, 20 s in demo)
  watchlist ──► screen (code) ──► top N candidates
                                        │
       held positions ──► Chief review (HOLD/SELL)
                                        ▼
        fact sheet ─► Scout ─► Analyst ┐
                              Charts  ─┼─► Skeptic ─► Chief ─► Decision
                                       ┘                         │
                                                                 ▼
                                   RiskGuard (pure code; clamps, caps, vetoes)
                                                                 │ approved
                                                                 ▼
                       Broker.submit_bracket_buy (stop + target attached)
                         SimBroker (demo)  |  AlpacaPaperBroker (paper)
                                                                 │
   SQLite (feed, decisions, equity, llm_usage, state) ◄──────────┘
        ▲
        └── FastAPI  /api/*  ──►  dashboard (single static HTML page, polls every 3 s)
```

## Quickstart: demo mode (zero keys)

```bash
uv sync
uv run python -m desk        # or: uv run desk
# open http://127.0.0.1:8000
```

A demo cycle runs every 20 seconds and advances the synthetic market by one trading day. The demo database
(`desk-demo.db`) is wiped on every start because the simulated account lives in memory. The server binds to
`127.0.0.1` only.

Run the tests: `uv run pytest -q` (no network needed).

## Use your Claude plan (recommended)

With Claude Code installed and logged in to a Claude Pro/Max plan, set `LLM_PROVIDER=claude_code`. Every agent
call runs `claude -p` with Opus 5.5 in an isolated, tool-less, one-shot session, so there's no API key and no
per-token bill. Calls do count against your plan's usage limits: one trading day is roughly 12 calls per cycle
× ~24 cycles. The dashboard's **AI usage** tile shows calls made today. If Claude replies with a usage limit,
the agents pause (default 30 min) and the desk keeps running, with no new buys in the meantime. Check your
plan's terms on automated use.

## Run it for free

Everything can run at zero cost:

| Piece | Free option |
|---|---|
| Market + broker | Alpaca **paper** account (free) – or demo mode's simulated market (no account at all) |
| Market data | Alpaca's free IEX feed |
| The AI agents | A local model through [Ollama](https://ollama.com) (default) – or a free hosted tier |

**Local model (default, `LLM_PROVIDER=ollama`)**: install the Ollama app, then `ollama pull qwen3.5:9b`
(about 7.6 GB). No key, no cost, nothing leaves your computer. On a 16 GB Apple Silicon Mac each agent answer takes
roughly 10–60 s, so a debate takes a few minutes; the 15-minute cycle leaves room for that. Any other pulled model works
via `MODEL_CHIEF` / `MODEL_ANALYSTS` (e.g. `gemma4:12b`). The desk checks at startup that Ollama is running and the
models are downloaded, and tells you the exact `ollama pull` command if not.

**Free hosted tier (`LLM_PROVIDER=openai_compat`)**: any OpenAI-compatible API. Set `OPENAI_COMPAT_BASE_URL`,
`OPENAI_COMPAT_API_KEY` and `MODEL_CHIEF`/`MODEL_ANALYSTS` (examples for Groq, Google Gemini and OpenRouter are in
`.env.example`). The key is optional for local servers such as LM Studio. If the endpoint is **paid**, set
`OPENAI_COMPAT_PRICE_IN`/`OUT` ($ per 1M tokens) so spend is metered and the daily budget cap applies; at 0 the desk
treats it as free and never budget-gates it. Free tiers have rate limits (a rate-limited candidate is skipped, never traded) and some providers
may use free-tier prompts to improve their models – read their terms.

**Claude (`LLM_PROVIDER=anthropic`)**: paid, strongest reasoning. See "Rough LLM cost" below.

Small local models reason noticeably worse than frontier models: expect shallower debates and more generic notes.
That's acceptable because RiskGuard, not the model, decides what can actually be traded – but judge the strategy on
its paper results, and consider a stronger model for the Chief if the decisions look poor.

**Watch real AI agents without any account**: in `.env` set `DESK_MODE=demo` and `DEMO_AGENTS=llm`. The agents then
debate the simulated market using your chosen `LLM_PROVIDER`.

## Paper mode setup

1. Create a free [Alpaca](https://alpaca.markets) account and generate **paper trading** API keys (not live keys).
2. Pick the AI (see above). For the default local model, have the Ollama app running.
3. `cp .env.example .env`, then edit `.env` yourself: set `DESK_MODE=paper`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
   (plus the provider's key if you chose a hosted one). `.env` is git-ignored; keys are never logged.
4. `uv run python -m desk`. If a key is missing the desk exits immediately with a message naming it.

In paper mode the scheduler runs a cycle every `CYCLE_MINUTES` while the US market is open. When it is closed the desk
posts one "market closed, desk sleeping" message per closure and records an equity snapshot at most hourly. Paper state
persists in `desk.db`; the desk refuses to open a database last used in a different mode.

All settings, with defaults, are documented in `.env.example`.

## Safety design

- **RiskGuard** (`desk/risk.py`) is pure code: no I/O, no clock. Rules: kill switch rejects all buys (sells always
  allowed); daily loss limit halts new buys for the rest of the day (persisted with the date); minimum confidence;
  BUY needs a stop below price and within `MAX_STOP_DISTANCE_PCT`, a target above price, and reward:risk at least
  `MIN_REWARD_RISK`; size is `min(requested, MAX_POSITION_PCT) x equity`, floored to whole shares and capped by
  remaining gross exposure and cash (with 0.2% headroom for slippage); quantity < 1 is rejected; no pyramiding;
  `MAX_OPEN_POSITIONS`. Every number the LLM returns is clamped, never trusted.
- **LLM budget cap**: estimated cost per call is recorded from `response.usage`. Once today's spend reaches
  `DAILY_LLM_BUDGET_USD` the desk skips all debates (with a feed message) until the next day. Bracket stop/target
  orders sit at the broker, so open positions stay protected.
- **Kill switch** (`POST /api/kill`, persisted) blocks new buys; `POST /api/resume` clears it. `POST /api/flatten`
  closes every position (the dashboard asks for confirmation first).
- Long only, whole shares, no shorting/options/margin (buying power is treated as cash).

## Rough LLM cost (Claude provider only)

Local and free-tier providers cost $0.

Per candidate there are about 4 Opus calls (Analyst, Charts, Skeptic, Chief); each reviewed open position is 1 more
(Chief). Scout and Risk cost nothing.

```
cost/cycle  ≈ candidates x (4 calls) + open_positions x (1 call)
daily cost  ≈ cycles/day x cost/cycle           (about 26 cycles in a 6.5 h session at 15 min)
```

Assumption (not measured, so treat as order-of-magnitude): about 8k input and 5k output tokens per candidate
including thinking at the default efforts, i.e. roughly 8k x $4/M + 5k x $20/M ≈ **$0.13 per candidate**. At defaults
(3 candidates, up to 5 positions reviewed at ~$0.04 each) that is roughly $0.4 to $0.6 per cycle, so around
$10 to $15 per full market day, which is why `DAILY_LLM_BUDGET_USD` defaults to $10 and will cut the desk off. To spend
less: lower `MAX_CANDIDATES_PER_CYCLE`, lengthen `CYCLE_MINUTES`, or set `MODEL_ANALYSTS` to a cheaper model
(prices in `desk/config.py`; unknown models are priced as Opus). Check actual spend on the dashboard header and in the
`llm_usage` table after the first day and adjust.

## Evaluation plan

1. Run in paper mode for **4 to 8 weeks** without changing the prompts or limits mid-run.
2. Watch the dashboard's equity curve against the SPY buy-and-hold line (both normalized to starting equity).
3. From the `decisions` table / closed trades, check win rate, average win versus average loss, and maximum drawdown,
   and include the LLM spend as a cost.
4. Only if the desk clearly beats SPY *after costs* over that period is it even worth discussing anything further, and
   any such step is entirely your own decision. A few weeks of paper results can easily be luck.

## Disclaimers

- Not financial advice. Nothing here recommends any security or strategy.
- LLM agents have no proven edge in trading. Backtests and short paper runs do not predict future returns.
- Paper fills are optimistic (no real market impact or queue position); real trading would be worse.
- Live trading is intentionally not supported in this version.

## Possible future work

Crypto (symbols like `BTC/USD` are currently rejected), shorting, richer position-sizing, trade analytics.

## API

`GET /api/state`, `/api/feed?after_id=N&limit=100`, `/api/decisions?limit=50`, `/api/positions`, `/api/equity`;
`POST /api/kill`, `/api/resume`, `/api/flatten`, `/api/run-cycle`.
