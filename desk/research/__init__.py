"""Historical research for Agent Desk: download ~10 years of free, reliable data, test which signals really
predicted moves (walk-forward, out-of-sample, after costs) and export calibrated base rates for the live desk.

Run: `uv run --group research python -m desk.research <command>` (see __main__.py).

Sources (all free, all with a public track record):
- Alpaca market data (SIP consolidated tape, split/dividend adjusted daily bars, 2016+)
- Alpaca News (Benzinga newswire, timestamped headlines, 2015+)
- FRED (St. Louis Fed): market-priced series only (VIX, Treasury yields, credit spread, dollar). These are not
  revised after publication, so there is no "revised data" look-ahead.
- SEC EDGAR: 8-K item 2.02 ("Results of Operations") filings = official earnings-release timestamps.
- FinBERT (ProsusAI/finbert, run locally): headline sentiment. Its own accuracy is measured, not assumed.

The live LLM agents are NOT backtested on history: models trained on data through 2025 already "know" how past
events turned out, so such a test would be contaminated. History is used only with fixed rules and FinBERT.
"""
