"""desk.history: base-rate context and VIX regime (offline, fake FRED)."""
import json

from desk.history import History

RATES = {"horizon_days": 10, "period": ["2016-03-21", "2026-10-29"],
         "overall": {"n": 100, "beat_spy": 0.52, "avg_excess": 0.007},
         "by_screen_score": [{"lo": -10, "hi": 0, "n": 5, "beat_spy": 0.54, "avg_excess": 0.0},
                             {"lo": 3, "hi": 4, "n": 9, "beat_spy": 0.542, "avg_excess": 0.0}],
         "by_symbol": {"AAPL": {"n": 50, "beat_spy": 0.55, "avg_excess": 0.01}},
         "spy_by_vix": [{"lo": 16, "hi": 20, "n": 700, "spy_avg_10d": 0.004, "spy_up_share": 0.62}]}
RESULTS = {"test_years": [2019, 2026], "chosen_model": None}
CSV = "observation_date,VIXCLS\n2026-11-11,17.5\n2026-11-12,.\n2026-11-13,18.25\n"


def _h(tmp_path, fetch=None):
    (tmp_path / "base_rates.json").write_text(json.dumps(RATES))
    (tmp_path / "results.json").write_text(json.dumps(RESULTS))
    calls = []

    def fake(url):
        calls.append(url)
        return CSV
    return History(tmp_path, fetch=fetch or fake), calls


def test_context_has_base_rates_bucket_symbol_and_regime(tmp_path):
    h, calls = _h(tmp_path)
    c = h.context("AAPL", 3.4)
    assert c["base_rate_beat_spy"] == 0.52 and "no signal tested" in c["study_verdict"]
    assert c["when_scout_score_similar"]["beat_spy"] == 0.542
    assert "survivorship" in c["this_symbol"]["caveat"]
    assert c["market_regime"]["vix"] == 18.25 and c["market_regime"]["as_of"] == "2026-11-13"
    assert c["market_regime"]["spy_next_10d_when_vix_similar"]["up_share"] == 0.62
    h.context("MSFT", None)
    assert len(calls) == 1  # VIX cached


def test_missing_study_or_fred_degrades_gracefully(tmp_path):
    assert History(tmp_path / "nope").context("AAPL", 1.0) is None

    def boom(url):
        raise OSError("offline")
    h, _ = _h(tmp_path, fetch=boom)
    c = h.context("AAPL", 1.0)
    assert c and "market_regime" not in c


def test_history_reaches_the_agents_fact_sheet(settings, store):
    from desk.agents.scripted import ScriptedTeam
    from desk.broker.sim import SimBroker
    from desk.desk import Desk

    class FakeHistory:
        def __init__(self):
            self.calls = []

        def context(self, symbol, score, index_buy_min_vix=25.0):
            self.calls.append((symbol, score))
            return {"base_rate_beat_spy": 0.52}
    fh = FakeHistory()
    broker = SimBroker(settings.watchlist, 1)
    d = Desk(settings, store, broker, ScriptedTeam(settings), history=fh)
    facts = d.fact_sheet("AAPL", broker.get_account(), broker.get_positions())
    assert facts["history"] == {"base_rate_beat_spy": 0.52} and fh.calls and fh.calls[0][0] == "AAPL"
    plain = Desk(settings, store, broker, ScriptedTeam(settings)).fact_sheet("AAPL", broker.get_account(), [])
    assert "history" not in plain


def test_failed_vix_fetch_is_retried_after_five_minutes(tmp_path):
    t = [0.0]
    n = [0]

    def flaky(url):
        n[0] += 1
        if n[0] == 1:
            raise OSError("offline")
        return CSV
    (tmp_path / "base_rates.json").write_text(json.dumps(RATES))
    h = History(tmp_path, fetch=flaky, clock=lambda: t[0])
    assert h.vix() == (None, "")
    t[0] += 60
    assert h.vix() == (None, "")       # still within the short failure cache
    t[0] += 300
    assert h.vix()[0] == 18.25         # retried
