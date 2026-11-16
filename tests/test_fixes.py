"""Regression tests for the review findings (alpaca close safety, kill/flatten, caching, pending orders, ...)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from desk import risk
from desk.agents.base import AgentUnavailable, LLMCaller
from desk.agents.scripted import ScriptedTeam
from desk.broker import alpaca_paper as ap
from desk.broker.sim import SimBroker, _Held
from desk.config import ConfigError, Settings, load_settings
from desk.desk import Desk, scout_pitch
from desk.models import Account, Bar, Decision, PendingOrder, Position


# ---------- fake alpaca trading client ----------
def order(oid, sym, *, stop=None, limit=None, side="sell", qty=10, filled=0):
    return SimpleNamespace(id=oid, symbol=sym, stop_price=stop, limit_price=limit, side=side, legs=None,
                           qty=qty, filled_qty=filled)


class FakeTrading:
    _base_url = ap.BaseURL.TRADING_PAPER

    def __init__(self, orders, positions=(), fail_cancel=(), fail_close=False, sticky=False):
        self.orders, self.positions = list(orders), list(positions)
        self.fail_cancel, self.fail_close, self.sticky = set(fail_cancel), fail_close, sticky
        self.closed, self.submitted = [], []

    def get_orders(self, f=None):
        syms = getattr(f, "symbols", None)
        return [o for o in self.orders if not syms or o.symbol in syms]

    def cancel_order_by_id(self, oid):
        if oid in self.fail_cancel:
            raise RuntimeError("cancel rejected")
        if not self.sticky:
            self.orders = [o for o in self.orders if o.id != oid]

    def close_position(self, sym):
        if self.fail_close:
            raise RuntimeError("close rejected")
        self.closed.append(sym)
        return SimpleNamespace(id="close-1")

    def get_all_positions(self):
        return self.positions

    def submit_order(self, req):
        self.submitted.append(req)
        self.orders.append(order("prot", req.symbol, stop=req.stop_price))
        return SimpleNamespace(id="prot")


def broker(trading, data=None):
    b = ap.AlpacaPaperBroker("k", "s", trading_client=trading, data_client=data or object(), news_client=object())
    b.cancel_timeout_s, b.cancel_poll_s = 0.05, 0.01
    b.alerts = []
    b.on_alert = b.alerts.append
    return b


POS = SimpleNamespace(symbol="AAPL", qty="10", current_price="100", avg_entry_price="98", unrealized_pl="20")
LEGS = lambda: [order("stop1", "AAPL", stop=95.0), order("tp1", "AAPL", limit=112.0)]  # noqa: E731


def test_close_waits_for_cancels_then_closes():
    t = FakeTrading(LEGS(), [POS])
    assert broker(t).close_position("AAPL") == "close-1"
    assert t.closed == ["AAPL"] and not t.orders


def test_close_aborts_if_cancel_fails_and_restores_stop():
    t = FakeTrading(LEGS(), [POS], fail_cancel={"tp1"})  # stop leg cancels fine, the target leg does not
    b = broker(t)
    with pytest.raises(RuntimeError):
        b.close_position("AAPL")
    assert t.closed == []  # never reached the close
    assert len(t.submitted) == 1 and t.submitted[0].stop_price == 95.0  # previous stop re-applied
    assert any("ALERT" in a for a in b.alerts)


def test_close_aborts_if_orders_never_clear():
    t = FakeTrading(LEGS(), [POS], sticky=True)  # cancel "succeeds" but orders stay open
    with pytest.raises(TimeoutError):
        broker(t).close_position("AAPL")
    assert t.closed == []


def test_failed_close_resubmits_protective_stop_with_fallback_price():
    t = FakeTrading([order("tp1", "AAPL", limit=112.0)], [POS], fail_close=True)  # no stop known
    b = broker(t)
    with pytest.raises(RuntimeError):
        b.close_position("AAPL")
    req = t.submitted[0]
    assert req.stop_price == 92.0 and req.qty == 10 and req.time_in_force == ap.TimeInForce.GTC  # 8% below last
    assert any("protective stop" in a for a in b.alerts)


def test_open_buy_orders_and_stale_price():
    t = FakeTrading([order("b1", "MSFT", limit=400.0, side=SimpleNamespace(value="buy"), qty=5, filled=2),
                     order("s1", "AAPL", stop=95.0)])
    bar = SimpleNamespace(timestamp=datetime(2026, 1, 2, 15, tzinfo=timezone.utc), open=1, high=2, low=1, close=1.5,
                          volume=10)

    class Data:
        def get_stock_latest_trade(self, req):
            raise RuntimeError("no quote")

        def get_stock_bars(self, req):
            return SimpleNamespace(data={"AAPL": [bar]})

    b = broker(t, Data())
    po = b.get_open_buy_orders()
    assert [(o.symbol, o.qty, o.price) for o in po] == [("MSFT", 3.0, 400.0)]
    assert b.get_latest_price("AAPL") == 1.5 and b.is_price_stale("AAPL")


# ---------- risk: pending orders and stale price ----------
S = Settings()
ACC = Account(100_000, 100_000, 100_000)


def buy(**kw):
    d = dict(symbol="AAPL", action="BUY", size_pct=0.05, stop_loss=95.0, take_profit=115.0, confidence=0.8)
    d.update(kw)
    return Decision(**d)


def test_pending_orders_count_for_symbol_positions_and_exposure():
    r = risk.evaluate(buy(), ACC, [], 0, 100.0, S, False, pending=[PendingOrder("AAPL", 10, 100)])
    assert not r.approved and any("pending" in x for x in r.reasons)
    pend = [PendingOrder(f"S{i}", 1, 10) for i in range(5)]
    r = risk.evaluate(buy(), ACC, [], 0, 100.0, S, False, pending=pend)
    assert not r.approved and any("max open positions" in x for x in r.reasons)
    r = risk.evaluate(buy(), ACC, [], 0, 100.0, S, False, pending=[PendingOrder("X", 590, 100)])  # 59k pending
    assert r.approved and r.qty == 9  # only ~$1k of the 60% gross cap left


def test_stale_price_refuses_buys_but_not_sells():
    r = risk.evaluate(buy(), ACC, [], 0, 100.0, S, False, price_stale=True)
    assert not r.approved and any("stale" in x for x in r.reasons)
    p = Position("AAPL", 5, 100, 100, 0)
    assert risk.evaluate(Decision(symbol="AAPL", action="SELL"), ACC, [p], 0, 100.0, S, False, price_stale=True).approved


# ---------- config ----------
@pytest.mark.parametrize("k,v", [("MAX_POSITION_PCT", "nan"), ("DAILY_LLM_BUDGET_USD", "inf"),
                                 ("MIN_REWARD_RISK", "-inf"), ("CYCLE_MINUTES", "nan"), ("DEMO_CYCLE_SECONDS", "inf")])
def test_non_finite_config_rejected(k, v):
    with pytest.raises(ConfigError) as e:
        load_settings({k: v}, use_dotenv=False)
    assert k in str(e.value)


def test_non_finite_set_directly_rejected():
    with pytest.raises(ConfigError):
        Settings(min_confidence=float("nan")).validate()


# ---------- hard LLM budget inside the caller ----------
def test_llm_caller_enforces_budget(settings, store):
    sent = []
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: sent.append(k)),
                             beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: sent.append(k))))
    store.add_usage("Chief", "claude-opus-5-5", 1, 1, settings.daily_llm_budget_usd)
    with pytest.raises(AgentUnavailable):
        LLMCaller(settings, store, client=client).call("Chief", "m", "low", "s", "u", {})
    assert sent == []


# ---------- sim: gaps, thread-safety lock ----------
def _gap_sim(open_px, held):
    b = SimBroker(["AAPL"], 3)
    sym = b.syms["AAPL"]

    def step(day):
        bar = Bar(day.isoformat(), open_px, max(open_px, 130), min(open_px, 85), open_px, 1e6)
        sym.price = open_px
        sym.bars.append(bar)
        return bar

    sym.step = step
    b.held["AAPL"] = held
    b.cash = 0.0
    return b


def test_sim_gap_up_through_target_fills_at_open():
    b = _gap_sim(120.0, _Held(10, 100.0, 90.0, 110.0))
    b.begin_cycle()
    assert "AAPL" not in b.held and b.cash == pytest.approx(1200.0)  # open 120, not the 110 target


def test_sim_gap_down_through_stop_fills_at_open():
    b = _gap_sim(80.0, _Held(10, 100.0, 90.0, 110.0))
    b.begin_cycle()
    assert "AAPL" not in b.held and b.cash == pytest.approx(800.0 * (1 - 0.0005))  # open 80, not the 90 stop


def test_sim_has_lock():
    assert hasattr(SimBroker(["AAPL"], 1)._lock, "acquire")


# ---------- desk: kill check before submit, rounding, cache ----------
class StubChief(ScriptedTeam):
    def __init__(self, s, on_chief=None):
        super().__init__(s)
        self.on_chief = on_chief

    def chief(self, facts, notes, limits):
        if self.on_chief:
            self.on_chief()
        px = facts["price"]
        return Decision(symbol=facts["symbol"], action="BUY", size_pct=0.05, confidence=0.9,
                        stop_loss=px * 0.95 + 0.004, take_profit=px * 1.12 + 0.003)


def test_kill_switch_rechecked_right_before_order(settings, store):
    team = StubChief(settings, on_chief=lambda: store.set_kill_switch(True))  # flips mid-debate
    d = Desk(settings, store, SimBroker(settings.watchlist, 1), team)
    d.run_cycle()
    assert not d.broker.get_positions()
    # the risk guard itself blocks (kill on) before we even reach the broker
    assert any("Rejected" in f["text"] and "kill switch" in f["text"] for f in store.get_feed(limit=200))


def test_kill_switch_abort_inside_order_lock(settings, store, monkeypatch):
    d = Desk(settings, store, SimBroker(settings.watchlist, 1), StubChief(settings))
    real = risk.evaluate

    def eval_then_flatten(*a, **k):  # kill switch lands after risk approved, before the order goes out
        res = real(*a, **k)
        store.set_kill_switch(True)
        return res

    monkeypatch.setattr("desk.desk.risk.evaluate", eval_then_flatten)
    d.run_cycle()
    assert not d.broker.get_positions()
    assert any("order aborted" in f["text"] for f in store.get_feed(limit=300))


def test_stop_and_target_rounded_before_risk_and_submit(settings, store):
    d = Desk(settings, store, SimBroker(settings.watchlist, 1), StubChief(settings))
    seen = []
    orig = d.broker.submit_bracket_buy
    d.broker.submit_bracket_buy = lambda s, q, st, tp: (seen.append((st, tp)), orig(s, q, st, tp))[1]
    d.run_cycle()
    assert seen
    for st, tp in seen:
        assert st == round(st, 2) and tp == round(tp, 2)
    for dec in store.get_decisions(50):
        if dec["action"] == "BUY":
            assert dec["stop"] == round(dec["stop"], 2) and dec["target"] == round(dec["target"], 2)


class CountingBroker(SimBroker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n = {"account": 0, "clock": 0, "positions": 0}

    def get_account(self):
        self.n["account"] += 1
        return super().get_account()

    def clock_info(self):
        self.n["clock"] += 1
        return super().clock_info()

    def get_positions(self):
        self.n["positions"] += 1
        return super().get_positions()


def test_read_paths_are_cached_and_use_one_clock_call(settings, store):
    b = CountingBroker(settings.watchlist, 1)
    d = Desk(settings, store, b, ScriptedTeam(settings))
    for _ in range(3):
        d.state()
        d.positions_view()
    assert b.n == {"account": 1, "clock": 1, "positions": 1}
    d.run_cycle()  # trading path uses fresh data and clears the cache
    before = b.n["account"]
    d.state()
    assert b.n["account"] == before + 1


# ---------- scripted wording ----------
def test_scripted_wording_matches_sign_of_numbers():
    t = ScriptedTeam(Settings())
    facts = {"symbol": "AAPL", "as_of": "d", "price": 100, "indicators": {"ret_20d": -0.072},
             "news": [{"headline": "AAPL beats earnings estimates and raises guidance"}]}
    for k in range(20):
        facts["as_of"] = f"d{k}"
        note = t.analyst(facts)
        assert "agrees" not in note.text and "follow-through" not in note.text
    pitch = scout_pitch("AAPL", {"sma20": 90, "sma50": 95, "ret_20d": -0.072}, 100, 1.0)
    assert "down 7.2%" in pitch and "-7.2%" not in pitch
    assert "up 7.2%" in scout_pitch("AAPL", {"ret_20d": 0.072}, 100, 1.0)
