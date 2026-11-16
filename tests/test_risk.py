import pytest

from desk import risk
from desk.config import Settings
from desk.models import Account, Decision, Position

S = Settings()
PRICE = 100.0


def buy(**kw) -> Decision:
    base = dict(symbol="AAPL", action="BUY", size_pct=0.05, stop_loss=95.0, take_profit=115.0, confidence=0.8)
    base.update(kw)
    return Decision(**base)


def acct(equity=100_000.0, cash=100_000.0) -> Account:
    return Account(equity=equity, cash=cash, last_equity=equity)


def pos(sym="MSFT", qty=10, price=100.0) -> Position:
    return Position(sym, qty, price, price, 0.0)


def ev(d=None, account=None, positions=(), day_pnl=0.0, price=PRICE, kill=False, halted=False, s=S):
    return risk.evaluate(d or buy(), account or acct(), list(positions), day_pnl, price, s, kill, halted)


def test_happy_path_sizes_whole_shares():
    r = ev()
    assert r.approved and r.qty == 49  # 5% of 100k = 5000 -> 49 after 0.2% slippage headroom
    assert r.qty * PRICE <= 0.05 * 100_000


def test_kill_switch_blocks_buys_but_not_sells():
    r = ev(kill=True)
    assert not r.approved and any("kill switch" in x for x in r.reasons)
    sell = Decision(symbol="MSFT", action="SELL", confidence=0.5)
    r = ev(sell, positions=[pos("MSFT", 10)], kill=True)
    assert r.approved and r.qty == 10


def test_sell_without_position_rejected():
    assert not ev(Decision(symbol="MSFT", action="SELL")).approved


def test_daily_loss_limit_halts_and_flags():
    r = ev(day_pnl=-2_100.0, account=acct(equity=97_900.0, cash=97_900.0))
    assert not r.approved and r.halt and any("daily loss" in x for x in r.reasons)
    assert not ev(day_pnl=-1_000.0).halt
    r = ev(halted=True)  # already halted earlier today
    assert not r.approved and any("halted" in x for x in r.reasons)


def test_low_confidence():
    r = ev(buy(confidence=0.59))
    assert not r.approved and any("confidence" in x for x in r.reasons)
    assert ev(buy(confidence=0.6)).approved


@pytest.mark.parametrize("stop", [None, 0, 100.0, 105.0, 90.0])
def test_bad_stop(stop):
    # None/0 missing, >= price wrong side, 90 is 10% away (> 8%)
    r = ev(buy(stop_loss=stop, take_profit=130.0))
    assert not r.approved and any("stop" in x for x in r.reasons)


def test_bad_target_and_reward_risk():
    assert any("take-profit" in x for x in ev(buy(take_profit=99.0)).reasons)
    assert any("take-profit" in x for x in ev(buy(take_profit=None)).reasons)
    r = ev(buy(stop_loss=95.0, take_profit=106.0))  # rr 1.2 < 1.5
    assert not r.approved and any("reward:risk" in x for x in r.reasons)
    assert ev(buy(stop_loss=95.0, take_profit=107.5)).approved  # rr exactly 1.5


def test_position_size_capped_at_max_pct():
    r = ev(buy(size_pct=0.9))  # LLM asks for 90%
    assert r.approved and r.qty * PRICE <= S.max_position_pct * 100_000
    assert r.qty == 99


def test_out_of_range_numbers_are_clamped():
    assert ev(buy(confidence=5.0, size_pct=-1.0)).qty == 0  # negative size -> 0 shares -> rejected
    assert ev(buy(confidence=float("nan"))).approved is False
    assert ev(buy(size_pct=float("nan"))).approved is False


def test_gross_exposure_cap():
    held = [pos("A", 250, 100), pos("B", 340, 100)]  # 59k of 60k allowed
    r = ev(positions=held, account=acct(cash=41_000))
    assert r.approved and r.qty == 9  # only ~$1000 left
    full = [pos("A", 300, 100), pos("B", 300, 100)]
    r = ev(positions=full, account=acct(cash=40_000))
    assert not r.approved and any("exposure" in x for x in r.reasons)


def test_cash_cap():
    r = ev(account=acct(equity=100_000, cash=1_500))
    assert r.approved and r.qty == 14
    assert not ev(account=acct(equity=100_000, cash=50)).approved


def test_max_open_positions():
    held = [pos(f"S{i}", 1, 10) for i in range(5)]
    r = ev(positions=held)
    assert not r.approved and any("max open positions" in x for x in r.reasons)
    assert ev(positions=held[:4]).approved


def test_already_held():
    r = ev(positions=[pos("AAPL", 5)])
    assert not r.approved and any("already holding" in x for x in r.reasons)


def test_qty_below_one():
    r = ev(price=2000.0, d=buy(stop_loss=1900.0, take_profit=2300.0), account=acct(equity=10_000, cash=10_000))
    assert not r.approved and r.qty == 0 and any("quantity would be 0" in x for x in r.reasons)


def test_non_order_actions_not_approved():
    assert not ev(Decision(symbol="X", action="WAIT")).approved


def _spy_buy():
    from desk.models import Decision
    return Decision(symbol="SPY", action="BUY", size_pct=0.1, stop_loss=480.0, take_profit=540.0, confidence=0.7,
                    rationale="")


def test_spy_buys_need_the_vix_window_in_real_market_modes(settings):
    import dataclasses
    from desk import risk
    from desk.models import Account
    paper = dataclasses.replace(settings, mode="paper", index_buy_min_vix=25.0)
    acct = Account(equity=100000.0, cash=100000.0, last_equity=100000.0)
    closed = risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, paper, False, vix=18.7)
    assert not closed.approved and any("VIX >= 25" in r for r in closed.reasons)
    unknown = risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, paper, False, vix=None)
    assert not unknown.approved and any("unavailable" in r for r in unknown.reasons)
    opened = risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, paper, False, vix=27.0)
    assert opened.approved and opened.qty > 0
    default = dataclasses.replace(settings, mode="paper")  # shipped default threshold is 30
    assert not risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, default, False, vix=27.0).approved
    assert risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, default, False, vix=31.0).approved
    # other symbols and the simulator are unaffected
    demo = risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, settings, False, vix=None)
    assert demo.approved
    off = dataclasses.replace(paper, index_buy_min_vix=0.0)
    assert risk.evaluate(_spy_buy(), acct, [], 0.0, 500.0, off, False, vix=None).approved
