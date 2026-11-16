import pytest

from desk.indicators import atr, dist_from_high, pct_return, rsi, sma, volume_ratio


def test_sma():
    assert sma([1, 2, 3, 4, 5], 5) == 3
    assert sma([1, 2, 3, 4, 5], 2) == 4.5
    assert sma([1, 2], 5) is None


def test_rsi_balanced_and_wilder_smoothing():
    closes = [10.0]
    for i in range(14):  # 7 up, 7 down by 1 -> avg gain == avg loss -> RSI 50
        closes.append(closes[-1] + (1 if i % 2 == 0 else -1))
    assert rsi(closes, 14) == pytest.approx(50.0)
    closes.append(closes[-1] + 1)  # Wilder: gain=(0.5*13+1)/14, loss=0.5*13/14
    assert rsi(closes, 14) == pytest.approx(100 - 100 / (1 + 7.5 / 6.5))


def test_rsi_extremes_and_short_input():
    assert rsi(list(range(1, 30)), 14) == 100.0
    assert rsi(list(range(30, 1, -1)), 14) == pytest.approx(0.0)
    assert rsi([1, 2, 3], 14) is None


def test_atr_constant_range_and_gap():
    n = 20
    assert atr([11.0] * n, [9.0] * n, [10.0] * n, 14) == pytest.approx(2.0)
    highs, lows, closes = [11.0] * n, [9.0] * n, [10.0] * n
    highs[-1], lows[-1], closes[-1] = 16.0, 14.0, 15.0  # gap up: TR = 16-10 = 6
    assert atr(highs, lows, closes, 14) == pytest.approx((2.0 * 13 + 6.0) / 14)
    assert atr([1], [1], [1], 14) is None


def test_volume_ratio_and_returns():
    vols = [100.0] * 20 + [250.0]
    assert volume_ratio(vols, 20) == pytest.approx(2.5)
    assert volume_ratio([1.0] * 5, 20) is None
    closes = [100.0, 101, 102, 103, 104, 110]
    assert pct_return(closes, 5) == pytest.approx(0.10)
    assert pct_return(closes, 9) is None


def test_dist_from_high():
    highs = [10.0] * 19 + [20.0]
    closes = [10.0] * 19 + [18.0]
    assert dist_from_high(closes, highs, 20) == pytest.approx(-0.1)


def test_volume_ratio_prorates_partial_day():
    from desk.indicators import session_volume_fraction
    vols = [100.0] * 20 + [50.0]  # half a normal day's volume so far...
    assert volume_ratio(vols, 20, 0.5) == pytest.approx(1.0)  # ...at the point a normal day is half done = normal
    assert volume_ratio(vols, 20, None) is None
    assert session_volume_fraction(5) is None  # too early to judge
    assert session_volume_fraction(60) == pytest.approx(0.21)
    assert session_volume_fraction(75) == pytest.approx(0.245)
    assert session_volume_fraction(400) == 1.0
    fr = [session_volume_fraction(m) for m in range(15, 391, 5)]
    assert all(0 < a <= b <= 1 for a, b in zip(fr, fr[1:]))


def test_desk_uses_partial_fraction_only_for_todays_intraday_bar(settings, store):
    from datetime import datetime
    from desk.desk import ET, Desk
    from desk.models import Bar
    from desk.agents.scripted import ScriptedTeam
    from desk.broker.sim import SimBroker

    class IntradayBroker(SimBroker):
        intraday_last_bar = True

    now = datetime(2026, 11, 13, 13, 30, tzinfo=ET)  # 4h into the session
    bars = [Bar("2026-11-12", 1, 1, 1, 1, 100.0), Bar("2026-11-13", 1, 1, 1, 1, 56.0)]
    d = Desk(settings, store, IntradayBroker(settings.watchlist, 1), ScriptedTeam(settings))
    d._now_et = lambda: now
    assert d.last_bar_fraction(bars) == pytest.approx(0.56)
    assert d.last_bar_fraction(bars[:1]) == 1.0  # last bar is yesterday's full day
    sim = Desk(settings, store, SimBroker(settings.watchlist, 1), ScriptedTeam(settings))
    sim._now_et = lambda: now
    assert sim.last_bar_fraction(bars) == 1.0  # simulator bars are always complete days
