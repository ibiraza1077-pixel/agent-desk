"""Day-trading backtest mechanics: cost accounting, stops, news alignment, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")
from desk.research import daytrade as dt  # noqa: E402

BARS = dt.BARS


def _day(closes, opens=None, highs=None, lows=None, prev=100.0, vol=None, dates=None):
    C = np.atleast_2d(np.asarray(closes, float))
    O = np.atleast_2d(np.asarray(opens, float)) if opens is not None else np.hstack([np.full((len(C), 1), prev), C[:, :-1]])
    H = np.atleast_2d(np.asarray(highs, float)) if highs is not None else np.maximum(O, C)
    L = np.atleast_2d(np.asarray(lows, float)) if lows is not None else np.minimum(O, C)
    V = np.ones_like(C) if vol is None else np.atleast_2d(np.asarray(vol, float))
    if dates is None:
        dates = np.datetime64("2024-01-02") + np.arange(len(C))
    return dt.Day("X", dates, O, H, L, C, V, np.full(len(C), prev))


def test_run_path_holds_all_day_pays_two_sides():
    c = np.linspace(100, 110, BARS)
    d = _day(c)
    pos = np.ones((1, BARS))
    pos[:, -1] = 0
    r = dt.run_path(d, pos, cost=0.001)
    gross = np.sum(c[1:] / c[:-1] - 1)
    assert r[0] == pytest.approx(gross - 0.002)


def test_flip_pays_twice():
    d = _day(np.full(BARS, 100.0))
    pos = np.zeros((1, BARS))
    pos[:, 10:20], pos[:, 20:40] = 1, -1
    assert dt.run_path(d, pos, cost=0.001)[0] == pytest.approx(-0.001 * (1 + 2 + 1))


def test_orb_long_stopped_at_range_low():
    c = np.full(BARS, 101.0)
    c[0] = 101.0                                  # first 5 minutes up: go long at bar 1 open
    o = np.full(BARS, 101.0); o[0] = 100.0
    h = np.full(BARS, 101.5); lo = np.full(BARS, 100.5); lo[0] = 99.0
    lo[5] = 98.0                                  # later bar trades through the range low (99)
    d = _day(c, o, h, lo)
    r = dt.strat_orb(d, 0.0, np.ones(1), minutes=5)
    assert r[0] == pytest.approx(99.0 / 101.0 - 1)


def test_orb_skips_when_entry_already_beyond_stop():
    o = np.full(BARS, 100.0); o[0] = 100.0; o[1] = 98.0
    c = np.full(BARS, 98.0); c[0] = 101.0
    h = np.full(BARS, 101.0); lo = np.full(BARS, 97.0); lo[0] = 99.0
    assert dt.strat_orb(_day(c, o, h, lo), 0.0, np.ones(1), minutes=5)[0] == 0.0


def test_intraday_momentum_uses_only_first_half_hour_for_the_sign():
    c = np.full(BARS, 100.0)
    c[5] = 101.0                                   # 10:00 price above previous close -> long last half hour
    c[71], c[77] = 100.0, 102.0
    r = dt.strat_intraday_momentum(_day(c), 0.0, np.ones(1))
    assert r[0] == pytest.approx(0.02)
    c2 = c.copy(); c2[40] = 50.0                   # a midday crash must not change the sign
    assert dt.strat_intraday_momentum(_day(c2), 0.0, np.ones(1))[0] == pytest.approx(0.02)


@pytest.mark.parametrize("fn", [dt.strat_vwap, dt.strat_ema_cross, dt.strat_rsi_revert])
def test_path_strategies_do_not_look_ahead(fn):
    rng = np.random.default_rng(0)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.002, (1, BARS)), axis=1)
    cut = 40
    alt = base.copy()
    alt[:, cut + 1:] *= 1.05                        # change only the future after bar `cut`
    r1 = dt.run_path  # noqa: F841 - keep reference for clarity
    pos = {}
    for name, closes in (("a", base), ("b", alt)):
        captured = {}
        orig = dt.run_path
        dt.run_path = lambda d, p, cost, _c=captured: (_c.setdefault("p", p.copy()), orig(d, p, cost))[1]
        try:
            fn(_day(closes, vol=np.ones((1, BARS))), 0.0, np.ones(1))
        finally:
            dt.run_path = orig
        pos[name] = captured["p"]
    # positions decided at or before bar `cut` must be identical
    assert np.array_equal(pos["a"][:, :cut + 1], pos["b"][:, :cut + 1])


def test_news_rolls_weekend_headlines_to_monday():
    dates = np.array(["2024-01-05", "2024-01-08"], dtype="datetime64[D]")   # Fri, Mon
    news = pd.DataFrame({"mean": [0.8, -0.4], "count": [1, 3]},
                        index=pd.to_datetime(["2024-01-06", "2024-01-08"]))   # Sat headline, Mon headlines
    m, c = dt.news_by_session(dates, news)
    assert np.isnan(m[0]) and c[0] == 0
    assert c[1] == 4 and m[1] == pytest.approx((0.8 - 1.2) / 4)
