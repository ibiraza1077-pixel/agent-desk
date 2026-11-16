"""Look-ahead and correctness tests for the research backtest (synthetic data, offline)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from desk.research import backtest as bt  # noqa: E402
from desk.research import db as rdb  # noqa: E402
from desk.research import features as ft  # noqa: E402


def _fill(con, n_days=900, symbols=("SPY", "AAA", "BBB", "CCC", "DDD", "EEE"), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=n_days)
    for sym in symbols:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        op = px * np.exp(rng.normal(0, 0.003, n_days))
        rows = [(sym, d.date().isoformat(), o, max(o, c) * 1.01, min(o, c) * 0.99, c, 1e6 * (1 + rng.random()), c)
                for d, o, c in zip(days, op, px)]
        con.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return days


@pytest.fixture
def con(tmp_path):
    c = rdb.connect(tmp_path / "h.db")
    yield c
    c.close()


def test_decision_day_cutoff_is_the_close():
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-01-31"))
    ts = pd.Series(["2024-01-05T20:59:00Z",   # 15:59 ET Friday -> Friday
                    "2024-01-05T21:00:00Z",   # 16:00 ET Friday -> Monday
                    "2024-01-06T15:00:00Z",   # Saturday -> Monday
                    "2024-01-08T13:00:00Z"])  # 08:00 ET Monday -> Monday
    got = ft.to_decision_day(ts, cal).dt.date.astype(str).tolist()
    assert got == ["2024-01-05", "2024-01-08", "2024-01-08", "2024-01-08"]


def test_forward_label_is_next_open_to_open_after_horizon(con):
    days = _fill(con, n_days=200)
    f = ft.build(con)
    bars = ft.load_bars(con)
    a = bars[bars.symbol == "AAA"].reset_index(drop=True)
    row = f[(f.symbol == "AAA") & (f.date == days[100])].iloc[0]
    expect = a.open[100 + 1 + ft.HORIZON] / a.open[101] - 1
    assert row.fwd_ret == pytest.approx(expect)
    # the features for day 100 must not change if every bar after day 100 is altered
    con.execute("UPDATE bars SET close = close * 3, open = open * 3, high = high * 3, low = low * 3 "
                "WHERE date > ?", (days[100].date().isoformat(),))
    con.commit()
    g = ft.build(con)
    row2 = g[(g.symbol == "AAA") & (g.date == days[100])].iloc[0]
    for c in ft.PRICE_FEATURES:
        if c != "rel_20d" and c != "rel_60d":
            assert row2[c] == pytest.approx(row[c], nan_ok=True), c


def test_macro_is_lagged_one_day(con):
    days = _fill(con, n_days=120)
    con.executemany("INSERT INTO macro VALUES ('VIXCLS', ?, ?)",
                    [(d.date().isoformat(), float(i)) for i, d in enumerate(days)])
    con.commit()
    m = ft.macro_features(con, pd.DatetimeIndex(days))
    assert m.vix.iloc[50] == 49.0  # day 50 sees day 49's value


def test_news_after_close_counts_next_day(con):
    days = _fill(con, n_days=120)
    d = days[80]
    con.executemany("INSERT INTO news VALUES (?,?,?,?,?,?)", [
        (1, f"{d.date()}T15:00:00Z", "S", "", "before close", ""),   # 10:00 ET
        (2, f"{d.date()}T22:00:00Z", "S", "", "after close", "")])   # 17:00 ET
    con.executemany("INSERT INTO news_symbol VALUES (?, 'AAA')", [(1,), (2,)])
    con.executemany("INSERT INTO news_sentiment VALUES (?, 'ProsusAI/finbert', 0.9, 0.05, 0.05, 0.85)", [(1,), (2,)])
    con.commit()
    f = ft.build(con)
    a = f[f.symbol == "AAA"].set_index("date")
    assert a.loc[d, "news_1d"] == 1 and a.loc[days[81], "news_1d"] == 1


def test_walk_forward_has_no_skill_on_random_data_and_respects_embargo(con, monkeypatch):
    _fill(con, n_days=1300, seed=3)
    f = ft.build(con)
    seen = {}
    real_fit = bt.make_model

    def spy_model(kind):
        m = real_fit(kind)
        orig = m.fit

        def fit(X, y):
            seen.setdefault("max_train", []).append(X.index.max())  # row labels of prepare(f)
            return orig(X, y)
        m.fit = fit
        return m
    monkeypatch.setattr(bt, "make_model", spy_model)
    specs = {"base_rate": ("const", []), "price": ("logit", ft.PRICE_FEATURES)}
    pred = bt.walk_forward(f, specs)
    assert pred.year.min() == f.date.dt.year.min() + bt.MIN_TRAIN_YEARS
    d = bt.prepare(f)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    test_years = sorted(pred.year.unique())
    assert len(seen["max_train"]) == len(test_years)  # one fit per test year
    for yr, last_idx in zip(test_years, seen["max_train"]):
        train_end = d.loc[d.index <= last_idx].date.max()  # prepare() keeps a date-sorted index
        first_test = dates[dates.searchsorted(pd.Timestamp(yr, 1, 1))]
        gap = dates.searchsorted(first_test) - dates.searchsorted(train_end)
        assert gap >= bt.EMBARGO, (yr, train_end, first_test)
    s = bt.score_models(pred)
    auc = s[(s.year == "all") & (s.model == "price")].auc.iloc[0]
    assert 0.45 < auc < 0.55  # random walk: no skill should appear


def test_simulation_matches_spy_when_holding_nothing(con):
    _fill(con, n_days=1300, seed=4)
    f = ft.build(con)
    pred = bt.walk_forward(f, {"base_rate": ("const", [])})
    r = bt.simulate(pred, "base_rate", threshold=1.01)  # nothing qualifies -> all cash
    assert r.invested_periods == 0 and r.cagr == pytest.approx(0.0)


def test_unscored_news_does_not_crash(con):
    days = _fill(con, n_days=120)
    con.execute("INSERT INTO news VALUES (1, ?, 'S', '', 'h', '')", (f"{days[60].date()}T15:00:00Z",))
    con.execute("INSERT INTO news_symbol VALUES (1, 'AAA')")
    con.commit()
    a = ft.build(con).set_index(["symbol", "date"]).loc["AAA"]
    assert a.loc[days[60], "news_1d"] == 1 and a.loc[days[60], "sent_1d"] == 0.0
