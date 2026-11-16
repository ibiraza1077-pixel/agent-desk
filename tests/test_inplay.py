"""In-play ORB backtest mechanics: entry, stop, exit, sizing, costs, prior-day filters."""
import pandas as pd
import pytest

pytest.importorskip("pandas")
from desk.research import inplay as ip  # noqa: E402


def _bars(rows):
    df = pd.DataFrame(rows, columns=["hm", "open", "high", "low", "close"])
    return df


def test_long_held_to_close():
    b = _bars([("09:35", 10.0, 10.2, 9.99, 10.1), ("09:36", 10.1, 10.5, 10.05, 10.4), ("15:59", 10.4, 10.6, 10.3, 10.5)])
    t = ip.trade(b, atr=1.0, first_open=9.8, first_close=10.0)   # green first candle -> long; stop 9.90
    assert t["side"] == 1 and t["reason"] == "close" and t["exit"] == 10.5


def test_short_stopped_and_gap_through_fills_at_worse_open():
    b = _bars([("09:35", 10.0, 10.05, 9.9, 9.95), ("09:36", 10.3, 10.4, 10.2, 10.3)])
    t = ip.trade(b, atr=1.0, first_open=10.2, first_close=10.0)  # red candle -> short; stop 10.10
    assert t["side"] == -1 and t["reason"] == "stop" and t["exit"] == 10.3


def test_flat_first_candle_or_missing_0935_bar_skips():
    b = _bars([("09:35", 10.0, 10.1, 9.9, 10.0)])
    assert ip.trade(b, 1.0, 10.0, 10.0) is None
    assert ip.trade(_bars([("09:37", 10.0, 10.1, 9.9, 10.0)]), 1.0, 9.0, 10.0) is None


def test_portfolio_sizing_and_costs():
    t = pd.DataFrame([{"date": "2024-01-02", "side": 1, "entry": 10.0, "exit": 11.0, "stop_dist": 0.1},
                      {"date": "2024-01-02", "side": -1, "entry": 50.0, "exit": 50.0, "stop_dist": 5.0}])
    unlev = ip.portfolio(t, 0.0, levered=False)
    comm1 = ip.COMMISSION * (1 / 10 + 1 / 11)
    comm2 = ip.COMMISSION * (2 / 50)
    assert unlev.iloc[0] == pytest.approx((0.1 - comm1) / 20 + (0 - comm2) / 20)
    lev = ip.portfolio(t, 0.0, levered=True)
    w1 = min(0.01 * 10 / 0.1, 4 / 20)       # 1% risk wants 100% of equity: capped at 4x/20 = 20%
    w2 = min(0.01 * 50 / 5.0, 4 / 20)       # 10%
    assert lev.iloc[0] == pytest.approx(w1 * (0.1 - comm1) + w2 * (0 - comm2))


def test_filters_use_only_prior_days_and_skip_splits():
    n = 30
    df = pd.DataFrame({"date": [f"2024-01-{i + 1:02d}" for i in range(n)], "high": [11.0] * n, "low": [9.0] * n,
                       "close": [10.0] * n, "volume": [1e6] * n})
    df.loc[20, ["high", "low", "close"]] = [100.0, 90.0, 95.0]     # day 20 spikes
    f = ip._filters(df).set_index("date")
    assert f.loc["2024-01-21", "prev_close"] == 10.0              # day 20's own close isn't known yet on day 20
    assert "2024-01-22" not in f.index                            # the >60% jump marks the next 15 days unusable
