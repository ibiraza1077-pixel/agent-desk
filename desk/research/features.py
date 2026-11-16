"""Point-in-time feature table: one row per (symbol, trading day D).

Timing rules (the whole point of this module is that nothing here can see the future):
- A row describes the moment of the close on day D (16:00 New York). The trade it could trigger is entered at the
  NEXT day's open and exited at the open H trading days later. So the label never overlaps the data used.
- Price features use bars up to and including D (the same `compute_indicators` / `screen_score` the live desk
  uses, so a base rate learned here means the same thing as the number the desk sees).
- News and earnings count only if published strictly before D's 16:00 close. Anything later is attributed to the
  next trading day.
- FRED macro values are lagged one trading day (use D-1's value at D) to be safe about publication timing.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from ..desk import screen_score
from ..indicators import compute_indicators
from ..models import Bar

ET = "America/New_York"
HORIZON = 10          # trading days held
BENCH = "SPY"

PRICE_FEATURES = ["sma20_gap", "sma50_gap", "sma20_over_50", "rsi14", "atr_pct", "volume_ratio20", "ret_1d",
                  "ret_5d", "ret_20d", "ret_60d", "dist_from_20d_high", "vol_20d", "rel_20d", "rel_60d",
                  "screen_score"]
MACRO_FEATURES = ["vix", "vix_chg5", "t10y2y", "dgs10_chg20", "credit_spread", "credit_chg20", "usd_chg20", "oil_chg20"]
NEWS_FEATURES = ["news_1d", "news_5d", "news_surge", "sent_1d", "sent_5d", "sent_pos_share_5d"]
EARN_FEATURES = ["days_since_earn", "earn_react", "earn_recent"]
ALL_GROUPS = {"price": PRICE_FEATURES, "macro": MACRO_FEATURES, "news": NEWS_FEATURES, "earnings": EARN_FEATURES}


def load_bars(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT symbol, date, open, high, low, close, volume FROM bars ORDER BY symbol, date", con)
    return df


def _calendar(bars: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(bars.loc[bars.symbol == BENCH, "date"]).unique()))


def to_decision_day(ts_utc: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """Map UTC publish times to the first trading day D whose 16:00 ET close is strictly after the publish time."""
    et = pd.to_datetime(ts_utc, utc=True).dt.tz_convert(ET)
    day = et.dt.tz_localize(None).dt.normalize()
    after_close = (et.dt.hour >= 16)
    day = day.where(~after_close, day + pd.Timedelta(days=1))
    idx = cal.searchsorted(day.values, side="left")  # first trading day >= that calendar day
    out = pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")
    ok = idx < len(cal)
    out[ok] = cal[idx[ok]]
    return out


def price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Desk indicators per (symbol, D), plus a few extra price features, plus forward labels."""
    rows = []
    bench = bars[bars.symbol == BENCH].set_index("date")
    for sym, g in bars.groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        # compute_indicators rounds SMAs to cents. Split-adjusted history can be under $1 (NVDA 2016), where that
        # rounding distorts the ratios, so prices are rescaled first; every feature used here is scale-free.
        # The factor uses the FIRST bar only (always in the past), never the series min, which would peek ahead.
        k = max(1.0, 1000.0 / max(1e-9, float(g.close.iloc[0])))
        blist = [Bar(r.date, r.open * k, r.high * k, r.low * k, r.close * k, r.volume) for r in g.itertuples()]
        for i in range(55, len(g)):  # the desk skips symbols with < 55 bars
            window = blist[max(0, i - 119): i + 1]  # the desk looks at 120 daily bars
            ind = compute_indicators(window)
            close = g.close[i] * k
            rows.append({
                "symbol": sym, "date": g.date[i], "close": g.close[i],
                "sma20_gap": close / ind["sma20"] - 1 if ind["sma20"] else np.nan,
                "sma50_gap": close / ind["sma50"] - 1 if ind["sma50"] else np.nan,
                "sma20_over_50": ind["sma20"] / ind["sma50"] - 1 if ind["sma20"] and ind["sma50"] else np.nan,
                "rsi14": ind["rsi14"], "atr_pct": ind["atr14"] / close if ind["atr14"] else np.nan,
                "volume_ratio20": ind["volume_ratio20"], "ret_5d": ind["ret_5d"], "ret_20d": ind["ret_20d"],
                "dist_from_20d_high": ind["dist_from_20d_high"],
                "screen_score": screen_score(ind, close),
            })
    f = pd.DataFrame(rows)
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)

    b = bars.sort_values(["symbol", "date"]).copy()
    b["ret_1d"] = b.groupby("symbol").close.pct_change()
    b["ret_60d"] = b.groupby("symbol").close.pct_change(60)
    b["vol_20d"] = b.groupby("symbol").ret_1d.transform(lambda s: s.rolling(20).std())
    # Forward label: enter next open, exit open HORIZON days after entry.
    nxt_open = b.groupby("symbol").open.shift(-1)
    exit_open = b.groupby("symbol").open.shift(-1 - HORIZON)
    b["fwd_ret"] = exit_open / nxt_open - 1
    f = f.merge(b[["symbol", "date", "ret_1d", "ret_60d", "vol_20d", "fwd_ret"]], on=["symbol", "date"], how="left")

    bench = f[f.symbol == BENCH].set_index("date")[["ret_20d", "ret_60d", "fwd_ret"]].add_prefix("spy_")
    f = f.merge(bench, left_on="date", right_index=True, how="left")
    f["rel_20d"] = f.ret_20d - f.spy_ret_20d
    f["rel_60d"] = f.ret_60d - f.spy_ret_60d
    f["excess_fwd"] = f.fwd_ret - f.spy_fwd_ret
    return f.drop(columns=["spy_ret_20d", "spy_ret_60d"])


def macro_features(con: sqlite3.Connection, cal: pd.DatetimeIndex) -> pd.DataFrame:
    m = pd.read_sql("SELECT series, date, value FROM macro", con)
    if m.empty:
        return pd.DataFrame(index=cal)
    w = m.pivot(index="date", columns="series", values="value")
    w.index = pd.to_datetime(w.index)
    w = w.reindex(w.index.union(cal)).sort_index().ffill().reindex(cal)
    # Publication lags (trading days): most series are out by the next morning; the broad dollar index is weekly
    # and WTI oil is often a few days late, so they are lagged more to only use values already published.
    lags = {"DTWEXBGS": 6, "DCOILWTICO": 3}
    w = pd.DataFrame({c: w[c].shift(lags.get(c, 1)) for c in w.columns}, index=cal)
    out = pd.DataFrame(index=cal)
    col = lambda s: w[s] if s in w else pd.Series(np.nan, index=cal)  # noqa: E731
    out["vix"] = col("VIXCLS")
    out["vix_chg5"] = col("VIXCLS").pct_change(5)
    out["t10y2y"] = col("T10Y2Y")
    out["dgs10_chg20"] = col("DGS10").diff(20)
    out["credit_spread"] = col("BAA10Y")
    out["credit_chg20"] = col("BAA10Y").diff(20)
    out["usd_chg20"] = col("DTWEXBGS").pct_change(20)
    out["oil_chg20"] = col("DCOILWTICO").pct_change(20)
    return out


def news_daily(con: sqlite3.Connection, cal: pd.DatetimeIndex, model: str = "ProsusAI/finbert") -> pd.DataFrame:
    """Per (symbol, decision day): article count and sentiment stats. Also returns article-level rows for audits."""
    q = """SELECT n.id, n.created_at, n.source, ns.symbol, s.score
           FROM news n JOIN news_symbol ns ON ns.id = n.id
           LEFT JOIN news_sentiment s ON s.id = n.id AND s.model = ?"""
    a = pd.read_sql(q, con, params=(model,))
    if a.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    a["date"] = to_decision_day(a.created_at, cal)
    a = a.dropna(subset=["date"])
    a["pos"] = (a.score > 0.5).astype(float).where(a.score.notna())
    d = a.groupby(["symbol", "date"]).agg(n=("id", "size"), s=("score", "sum"), k=("score", "count"),
                                          pos=("pos", "sum"))
    d = d.reset_index()
    full = []
    for sym, g in d.groupby("symbol"):
        g = g.drop(columns="symbol").set_index("date").reindex(cal, fill_value=0).astype(float)
        r5 = g.rolling(5, min_periods=1).sum()
        r60 = g.n.rolling(60, min_periods=20).mean().shift(1)
        out = pd.DataFrame({
            "symbol": sym,
            "news_1d": g.n, "news_5d": r5.n,
            "news_surge": (g.n + 1) / (r60 + 1),
            "sent_1d": (g.s / g.k.replace(0, np.nan)).fillna(0.0),   # 0 = no scored news that day
            "sent_5d": (r5.s / r5.k.replace(0, np.nan)).fillna(0.0),
            "sent_pos_share_5d": r5.pos / r5.k.replace(0, np.nan),
        }, index=cal)
        full.append(out.rename_axis("date").reset_index())
    return pd.concat(full, ignore_index=True)


def earnings_features(con: sqlite3.Connection, bars: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    e = pd.read_sql("SELECT symbol, filing_date, accepted_at FROM earnings", con)
    if e.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    ts = e.accepted_at.fillna(e.filing_date + "T21:00:00Z")  # no acceptance time: assume after the close
    e["day"] = to_decision_day(ts, cal)   # first close at which the release was public
    e = e.dropna(subset=["day"])
    closes = bars.pivot(index="date", columns="symbol", values="close")
    closes.index = pd.to_datetime(closes.index)
    out = []
    for sym, g in e.groupby("symbol"):
        if sym not in closes:
            continue
        c = closes[sym].reindex(cal)
        pos = pd.Series(np.arange(len(cal)), index=cal)
        days = sorted(set(g.day))
        # reaction = close on the release's decision day vs the close before it
        react = {d: c[d] / c.shift(1)[d] - 1 for d in days}
        last_day = pd.Series(pd.NaT, index=cal, dtype="datetime64[ns]")
        last_day.loc[days] = days
        last_day = last_day.ffill()
        since = pos - pos.reindex(last_day.values).values
        rx = last_day.map(react).astype(float)
        out.append(pd.DataFrame({"symbol": sym, "date": cal,
                                 "days_since_earn": since.values,
                                 "earn_react": np.where(since.values <= 20, rx.values, 0.0),
                                 "earn_recent": (since.values <= 5).astype(float)}))
    return pd.concat(out, ignore_index=True)


def build(con: sqlite3.Connection, symbols: Optional[list[str]] = None) -> pd.DataFrame:
    bars = load_bars(con)
    if symbols:
        bars = bars[bars.symbol.isin(set(symbols) | {BENCH})]
    cal = _calendar(bars)
    f = price_features(bars)
    f["date"] = pd.to_datetime(f.date)
    f = f.merge(macro_features(con, cal), left_on="date", right_index=True, how="left")
    nd = news_daily(con, cal)
    if not nd.empty:
        f = f.merge(nd, on=["symbol", "date"], how="left")
    ed = earnings_features(con, bars, cal)
    if not ed.empty:
        f = f.merge(ed, on=["symbol", "date"], how="left")
    for c in MACRO_FEATURES + NEWS_FEATURES + EARN_FEATURES:
        if c not in f:
            f[c] = np.nan
    return f.sort_values(["date", "symbol"]).reset_index(drop=True)
