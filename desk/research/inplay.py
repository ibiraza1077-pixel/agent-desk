"""'Stocks in play' opening-range breakout, tested as published (Zarattini, Barbon & Aziz, 2024):

  universe   US-listed stocks, price > $5, 14-day average volume > 1M shares, ATR(14) > $0.50 (all known by
             the previous close)
  in play    the 20 with the highest relative volume in the first 5 minutes (vs their own previous 14 days),
             relative volume >= 1
  trade      direction of the first 5-minute candle (skip if flat); enter at 09:35; stop 10% of ATR(14) away;
             otherwise exit at the 16:00 close
  sizing     risk 1% of equity per trade, at most 4x leverage across the 20 (also reported unlevered)
  costs      $0.0035 per share commission + slippage on entry and exit (2 bp; 5 bp sensitivity)

Delisted symbols are included (Alpaca 'inactive' assets) to limit survivorship bias. Prices are unadjusted
(as traded that day). Pipeline, each step resumable:

    uv run --group research python -m desk.research.inplay assets
    uv run --group research python -m desk.research.inplay daily      # daily bars -> eligible days
    uv run --group research python -m desk.research.inplay opening    # first 5-min volume per eligible day
    uv run --group research python -m desk.research.inplay minutes    # 1-min bars for each day's top 20
    uv run --group research python -m desk.research.inplay backtest
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
from datetime import UTC, datetime, time as dtime, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from desk.research.download import _need_keys, _with_backoff
from desk.research.intraday import connect

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
START = "2017-11-01"            # 14-day lookbacks are ready by Jan 2018
TEST_START = "2018-01-01"
TRAIN = range(2018, 2023)
HOLDOUT = range(2023, 2027)
TOP_N = 20
MIN_PRICE, MIN_AVG_VOL, MIN_ATR = 5.0, 1_000_000, 0.50
STOP_ATR = 0.10
RISK_PER_TRADE, MAX_LEVERAGE = 0.01, 4.0
COMMISSION = 0.0035
EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "BATS", "ARCA"}
NOT_COMMON = re.compile(r"\b(warrants?|units?|rights?|preferred|notes due|ETF|ETN|fund|trust)\b", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT, status TEXT);
-- one row per symbol-day that passes the universe filter, using only data known at the previous close
CREATE TABLE IF NOT EXISTS eligible (date TEXT, symbol TEXT, prev_close REAL, atr14 REAL, avgvol14 REAL,
    PRIMARY KEY (date, symbol)) WITHOUT ROWID;
-- loose superset for the relative-volume lookback (price > 3, avg vol > 500k)
CREATE TABLE IF NOT EXISTS watch (date TEXT, symbol TEXT, PRIMARY KEY (date, symbol)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS open5 (date TEXT, symbol TEXT, open REAL, close REAL, volume REAL,
    PRIMARY KEY (date, symbol)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS picks (date TEXT, symbol TEXT, rvol REAL, PRIMARY KEY (date, symbol)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS min1 (symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
"""


def db() -> sqlite3.Connection:
    con = connect()
    con.executescript(SCHEMA)
    return con


def _clients(settings: Any):
    from alpaca.data.historical import StockHistoricalDataClient
    _need_keys(settings)
    return StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)


def _progress(con, key, value=None):
    if value is None:
        r = con.execute("SELECT value FROM progress WHERE key=?", (key,)).fetchone()
        return r[0] if r else None
    con.execute("INSERT OR REPLACE INTO progress VALUES(?,?)", (key, value))
    con.commit()


# ----------------------------------------------------------------------------------------------- steps

def step_assets(con, settings) -> None:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    tc = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=True)
    rows = []
    for st in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for a in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=st)):
            ex = str(a.exchange).split(".")[-1]
            if ex in EXCHANGES and not NOT_COMMON.search(a.name or "") and re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", a.symbol):
                rows.append((a.symbol, a.name, ex, str(st).split(".")[-1]))
    con.executemany("INSERT OR REPLACE INTO assets VALUES(?,?,?,?)", rows)
    con.commit()
    log.info("assets: %d common-stock symbols kept", len(rows))


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars for one symbol -> per-day prior-close stats (shifted: known before the session opens)."""
    df = df.sort_values("date")
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    out = pd.DataFrame({"date": df.date, "prev_close": pc,
                        "atr14": tr.rolling(14).mean().shift(1),
                        "avgvol14": df.volume.rolling(14).mean().shift(1)})
    # a split inside the window corrupts unadjusted ATR / volume: skip those days
    jump = (df.close / pc - 1).abs().gt(0.6).rolling(15).max().shift(1).fillna(1).astype(bool)
    return out[~jump.values]


def step_daily(con, settings, chunk: int = 200) -> None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = _clients(settings)
    syms = [r[0] for r in con.execute("SELECT symbol FROM assets ORDER BY symbol")]
    done = int(_progress(con, "inplay:daily") or 0)
    end = datetime.now(UTC) - timedelta(minutes=16)
    for i in range(done, len(syms), chunk):
        batch = syms[i:i + chunk]
        req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=datetime.fromisoformat(START).replace(tzinfo=UTC),
                               end=end, adjustment=Adjustment.RAW, feed=DataFeed.SIP)
        res = _with_backoff(lambda: client.get_stock_bars(req), f"daily {batch[0]}..")
        elig, watch = [], []
        for sym, bars in res.data.items():
            if len(bars) < 20:
                continue
            df = pd.DataFrame({"date": [b.timestamp.astimezone(NY).date().isoformat() for b in bars],
                               "high": [b.high for b in bars], "low": [b.low for b in bars],
                               "close": [b.close for b in bars], "volume": [b.volume for b in bars]})
            f = _filters(df).dropna()
            f = f[(f.date >= TEST_START) & (f.date < datetime.now(NY).date().isoformat())]
            e = f[(f.prev_close > MIN_PRICE) & (f.avgvol14 > MIN_AVG_VOL) & (f.atr14 > MIN_ATR)]
            w = f[(f.prev_close > 3) & (f.avgvol14 > 500_000)]
            elig += [(d, sym, pc, a, v) for d, pc, a, v in zip(e.date, e.prev_close, e.atr14, e.avgvol14)]
            watch += [(d, sym) for d in w.date]
        con.executemany("INSERT OR REPLACE INTO eligible VALUES(?,?,?,?,?)", elig)
        con.executemany("INSERT OR REPLACE INTO watch VALUES(?,?)", watch)
        _progress(con, "inplay:daily", str(i + chunk))
        log.info("daily %d/%d symbols: +%d eligible days", min(i + chunk, len(syms)), len(syms), len(elig))


def step_opening(con, settings, chunk: int = 400) -> None:
    """First 5-minute bar (09:30-09:35) for every watched symbol-day."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = _clients(settings)
    tf = TimeFrame(5, TimeFrameUnit.Minute)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM watch ORDER BY date")]
    last = _progress(con, "inplay:opening") or ""
    for day in days:
        if day <= last:
            continue
        syms = [r[0] for r in con.execute("SELECT symbol FROM watch WHERE date=? ORDER BY symbol", (day,))]
        d = datetime.fromisoformat(day)
        start = datetime.combine(d, dtime(9, 30), NY)
        rows = []
        for i in range(0, len(syms), chunk):
            req = StockBarsRequest(symbol_or_symbols=syms[i:i + chunk], timeframe=tf, start=start,
                                   end=start + timedelta(minutes=1), adjustment=Adjustment.RAW, feed=DataFeed.SIP)
            res = _with_backoff(lambda: client.get_stock_bars(req), f"open5 {day}")
            for sym, bars in res.data.items():
                for b in bars:
                    if b.timestamp.astimezone(NY).time() == dtime(9, 30):
                        rows.append((day, sym, b.open, b.close, b.volume))
        con.executemany("INSERT OR REPLACE INTO open5 VALUES(?,?,?,?,?)", rows)
        _progress(con, "inplay:opening", day)
        log.info("open5 %s: %d/%d symbols", day, len(rows), len(syms))


def select_picks(con) -> pd.DataFrame:
    o = pd.read_sql("SELECT date, symbol, volume FROM open5", con)
    o = o.sort_values(["symbol", "date"])
    # previous 14 sessions that have an opening bar for this symbol (needs at least 10)
    o["avg"] = o.groupby("symbol").volume.transform(lambda v: v.rolling(14, min_periods=10).mean().shift(1))
    o["rvol"] = o.volume / o.avg
    e = pd.read_sql("SELECT date, symbol FROM eligible", con)
    m = o.merge(e, on=["date", "symbol"]).dropna(subset=["rvol"])
    m = m[m.rvol >= 1.0]
    top = m.sort_values(["date", "rvol"], ascending=[True, False]).groupby("date").head(TOP_N)
    con.execute("DELETE FROM picks")
    con.executemany("INSERT INTO picks VALUES(?,?,?)", top[["date", "symbol", "rvol"]].itertuples(index=False))
    con.commit()
    return top


def step_minutes(con, settings) -> None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = _clients(settings)
    if not con.execute("SELECT 1 FROM picks LIMIT 1").fetchone():
        select_picks(con)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM picks ORDER BY date")]
    last = _progress(con, "inplay:minutes") or ""
    for day in days:
        if day <= last:
            continue
        syms = [r[0] for r in con.execute("SELECT symbol FROM picks WHERE date=?", (day,))]
        d = datetime.fromisoformat(day)
        req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                               start=datetime.combine(d, dtime(9, 30), NY), end=datetime.combine(d, dtime(16, 0), NY),
                               adjustment=Adjustment.RAW, feed=DataFeed.SIP)
        res = _with_backoff(lambda: client.get_stock_bars(req), f"min1 {day}")
        rows = []
        for sym, bars in res.data.items():
            for b in bars:
                t = b.timestamp.astimezone(NY)
                if dtime(9, 30) <= t.time() < dtime(16, 0):
                    rows.append((sym, t.strftime("%Y-%m-%d %H:%M"), b.open, b.high, b.low, b.close, b.volume))
        con.executemany("INSERT OR REPLACE INTO min1 VALUES(?,?,?,?,?,?,?)", rows)
        _progress(con, "inplay:minutes", day)
        log.info("min1 %s: %d symbols, %d bars", day, len(res.data), len(rows))


# -------------------------------------------------------------------------------------------- backtest

def trade(bars: pd.DataFrame, atr: float, first_open: float, first_close: float) -> dict | None:
    """bars: one symbol's 1-minute bars for the day (ts 'HH:MM' >= 09:35, sorted). Returns the trade or None."""
    s = np.sign(first_close - first_open)
    if s == 0 or bars.empty or bars.iloc[0].hm != "09:35":
        return None
    entry = float(bars.iloc[0].open)
    stop = entry - s * STOP_ATR * atr
    exit_px, reason = float(bars.iloc[-1].close), "close"
    lows, highs, opens = bars.low.values, bars.high.values, bars.open.values
    for i in range(len(bars)):
        if s > 0 and lows[i] <= stop:
            exit_px, reason = min(opens[i], stop), "stop"
            break
        if s < 0 and highs[i] >= stop:
            exit_px, reason = max(opens[i], stop), "stop"
            break
    return {"side": int(s), "entry": entry, "exit": exit_px, "stop_dist": STOP_ATR * atr, "reason": reason,
            "last_hm": bars.iloc[-1].hm}


def build_trades(con) -> pd.DataFrame:
    picks = pd.read_sql("SELECT p.date, p.symbol, p.rvol, e.atr14, o.open AS o5, o.close AS c5 FROM picks p "
                        "JOIN eligible e USING(date, symbol) JOIN open5 o USING(date, symbol)", con)
    out = []
    for day, grp in picks.groupby("date"):
        m = pd.read_sql("SELECT symbol, ts, open, high, low, close FROM min1 WHERE ts >= ? AND ts < ? ",
                        con, params=(f"{day} 09:35", f"{day} 16:00"))
        if m.empty:
            continue
        m["hm"] = m.ts.str[11:16]
        for r in grp.itertuples():
            t = trade(m[m.symbol == r.symbol].sort_values("ts"), r.atr14, r.o5, r.c5)
            if t:
                out.append({"date": day, "symbol": r.symbol, "rvol": r.rvol, **t})
    return pd.DataFrame(out)


def portfolio(trades: pd.DataFrame, slip_bps: float, levered: bool) -> pd.Series:
    """Daily return on equity. Levered: risk 1% per trade, total gross <= 4x. Unlevered: 1/20 of equity each."""
    t = trades.copy()
    if levered:
        w = RISK_PER_TRADE * t.entry / t.stop_dist               # position value / equity for 1% risk
        w = np.minimum(w, MAX_LEVERAGE / TOP_N)
    else:
        w = pd.Series(1.0 / TOP_N, index=t.index)
    gross = t.side * (t.exit / t.entry - 1)
    cost = 2 * slip_bps / 1e4 + COMMISSION * (1 / t.entry + 1 / t.exit)
    t["contrib"] = w * (gross - cost)
    return t.groupby("date").contrib.sum()


def stats(daily: pd.Series, all_days: pd.Index) -> dict:
    daily = daily.reindex(all_days, fill_value=0.0)
    if len(daily) < 20:
        return {}
    mu, sd = daily.mean(), daily.std()
    eq = (1 + daily).cumprod()
    yrs = daily.groupby(pd.to_datetime(daily.index).year).apply(lambda x: (1 + x).prod() - 1)
    return {"ann_return": float(eq.iloc[-1] ** (252 / len(daily)) - 1), "sharpe": float(mu / sd * math.sqrt(252)),
            "t_stat": float(mu / sd * math.sqrt(len(daily))), "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}", "by_year": {int(k): round(float(v), 4) for k, v in yrs.items()}}


def step_backtest(con) -> dict:
    from desk.research.daytrade import OUT
    trades = build_trades(con)
    days = pd.Index(sorted(r[0] for r in con.execute("SELECT DISTINCT date FROM eligible")))
    res: dict = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"), "n_trades": int(len(trades)),
                 "trades_per_day": float(len(trades) / max(1, trades.date.nunique())),
                 "stop_rate": float((trades.reason == "stop").mean()),
                 "win_rate_gross": float(((trades.side * (trades.exit - trades.entry)) > 0).mean()),
                 "median_price": float(trades.entry.median()), "runs": {}}
    for lev in (False, True):
        for slip in (0.0, 2.0, 5.0, 10.0):
            d = portfolio(trades, slip, lev)
            key = f"{'levered' if lev else 'unlevered'}_slip{slip:g}bp"
            res["runs"][key] = {
                "train": stats(d[pd.to_datetime(d.index).year.isin(list(TRAIN))], days[pd.to_datetime(days).year.isin(list(TRAIN))]),
                "holdout": stats(d[pd.to_datetime(d.index).year.isin(list(HOLDOUT))], days[pd.to_datetime(days).year.isin(list(HOLDOUT))])}
    OUT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT / "inplay_trades.csv", index=False)
    (OUT / "inplay.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m desk.research.inplay")
    ap.add_argument("step", choices=["assets", "daily", "opening", "picks", "minutes", "backtest", "all"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = db()
    from desk.config import load_settings
    s = load_settings()
    steps = ["assets", "daily", "opening", "picks", "minutes", "backtest"] if a.step == "all" else [a.step]
    for st in steps:
        if st == "assets":
            step_assets(con, s)
        elif st == "daily":
            step_daily(con, s)
        elif st == "opening":
            step_opening(con, s)
        elif st == "picks":
            log.info("picks: %d symbol-days", len(select_picks(con)))
        elif st == "minutes":
            step_minutes(con, s)
        elif st == "backtest":
            print(json.dumps(step_backtest(con), indent=1))


if __name__ == "__main__":
    main()
