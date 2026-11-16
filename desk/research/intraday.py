"""Intraday (5-minute) bar history for the day-trading study, in its own database (data/intraday.db).

Bars are SIP, split+dividend adjusted, stamped by their START time in New York ("YYYY-MM-DD HH:MM").
Only 08:00-16:00 New York is kept: the pre-market hour and a half (for gaps and pre-open news reaction)
and the regular session. Resumable: each symbol remembers the last day it stored.

    uv run --group research python -m desk.research.intraday download
    uv run --group research python -m desk.research.intraday status
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from desk.research.download import _need_keys, _with_backoff

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
DEFAULT_PATH = Path(os.environ.get("INTRADAY_DB", Path(__file__).resolve().parents[2] / "data" / "intraday.db"))

# Liquid ETFs (index, gold, silver, bonds, oil, sectors) plus the desk's large-cap stocks.
ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO", "XLE", "XLF", "XLK"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "XOM"]
UNIVERSE = ETFS + STOCKS
START = "2018-01-01"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars5 (
    symbol TEXT NOT NULL, ts TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL, trades INTEGER,
    PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: Path | str = DEFAULT_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _keep(local: datetime) -> bool:
    hm = local.hour * 60 + local.minute
    return 8 * 60 <= hm < 16 * 60


def rows_from_bars(symbol: str, bars: Any) -> list[tuple]:
    out = []
    for b in bars:
        ts = b.timestamp if b.timestamp.tzinfo else b.timestamp.replace(tzinfo=UTC)
        local = ts.astimezone(NY)
        if _keep(local):
            out.append((symbol, local.strftime("%Y-%m-%d %H:%M"), b.open, b.high, b.low, b.close, b.volume,
                        b.vwap, getattr(b, "trade_count", None)))
    return out


def download(con: sqlite3.Connection, settings: Any, symbols: Sequence[str] = UNIVERSE, start: str = START,
             *, client: Any = None, chunk_days: int = 120) -> None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        _need_keys(settings)
        client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    end_all = datetime.now(UTC) - timedelta(minutes=16)  # free plan: no SIP from the last 15 minutes
    tf = TimeFrame(5, TimeFrameUnit.Minute)
    for sym in symbols:
        row = con.execute("SELECT value FROM progress WHERE key=?", (f"bars5:{sym}",)).fetchone()
        cur = datetime.fromisoformat(row[0]) if row else datetime.fromisoformat(start).replace(tzinfo=UTC)
        total = 0
        while cur < end_all:
            nxt = min(cur + timedelta(days=chunk_days), end_all)
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=cur, end=nxt,
                                   adjustment=Adjustment.ALL, feed=DataFeed.SIP)
            res = _with_backoff(lambda: client.get_stock_bars(req), f"bars5 {sym} {cur:%Y-%m-%d}")
            rows = rows_from_bars(sym, res.data.get(sym, []))
            con.executemany("INSERT OR REPLACE INTO bars5 VALUES(?,?,?,?,?,?,?,?,?)", rows)
            con.execute("INSERT OR REPLACE INTO progress VALUES(?,?)", (f"bars5:{sym}", nxt.isoformat()))
            con.commit()
            total += len(rows)
            cur = nxt
        log.info("bars5 %s: +%d rows", sym, total)


def status(con: sqlite3.Connection) -> None:
    for sym, n, lo, hi in con.execute("SELECT symbol, COUNT(*), MIN(ts), MAX(ts) FROM bars5 GROUP BY symbol"):
        print(f"{sym:6} {n:>9,}  {lo} .. {hi}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m desk.research.intraday")
    ap.add_argument("command", choices=["download", "status"])
    ap.add_argument("--symbols", nargs="*", default=UNIVERSE)
    ap.add_argument("--start", default=START)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = connect()
    if a.command == "download":
        from desk.config import load_settings
        download(con, load_settings(), a.symbols, a.start)
    status(con)


if __name__ == "__main__":
    main()
