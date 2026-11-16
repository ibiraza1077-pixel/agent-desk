"""The local history database (data/history.db). One schema shared by the downloaders and the backtest."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("HISTORY_DB", Path(__file__).resolve().parents[2] / "data" / "history.db"))

SCHEMA = """
-- Daily bars, split+dividend adjusted (Adjustment.ALL), SIP feed. date = trading day in New York (YYYY-MM-DD).
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL, vwap REAL,
    PRIMARY KEY (symbol, date));

-- News articles (Alpaca / Benzinga). created_at is UTC ISO-8601 'YYYY-MM-DDTHH:MM:SSZ'.
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, source TEXT, author TEXT,
    headline TEXT NOT NULL, summary TEXT);
-- Which watchlist symbols each article is tagged with (an article can name several).
CREATE TABLE IF NOT EXISTS news_symbol (
    id INTEGER NOT NULL, symbol TEXT NOT NULL, PRIMARY KEY (id, symbol));
CREATE INDEX IF NOT EXISTS news_symbol_sym ON news_symbol(symbol);
CREATE INDEX IF NOT EXISTS news_created ON news(created_at);

-- Headline sentiment from a local model. score = p_positive - p_negative, in [-1, 1].
CREATE TABLE IF NOT EXISTS news_sentiment (
    id INTEGER NOT NULL, model TEXT NOT NULL, p_pos REAL, p_neg REAL, p_neu REAL, score REAL,
    PRIMARY KEY (id, model));

-- FRED series. date = observation date (YYYY-MM-DD). Missing values ('.') are not stored.
CREATE TABLE IF NOT EXISTS macro (
    series TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY (series, date));

-- SEC EDGAR 8-K filings with item 2.02 (earnings releases). accepted_at = EDGAR acceptance time, UTC ISO-8601.
CREATE TABLE IF NOT EXISTS earnings (
    symbol TEXT NOT NULL, accession TEXT NOT NULL, filing_date TEXT NOT NULL, accepted_at TEXT,
    items TEXT, PRIMARY KEY (symbol, accession));

-- Bookkeeping: what each downloader has finished, so runs can resume.
CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: str | os.PathLike | None = None) -> sqlite3.Connection:
    p = Path(path or DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def get_progress(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM progress WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_progress(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO progress(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))
    con.commit()
