"""CLI: `uv run --group research python -m desk.research <command> [--symbols A,B] [--db PATH]`.

Heavy modules (torch, alpaca) are imported lazily inside the handlers so `status` and `--help` start instantly.
To add a command, write a handler `def _cmd_x(args) -> None` and add one line to COMMANDS.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from typing import Callable

from desk.config import load_settings
from desk.research.db import connect


def _ctx(args: argparse.Namespace):
    settings = load_settings()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else settings.watchlist
    return connect(args.db), settings, symbols


def _cmd_bars(args: argparse.Namespace) -> None:
    from desk.research import download
    con, settings, symbols = _ctx(args)
    download.download_bars(con, settings, symbols, **({"start": args.start} if args.start else {}))


def _cmd_news(args: argparse.Namespace) -> None:
    from desk.research import download
    con, settings, symbols = _ctx(args)
    kw = {"start": args.start} if args.start else {}
    download.download_news(con, settings, symbols, end=args.end, **kw)


def _cmd_macro(args: argparse.Namespace) -> None:
    from desk.research import download
    download.download_macro(connect(args.db))


def _cmd_earnings(args: argparse.Namespace) -> None:
    from desk.research import download
    con, _, symbols = _ctx(args)
    download.download_earnings(con, symbols)


def _cmd_sentiment(args: argparse.Namespace) -> None:
    from desk.research import sentiment
    sentiment.score_news(connect(args.db))


def _cmd_download_all(args: argparse.Namespace) -> None:
    for step in (_cmd_bars, _cmd_macro, _cmd_earnings, _cmd_news, _cmd_sentiment):
        step(args)


def _cmd_status(args: argparse.Namespace) -> None:
    con = connect(args.db)
    # (table, date column) -> row count and date range
    for table, col in (("bars", "date"), ("news", "created_at"), ("news_sentiment", None), ("macro", "date"),
                       ("earnings", "filing_date")):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        rng = ""
        if col and n:
            lo, hi = con.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
            rng = f"  {lo} .. {hi}"
        print(f"{table:15s} {n:>9,d}{rng}")
    _breakdown(con, "bars per symbol", "SELECT symbol, COUNT(*) FROM bars GROUP BY symbol ORDER BY symbol")
    _breakdown(con, "macro per series", "SELECT series, COUNT(*) FROM macro GROUP BY series ORDER BY series")
    _breakdown(con, "news per symbol", "SELECT symbol, COUNT(*) FROM news_symbol GROUP BY symbol ORDER BY symbol")
    _breakdown(con, "earnings per symbol", "SELECT symbol, COUNT(*) FROM earnings GROUP BY symbol ORDER BY symbol")
    _breakdown(con, "progress", "SELECT key, value FROM progress ORDER BY key")


def _breakdown(con: sqlite3.Connection, title: str, sql: str) -> None:
    rows = con.execute(sql).fetchall()
    if rows:
        print(f"\n{title}:")
        for k, v in rows:
            print(f"  {k:12s} {v}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Walk-forward backtest + signal/source audits; writes data/research/report.md."""
    from desk.research import report
    con = connect(args.db)
    try:
        r = report.run(con)
    finally:
        con.close()
    print(f"chosen model: {r['chosen_model']}  base rate: {r['base_rate_beat_spy']:.1%}  -> data/research/report.md")


COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "bars": _cmd_bars,
    "news": _cmd_news,
    "macro": _cmd_macro,
    "earnings": _cmd_earnings,
    "sentiment": _cmd_sentiment,
    "download-all": _cmd_download_all,
    "status": _cmd_status,
    "report": _cmd_report,
}


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", help="comma-separated symbols (default: WATCHLIST from settings)")
    common.add_argument("--db", help="history database path (default: data/history.db or $HISTORY_DB)")
    common.add_argument("--start", help="bars/news: first date, YYYY-MM-DD (defaults 2016-01-01 / 2015-01-01)")
    common.add_argument("--end", help="news: last date, YYYY-MM-DD (default: now)")
    ap = argparse.ArgumentParser(prog="python -m desk.research", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in COMMANDS.items():
        sub.add_parser(name, parents=[common], help=(fn.__doc__ or name).strip().splitlines()[0])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
