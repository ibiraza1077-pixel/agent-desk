"""Downloaders for the research database. Every function is idempotent (upserts) and resumable (progress keys),
so an interrupted run can simply be started again. API keys are never logged."""
from __future__ import annotations

import csv
import io
import logging
import sqlite3
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import httpx

from desk.config import Settings
from desk.research.db import get_progress, set_progress

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"

DEFAULT_MACRO = ["VIXCLS", "DGS10", "DGS2", "T10Y2Y", "BAA10Y", "DTWEXBGS", "DCOILWTICO"]
MACRO_START = "2014-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Free-tier news limit is 200 requests/minute; stay under it.
NEWS_REQUESTS_PER_MINUTE = 180
NEWS_PAGE_LIMIT = 50
MAX_TRIES = 5

# SEC asks automated clients to identify themselves; no personal details here on purpose.
SEC_USER_AGENT = "AgentDesk research (personal, non-commercial)"
SEC_MIN_INTERVAL = 0.25  # 4 requests/second, under SEC's 10/s fair-access cap and our 5/s budget
EARNINGS_START = "2015-01-01"
# Verified against each company's `tickers` field in its submissions JSON at run time.
CIKS: dict[str, int] = {
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724, "GOOGL": 1652044,
    "META": 1326801, "TSLA": 1318605, "AMD": 2488, "JPM": 19617, "XOM": 34088,
}
# Fallback identity check for filers whose EDGAR `tickers` list is empty (Exxon's is, as of 2026-11).
CIK_NAMES: dict[str, str] = {"XOM": "EXXON MOBIL CORP"}
ETFS = {"SPY", "QQQ"}


# --------------------------------------------------------------------------------------------- helpers

def _utc_str(value: str | datetime) -> str:
    """Normalise an ISO-8601 timestamp (with 'Z' or an offset, optional fractions) to UTC 'YYYY-MM-DDTHH:MM:SSZ'."""
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime(UTC_FMT)


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, UTC_FMT).replace(tzinfo=UTC)


def _status(exc: BaseException):
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _is_rate_limited(exc: BaseException) -> bool:
    """True for an HTTP 429 from either alpaca-py (APIError.status_code) or httpx."""
    return _status(exc) == 429


def _is_retryable(exc: BaseException) -> bool:
    """429, any 5xx, or a network timeout / connection error: worth retrying in a multi-hour download."""
    code = _status(exc)
    if code == 429 or (code is not None and 500 <= code < 600):
        return True
    import requests  # alpaca-py's transport
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError, requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError))


def _with_backoff(fn: Callable[[], Any], what: str, sleep: Callable[[float], None] = time.sleep,
                  tries: int = MAX_TRIES) -> Any:
    """Call fn(); on a 429, 5xx or network error wait 2, 4, 8, ... seconds (capped at 60) and retry, up to `tries`."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - only rate-limit errors are retried, everything else re-raised
            if not _is_retryable(exc) or attempt == tries:
                raise
            wait = min(2.0 ** attempt, 60.0)
            log.warning("%s: %s, retry %d/%d in %.0fs", what,
                        "rate limited (429)" if _is_rate_limited(exc) else f"transient error ({type(exc).__name__})",
                        attempt, tries - 1, wait)
            sleep(wait)
    raise AssertionError("unreachable")


class _Throttle:
    """Block so that calls are at least 60/per_minute seconds apart."""

    def __init__(self, per_minute: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._sleep, self._clock = sleep, clock
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


def _need_keys(settings: Settings) -> None:
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set (see .env.example).")


# --------------------------------------------------------------------------------------------- bars

def _bar_rows(symbol: str, bars: Iterable[Any]) -> list[tuple]:
    """alpaca Bar objects -> rows. Daily bars are stamped at midnight New York time, so convert before taking the date."""
    rows = []
    for b in bars:
        ts = b.timestamp if b.timestamp.tzinfo else b.timestamp.replace(tzinfo=UTC)
        rows.append((symbol, ts.astimezone(NY).date().isoformat(), b.open, b.high, b.low, b.close, b.volume, b.vwap))
    return rows


def download_bars(con: sqlite3.Connection, settings: Settings, symbols: Sequence[str], start: str = "2016-01-01",
                  *, client: Any = None) -> None:
    """Daily SIP bars, split+dividend adjusted, one request per symbol (alpaca-py paginates internally)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        _need_keys(settings)
        client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    # The free plan may not query the most recent 15 minutes of SIP data; stop just short of "now".
    end_dt = datetime.now(UTC) - timedelta(minutes=16)
    for sym in symbols:
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=start_dt, end=end_dt,
                               adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        res = _with_backoff(lambda: client.get_stock_bars(req), f"bars {sym}")
        rows = _bar_rows(sym, res.data.get(sym, []))
        con.executemany("INSERT OR REPLACE INTO bars(symbol, date, open, high, low, close, volume, vwap) "
                        "VALUES(?,?,?,?,?,?,?,?)", rows)
        con.commit()
        log.info("bars %s: %d rows (%s .. %s)", sym, len(rows), rows[0][1] if rows else "-", rows[-1][1] if rows else "-")


# --------------------------------------------------------------------------------------------- news

def _store_articles(con: sqlite3.Connection, articles: Iterable[dict], watchlist: set[str]) -> str | None:
    """Upsert a page of raw article dicts; returns the latest created_at stored (UTC string)."""
    last: str | None = None
    for a in articles:
        created = _utc_str(a["created_at"])
        con.execute("INSERT OR REPLACE INTO news(id, created_at, source, author, headline, summary) VALUES(?,?,?,?,?,?)",
                    (int(a["id"]), created, a.get("source"), a.get("author"), a.get("headline") or "",
                     a.get("summary")))
        con.executemany("INSERT OR IGNORE INTO news_symbol(id, symbol) VALUES(?,?)",
                        [(int(a["id"]), s) for s in (a.get("symbols") or []) if s in watchlist])
        if last is None or created > last:
            last = created
    return last


def download_news(con: sqlite3.Connection, settings: Settings, symbols: Sequence[str], start: str = "2015-01-01",
                  end: str | None = None, *, client: Any = None,
                  requests_per_minute: float = NEWS_REQUESTS_PER_MINUTE,
                  sleep: Callable[[float], None] = time.sleep) -> None:
    """Alpaca (Benzinga) news, oldest first, one symbol at a time, resumable via progress key news:<symbol>.

    Pages are fetched with the raw REST call because the high-level NewsSet drops `next_page_token`. Each page is
    committed together with the progress marker, so a crash loses at most the page in flight; a rerun restarts at
    the last stored created_at (inclusive) and the upserts absorb the overlap.
    """
    from alpaca.data.requests import NewsRequest

    if client is None:
        from alpaca.data.historical.news import NewsClient
        _need_keys(settings)
        client = NewsClient(settings.alpaca_api_key, settings.alpaca_secret_key, raw_data=True)
    watchlist = set(settings.watchlist) | set(symbols)
    end_dt = _parse_utc(_utc_str(end)) if end else None
    throttle = _Throttle(requests_per_minute, sleep)

    for sym in symbols:
        key = f"news:{sym}"
        resume = get_progress(con, key)
        start_dt = _parse_utc(_utc_str(start if resume is None else max(resume, _utc_str(start))))
        token: str | None = None
        pages = articles = 0
        while True:
            req = NewsRequest(symbols=sym, start=start_dt, end=end_dt, limit=NEWS_PAGE_LIMIT, sort="asc",
                              include_content=False, page_token=token)
            params = req.to_request_fields()

            def fetch() -> dict:
                throttle.wait()
                return client.get("/news", params)

            resp = _with_backoff(fetch, f"news {sym}", sleep)
            batch = resp.get("news") or []
            last = _store_articles(con, batch, watchlist)
            if last:
                set_progress(con, key, last)  # commits the page too
            else:
                con.commit()
            pages += 1
            articles += len(batch)
            if pages % 10 == 0:
                log.info("news %s: %d pages, %d articles, up to %s", sym, pages, articles, last)
            token = resp.get("next_page_token")
            if not token:
                break
        log.info("news %s: done, %d pages, %d articles (progress %s)", sym, pages, articles, get_progress(con, key))


# --------------------------------------------------------------------------------------------- macro

def parse_fred_csv(text: str, series: str, start: str = MACRO_START) -> list[tuple[str, str, float]]:
    """FRED graph CSV (header `observation_date,<SERIES>`) -> (series, date, value) rows; '.'/empty values skipped."""
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header
    rows = []
    for rec in reader:
        if len(rec) < 2:
            continue
        d, v = rec[0].strip(), rec[1].strip()
        if not d or d < start or v in ("", "."):
            continue
        try:
            rows.append((series, d, float(v)))
        except ValueError:
            continue
    return rows


def download_macro(con: sqlite3.Connection, series: Sequence[str] = DEFAULT_MACRO, *,
                   http: httpx.Client | None = None) -> None:
    """FRED series as CSV (no API key needed). Always refetches the whole series; they are small."""
    own = http is None
    http = http or httpx.Client(timeout=60, follow_redirects=True)
    try:
        for sid in series:
            def fetch() -> str:
                r = http.get(FRED_URL, params={"id": sid})
                r.raise_for_status()
                return r.text

            text = _with_backoff(fetch, f"macro {sid}")
            rows = parse_fred_csv(text, sid)
            con.executemany("INSERT OR REPLACE INTO macro(series, date, value) VALUES(?,?,?)", rows)
            con.commit()
            log.info("macro %s: %d rows (%s .. %s)", sid, len(rows), rows[0][1] if rows else "-",
                     rows[-1][1] if rows else "-")
    finally:
        if own:
            http.close()


# --------------------------------------------------------------------------------------------- earnings

def parse_filings(symbol: str, cols: dict, since: str = EARNINGS_START) -> list[tuple]:
    """EDGAR columnar filings block -> earnings rows: form == '8-K' with item 2.02, filed on/after `since`."""
    rows = []
    n = len(cols.get("accessionNumber", []))
    for i in range(n):
        if cols["form"][i] != "8-K":
            continue
        items = [x.strip() for x in (cols["items"][i] or "").split(",")]
        if "2.02" not in items:
            continue
        filed = cols["filingDate"][i]
        if filed < since:
            continue
        acc = cols.get("acceptanceDateTime", [None] * n)[i]
        rows.append((symbol, cols["accessionNumber"][i], filed, _utc_str(acc) if acc else None, cols["items"][i]))
    return rows


def sec_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(timeout=60, headers={"User-Agent": SEC_USER_AGENT}, follow_redirects=True, transport=transport)


def download_earnings(con: sqlite3.Connection, symbols: Sequence[str], *, http: httpx.Client | None = None,
                      min_interval: float = SEC_MIN_INTERVAL, sleep: Callable[[float], None] = time.sleep) -> None:
    """Earnings-release timestamps from SEC EDGAR (8-K item 2.02), including the older-filings overflow files."""
    own = http is None
    http = http or sec_client()
    last_call = [0.0]

    def get_json(url: str) -> dict:
        def fetch() -> dict:
            gap = min_interval - (time.monotonic() - last_call[0])
            if gap > 0:
                sleep(gap)
            last_call[0] = time.monotonic()
            r = http.get(url)
            r.raise_for_status()
            return r.json()
        return _with_backoff(fetch, "edgar", sleep)

    try:
        for sym in symbols:
            if sym in ETFS:
                log.info("earnings %s: ETF, skipped", sym)
                continue
            cik = CIKS.get(sym)
            if cik is None:
                raise ValueError(f"No CIK configured for {sym}; add it to download.CIKS.")
            sub = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            tickers = [t.upper() for t in sub.get("tickers", [])]
            name_ok = not tickers and sub.get("name", "").upper() == CIK_NAMES.get(sym)
            if name_ok:
                log.warning("earnings %s: EDGAR lists no tickers for CIK %s; matched by name %r", sym, cik, sub["name"])
            elif sym not in tickers:
                raise ValueError(f"CIK {cik} is not {sym}: EDGAR lists tickers {tickers}. Fix download.CIKS.")
            rows = parse_filings(sym, sub["filings"]["recent"])
            oldest = min(sub["filings"]["recent"]["filingDate"], default="")
            for f in sub["filings"].get("files", []):
                if f.get("filingTo", "9999") < EARNINGS_START:
                    continue
                rows += parse_filings(sym, get_json(f"https://data.sec.gov/submissions/{f['name']}"))
                oldest = min(oldest, f.get("filingFrom", oldest))
            con.executemany("INSERT OR REPLACE INTO earnings(symbol, accession, filing_date, accepted_at, items) "
                            "VALUES(?,?,?,?,?)", rows)
            con.commit()
            dates = sorted(r[2] for r in rows)
            log.info("earnings %s: %d releases (%s .. %s); filings covered back to %s", sym, len(rows),
                     dates[0] if dates else "-", dates[-1] if dates else "-", oldest)
    finally:
        if own:
            http.close()
