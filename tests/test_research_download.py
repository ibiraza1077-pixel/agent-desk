"""Offline tests for the research downloaders: fake Alpaca clients, httpx.MockTransport, temp database."""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from desk.config import load_settings
from desk.research import download
from desk.research.db import connect, get_progress


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "h.db")
    yield c
    c.close()


@pytest.fixture
def cfg():
    return load_settings({"DESK_MODE": "demo", "LLM_PROVIDER": "anthropic"}, use_dotenv=False)


def mock_http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------------------------------------ macro

def test_fred_csv_skips_missing_and_old(con):
    csv_text = ("observation_date,VIXCLS\n2013-12-31,20.0\n2014-01-02,14.23\n2014-01-03,.\n2014-01-06,\n2014-01-07,13.5\n")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.params["id"])
        return httpx.Response(200, text=csv_text)

    download.download_macro(con, ["VIXCLS"], http=mock_http(handler))
    assert seen == ["VIXCLS"]
    assert con.execute("SELECT date, value FROM macro ORDER BY date").fetchall() == [("2014-01-02", 14.23), ("2014-01-07", 13.5)]
    download.download_macro(con, ["VIXCLS"], http=mock_http(handler))  # idempotent
    assert con.execute("SELECT COUNT(*) FROM macro").fetchone()[0] == 2


# ------------------------------------------------------------------------------------------------ earnings

def _submissions(tickers, recent, files=()):
    return {"tickers": tickers, "filings": {"recent": recent, "files": list(files)}}


RECENT = {
    "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
    "filingDate": ["2024-08-01", "2024-07-15", "2024-05-02", "2024-05-02", "2014-12-30"],
    "acceptanceDateTime": ["2024-08-01T16:30:42.000Z", "2024-07-15T10:00:00.000Z", "2024-05-02T20:31:05.000Z",
                           "2024-05-02T20:40:00.000Z", "2014-12-30T21:00:00.000Z"],
    "form": ["8-K", "8-K", "8-K/A", "10-Q", "8-K"],
    "items": ["2.02,9.01", "5.02", "2.02", "", "2.02"],
}
OLDER = {
    "accessionNumber": ["o1", "o2"], "filingDate": ["2016-02-01", "2015-01-20"],
    "acceptanceDateTime": ["2016-02-01T21:15:00.000Z", "2015-01-20T21:30:00-05:00"],
    "form": ["8-K", "8-K"], "items": ["2.02,9.01", "2.02"],
}


def test_earnings_filters_and_normalises(con):
    paths = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        assert req.headers["user-agent"] == download.SEC_USER_AGENT
        if req.url.path == "/submissions/CIK0000320193.json":
            return httpx.Response(200, json=_submissions(["AAPL"], RECENT, [{"name": "CIK0000320193-submissions-001.json",
                                                                              "filingFrom": "2010-01-01", "filingTo": "2023-12-31"},
                                                                             {"name": "ancient.json", "filingFrom": "1996-01-01",
                                                                              "filingTo": "2009-12-31"}]))
        if req.url.path == "/submissions/CIK0000320193-submissions-001.json":
            return httpx.Response(200, json=OLDER)
        return httpx.Response(404)

    download.download_earnings(con, ["SPY", "AAPL"], http=download.sec_client(httpx.MockTransport(handler)), min_interval=0)
    assert "/submissions/ancient.json" not in paths  # entirely before 2015: not fetched
    rows = con.execute("SELECT symbol, accession, filing_date, accepted_at, items FROM earnings ORDER BY filing_date").fetchall()
    assert rows == [
        ("AAPL", "o2", "2015-01-20", "2015-01-21T02:30:00Z", "2.02"),
        ("AAPL", "o1", "2016-02-01", "2016-02-01T21:15:00Z", "2.02,9.01"),
        ("AAPL", "a1", "2024-08-01", "2024-08-01T16:30:42Z", "2.02,9.01"),
    ]


def test_earnings_rejects_cik_ticker_mismatch(con):
    http = download.sec_client(httpx.MockTransport(lambda req: httpx.Response(200, json=_submissions(["MSFT"], RECENT))))
    with pytest.raises(ValueError, match="not AAPL"):
        download.download_earnings(con, ["AAPL"], http=http, min_interval=0)


# ------------------------------------------------------------------------------------------------ news

def art(i, created, symbols):
    return {"id": i, "created_at": created, "updated_at": created, "headline": f"h{i}", "summary": f"s{i}",
            "source": "benzinga", "author": "a", "symbols": symbols, "url": None, "content": ""}


class FakeNews:
    """Serves pre-made pages per symbol; page tokens are 'p<N>'. Records each request's params."""

    def __init__(self, pages: dict[str, list[list[dict]]], fail_429: int = 0):
        self.pages, self.calls, self.fail_429 = pages, [], fail_429

    def get(self, path, data=None):
        assert path == "/news"
        self.calls.append(dict(data))
        if self.fail_429:
            self.fail_429 -= 1
            raise SimpleNamespace_error(429)
        pages = self.pages[data["symbols"]]
        n = int(data["page_token"][1:]) if data.get("page_token") else 0
        out = {"news": pages[n]}
        if n + 1 < len(pages):
            out["next_page_token"] = f"p{n + 1}"
        return out


class SimpleNamespace_error(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


NEWS_KW = dict(requests_per_minute=0, sleep=lambda s: None)


def test_news_pagination_tagging_dedup(con, cfg):
    shared = art(2, "2020-01-01T10:00:00Z", ["AAPL", "MSFT", "ZZZZ"])  # ZZZZ is not on the watchlist
    fake = FakeNews({
        "AAPL": [[art(1, "2020-01-01T09:00:00Z", ["AAPL"]), shared], [art(3, "2020-01-02T09:00:00.500Z", ["AAPL", "SPY"])]],
        "MSFT": [[shared]],
    })
    download.download_news(con, cfg, ["AAPL", "MSFT"], client=fake, **NEWS_KW)
    assert con.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 3  # article 2 stored once
    tags = con.execute("SELECT id, symbol FROM news_symbol ORDER BY id, symbol").fetchall()
    assert tags == [(1, "AAPL"), (2, "AAPL"), (2, "MSFT"), (3, "AAPL"), (3, "SPY")]
    assert con.execute("SELECT created_at FROM news WHERE id=3").fetchone()[0] == "2020-01-02T09:00:00Z"
    assert get_progress(con, "news:AAPL") == "2020-01-02T09:00:00Z"
    assert get_progress(con, "news:MSFT") == "2020-01-01T10:00:00Z"
    first = fake.calls[0]
    assert first["sort"] == "asc" and first["limit"] == 50 and first["symbols"] == "AAPL"
    assert fake.calls[1]["page_token"] == "p1"


def test_news_resumes_from_progress(con, cfg):
    fake = FakeNews({"AAPL": [[art(1, "2020-01-01T09:00:00Z", ["AAPL"])]]})
    download.download_news(con, cfg, ["AAPL"], client=fake, **NEWS_KW)
    fake2 = FakeNews({"AAPL": [[art(1, "2020-01-01T09:00:00Z", ["AAPL"]), art(4, "2020-03-01T00:00:00Z", ["AAPL"])]]})
    download.download_news(con, cfg, ["AAPL"], client=fake2, **NEWS_KW)
    assert "2020-01-01" in str(fake2.calls[0]["start"])  # restarted at the stored progress, not at 2015
    assert con.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 2
    assert get_progress(con, "news:AAPL") == "2020-03-01T00:00:00Z"


def test_news_retries_on_429(con, cfg):
    sleeps: list[float] = []
    fake = FakeNews({"AAPL": [[art(1, "2020-01-01T09:00:00Z", ["AAPL"])]]}, fail_429=2)
    download.download_news(con, cfg, ["AAPL"], client=fake, requests_per_minute=0, sleep=sleeps.append)
    assert sleeps == [2.0, 4.0]
    assert con.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 1


def test_news_gives_up_after_max_tries(con, cfg):
    fake = FakeNews({"AAPL": [[]]}, fail_429=99)
    with pytest.raises(SimpleNamespace_error):
        download.download_news(con, cfg, ["AAPL"], client=fake, **NEWS_KW)
    assert len(fake.calls) == download.MAX_TRIES


# ------------------------------------------------------------------------------------------------ bars

def test_bars_use_new_york_date(con, cfg):
    def bar(ts, close):
        return SimpleNamespace(timestamp=ts, open=close - 1, high=close + 1, low=close - 2, close=close, volume=100.0, vwap=close - 0.5)

    class FakeBars:
        def get_stock_bars(self, req):
            assert req.symbol_or_symbols in ("AAPL", "NOPE") and req.feed.value == "sip" and req.adjustment.value == "all"
            return SimpleNamespace(data={"AAPL": [
                bar(datetime(2016, 1, 4, 5, 0, tzinfo=UTC), 10.0),   # midnight EST
                bar(datetime(2016, 7, 1, 4, 0, tzinfo=UTC), 11.0),   # midnight EDT
                bar(datetime(2016, 7, 5, 1, 0, tzinfo=UTC), 12.0),   # 21:00 on the 4th in New York
            ]})

    download.download_bars(con, cfg, ["AAPL", "NOPE"], client=FakeBars())
    rows = con.execute("SELECT date, close, vwap FROM bars WHERE symbol='AAPL' ORDER BY date").fetchall()
    assert rows == [("2016-01-04", 10.0, 9.5), ("2016-07-01", 11.0, 10.5), ("2016-07-04", 12.0, 11.5)]
