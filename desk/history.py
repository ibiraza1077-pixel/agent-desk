"""What 10 years of history say, in a form the agents can quote: honest base rates and the market regime.

Sources: data/research/base_rates.json and results.json (written by `python -m desk.research report`) and the
latest VIX close from FRED (free, cached for 6 hours). Everything here is descriptive context, never an order
signal: the walk-forward study found no stock-picking signal that reliably beat the base rate out of sample.
If the study hasn't been run, or FRED can't be reached, the block simply omits that part.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import httpx

from .config import PROJECT_DIR

log = logging.getLogger("desk.history")
RESEARCH_DIR = PROJECT_DIR / "data" / "research"
FRED_VIX = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd={start}"


def _bucket(rows: list[dict], value: Optional[float]) -> Optional[dict]:
    if value is None:
        return None
    for r in rows:
        if r["lo"] < value <= r["hi"]:
            return r
    return None


class History:
    def __init__(self, research_dir: Path = RESEARCH_DIR, fetch: Optional[Callable[[str], str]] = None,
                 clock: Callable[[], float] = time.monotonic, vix_ttl_s: float = 6 * 3600):
        self.dir = Path(research_dir)
        self._fetch = fetch or self._http_get
        self._clock = clock
        self._vix_ttl = vix_ttl_s
        self._vix: Optional[tuple[float, Optional[float], str]] = None  # (fetched_at, value, date)
        self._files: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _http_get(url: str) -> str:
        r = httpx.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        return r.text

    def _json(self, name: str) -> dict:
        p = self.dir / name
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return {}
        hit = self._files.get(name)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            data = {}
        self._files[name] = (mtime, data)
        return data

    FAIL_TTL_S = 300.0  # retry a failed FRED fetch after 5 minutes, not 6 hours

    def vix(self) -> tuple[Optional[float], str]:
        """Latest VIX close and its date (cached; (None, '') if FRED is unreachable)."""
        with self._lock:
            if self._vix:
                ttl = self._vix_ttl if self._vix[1] is not None else self.FAIL_TTL_S
                if self._clock() - self._vix[0] < ttl:
                    return self._vix[1], self._vix[2]
        value, day = None, ""  # fetch outside the lock: a slow FRED must not block other callers
        try:
            text = self._fetch(FRED_VIX.format(start=(date.today() - timedelta(days=30)).isoformat()))
            for row in csv.reader(io.StringIO(text)):
                if len(row) == 2 and row[1] not in ("", ".") and row[0][:2].isdigit():
                    try:
                        value, day = float(row[1]), row[0]
                    except ValueError:
                        continue
        except Exception as e:  # noqa: BLE001 - context only; never block a cycle
            log.warning("VIX fetch failed: %s", e)
        with self._lock:
            self._vix = (self._clock(), value, day)
        return value, day

    def context(self, symbol: str, screen_score: Optional[float], index_buy_min_vix: float = 30.0) -> Optional[dict]:
        rates, results = self._json("base_rates.json"), self._json("results.json")
        if not rates:
            return None
        overall = rates.get("overall", {})
        out: dict = {
            "what_this_is": (f"Descriptive history {'–'.join(rates.get('period', []))}: how often a watchlist stock "
                             f"beat simply holding SPY over the next {rates.get('horizon_days', 10)} trading days. "
                             "Not a forecast."),
            "base_rate_beat_spy": round(overall.get("beat_spy", 0.0), 3),
            "study_verdict": (f"Walk-forward test {results['test_years'][0]}–{results['test_years'][-1]}: "
                              + (f"model {results['chosen_model']} showed out-of-sample skill."
                                 if results.get("chosen_model") else
                                 "no signal tested (Scout score, technicals, news sentiment, earnings, macro) "
                                 "beat this base rate reliably out of sample.")
                              if results.get("test_years") else "study not run"),
        }
        b = _bucket(rates.get("by_screen_score", []), screen_score)
        if b:
            out["when_scout_score_similar"] = {"beat_spy": round(b["beat_spy"], 3), "n": b["n"]}
        sym = rates.get("by_symbol", {}).get(symbol)
        if sym:
            out["this_symbol"] = {"beat_spy": round(sym["beat_spy"], 3), "n": sym["n"],
                                  "caveat": "survivorship-biased: today's watchlist names were past winners"}
        v, day = self.vix()
        if v is not None:
            reg = {"vix": round(v, 2), "as_of": day}
            vb = _bucket(rates.get("spy_by_vix", []), v)
            if vb:
                reg["spy_next_10d_when_vix_similar"] = {"avg_return": round(vb["spy_avg_10d"], 4),
                                                        "up_share": round(vb["spy_up_share"], 3)}
            timing = rates.get("vix_timing")
            if timing:
                reg["study_finding"] = timing.get("summary", "")
            if index_buy_min_vix > 0:
                reg["index_buy_window_open"] = v >= index_buy_min_vix
                reg["index_rule"] = (f"SPY may only be bought when VIX >= {index_buy_min_vix:g} (enforced by the Risk "
                                     "Guard). The historical edge is over about 10 trading days after the fear spike.")
            out["market_regime"] = reg
        return out
