"""Read-only summary of the historical study and the background data download, for the dashboard."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .config import PROJECT_DIR

RESEARCH_DIR = PROJECT_DIR / "data" / "research"
HISTORY_DB = Path(os.environ.get("HISTORY_DB", PROJECT_DIR / "data" / "history.db"))
_cache: dict[str, tuple[float, dict]] = {}


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _history_counts() -> dict:
    if not HISTORY_DB.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True, timeout=2)
        try:
            q = lambda s: con.execute(s).fetchone()  # noqa: E731
            return {
                "bars": q("SELECT COUNT(*) FROM bars")[0],
                "bars_range": q("SELECT MIN(date), MAX(date) FROM bars"),
                "news": q("SELECT COUNT(*) FROM news")[0],
                "news_range": q("SELECT MIN(created_at), MAX(created_at) FROM news"),
                "news_scored": q("SELECT COUNT(*) FROM news_sentiment")[0],
                "macro": q("SELECT COUNT(*) FROM macro")[0],
                "earnings": q("SELECT COUNT(*) FROM earnings")[0],
            }
        finally:
            con.close()
    except sqlite3.Error:
        return {}


def research_view(ttl_s: float = 30.0) -> dict:
    hit = _cache.get("v")
    if hit and time.monotonic() - hit[0] < ttl_s:
        return hit[1]
    results = _json(RESEARCH_DIR / "results.json") or {}
    rates = _json(RESEARCH_DIR / "base_rates.json") or {}
    view = {
        "has_study": bool(results),
        "generated_at": results.get("generated_at"),
        "test_years": results.get("test_years"),
        "chosen_model": results.get("chosen_model"),
        "base_rate_beat_spy": results.get("base_rate_beat_spy"),
        "skill": results.get("skill"),
        "highlights": results.get("highlights") or [],
        "by_screen_score": rates.get("by_screen_score") or [],
        "by_symbol": rates.get("by_symbol") or {},
        "spy_by_vix": rates.get("spy_by_vix") or [],
        "vix_timing": rates.get("vix_timing"),
        "history": _history_counts(),
    }
    _cache["v"] = (time.monotonic(), view)
    return view
