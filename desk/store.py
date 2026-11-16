"""SQLite persistence (stdlib sqlite3, one shared connection guarded by a lock)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, mode TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS feed (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, cycle_id INTEGER, symbol TEXT,
    agent TEXT, stance TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, cycle_id INTEGER, symbol TEXT,
    action TEXT, confidence REAL, size_pct REAL, stop REAL, target REAL, rationale TEXT, risk_status TEXT,
    risk_reasons TEXT, qty INTEGER, order_id TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, equity REAL, cash REAL, spy_price REAL);
CREATE TABLE IF NOT EXISTS llm_usage (ts TEXT, agent TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""
TABLES = ("cycles", "feed", "decisions", "equity", "llm_usage", "state")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def reset(self) -> None:
        with self._lock:
            for t in TABLES:
                self._db.execute(f"DELETE FROM {t}")
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid or 0

    def _rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # --- state ---
    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        rows = self._rows("SELECT value FROM state WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_state(self, key: str, value: str) -> None:
        self._exec("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, value))

    @property
    def kill_switch(self) -> bool:
        return self.get_state("kill_switch", "0") == "1"

    def set_kill_switch(self, on: bool) -> None:
        self.set_state("kill_switch", "1" if on else "0")

    @property
    def halted_date(self) -> Optional[str]:
        return self.get_state("halted_date")

    def set_halted(self, date: str) -> None:
        self.set_state("halted_date", date)

    # --- cycles ---
    def start_cycle(self, mode: str) -> int:
        return self._exec("INSERT INTO cycles(started_at,mode,status) VALUES(?,?,?)", (now_iso(), mode, "running"))

    def finish_cycle(self, cycle_id: int, status: str) -> None:
        self._exec("UPDATE cycles SET status=? WHERE id=?", (status, cycle_id))

    def last_cycle_at(self) -> Optional[str]:
        rows = self._rows("SELECT started_at FROM cycles ORDER BY id DESC LIMIT 1")
        return rows[0]["started_at"] if rows else None

    # --- feed ---
    def add_feed(self, cycle_id: Optional[int], symbol: Optional[str], agent: str, stance: str, text: str) -> int:
        return self._exec("INSERT INTO feed(ts,cycle_id,symbol,agent,stance,text) VALUES(?,?,?,?,?,?)",
                          (now_iso(), cycle_id, symbol, agent, stance, text))

    def get_feed(self, after_id: int = 0, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        if after_id > 0:
            return self._rows("SELECT * FROM feed WHERE id>? ORDER BY id ASC LIMIT ?", (after_id, limit))
        rows = self._rows("SELECT * FROM feed ORDER BY id DESC LIMIT ?", (limit,))
        return rows[::-1]

    # --- decisions ---
    def add_decision(self, *, cycle_id: int, symbol: str, action: str, confidence: float, size_pct: float,
                     stop: Optional[float], target: Optional[float], rationale: str, risk_status: str,
                     risk_reasons: list[str], qty: int, order_id: Optional[str]) -> int:
        return self._exec(
            "INSERT INTO decisions(ts,cycle_id,symbol,action,confidence,size_pct,stop,target,rationale,"
            "risk_status,risk_reasons,qty,order_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), cycle_id, symbol, action, confidence, size_pct, stop, target, rationale, risk_status,
             json.dumps(risk_reasons), qty, order_id))

    def get_decisions(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        rows = self._rows("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["risk_reasons"] = json.loads(r["risk_reasons"] or "[]")
        return rows

    def latest_buy_levels(self) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Most recent approved BUY (stop, target) per symbol; fallback for positions lacking broker levels."""
        out: dict[str, tuple[Optional[float], Optional[float]]] = {}
        for r in self._rows("SELECT symbol,stop,target FROM decisions WHERE action='BUY' AND risk_status='approved' "
                            "ORDER BY id ASC"):
            out[r["symbol"]] = (r["stop"], r["target"])
        return out

    # --- equity ---
    def add_equity(self, equity: float, cash: float, spy_price: Optional[float]) -> None:
        self._exec("INSERT INTO equity(ts,equity,cash,spy_price) VALUES(?,?,?,?)",
                   (now_iso(), equity, cash, spy_price))

    def get_equity(self, limit: int = 2000) -> list[dict]:
        rows = self._rows("SELECT * FROM equity ORDER BY rowid DESC LIMIT ?", (limit,))
        return rows[::-1]

    def last_equity_ts(self) -> Optional[str]:
        rows = self._rows("SELECT ts FROM equity ORDER BY rowid DESC LIMIT 1")
        return rows[0]["ts"] if rows else None

    # --- LLM usage ---
    def add_usage(self, agent: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self._exec("INSERT INTO llm_usage(ts,agent,model,input_tokens,output_tokens,cost_usd) VALUES(?,?,?,?,?,?)",
                   (now_iso(), agent, model, input_tokens, output_tokens, cost_usd))

    def calls_today(self) -> int:
        """LLM calls made in the current UTC day (plan users watch this against their usage limits)."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._rows("SELECT COUNT(*) AS n FROM llm_usage WHERE substr(ts,1,10)=?", (day,))
        return int(rows[0]["n"])

    def spend_today(self) -> float:
        """Estimated LLM spend for the current UTC day."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._rows("SELECT COALESCE(SUM(cost_usd),0) AS s FROM llm_usage WHERE substr(ts,1,10)=?", (day,))
        return float(rows[0]["s"])
