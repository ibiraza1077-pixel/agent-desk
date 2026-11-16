"""asyncio scheduler: runs desk.tick() every cycle interval in a worker thread."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from .desk import Desk

log = logging.getLogger("desk.scheduler")


class Scheduler:
    def __init__(self, desk: Desk, interval_s: float):
        self.desk = desk
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        from .agents.base import SHUTDOWN
        SHUTDOWN.clear()
        self.desk.running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        from .agents.base import SHUTDOWN
        SHUTDOWN.set()  # no new LLM requests; an in-flight cycle winds down after its current call
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.desk.running = False

    async def _loop(self) -> None:
        due = time.monotonic()  # first beat immediately
        while not self._stop.is_set():
            if time.monotonic() >= due:
                try:
                    await asyncio.to_thread(self.desk.tick)
                except Exception as e:  # noqa: BLE001 - never let the loop die
                    log.exception("tick failed")
                    self.desk.system(f"Scheduler error: {type(e).__name__}: {e}")
                due = time.monotonic() + self.interval_s
                nxt = datetime.now(timezone.utc) + timedelta(seconds=self.interval_s)
                self.desk.next_cycle_at = nxt.isoformat(timespec="seconds")
            await asyncio.sleep(0.5)
