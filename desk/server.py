"""FastAPI app: JSON API, background scheduler, static dashboard. Bound to 127.0.0.1 by __main__."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from urllib.parse import parse_qs, urlparse

from .agents.scripted import ScriptedTeam
from .config import Settings, load_settings
from .desk import Desk
from .scheduler import Scheduler
from .store import Store

log = logging.getLogger("desk.server")
STATIC = Path(__file__).parent / "static"


def build_desk(settings: Settings) -> Desk:
    """Wire store + broker + agents for the configured mode."""
    store = Store(settings.db_path)
    prev_mode = store.get_state("db_mode")
    if prev_mode and prev_mode != settings.mode:
        raise RuntimeError(
            f"{settings.db_path} was used in {prev_mode!r} mode; refusing to mix with {settings.mode!r}. "
            "Point DESK_DB at a different file."
        )
    if settings.is_demo:
        store.reset()  # the simulated account is in-memory, so demo history starts fresh every run
    store.set_state("db_mode", settings.mode)

    if settings.uses_llm:
        from .agents.llm import LLMTeam
        from .agents.providers import make_caller
        team = LLMTeam(settings, make_caller(settings, store))
    else:
        team = ScriptedTeam(settings)
    if settings.is_demo:
        from .broker.sim import SimBroker
        broker = SimBroker(settings.watchlist, seed=settings.demo_seed)
    else:
        from .broker.alpaca_paper import AlpacaPaperBroker
        broker = AlpacaPaperBroker(settings.alpaca_api_key, settings.alpaca_secret_key)
    history = None
    if not settings.is_demo:  # the study and VIX describe the real market, not the simulator
        from .history import History
        history = History()
    return Desk(settings, store, broker, team, history=history)


def create_app(settings: Optional[Settings] = None, desk: Optional[Desk] = None,
               extra_allowed_hosts: tuple[str, ...] = ()) -> FastAPI:
    """`extra_allowed_hosts` exists for tests (TestClient sends Host: testserver); never set it in production."""
    settings = settings or load_settings()
    desk = desk or build_desk(settings)
    scheduler = Scheduler(desk, settings.cycle_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.autostart:
            scheduler.start()
        yield
        await scheduler.stop()

    # No interactive API docs: nothing needs them, and they'd be one more page reachable on the network.
    app = FastAPI(title="Agent Desk", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.desk = desk
    store = desk.store
    allowed = {f"127.0.0.1:{settings.port}", f"localhost:{settings.port}", *extra_allowed_hosts}
    key = None
    if settings.phone_access:
        from .access import FailureLimiter, access_key, lan_addresses, local_hostname
        key = access_key()
        limiter = FailureLimiter()
        allowed |= {f"{ip}:{settings.port}" for ip in lan_addresses()}
        allowed.add(f"{local_hostname()}:{settings.port}".lower())
        app.state.phone_urls = [f"http://{ip}:{settings.port}/" for ip in lan_addresses()]
        app.state.access_key = key

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # DNS-rebinding defence: only our own names/addresses are accepted as Host, on every route.
        if request.headers.get("host", "").lower() not in allowed:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        if key is not None:
            from .access import COOKIE, COOKIE_MAX_AGE, PUBLIC_PATHS, cookie_ok, cookie_value, key_ok, unlock_page
            client = request.client.host if request.client else ""
            # Loopback skips the key, except when a local tunnel/proxy forwarded the request on someone's behalf.
            local = client in ("127.0.0.1", "::1") and "x-forwarded-for" not in request.headers \
                and "forwarded" not in request.headers
            path = request.url.path
            signed_in = local or cookie_ok(request.cookies.get(COOKIE), key)
            nostore = {"Cache-Control": "no-store"}  # never let a browser re-show a stale unlock page

            def page(msg: str = "", status: int = 401) -> HTMLResponse:
                return HTMLResponse(unlock_page(msg), status_code=status, headers=nostore)

            def unlocked() -> RedirectResponse:
                resp = RedirectResponse("/", status_code=303, headers=nostore)  # also drops ?key= from the URL
                resp.set_cookie(COOKIE, cookie_value(key), max_age=COOKIE_MAX_AGE, httponly=True, samesite="strict")
                return resp

            # count failures per real peer; only trust X-Forwarded-For when the peer is a local proxy/tunnel
            ip = client
            if client in ("127.0.0.1", "::1") and request.headers.get("x-forwarded-for"):
                ip = request.headers["x-forwarded-for"].split(",")[0].strip() or client

            if path == "/unlock":
                # Always handled here (signed in or not), so a cached unlock form can never fall through to the
                # API's POST guard.
                if request.method != "POST":
                    return RedirectResponse("/", status_code=303, headers=nostore)
                origin = request.headers.get("origin")
                if origin not in (None, "null") and urlparse(origin).netloc not in allowed:
                    return JSONResponse({"error": "forbidden origin"}, status_code=403)
                if limiter.blocked(ip):
                    return page("Too many wrong keys. Try again in 10 minutes.", 429)
                body = (await request.body())[:2048].decode("utf-8", "replace")
                given = (parse_qs(body).get("key") or [""])[0].strip()
                if key_ok(given, key):
                    limiter.reset(ip)
                    return unlocked()
                if signed_in:
                    return RedirectResponse("/", status_code=303, headers=nostore)
                limiter.fail(ip)
                return page("That key is not right.")

            given = request.query_params.get("key") if request.method == "GET" and path == "/" else None
            if given is not None:
                if limiter.blocked(ip):
                    return page("Too many wrong keys. Try again in 10 minutes.", 429)
                if key_ok(given, key):
                    limiter.reset(ip)
                    return unlocked()
                if not signed_in:
                    limiter.fail(ip)
                    return page("That key is not right.")
                return RedirectResponse("/", status_code=303, headers=nostore)

            if not signed_in and path not in PUBLIC_PATHS:
                if limiter.blocked(ip):
                    return page("Too many wrong keys. Try again in 10 minutes.", 429)
                return page()
        if request.method == "POST":
            # CSRF defence: browsers cannot send a custom header cross-site without a CORS preflight (which we
            # never approve), and a foreign Origin is refused outright.
            origin = request.headers.get("origin")
            if origin is not None and urlparse(origin).netloc not in allowed:
                return JSONResponse({"error": "forbidden origin"}, status_code=403)
            if request.headers.get("x-desk") != "1":
                return JSONResponse({"error": "missing X-Desk header"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/apple-touch-icon.png")
    def touch_icon():
        return FileResponse(STATIC / "apple-touch-icon.png", media_type="image/png")

    @app.get("/icon.svg")
    def icon():
        return FileResponse(STATIC / "icon.svg", media_type="image/svg+xml")

    @app.get("/api/state")
    def state():
        return desk.state()

    @app.get("/api/feed")
    def feed(after_id: int = 0, limit: int = Query(100, ge=1, le=500)):
        return store.get_feed(after_id, limit)

    @app.get("/api/decisions")
    def decisions(limit: int = Query(50, ge=1, le=500)):
        return store.get_decisions(limit)

    @app.get("/api/positions")
    def positions():
        return desk.positions_view()

    @app.get("/api/watchlist")
    def watchlist():
        return desk.watchlist_view()

    @app.get("/api/research")
    def research():
        from .research_view import research_view
        return research_view()

    @app.get("/api/equity")
    def equity():
        return desk.equity_series()

    @app.post("/api/kill")
    def kill():
        store.set_kill_switch(True)
        desk.system("KILL SWITCH ON: new buys are blocked. Sells and broker-side stops still work.")
        return {"kill_switch": True}

    @app.post("/api/resume")
    def resume():
        store.set_kill_switch(False)
        desk.system("Kill switch cleared: buying may resume.")
        return {"kill_switch": False}

    @app.post("/api/flatten")
    def flatten():
        try:
            return {"closed": desk.flatten()}
        except Exception as e:  # noqa: BLE001
            desk.system(f"Flatten failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=502)

    @app.post("/api/run-cycle")
    def run_cycle():
        if desk.lock.locked():
            return JSONResponse({"started": False, "reason": "a cycle is already running"}, status_code=409)
        if not settings.is_demo and not desk.broker.is_market_open():
            return JSONResponse({"started": False, "reason": "market is closed"}, status_code=409)
        threading.Thread(target=desk.run_cycle, daemon=True, name="manual-cycle").start()
        return {"started": True}

    return app
