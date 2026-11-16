"""Phone access: serve the dashboard on the home network behind a secret access key.

Off by default (the dashboard has Kill/Flatten buttons and no login). With DESK_PHONE_ACCESS=1 the server listens
on all interfaces, and every request from another device must carry the key: once via ?key=... (which sets an
HttpOnly cookie and redirects), afterwards via the cookie. Requests from this Mac itself (loopback) need no key.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import socket
import threading
import time
from pathlib import Path

from .config import PROJECT_DIR

KEY_FILE = PROJECT_DIR / "data" / ".desk_access_key"
COOKIE = "desk_key"
MIN_KEY_LEN = 16
COOKIE_MAX_AGE = 90 * 86400
# Static, non-sensitive files the browser fetches WITHOUT cookies (web manifest, home-screen icons).
PUBLIC_PATHS = {"/manifest.webmanifest", "/icon.svg", "/apple-touch-icon.png"}


def access_key(path: Path = KEY_FILE) -> str:
    """DESK_ACCESS_KEY if set, else a random key persisted (mode 600) so the phone stays signed in across restarts."""
    env = os.environ.get("DESK_ACCESS_KEY", "").strip()
    if env:
        if len(env) < MIN_KEY_LEN:
            raise RuntimeError(f"DESK_ACCESS_KEY must be at least {MIN_KEY_LEN} characters (it guards the Kill/Flatten "
                               "controls on your network). Remove it to use a generated key.")
        return env
    try:
        k = path.read_text().strip()
        if len(k) >= 16:
            return k
    except OSError:
        pass
    k = secrets.token_urlsafe(18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(k)
    return k


def key_ok(given: str | None, key: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), key.encode())


def cookie_value(key: str) -> str:
    """The cookie holds a keyed hash, not the key itself, so a leaked cookie isn't the printed link."""
    return hmac.new(key.encode(), b"agent-desk-cookie-v1", hashlib.sha256).hexdigest()


def cookie_ok(given: str | None, key: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), cookie_value(key).encode())


class FailureLimiter:
    """Slow down key guessing: after 5 wrong keys from one address in 10 minutes, refuse it for 10 minutes."""

    def __init__(self, limit: int = 5, window_s: float = 600.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_s, clock
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, ip: str) -> bool:
        with self._lock:
            now = self.clock()
            recent = [t for t in self._fails.get(ip, []) if now - t < self.window]
            self._fails[ip] = recent
            return len(recent) >= self.limit

    def fail(self, ip: str) -> None:
        with self._lock:
            self._fails.setdefault(ip, []).append(self.clock())

    def reset(self, ip: str) -> None:
        with self._lock:
            self._fails.pop(ip, None)


UNLOCK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#15141a"><title>Agent Desk</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#15141a;color:#f1f0f5;
font:16px/1.4 -apple-system,BlinkMacSystemFont,sans-serif;padding:24px}form{width:min(360px,100%);background:#1b1a21;
border:1px solid #2a2932;border-radius:22px;padding:24px}h1{font-size:22px;margin:0 0 6px}p{color:#8d8b99;margin:0 0 16px;
font-size:14px}input{width:100%;box-sizing:border-box;font:16px ui-monospace,Menlo,monospace;padding:12px;border-radius:12px;
border:1px solid #3a3944;background:#121117;color:#f1f0f5}button{margin-top:12px;width:100%;padding:13px;border:0;
border-radius:99px;background:#b6a3f7;color:#15141a;font-weight:700;font-size:16px}.e{color:#ff6f8a}</style></head>
<body><form method="post" action="/unlock"><h1>Agent Desk</h1>
<p>Enter the access key from the link the desk printed in its terminal (the part after <b>key=</b>).</p>__MSG__
<input name="key" autocomplete="current-password" autocapitalize="off" spellcheck="false" placeholder="access key" required>
<button type="submit">Unlock</button></form></body></html>"""


def unlock_page(msg: str = "") -> str:
    return UNLOCK_PAGE.replace("__MSG__", f'<p class="e">{msg}</p>' if msg else "")


def lan_addresses() -> list[str]:
    """This machine's private IPv4 addresses (for the Host check and the printed phone link)."""
    ips: set[str] = set()
    try:  # the interface used for outbound traffic; UDP connect sends no packets
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if _private(ip))


def _private(ip: str) -> bool:
    a = [int(x) for x in ip.split(".")]
    return a[0] == 10 or (a[0] == 172 and 16 <= a[1] <= 31) or (a[0] == 192 and a[1] == 168)


def local_hostname() -> str:
    name = socket.gethostname()
    return name if name.endswith(".local") else name + ".local"
