"""Phone access: the key gate in front of every route when DESK_PHONE_ACCESS=1."""
import dataclasses

from fastapi.testclient import TestClient

from desk import access
from desk.server import build_desk, create_app


def _app(settings, monkeypatch, tmp_path, key="k" * 24):
    monkeypatch.setenv("DESK_ACCESS_KEY", key)
    s = dataclasses.replace(settings, phone_access=True, autostart=False)
    return create_app(s, build_desk(s), extra_allowed_hosts=("testserver",))


def test_other_devices_need_the_key(settings, monkeypatch, tmp_path):
    app = _app(settings, monkeypatch, tmp_path)
    c = TestClient(app)  # TestClient's client address is not loopback
    assert c.get("/api/state").status_code == 401
    assert c.get("/").status_code == 401
    assert c.get("/?key=wrong", follow_redirects=False).status_code == 401
    r = c.get("/?key=" + "k" * 24, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert c.get("/api/state").status_code == 200          # cookie now set
    assert c.post("/api/kill").status_code == 403           # POST still needs the X-Desk header
    assert c.post("/api/kill", headers={"X-Desk": "1"}).status_code == 200


def test_cookie_with_wrong_key_is_refused(settings, monkeypatch, tmp_path):
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    c.cookies.set(access.COOKIE, "nope")
    assert c.get("/api/state").status_code == 401
    c.cookies.set(access.COOKIE, "k" * 24)  # the raw key is not a valid cookie: only its keyed hash is
    assert c.get("/api/state").status_code == 401


def test_unlock_form_for_home_screen_app(settings, monkeypatch, tmp_path):
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    page = c.get("/")
    assert page.status_code == 401 and 'action="/unlock"' in page.text
    r = c.post("/unlock", data={"key": "k" * 24}, follow_redirects=False)
    assert r.status_code == 303 and "k" * 24 not in r.headers["set-cookie"]
    assert c.get("/api/state").status_code == 200


def test_wrong_keys_are_rate_limited(settings, monkeypatch, tmp_path):
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    for _ in range(5):
        assert c.post("/unlock", data={"key": "guess"}).status_code == 401
    assert c.post("/unlock", data={"key": "k" * 24}).status_code == 429  # even the right key, while locked out


def test_manifest_and_icons_are_public_but_nothing_else(settings, monkeypatch, tmp_path):
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    for path in ("/manifest.webmanifest", "/icon.svg", "/apple-touch-icon.png"):
        assert c.get(path).status_code == 200, path
    for path in ("/docs", "/openapi.json", "/api/feed"):
        assert c.get(path).status_code in (401, 404), path


def test_short_custom_key_is_refused(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setenv("DESK_ACCESS_KEY", "1234")
    with pytest.raises(RuntimeError, match="at least"):
        access.access_key(tmp_path / ".k")


def test_loopback_needs_no_key_and_unknown_hosts_are_refused(settings, monkeypatch, tmp_path):
    app = _app(settings, monkeypatch, tmp_path)
    c = TestClient(app, client=("127.0.0.1", 50000))
    assert c.get("/api/state").status_code == 200
    # a local tunnel/proxy forwarding someone else's request does not get the loopback pass
    assert c.get("/api/state", headers={"x-forwarded-for": "203.0.113.9"}).status_code == 401
    assert c.get("/api/state", headers={"host": "evil.example:8000"}).status_code == 403


def test_off_by_default(settings):
    assert not settings.phone_access and settings.host == "127.0.0.1"


def test_generated_key_is_private_and_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("DESK_ACCESS_KEY", raising=False)
    p = tmp_path / ".k"
    k1 = access.access_key(p)
    assert len(k1) >= 20 and access.access_key(p) == k1
    assert (p.stat().st_mode & 0o777) == 0o600


def test_stale_unlock_form_while_signed_in_still_works(settings, monkeypatch, tmp_path):
    """Regression: a cached unlock page submitted by an already signed-in phone must not hit the API's POST guard."""
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    assert c.post("/unlock", data={"key": "k" * 24}, follow_redirects=False).status_code == 303   # signs in
    r = c.post("/unlock", data={"key": "k" * 24}, follow_redirects=False)                        # stale form again
    assert r.status_code == 303 and r.headers["location"] == "/"
    r = c.post("/unlock", data={"key": "typo"}, follow_redirects=False)                          # signed in + typo
    assert r.status_code == 303
    assert c.get("/unlock", follow_redirects=False).status_code == 303
    page = TestClient(_app(settings, monkeypatch, tmp_path)).get("/")
    assert page.status_code == 401 and page.headers["cache-control"] == "no-store"


def test_key_link_when_already_signed_in_just_cleans_the_url(settings, monkeypatch, tmp_path):
    c = TestClient(_app(settings, monkeypatch, tmp_path))
    c.get("/?key=" + "k" * 24)
    r = c.get("/?key=" + "k" * 24, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
