import time

import pytest
from fastapi.testclient import TestClient

from desk.server import build_desk, create_app


@pytest.fixture
def client(settings):
    desk = build_desk(settings)
    assert settings.autostart is False  # no background loop in tests
    app = create_app(settings, desk, extra_allowed_hosts=("testserver",))
    c = TestClient(app)
    c.headers.update({"X-Desk": "1"})
    return c, desk


def test_state_feed_and_pages(client):
    c, desk = client
    desk.run_cycle()
    s = c.get("/api/state").json()
    for k in ("mode", "running", "market_open", "kill_switch", "halted_today", "equity", "cash", "day_pnl",
              "day_pnl_pct", "open_positions", "llm_spend_today", "llm_budget", "last_cycle_at", "next_cycle_at", "risk"):
        assert k in s
    assert s["mode"] == "demo" and s["running"] is False and s["equity"] > 0
    feed = c.get("/api/feed?limit=5").json()
    assert 0 < len(feed) <= 5
    later = c.get(f"/api/feed?after_id={feed[0]['id']}").json()
    assert all(f["id"] > feed[0]["id"] for f in later)
    assert isinstance(c.get("/api/decisions").json(), list)
    assert isinstance(c.get("/api/positions").json(), list)
    assert len(c.get("/api/equity").json()) == 1
    html = c.get("/").text
    assert "Agent Desk" in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")


def test_kill_resume(client):
    c, desk = client
    assert c.post("/api/kill").json() == {"kill_switch": True}
    assert c.get("/api/state").json()["kill_switch"] is True
    assert desk.store.kill_switch
    assert c.post("/api/resume").json() == {"kill_switch": False}
    assert c.get("/api/state").json()["kill_switch"] is False


def test_run_cycle_and_flatten(client):
    c, desk = client
    assert c.post("/api/run-cycle").json() == {"started": True}
    for _ in range(100):
        if not desk.lock.locked() and desk.store.last_cycle_at():
            break
        time.sleep(0.05)
    assert desk.store.get_feed()
    for _ in range(8):
        desk.run_cycle()
    r = c.post("/api/flatten")
    assert r.status_code == 200 and c.get("/api/positions").json() == []
    held = desk.lock.acquire()
    try:
        assert c.post("/api/run-cycle").status_code == 409
    finally:
        desk.lock.release()


def test_csrf_and_host_guards(client):
    c, desk = client
    assert c.post("/api/kill", headers={"Origin": "http://evil.example"}).status_code == 403
    assert not desk.store.kill_switch
    no_hdr = TestClient(c.app)  # no X-Desk header
    assert no_hdr.post("/api/kill").status_code == 403
    assert not desk.store.kill_switch
    assert c.get("/api/state", headers={"Host": "evil.com"}).status_code == 403
    assert c.get("/api/state", headers={"Host": "127.0.0.1:8000.evil.com"}).status_code == 403
    # legit dashboard-style requests still work
    ok = c.post("/api/kill", headers={"Origin": "http://127.0.0.1:8000", "Host": "127.0.0.1:8000"})
    assert ok.status_code == 200 and desk.store.kill_switch
    assert c.get("/api/state", headers={"Host": "localhost:8000"}).status_code == 200


def test_dashboard_sends_x_desk_header():
    from pathlib import Path
    html = (Path(__file__).parent.parent / "desk/static/index.html").read_text()
    assert '"X-Desk":"1"' in html


def test_flatten_sets_kill_switch(client):
    c, desk = client
    assert c.post("/api/flatten").status_code == 200
    assert desk.store.kill_switch
