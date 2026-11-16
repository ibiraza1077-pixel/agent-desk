from desk.server import build_desk


def test_demo_cycles_end_to_end(settings):
    desk = build_desk(settings)
    entries = []
    orig = desk.broker.submit_bracket_buy

    def spy(symbol, qty, stop, tp):
        eq = desk.broker.get_account().equity
        entries.append(qty * desk.broker.get_latest_price(symbol) / eq)
        return orig(symbol, qty, stop, tp)

    desk.broker.submit_bracket_buy = spy
    for _ in range(25):
        assert desk.run_cycle() is not None
        assert len(desk.broker.get_positions()) <= settings.max_open_positions
    store = desk.store
    feed = store.get_feed(limit=500)
    agents = {f["agent"] for f in feed}
    assert {"Scout", "Analyst", "Charts", "Skeptic", "Chief", "Risk"} <= agents
    decisions = store.get_decisions(500)
    actions = {d["action"] for d in decisions}
    assert "BUY" in actions and actions & {"WAIT", "SKIP"}
    assert len(store.get_equity()) == 25
    assert entries, "demo should have produced at least one entry"
    assert max(entries) <= settings.max_position_pct
    approved = [d for d in decisions if d["action"] == "BUY" and d["risk_status"] == "approved"]
    assert len(approved) == len(entries) and all(d["qty"] >= 1 and d["order_id"] for d in approved)
    assert desk.equity_series()[0]["spy"] is not None


def test_kill_switch_stops_entries_but_cycles_continue(settings):
    desk = build_desk(settings)
    desk.store.set_kill_switch(True)
    for _ in range(5):
        desk.run_cycle()
    assert not desk.broker.get_positions()
    assert any("kill switch" in f["text"] for f in desk.store.get_feed())


def test_demo_is_deterministic(settings, tmp_path):
    import dataclasses
    # separate databases: the Scout's rotation reads each desk's own recent decisions
    a = build_desk(dataclasses.replace(settings, db_path=str(tmp_path / "a.db")))
    b = build_desk(dataclasses.replace(settings, db_path=str(tmp_path / "b.db")))
    for _ in range(6):
        a.run_cycle(); b.run_cycle()
    assert a.broker.get_account().equity == b.broker.get_account().equity


def test_scout_rotates_recently_debated_names_to_the_back(settings, tmp_path):
    import dataclasses
    d = build_desk(dataclasses.replace(settings, db_path=str(tmp_path / "r.db")))
    acct, pos = d.broker.get_account(), d.broker.get_positions()
    first = [sym for _, sym, _, _ in d.screen(acct, pos)]
    for sym in first:
        d.store.add_decision(cycle_id=1, symbol=sym, action="WAIT", confidence=0.1, size_pct=0.0, stop=None,
                             target=None, rationale="", risk_status="n/a", risk_reasons=[], qty=0, order_id=None)
    second = [sym for _, sym, _, _ in d.screen(acct, pos)]
    assert first and second and not set(first) & set(second)
