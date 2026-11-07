"""B-Show B5: app server, WebSocket bridge, replay, live fallback, reveal/fix/shield."""
import time

import pytest
from fastapi.testclient import TestClient

import app.server as server

N_FIXTURE = None


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from store.store import JsonStore

    store = JsonStore(tmp_path / "runs")              # never write test runs into runs/
    monkeypatch.setattr(server, "_store", lambda: store)
    monkeypatch.setenv("USE_MOCKS", "1")
    monkeypatch.setenv("FALLBACK_SPEED", "0")          # instant fallback replays
    server.ensure_fixture()
    with TestClient(server.app) as c:
        yield c


def fixture_events():
    return server.load_jsonl(server.ensure_fixture())


def drain(ws, until="run_finished", limit=500):
    out = []
    for _ in range(limit):
        ev = ws.receive_json()
        out.append(ev)
        if ev["type"] == until:
            break
    return out


def wait_finished(client, run_id, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if client.get(f"/api/runs/{run_id}").json()["finished"]:
            return
        time.sleep(0.05)
    raise AssertionError("run never finished")


def test_index_and_config(client):
    r = client.get("/")
    assert r.status_code == 200 and "Unbox" in r.text
    assert client.get("/api/config").json()["use_mocks"] is True
    pols = client.get("/api/policies").json()["policies"]
    assert pols and {"bb", "label", "env_id"} <= set(pols[0])
    tw = client.get("/api/twists", params={"bb": pols[0]["bb"]}).json()["twists"]
    assert tw and set(tw[0]) == {"twist_id", "hidden_label"}
    assert "Hidden flaw" in tw[0]["hidden_label"]


def test_replay_streams_every_event_in_order(client):
    src = fixture_events()
    r = client.post("/api/replay", json={"fixture": "demo_events", "speed": 0, "hold_reveal": False})
    assert r.status_code == 200
    rid = r.json()["run_id"]
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws)
    assert [e["type"] for e in got] == [e["type"] for e in src]
    assert all(e["run_id"] == rid for e in got)
    assert got[-1]["data"]["cached"] is True
    ts = [e["t"] for e in got]
    assert ts == sorted(ts)


def test_replay_keeps_original_spacing(client):
    src = fixture_events()[:6]
    span = src[-1]["t"] - src[0]["t"]
    b = server.new_bridge("replay")
    t0 = time.time()
    server.replay_into(b, src, speed=span / 0.5)        # whole slice in ~0.5 s
    assert 0.4 <= time.time() - t0 <= 1.5


def test_late_socket_gets_buffer_first(client):
    rid = client.post("/api/replay", json={"speed": 0, "hold_reveal": False}).json()["run_id"]
    wait_finished(client, rid)
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws)
    assert len(got) == len(fixture_events())


def test_two_sockets_see_the_same_stream(client):
    rid = client.post("/api/replay", json={"speed": 40, "hold_reveal": False}).json()["run_id"]
    with client.websocket_connect(f"/ws/{rid}") as a, client.websocket_connect(f"/ws/{rid}") as b:
        ga, gb = drain(a), drain(b)
    assert [e["type"] for e in ga] == [e["type"] for e in gb]
    assert len(ga) == len(fixture_events())


def test_buttons_on_a_held_replay(client):
    r = client.post("/api/replay", json={"speed": 0}).json()
    assert set(r["held"]) == {"reveal", "fix", "shield"}
    rid = r["run_id"]
    wait_finished(client, rid)
    assert client.post("/api/fix", json={"run_id": rid}).status_code == 409   # reveal first
    rev = client.post("/api/reveal", json={"run_id": rid}).json()
    assert rev["feature"] == "cart_v" and rev["highlight_rule_ids"] == ["r4"]
    assert "cart_v_limit" in client.post("/api/fix", json={"run_id": rid}).json()["diff"]
    assert client.post("/api/shield", json={"run_id": rid, "on": True}).json()["frames"] > 0
    assert client.post("/api/shield", json={"run_id": rid, "on": False}).json()["on"] is False
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws, until="__never__", limit=len(fixture_events()) - 3 + 4)
    tail = [e["type"] for e in got[-4:]]
    assert tail == ["reveal", "fix", "shield", "shield"]
    assert got[-1]["data"] == {"on": False, "bb": None}


def test_run_with_stub_agent_falls_back_loudly(client, monkeypatch):
    import agent.loop

    def stub(*a, **k):
        raise NotImplementedError

    monkeypatch.setattr(agent.loop, "run", stub)
    rid = client.post("/api/run", json={"bb": "mock", "twist_id": "t-cp-1", "brain": "muse"}).json()["run_id"]
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws)
    kinds = [e["type"] for e in got]
    assert kinds[0] == "error" and "NotImplementedError" in got[0]["data"]["message"]
    banner = got[1]
    assert banner["type"] == "status" and banner["data"]["banner"] == "fallback"
    assert "fixture" in banner["data"]["text"] or "cached" in banner["data"]["text"]
    assert got[-1]["type"] == "run_finished" and got[-1]["data"]["cached"] is True
    # the replayed source's reveal is held for the button
    rev = client.post("/api/reveal", json={"run_id": rid}).json()
    assert rev["feature"] == "cart_v"


def test_live_timeout_switches_to_cache(client, monkeypatch, tmp_path):
    import agent.loop

    seen = {}

    def slow(bb, brain, **kw):
        seen.update(kw)
        kw["emit"]({"type": "status", "data": {"text": "thinking..."}})
        time.sleep(1.5)
        kw["emit"]({"type": "status", "data": {"text": "too late, should be dropped"}})

    monkeypatch.setattr(agent.loop, "run", slow)
    monkeypatch.setenv("LIVE_TIMEOUT_S", "0.2")
    monkeypatch.setenv("LIVE_GRACE_S", "0.1")
    key = server.cache_key("mock", "t-cp-1", "muse")
    cache = tmp_path / key
    cache.mkdir()
    (cache / "events.jsonl").write_text(server.fixture_path("demo_events").read_text())
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path)

    rid = client.post("/api/run", json={"bb": "mock", "twist_id": "t-cp-1"}).json()["run_id"]
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws)
    texts = [e["data"].get("text") for e in got if e["type"] == "status"]
    assert texts[0] == "thinking..."
    assert server.FALLBACK_TIMEOUT_TEXT in texts
    time.sleep(1.6)
    assert all("too late" not in str(e["data"]) for e in server.get_bridge(rid).buffer)
    # the agent got the LANES.md section 2 signature
    assert {"env_factory", "numerics", "emit", "store", "run_id", "budget", "max_turns", "live",
            "timeout_s", "policy_label", "demo_seed"} <= set(seen)
    env, obs = seen["env_factory"](0)
    assert len(obs) == 4
    env.close()


def test_agent_events_are_wrapped_into_envelopes(client, monkeypatch):
    import agent.loop

    def fake(bb, brain, **kw):
        kw["emit"]({"type": "status", "data": {"text": "hi"}})
        kw["emit"]({"type": "query_count", "n": 7})                 # flat form tolerated
        kw["emit"]({"type": "run_finished", "data": {"eval": None, "total_queries": 7,
                                                     "turns": 1, "seconds": 0.1, "cached": False}})
        return {}

    monkeypatch.setattr(agent.loop, "run", fake)
    rid = client.post("/api/run", json={"bb": "mock"}).json()["run_id"]
    with client.websocket_connect(f"/ws/{rid}") as ws:
        got = drain(ws)
    assert [e["type"] for e in got] == ["status", "query_count", "run_finished"]
    assert got[1]["data"] == {"n": 7}
    assert all(set(e) == {"type", "run_id", "t", "data"} for e in got)


def test_bad_names_rejected(client):
    assert client.post("/api/replay", json={"run_id": "../etc"}).status_code == 400
    assert client.post("/api/replay", json={"cache_key": "nope"}).status_code == 404
    assert client.post("/api/reveal", json={"run_id": "missing"}).status_code == 404


def test_results_endpoint(client):
    r = client.get("/api/results").json()
    assert {"available", "results", "source"} == set(r)


def test_reveal_and_fix_are_idempotent(client):
    rid = client.post("/api/replay", json={"speed": 0}).json()["run_id"]
    wait_finished(client, rid)
    for _ in range(3):
        client.post("/api/reveal", json={"run_id": rid})
        client.post("/api/fix", json={"run_id": rid})
    kinds = [e["type"] for e in server.get_bridge(rid).buffer]
    assert kinds.count("reveal") == 1 and kinds.count("fix") == 1


def test_ui_has_one_handler_per_name():
    import re
    js = (server.STATIC_DIR / "app.js").read_text()
    names = re.findall(r"^\s*(?:async\s+)?function\s+(\w+)\s*\(", js, re.M)
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate function declarations shadow each other: {dupes}"


def test_persisted_events_are_full_envelopes(client, monkeypatch):
    import agent.loop

    def stub(*a, **k):
        raise NotImplementedError

    monkeypatch.setattr(agent.loop, "run", stub)
    rid = client.post("/api/run", json={"bb": "mock"}).json()["run_id"]
    wait_finished(client, rid)
    saved = server._store().load_events(rid)
    assert saved and all(set(e) == {"type", "run_id", "t", "data"} and e["run_id"] == rid for e in saved)


def test_results_fixture_source(client):
    r = client.get("/api/results", params={"source": "fixture"}).json()
    assert r["available"] and r["source"] == "fixture"
    rows = r["results"]["summary"]["rows"]
    assert {True, False} == {row["snap"] for row in rows}
    assert any(row["twist_found"] is False and row["twist"] != "none" for row in rows)
