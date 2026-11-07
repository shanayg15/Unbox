"""Report card endpoints (app/report.py) on a replayed cached run: no network, no LLM."""
import csv
import io
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KEY = "bb-01__t-cp-2__muse"
CARTPOLE = dict(obs_names=["cart_x", "cart_v", "pole_angle", "pole_ang_v"],
                obs_low=[-4.8, -3.0, -0.418, -3.5], obs_high=[4.8, 3.0, 0.418, 3.5],
                action_type="discrete", action_names=["push_left", "push_right"])

pytestmark = pytest.mark.skipif(not (ROOT / "cache" / KEY / "events.jsonl").exists(),
                                reason=f"cache/{KEY} not generated")


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("USE_MOCKS", "0")
    from fastapi.testclient import TestClient

    from app import report as R
    from app import server as S
    from contracts.types import SensorSpec

    R._SPECS["bb-01"] = SensorSpec(**CARTPOLE)
    bridge = S.new_bridge("replay", {"source": f"cache/{KEY}", "speed": 0})
    bridge.ctx["fallback_source"] = f"cache/{KEY}"
    bridge.ctx["report_truth"] = None              # no admin call: the found badge comes from summary.json
    events = S.load_jsonl(S.cache_path(KEY))
    play, held = S._replay_source_reveal_only(events)
    bridge.source.update(held)
    S.replay_into(bridge, play, speed=0)
    return TestClient(S.app), bridge, R, S


def _reveal(bridge):
    data = {"label": "Drifts in one zone of the track", "feature": "cart_x", "highlight_rule_ids": []}
    bridge.ctx["reveal"] = data
    bridge.emit({"type": "reveal", "data": data})


def _fix(bridge, R):
    """A generic minimal fix: disable the if-line the report blames for the flaw."""
    rep = R.build_report(bridge)
    prog = R.best_program(R._events(bridge))
    lines = prog["code"].split("\n")
    n = rep["failing"]["line"] - 1
    lines[n] = re.sub(r"^(\s*)(el)?if\b.*:\s*$", lambda m: f"{m[1]}{m[2] or ''}if False:", lines[n])
    fix = {"code": "\n".join(lines), "params": prog["params"], "diff": "", "changed_rule_ids": []}
    bridge.ctx["fix"] = fix
    bridge.emit({"type": "fix", "data": fix})


def test_card_before_and_after_reveal(env):
    client, bridge, R, _ = env
    r = client.get(f"/api/report/{bridge.run_id}")
    assert r.status_code == 200
    rep = r.json()
    assert rep["ready"] and rep["has_flaw"] and not rep["revealed"]
    best = max((e["data"] for e in bridge.buffer if e["type"] == "program"), key=lambda d: d["agreement"])
    assert rep["match"] == pytest.approx(best["agreement"])
    f = rep["failing"]
    assert f["feature"] == "cart_x" and f["op"] == "<" and f["action_name"] == "push_right"
    assert f["line"] and "if" in rep["program"]["code"].split("\n")[f["line"] - 1]
    assert f["sentence"].startswith("When cart_x is below")
    reg = rep["region"]
    assert reg["text"].startswith("cart_x <") and 0 < reg["fire_pct"] < 1
    assert 0 < reg["episodes_hit"] <= reg["episodes_total"] == 8
    assert rep["evidence"]["probe_queries"] > 0 and "rules proven" in rep["evidence"]["text"]
    assert all(x["rule_id"] != f["rule_id"] for x in rep["regression"])
    assert rep["found"] is None and rep["action"] is None and rep["guard_check"] is None

    _reveal(bridge)
    rep = client.get(f"/api/report/{bridge.run_id}").json()
    assert rep["revealed"] and rep["reported_label"] == "Drifts in one zone of the track"
    assert rep["found"] == "match"
    assert rep["failing"]["feature"] == "cart_x"


def test_fix_downloads(env):
    client, bridge, R, _ = env
    _reveal(bridge)
    assert client.get(f"/api/report/{bridge.run_id}/training.csv").status_code == 404   # not fixed yet
    _fix(bridge, R)
    rep = client.get(f"/api/report/{bridge.run_id}").json()
    assert rep["fixed"] and rep["action"]["n_training"] > 0
    assert rep["action"]["correct_action"] in ("push_left", "push_right")
    assert rep["action"]["text"].startswith("Retrain on states where cart_x <")
    g = rep["guard_check"]
    assert g["n"] > 0 and g["pct"] > 0.99 and g["text"].startswith("Outside the flaw zone")

    r = client.get(f"/api/report/{bridge.run_id}/training.csv")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == CARTPOLE["obs_names"] + ["label_action_id", "label_action_name"]
    thr = rep["failing"]["value"]
    assert len(rows) - 1 == rep["action"]["n_training"]
    assert all(float(row[0]) < thr for row in rows[1:])

    r = client.get(f"/api/report/{bridge.run_id}/tests.json")
    tests = json.loads(r.text)["tests"]
    assert {t["rule_id"] for t in tests} == {x["rule_id"] for x in rep["regression"]}
    assert all(len(t["states"]) <= 20 for t in tests)

    for url in (f"/api/report/{bridge.run_id}.md", f"/api/report/{bridge.run_id}/report.md"):
        r = client.get(url)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert "## Failing decision" in r.text and "def policy" in r.text and "## Ledger" in r.text


def test_no_flaw_card(env, monkeypatch):
    client, _, R, S = env
    key = "bb-01__none__muse"
    if not (ROOT / "cache" / key / "events.jsonl").exists():
        pytest.skip("no cached no-flaw run")
    bridge = S.new_bridge("replay", {"source": f"cache/{key}", "speed": 0})
    bridge.ctx["fallback_source"] = f"cache/{key}"
    play, held = S._replay_source_reveal_only(S.load_jsonl(S.cache_path(key)))
    S.replay_into(bridge, play, speed=0)
    rep = client.get(f"/api/report/{bridge.run_id}").json()
    assert rep["has_flaw"] is False and rep["failing"] is None
    assert rep["no_flaw_text"].startswith("No failing decision found. Readable copy matches the robot on")
    assert isinstance(rep["regression"], list)


def test_unknown_run_is_404(env):
    client, *_ = env
    assert client.get("/api/report/nope-123").status_code == 404
