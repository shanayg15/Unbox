"""B-Show: the UI fixture is a valid, realistic event stream."""
import base64
import json
from pathlib import Path

import pytest

from contracts.events import EVENT_TYPES

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "demo_events.jsonl"

# B-Brain adds "snap" to contracts/events.py (LANES.md section 4); accept it until then.
KNOWN = set(EVENT_TYPES) | {"snap"}


@pytest.fixture(scope="module")
def events():
    if not FIXTURE.exists():
        from scripts import make_fixture
        make_fixture.main(["--quiet"])
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def of(events, kind):
    return [e["data"] for e in events if e["type"] == kind]


def test_envelopes(events):
    run_ids = {e["run_id"] for e in events}
    assert len(run_ids) == 1
    for e in events:
        assert set(e) == {"type", "run_id", "t", "data"}
        assert e["type"] in KNOWN, e["type"]
        assert isinstance(e["data"], dict)
    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "run_finished"


def test_timestamps_realistic(events):
    ts = [e["t"] for e in events]
    assert ts == sorted(ts)
    assert 45 <= ts[-1] <= 90


def test_programs_grow_and_metrics_climb(events):
    progs = of(events, "program")
    assert [p["version"] for p in progs] == [1, 2, 3]
    lengths = [len(p["code"].splitlines()) for p in progs]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]
    for p in progs:
        assert p["code"].startswith("def policy(obs, P):")
        assert p["changed_lines"] and all(isinstance(i, int) for i in p["changed_lines"])
        for name in p["params"]:
            assert f'P["{name}"]' in p["code"]
    agr = [m["agreement"] for m in of(events, "metric")]
    assert agr == sorted(agr) and agr[0] <= 0.45 and agr[-1] >= 0.93
    counts = [q["n"] for q in of(events, "query_count")]
    assert counts == sorted(counts) and counts[-1] <= of(events, "run_started")[0]["budget"]


def test_ledger_story(events):
    snaps = of(events, "ledger")
    statuses = {(e["rule_id"], e["status"]) for s in snaps for e in s["entries"]}
    assert ("r3", "inferred") in statuses and ("r3", "refuted") in statuses
    assert ("r4", "inferred") in statuses and ("r4", "confirmed") in statuses
    final = {e["rule_id"]: e for e in snaps[-1]["entries"]}
    assert "speed" in final["r4"]["text"]


def test_one_snap_per_turn_with_rising_anomaly(events):
    snaps = of(events, "snap")
    assert [s["turn"] for s in snaps] == [1, 2, 3, 4]
    feature = of(events, "reveal")[0]["feature"]
    p = [s["anomaly_feature"]["probabilities"][feature] for s in snaps]
    assert p == sorted(p) and p[-1] > 0.9
    assert snaps[-1]["anomaly_feature"]["choice"] == feature
    for s in snaps:
        for key in ("n_rows", "input_tokens", "feature_relevance", "anomaly_feature",
                    "next_probe", "rule_support"):
            assert key in s
        assert abs(sum(s["anomaly_feature"]["probabilities"].values()) - 1) < 0.02
        assert s["input_tokens"] < 384_000


def test_frames_are_real_jpegs(events):
    clips = of(events, "frames")
    assert len(clips) >= 2
    for c in clips:
        assert len(c["bb"]) == len(c["clone"]) == len(c["disagree"]) > 0
        assert c["fps"] > 0
        assert base64.b64decode(c["bb"][0])[:2] == b"\xff\xd8"
        assert base64.b64decode(c["clone"][-1])[:2] == b"\xff\xd8"
    red = [sum(c["disagree"]) for c in clips]
    assert red[0] > red[-1]
    shield = of(events, "shield")[0]
    assert shield["on"] and base64.b64decode(shield["bb"][0])[:2] == b"\xff\xd8"


def test_reveal_fix_finish(events):
    rev = of(events, "reveal")[0]
    assert rev["highlight_rule_ids"] == ["r4"]
    fix = of(events, "fix")[0]
    assert fix["diff"].startswith("---") and "cart_v_limit" in fix["diff"]
    fin = of(events, "run_finished")[0]
    assert {"eval", "total_queries", "turns", "seconds", "cached"} <= set(fin)
    assert fin["eval"]["open_loop_agreement"] >= 0.93
