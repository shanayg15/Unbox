"""B-Brain tests: event-schema validation, offline end-to-end loop on mocks, brain JSON/backoff, ledger, probes.

No network: the brain is replaced by a scripted fake and SNAP=0.
    python -m tests.test_b_brain runs/<run_id>/events.jsonl     # validate a real run's event log
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from contracts.events import EVENT_TYPES


# ------------------------------------------------------------------ event schema
def _top_fields(desc: str) -> list[str]:
    """'{a, b: [x], c: {d, e}}  # comment' -> ['a', 'b', 'c']"""
    start = desc.index("{")
    depth, buf, fields = 0, "", []
    for ch in desc[start:]:
        if ch in "{[":
            depth += 1
            if depth == 1:
                continue
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and ch == ",":
            fields.append(buf)
            buf = ""
        elif depth >= 1:
            buf += ch
    fields.append(buf)
    out = []
    for f in fields:
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", f)
        if m:
            out.append(m.group(1))
    return out


REQUIRED = {k: _top_fields(v) for k, v in EVENT_TYPES.items()}


def validate_events(events: list[dict]) -> list[str]:
    """Problems found (empty list = valid)."""
    errs = []
    if not events:
        return ["no events"]
    if events[0].get("type") != "run_started":
        errs.append(f"first event is {events[0].get('type')!r}, not run_started")
    if events[-1].get("type") != "run_finished":
        errs.append(f"last event is {events[-1].get('type')!r}, not run_finished")
    t_prev = -1.0
    for i, ev in enumerate(events):
        for k in ("type", "run_id", "t", "data"):
            if k not in ev:
                errs.append(f"event {i}: missing envelope key {k!r}")
        typ = ev.get("type")
        if typ not in EVENT_TYPES:
            errs.append(f"event {i}: unknown type {typ!r}")
            continue
        if not isinstance(ev.get("t"), (int, float)) or ev["t"] < t_prev:
            errs.append(f"event {i}: t is not a non-decreasing number")
        t_prev = ev.get("t", t_prev)
        missing = [f for f in REQUIRED[typ] if f not in (ev.get("data") or {})]
        if missing:
            errs.append(f"event {i} ({typ}): missing data fields {missing}")
    return errs


def test_schema_parser_reads_contract():
    assert REQUIRED["frames"] == ["version", "seed", "fps", "bb", "clone", "disagree"]
    assert REQUIRED["snap"] == ["turn", "n_rows", "input_tokens", "feature_relevance", "anomaly_feature",
                                "next_probe", "rule_support"]
    assert REQUIRED["status"] == ["text"]


# ------------------------------------------------------------------ offline end-to-end loop
TURN1 = {
    "summary": "Pole angle and angular velocity drive the action.",
    "ledger": [{"rule_id": None, "text": "pushes right when pole_angle + k*pole_ang_v > 0", "status": "inferred",
                "confidence": 0.7},
               {"rule_id": None, "text": "cart_x decides the action", "status": "confirmed", "confidence": 0.9}],
    "program": {"code": "def policy(obs, P):\n    return 1 if obs[2] + P['k'] * obs[3] > 0 else 0\n",
                "params": {"k": 0.3}},
    "probes": [
        {"rule_id": "r1", "sweep": {"feature": "pole_angle", "start": -0.2, "stop": 0.2, "n": 20},
         "expect": {"feature": "pole_angle", "op": ">", "value": 0.05, "action": "push_right"}},
        # pole tilted right at a fixed base, so the robot pushes right everywhere: "left when cart_x > 0.5" is refuted
        {"rule_id": "r2", "base": [0.0, 0.0, 0.05, 0.0], "sweep": {"feature": "cart_x", "start": -2, "stop": 2, "n": 20},
         "expect": {"feature": "cart_x", "op": ">", "value": 0.5, "action": 0}},
    ],
}
TURN2 = {
    "summary": "Counterexamples at high cart_v: add an override.",
    "ledger": [{"rule_id": None, "text": "pushes left whenever cart_v > t", "status": "probing", "confidence": 0.6}],
    "program": {"code": ("def policy(obs, P):\n    if obs[1] > P['t']:\n        return 0\n"
                         "    return 1 if obs[2] + P['k'] * obs[3] > 0 else 0\n"), "params": {"k": 0.5, "t": 0.6}},
    "probes": [{"rule_id": "r3", "sweep": {"feature": "cart_v", "start": -1.5, "stop": 1.5, "n": 30},
                "expect": {"feature": "cart_v", "op": ">", "value": 0.6, "action": 0}}],
    "done": True,
}


class FakeBrain:
    script = [TURN1, TURN2]

    def __init__(self, provider, *, run_id, store=None, live=False, model=None):
        self.model, self.i = "fake", 0

    def turn(self, messages, action_names=None):
        from agent.protocol import BrainTurn, normalize_turn

        prompt = messages[-1]["content"]
        assert "CartPole" not in prompt and "t-cp-1" not in prompt, "rule 2: env/twist leaked into the prompt"
        out = json.loads(json.dumps(self.script[min(self.i, len(self.script) - 1)]))
        self.i += 1
        return BrainTurn.model_validate(normalize_turn(out, action_names)).model_dump()


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAP", "0")
    import agent.loop as loop
    from contracts.mocks import MockBlackBox, mock_env_factory, numerics_stub
    from store.store import JsonStore

    monkeypatch.setattr(loop, "Brain", FakeBrain)
    store = JsonStore(tmp_path)
    events = []
    out = loop.run(MockBlackBox(), "muse", env_factory=mock_env_factory, numerics=numerics_stub,
                   emit=events.append, store=store, run_id="t1", budget=5000, max_turns=4, live=False,
                   timeout_s=None, policy_label="Mock CartPole")
    return out, events, store


def test_offline_run_end_to_end(offline_run):
    out, events, store = offline_run
    assert validate_events(events) == []
    assert validate_events(store.load_events("t1")) == []          # the stored copy replays identically
    assert len(store.load_events("t1")) == len(events)
    status = {e.rule_id: e.status for e in out["ledger"]}
    assert status["r1"] == "confirmed"                              # probe verdicts come from Python
    assert status["r2"] == "refuted"                                # the brain's "confirmed" was downgraded
    assert status["r3"] == "confirmed"                              # the twist rule (cart_v > t -> left)
    assert out["program"].version == 2 and out["program"].agreement >= 0.9
    types = [e["type"] for e in events]
    assert "frames" in types and types.count("program") == 2


def test_replay_round_trip(offline_run):
    from store.store import replay

    _, events, store = offline_run
    got = []
    assert replay("t1", got.append, speed=0, store=store) == len(events)
    assert [e["type"] for e in got] == [e["type"] for e in events]


# ------------------------------------------------------------------ brain client
def test_extract_json_variants():
    from agent.brain import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('blah ```json\n{"a": 2}\n``` tail') == {"a": 2}
    assert extract_json('thinking... {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def _status_error(code: int):
    import httpx
    import openai

    req = httpx.Request("POST", "https://x/v1/chat/completions")
    resp = httpx.Response(code, request=req, json={"error": {"message": "x"}})
    cls = openai.RateLimitError if code == 429 else openai.InternalServerError
    return cls("boom", response=resp, body=None)


class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()
        self.finish_reason = "stop"


class _Resp:
    def __init__(self, text):
        self.choices = [_Choice(text)]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def test_backoff_retries_429_and_5xx(monkeypatch, tmp_path):
    import agent.brain as brain_mod
    from store.store import JsonStore

    monkeypatch.setenv("MODEL_API_KEY", "test")
    sleeps = []
    monkeypatch.setattr(brain_mod.time, "sleep", sleeps.append)
    b = brain_mod.Brain("muse", run_id="bk", store=JsonStore(tmp_path))
    calls = iter([_status_error(429), _status_error(503), _status_error(500), _Resp('{"summary": "ok"}')])

    def create(**kw):
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(b.client.chat.completions, "create", create)
    assert b.turn([{"role": "user", "content": "x"}])["summary"] == "ok"
    assert sleeps == [1.0, 2.0, 4.0]
    log = [json.loads(line) for line in (tmp_path / "bk" / "brain_calls.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in log] == [False, False, False, True]


def test_backoff_gives_up_after_three(monkeypatch, tmp_path):
    import agent.brain as brain_mod
    from store.store import JsonStore

    monkeypatch.setenv("MODEL_API_KEY", "test")
    monkeypatch.setattr(brain_mod.time, "sleep", lambda s: None)
    b = brain_mod.Brain("muse", run_id="bk2", store=JsonStore(tmp_path))

    def create(**kw):
        raise _status_error(429)

    monkeypatch.setattr(b.client.chat.completions, "create", create)
    with pytest.raises(brain_mod.BrainError):
        b.complete([{"role": "user", "content": "x"}])


def test_newest_muse_spark():
    from agent.brain import newest_muse_spark

    ids = ["muse-spark-1.1", "muse-spark-1.3-contributor", "muse-spark-1.3", "muse-spark-1.2", "sam-3.1"]
    assert newest_muse_spark(ids) == "muse-spark-1.3"


# ------------------------------------------------------------------ ledger + probes + snap state
def test_ledger_never_lets_brain_confirm():
    from agent.ledger import Ledger

    lg = Ledger()
    lg.apply([{"rule_id": None, "text": "a", "status": "confirmed", "confidence": 0.9}], 1)
    assert lg.get("r1").status == "inferred"
    lg.verdict("r1", "confirmed", 0.95, "q1")
    lg.apply([{"rule_id": "r1", "text": "a", "status": "inferred", "confidence": 0.5}], 2)
    assert lg.get("r1").status == "confirmed"                       # Python verdict survives an unchanged rule
    lg.apply([{"rule_id": "r1", "text": "a", "status": "refuted", "confidence": 0.1}], 3)
    assert lg.get("r1").status == "refuted"                         # the brain may still refute


def test_probe_verdicts():
    from agent.probes import run_probe
    from contracts.mocks import MockBlackBox

    bb = MockBlackBox()
    seen = [[0.0, 0.0, 0.0, 0.0]] * 5
    twist = {"sweep": {"feature": "cart_v", "start": -1.5, "stop": 1.5, "n": 31},
             "expect": {"feature": "cart_v", "op": ">", "value": 0.6, "action": 0}}
    wrong = {"sweep": {"feature": "cart_v", "start": -1.5, "stop": 1.5, "n": 31},
             "expect": {"feature": "cart_v", "op": ">", "value": 0.6, "action": 1}}
    assert run_probe(twist, "r1", bb, bb.spec(), seen)["status"] == "confirmed"
    assert run_probe(wrong, "r2", bb, bb.spec(), seen)["status"] == "refuted"
    assert bb.query_count() == 62


def test_snap_state_drops_oldest_rollouts_first(monkeypatch):
    from agent import snap
    from contracts.mocks import CARTPOLE_SPEC

    monkeypatch.setattr(snap, "MAX_STATE_CHARS", 3000)
    rows = [(f"roll{i}", [0.1, 0.2, 0.3, 0.4], 1, "rollout") for i in range(100)]
    rows += [("q9", [0.5, 0.6, 0.7, 0.8], 0, "probe")]
    text, n = snap.build_state(CARTPOLE_SPEC, rows, [], [])
    assert len(text) <= 3000 and n < len(rows)
    assert "q9 |" in text and "roll0 |" not in text


if __name__ == "__main__":
    evs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    problems = validate_events(evs)
    counts: dict[str, int] = {}
    for e in evs:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"{len(evs)} events: {counts}")
    print("VALID" if not problems else "INVALID:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)
