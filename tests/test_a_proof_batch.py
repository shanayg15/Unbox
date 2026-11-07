"""A6: scripts/batch.py. No network and no real brain: RuleBlackBox plus a scripted fake."""
import json
import os

import pytest

import numerics
from agent import loop as agent_loop
from agent.protocol import BrainTurn, normalize_turn
from scripts import batch
from store.store import JsonStore
from tests.test_a_proof_helpers import CARTPOLE_SPEC, RuleBlackBox, make_env_factory

NAMES = ["cart_x", "cart_v", "pole_angle", "pole_ang_v"]
# The in-memory world has no renderer, so the agent's clone clip is skipped; gymnasium warns about it.
pytestmark = pytest.mark.filterwarnings("ignore:.*render method without specifying any render mode")


# --- planning ------------------------------------------------------------------------------------------

CATALOG = {
    "bb-01": {"label": "CartPole-PPO", "env_id": "CartPole-v1",
              "twists": [{"id": "t-cp-1", "feature": "cart_v", "threshold": 0.03},
                         {"id": "t-cp-2", "feature": "cart_x", "threshold": -0.08}]},
    "bb-02": {"label": "CartPole-DQN", "env_id": "CartPole-v1",
              "twists": [{"id": "t-cp-1", "feature": "cart_v", "threshold": 0.03},
                         {"id": "t-cp-2", "feature": "cart_x", "threshold": -0.08}]},
    "bb-04": {"label": "Acrobot-PPO", "env_id": "Acrobot-v1", "twists": []},
    "bb-05": {"label": "LunarLander-PPO", "env_id": "LunarLander-v3",
              "twists": [{"id": "t-ll-1", "feature": "x", "threshold": 0.05}]},
}


def test_plan_covers_every_policy_flaw_and_snap_setting():
    configs = batch.plan_configs(CATALOG, snaps=[True, False])
    got = {(c.bb, c.twist, c.snap) for c in configs}
    want = set()
    for bb, twists in [("bb-01", ["t-cp-1", "t-cp-2", None]), ("bb-02", ["t-cp-1", "t-cp-2", None]),
                       ("bb-04", [None]), ("bb-05", ["t-ll-1", None])]:
        want |= {(bb, t, s) for t in twists for s in (True, False)}
    assert got == want and len(configs) == len(want)


def test_plan_runs_the_demo_config_first():
    first = batch.plan_configs(CATALOG, snaps=[True, False])[0]
    assert (first.bb, first.twist, first.snap) == ("bb-05", "t-ll-1", True)


def test_plan_carries_the_truth_needed_to_score_the_run():
    config = next(c for c in batch.plan_configs(CATALOG, snaps=[True]) if c.twist == "t-cp-2")
    assert config.feature == "cart_x" and config.threshold == -0.08 and config.label == "CartPole-PPO"


@pytest.mark.parametrize("bbs, twists, want", [
    (["bb-01"], ["t-cp-1"], {("bb-01", "t-cp-1")}),
    (["bb-01"], ["none"], {("bb-01", None)}),
    (None, ["t-ll-1"], {("bb-05", "t-ll-1")}),
])
def test_plan_filters(bbs, twists, want):
    configs = batch.plan_configs(CATALOG, snaps=[True], bbs=bbs, twists=twists)
    assert {(c.bb, c.twist) for c in configs} == want


def test_plan_rejects_a_policy_it_does_not_know():
    with pytest.raises(ValueError, match="bb-99"):
        batch.plan_configs(CATALOG, snaps=[True], bbs=["bb-99"])


def test_each_run_group_holds_one_policy_because_its_twist_is_server_wide():
    groups = batch.group_by_policy(batch.plan_configs(CATALOG, snaps=[True, False]))
    assert [g[0].bb for g in groups][0] == "bb-05"
    assert all(len({c.bb for c in g}) == 1 for g in groups)
    assert sum(map(len, groups)) == 18     # bb-01: 6, bb-02: 6, bb-04: 2, bb-05: 4


@pytest.mark.parametrize("text, want", [("1,0", [True, False]), ("1", [True]), ("0", [False])])
def test_snap_flag(text, want):
    assert batch.parse_args(["--snap", text]).snap == want


# --- did the run find the flaw? ------------------------------------------------------------------------

def turn(n, queries=None):
    events = [{"type": "status", "data": {"text": f"Turn {n}: Muse is thinking"}}]
    if queries is not None:
        events.insert(0, {"type": "query_count", "data": {"n": queries}})
    return events


def ledger(*entries):
    return {"type": "ledger", "data": {"entries": [
        {"rule_id": rid, "text": text, "status": status} for rid, text, status in entries]}}


def probing(rid, feature, op, value):
    return {"type": "status", "data": {"text": (f"Probing {rid}: sweep {feature} 0.9→0.95, "
                                                f"expect action 0 where {feature} {op} {value}")}}


def program(code, params):
    return {"type": "program", "data": {"version": 1, "code": code, "params": params, "agreement": 0.9}}


def evidence(events, feature="cart_v", threshold=0.03):
    return batch.twist_evidence(events, feature, threshold, NAMES)


@pytest.mark.parametrize("text, found", [
    ("When cart_v > 0.03, the robot chooses push_left", True),
    ("pushes left once cart_v exceeds 0.035", True),            # 16.7% off, inside 20%
    ("pushes left once cart_v exceeds 0.024", True),            # exactly 20% low
    ("pushes left once cart_v exceeds 0.037", False),           # 23% off
    ("pushes left once cart_v exceeds 0.023", False),
    ("pushes left once cart_v exceeds -0.03", False),           # wrong sign
    ("pushes left when cart_x > 0.03", False),                  # right number, wrong feature
    ("pushes left when xcart_v > 0.03", False),                 # the feature name is only a substring
    ("rule r3: pushes left when cart_v is high, q3", False),    # r3 and q3 are ids, not thresholds
])
def test_confirmed_rule_text_scored_against_the_true_threshold(text, found):
    got = evidence([*turn(1), ledger(("r1", text, "confirmed"))])
    assert got["found"] is found


@pytest.mark.parametrize("status", ["observed", "inferred", "probing", "refuted"])
def test_only_a_confirmed_rule_counts(status):
    assert evidence([*turn(1), ledger(("r1", "cart_v > 0.03 means left", status))])["found"] is False


def test_negative_threshold_needs_a_negative_number():
    events = [*turn(1), ledger(("r1", "pushes right when cart_x < -0.08", "confirmed"))]
    assert batch.twist_evidence(events, "cart_x", -0.08, NAMES)["found"] is True


def test_rule_with_no_number_is_scored_on_the_probe_that_confirmed_it():
    events = [*turn(1), probing("r1", "cart_v", ">", 0.03), ledger(("r1", "over-reacts when moving fast", "confirmed"))]
    assert evidence(events)["found"] is True


def test_numbers_in_the_sweep_range_are_not_a_threshold():
    # Sweeping 0.03 to 0.5 proves nothing about 0.03; the expected region starts at 0.2 here.
    events = [*turn(1), {"type": "status", "data": {
        "text": "Probing r1: sweep cart_v 0.03→0.5, expect action 0 where cart_v > 0.2"}},
        ledger(("r1", "pushes left when moving fast", "confirmed"))]
    assert evidence(events)["found"] is False


def test_probe_on_another_feature_does_not_count():
    events = [*turn(1), probing("r1", "cart_x", ">", 0.03), ledger(("r1", "over-reacts", "confirmed"))]
    assert evidence(events)["found"] is False


@pytest.mark.parametrize("code", [
    'def policy(obs, P):\n    v = obs[1]\n    if v > P["t"]:\n        return 0\n    return 1\n',
    'def policy(obs, P):\n    if obs[1] > P["t"]:\n        return 0\n    return 1\n',
    'def policy(obs, P):\n    if P["t"] < obs[1]:\n        return 0\n    return 1\n',
    'def policy(obs, P):\n    x, v, a, w = obs\n    return 0 if v > P["t"] else 1\n',
    'def policy(obs, P):\n    x, v = obs[0], obs[1]\n    return 0 if v > P["t"] else 1\n',
])
def test_fitted_threshold_on_the_feature_counts_because_the_optimizer_sets_the_numbers(code):
    # Hard rule 3: the brain guesses and the optimizer fits, so the fitted value is the real threshold.
    events = [*turn(1), program(code, {"t": 0.031}), ledger(("r1", "pushes left when cart_v is high", "confirmed"))]
    assert evidence(events)["found"] is True


@pytest.mark.parametrize("code, params", [
    ('def policy(obs, P):\n    return 0 if obs[1] > P["t"] else 1\n', {"t": 0.5}),      # fitted far from the truth
    ('def policy(obs, P):\n    return 0 if obs[0] > P["t"] else 1\n', {"t": 0.03}),     # on another feature
    ('def policy(obs, P):\n    return 0 if obs[1] * P["t"] > 0.2 else 1\n', {"t": 0.03}),   # a gain, not a limit
])
def test_fitted_value_must_be_a_limit_on_that_feature(code, params):
    events = [*turn(1), program(code, params), ledger(("r1", "pushes left when cart_v is high", "confirmed"))]
    assert evidence(events)["found"] is False


def test_fitted_threshold_without_a_confirmed_rule_is_not_enough():
    code = 'def policy(obs, P):\n    return 0 if obs[1] > P["t"] else 1\n'
    assert evidence([*turn(1), program(code, {"t": 0.03})])["found"] is False


def test_records_the_first_turn_and_the_queries_spent_to_get_there():
    events = [*turn(1, queries=0), ledger(("r1", "cart_v > 0.5", "probing")),
              {"type": "query_count", "data": {"n": 900}},
              *turn(2), ledger(("r1", "cart_v > 0.03", "confirmed")),
              {"type": "query_count", "data": {"n": 1400}},      # the probes that confirmed it are counted
              *turn(3), ledger(("r1", "cart_v > 0.03", "confirmed")),
              {"type": "query_count", "data": {"n": 2000}}]
    got = evidence(events)
    assert got == {"found": True, "queries_to_twist": 1400, "turns_to_twist": 2}


def test_never_found_has_null_counts():
    assert evidence(turn(1)) == {"found": False, "queries_to_twist": None, "turns_to_twist": None}


def test_a_rule_refuted_later_no_longer_counts():
    events = [*turn(1), ledger(("r1", "cart_v > 0.03", "confirmed")),
              *turn(2), ledger(("r1", "cart_v > 0.03", "refuted"))]
    assert evidence(events)["found"] is False


def test_a_rule_merely_reworded_later_still_counts():
    # The ledger drops a confirmed status when the brain edits the text, but the evidence still stands.
    events = [*turn(1), ledger(("r1", "cart_v > 0.03", "confirmed")),
              {"type": "query_count", "data": {"n": 700}},
              *turn(2), ledger(("r1", "pushes left when cart_v is high", "inferred"))]
    assert evidence(events) == {"found": True, "queries_to_twist": 700, "turns_to_twist": 1}


def test_found_again_after_a_refutation_reports_the_later_time():
    events = [*turn(1), ledger(("r1", "cart_v > 0.03", "confirmed")), {"type": "query_count", "data": {"n": 100}},
              *turn(2), ledger(("r1", "cart_v > 0.03", "refuted")), {"type": "query_count", "data": {"n": 200}},
              *turn(3), ledger(("r1", "cart_v > 0.03", "refuted"), ("r2", "cart_v > 0.031", "confirmed")),
              {"type": "query_count", "data": {"n": 300}}]
    assert evidence(events) == {"found": True, "queries_to_twist": 300, "turns_to_twist": 3}


# --- running a config end to end -----------------------------------------------------------------------

BALANCE_GAINS = {"x": 1.0, "v": 2.0, "a": 30.0, "w": 7.0}
MOVING_FAST = {"condition": lambda o: o[1] > 0.5, "override": lambda o, a: 0}
LINEAR = ('def policy(obs, P):\n    return 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] '
          '+ P["w"] * obs[3] > 0 else 0\n')


def balance(obs, P):
    return 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] + P["w"] * obs[3] > 0 else 0


def anxious(obs, P):
    """A second policy on the same env, disagreeing with `balance` wherever the pole spins fast."""
    return 0 if obs[3] > 0.2 else balance(obs, P)


class FakeWorld:
    """HttpWorld's interface, in memory."""

    twists = {"t-cp-1": MOVING_FAST}

    def __init__(self):
        self.configured = []
        self.boxes = {}
        self.rules = {"bb-01": balance, "bb-02": anxious}

    def configure(self, bb, twist):
        self.configured.append((bb, twist))
        self.boxes[bb] = RuleBlackBox(self.rules[bb], BALANCE_GAINS, CARTPOLE_SPEC, twist=self.twists.get(twist))

    def client(self, bb):
        return self.boxes[bb]

    def env_factory(self, bb):
        return make_env_factory()


class ScriptedBrain:
    """Writes a program and probes the flaw. Python still decides every verdict."""
    seen_snap = []
    prompts = []

    def __init__(self, provider, *, run_id, store=None, live=False, model=None):
        self.model, self.calls = "scripted-1", 0
        ScriptedBrain.seen_snap.append(os.environ.get("SNAP"))

    def turn(self, messages, action_names=None):
        ScriptedBrain.prompts.append(messages[-1]["content"])
        self.calls += 1
        reply = {
            "summary": "A linear balance rule, then test for an override when the cart moves fast.",
            "ledger": [{"rule_id": None, "text": "pushes left when cart_v > 0.5", "status": "probing",
                        "confidence": 0.5}],
            "program": {"code": LINEAR, "params": {"x": 0.5, "v": 0.5, "a": 5.0, "w": 2.0}},
            "probes": [{"rule_id": "r1", "sweep": {"feature": "cart_v", "start": -1.0, "stop": 1.0, "n": 21},
                        "expect": {"feature": "cart_v", "op": ">", "value": 0.5, "action": 0}}],
            "done": self.calls >= 2,
        }
        return BrainTurn.model_validate(normalize_turn(reply, action_names)).model_dump()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_loop, "Brain", ScriptedBrain)
    monkeypatch.setattr(batch, "get_store", lambda: JsonStore(tmp_path / "runs"))
    monkeypatch.setenv("SNAP", "untouched")
    ScriptedBrain.seen_snap, ScriptedBrain.prompts = [], []
    paths = batch.Paths(results=tmp_path / "results", cache=tmp_path / "cache")
    return FakeWorld(), paths, batch.RunOptions(max_turns=2, budget=5000)


def config(twist="t-cp-1", snap=False, bb="bb-01", feature="cart_v", threshold=0.5):
    return batch.Config(bb=bb, label="CartPole-PPO", env_id="CartPole-v1", twist=twist,
                        feature=feature if twist else None, threshold=threshold if twist else None,
                        snap=snap, brain="muse")


def test_run_config_writes_a_row_with_every_summary_field(env):
    world, paths, options = env
    row = batch.run_config(config(), world, paths, options)
    expected = {"bb", "label", "twist", "brain", "model", "agreement", "clone_reward", "bb_reward", "bound_95",
                "lines_of_code", "queries", "turns", "seconds", "speedup", "twist_found", "snap",
                "queries_to_twist", "turns_to_twist"}
    assert expected <= set(row)
    assert row["bb"] == "bb-01" and row["twist"] == "t-cp-1" and row["snap"] is False
    assert row["model"] == "scripted-1" and row["turns"] == len(ScriptedBrain.prompts) >= 1
    assert row["lines_of_code"] == 2
    assert row["twist_found"] is True
    assert row["turns_to_twist"] == 1 and 0 < row["queries_to_twist"] <= row["queries"]


def test_speedup_is_black_box_time_over_clone_time(env):
    world, paths, options = env
    row = batch.run_config(config(), world, paths, options)
    assert row["speedup"] > 1      # RuleBlackBox reports 40 us; a two-line clone is far below that


def test_the_twist_is_set_on_the_server_before_the_run_starts(env):
    world, paths, options = env
    batch.run_config(config(twist="t-cp-1"), world, paths, options)
    assert world.configured == [("bb-01", "t-cp-1")]
    batch.run_config(config(twist=None), world, paths, options)
    assert world.configured[-1] == ("bb-01", None)


def test_flawless_run_has_no_twist_to_find(env):
    world, paths, options = env
    row = batch.run_config(config(twist=None), world, paths, options)
    assert row["twist"] == "none" and row["twist_found"] is False
    assert row["queries_to_twist"] is None and row["turns_to_twist"] is None


@pytest.mark.parametrize("snap, want", [(True, "1"), (False, "0")])
def test_snap_setting_reaches_the_agent_and_is_restored_after(env, snap, want, monkeypatch):
    world, paths, options = env
    # With SNAP on, the agent would call the real SubQ API; a failed scan is skipped by design.
    import agent.snap as snap_mod
    monkeypatch.setattr(snap_mod, "scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    batch.run_config(config(snap=snap), world, paths, options)
    assert ScriptedBrain.seen_snap == [want]
    assert os.environ["SNAP"] == "untouched"


def test_brain_prompts_never_name_the_env_policy_or_twist(env):
    world, paths, options = env
    batch.run_config(config(), world, paths, options)
    text = "\n".join(ScriptedBrain.prompts)
    assert ScriptedBrain.prompts
    for secret in ("CartPole", "t-cp-1", "bb-01", "PPO"):
        assert secret not in text


def test_snap_on_run_is_cached_where_the_app_looks_for_it(env, monkeypatch):
    world, paths, options = env
    import agent.snap as snap_mod
    monkeypatch.setattr(snap_mod, "scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    batch.run_config(config(snap=True), world, paths, options)
    cached = paths.cache / "bb-01__t-cp-1__muse" / "events.jsonl"
    assert cached.exists()
    types = [json.loads(line)["type"] for line in cached.read_text().splitlines()]
    assert types[0] == "run_started" and types[-1] == "run_finished"


def test_snap_off_run_is_not_cached(env):
    world, paths, options = env
    batch.run_config(config(snap=False), world, paths, options)
    assert not (paths.cache / "bb-01__t-cp-1__muse").exists()


def test_queries_are_left_out_of_the_cache_because_they_are_large_and_unused(env, monkeypatch):
    world, paths, options = env
    import agent.snap as snap_mod
    monkeypatch.setattr(snap_mod, "scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    batch.run_config(config(snap=True), world, paths, options)
    assert not (paths.cache / "bb-01__t-cp-1__muse" / "queries.jsonl").exists()


def test_a_finished_config_is_skipped_on_rerun(env):
    world, paths, options = env
    first = batch.run_config(config(), world, paths, options)
    n_runs = len(world.configured)
    again = batch.run_config(config(), world, paths, options)
    assert len(world.configured) == n_runs and again == first


def test_force_reruns_a_finished_config(env):
    world, paths, options = env
    batch.run_config(config(), world, paths, options)
    n_runs = len(world.configured)
    batch.run_config(config(), world, paths, batch.RunOptions(max_turns=2, budget=5000, force=True))
    assert len(world.configured) == n_runs + 1


def test_group_keeps_going_after_one_config_fails_and_reports_it(env, monkeypatch):
    world, paths, options = env
    real_run = agent_loop.run
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(agent_loop, "run", flaky)
    failures = batch.run_group([config(twist="t-cp-1"), config(twist=None)], world, paths, options)
    assert len(failures) == 1 and "bb-01__t-cp-1__muse__snap0" in failures[0] and "boom" in failures[0]
    summary = json.loads((paths.results / "summary.json").read_text())
    assert [r["twist"] for r in summary["rows"]] == ["none"]     # the failed one left no row, so a rerun retries it


def test_summary_is_rewritten_after_every_row_and_leaves_out_the_code(env):
    world, paths, options = env
    batch.run_config(config(twist="t-cp-1"), world, paths, options)
    batch.write_summary(paths)
    rows = json.loads((paths.results / "summary.json").read_text())["rows"]
    assert len(rows) == 1 and "code" not in rows[0] and "params" not in rows[0]
    batch.run_config(config(twist=None), world, paths, options)
    batch.write_summary(paths)
    assert len(json.loads((paths.results / "summary.json").read_text())["rows"]) == 2


def test_results_folder_holds_only_what_the_results_tab_should_show(env):
    # The app turns every results/*.json into a table, so per-run working files must sit in a subfolder.
    world, paths, options = env
    batch.run_config(config(), world, paths, options)
    batch.write_summary(paths)
    assert sorted(p.name for p in paths.results.glob("*.json")) == ["summary.json"]


def test_demo_seed_follows_the_same_settings_as_the_app(env, monkeypatch):
    monkeypatch.setenv("DEMO_SEED", "7")
    monkeypatch.setenv("DEMO_SEED_BB_05", "42")
    assert batch.demo_seed_for("bb-05") == 42 and batch.demo_seed_for("bb-01") == 7


# --- cross-match ---------------------------------------------------------------------------------------

def write_row(paths, bb, twist, snap, code, params):
    paths.rows.mkdir(parents=True, exist_ok=True)
    key = f"{bb}__{twist}__muse__snap{int(snap)}"
    (paths.rows / f"{key}.json").write_text(json.dumps(
        {"bb": bb, "twist": twist, "snap": snap, "brain": "muse", "code": code, "params": params, "ok": True}))


CROSS_CATALOG = {"bb-01": {"label": "A", "env_id": "CartPole-v1", "twists": []},
                 "bb-02": {"label": "B", "env_id": "CartPole-v1", "twists": []}}


def test_crossmatch_scores_each_clean_program_against_both_policies(env):
    world, paths, _ = env
    balance_code = LINEAR
    anxious_code = ('def policy(obs, P):\n    if obs[3] > P["spin"]:\n        return 0\n'
                    '    return 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] + P["w"] * obs[3] > 0 else 0\n')
    write_row(paths, "bb-01", "none", True, balance_code, BALANCE_GAINS)
    write_row(paths, "bb-02", "none", True, anxious_code, {**BALANCE_GAINS, "spin": 0.2})
    batch.crossmatch(CROSS_CATALOG, world, paths)
    matrix = json.loads((paths.results / "crossmatch.json").read_text())
    assert set(matrix) == {"bb-01 program", "bb-02 program"}
    assert matrix["bb-01 program"]["bb-01"] == 1.0 and matrix["bb-02 program"]["bb-02"] == 1.0
    assert matrix["bb-01 program"]["bb-02"] < 1.0 and matrix["bb-02 program"]["bb-01"] < 1.0


def test_crossmatch_uses_the_flawless_run_not_a_twisted_one(env):
    world, paths, _ = env
    write_row(paths, "bb-01", "t-cp-1", True, 'def policy(obs, P):\n    return 0\n', {})
    write_row(paths, "bb-01", "none", False, LINEAR, BALANCE_GAINS)
    batch.crossmatch(CROSS_CATALOG, world, paths)
    matrix = json.loads((paths.results / "crossmatch.json").read_text())
    assert matrix["bb-01 program"]["bb-01"] == 1.0


def test_crossmatch_clears_any_twist_before_measuring(env):
    world, paths, _ = env
    write_row(paths, "bb-01", "none", True, LINEAR, BALANCE_GAINS)
    batch.crossmatch(CROSS_CATALOG, world, paths)
    assert set(world.configured) >= {("bb-01", None), ("bb-02", None)}


def test_crossmatch_skips_an_env_with_only_one_policy(env):
    world, paths, _ = env
    catalog = {"bb-01": {"label": "A", "env_id": "CartPole-v1", "twists": []}}
    write_row(paths, "bb-01", "none", True, LINEAR, BALANCE_GAINS)
    batch.crossmatch(catalog, world, paths)
    assert json.loads((paths.results / "crossmatch.json").read_text()) == {}


# --- zero-query baseline -------------------------------------------------------------------------------

class BaselineBrain:
    """Answers from a script, one reply per complete() call."""
    replies = []
    seen = []

    def __init__(self, provider, *, run_id, store=None, live=False, model=None):
        self.model = "scripted-1"

    def complete(self, messages, **kwargs):
        BaselineBrain.seen.append(messages)
        return BaselineBrain.replies.pop(0)


def good_controller():
    return json.dumps({"program": {"code": LINEAR, "params": BALANCE_GAINS}})


@pytest.fixture
def baseline_env(env, monkeypatch):
    monkeypatch.setattr(batch, "Brain", BaselineBrain)
    BaselineBrain.seen, BaselineBrain.replies = [], []
    return env


def test_baseline_row_for_every_policy_from_one_brain_call_per_env(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = [good_controller()]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    rows = json.loads((paths.results / "baselines.json").read_text())["rows"]
    assert len(BaselineBrain.seen) == 1
    assert [r["bb"] for r in rows] == ["bb-01", "bb-02"]
    assert all(r["kind"] == "zero-query" for r in rows)
    first = rows[0]
    assert first["label"] == "A" and first["model"] == "scripted-1"
    assert first["agreement"] == 1.0 and first["clone_reward"] == 500.0 and first["bb_reward"] == 500.0
    assert rows[1]["agreement"] < 1.0       # same controller, other policy


def test_baseline_prompt_has_the_task_and_spec_and_no_data(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = [good_controller()]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    prompt = json.dumps(BaselineBrain.seen[0])
    assert "CartPole" in prompt and "cart_v" in prompt and "You get no data" in prompt
    assert "COUNTEREXAMPLES" not in prompt and "PATTERN HINTS" not in prompt


def test_baseline_params_are_not_fitted_because_it_gets_no_data(baseline_env):
    world, paths, _ = baseline_env
    bad_gains = {"x": 0.0, "v": 0.0, "a": -1.0, "w": -1.0}    # falls over, and a fit would repair it
    BaselineBrain.replies = [json.dumps({"program": {"code": LINEAR, "params": bad_gains}})]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    rows = json.loads((paths.results / "baselines.json").read_text())["rows"]
    assert rows[0]["clone_reward"] < 100


def test_baseline_feeds_a_rejected_program_back_to_the_brain(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = [json.dumps({"program": {"code": "import os\ndef policy(obs, P):\n    return 0\n", "params": {}}}),
                             good_controller()]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    assert "import" in BaselineBrain.seen[1][-1]["content"]
    assert json.loads((paths.results / "baselines.json").read_text())["rows"][0]["agreement"] == 1.0


def test_baseline_gives_up_with_an_honest_row_after_repeated_bad_programs(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = ["not json"] * 3
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    rows = json.loads((paths.results / "baselines.json").read_text())["rows"]
    assert all("error" in r and r["agreement"] is None for r in rows)


def test_baseline_is_not_asked_for_again_on_rerun(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = [good_controller()]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    assert len(BaselineBrain.seen) == 1


def test_baseline_error_row_is_retried_on_rerun(baseline_env):
    world, paths, _ = baseline_env
    BaselineBrain.replies = ["not json"] * 3 + [good_controller()]
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    batch.baselines(CROSS_CATALOG, world, paths, "muse")
    assert json.loads((paths.results / "baselines.json").read_text())["rows"][0]["agreement"] == 1.0


def test_numerics_is_the_real_module_not_the_stub():
    assert batch.numerics is numerics
