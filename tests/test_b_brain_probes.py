"""Probes start from counterexample states when there are any (conditional flaws), else from the median."""
import pytest

from agent import probes
from agent.ledger import Ledger
from contracts.types import Counterexample, QueryBatch, SensorSpec

SPEC = SensorSpec(obs_names=["x", "speed", "angle"], obs_low=[-1, -1, -1], obs_high=[1, 1, 1],
                  action_type="discrete", action_names=["noop", "fire", "skip"])


class ConditionalFlawBB:
    """Fires when angle > 0. Hidden flaw: when speed > 0.5 AND it would fire, it skips instead.

    The flaw only exists while firing, so a sweep over `speed` from a state where it is not firing
    never sees it."""

    def __init__(self):
        self.n = 0
        self.q = 0

    def _act(self, o):
        a = 1 if o[2] > 0 else 0
        return 2 if (o[1] > 0.5 and a == 1) else a

    def act(self, obs_batch, source="probe", rule_id=None):
        self.q += 1
        self.n += len(obs_batch)
        return QueryBatch(f"q{self.q}", [list(o) for o in obs_batch], [self._act(o) for o in obs_batch],
                          source, rule_id)

    def spec(self):
        return SPEC

    def query_count(self):
        return self.n


# most of the seen data is not firing (angle < 0), so the median state is not firing either
SEEN = [[0.0, 0.0, -0.5]] * 9 + [[0.0, 0.0, 0.4]]
FLAW_RULE = {"sweep": {"feature": "speed", "start": -1.0, "stop": 1.0, "n": 21},
             "expect": {"feature": "speed", "op": ">", "value": 0.5, "action": 2}}


def cex(i, angle=0.4):
    return Counterexample(obs=[0.1 * i, 0.7, angle], bb_action=2, clone_action=1, query_id=f"c{i}")


def test_median_only_without_counterexamples():
    bases = probes.choose_bases(FLAW_RULE, SPEC, SEEN, counterexamples=[])
    assert [b["kind"] for b in bases] == ["median"]
    pts = probes.expand(FLAW_RULE, SPEC, SEEN)                    # old call signature still works
    assert len(pts) == 21 and all(p[2] == -0.5 for p in pts)


def test_sweeps_start_from_up_to_three_counterexamples():
    five = [cex(i) for i in range(5)]
    pts, bases = probes.expand_with_bases(FLAW_RULE, SPEC, SEEN, five)
    assert [b["query_id"] for b in bases] == ["c0", "c1", "c2"]
    assert len(pts) == 3 * 21
    for k, base in enumerate(bases):                              # non-swept features come from each base
        chunk = pts[k * 21:(k + 1) * 21]
        assert {(p[0], p[2]) for p in chunk} == {(round(base["obs"][0], 6), base["obs"][2])}


def test_explicit_base_plus_counterexamples():
    probe = {**FLAW_RULE, "base": [0.0, 0.0, 0.3]}
    bases = probes.choose_bases(probe, SPEC, SEEN, [cex(i) for i in range(5)])
    assert bases[0] == {"kind": "explicit", "obs": [0.0, 0.0, 0.3], "query_id": None}
    assert [b["query_id"] for b in bases[1:]] == ["c0", "c1", "c2"]
    # explicit base alone when there are no counterexamples (never the median)
    assert [b["kind"] for b in probes.choose_bases(probe, SPEC, SEEN, [])] == ["explicit"]
    with pytest.raises(probes.ProbeError):
        probes.choose_bases({**FLAW_RULE, "base": [0.0]}, SPEC, SEEN, [cex(0)])


def test_explicit_base_equal_to_a_counterexample_is_not_swept_twice():
    probe = {**FLAW_RULE, "base": cex(0).obs}
    bases = probes.choose_bases(probe, SPEC, SEEN, [cex(0), cex(1)])
    assert [b["kind"] for b in bases] == ["explicit", "counterexample"]
    assert bases[1]["query_id"] == "c1"


def test_dormant_hand_built_base_no_longer_hides_a_conditional_flaw():
    # Muse picks a base where it is not firing (angle < 0): alone that refutes the flaw rule.
    # The counterexample sweeps show the flaw: 2 of 3 bases match -> 0.67, stays probing, not refuted.
    probe = {**FLAW_RULE, "base": [0.0, 0.0, -0.5]}
    bb = ConditionalFlawBB()
    assert probes.run_probe(probe, "r1", bb, SPEC, SEEN, counterexamples=None)["status"] == "refuted"
    res = probes.run_probe(probe, "r1", bb, SPEC, SEEN, counterexamples=[cex(0), cex(1)])
    assert res["n_points"] == 3 * 21 and res["status"] == "probing"
    assert abs(res["fraction"] - 2 / 3) < 1e-9
    assert res["base_evidence"] == ["c0", "c1"]
    assert "from the given base state + 2 counterexample states (c0, c1)" in res["description"]


def test_malformed_counterexamples_are_skipped():
    bad = Counterexample(obs=[0.1], bb_action=2, clone_action=1, query_id="bad")
    bases = probes.choose_bases(FLAW_RULE, SPEC, SEEN, [bad, {"obs": [0.0, 0.7, 0.4], "query_id": "d1"}])
    assert [b["query_id"] for b in bases] == ["d1"]
    assert [b["kind"] for b in probes.choose_bases(FLAW_RULE, SPEC, SEEN, [bad])] == ["median"]


def test_conditional_flaw_refuted_from_median_confirmed_from_counterexamples():
    bb = ConditionalFlawBB()
    from_median = probes.run_probe(FLAW_RULE, "r1", bb, SPEC, SEEN, counterexamples=None)
    assert from_median["status"] == "refuted"                     # the old behaviour: flaw invisible
    from_cex = probes.run_probe(FLAW_RULE, "r1", bb, SPEC, SEEN, counterexamples=[cex(0), cex(1)])
    assert from_cex["status"] == "confirmed" and from_cex["fraction"] == 1.0
    assert from_cex["n_points"] == 42 and from_cex["n_in_region"] == 2 * 5
    assert from_cex["base_evidence"] == ["c0", "c1"]
    assert "from 2 counterexample states (c0, c1)" in from_cex["description"]


def test_verdict_counts_points_from_every_base():
    # one base is firing (flaw shows), one is not (flaw hidden): 50% match -> stays probing
    res = probes.run_probe(FLAW_RULE, "r1", ConditionalFlawBB(), SPEC, SEEN,
                           counterexamples=[cex(0, angle=0.4), cex(1, angle=-0.4)])
    assert res["n_in_region"] == 10 and res["fraction"] == 0.5 and res["status"] == "probing"


def test_bases_recorded_as_ledger_evidence():
    lg = Ledger()
    rid = lg.ensure_rule(None, "When speed > 0.5 while firing, it skips")
    res = probes.run_probe(FLAW_RULE, rid, ConditionalFlawBB(), SPEC, SEEN, counterexamples=[cex(0), cex(1)])
    lg.verdict(rid, res["status"], res["fraction"], res["query_id"], extra_evidence=res["base_evidence"])
    e = lg.get(rid)
    assert e.status == "confirmed" and e.evidence[-3:] == [res["query_id"], "c0", "c1"]
    lg.verdict(rid, "confirmed", 1.0, res["query_id"], extra_evidence=["c0"])   # no duplicates
    assert e.evidence.count("c0") == 1


def test_system_prompt_mentions_counterexample_base():
    from agent import prompts
    assert ("If a rule only applies in a specific situation (e.g. while firing an engine), "
            "set base to a counterexample state, not the median.") in prompts.system_prompt()
