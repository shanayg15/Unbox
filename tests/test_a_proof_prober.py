"""A5: find_disagreements. Done-when on RuleBlackBox per LANES.md; re-check on real bb-01 at integration."""
import math
import random

import numpy as np
import pytest

from contracts.types import SensorSpec
from numerics.fit import agreement, fit_params
from numerics.prober import find_disagreements
from numerics.sandbox import load_program
from tests.test_a_proof_helpers import ARM_SPEC, CARTPOLE_SPEC, LUNAR_SPEC, RuleBlackBox

CLONE = '''
def policy(obs, P):
    score = P["k_x"] * obs[0] + P["k_v"] * obs[1] + P["k_a"] * obs[2] + P["k_w"] * obs[3]
    return 1 if score > 0 else 0
'''
GAINS = {"x": -0.35, "v": 0.51, "a": 3.9, "w": 0.45}
CLONE_START = {"k_x": -0.2, "k_v": 0.2, "k_a": 2.0, "k_w": 1.0}
MOVING_FAST = {"condition": lambda o: o[1] > 0.5, "override": lambda o, a: 0}


def rule(obs, P):
    return 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] + P["w"] * obs[3] > 0 else 0


# Terms a linear clone can't express, so its fit is imperfect the way a fit to a real net is. Each one also
# keeps the twist firing in rollouts, which a real twist is tuned to do (5-30% of steps, CLAUDE.md A2).
EXTRA_TERMS = {
    "wobble": lambda o: 0.08 * math.sin(6.0 * o[3]),
    "soft-spin": lambda o: -0.1 * math.sin(3.0 * o[3]),
    "mixed": lambda o: -0.25 * math.sin(5.0 * o[3]) * math.cos(3.0 * o[1]),
    "coupled": lambda o: 0.2 * math.cos(8.0 * o[3]) * math.sin(4.0 * o[0]),
}


def rule_plus(extra):
    return lambda obs, P: 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] \
        + P["w"] * obs[3] + extra(obs) > 0 else 0


def rollouts(bb, seeds=range(8)):
    X, y = [], []
    for seed in seeds:
        episode = bb.episode(seed)
        X += episode.obs
        y += episode.actions
    return X, y


def clean_fit(rule_fn):
    """A program fit only on clean behaviour: it has never seen the twist."""
    fn = load_program(CLONE, CARTPOLE_SPEC)
    X, y = rollouts(RuleBlackBox(rule_fn, GAINS, CARTPOLE_SPEC))
    params, score = fit_params(fn, CLONE_START, X, y, CARTPOLE_SPEC, max_seconds=30.0)   # converges, so repeatable
    return fn, params, score


def counterexamples_inside_twist(rule_fn) -> tuple[int, int, float]:
    fn, params, score = clean_fit(rule_fn)
    twisted = RuleBlackBox(rule_fn, GAINS, CARTPOLE_SPEC, twist=MOVING_FAST)
    seed_obs, _ = rollouts(twisted)
    found = find_disagreements(fn, params, twisted, seed_obs, CARTPOLE_SPEC, k=8, max_queries=400)
    return sum(c.obs[1] > 0.5 for c in found), len(found), score


def test_done_when_counterexamples_land_in_the_twist_region():
    inside, total, score = counterexamples_inside_twist(rule)
    assert score > 0.97, f"bad fixture, clean fit only reached {score:.3f}"
    assert total == 8
    assert inside / total >= 0.8


# When the clone is also wrong about clean behaviour, those errors are real, thick, stable regions, and no
# black-box signal can tell them from a twist. The share in the twist falls, so this pins a floor: the twist
# must still surface. Without support ranking soft-spin scores 3/8 and mixed 4/8, so their floors catch that.
@pytest.mark.parametrize("name, at_least", [("wobble", 1), ("soft-spin", 5), ("mixed", 5), ("coupled", 1)])
def test_imperfect_clone_still_surfaces_the_twist(name, at_least):
    inside, total, score = counterexamples_inside_twist(rule_plus(EXTRA_TERMS[name]))
    assert 0.85 < score < 0.97, f"bad fixture, clean fit reached {score:.3f}"
    assert total == 8 and inside >= at_least


def test_thin_boundary_slivers_do_not_crowd_out_a_real_region():
    # The clone's boundary is slightly off everywhere and a thick twisted block sits to one side.
    spec = SensorSpec(["a", "b"], [-1.0, -1.0], [1.0, 1.0], "discrete", ["zero", "one"])
    clone = load_program('def policy(obs, P):\n    return 1 if obs[0] > P["t"] else 0\n', spec)
    block = {"condition": lambda o: o[1] > 0.4, "override": lambda o, a: 1 - a}
    bb = RuleBlackBox(lambda o, P: 1 if o[0] > 0.0 else 0, {}, spec, twist=block)
    rng = random.Random(0)
    seed_obs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(1000)]
    found = find_disagreements(clone, {"t": 0.02}, bb, seed_obs, spec, k=8, max_queries=400)
    assert len(found) == 8
    assert sum(c.obs[1] > 0.4 for c in found) >= 7


@pytest.mark.parametrize("max_queries", [0, 1, 2, 7, 50, 399, 400])
def test_never_exceeds_the_query_budget(max_queries):
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 1 if o[1] < 0.2 else 0, {}, CARTPOLE_SPEC)
    rng = random.Random(1)
    seed_obs = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(300)]
    find_disagreements(fn, {}, bb, seed_obs, CARTPOLE_SPEC, k=8, max_queries=max_queries)
    assert bb.query_count() <= max_queries


def test_a_large_budget_is_actually_spent_looking():
    # A perfect clone means nothing is found, so the whole budget goes to searching.
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 1, {}, CARTPOLE_SPEC)
    rng = random.Random(1)
    seed_obs = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(300)]
    assert find_disagreements(fn, {}, bb, seed_obs, CARTPOLE_SPEC, max_queries=400) == []
    assert bb.query_count() == 400


def test_counterexamples_are_true_and_traceable():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 0 if o[1] > 0.3 else 1, {}, CARTPOLE_SPEC)
    rng = random.Random(2)
    seed_obs = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(300)]
    found = find_disagreements(fn, {}, bb, seed_obs, CARTPOLE_SPEC, k=8, max_queries=200)
    assert found
    by_id = {batch.query_id: batch for batch in bb.log}
    for c in found:
        assert c.bb_action == 0 and c.clone_action == 1 and c.obs[1] > 0.3
        batch = by_id[c.query_id]
        assert c.obs in batch.obs and batch.actions[batch.obs.index(c.obs)] == 0


def test_queries_are_tagged_prober():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    find_disagreements(fn, {}, bb, [[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]], CARTPOLE_SPEC, max_queries=20)
    assert bb.log and all(batch.source == "prober" for batch in bb.log)


def lander_seeds(n=400):
    rng = random.Random(3)
    return [[rng.uniform(-1, 1), rng.uniform(0, 1), rng.gauss(0, 0.3), rng.gauss(-0.3, 0.2),
             rng.gauss(0, 0.2), rng.gauss(0, 0.2), float(rng.random() < 0.1), float(rng.random() < 0.1)]
            for _ in range(n)]


def test_binary_features_only_ever_take_their_observed_values():
    fn = load_program("def policy(obs, P):\n    return 0\n", LUNAR_SPEC)
    # Disagree everywhere, so the refinement round runs too.
    bb = RuleBlackBox(lambda o, P: 2, {}, LUNAR_SPEC)
    find_disagreements(fn, {}, bb, lander_seeds(), LUNAR_SPEC, max_queries=400)
    queried = [o for batch in bb.log for o in batch.obs]
    assert len(queried) > 100
    assert {o[6] for o in queried} <= {0.0, 1.0} and {o[7] for o in queried} <= {0.0, 1.0}


def test_binary_feature_is_swept_so_a_rare_value_gets_tested():
    # Both legs down is uncommon in seeds; a rule that only differs there must still be found.
    fn = load_program("def policy(obs, P):\n    return 0\n", LUNAR_SPEC)
    bb = RuleBlackBox(lambda o, P: 2 if o[6] == 1.0 else 0, {}, LUNAR_SPEC)
    seeds = lander_seeds()
    seeds = [s[:6] + [0.0, s[7]] for s in seeds[:-1]] + [seeds[-1][:6] + [1.0, 0.0]]
    found = find_disagreements(fn, {}, bb, seeds, LUNAR_SPEC, max_queries=60)
    assert found and all(c.obs[6] == 1.0 for c in found)


def test_median_of_a_half_and_half_binary_feature_is_not_a_fraction():
    fn = load_program("def policy(obs, P):\n    return 0\n", LUNAR_SPEC)
    bb = RuleBlackBox(lambda o, P: 0, {}, LUNAR_SPEC)
    seeds = [s[:6] + [float(i % 2), 0.0] for i, s in enumerate(lander_seeds())]
    find_disagreements(fn, {}, bb, seeds, LUNAR_SPEC, max_queries=200)
    assert {o[6] for batch in bb.log for o in batch.obs} <= {0.0, 1.0}


def test_sweeps_use_the_data_range_not_infinite_spec_bounds():
    spec = SensorSpec(["a", "b"], [-math.inf, -math.inf], [math.inf, math.inf], "discrete", ["zero", "one"])
    fn = load_program("def policy(obs, P):\n    return 0\n", spec)
    bb = RuleBlackBox(lambda o, P: 0, {}, spec)
    rng = random.Random(4)
    seeds = [[rng.gauss(0, 1), rng.gauss(10, 0.5)] for _ in range(500)]
    find_disagreements(fn, {}, bb, seeds, spec, max_queries=200)
    queried = np.array([o for batch in bb.log for o in batch.obs])
    assert np.isfinite(queried).all()
    assert queried[:, 1].min() > 7.0 and queried[:, 1].max() < 13.0
    assert queried[:, 1].max() - queried[:, 1].min() > 1.5      # the sweep really spanned the feature


def test_diverse_selection_represents_every_region():
    spec = SensorSpec(["a", "b"], [-1.0, -1.0], [1.0, 1.0], "discrete", ["zero", "one"])
    fn = load_program("def policy(obs, P):\n    return 0\n", spec)
    bb = RuleBlackBox(lambda o, P: 1 if o[0] > 0.6 or o[0] < -0.6 else 0, {}, spec)
    rng = random.Random(5)
    seeds = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(1000)]
    found = find_disagreements(fn, {}, bb, seeds, spec, k=6, max_queries=400)
    assert len(found) == 6
    assert any(c.obs[0] > 0.6 for c in found) and any(c.obs[0] < -0.6 for c in found)
    gaps = [math.dist(a.obs, b.obs) for i, a in enumerate(found) for b in found[i + 1:]]
    assert min(gaps) > 0.05         # k spread-out points, not k copies of one


def test_never_returns_more_than_k_and_handles_empty_input():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    rng = random.Random(6)
    seeds = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(100)]
    assert len(find_disagreements(fn, {}, bb, seeds, CARTPOLE_SPEC, k=3, max_queries=100)) == 3
    assert find_disagreements(fn, {}, bb, seeds, CARTPOLE_SPEC, k=0, max_queries=100) == []
    before = bb.query_count()
    assert find_disagreements(fn, {}, bb, [], CARTPOLE_SPEC, k=8, max_queries=100) == []
    assert bb.query_count() == before


def test_a_single_seed_obs_still_works():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    found = find_disagreements(fn, {}, bb, [[0.1, 0.2, 0.3, 0.4]], CARTPOLE_SPEC, max_queries=50)
    assert [c.obs for c in found] == [[0.1, 0.2, 0.3, 0.4]]


def test_clone_crash_on_a_candidate_is_not_fatal():
    # Crashes wherever the first feature is positive: those points carry no clone action to report.
    fn = load_program("def policy(obs, P):\n    x = 1.0 / (obs[0] - 7.0)\n    return 1\n", CARTPOLE_SPEC)
    crashy = lambda o, P: fn(o, P) if o[0] <= 0 else 1 // 0
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    rng = random.Random(7)
    seeds = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(100)]
    found = find_disagreements(crashy, {}, bb, seeds, CARTPOLE_SPEC, max_queries=100)
    assert found and all(c.obs[0] <= 0 for c in found)


def test_continuous_differences_inside_the_tolerance_are_not_disagreements():
    fn = load_program("def policy(obs, P):\n    return [obs[0], 0.5]\n", ARM_SPEC)
    # torque range 4.0: off by 0.2 is 0.05 of the range (fine); off by 0.8 is 0.2 (a disagreement).
    bb = RuleBlackBox(lambda o, P: [o[0] + (0.8 if o[1] > 0.5 else 0.2), 0.5], {}, ARM_SPEC)
    rng = random.Random(8)
    seeds = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(500)]
    found = find_disagreements(fn, {}, bb, seeds, ARM_SPEC, k=8, max_queries=300)
    assert found and all(c.obs[1] > 0.5 for c in found)
    assert all(type(c.clone_action) is list for c in found)


def test_same_inputs_give_the_same_answer():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    rng = random.Random(9)
    seeds = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(100)]
    runs = []
    for _ in range(2):
        bb = RuleBlackBox(lambda o, P: 0 if o[1] > 0 else 1, {}, CARTPOLE_SPEC)
        runs.append([c.obs for c in find_disagreements(fn, {}, bb, seeds, CARTPOLE_SPEC, max_queries=100)])
    assert runs[0] == runs[1]
