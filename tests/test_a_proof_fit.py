"""A4 Done-when (CLAUDE.md names tests/test_a_fit.py; LANES.md gives this session test_a_proof*.py)."""
import math
import random
import time

import pytest

from contracts.types import SensorSpec
from numerics.fit import agreement, fit_params
from numerics.sandbox import load_program
from tests.test_a_proof_helpers import ARM_SPEC, CARTPOLE_SPEC, LUNAR_SPEC, RuleBlackBox

LINEAR = '''
def policy(obs, P):
    score = P["k_v"] * obs[1] + P["k_a"] * obs[2] + P["k_w"] * obs[3]
    if score > P["bias"]:
        return 1
    return 0
'''

THRESHOLDS = '''
def policy(obs, P):
    if obs[3] > P["w_hi"]:
        return 1
    if obs[3] < P["w_lo"]:
        return 0
    return 1 if obs[2] > P["a_mid"] else 0
'''

ONE_PARAM = '''
def policy(obs, P):
    return 1 if obs[2] > P["limit"] else 0
'''

LANDER = '''
def policy(obs, P):
    if obs[6] > 0.5 and obs[7] > 0.5:
        return 0
    if obs[3] < P["vy_limit"]:
        return 2
    if obs[4] > P["tilt"]:
        return 3
    if obs[4] < -P["tilt"]:
        return 1
    return 0
'''


CALL_LIMIT = 15_000_000       # about 8M with the restart limit, about 28M without it


def sample(spec: SensorSpec, n: int, seed: int = 1, shrink: float = 0.5) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(lo * shrink, hi * shrink) for lo, hi in zip(spec.obs_low, spec.obs_high)]
            for _ in range(n)]


def labelled(code, true_params, spec, n=4000):
    fn = load_program(code, spec)
    bb = RuleBlackBox(fn, true_params, spec)
    X = sample(spec, n)
    return fn, X, bb.act(X, source="rollout").actions


# --- the spec's Done-when ------------------------------------------------------------------------------

@pytest.mark.parametrize("code, truth, wrong, spec", [
    (LINEAR, {"k_v": 0.3, "k_a": 2.0, "k_w": 0.8, "bias": 0.1},
     {"k_v": -1.0, "k_a": 0.1, "k_w": -0.4, "bias": 1.0}, CARTPOLE_SPEC),
    (THRESHOLDS, {"w_hi": 0.9, "w_lo": -0.7, "a_mid": 0.05},
     {"w_hi": -0.4, "w_lo": 0.6, "a_mid": 0.15}, CARTPOLE_SPEC),
    (LANDER, {"vy_limit": -0.3, "tilt": 0.2}, {"vy_limit": 0.6, "tilt": 1.5}, LUNAR_SPEC),
], ids=["linear", "thresholds", "lander-4-actions"])
def test_fit_raises_agreement_from_below_07_to_above_097(code, truth, wrong, spec):
    fn, X, y = labelled(code, truth, spec)
    before = agreement(fn, wrong, X, y, spec)
    params, after = fit_params(fn, wrong, X, y, spec, max_seconds=4.0)
    assert before < 0.7, f"bad fixture, start agreement {before:.3f}"
    assert after > 0.97
    assert agreement(fn, params, X, y, spec) == pytest.approx(after)


# --- traps ---------------------------------------------------------------------------------------------

def test_single_param_program_fits():
    fn, X, y = labelled(ONE_PARAM, {"limit": 0.05}, CARTPOLE_SPEC)
    params, after = fit_params(fn, {"limit": 0.2}, X, y, CARTPOLE_SPEC, max_seconds=3.0)
    assert after > 0.97
    assert params["limit"] == pytest.approx(0.05, abs=0.02)


def test_fit_escapes_a_flat_start():
    # Threshold starts far outside the data, so every nearby candidate scores the same.
    fn, X, y = labelled(ONE_PARAM, {"limit": 0.05}, CARTPOLE_SPEC)
    start = {"limit": 3.0}
    assert agreement(fn, start, X, y, CARTPOLE_SPEC) < 0.7
    _, after = fit_params(fn, start, X, y, CARTPOLE_SPEC, max_seconds=3.0)
    assert after > 0.97


def test_fit_never_makes_things_worse():
    fn, X, y = labelled(LINEAR, {"k_v": 0.3, "k_a": 2.0, "k_w": 0.8, "bias": 0.1}, CARTPOLE_SPEC)
    perfect = {"k_v": 0.3, "k_a": 2.0, "k_w": 0.8, "bias": 0.1}
    params, after = fit_params(fn, perfect, X, y, CARTPOLE_SPEC, max_seconds=0.2)
    assert after == 1.0 and agreement(fn, params, X, y, CARTPOLE_SPEC) == 1.0


def test_fit_is_reproducible():
    fn, X, y = labelled(LINEAR, {"k_v": 0.3, "k_a": 2.0, "k_w": 0.8, "bias": 0.1}, CARTPOLE_SPEC)
    start = {"k_v": -1.0, "k_a": 0.1, "k_w": -0.4, "bias": 1.0}
    # A generation cap would be needed for byte equality under a time limit, so compare a run that
    # converges well before the cap: both must land on the same answer.
    a = fit_params(fn, start, X[:400], y[:400], CARTPOLE_SPEC, max_seconds=30.0)
    b = fit_params(fn, start, X[:400], y[:400], CARTPOLE_SPEC, max_seconds=30.0)
    assert a == b


def test_time_cap_holds_with_a_slow_program():
    slow = '''
def policy(obs, P):
    a = np.tanh(np.dot(obs, [P["a"], P["b"], P["c"], P["d"]]))
    b = np.tanh(np.dot(obs, [P["d"], P["c"], P["b"], P["a"]]))
    return 1 if a + b > P["e"] else 0
'''
    fn = load_program(slow, CARTPOLE_SPEC)
    truth = {"a": 0.3, "b": 0.5, "c": 2.0, "d": 0.7, "e": 0.1}
    X = sample(CARTPOLE_SPEC, 4000)
    y = RuleBlackBox(fn, truth, CARTPOLE_SPEC).act(X).actions
    t0 = time.time()
    fit_params(fn, {k: 1.5 for k in truth}, X, y, CARTPOLE_SPEC, max_seconds=1.0)
    assert time.time() - t0 < 1.5


def test_fit_that_cannot_improve_stops_well_before_a_long_cap():
    # The program can't express the rule, so 100% is out of reach. A live run has about 75 s for everything,
    # so restarts that find nothing must not keep spending the cap. Counts calls, not seconds, because this
    # laptop is often running other sessions.
    program = load_program(LINEAR, CARTPOLE_SPEC)
    calls = {"n": 0}

    def counted(obs, P):
        calls["n"] += 1
        return program(obs, P)

    counted.param_names = program.param_names
    X = sample(CARTPOLE_SPEC, 3000)
    y = [1 if 0.3 * o[1] + 2.0 * o[2] + 0.8 * o[3] + 0.4 * math.sin(6.0 * o[3]) > 0.1 else 0 for o in X]
    start = {"k_v": -1.0, "k_a": 0.1, "k_w": -0.4, "bias": 1.0}
    _, after = fit_params(counted, start, X, y, CARTPOLE_SPEC, max_seconds=60.0)
    assert after > 0.9
    assert calls["n"] < CALL_LIMIT


def test_bad_param_region_scores_zero_and_never_raises():
    crashy = '''
def policy(obs, P):
    return 1 if obs[2] > math.log(P["limit"]) else 0
'''
    fn = load_program(crashy, CARTPOLE_SPEC)
    X = sample(CARTPOLE_SPEC, 500)
    y = [1 if o[2] > -2.0 else 0 for o in X]
    assert agreement(fn, {"limit": -1.0}, X, y, CARTPOLE_SPEC) == 0.0   # math domain error on every point
    params, after = fit_params(fn, {"limit": 0.2}, X, y, CARTPOLE_SPEC, max_seconds=1.0)
    assert after >= agreement(fn, {"limit": 0.2}, X, y, CARTPOLE_SPEC)
    assert params["limit"] > 0


def test_param_the_brain_forgot_is_added_and_fitted():
    fn, X, y = labelled(ONE_PARAM, {"limit": 0.05}, CARTPOLE_SPEC)
    params, after = fit_params(fn, {}, X, y, CARTPOLE_SPEC, max_seconds=3.0)
    assert "limit" in params and after > 0.97


def test_unused_param_is_kept_but_not_moved():
    fn, X, y = labelled(ONE_PARAM, {"limit": 0.05}, CARTPOLE_SPEC)
    params, _ = fit_params(fn, {"limit": 0.2, "unused": 7.0}, X, y, CARTPOLE_SPEC, max_seconds=1.0)
    assert params["unused"] == 7.0


def test_program_with_no_params_returns_its_agreement():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    assert fit_params(fn, {}, [[0.0] * 4] * 4, [1, 1, 1, 0], CARTPOLE_SPEC, max_seconds=1.0) == ({}, 0.75)


def test_returned_agreement_is_on_all_of_X_not_the_subsample():
    # Every point past the first 3,000 agrees only under a specific value the subsample still sees most of.
    fn = load_program(ONE_PARAM, CARTPOLE_SPEC)
    X = sample(CARTPOLE_SPEC, 6000)
    y = [1 if o[2] > 0.05 else 0 for o in X]
    params, after = fit_params(fn, {"limit": 0.1}, X, y, CARTPOLE_SPEC, max_seconds=2.0)
    assert after == agreement(fn, params, X, y, CARTPOLE_SPEC)


def test_params_come_back_as_plain_floats():
    fn, X, y = labelled(ONE_PARAM, {"limit": 0.05}, CARTPOLE_SPEC)
    params, after = fit_params(fn, {"limit": 0.2}, X[:300], y[:300], CARTPOLE_SPEC, max_seconds=1.0)
    assert type(params["limit"]) is float and type(after) is float


# --- agreement -----------------------------------------------------------------------------------------

def test_agreement_is_exact_match_fraction():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    assert agreement(fn, {}, [[0.0] * 4] * 4, [1, 1, 1, 0], CARTPOLE_SPEC) == 0.75


def test_agreement_on_no_data_is_zero():
    fn = load_program("def policy(obs, P):\n    return 1\n", CARTPOLE_SPEC)
    assert agreement(fn, {}, [], [], CARTPOLE_SPEC) == 0.0


def test_continuous_agreement_uses_l2_within_a_tenth_of_the_range():
    # ARM_SPEC ranges: torque 4.0, brake 1.0. Normalised L2 error must be <= 0.1.
    fn = load_program("def policy(obs, P):\n    return [0.0, 0.5]\n", ARM_SPEC)
    y = [
        [0.3, 0.5],     # 0.3 / 4 = 0.075                    inside
        [0.0, 0.59],    # 0.09 / 1 = 0.09                    inside
        [0.5, 0.5],     # 0.5 / 4 = 0.125                    outside
        [0.3, 0.58],    # sqrt(0.075^2 + 0.08^2) = 0.1097    outside: each axis alone would pass
    ]
    assert agreement(fn, {}, [[0.0, 0.0]] * 4, y, ARM_SPEC) == 0.5


def test_continuous_fit_recovers_gains():
    code = '''
def policy(obs, P):
    return [P["kp"] * obs[0] + P["kd"] * obs[1], P["b0"] + P["b1"] * abs(obs[0])]
'''
    truth = {"kp": 1.4, "kd": -0.6, "b0": 0.2, "b1": 0.5}
    fn, X, y = labelled(code, truth, ARM_SPEC, n=1000)
    start = {"kp": -0.5, "kd": 0.9, "b0": 0.9, "b1": -0.2}
    assert agreement(fn, start, X, y, ARM_SPEC) < 0.7
    params, after = fit_params(fn, start, X, y, ARM_SPEC, max_seconds=3.0)
    assert after > 0.97
    assert params["kp"] == pytest.approx(1.4, abs=0.05)
