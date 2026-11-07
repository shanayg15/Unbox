"""A5: evaluate, shield and the numerics package surface."""
import inspect
import time

import pytest

import contracts.numerics_api as api
import numerics
from contracts.types import Episode, EvalReport, QueryBatch
from numerics.evaluate import evaluate
from numerics.sandbox import load_program
from numerics.shield import shielded_action
from tests.test_a_proof_helpers import ARM_SPEC, CARTPOLE_SPEC, RuleBlackBox, make_env_factory

BALANCE = '''
def policy(obs, P):
    return 1 if P["x"] * obs[0] + P["v"] * obs[1] + P["a"] * obs[2] + P["w"] * obs[3] > 0 else 0
'''
GAINS = {"x": 1.0, "v": 2.0, "a": 30.0, "w": 7.0}
ALWAYS_LEFT = "def policy(obs, P):\n    return 0\n"


class FakeEnv:
    """Pays reward 1 per step for `steps` steps, recording what it was given."""

    def __init__(self, steps):
        self.steps, self.count, self.closed, self.actions = steps, 0, False, []

    def step(self, action):
        self.count += 1
        self.actions.append(action)
        return [0.0, 0.0], 1.0, self.count >= self.steps, False, {}

    def close(self):
        self.closed = True


class FixedRewardBox:
    """A black box whose only job is to report a known episode reward and inference time."""

    def __init__(self, reward, inference_us=None):
        self.reward, self.inference_us, self.queries = reward, inference_us, 0

    def spec(self):
        return ARM_SPEC

    def act(self, obs_batch, source="probe", rule_id=None):
        self.queries += len(obs_batch)
        return QueryBatch("q1", list(obs_batch), [0] * len(obs_batch), source, rule_id, self.inference_us)

    def episode(self, seed, render=False, frame_every=2):
        return Episode(seed, [], [], [], self.reward, True, False)

    def query_count(self):
        return self.queries


def holdout_for(bb, seeds=(100, 101)):
    batches = []
    for seed in seeds:
        episode = bb.episode(seed)
        batches.append(QueryBatch(f"h{seed}", episode.obs, episode.actions, "holdout"))
    return batches


def test_done_when_full_report_under_20_seconds_for_20_seeds():
    fn = load_program(BALANCE, CARTPOLE_SPEC)
    bb = RuleBlackBox(fn, GAINS, CARTPOLE_SPEC)
    holdout = holdout_for(bb)
    t0 = time.time()
    report = evaluate(fn, GAINS, bb, CARTPOLE_SPEC, holdout, make_env_factory(), list(range(20)))
    seconds = time.time() - t0
    assert seconds < 20.0
    assert isinstance(report, EvalReport)
    assert report.open_loop_agreement == 1.0
    assert report.n_holdout_obs == 1000
    assert report.clone_mean_reward == 500.0 and report.bb_mean_reward == 500.0
    assert report.n_episodes == 20 and report.n_failures == 0
    assert report.failure_rate_bound_95 == pytest.approx(0.15)
    assert report.us_per_action_clone > 0
    assert report.us_per_action_bb == 40.0


def test_a_clone_that_falls_over_counts_as_failures_and_has_no_rule_of_three_bound():
    clone = load_program(ALWAYS_LEFT, CARTPOLE_SPEC)
    bb = RuleBlackBox(load_program(BALANCE, CARTPOLE_SPEC), GAINS, CARTPOLE_SPEC)
    report = evaluate(clone, {}, bb, CARTPOLE_SPEC, holdout_for(bb, (100,)), make_env_factory(), [0, 1, 2])
    assert report.n_failures == 3 and report.n_episodes == 3
    assert report.failure_rate_bound_95 is None
    assert report.clone_mean_reward < 20 and report.bb_mean_reward == 500.0
    assert 0.3 < report.open_loop_agreement < 0.7


@pytest.mark.parametrize("bb_reward, clone_steps, is_failure", [
    (500.0, 451, False),    # margin 50: short by 49
    (500.0, 450, False),    # short by exactly the margin: not below it
    (500.0, 449, True),     # short by 51
    (20.0, 11, False),      # margin is the floor of 10, short by 9
    (20.0, 10, False),
    (20.0, 9, True),
    (-200.0, 1, False),     # a negative bb reward: clone beat it
])
def test_failure_uses_the_larger_of_10_and_a_tenth_of_bb_reward(bb_reward, clone_steps, is_failure):
    fn = load_program(ALWAYS_LEFT, CARTPOLE_SPEC)
    factory = lambda seed: (FakeEnv(clone_steps), [0.0, 0.0, 0.0, 0.0])
    report = evaluate(fn, {}, FixedRewardBox(bb_reward), CARTPOLE_SPEC, [], factory, [0])
    assert report.n_failures == (1 if is_failure else 0)
    assert report.clone_mean_reward == clone_steps


def test_every_env_is_closed_even_when_the_clone_crashes_mid_episode():
    envs = []

    def factory(seed):
        envs.append(FakeEnv(50))
        return envs[-1], [0.0, 0.0, 0.0, 0.0]

    calls = {"n": 0}

    def crashes_on_third_step(obs, P):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("boom")
        return 0

    report = evaluate(crashes_on_third_step, {}, FixedRewardBox(50.0), CARTPOLE_SPEC, [], factory, [0, 1])
    assert len(envs) == 2 and all(env.closed for env in envs)
    assert report.n_failures == 2       # the crash ended the episode early, far below 50


def test_same_seed_reaches_both_the_black_box_and_the_local_env():
    seen = []

    def factory(seed):
        seen.append(seed)
        return FakeEnv(1), [0.0, 0.0, 0.0, 0.0]

    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    fn = load_program(ALWAYS_LEFT, CARTPOLE_SPEC)
    evaluate(fn, {}, bb, CARTPOLE_SPEC, [], factory, [3, 9])
    assert seen == [3, 9] and bb.episode_seeds == [3, 9]


def test_clone_sees_plain_floats_on_every_step():
    # gymnasium hands back numpy arrays; a program's obs must be a plain list of plain floats.
    seen = []

    def spy(obs, P):
        seen.append(obs)
        return 0

    bb = FixedRewardBox(10.0)
    evaluate(spy, {}, bb, CARTPOLE_SPEC, [], make_env_factory(), [0])
    assert seen and all(type(o) is list and all(type(v) is float for v in o) for o in seen)


def test_clone_timing_is_over_at_least_1000_calls():
    calls = {"n": 0}

    def counted(obs, P):
        calls["n"] += 1
        return 0

    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    holdout = [QueryBatch("h", [[0.0] * 4] * 3, [0, 0, 0], "holdout")]
    report = evaluate(counted, {}, bb, CARTPOLE_SPEC, holdout, make_env_factory(), [])
    assert calls["n"] >= 1000       # agreement's 3 calls are in here too
    assert report.us_per_action_clone > 0


def test_bb_time_is_the_obs_weighted_mean_of_holdout_batches():
    holdout = [QueryBatch("a", [[0.0] * 4] * 1, [0], "holdout", inference_us=100.0),
               QueryBatch("b", [[0.0] * 4] * 3, [0, 0, 0], "holdout", inference_us=20.0),
               QueryBatch("c", [[0.0] * 4] * 5, [0] * 5, "holdout")]        # no timing on this one
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    report = evaluate(load_program(ALWAYS_LEFT, CARTPOLE_SPEC), {}, bb, CARTPOLE_SPEC, holdout, make_env_factory(), [])
    assert report.us_per_action_bb == 40.0
    assert bb.query_count() == 0    # the timing was already there, so nothing extra was asked


def test_bb_time_is_measured_with_one_small_query_when_holdout_has_none():
    # Holdout built from /episode carries no inference_us, and that's the usual case.
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC, inference_us=55.0)
    holdout = [QueryBatch("h", [[0.0] * 4] * 500, [0] * 500, "holdout")]
    report = evaluate(load_program(ALWAYS_LEFT, CARTPOLE_SPEC), {}, bb, CARTPOLE_SPEC, holdout, make_env_factory(), [])
    assert report.us_per_action_bb == 55.0
    assert 0 < bb.query_count() <= 64
    assert bb.log[-1].source == "holdout"


def test_bb_time_stays_none_when_the_server_reports_none():
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC, inference_us=None)
    report = evaluate(load_program(ALWAYS_LEFT, CARTPOLE_SPEC), {}, bb, CARTPOLE_SPEC, [], make_env_factory(), [])
    assert report.us_per_action_bb is None


def test_no_holdout_and_no_seeds_is_an_honest_empty_report():
    bb = RuleBlackBox(lambda o, P: 0, {}, CARTPOLE_SPEC)
    report = evaluate(load_program(ALWAYS_LEFT, CARTPOLE_SPEC), {}, bb, CARTPOLE_SPEC, [], make_env_factory(), [])
    assert report.open_loop_agreement == 0.0 and report.n_holdout_obs == 0
    assert report.clone_mean_reward is None and report.bb_mean_reward is None
    assert report.n_episodes == 0 and report.n_failures == 0
    assert report.failure_rate_bound_95 is None       # no evidence, so no bound


def test_continuous_actions_are_clipped_to_bounds_before_the_env_sees_them():
    env = FakeEnv(1)
    fn = load_program("def policy(obs, P):\n    return [9.0, -9.0]\n", ARM_SPEC)
    evaluate(fn, {}, FixedRewardBox(1.0), ARM_SPEC, [], lambda seed: (env, [0.0, 0.0]), [0])
    assert [float(v) for v in env.actions[0]] == [2.0, 0.0]


# --- shield --------------------------------------------------------------------------------------------

def rule_at(limit):
    return lambda obs, P: 1 if obs[0] > limit else 0


@pytest.mark.parametrize("obs, bb_action, want", [
    ([0.0], 7, 7),      # recovered == fixed (both 0): the black box acts, even with an action neither chose
    ([2.0], 7, 7),      # recovered == fixed (both 1): the black box acts
    ([0.7], 7, 0),      # recovered 1, fixed 0: the fix is what changed here, so the fix wins
])
def test_shield_overrides_only_where_the_fix_changed_the_decision(obs, bb_action, want):
    assert shielded_action(rule_at(0.5), {}, rule_at(1.0), {}, obs, bb_action) == want


def test_shield_passes_params_to_each_program():
    recovered = lambda obs, P: 1 if obs[0] > P["t"] else 0
    assert shielded_action(recovered, {"t": 0.5}, recovered, {"t": 1.0}, [0.7], 7) == 0
    assert shielded_action(recovered, {"t": 1.0}, recovered, {"t": 1.0}, [0.7], 7) == 7


def test_shield_compares_continuous_actions():
    a, b = (lambda obs, P: [0.0, 1.0]), (lambda obs, P: [0.0, 2.0])
    assert shielded_action(a, {}, b, {}, [0.0], [9.0, 9.0]) == [0.0, 2.0]
    assert shielded_action(a, {}, a, {}, [0.0], [9.0, 9.0]) == [9.0, 9.0]


def test_shield_falls_back_to_the_black_box_when_a_program_crashes():
    def crash(obs, P):
        raise ValueError("boom")

    assert shielded_action(crash, {}, rule_at(0.0), {}, [1.0], 7) == 7
    assert shielded_action(rule_at(0.0), {}, crash, {}, [1.0], 7) == 7


# --- package surface -----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["load_program", "agreement", "fit_params", "pattern_hints",
                                  "find_disagreements", "evaluate", "shielded_action"])
def test_numerics_matches_the_contract_signature(name):
    ours = inspect.signature(getattr(numerics, name)).parameters
    theirs = inspect.signature(getattr(api, name)).parameters
    assert [(p.name, p.default) for p in ours.values()] == [(p.name, p.default) for p in theirs.values()]


def test_program_error_is_the_contract_class():
    assert numerics.ProgramError is api.ProgramError
