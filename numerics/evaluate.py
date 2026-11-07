"""Lane A, task A5: evaluator"""
import time
from typing import Callable

import numpy as np

from contracts.types import Action, BlackBox, EvalReport, Params, ProgramFn, QueryBatch, SensorSpec
from numerics.fit import agreement
from numerics.sandbox import sample_obs

N_TIMING_CALLS = 1000
N_TIMING_QUERY = 64


def _env_action(action: Action, spec: SensorSpec):
    if spec.action_type == "discrete":
        return action
    values = np.asarray(action, dtype=np.float32)
    if spec.action_low is not None and spec.action_high is not None:
        values = np.clip(values, spec.action_low, spec.action_high)
    return values


def _drive_clone(fn: ProgramFn, params: Params, env_factory: Callable[[int], object], seed: int,
                 spec: SensorSpec) -> float:
    """Total reward with the clone in control. A crash ends the episode, and the reward so far stands."""
    env, obs = env_factory(seed)
    total = 0.0
    try:
        done = False
        while not done:
            try:
                action = fn([float(v) for v in obs], params)
            except Exception:
                break
            obs, reward, terminated, truncated, _ = env.step(_env_action(action, spec))
            total += float(reward)
            done = terminated or truncated
    finally:
        env.close()
    return total


def _is_failure(clone_reward: float, bb_reward: float) -> bool:
    return clone_reward < bb_reward - max(10.0, 0.1 * abs(bb_reward))


def _time_clone(fn: ProgramFn, params: Params, X: list, spec: SensorSpec) -> float:
    states = X or sample_obs(spec, N_TIMING_QUERY)
    start = time.perf_counter()
    for i in range(N_TIMING_CALLS):
        try:
            fn(states[i % len(states)], params)
        except Exception:
            pass
    return (time.perf_counter() - start) / N_TIMING_CALLS * 1e6


def _bb_inference_us(bb: BlackBox, holdout: list[QueryBatch], X: list, spec: SensorSpec) -> float | None:
    timed = [(b.inference_us, len(b.obs)) for b in holdout if b.inference_us is not None and b.obs]
    if not timed:
        # Holdout built from /episode has no timing, so ask the server for a small sample.
        states = X[:N_TIMING_QUERY] or sample_obs(spec, N_TIMING_QUERY)
        batch = bb.act(states, source="holdout")
        return batch.inference_us
    return sum(us * n for us, n in timed) / sum(n for _, n in timed)


def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
             holdout: list[QueryBatch], env_factory: Callable[[int], object],
             seeds: list[int]) -> EvalReport:
    """env_factory(seed) -> gymnasium env already reset with that seed; returns (env, first_obs)"""
    X = [obs for batch in holdout for obs in batch.obs]
    y = [action for batch in holdout for action in batch.actions]

    bb_rewards, clone_rewards, n_failures = [], [], 0
    for seed in seeds:
        bb_reward = float(bb.episode(seed).total_reward)
        clone_reward = _drive_clone(fn, params, env_factory, seed, spec)
        bb_rewards.append(bb_reward)
        clone_rewards.append(clone_reward)
        n_failures += _is_failure(clone_reward, bb_reward)

    n = len(seeds)
    return EvalReport(
        open_loop_agreement=agreement(fn, params, X, y, spec),
        n_holdout_obs=len(X),
        clone_mean_reward=sum(clone_rewards) / n if n else None,
        bb_mean_reward=sum(bb_rewards) / n if n else None,
        n_episodes=n,
        n_failures=n_failures,
        failure_rate_bound_95=3 / n if n and n_failures == 0 else None,
        us_per_action_clone=_time_clone(fn, params, X, spec),
        us_per_action_bb=_bb_inference_us(bb, holdout, X, spec),
    )
