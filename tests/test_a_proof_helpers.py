"""A-Proof test support: an in-memory black box, so numerics never waits on the real server."""
from __future__ import annotations

import gymnasium as gym

from contracts.types import Action, Episode, Obs, Params, ProgramFn, QueryBatch, SensorSpec

CARTPOLE_SPEC = SensorSpec(
    obs_names=["cart_x", "cart_v", "pole_angle", "pole_ang_v"],
    obs_low=[-4.8, -3.0, -0.418, -3.0],
    obs_high=[4.8, 3.0, 0.418, 3.0],
    action_type="discrete",
    action_names=["push_left", "push_right"],
)

LUNAR_SPEC = SensorSpec(
    obs_names=["x", "y", "vx", "vy", "angle", "ang_v", "leg_left", "leg_right"],
    obs_low=[-1.5, -0.5, -2.0, -2.0, -3.14, -5.0, 0.0, 0.0],
    obs_high=[1.5, 1.5, 2.0, 2.0, 3.14, 5.0, 1.0, 1.0],
    action_type="discrete",
    action_names=["noop", "fire_left", "fire_main", "fire_right"],
)

ARM_SPEC = SensorSpec(
    obs_names=["error", "error_rate"],
    obs_low=[-1.0, -1.0],
    obs_high=[1.0, 1.0],
    action_type="continuous",
    action_names=["torque", "brake"],
    action_low=[-2.0, 0.0],
    action_high=[2.0, 1.0],
)


class RuleBlackBox:
    """Implements contracts.types.BlackBox in memory, with an optional twist applied after the rule.

    twist: {"condition": obs -> bool, "override": (obs, action) -> action}, the blackbox/twists.py shape.
    """

    def __init__(self, fn: ProgramFn, params: Params, spec: SensorSpec, twist: dict | None = None,
                 env_id: str = "CartPole-v1", inference_us: float | None = 40.0):
        self._fn, self._params, self._spec, self._twist = fn, params, spec, twist
        self._env_id, self._inference_us = env_id, inference_us
        self._queries = 0
        self._batches = 0
        self.log: list[QueryBatch] = []      # every batch asked of act(), so tests can inspect real queries
        self.episode_seeds: list[int] = []

    def _decide(self, obs: Obs) -> Action:
        action = self._fn(obs, self._params)
        if self._twist and self._twist["condition"](obs):
            action = self._twist["override"](obs, action)
        return action

    def spec(self) -> SensorSpec:
        return self._spec

    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch:
        self._queries += len(obs_batch)
        self._batches += 1
        batch = QueryBatch(
            query_id=f"q{self._batches}",
            obs=[list(o) for o in obs_batch],
            actions=[self._decide(list(o)) for o in obs_batch],
            source=source,
            rule_id=rule_id,
            inference_us=self._inference_us,
        )
        self.log.append(batch)
        return batch

    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode:
        self.episode_seeds.append(seed)
        env = gym.make(self._env_id)
        try:
            raw, _ = env.reset(seed=seed)
            obs, actions, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                state = [float(v) for v in raw]
                action = self._decide(state)
                raw, reward, terminated, truncated, _ = env.step(action)
                obs.append(state)
                actions.append(action)
                rewards.append(float(reward))
            return Episode(seed=seed, obs=obs, actions=actions, rewards=rewards,
                           total_reward=sum(rewards), terminated=terminated, truncated=truncated)
        finally:
            env.close()

    def query_count(self) -> int:
        return self._queries


def make_env_factory(env_id: str = "CartPole-v1", opened: list | None = None):
    """env_factory(seed) -> (env, first_obs), the shape the app hands to numerics.evaluate."""
    def factory(seed: int):
        env = gym.make(env_id)
        first_obs, _ = env.reset(seed=seed)
        if opened is not None:
            opened.append(env)
        return env, first_obs
    return factory
