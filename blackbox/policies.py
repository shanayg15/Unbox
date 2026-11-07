"""Lane A, task A1: policy registry, loading and rollouts.

Third-party policies: SB3 RL Zoo checkpoints from Hugging Face (huggingface.co/sb3).

Run `python -m blackbox.policies --check` to compare each policy's mean reward
over 5 seeds against the score on its model card.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

# pygame starts SDL's video driver on the first env.render(). On macOS the default Cocoa
# driver aborts the whole process when that happens off the main thread, and FastAPI runs
# sync endpoints in a threadpool. Frames still render: rgb_array draws to an offscreen surface.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402
from PIL import Image  # noqa: E402

from contracts.types import Episode, SensorSpec  # noqa: E402

FRAME_WIDTH = 360
JPEG_QUALITY = 60
DEFAULT_FRAME_EVERY = 3
SPEC_SEEDS = range(1000, 1005)  # rollouts behind the empirical p1/p99 bounds; clean policy only
UNBOUNDED = 1e6


@dataclass(frozen=True)
class PolicyDef:
    bb: str
    label: str
    env_id: str
    repo: str
    filename: str
    algo: str
    obs_names: tuple[str, ...]
    action_names: tuple[str, ...]
    card_reward: float
    card_std: float
    vec_normalize: bool = False


CARTPOLE = dict(
    env_id="CartPole-v1",
    obs_names=("cart_x", "cart_v", "pole_angle", "pole_ang_v"),
    action_names=("push_left", "push_right"),
)
LUNARLANDER = dict(
    env_id="LunarLander-v3",  # the v2 checkpoints run on v3: same 8-dim obs, 4 actions
    obs_names=("x", "y", "vx", "vy", "angle", "ang_v", "leg_left", "leg_right"),
    action_names=("noop", "fire_left", "fire_main", "fire_right"),
)
MOUNTAINCAR = dict(
    env_id="MountainCar-v0",
    obs_names=("position", "velocity"),
    action_names=("push_left", "noop", "push_right"),
)
ACROBOT = dict(
    env_id="Acrobot-v1",
    obs_names=("cos1", "sin1", "cos2", "sin2", "v1", "v2"),
    action_names=("torque_neg", "torque_0", "torque_pos"),
)

REGISTRY: dict[str, PolicyDef] = {
    p.bb: p
    for p in [
        PolicyDef("bb-01", "CartPole-PPO", repo="sb3/ppo-CartPole-v1", filename="ppo-CartPole-v1.zip",
                  algo="ppo", card_reward=500.0, card_std=0.0, **CARTPOLE),
        PolicyDef("bb-02", "CartPole-DQN", repo="sb3/dqn-CartPole-v1", filename="dqn-CartPole-v1.zip",
                  algo="dqn", card_reward=500.0, card_std=0.0, **CARTPOLE),
        PolicyDef("bb-03", "MountainCar-DQN", repo="sb3/dqn-MountainCar-v0", filename="dqn-MountainCar-v0.zip",
                  algo="dqn", card_reward=-103.40, card_std=7.49, **MOUNTAINCAR),
        PolicyDef("bb-04", "Acrobot-PPO", repo="sb3/ppo-Acrobot-v1", filename="ppo-Acrobot-v1.zip",
                  algo="ppo", card_reward=-74.60, card_std=11.48, vec_normalize=True, **ACROBOT),
        PolicyDef("bb-05", "LunarLander-PPO", repo="sb3/ppo-LunarLander-v2", filename="ppo-LunarLander-v2.zip",
                  algo="ppo", card_reward=233.56, card_std=53.89, **LUNARLANDER),
        PolicyDef("bb-06", "LunarLander-DQN", repo="sb3/dqn-LunarLander-v2", filename="dqn-LunarLander-v2.zip",
                  algo="dqn", card_reward=136.79, card_std=42.72, **LUNARLANDER),
    ]
}


class ActionTwist(Protocol):
    def apply(self, obs: list[float], action: int) -> int: ...


def _download(repo: str, filename: str) -> str:
    """Cached copy first, so a loaded laptop never needs venue Wi-Fi."""
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename)


class _GymSpacesUnpickler(pickle.Unpickler):
    """SB3 1.5-era pickles reference the retired `gym` package; gymnasium has the same classes."""

    def find_class(self, module: str, name: str):
        if module == "gym" or module.startswith("gym."):
            module = "gymnasium" + module[len("gym"):]
        return super().find_class(module, name)


def _load_obs_normalizer(repo: str) -> Callable[[np.ndarray], np.ndarray]:
    """The saved VecNormalize stats as a plain function. Without them the policy acts nonsensically."""
    with open(_download(repo, "vec_normalize.pkl"), "rb") as file:
        saved = _GymSpacesUnpickler(file).load()
    mean, std = saved.obs_rms.mean, np.sqrt(saved.obs_rms.var + saved.epsilon)
    clip = saved.clip_obs

    def normalize(obs_batch: np.ndarray) -> np.ndarray:
        return np.clip((obs_batch - mean) / std, -clip, clip).astype(np.float32)

    return normalize


class LoadedPolicy:
    def __init__(self, defn: PolicyDef) -> None:
        import torch
        from stable_baselines3 import DQN, PPO

        torch.set_num_threads(1)  # tiny MLPs: extra threads only add latency
        self.defn = defn
        self._lock = threading.Lock()
        path = _download(defn.repo, defn.filename)
        no_schedule = lambda _: 0.0
        env = gym.make(defn.env_id)
        self.model = {"ppo": PPO, "dqn": DQN}[defn.algo].load(
            path,
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": no_schedule,
                "clip_range": no_schedule,
                "exploration_schedule": no_schedule,
                # SB3 1.5-era checkpoints pickle these as old `gym` spaces, which no longer unpickle
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
        env.close()
        self._normalizer = _load_obs_normalizer(defn.repo) if defn.vec_normalize else None

    def predict(self, obs_batch: np.ndarray) -> np.ndarray:
        """Clean policy actions for a whole (n, obs_dim) batch."""
        with self._lock:
            batch = self._normalizer(obs_batch) if self._normalizer else obs_batch
            actions, _ = self.model.predict(batch, deterministic=True)
        return actions


_loaded: dict[str, LoadedPolicy] = {}
_specs: dict[str, SensorSpec] = {}
_load_lock = threading.Lock()


def get_policy(bb: str) -> LoadedPolicy:
    with _load_lock:
        if bb not in _loaded:
            _loaded[bb] = LoadedPolicy(REGISTRY[bb])
        return _loaded[bb]


def _encode_frame(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb)
    height = round(image.height * FRAME_WIDTH / image.width)
    buffer = io.BytesIO()
    image.resize((FRAME_WIDTH, height), Image.BILINEAR).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def rollout(bb: str, seed: int, twist: ActionTwist | None = None, render: bool = False,
            frame_every: int = DEFAULT_FRAME_EVERY) -> Episode:
    """One episode: the policy drives itself, with the twist applied after it acts.

    obs[t] is the state actions[t] was chosen in, so obs, actions and rewards line up;
    frames[i] shows obs[i * frame_every].
    """
    policy = get_policy(bb)
    env = gym.make(policy.defn.env_id, render_mode="rgb_array" if render else None)
    try:
        obs, _ = env.reset(seed=seed)
        all_obs, actions, rewards, frames = [], [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            if render and len(all_obs) % frame_every == 0:
                frames.append(_encode_frame(env.render()))
            row = [float(v) for v in obs]
            action = int(policy.predict(np.asarray(obs, dtype=np.float32)[None])[0])
            if twist is not None:
                action = twist.apply(row, action)
            obs, reward, terminated, truncated, _ = env.step(action)
            all_obs.append(row)
            actions.append(action)
            rewards.append(float(reward))
    finally:
        env.close()
    return Episode(
        seed=seed,
        obs=all_obs,
        actions=actions,
        rewards=rewards,
        total_reward=float(sum(rewards)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        frames=frames if render else None,
        frame_every=frame_every,
    )


def sensor_spec(bb: str) -> SensorSpec:
    """Bounds come from the space, or the clean policy's empirical p1/p99 where it is unbounded."""
    with _load_lock:
        if bb in _specs:
            return _specs[bb]
    defn = REGISTRY[bb]
    env = gym.make(defn.env_id)
    low = np.asarray(env.observation_space.low, dtype=float)
    high = np.asarray(env.observation_space.high, dtype=float)
    env.close()
    unbounded = (np.abs(low) > UNBOUNDED) | (np.abs(high) > UNBOUNDED)
    if unbounded.any():
        seen = np.array([row for seed in SPEC_SEEDS for row in rollout(bb, seed).obs])
        low = np.where(unbounded, np.percentile(seen, 1, axis=0), low)
        high = np.where(unbounded, np.percentile(seen, 99, axis=0), high)
    spec = SensorSpec(
        obs_names=list(defn.obs_names),
        obs_low=[round(float(v), 6) for v in low],
        obs_high=[round(float(v), 6) for v in high],
        action_type="discrete",
        action_names=list(defn.action_names),
    )
    with _load_lock:
        _specs[bb] = spec
    return spec


def check(seeds: range = range(5)) -> bool:
    """Mean reward over a few seeds vs the model card (itself a 10-episode score)."""
    print(f"{'bb':6} {'label':17} {'env':15} {'card':>16} {'mean over 5':>12} {'min':>9} {'max':>9} {'s':>5}  ok")
    all_ok = True
    for bb, defn in REGISTRY.items():
        started = time.perf_counter()
        rewards = [rollout(bb, seed).total_reward for seed in seeds]
        mean = float(np.mean(rewards))
        tolerance = max(2 * defn.card_std, 0.02 * abs(defn.card_reward))
        ok = abs(mean - defn.card_reward) <= tolerance
        all_ok &= ok
        card = f"{defn.card_reward:.1f} +/- {defn.card_std:.1f}"
        print(f"{bb:6} {defn.label:17} {defn.env_id:15} {card:>16} {mean:12.1f} {min(rewards):9.1f} "
              f"{max(rewards):9.1f} {time.perf_counter() - started:5.1f}  {'yes' if ok else 'NO'}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy registry checks")
    parser.add_argument("--check", action="store_true", help="compare mean reward with each model card")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
