"""Owned by Lane B, task B1: an in-process black box and a numerics stub, so Lane B runs with no Lane A code.

MockBlackBox  = local CartPole + hand-written policy + the t-cp-1 twist, implementing contracts.types.BlackBox.
numerics_stub = same surface as contracts/numerics_api.py, simple implementations (exec, quick search, random prober).
"""
from __future__ import annotations

import base64
import io
import math
import random
import time
from typing import Callable

import numpy as np

from contracts.numerics_api import ProgramError as _ProgramError
from contracts.types import (
    Action,
    BlackBox,
    Counterexample,
    Episode,
    EvalReport,
    Obs,
    Params,
    ProgramFn,
    QueryBatch,
    SensorSpec,
)

CARTPOLE_SPEC = SensorSpec(
    obs_names=["cart_x", "cart_v", "pole_angle", "pole_ang_v"],
    obs_low=[-2.4, -2.0, -0.21, -3.0],
    obs_high=[2.4, 2.0, 0.21, 3.0],
    action_type="discrete",
    action_names=["push_left", "push_right"],
)

TWIST_CP1_THRESHOLD = 0.5      # t-cp-1: cart_v > 0.5 -> push_left
INIT_CART_V = 0.9              # mock episodes start with a seeded cart velocity in [-0.9, 0.9]


def reset_cartpole(env, seed: int) -> list[float]:
    """Reset with `seed`, then give the cart a seeded starting velocity.

    The hand-written policy balances the pole but never regulates the cart, so it keeps whatever drift it starts
    with. Without a starting velocity, cart_v never reaches the t-cp-1 region and the twist never shows up in
    rollouts. The clone's env (mock_env_factory) uses this same reset, so both start from the same state.
    """
    obs, _ = env.reset(seed=seed)
    st = np.array(env.unwrapped.state, dtype=np.float64)
    st[1] = float(np.random.default_rng(10_000 + seed).uniform(-INIT_CART_V, INIT_CART_V))
    env.unwrapped.state = st
    return [float(v) for v in st]


def encode_frame(rgb, width: int = 360, quality: int = 60) -> str:
    """RGB array -> base64 JPEG, downscaled to `width` px wide."""
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class MockBlackBox:
    """CartPole with `push_right if pole_angle + 0.5*pole_ang_v > 0 else push_left`, plus t-cp-1 when twist=True."""

    def __init__(self, twist: bool = True, threshold: float = TWIST_CP1_THRESHOLD) -> None:
        self.twist = twist
        self.threshold = threshold
        self._count = 0
        self._qid = 0
        self._episodes: dict[tuple, Episode] = {}

    # the hidden behavior ---------------------------------------------------
    def _policy(self, obs: Obs) -> int:
        a = 1 if obs[2] + 0.5 * obs[3] > 0 else 0
        if self.twist and obs[1] > self.threshold:
            a = 0
        return a

    # BlackBox protocol -------------------------------------------------------
    def spec(self) -> SensorSpec:
        return CARTPOLE_SPEC

    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch:
        t0 = time.perf_counter()
        actions = [self._policy(list(map(float, o))) for o in obs_batch]
        us = (time.perf_counter() - t0) * 1e6 / max(1, len(obs_batch))
        self._count += len(obs_batch)
        self._qid += 1
        return QueryBatch(query_id=f"q{self._qid}", obs=[list(map(float, o)) for o in obs_batch],
                          actions=actions, source=source, rule_id=rule_id, inference_us=us)

    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode:
        key = (self.twist, seed, render, frame_every)
        if key in self._episodes:
            return self._episodes[key]
        import gymnasium as gym

        env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
        obs = reset_cartpole(env, seed)
        obs_l, act_l, rew_l, frames = [], [], [], []
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            o = [float(v) for v in obs]
            a = self._policy(o)
            if render and step % frame_every == 0:
                frames.append(encode_frame(env.render()))
            obs_l.append(o)
            act_l.append(a)
            obs, r, terminated, truncated, _ = env.step(a)
            rew_l.append(float(r))
            step += 1
        env.close()
        ep = Episode(seed=seed, obs=obs_l, actions=act_l, rewards=rew_l, total_reward=float(sum(rew_l)),
                     terminated=bool(terminated), truncated=bool(truncated),
                     frames=frames if render else None, frame_every=frame_every)
        self._episodes[key] = ep
        return ep

    def query_count(self) -> int:
        return self._count


def mock_env_factory(seed: int, render: bool = True):
    """env_factory for the mock: (env already reset with seed, first_obs)."""
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    return env, reset_cartpole(env, seed)


# --------------------------------------------------------------------------- numerics stub
def _same(a, b, spec: SensorSpec) -> bool:
    if spec.action_type == "discrete":
        return int(a) == int(b)
    lo, hi = np.asarray(spec.action_low, float), np.asarray(spec.action_high, float)
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))) <= 0.1 * float(np.linalg.norm(hi - lo))


def _safe_call(fn, obs, params, spec):
    try:
        a = fn(obs, params)
        if spec.action_type == "discrete":
            a = int(a)
            if not 0 <= a < len(spec.action_names):
                return None
        return a
    except Exception:
        return None


class numerics_stub:
    """Same surface as contracts/numerics_api.py."""

    ProgramError = _ProgramError

    @staticmethod
    def load_program(code: str, spec: SensorSpec) -> ProgramFn:
        if "import" in code or "__" in code:
            raise _ProgramError("imports and dunder names are not allowed")
        ns: dict = {"math": math, "np": np}
        try:
            exec(compile(code, "<program>", "exec"), ns)
        except SyntaxError as e:
            raise _ProgramError(f"SyntaxError on line {e.lineno}: {e.msg}") from None
        except Exception as e:  # noqa: BLE001
            raise _ProgramError(f"{type(e).__name__} while defining the program: {e}") from None
        fn = ns.get("policy")
        if not callable(fn):
            raise _ProgramError("the code must define `def policy(obs, P):`")
        rng = random.Random(0)
        for _ in range(50):
            obs = [rng.uniform(lo, hi) for lo, hi in zip(spec.obs_low, spec.obs_high)]
            try:
                a = fn(obs, _ParamProbe())
            except KeyError as e:
                raise _ProgramError(f"P has no parameter {e}") from None
            except Exception as e:  # noqa: BLE001
                raise _ProgramError(f"{type(e).__name__} when running policy on {obs}: {e}") from None
            if spec.action_type == "discrete":
                try:
                    ai = int(a)
                except Exception:
                    raise _ProgramError(f"policy returned {a!r}; it must return an int action id") from None
                if not 0 <= ai < len(spec.action_names):
                    raise _ProgramError(f"policy returned {ai}; valid ids are 0..{len(spec.action_names) - 1}")
        return fn

    @staticmethod
    def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float:
        if not X:
            return 0.0
        ok = 0
        for o, a in zip(X, y):
            c = _safe_call(fn, o, params, spec)
            ok += c is not None and _same(c, a, spec)
        return ok / len(X)

    @staticmethod
    def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
                   max_seconds: float = 8.0) -> tuple[Params, float]:
        """Stub: a short coordinate search (≤ 2 s), so mock runs reach realistic agreement."""
        rng = random.Random(0)
        idx = list(range(len(X)))
        if len(idx) > 1500:
            idx = rng.sample(idx, 1500)
        Xs, ys = [X[i] for i in idx], [y[i] for i in idx]
        best = dict(init)
        best_s = numerics_stub.agreement(fn, best, Xs, ys, spec)
        t0, budget = time.time(), min(max_seconds, 2.0)
        step = {k: max(abs(v), 0.1) for k, v in best.items()}
        while time.time() - t0 < budget and step:
            improved = False
            for k in list(best):
                for d in (1, -1, 0.5, -0.5, 0.1, -0.1):
                    cand = dict(best)
                    cand[k] = best[k] + d * step[k]
                    s = numerics_stub.agreement(fn, cand, Xs, ys, spec)
                    if s > best_s:
                        best, best_s, improved = cand, s, True
                if time.time() - t0 >= budget:
                    break
            if not improved:
                step = {k: v * 0.5 for k, v in step.items()}
                if max(step.values()) < 1e-3:
                    break
        return best, numerics_stub.agreement(fn, best, X, y, spec)

    @staticmethod
    def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str:
        if len(X) < 20:
            return "(not enough data)"
        from sklearn.tree import DecisionTreeClassifier, export_text

        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20, random_state=0)
        tree.fit(np.asarray(X), np.asarray([int(a) for a in y]))
        names = [spec.action_names[int(c)] for c in tree.classes_]
        return export_text(tree, feature_names=list(spec.obs_names), class_names=names,
                           decimals=3, show_weights=True)

    @staticmethod
    def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                           spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]:
        """Stub: random candidates (seed obs + noise, uniform in the spec box, per-feature sweeps), one batch."""
        if max_queries <= 0 or not seed_obs:
            return []
        rng = np.random.default_rng(0)
        S = np.asarray(seed_obs, float)
        lo, hi = np.asarray(spec.obs_low, float), np.asarray(spec.obs_high, float)
        sd = S.std(axis=0) + 1e-9
        med = np.median(S, axis=0)
        n = int(max_queries)
        n_noise, n_unif = n // 3, n // 3
        n_sweep = n - n_noise - n_unif
        noisy = S[rng.integers(0, len(S), n_noise)] + rng.normal(0, 0.3, (n_noise, S.shape[1])) * sd
        unif = rng.uniform(lo, hi, (n_unif, S.shape[1]))
        sweeps = np.tile(med, (n_sweep, 1))
        for i in range(n_sweep):
            f = i % S.shape[1]
            sweeps[i, f] = rng.uniform(lo[f], hi[f])
        cand = np.vstack([noisy, unif, sweeps])[:n].round(4).tolist()
        batch = bb.act(cand, source="prober")
        found = []
        for o, a in zip(batch.obs, batch.actions):
            c = _safe_call(fn, o, params, spec)
            if c is None or not _same(c, a, spec):
                found.append(Counterexample(obs=o, bb_action=a, clone_action=c, query_id=batch.query_id))
        rng2 = random.Random(0)
        rng2.shuffle(found)
        return found[:k]

    @staticmethod
    def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
                 holdout: list[QueryBatch], env_factory: Callable[[int], object],
                 seeds: list[int]) -> EvalReport:
        """Stub: open-loop agreement on the holdout only."""
        X = [o for b in holdout for o in b.obs]
        y = [a for b in holdout for a in b.actions]
        t0 = time.perf_counter()
        for o in X[:1000]:
            _safe_call(fn, o, params, spec)
        us = (time.perf_counter() - t0) * 1e6 / max(1, min(1000, len(X)))
        return EvalReport(open_loop_agreement=numerics_stub.agreement(fn, params, X, y, spec),
                          n_holdout_obs=len(X), us_per_action_clone=us)

    @staticmethod
    def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                        obs: Obs, bb_action: Action) -> Action:
        r, f = recovered(obs, rec_params), fixed(obs, fix_params)
        return f if r != f else bb_action


class _ParamProbe(dict):
    """Lets load_program test-run a program before params are known: any P["name"] reads as 0.0."""

    def __missing__(self, key):
        return 0.0
