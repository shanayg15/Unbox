"""Lane A, task A4: agreement and parameter fitting"""
import math
import random
import time
import warnings

import numpy as np
from scipy.optimize import least_squares

with warnings.catch_warnings():
    warnings.simplefilter("ignore")     # cma warns when matplotlib is missing; plotting is never used here
    import cma

from contracts.types import Action, Obs, Params, ProgramFn, SensorSpec

MAX_FIT_POINTS = 3000
CONTINUOUS_TOLERANCE = 0.1
POPSIZE = 16
START_SIGMA = 0.5
# Agreement is a step function, so CMA-ES stops the moment a generation scores flat. Each restart without
# progress doubles sigma, which walks out of a flat region from 0.5 up to 32 in normalized units. Those flat
# restarts cost one generation each. A full search that finds nothing costs far more, so only a few are allowed.
MAX_STALE_RESTARTS = 6
MAX_FRUITLESS_SEARCHES = 2


def action_ranges(spec: SensorSpec, y: list[Action] | None = None) -> list[float]:
    """Per-dimension action range, from spec bounds, else from the data, else 1.0."""
    n = len(spec.action_names)
    low, high = spec.action_low or [math.inf] * n, spec.action_high or [math.inf] * n
    ranges = [hi - lo if math.isfinite(lo) and math.isfinite(hi) and hi > lo else None
              for lo, hi in zip(low, high)]
    if None in ranges and y:
        data = np.asarray(y, dtype=float)
        ranges = [r if r is not None else float(data[:, i].max() - data[:, i].min()) or None
                  for i, r in enumerate(ranges)]
    return [r if r is not None else 1.0 for r in ranges]


def continuous_match(a: list[float], b: list[float], ranges: list[float]) -> bool:
    """L2 error <= 0.1 x action range, with each dimension measured in its own range."""
    total = 0.0
    for ai, bi, r in zip(a, b, ranges):
        total += ((ai - bi) / r) ** 2
    return total <= CONTINUOUS_TOLERANCE ** 2


def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float:
    """discrete = exact-match fraction; continuous = fraction with L2 error <= 0.1 x action range."""
    if not X:
        return 0.0
    if spec.action_type == "discrete":
        return _count_discrete(fn, params, X, y) / len(X)
    ranges = action_ranges(spec, y)
    hits = 0
    for obs, target in zip(X, y):
        try:
            hits += continuous_match(fn(obs, params), target, ranges)
        except Exception:
            pass
    return hits / len(X)


def _count_discrete(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action]) -> int:
    hits = 0
    for obs, target in zip(X, y):
        try:
            if fn(obs, params) == target:
                hits += 1
        except Exception:       # a crash on this point or in this param region just scores 0
            pass
    return hits


def _subsample(X: list[Obs], y: list[Action]) -> tuple[list[Obs], list[Action]]:
    if len(X) <= MAX_FIT_POINTS:
        return X, y
    keep = sorted(random.Random(0).sample(range(len(X)), MAX_FIT_POINTS))
    return [X[i] for i in keep], [y[i] for i in keep]


def _cma_search(fn: ProgramFn, start: Params, free: list[str], X: list[Obs], y: list[Action],
                deadline: float) -> Params:
    scale = np.array([max(abs(start[name]), 0.1) for name in free])
    origin = np.array([start[name] for name in free])

    def to_params(z) -> Params:
        return {**start, **dict(zip(free, (origin + scale * np.asarray(z)).tolist()))}

    def loss(z) -> float:
        if time.time() >= deadline:     # checked per candidate, so one generation can't overrun the cap
            return 1.0
        return 1.0 - _count_discrete(fn, to_params(z), X, y) / len(X)

    best_z, best_loss = np.zeros(len(free)), loss(np.zeros(len(free)))
    sigma, stale, fruitless, seed = START_SIGMA, 0, 0, 0
    while (best_loss > 0.0 and stale <= MAX_STALE_RESTARTS and fruitless < MAX_FRUITLESS_SEARCHES
           and time.time() < deadline):
        seed += 1                       # never 0: pycma reads seed 0 as "use the clock"
        es = cma.CMAEvolutionStrategy(best_z.tolist(), sigma, {
            "popsize": POPSIZE, "verbose": -9, "seed": seed, "verb_log": 0, "verb_disp": 0,
        })
        before, generations = best_loss, 0
        while not es.stop() and best_loss > 0.0 and time.time() < deadline:
            generations += 1
            Z = es.ask()
            scores = [loss(z) for z in Z]
            es.tell(Z, scores)
            i = int(np.argmin(scores))
            if scores[i] < best_loss:
                best_loss, best_z = scores[i], np.array(Z[i])
        improved = best_loss < before
        stale = 0 if improved else stale + 1
        fruitless += not improved and generations > 1
        sigma = START_SIGMA * 2 ** stale
    return to_params(best_z)


class _OutOfTime(Exception):
    pass


def _least_squares_search(fn: ProgramFn, start: Params, free: list[str], X: list[Obs], y: list[Action],
                          spec: SensorSpec, deadline: float) -> Params:
    scale = np.array([max(abs(start[name]), 0.1) for name in free])
    origin = np.array([start[name] for name in free])
    ranges = action_ranges(spec, y)
    targets = np.asarray(y, dtype=float)
    penalty = 10.0      # residual for a point the program crashed on, in action-range units
    best = {"cost": math.inf, "params": dict(start)}

    def to_params(z) -> Params:
        return {**start, **dict(zip(free, (origin + scale * np.asarray(z)).tolist()))}

    def residuals(z):
        if time.time() >= deadline:
            raise _OutOfTime
        params = to_params(z)
        out = np.full(targets.shape, penalty)
        for i, obs in enumerate(X):
            try:
                out[i] = (np.asarray(fn(obs, params)) - targets[i]) / ranges
            except Exception:
                pass
        out = np.nan_to_num(out, nan=penalty, posinf=penalty, neginf=-penalty).ravel()
        cost = float(out @ out)
        if cost < best["cost"]:
            best["cost"], best["params"] = cost, params
        return out

    try:
        least_squares(residuals, np.zeros(len(free)))
    except _OutOfTime:
        pass
    except Exception:
        pass            # a program the solver can't handle keeps its best params so far
    return best["params"]


def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
               max_seconds: float = 8.0) -> tuple[Params, float]:
    """returns (best_params, agreement_on_X)"""
    deadline = time.time() + max_seconds
    used = getattr(fn, "param_names", None)
    # The brain can forget a param or send one the code never reads. Fit what the code actually uses.
    start = {name: float(value) for name, value in init.items()}
    for name in used or ():
        start.setdefault(name, 0.0)
    free = [name for name in start if used is None or name in used]

    X = [[float(v) for v in row] for row in X]      # plain lists: far faster to index in a Python loop
    if spec.action_type == "discrete":
        y = [int(a) for a in y]
    if not free or not X:
        return start, agreement(fn, start, X, y, spec)

    X_fit, y_fit = _subsample(X, y)
    if spec.action_type == "discrete":
        found = _cma_search(fn, start, free, X_fit, y_fit, deadline)
    else:
        found = _least_squares_search(fn, start, free, X_fit, y_fit, spec, deadline)

    # The search scored a subsample. Compare on all of X so fitting can never make things worse.
    found_score, start_score = agreement(fn, found, X, y, spec), agreement(fn, start, X, y, spec)
    return (found, found_score) if found_score >= start_score else (start, start_score)
