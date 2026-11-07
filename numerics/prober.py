"""Lane A, task A5: prober.

Finds states where the clone and the black box disagree, in two rounds:
  1. Seed obs with small Gaussian noise, plus per-feature sweeps over the empirical p1-p99 range.
  2. Refinement: perturb around what round 1 found, to map each region.

Round 2 is also how counterexamples get ranked. Any imperfect clone disagrees in a thin band along its
decision boundary, and farthest-point selection alone loves those stray points, so they would crowd out a
real behaviour the clone is missing. A step off a thin band agrees again, while a step inside a real region
still disagrees, so the fraction of a point's neighbours that also disagree (its "support") tells them apart.

Limit: if the clone is also wrong about clean behaviour in a thick region, that region has high support too,
because it is a real error. Those counterexamples are worth showing the brain, but they dilute the twist.
"""
from dataclasses import dataclass

import numpy as np

from contracts.types import Action, BlackBox, Counterexample, Obs, Params, ProgramFn, SensorSpec
from numerics.fit import action_ranges, continuous_match

BATCH_SIZE = 256
FIRST_ROUND_SHARE = 0.6
SWEEP_SHARE = 0.4               # of the first round; the rest is noise around seeds
NOISE_SCALE = 0.1               # sigma, as a fraction of each feature's empirical std
SEARCH_NOISE_SCALE = 0.3        # used when round 1 found nothing and there's budget left to look wider
REFINE_SCALE = 0.25
CHILDREN_PER_PARENT = 6
RELATIVE_SUPPORT = 0.75         # a point must reach this fraction of the best region's support


class _SeedStats:
    """What the seed obs say about each feature. Spec bounds aren't used: some are infinite."""

    def __init__(self, seed_obs: list[Obs]):
        self.data = np.asarray(seed_obs, dtype=float)
        self.low, self.high = np.percentile(self.data, [1, 99], axis=0)
        self.std = self.data.std(axis=0)
        self.base = np.median(self.data, axis=0)
        self.binary_values: dict[int, np.ndarray] = {}
        for j in range(self.data.shape[1]):
            values, counts = np.unique(self.data[:, j], return_counts=True)
            if len(values) <= 2:
                self.binary_values[j] = values
                # A half-and-half feature has a median of 0.5, a state that can never occur.
                self.base[j] = values[int(counts.argmax())]
        self.binary = np.array([j in self.binary_values for j in range(self.data.shape[1])])

    def noisy(self, centers: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(size=centers.shape) * (scale * self.std)
        noise[:, self.binary] = 0.0     # legs are 0 or 1; a leg at 0.03 is not a real state
        return centers + noise

    def sweeps(self, budget: int) -> np.ndarray:
        """Vary one feature at a time, others at their median. Binary values first, they cost little."""
        rows = [(j, v) for j, values in self.binary_values.items() for v in values]
        continuous = [j for j in range(self.data.shape[1]) if j not in self.binary_values]
        if continuous:
            per_feature = max(budget - len(rows), 0) // len(continuous)
            for j in continuous:
                rows += [(j, v) for v in np.linspace(self.low[j], self.high[j], per_feature)]
        out = np.tile(self.base, (len(rows), 1))
        for i, (j, v) in enumerate(rows):
            out[i, j] = v
        return out[:budget]


@dataclass
class _Probe:
    """Result for one queried state. cex is set only when the clone and the black box disagree."""
    obs: Obs
    cex: Counterexample | None


class _Asker:
    """Queries the black box, compares with the clone, and never spends past the budget or repeats a state."""

    def __init__(self, fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec, budget: int):
        self.fn, self.params, self.bb, self.spec = fn, params, bb, spec
        self.remaining = budget
        self.ranges = action_ranges(spec)
        self.seen: set[tuple[float, ...]] = set()

    def _differs(self, bb_action: Action, clone_action: Action) -> bool:
        if self.spec.action_type == "discrete":
            return bb_action != clone_action
        return not continuous_match(clone_action, bb_action, self.ranges)

    def ask(self, candidates: np.ndarray) -> list[_Probe]:
        fresh: list[Obs] = []
        for row in candidates.tolist():
            key = tuple(row)
            if key not in self.seen and len(fresh) < self.remaining:
                self.seen.add(key)
                fresh.append(row)
        probes: list[_Probe] = []
        for start in range(0, len(fresh), BATCH_SIZE):
            chunk = fresh[start:start + BATCH_SIZE]
            self.remaining -= len(chunk)
            batch = self.bb.act(chunk, source="prober")
            for obs, bb_action in zip(chunk, batch.actions):
                try:
                    clone_action = self.fn(obs, self.params)
                except Exception:       # no clone action to report, so this can't be a counterexample
                    probes.append(_Probe(obs, None))
                    continue
                cex = None
                if self._differs(bb_action, clone_action):
                    cex = Counterexample(obs, bb_action, clone_action, batch.query_id)
                probes.append(_Probe(obs, cex))
        return probes


def _farthest_points(cexes: list[Counterexample], k: int, stats: _SeedStats) -> list[Counterexample]:
    """Greedy farthest-point selection, starting from the first item, in std-normalized units."""
    if not cexes or k <= 0:
        return []
    unit = np.where(stats.std > 0, stats.std, 1.0)
    rows = np.asarray([c.obs for c in cexes], dtype=float) / unit
    chosen = [0]
    distance = np.linalg.norm(rows - rows[0], axis=1)
    while len(chosen) < min(k, len(cexes)):
        i = int(distance.argmax())
        if distance[i] == 0.0:          # everything left duplicates a chosen point
            break
        chosen.append(i)
        distance = np.minimum(distance, np.linalg.norm(rows - rows[i], axis=1))
    return [cexes[i] for i in chosen]


def _refine(found: list[Counterexample], asker: _Asker, stats: _SeedStats, rng: np.random.Generator,
            ) -> list[tuple[Counterexample, float]]:
    """Perturb around the found points. Returns every disagreement with its neighbourhood's support."""
    budget = asker.remaining
    if budget <= 0:
        return [(c, 0.0) for c in found]
    n_parents = max(1, budget // CHILDREN_PER_PARENT)
    parents = _farthest_points(found, n_parents, stats)
    each, extra = divmod(budget, len(parents))

    pool: list[tuple[Counterexample, float]] = []
    for i, parent in enumerate(parents):
        n_children = each + (1 if i < extra else 0)
        centers = np.tile(np.asarray(parent.obs, dtype=float), (n_children, 1))
        children = asker.ask(stats.noisy(centers, REFINE_SCALE, rng))
        hits = [p.cex for p in children if p.cex is not None]
        support = len(hits) / len(children) if children else 0.0
        pool.extend((c, support) for c in [parent, *hits])

    refined = {id(c) for c, _ in pool}
    pool.extend((c, 0.0) for c in found if id(c) not in refined)    # never tested: no evidence either way
    return pool


def _select(pool: list[tuple[Counterexample, float]], k: int, stats: _SeedStats) -> list[Counterexample]:
    ranked = sorted(pool, key=lambda item: -item[1])    # stable: ties keep discovery order
    bar = RELATIVE_SUPPORT * ranked[0][1]
    strong = [c for c, support in ranked if support >= bar]
    weak = [c for c, support in ranked if support < bar]
    # Weak points only fill up what strong regions can't: a real sliver is still worth showing the brain.
    return _farthest_points(strong + weak[:max(0, k - len(strong))], k, stats)


def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                       spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]:
    """returns the k most diverse disagreements; respects max_queries exactly"""
    if not seed_obs or k <= 0 or max_queries <= 0:
        return []
    stats = _SeedStats(seed_obs)
    # Seeded by size so reruns reproduce, but a later turn with more seed obs doesn't re-ask the same states.
    rng = np.random.default_rng(len(seed_obs))
    asker = _Asker(fn, params, bb, spec, max_queries)

    first_round = max(1, int(max_queries * FIRST_ROUND_SHARE))
    n_sweeps = int(first_round * SWEEP_SHARE)
    sweeps = stats.sweeps(n_sweeps)
    centers = stats.data[rng.integers(len(stats.data), size=first_round - len(sweeps))]
    candidates = np.vstack([sweeps, stats.noisy(centers, NOISE_SCALE, rng)])
    found = [p.cex for p in asker.ask(candidates) if p.cex is not None]

    if not found and asker.remaining > 0:
        centers = stats.data[rng.integers(len(stats.data), size=asker.remaining)]
        found = [p.cex for p in asker.ask(stats.noisy(centers, SEARCH_NOISE_SCALE, rng)) if p.cex is not None]
    if not found:
        return []
    return _select(_refine(found, asker, stats, rng), k, stats)
