"""Lane B, task B3: probe expansion and Python verdicts.

A probe sweeps one (or two) features while holding every other feature at a base state, queries the black box,
and Python decides: among probe points inside the `expect` region, the fraction where the black-box action
equals the expected action -> >= 0.85 confirmed, <= 0.4 refuted, else probing.

Base states (choose_bases):
- an explicit `base` from the brain is always swept;
- when counterexamples exist, the sweep is also repeated from up to MAX_CEX_BASES counterexample states
  (in addition to an explicit base, or instead of the median). A hand-built base can easily sit in a state where
  a conditional flaw is dormant, so the counterexample states are always included.
  Conditional flaws ("x > 0.05 *while firing the main engine*") only show up in states where the condition's
  other half holds, and the median state usually isn't one of them, so a median sweep can wrongly refute
  the rule. Counterexamples are, by definition, states where the clone and the robot disagree;
- the median of the data seen so far is used only when there are no counterexamples.
The verdict is computed over every point from every base, and the bases are returned as evidence.
"""
from __future__ import annotations

import numpy as np

CONFIRM_AT = 0.85
REFUTE_AT = 0.40
MAX_N_2D = 15
MAX_CEX_BASES = 3


class ProbeError(ValueError):
    pass


def _feature_index(spec, name: str) -> int:
    if name not in spec.obs_names:
        raise ProbeError(f"unknown feature {name!r}; features are {spec.obs_names}")
    return spec.obs_names.index(name)


def _cex_obs_and_id(c) -> tuple[list[float] | None, str | None]:
    """Counterexample dataclass or dict -> (obs, query_id)."""
    if isinstance(c, dict):
        return c.get("obs"), c.get("query_id")
    return getattr(c, "obs", None), getattr(c, "query_id", None)


def choose_bases(probe: dict, spec, seen_X: list[list[float]], counterexamples=None) -> list[dict]:
    """-> [{"kind": "explicit" | "counterexample" | "median", "obs": [...], "query_id": str | None}]."""
    width = len(spec.obs_names)
    base = probe.get("base", "median")
    bases = []
    if base != "median":
        if len(base) != width:
            raise ProbeError(f"base has {len(base)} values; expected {width}")
        bases.append({"kind": "explicit", "obs": [float(v) for v in base], "query_id": None})

    seen = {tuple(round(v, 6) for v in b["obs"]) for b in bases}
    n_cex = 0
    for c in counterexamples or []:
        obs, qid = _cex_obs_and_id(c)
        if obs is None or len(obs) != width:
            continue
        key = tuple(round(float(v), 6) for v in obs)
        if key in seen:                       # the brain already chose this counterexample as its base
            continue
        seen.add(key)
        bases.append({"kind": "counterexample", "obs": [float(v) for v in obs], "query_id": qid})
        n_cex += 1
        if n_cex == MAX_CEX_BASES:
            break
    if bases:
        return bases

    if not seen_X:
        raise ProbeError("no data yet to take a median from")
    median = np.median(np.asarray(seen_X, float), axis=0)
    return [{"kind": "median", "obs": [float(v) for v in median], "query_id": None}]


def _sweep_from(base_v: np.ndarray, probe: dict, spec) -> list[np.ndarray]:
    s1 = probe["sweep"]
    i1 = _feature_index(spec, s1["feature"])
    s2 = probe.get("sweep2")
    pts = []
    if s2:
        i2 = _feature_index(spec, s2["feature"])
        n1, n2 = min(int(s1["n"]), MAX_N_2D), min(int(s2["n"]), MAX_N_2D)
        for a in np.linspace(s1["start"], s1["stop"], n1):
            for b in np.linspace(s2["start"], s2["stop"], n2):
                v = base_v.copy()
                v[i1], v[i2] = a, b
                pts.append(v)
    else:
        for a in np.linspace(s1["start"], s1["stop"], int(s1["n"])):
            v = base_v.copy()
            v[i1] = a
            pts.append(v)
    return pts


def expand_with_bases(probe: dict, spec, seen_X: list[list[float]],
                      counterexamples=None) -> tuple[list[list[float]], list[dict]]:
    """-> (probe points from every base, the bases used)."""
    bases = choose_bases(probe, spec, seen_X, counterexamples)
    pts: list[np.ndarray] = []
    for base in bases:
        pts += _sweep_from(np.asarray(base["obs"], float), probe, spec)
    return [[round(float(x), 6) for x in p] for p in pts], bases


def expand(probe: dict, spec, seen_X: list[list[float]], counterexamples=None) -> list[list[float]]:
    return expand_with_bases(probe, spec, seen_X, counterexamples)[0]


def describe_bases(bases: list[dict]) -> str:
    if not bases or bases[0]["kind"] == "median":
        return "from the median state"
    parts = []
    if any(b["kind"] == "explicit" for b in bases):
        parts.append("the given base state")
    cex = [b for b in bases if b["kind"] == "counterexample"]
    if cex:
        ids = ", ".join(b["query_id"] or "?" for b in cex)
        parts.append(f"{len(cex)} counterexample state{'s' if len(cex) != 1 else ''} ({ids})")
    return "from " + " + ".join(parts)


def describe(probe: dict) -> str:
    s, e = probe["sweep"], probe["expect"]
    d = f"sweep {s['feature']} {s['start']:.3g}→{s['stop']:.3g}"
    if probe.get("sweep2"):
        s2 = probe["sweep2"]
        d += f" × {s2['feature']} {s2['start']:.3g}→{s2['stop']:.3g}"
    return d + f", expect action {e['action']} where {e['feature']} {e['op']} {e['value']:.3g}"


def rule_text(probe: dict, spec) -> str:
    e = probe["expect"]
    a = e["action"]
    name = spec.action_names[int(a)] if spec.action_type == "discrete" and isinstance(a, int) \
        and 0 <= a < len(spec.action_names) else str(a)
    return f"When {e['feature']} {e['op']} {e['value']:.3g}, the robot chooses {name}"


def _same(a, b, spec) -> bool:
    if spec.action_type == "discrete":
        return int(a) == int(b)
    lo, hi = np.asarray(spec.action_low, float), np.asarray(spec.action_high, float)
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))) <= 0.1 * float(np.linalg.norm(hi - lo))


def verdict(batch, probe: dict, spec) -> tuple[str, float, int]:
    """-> (status, fraction matched, n points in the expect region)."""
    e = probe["expect"]
    j = _feature_index(spec, e["feature"])
    inside = [(o, a) for o, a in zip(batch.obs, batch.actions)
              if (o[j] > e["value"] if e["op"] == ">" else o[j] < e["value"])]
    if not inside:
        return "probing", 0.0, 0
    frac = sum(_same(a, e["action"], spec) for _, a in inside) / len(inside)
    if frac >= CONFIRM_AT:
        return "confirmed", frac, len(inside)
    if frac <= REFUTE_AT:
        return "refuted", frac, len(inside)
    return "probing", frac, len(inside)


def run_probe(probe: dict, rule_id: str, bb, spec, seen_X, counterexamples=None) -> dict:
    """Expand from the chosen bases, query (source='probe', rule_id), and judge over all those points.

    Returns a result dict incl. the QueryBatch, the bases used and `base_evidence` (the query ids of the
    counterexample states the sweep started from)."""
    obs, bases = expand_with_bases(probe, spec, seen_X, counterexamples)
    batch = bb.act(obs, source="probe", rule_id=rule_id)
    status, frac, n_in = verdict(batch, probe, spec)
    note = "no probe points fell inside the expect region; widen the sweep" if n_in == 0 else ""
    return {"rule_id": rule_id, "query_id": batch.query_id,
            "description": f"{describe(probe)} {describe_bases(bases)}",
            "n_points": len(obs), "n_in_region": n_in, "fraction": frac, "status": status,
            "note": note, "batch": batch,
            "bases": [{"kind": b["kind"], "query_id": b["query_id"]} for b in bases],
            "base_evidence": [b["query_id"] for b in bases if b["query_id"]]}
