"""Lane B, task B3: probe expansion and Python verdicts.

A probe sweeps one (or two) features with every other feature at the median of the data seen so far, queries
the black box, and Python decides: among probe points inside the `expect` region, the fraction where the
black-box action equals the expected action -> >= 0.85 confirmed, <= 0.4 refuted, else probing.
"""
from __future__ import annotations

import numpy as np

CONFIRM_AT = 0.85
REFUTE_AT = 0.40
MAX_N_2D = 15


class ProbeError(ValueError):
    pass


def _feature_index(spec, name: str) -> int:
    if name not in spec.obs_names:
        raise ProbeError(f"unknown feature {name!r}; features are {spec.obs_names}")
    return spec.obs_names.index(name)


def expand(probe: dict, spec, seen_X: list[list[float]]) -> list[list[float]]:
    base = probe.get("base", "median")
    if base == "median":
        if not seen_X:
            raise ProbeError("no data yet to take a median from")
        base_v = np.median(np.asarray(seen_X, float), axis=0)
    else:
        if len(base) != len(spec.obs_names):
            raise ProbeError(f"base has {len(base)} values; expected {len(spec.obs_names)}")
        base_v = np.asarray(base, float)
    s1 = probe["sweep"]
    i1 = _feature_index(spec, s1["feature"])
    s2 = probe.get("sweep2")
    if s2:
        i2 = _feature_index(spec, s2["feature"])
        n1, n2 = min(int(s1["n"]), MAX_N_2D), min(int(s2["n"]), MAX_N_2D)
        pts = []
        for a in np.linspace(s1["start"], s1["stop"], n1):
            for b in np.linspace(s2["start"], s2["stop"], n2):
                v = base_v.copy()
                v[i1], v[i2] = a, b
                pts.append(v)
    else:
        pts = []
        for a in np.linspace(s1["start"], s1["stop"], int(s1["n"])):
            v = base_v.copy()
            v[i1] = a
            pts.append(v)
    return [[round(float(x), 6) for x in p] for p in pts]


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


def run_probe(probe: dict, rule_id: str, bb, spec, seen_X) -> dict:
    """Expand, query (source='probe', rule_id), and judge. Returns a result dict incl. the QueryBatch."""
    obs = expand(probe, spec, seen_X)
    batch = bb.act(obs, source="probe", rule_id=rule_id)
    status, frac, n_in = verdict(batch, probe, spec)
    note = "no probe points fell inside the expect region; widen the sweep" if n_in == 0 else ""
    return {"rule_id": rule_id, "query_id": batch.query_id, "description": describe(probe),
            "n_points": len(obs), "n_in_region": n_in, "fraction": frac, "status": status,
            "note": note, "batch": batch}
