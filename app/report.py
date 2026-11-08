"""Report card: "What your robotics team gets".

Everything here is computed from one run's own data: its event stream (ledger, probes, programs, SubQ
scans, reveal / fix / guard), its query log (``queries.jsonl``: rollouts, holdout, probes) and, for the
cost line only, fresh closed-loop episodes of the guarded robot. Nothing is hardcoded.

    GET /api/report/{run_id}                 JSON for the UI card
    GET /api/report/{run_id}.md              full report (Markdown download)
    GET /api/report/{run_id}/report.md       same
    GET /api/report/{run_id}/training.csv    targeted training set (needs a fix)
    GET /api/report/{run_id}/tests.json      regression tests (other confirmed rules)

``app.server`` is imported lazily inside functions (it includes this router at import time).
"""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()

EVAL_SEEDS = [200, 201, 202, 203, 204]
NEAR = 0.2                                     # twist_found: threshold within 20% of the true one
_KEY = re.compile(r"(bb-\d+|mock)__(.+?)__")
_PROBE_STATUS = re.compile(r"^Probing (\S+): .*expect action (.+?) where (\S+) ([<>]=?) (\S+)$")
_TEXT_REGION = re.compile(r"\b([A-Za-z_]\w*)\s*(<=|>=|<|>)\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_OP_WORDS = {"<": "below", ">": "above", "<=": "at or below", ">=": "at or above"}
_FILE_CACHE: dict[str, tuple[float, list]] = {}
_SPECS: dict[str, Any] = {}
_LOCK = threading.Lock()


def _server():
    from app import server
    return server


# ------------------------------------------------------------------------------------------ run data
def _source(bridge) -> str | None:
    """Where a replay / fallback came from ("cache/<key>", "runs/<id>", "fixtures/x.jsonl")."""
    if bridge.kind == "replay" or bridge.superseded:
        return bridge.ctx.get("fallback_source")
    return None


def _buffer(bridge) -> list[dict]:
    with bridge.lock:
        return list(bridge.buffer)


def _events(bridge) -> list[dict]:
    """The run's events. A fallback mixes a partial live run with the cached run: use the cached
    stream, plus the reveal / fix / shield the buttons added to this bridge."""
    buf = _buffer(bridge)
    src = _source(bridge)
    if bridge.superseded and src and src.startswith("cache/"):
        S = _server()
        try:
            base = S.load_jsonl(S.cache_path(src.split("/", 1)[1]))
        except Exception:  # noqa: BLE001
            return buf
        return base + [e for e in buf if e.get("type") in ("reveal", "fix", "shield")]
    return buf


def _read_jsonl(path: Path) -> list:
    key = str(path)
    mtime = path.stat().st_mtime
    hit = _FILE_CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    _FILE_CACHE[key] = (mtime, rows)
    return rows


def _queries_path(bridge) -> Path | None:
    S = _server()
    src = _source(bridge)
    cands: list[Path] = []
    if src is None:
        cands.append(S.RUNS_DIR / bridge.run_id / "queries.jsonl")
    elif src.startswith("runs/"):
        cands.append(S.RUNS_DIR / src.split("/", 1)[1] / "queries.jsonl")
    elif src.startswith("cache/"):
        cdir = S.CACHE_DIR / src.split("/", 1)[1]
        try:
            first = json.loads((cdir / "events.jsonl").open().readline())
            cands.append(S.RUNS_DIR / first["run_id"] / "queries.jsonl")
        except Exception:  # noqa: BLE001
            pass
        cands.append(cdir / "queries.jsonl")
    return next((p for p in cands if p.exists()), None)


def _queries(bridge) -> list[dict]:
    p = _queries_path(bridge)
    return _read_jsonl(p) if p else []


def _source_run_id(bridge) -> str | None:
    """The original run id behind a replay of a cached run (matches results/summary.json rows)."""
    src = _source(bridge)
    if not src:
        return bridge.run_id
    if src.startswith("runs/"):
        return src.split("/", 1)[1]
    if src.startswith("cache/"):
        try:
            cdir = _server().CACHE_DIR / src.split("/", 1)[1]
            return json.loads((cdir / "events.jsonl").open().readline())["run_id"]
        except Exception:  # noqa: BLE001
            return None
    return None


def _bb_and_twist(bridge, events: list[dict]) -> tuple[str | None, str | None, bool]:
    """-> (bb id, twist id, twist known). Live: the run config. Replay: the cache key in the source."""
    cfg = bridge.config or {}
    if bridge.kind == "live" and cfg.get("bb"):
        return cfg["bb"], cfg.get("twist_id"), True
    src = _source(bridge) or ""
    m = _KEY.search(src)
    if m:
        tw = m[2]
        return m[1], (None if tw == "none" else tw), True
    label = next((e["data"].get("policy_label") for e in events if e.get("type") == "run_started"), None)
    bb = None
    if label:
        try:
            pols, _ = _server().list_policies()
            bb = next((p["bb"] for p in pols if p.get("label") == label), None)
        except Exception:  # noqa: BLE001
            bb = None
    return bb, None, False


def _spec(bridge, bb_id: str | None):
    bb = bridge.ctx.get("bb")
    if bb is not None:
        key = f"ctx:{bridge.run_id}"
        if key not in _SPECS:
            _SPECS[key] = bb.spec()
        return _SPECS[key]
    if bb_id is None:
        return None
    if bb_id not in _SPECS:
        try:
            _SPECS[bb_id] = _server().make_blackbox(bb_id).spec()
        except Exception:  # noqa: BLE001
            return None
    return _SPECS[bb_id]


def _numerics(bridge):
    n = bridge.ctx.get("numerics")
    if n is not None:
        return n
    import numerics
    return numerics


# --------------------------------------------------------------------------------------- rule logic
def _last(events: list[dict], etype: str) -> dict | None:
    for e in reversed(events):
        if e.get("type") == etype:
            return e.get("data") or {}
    return None


def best_program(events: list[dict]) -> dict | None:
    """Max held-out agreement, latest on a tie: the program the run evaluates and the UI shows."""
    best = None
    for e in events:
        if e.get("type") == "program":
            d = e.get("data") or {}
            if best is None or (d.get("agreement") or 0.0) >= (best.get("agreement") or 0.0):
                best = d
    return best


def _parse_action(text: str):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            return text


def _probe_regions(events: list[dict]) -> dict[str, list[dict]]:
    """rule_id -> [{feature, op, value, action}] from the 'Probing rN: ... expect action A where f op v' lines."""
    out: dict[str, list[dict]] = {}
    for e in events:
        if e.get("type") != "status":
            continue
        m = _PROBE_STATUS.match((e.get("data") or {}).get("text", ""))
        if not m:
            continue
        try:
            value = float(m[5])
        except ValueError:
            continue                            # status line cut off mid-number
        out.setdefault(m[1], []).append({"feature": m[3], "op": m[4], "value": value,
                                         "action": _parse_action(m[2])})
    return out


def _text_region(text: str, obs_names: list[str]) -> dict | None:
    for m in _TEXT_REGION.finditer(text or ""):
        if m[1] in obs_names:
            return {"feature": m[1], "op": m[2], "value": float(m[3]), "action": None}
    return None


def _inside(obs, idx: int, op: str, value: float) -> bool:
    v = obs[idx]
    return v > value if op == ">" else v < value if op == "<" else v >= value if op == ">=" else v <= value


def _action_name(spec, a) -> str:
    try:
        if spec is not None and spec.action_type == "discrete":
            return spec.action_names[int(a)]
    except Exception:  # noqa: BLE001
        pass
    return str(a)


def _mentions(text: str, feature: str) -> bool:
    try:
        S = _server()
        return bool(S.rules_mentioning([{"rule_id": "x", "text": text}], feature))
    except Exception:  # noqa: BLE001
        return re.search(rf"(?<!\w){re.escape(feature)}(?!\w)", text or "") is not None


def _rule_infos(events, spec, queries) -> list[dict]:
    """One entry per ledger rule: its sensor region (from the probe that tested it), its probe batches,
    and `proven`: confirmed in the ledger, or its latest probe says >= 85% (the same Python verdict)."""
    ledger = (_last(events, "ledger") or {}).get("entries", [])
    regions = _probe_regions(events)
    names = list(spec.obs_names) if spec is not None else []
    by_rule: dict[str, list[dict]] = {}
    for b in queries:
        if b.get("source") == "probe" and b.get("rule_id"):
            by_rule.setdefault(b["rule_id"], []).append(b)
    infos = []
    for e in ledger:
        rid = e.get("rule_id")
        regs = [r for r in regions.get(rid, []) if not names or r["feature"] in names]
        batches = by_rule.get(rid, [])
        reg = regs[-1] if regs else (_text_region(e.get("text", ""), names) if names else None)
        last_batches = [batches[-1]] if (batches and len(regs) == len(regions.get(rid, [])) == len(batches)) \
            else batches
        info = {"entry": e, "rule_id": rid, "status": e.get("status"), "region": reg,
                "batches": batches, "n_probe": sum(len(b.get("obs", [])) for b in batches)}
        frac, n_in = _probe_fraction({**info, "batches": last_batches}, spec)
        info["proven"] = e.get("status") == "confirmed" or (
            e.get("status") != "refuted" and frac is not None and frac >= 0.85 and n_in > 0)
        infos.append(info)
    return infos


def _probe_fraction(info: dict, spec) -> tuple[float | None, int]:
    """Among this rule's probe points inside its region: fraction where the robot took the expected action."""
    reg = info.get("region")
    if not reg or spec is None or reg["feature"] not in spec.obs_names or reg.get("action") is None:
        return None, 0
    j = spec.obs_names.index(reg["feature"])
    pts = [(o, a) for b in info["batches"] for o, a in zip(b["obs"], b["actions"])
           if _inside(o, j, reg["op"], reg["value"])]
    if not pts:
        return None, 0
    try:
        hits = sum(int(a) == int(reg["action"]) for _, a in pts)
    except Exception:  # noqa: BLE001
        hits = sum(a == reg["action"] for _, a in pts)
    return hits / len(pts), len(pts)


def _pick_failing(infos, events, revealed: dict | None, truth: dict | None, code: str | None,
                  spec) -> dict | None:
    """The flaw rule. After Reveal: a proven rule on the revealed feature (nearest the true threshold when
    known). Before: the top suspect = a proven rule implemented as an early guard in the copy (planted
    flaws are overrides, so the copy checks them first), then SubQ's anomaly votes, then probe evidence."""
    proven = [i for i in infos if i["proven"] and i["region"]]
    if not proven:
        return None
    votes: Counter = Counter()
    for e in events:
        if e.get("type") == "snap":
            a = (e.get("data") or {}).get("anomaly_feature") or {}
            if a.get("choice") and a["choice"] != "none":
                votes[a["choice"]] += float((a.get("probabilities") or {}).get(a["choice"], 0.0))
    lines = {i["rule_id"]: (_code_line(code, i["region"]["feature"], spec) if code else None) for i in proven}

    def rank(i):
        ln = lines.get(i["rule_id"])
        return (ln is None, ln or 0, -votes.get(i["region"]["feature"], 0.0), -i["n_probe"])

    if revealed and revealed.get("feature"):
        cands = [i for i in proven if i["region"]["feature"] == revealed["feature"]]
        if not cands:
            return None
        if truth and truth.get("threshold") is not None:
            thr = truth["threshold"]
            return min(cands, key=lambda i: (abs(i["region"]["value"] - thr),) + rank(i))
        return min(cands, key=rank)
    return min(proven, key=rank)


def _code_line(code: str, feature: str, spec) -> int | None:
    """1-based line of the first if/elif in `code` whose test reads `feature` (by alias or obs[i])."""
    if spec is None or feature not in spec.obs_names:
        return None
    idx = spec.obs_names.index(feature)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    alias: set[str] = {feature}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and _is_obs(node.value, idx):
                    alias.add(t.id)
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(t.elts, node.value.elts):
                        if isinstance(n, ast.Name) and _is_obs(v, idx):
                            alias.add(n.id)
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Name) and node.value.id == "obs":
                    if idx < len(t.elts) and isinstance(t.elts[idx], ast.Name):
                        alias.add(t.elts[idx].id)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            for sub in ast.walk(node.test):
                if (isinstance(sub, ast.Name) and sub.id in alias) or _is_obs(sub, idx):
                    lines.append(node.test.lineno)
                    break
    return min(lines) if lines else None


def _is_obs(node, idx: int) -> bool:
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "obs"
            and isinstance(node.slice, ast.Constant) and node.slice.value == idx)


# ------------------------------------------------------------------------------------ truth / found
def _truth(bridge, bb_id: str | None, twist_id: str | None, feature: str | None) -> dict | None:
    """The planted rule's (feature, threshold), when the app can know it: admin reveal for the active twist."""
    if "report_truth" in bridge.ctx:
        return bridge.ctx["report_truth"]
    truth = None
    S = _server()
    if bb_id and twist_id and not S.use_mocks() and bb_id != "mock":
        try:
            info = S.admin_get("/admin/reveal", bb=bb_id)
            if info.get("twist_id") == twist_id and info.get("rule_text"):
                feat = feature or info.get("feature")
                for m in _TEXT_REGION.finditer(info["rule_text"]):
                    if m[1] == feat:
                        truth = {"feature": feat, "op": m[2], "threshold": float(m[3]),
                                 "rule_text": info["rule_text"]}
                        break
        except Exception:  # noqa: BLE001
            truth = None
    if truth is not None:
        bridge.ctx["report_truth"] = truth
    return truth


def _summary_found(run_id: str | None) -> bool | None:
    if not run_id:
        return None
    try:
        rows = json.loads((_server().RESULTS_DIR / "summary.json").read_text())
        rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        row = next((r for r in rows if r.get("run_id") == run_id), None)
        return None if row is None else bool(row.get("twist_found"))
    except Exception:  # noqa: BLE001
        return None


def _found(bridge, events, spec, revealed, truth, failing) -> str | None:
    if not revealed:
        return None
    if truth is not None and spec is not None:
        try:
            from scripts.batch import twist_evidence
            ok = twist_evidence(events, truth["feature"], truth["threshold"], list(spec.obs_names))["found"]
            return "match" if ok else "not_found"
        except Exception:  # noqa: BLE001
            pass
    if truth is not None and failing and failing.get("region"):
        thr = truth["threshold"]
        near = abs(failing["region"]["value"] - thr) <= NEAR * max(abs(thr), 1e-9)
        return "match" if near else "not_found"
    from_summary = _summary_found(_source_run_id(bridge))
    if from_summary is not None:
        return "match" if from_summary else "not_found"
    return "match" if failing else "not_found"


# ------------------------------------------------------------------------------- fixed-copy numbers
def guarded_action(n, rec, rp, fixed, fp, obs, a_bb, reg: dict | None = None, j: int | None = None):
    """Runtime guard. Inside the diagnosed flaw zone the corrected policy decides; elsewhere the
    contract's shielded_action (network's own action unless the flawed and fixed copies disagree)."""
    if reg is not None and j is not None and _inside(obs, j, reg["op"], reg["value"]):
        return fixed(obs, fp)
    return n.shielded_action(rec, rp, fixed, fp, obs, a_bb)


def failing_region(bridge) -> dict | None:
    """{feature, op, value, sentence, line} of the diagnosed failure (cached once the flaw is revealed)."""
    reg = bridge.ctx.get("failing_region")
    if reg:
        return reg
    try:
        f = build_report(bridge).get("failing")
    except Exception:  # noqa: BLE001
        return None
    if f and f.get("feature") and f.get("op") and f.get("value") is not None:
        reg = {k: f.get(k) for k in ("feature", "op", "value", "sentence", "line", "action_name")}
        if bridge.ctx.get("reveal") is not None:
            bridge.ctx["failing_region"] = reg
        return reg
    return None


def _programs(bridge, spec, prog: dict, fix: dict):
    n = _numerics(bridge)
    rec = n.load_program(prog["code"], spec)
    fixed = n.load_program(fix["code"], spec)
    return n, rec, fixed


def _all_states(queries: list[dict], sources: tuple[str, ...]) -> list[list[float]]:
    return [o for b in queries if b.get("source") in sources for o in b.get("obs", [])]


def training_rows(bridge, spec, queries, failing: dict, prog: dict, fix: dict) -> list[tuple[list[float], int]]:
    """States inside the failing region (probes, prober, the robot's own runs) + seeded perturbations,
    labeled with the fixed copy's action."""
    reg = failing
    j = spec.obs_names.index(reg["feature"])
    pool = [o for o in _all_states(queries, ("probe", "prober", "rollout", "holdout"))
            if _inside(o, j, reg["op"], reg["value"])]
    seen, base = set(), []
    for o in pool:
        k = tuple(round(float(v), 6) for v in o)
        if k not in seen:
            seen.add(k)
            base.append([float(v) for v in o])
    allX = np.asarray(_all_states(queries, ("rollout", "holdout", "probe")) or base or [[0.0] * len(spec.obs_names)], float)
    std = allX.std(axis=0)
    binary = [len(np.unique(allX[:, i])) <= 2 for i in range(allX.shape[1])]
    sigma = np.where(binary, 0.0, 0.05 * std)
    rng = np.random.default_rng(0)
    extra: list[list[float]] = []
    if base:
        B = np.asarray(base, float)
        for _ in range(600):
            if len(extra) >= 200:
                break
            o = B[rng.integers(len(B))] + rng.normal(0.0, 1.0, B.shape[1]) * sigma
            o = [float(v) for v in o]
            if _inside(o, j, reg["op"], reg["value"]):
                extra.append(o)
    n, _rec, fixed = _programs(bridge, spec, prog, fix)
    rows = []
    for o in base + extra:
        try:
            rows.append((o, int(fixed(o, fix.get("params") or {}))))
        except Exception:  # noqa: BLE001
            continue
    return rows


def _guard_check(bridge, spec, queries, failing: dict | None, prog: dict, fix: dict) -> dict | None:
    n, rec, fixed = _programs(bridge, spec, prog, fix)
    rp, fp = prog.get("params") or {}, fix.get("params") or {}
    reg = failing if failing and failing.get("feature") in spec.obs_names else None
    j = spec.obs_names.index(reg["feature"]) if reg else None
    same = total = 0
    for b in queries:
        if b.get("source") != "holdout":
            continue
        for o, a in zip(b["obs"], b["actions"]):
            if reg is not None and _inside(o, j, reg["op"], reg["value"]):
                continue
            g = n.shielded_action(rec, rp, fixed, fp, o, a)
            total += 1
            try:
                same += int(g) == int(a)
            except Exception:  # noqa: BLE001
                same += g == a
    if not total:
        return None
    pct = same / total
    where = "Outside the flaw zone, the" if reg is not None else "The"
    return {"pct": pct, "n": total,
            "text": f"{where} guarded policy selects the same action as the original network on {pct:.1%} of held-out situations"}


def _cost(bridge, prog: dict, fix: dict) -> dict | None:
    """Robot vs guarded robot on the same seeds. Live bridges only; computed once in the background."""
    ctx = bridge.ctx
    # replays and fallbacks set the same robot + flaw on the live black box (app.server), so the
    # measurement is just as real there; only a bridge without a black box is skipped
    if ctx.get("bb") is None or not ctx.get("env_id"):
        return None
    with _LOCK:
        if "report_cost" in ctx:
            return ctx["report_cost"]
        ctx["report_cost"] = {"pending": True, "seeds": EVAL_SEEDS, "n": len(EVAL_SEEDS)}
    threading.Thread(target=_compute_cost, args=(bridge, prog, fix), daemon=True,
                     name=f"report-cost-{bridge.run_id}").start()
    return ctx["report_cost"]


def _compute_cost(bridge, prog: dict, fix: dict) -> None:
    S = _server()
    ctx = bridge.ctx
    try:
        bb, env_id = ctx["bb"], ctx["env_id"]
        spec = bb.spec()
        n, rec, fixed = _programs(bridge, spec, prog, fix)
        rp, fp = prog.get("params") or {}, fix.get("params") or {}
        reg = ctx.get("failing_region")
        j = spec.obs_names.index(reg["feature"]) if reg and reg.get("feature") in spec.obs_names else None
        threshold = None
        try:
            import gymnasium as gym
            threshold = gym.spec(env_id).reward_threshold
        except Exception:  # noqa: BLE001
            threshold = None
        if str(env_id).startswith("MountainCar"):
            threshold = -199.5        # success = reached the goal before the 200-step limit
        robot, guarded = [], []
        for seed in EVAL_SEEDS:
            robot.append(float(bb.episode(seed).total_reward))
            env, obs = S.make_env_factory(env_id)(seed)
            total = 0.0
            try:
                for _ in range(1000):
                    a_bb = bb.act([obs], source="probe").actions[0]
                    a = guarded_action(n, rec, rp, fixed, fp, obs, a_bb, reg, j)
                    o, r, term, trunc, _ = env.step(int(a) if spec.action_type == "discrete" else a)
                    total += float(r)
                    obs = [float(v) for v in o]
                    if term or trunc:
                        break
            finally:
                env.close()
            guarded.append(total)
        out = {"robot_reward": float(np.mean(robot)), "guarded_reward": float(np.mean(guarded)),
               "seeds": EVAL_SEEDS, "n": len(EVAL_SEEDS),
               "robot_success": None if threshold is None else sum(r >= threshold for r in robot),
               "guarded_success": None if threshold is None else sum(g >= threshold for g in guarded),
               "success_threshold": threshold}
    except Exception as e:  # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"[:200], "seeds": EVAL_SEEDS, "n": len(EVAL_SEEDS)}
    ctx["report_cost"] = out


# ----------------------------------------------------------------------------------------- the card
def build_report(bridge) -> dict:
    events = _events(bridge)
    buf = _buffer(bridge)
    queries = _queries(bridge)
    bb_id, twist_id, twist_known = _bb_and_twist(bridge, events)
    spec = _spec(bridge, bb_id)
    prog = best_program(events)
    revealed = bridge.ctx.get("reveal") or _last(events, "reveal")
    fix = bridge.ctx.get("fix") or _last(events, "fix")
    shield = _last(buf, "shield")
    has_flaw = bool(twist_id) if twist_known else ("reveal" in bridge.source or revealed is not None)
    ready = any(e.get("type") == "run_finished" for e in buf)
    match = prog.get("agreement") if prog else None

    infos = _rule_infos(events, spec, queries) if spec is not None else []
    truth = _truth(bridge, bb_id, twist_id, (revealed or {}).get("feature")) if revealed else None
    failing_info = _pick_failing(infos, events, revealed, truth, prog["code"] if prog else None, spec) \
        if has_flaw else None

    failing = region = evidence = action = guard = cost = None
    if failing_info is not None:
        e, reg = failing_info["entry"], failing_info["region"]
        frac, n_in = _probe_fraction(failing_info, spec)
        act_name = _action_name(spec, reg.get("action")) if reg and reg.get("action") is not None else None
        if reg:
            head = f"When {reg['feature']} is {_OP_WORDS.get(reg['op'], reg['op'])} {reg['value']:.3g}, the robot"
            act = act_name or "a different action"
            if frac is not None and frac >= 0.995:
                sentence = f"{head} always chooses {act}."
            elif frac is not None:
                sentence = f"{head} chooses {act} ({frac:.0%} of probed cases)."
            else:
                sentence = f"{head} chooses {act}."
        else:
            sentence = e.get("text", "")
        failing = {"rule_id": failing_info["rule_id"], "text": e.get("text", ""), "sentence": sentence,
                   "line": _code_line(prog["code"], reg["feature"], spec) if (prog and reg) else None,
                   "feature": reg["feature"] if reg else None, "op": reg["op"] if reg else None,
                   "value": reg["value"] if reg else None, "action": reg.get("action") if reg else None,
                   "action_name": act_name, "probe_fraction": frac, "probe_points_in_region": n_in}
        if reg and spec is not None and reg["feature"] in spec.obs_names:
            j = spec.obs_names.index(reg["feature"])
            rollouts = [b for b in queries if b.get("source") == "rollout"]
            steps = sum(len(b["obs"]) for b in rollouts)
            hits = [sum(_inside(o, j, reg["op"], reg["value"]) for o in b["obs"]) for b in rollouts]
            region = {"text": f"{reg['feature']} {reg['op']} {reg['value']:.3g}",
                      "fire_pct": (sum(hits) / steps) if steps else None,
                      "episodes_hit": sum(h > 0 for h in hits), "episodes_total": len(rollouts)}

    snaps = [e.get("data") or {} for e in events if e.get("type") == "snap"]
    feat = failing.get("feature") if failing else None
    subq_first = subq_turn = None
    for s in snaps:
        if feat and (s.get("anomaly_feature") or {}).get("choice") == feat:
            subq_first, subq_turn = feat, s.get("turn")
            break
    if subq_first is None:
        first = next(((s.get("anomaly_feature") or {}).get("choice") for s in snaps
                      if (s.get("anomaly_feature") or {}).get("choice") not in (None, "none")), None)
        subq_first = first
    n_conf = sum(i["status"] == "confirmed" for i in infos)
    n_ref = sum(i["status"] == "refuted" for i in infos)
    probe_q = failing_info["n_probe"] if failing_info else sum(i["n_probe"] for i in infos)
    parts = [f"Confirmed by {probe_q:,} probe queries" if failing_info else f"{probe_q:,} probe queries"]
    if subq_first:
        parts.append(f"SubQ flagged {subq_first} first" + (f" (turn {subq_turn})" if subq_turn else ""))
    elif not snaps:
        parts.append("SubQ scan off for this run")
    parts.append(f"{n_conf} rules proven, {n_ref} ruled out")
    evidence = {"probe_queries": probe_q, "subq_first": subq_first, "subq_turn": subq_turn,
                "confirmed": n_conf, "refuted": n_ref, "text": " · ".join(parts)}

    dup = set()
    if failing_info and failing_info["region"]:
        fr = failing_info["region"]
        dup = {i["rule_id"] for i in infos if i["region"] and i["region"]["feature"] == fr["feature"]
               and i["region"]["op"] == fr["op"]}
    regression = [{"rule_id": i["rule_id"], "text": i["entry"].get("text", "")} for i in infos
                  if i["status"] == "confirmed" and i["rule_id"] not in dup
                  and (failing_info is None or i["rule_id"] != failing_info["rule_id"])]

    errors = []
    if fix and prog and spec is not None:
        if failing and failing.get("feature"):
            try:
                rows = training_rows(bridge, spec, queries, failing, prog, fix)
                acts = Counter(a for _, a in rows)
                if acts:
                    best_a = acts.most_common(1)[0][0]
                    name = _action_name(spec, best_a)
                    action = {"region_text": region["text"] if region else "", "correct_action": name,
                              "correct_action_id": best_a, "n_training": len(rows),
                              "text": f"Retrain on states where {region['text'] if region else failing['feature']}. "
                                      f"Target action in this region: {name}, taken from the corrected policy."}
            except Exception as e:  # noqa: BLE001
                errors.append(f"training set: {type(e).__name__}: {e}"[:200])
        try:
            guard = _guard_check(bridge, spec, queries, failing, prog, fix)
        except Exception as e:  # noqa: BLE001
            errors.append(f"guard check: {type(e).__name__}: {e}"[:200])
        if shield and shield.get("on"):
            if failing and failing.get("feature") and "failing_region" not in bridge.ctx:
                bridge.ctx["failing_region"] = {k: failing.get(k) for k in
                                                ("feature", "op", "value", "sentence", "line", "action_name")}
            cost = _cost(bridge, prog, fix)

    found = _found(bridge, events, spec, revealed, truth, failing_info) if has_flaw else None
    return {
        "run_id": bridge.run_id, "ready": ready, "has_flaw": has_flaw,
        "revealed": revealed is not None, "fixed": fix is not None,
        "guarded": bool(shield and shield.get("on")),
        "match": match,
        "program": ({"version": prog.get("version"), "lines": len(prog["code"].rstrip().splitlines()),
                     "code": prog["code"], "params": prog.get("params") or {}} if prog else None),
        "failing": failing, "region": region, "cost": cost, "evidence": evidence, "action": action,
        "regression": regression, "guard_check": guard, "found": found,
        "reported_label": (revealed or {}).get("label"),
        "no_flaw_text": (f"No failing decision found. Readable copy matches the robot on {match:.1%}"
                         if not has_flaw and match is not None else
                         ("No failing decision found." if not has_flaw else None)),
        "queries_available": bool(queries), "errors": errors,
    }


# ----------------------------------------------------------------------------------------- downloads
def regression_tests(bridge) -> list[dict]:
    events = _events(bridge)
    queries = _queries(bridge)
    bb_id, twist_id, twist_known = _bb_and_twist(bridge, events)
    spec = _spec(bridge, bb_id)
    if spec is None:
        raise HTTPException(503, "sensor spec unavailable (is the black box running?)")
    rep = build_report(bridge)
    keep = {r["rule_id"] for r in rep["regression"]}
    out = []
    for i in _rule_infos(events, spec, queries):
        if i["rule_id"] not in keep:
            continue
        reg = i["region"]
        states = []
        if reg and reg["feature"] in spec.obs_names:
            j = spec.obs_names.index(reg["feature"])
            states = [o for b in i["batches"] for o in b["obs"] if _inside(o, j, reg["op"], reg["value"])][:20]
        a = reg.get("action") if reg else None
        out.append({"rule_id": i["rule_id"], "text": i["entry"].get("text", ""),
                    "feature": reg["feature"] if reg else None, "op": reg["op"] if reg else None,
                    "value": reg["value"] if reg else None, "expected_action": a,
                    "expected_action_name": _action_name(spec, a) if a is not None else None,
                    "evidence": i["entry"].get("evidence", []), "states": states})
    return out


def markdown(bridge) -> str:
    rep = build_report(bridge)
    events = _events(bridge)
    ledger = (_last(events, "ledger") or {}).get("entries", [])
    fix = bridge.ctx.get("fix") or _last(events, "fix")
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
    L = [f"# Unbox report: {rep['run_id']}", ""]
    prog = rep["program"]
    if prog:
        L.append(f"Readable copy v{prog['version']} ({prog['lines']} lines) matches the robot on "
                 f"{pct(rep['match'])} of held-out situations.")
    L.append("")
    if not rep["has_flaw"]:
        L += ["## Failing decision", rep["no_flaw_text"] or "No failing decision found.", ""]
    else:
        f = rep["failing"]
        L += ["## Failing decision"]
        if f:
            L.append(f"{f['sentence']}" + (f" (line {f['line']} of the readable copy)" if f.get("line") else ""))
            L.append(f"Ledger rule {f['rule_id']}: {f['text']}")
        else:
            L.append("No confirmed rule describes the failing decision.")
        if rep["reported_label"]:
            L.append(f"Reported flaw: {rep['reported_label']} · Unbox verdict: "
                     f"{'Match' if rep['found'] == 'match' else 'Not found'}")
        L.append("")
        if rep["region"]:
            r = rep["region"]
            L += ["## Where and how often",
                  f"Region: {r['text']}. Fires on {pct(r['fire_pct'])} of the robot's own steps, in "
                  f"{r['episodes_hit']} of {r['episodes_total']} rollout episodes.", ""]
        if rep["cost"] and "robot_reward" in rep["cost"]:
            c = rep["cost"]
            line = (f"Mean reward over seeds {c['seeds']}: robot {c['robot_reward']:.1f}, guarded robot "
                    f"{c['guarded_reward']:.1f}.")
            if c.get("robot_success") is not None:
                line += f" Success: robot {c['robot_success']}/{c['n']} -> guarded {c['guarded_success']}/{c['n']}."
            L += ["## What it costs", line, ""]
    L += ["## Evidence", rep["evidence"]["text"], ""]
    if rep["action"]:
        L += ["## Post-training action", rep["action"]["text"],
              f"Targeted training set: {rep['action']['n_training']} labeled situations "
              f"(/api/report/{rep['run_id']}/training.csv).", ""]
    L += ["## Regression tests", "After retraining, these behaviors must not change:"]
    L += [f"- {r['rule_id']}: {r['text']}" for r in rep["regression"]] or ["- (none confirmed)"]
    L.append("")
    if rep["guard_check"]:
        L += ["## Guard check", rep["guard_check"]["text"] + ".", ""]
    if prog:
        L += ["## Readable copy", "```python", prog["code"].rstrip(), "```", "", "Fitted parameters:"]
        L += [f"- {k} = {v:.4g}" for k, v in prog["params"].items()] or ["- (none)"]
        L.append("")
    if fix:
        L += ["## Fixed copy", "```python", fix.get("code", "").rstrip(), "```", ""]
        if fix.get("diff"):
            L += ["```diff", fix["diff"].rstrip(), "```", ""]
    L += ["## Ledger", "| rule | status | text |", "|---|---|---|"]
    L += [f"| {e.get('rule_id')} | {e.get('status')} | {str(e.get('text', '')).replace('|', '/')} |" for e in ledger]
    L.append("")
    return "\n".join(L)


def _download(body: str | bytes, media: str, filename: str) -> Response:
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/api/report/{run_id}/report.md")
def report_md(run_id: str) -> Response:
    bridge = _server().get_bridge(run_id)
    return _download(markdown(bridge), "text/markdown; charset=utf-8", f"unbox-report-{run_id}.md")


@router.get("/api/report/{run_id}/training.csv")
def report_training(run_id: str) -> Response:
    bridge = _server().get_bridge(run_id)
    events = _events(bridge)
    fix = bridge.ctx.get("fix") or _last(events, "fix")
    if fix is None:
        raise HTTPException(404, "no fix yet: press Reveal, then Fix our copy")
    rep = build_report(bridge)
    if not rep["failing"] or not rep["failing"].get("feature"):
        raise HTTPException(404, "no confirmed failing decision with a sensor region")
    bb_id, _, _ = _bb_and_twist(bridge, events)
    spec = _spec(bridge, bb_id)
    rows = training_rows(bridge, spec, _queries(bridge), rep["failing"], best_program(events), fix)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(spec.obs_names) + ["label_action_id", "label_action_name"])
    for o, a in rows:
        w.writerow([f"{v:.6g}" for v in o] + [a, _action_name(spec, a)])
    return _download(buf.getvalue(), "text/csv; charset=utf-8", f"unbox-training-{run_id}.csv")


@router.get("/api/report/{run_id}/tests.json")
def report_tests(run_id: str) -> Response:
    bridge = _server().get_bridge(run_id)
    body = json.dumps({"run_id": run_id, "tests": regression_tests(bridge)}, indent=1)
    return _download(body, "application/json", f"unbox-regression-tests-{run_id}.json")


@router.get("/api/report/{run_id}")
def report(run_id: str):
    if run_id.endswith(".md"):
        return report_md(run_id[:-3])
    bridge = _server().get_bridge(run_id)
    from app.server import _jsonable
    return _jsonable(build_report(bridge))
