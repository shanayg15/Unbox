"""Lane A, task A6: batch runs, cross-match, baselines.

    python -m scripts.batch --snap 1,0                    # everything: every policy x flaw x SubQ on/off
    python -m scripts.batch --snap 1 --bb bb-01 --twist t-cp-1 --stages runs     # one run

Every run goes through agent.loop.run headless (live=False), exactly as the app runs it, against a private
black-box server this script starts on --port. Setting a twist and resetting the query counter are
server-wide per policy, so a shared server would let the UI or another session corrupt a batch run.

Writes, and rewrites after every finished run so the results tab fills in as it goes:
  results/summary.json      one row per run
  results/crossmatch.json   each clean program scored against every policy of its env
  results/baselines.json    a controller written from the task name alone, no queries
  cache/<bb>__<twist>__<brain>/   SubQ-on runs, for the app's live fallback
Working files sit in results/rows/, so a rerun skips what already finished (--force reruns it).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numerics
from agent import loop as agent_loop
from agent import prompts
from agent.brain import Brain, extract_json
from store.store import get_store

ROOT = Path(__file__).resolve().parent.parent
# Demo policy first, then its CartPole integration target, then their cross-match partners. The cut rule
# drops MountainCar and Acrobot first, so they go last.
POLICY_ORDER = ["bb-05", "bb-01", "bb-06", "bb-02", "bb-03", "bb-04"]
CACHED_FILES = ["events.jsonl", "brain_calls.jsonl", "programs.jsonl", "ledger.jsonl", "eval.json", "queries.jsonl"]
NEAR = 0.2                  # twist_found: a threshold within 20% of the true one
TASKS = {
    "CartPole-v1": "CartPole: keep a pole balanced upright on a cart by pushing the cart left or right, "
                   "and keep the cart near the middle of the track",
    "LunarLander-v3": "LunarLander: land the lander gently on the pad between the flags, using the main "
                      "engine and the left and right side engines",
    "MountainCar-v0": "MountainCar: drive an underpowered car up the right-hand hill to the flag by "
                      "rocking back and forth to build momentum",
    "Acrobot-v1": "Acrobot: swing a two-link arm hanging from a bar until its tip rises above the bar",
}


# --------------------------------------------------------------------------------------------- plan

@dataclass(frozen=True)
class Config:
    bb: str
    label: str
    env_id: str
    twist: str | None
    feature: str | None         # the truth, kept here to score the run; never passed to the agent
    threshold: float | None
    snap: bool
    brain: str

    @property
    def cache_key(self) -> str:
        return f"{self.bb}__{self.twist or 'none'}__{self.brain}"

    @property
    def key(self) -> str:
        return f"{self.cache_key}__snap{int(self.snap)}"


@dataclass(frozen=True)
class Paths:
    results: Path = ROOT / "results"
    cache: Path = ROOT / "cache"

    @property
    def rows(self) -> Path:
        return self.results / "rows"

    @property
    def baseline_rows(self) -> Path:
        return self.rows / "baselines"


@dataclass(frozen=True)
class RunOptions:
    max_turns: int = 6
    budget: int = 5000
    force: bool = False
    timeout_s: float | None = None


def _policy_rank(bb: str) -> tuple[int, str]:
    return (POLICY_ORDER.index(bb) if bb in POLICY_ORDER else len(POLICY_ORDER), bb)


def plan_configs(catalog: dict, snaps: list[bool], bbs: list[str] | None = None,
                 twists: list[str] | None = None, brain: str = "muse") -> list[Config]:
    """Every policy x {each twist, no twist} x snap setting, demo config first."""
    unknown = [bb for bb in bbs or [] if bb not in catalog]
    if unknown:
        raise ValueError(f"unknown policy {', '.join(unknown)}; known: {', '.join(sorted(catalog))}")
    configs = []
    for bb in sorted(bbs or catalog, key=_policy_rank):
        entry = catalog[bb]
        options = sorted(entry["twists"], key=lambda t: t["id"]) + [None]
        for twist in options:
            twist_id = twist["id"] if twist else None
            if twists is not None and (twist_id or "none") not in twists:
                continue
            for snap in snaps:
                configs.append(Config(bb=bb, label=entry["label"], env_id=entry["env_id"], twist=twist_id,
                                      feature=twist["feature"] if twist else None,
                                      threshold=twist["threshold"] if twist else None, snap=snap, brain=brain))
    return configs


def group_by_policy(configs: list[Config]) -> list[list[Config]]:
    """One group per policy, in plan order. A group runs in order: its twist is server-wide."""
    groups: dict[str, list[Config]] = {}
    for config in configs:
        groups.setdefault(config.bb, []).append(config)
    return list(groups.values())


def demo_seed_for(bb: str) -> int:
    """Same lookup as the app, so a cached run's clip shows the seed the live demo uses."""
    return int(os.getenv(f"DEMO_SEED_{bb.replace('-', '_').upper()}", os.getenv("DEMO_SEED", "0")))


# ------------------------------------------------------------------------------- did it find the flaw?

_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?!\w)")
_TURN = re.compile(r"^Turn (\d+):")
_PROBE = re.compile(r"^Probing (\S+): .* where (\S+) ([<>]) (\S+)$")


def _near(value: float, truth: float) -> bool:
    return abs(value - truth) <= NEAR * abs(truth) + 1e-12


def _mentions(text: str, feature: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(feature)}(?!\w)", text) is not None


def fitted_limits(code: str, params: dict, obs_names: list[str]) -> dict[str, list[float]]:
    """Fitted values a program compares a feature against, e.g. `v > P["t"]` with v = obs[1] -> {cart_v: [t]}."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    alias: dict[str, int] = {}

    def index_of(node) -> int | None:
        if isinstance(node, ast.Name):
            return alias.get(node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "obs" \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            return node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" \
                and len(node.args) == 1:
            return index_of(node.args[0])
        return None

    def param_of(node) -> float | None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = param_of(node.operand)
            return None if value is None else -value
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "P" \
                and isinstance(node.slice, ast.Constant) and node.slice.value in params:
            return float(params[node.slice.value])
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and index_of(node.value) is not None:
                alias[target.id] = index_of(node.value)
            elif isinstance(target, ast.Tuple):
                if isinstance(node.value, ast.Name) and node.value.id == "obs":
                    sources = [ast.Subscript(ast.Name("obs"), ast.Constant(i)) for i in range(len(target.elts))]
                elif isinstance(node.value, ast.Tuple):
                    sources = node.value.elts
                else:
                    continue
                for name, source in zip(target.elts, sources):
                    if isinstance(name, ast.Name) and index_of(source) is not None:
                        alias[name.id] = index_of(source)

    limits: dict[str, list[float]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for a, b in zip(operands, operands[1:]):
            for side, other in ((a, b), (b, a)):
                index, value = index_of(side), param_of(other)
                if index is not None and value is not None and 0 <= index < len(obs_names):
                    is_abs = isinstance(side, ast.Call)
                    limits.setdefault(obs_names[index], []).extend([value, -value] if is_abs else [value])
    return limits


def twist_evidence(events: list[dict], feature: str, threshold: float, obs_names: list[str]) -> dict:
    """When (if ever) a confirmed ledger rule pinned down the twist.

    A rule counts once it is confirmed (only Python probes confirm), mentions the twist's feature, and a
    threshold within 20% of the true one appears in its text, in the region its confirming probe expected,
    or in the current program's fitted limit on that feature. Hard rule 3 makes that last one the real
    threshold: the brain only guesses numbers and the optimizer fits them.
    """
    turn = queries = None
    regions: dict[str, list[tuple[str, float]]] = {}
    limits: dict[str, list[float]] = {}
    entries: list[dict] = []
    found: dict | None = None

    def hit(entry: dict) -> bool:
        if entry.get("status") != "confirmed":
            return False
        text = entry.get("text", "")
        probed = [value for f, value in regions.get(entry.get("rule_id"), []) if f == feature]
        if not (_mentions(text, feature) or probed):
            return False
        numbers = [float(n) for n in _NUMBER.findall(text)] if _mentions(text, feature) else []
        return any(_near(v, threshold) for v in numbers + probed + limits.get(feature, []))

    def look() -> None:
        nonlocal found
        if found is None:
            rule = next((e for e in entries if hit(e)), None)
            if rule is not None:
                found = {"rule_id": rule["rule_id"], "turn": turn, "queries": None}

    for event in events:
        kind, data = event.get("type"), event.get("data", {})
        if kind == "status":
            text = data.get("text", "")
            if (m := _TURN.match(text)):
                turn = int(m[1])
            elif (m := _PROBE.match(text)):
                try:
                    regions.setdefault(m[1], []).append((m[2], float(m[4])))
                except ValueError:
                    pass        # the status line was cut off mid-number
        elif kind == "snap" and data.get("turn") is not None:
            turn = data["turn"]
        elif kind in ("query_count", "metric"):
            queries = data.get("n", data.get("queries"))
            if found is not None and found["queries"] is None:
                found["queries"] = queries
        elif kind == "program":
            limits = fitted_limits(data.get("code", ""), data.get("params", {}), obs_names)
            look()
        elif kind == "ledger":
            entries = data.get("entries", [])
            if found is not None and any(e.get("rule_id") == found["rule_id"] and e.get("status") == "refuted"
                                         for e in entries):
                found = None
            look()

    if found is None:
        return {"found": False, "queries_to_twist": None, "turns_to_twist": None}
    return {"found": True, "queries_to_twist": found["queries"] if found["queries"] is not None else queries,
            "turns_to_twist": found["turn"]}


# ------------------------------------------------------------------------------------------- one run

def _round(value, digits):
    return None if value is None else round(float(value), digits)


def make_row(config: Config, events: list[dict], out: dict, run_id: str, obs_names: list[str]) -> dict:
    started = next((e["data"] for e in events if e["type"] == "run_started"), {})
    finished = next((e["data"] for e in reversed(events) if e["type"] == "run_finished"), {})
    report = asdict(out["eval"])
    program = out["program"]
    us_bb, us_clone = report.get("us_per_action_bb"), report.get("us_per_action_clone")
    if config.twist:
        evidence = twist_evidence(events, config.feature, config.threshold, obs_names)
    else:
        evidence = {"found": False, "queries_to_twist": None, "turns_to_twist": None}
    return {
        "bb": config.bb, "label": config.label, "twist": config.twist or "none", "brain": config.brain,
        "model": started.get("model"), "snap": config.snap,
        "agreement": _round(report["open_loop_agreement"], 4),
        "clone_reward": _round(report.get("clone_mean_reward"), 1),
        "bb_reward": _round(report.get("bb_mean_reward"), 1),
        "bound_95": _round(report.get("failure_rate_bound_95"), 3),
        "n_failures": report.get("n_failures"),
        "lines_of_code": sum(1 for line in program.code.splitlines() if line.strip()),
        "queries": finished.get("total_queries"), "turns": finished.get("turns"),
        "seconds": finished.get("seconds"),
        "speedup": _round(us_bb / us_clone, 1) if us_bb and us_clone else None,
        "twist_found": evidence["found"], "queries_to_twist": evidence["queries_to_twist"],
        "turns_to_twist": evidence["turns_to_twist"],
        "brain_errors": sum(1 for e in events if e["type"] == "error"),
        "ok": program.version > 0,     # false: the brain never produced a program and a constant stood in
        "run_id": run_id, "code": program.code, "params": program.params,
    }


def _write_json(path: Path, obj) -> None:
    """Write then rename, so the app never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _cache_run(run_dir: Path, dest: Path) -> None:
    staging = dest.with_name(dest.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in CACHED_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, staging / name)
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)


def run_config(config: Config, world, paths: Paths, options: RunOptions) -> dict:
    row_path = paths.rows / f"{config.key}.json"
    if row_path.exists() and not options.force:
        return json.loads(row_path.read_text())

    world.configure(config.bb, config.twist)
    store = get_store()
    run_id = f"batch-{time.strftime('%m%d-%H%M%S')}-{config.key}"
    previous = os.environ.get("SNAP")
    os.environ["SNAP"] = "1" if config.snap else "0"     # agent.loop reads it at the start of each run
    try:
        out = agent_loop.run(world.client(config.bb), config.brain, env_factory=world.env_factory(config.bb),
                             numerics=numerics, emit=None, store=store, run_id=run_id,
                             budget=options.budget, max_turns=options.max_turns, live=False,
                             timeout_s=options.timeout_s, policy_label=config.label,
                             demo_seed=demo_seed_for(config.bb))
    finally:
        if previous is None:
            os.environ.pop("SNAP", None)
        else:
            os.environ["SNAP"] = previous

    obs_names = world.client(config.bb).spec().obs_names
    row = make_row(config, store.load_events(run_id), out, run_id, obs_names)
    if config.snap and row["ok"]:
        _cache_run(Path(store.events_path(run_id)).parent, paths.cache / config.cache_key)
    _write_json(row_path, row)
    return row


def _row_rank(row: dict) -> tuple:
    return (_policy_rank(row["bb"]), row["twist"] == "none", row["twist"], not row.get("snap"))


def write_summary(paths: Paths) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in paths.rows.glob("*.json")]
    rows = sorted(rows, key=_row_rank)
    public = [{k: v for k, v in row.items() if k not in ("code", "params")} for row in rows]
    _write_json(paths.results / "summary.json", {"rows": public})
    return public


def _log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def run_group(configs: list[Config], world, paths: Paths, options: RunOptions) -> list[str]:
    """Run one policy's configs in order. A failed run leaves no row, so the next batch retries it."""
    failures = []
    for config in configs:
        done = (paths.rows / f"{config.key}.json").exists() and not options.force
        _log(f"{config.key}: {'already done, skipping' if done else 'start'}")
        t0 = time.time()
        try:
            row = run_config(config, world, paths, options)
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the batch
            failures.append(f"{config.key}: {type(e).__name__}: {e}")
            _log(f"{config.key}: FAILED {type(e).__name__}: {e}")
            continue
        finally:
            write_summary(paths)
        if not done:
            _log(f"{config.key}: done in {time.time() - t0:.0f}s, agreement {row['agreement']}, "
                 f"twist found {row['twist_found']} (turn {row['turns_to_twist']}, {row['queries_to_twist']} "
                 f"queries), {row['queries']} queries, {row['turns']} turns, brain errors {row['brain_errors']}")
    return failures


# --------------------------------------------------------------------------------- cross-match + baseline

def _holdout(world, bb: str):
    """Clean behaviour on the agent's holdout seeds."""
    world.configure(bb, None)
    client = world.client(bb)
    batches = []
    for seed in agent_loop.HOLDOUT_SEEDS:
        episode = client.episode(seed)
        batches.append(agent_loop._episode_batch(episode, f"hold-s{seed}", "holdout"))
    return client, client.spec(), batches


def _envs(catalog: dict) -> dict[str, list[str]]:
    envs: dict[str, list[str]] = {}
    for bb in sorted(catalog, key=_policy_rank):
        envs.setdefault(catalog[bb]["env_id"], []).append(bb)
    return envs


def crossmatch(catalog: dict, world, paths: Paths) -> dict:
    """Each policy's no-flaw program, scored on held-out clean behaviour of every policy of its env."""
    rows = [json.loads(p.read_text()) for p in paths.rows.glob("*.json")]
    matrix: dict[str, dict[str, float]] = {}
    for bbs in _envs(catalog).values():
        if len(bbs) < 2:
            continue
        clean = {}
        for bb in bbs:
            _, spec, batches = _holdout(world, bb)
            clean[bb] = (spec, [o for b in batches for o in b.obs], [a for b in batches for a in b.actions])
        for bb in bbs:
            candidates = [r for r in rows if r["bb"] == bb and r["twist"] == "none" and r.get("ok", True)]
            if not candidates:
                continue
            row = sorted(candidates, key=lambda r: not r.get("snap"))[0]
            spec = clean[bb][0]
            fn = numerics.load_program(row["code"], spec)
            matrix[f"{bb} program"] = {other: round(numerics.agreement(fn, row["params"], X, y, spec), 3)
                                       for other, (_, X, y) in clean.items()}
    _write_json(paths.results / "crossmatch.json", matrix)
    return matrix


def _zero_query_program(brain_name: str, env_id: str, spec, store):
    """One controller from the task name and sensor spec alone. Returns (brain, fn, code, params)."""
    brain = Brain(brain_name, run_id=f"batch-baseline-{env_id}", store=store)
    prompt = prompts.ZERO_QUERY.format(task_name=TASKS.get(env_id, env_id), spec=prompts.spec_text(spec))
    messages = [{"role": "user", "content": prompt}]
    error = None
    for _ in range(3):
        text = brain.complete(messages)
        try:
            program = extract_json(text)["program"]
            code, params = program["code"], {k: float(v) for k, v in (program.get("params") or {}).items()}
            return brain, numerics.load_program(code, spec), code, params
        except Exception as e:  # noqa: BLE001 - bad JSON or a rejected program goes back to the brain
            error = f"{type(e).__name__}: {e}"
            messages = messages + [{"role": "assistant", "content": text[:6000]},
                                   {"role": "user", "content": f"That was rejected: {error}\n"
                                    "Reply again with ONLY the JSON object."}]
    raise RuntimeError(f"no valid program after 3 tries: {error}")


def baselines(catalog: dict, world, paths: Paths, brain_name: str) -> list[dict]:
    """Zero-query baseline per env (one brain call), evaluated against each of its policies.
    Its params are the brain's guesses: with no data there is nothing to fit them to."""
    for env_id, bbs in _envs(catalog).items():
        row_paths = {bb: paths.baseline_rows / f"{bb}.json" for bb in bbs}
        if all(p.exists() and "error" not in json.loads(p.read_text()) for p in row_paths.values()):
            continue
        _log(f"zero-query baseline for {env_id}")
        client, spec, _ = _holdout(world, bbs[0])
        try:
            brain, fn, code, params = _zero_query_program(brain_name, env_id, spec, get_store())
            model, failure = brain.model, None
        except Exception as e:  # noqa: BLE001
            model, failure = None, f"{type(e).__name__}: {e}"
            _log(f"zero-query baseline for {env_id} FAILED: {failure}")
        for bb in bbs:
            row = {"bb": bb, "label": catalog[bb]["label"], "kind": "zero-query", "model": model}
            if failure:
                row.update(agreement=None, clone_reward=None, bb_reward=None, error=failure)
            else:
                client, spec, holdout = _holdout(world, bb)
                report = numerics.evaluate(fn, params, client, spec, holdout, world.env_factory(bb),
                                           agent_loop.EVAL_SEEDS)
                row.update(agreement=_round(report.open_loop_agreement, 4),
                           clone_reward=_round(report.clone_mean_reward, 1),
                           bb_reward=_round(report.bb_mean_reward, 1),
                           bound_95=_round(report.failure_rate_bound_95, 3),
                           lines_of_code=sum(1 for line in code.splitlines() if line.strip()), code=code)
            _write_json(row_paths[bb], row)

    rows = [json.loads(p.read_text()) for p in paths.baseline_rows.glob("*.json")]
    rows = sorted(rows, key=lambda r: _policy_rank(r["bb"]))
    public = [{k: v for k, v in r.items() if k != "code"} for r in rows]
    _write_json(paths.results / "baselines.json", {"rows": public})
    return public


# ------------------------------------------------------------------------------------- real black box

class HttpWorld:
    """Admin access to the private black-box server, plus the env each policy runs in."""

    def __init__(self, url: str, token: str, catalog: dict):
        self.url, self.token, self.catalog = url, token, catalog

    def configure(self, bb: str, twist: str | None) -> None:
        import httpx

        response = httpx.post(f"{self.url}/admin/config", json={"bb": bb, "twist_id": twist},
                              headers={"X-Admin-Token": self.token}, timeout=60)
        response.raise_for_status()

    def client(self, bb: str):
        from blackbox.client import HttpBlackBox

        return HttpBlackBox(bb, self.url)

    def env_factory(self, bb: str):
        env_id = self.catalog[bb]["env_id"]

        def factory(seed: int):
            import gymnasium as gym

            env = gym.make(env_id, render_mode="rgb_array")    # the clone clip renders from this env
            obs, _ = env.reset(seed=int(seed))
            return env, [float(v) for v in obs]
        return factory


def load_catalog() -> dict:
    from blackbox.policies import REGISTRY
    from blackbox.twists import twists_for

    return {bb: {"label": d.label, "env_id": d.env_id,
                 "twists": [{"id": t.twist_id, "feature": t.feature, "threshold": t.threshold}
                            for t in twists_for(bb)]}
            for bb, d in REGISTRY.items()}


def start_server(port: int) -> tuple[str, str]:
    """The black box in a thread of this process. Returns (url, admin token)."""
    import httpx
    import uvicorn

    from blackbox.server import app

    token = os.environ.get("BLACKBOX_ADMIN_TOKEN") or secrets.token_hex(16)
    os.environ["BLACKBOX_ADMIN_TOKEN"] = token
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="blackbox", daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if server.started:
            try:
                httpx.get(f"{url}/health", timeout=2).raise_for_status()
                return url, token
            except httpx.HTTPError:
                pass
        if not threading.main_thread().is_alive():
            break
        time.sleep(0.2)
    raise RuntimeError(f"black-box server did not start on port {port} (is the port taken?)")


def _run_group_in_worker(configs, url, token, catalog, paths, options) -> list[str]:
    os.environ["BLACKBOX_ADMIN_TOKEN"] = token
    return run_group(configs, HttpWorld(url, token, catalog), paths, options)


# ---------------------------------------------------------------------------------------------- CLI

def _snaps(text: str) -> list[bool]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values or any(v not in ("0", "1") for v in values):
        raise argparse.ArgumentTypeError("--snap takes 1, 0, or 1,0")
    return [v == "1" for v in values]


def _names(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="A6: batch runs with SubQ on/off, cross-match, baselines.")
    ap.add_argument("--snap", type=_snaps, default=[True, False], help="1,0 (default), 1 or 0")
    ap.add_argument("--bb", type=_names, default=None, help="comma list of policies (default: all)")
    ap.add_argument("--twist", type=_names, default=None, help="comma list of twist ids and/or none")
    ap.add_argument("--brain", default=os.getenv("BRAIN", "muse"))
    ap.add_argument("--stages", type=_names, default=["runs", "crossmatch", "baselines"])
    ap.add_argument("--jobs", type=int, default=2, help="policies run in parallel, one process each")
    ap.add_argument("--port", type=int, default=8002, help="private black-box server port")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--budget", type=int, default=int(os.getenv("QUERY_BUDGET", "5000")))
    ap.add_argument("--run-timeout", type=float, default=900.0, help="seconds per run before it wraps up")
    ap.add_argument("--force", action="store_true", help="rerun configs that already have a row")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    full_catalog = load_catalog()
    configs = plan_configs(full_catalog, args.snap, bbs=args.bb, twists=args.twist, brain=args.brain)
    catalog = {bb: full_catalog[bb] for bb in (args.bb or full_catalog)}
    paths = Paths()
    options = RunOptions(max_turns=args.max_turns, budget=args.budget, force=args.force,
                         timeout_s=args.run_timeout)
    url, token = start_server(args.port)
    world = HttpWorld(url, token, catalog)
    _log(f"black box on {url}; {len(configs)} runs planned in {len(group_by_policy(configs))} policy groups, "
         f"{args.jobs} at a time; stages {', '.join(args.stages)}")

    failures: list[str] = []
    if "runs" in args.stages and configs:
        groups = group_by_policy(configs)
        if args.jobs <= 1:
            for group in groups:
                failures += run_group(group, world, paths, options)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(_run_group_in_worker, g, url, token, full_catalog, paths, options)
                           for g in groups]
                for future in as_completed(futures):
                    try:
                        failures += future.result()
                    except Exception as e:  # noqa: BLE001
                        failures.append(f"worker crashed: {type(e).__name__}: {e}")
        write_summary(paths)
    if "crossmatch" in args.stages:
        matrix = crossmatch(catalog, world, paths)
        _log(f"cross-match: {json.dumps(matrix)}")
    if "baselines" in args.stages:
        for row in baselines(catalog, world, paths, args.brain):
            _log(f"baseline {row['bb']}: agreement {row.get('agreement')}, clone reward {row.get('clone_reward')}, "
                 f"bb reward {row.get('bb_reward')}{', ERROR ' + row['error'] if 'error' in row else ''}")

    for failure in failures:
        _log(f"FAILED {failure}")
    _log(f"batch finished: {len(failures)} failed runs")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
