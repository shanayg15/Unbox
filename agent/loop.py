"""Lane B, task B3: the agent loop.

    USE_MOCKS=1 python -m agent.loop --bb mock --brain muse
    python -m agent.loop --bb bb-01 --env CartPole-v1 --brain muse     # real black box (twist set in the app)

The agent never builds envs or learns the env id: the caller passes `bb`, `env_factory` and `numerics`.
(The CLI builds env_factory from the --env the human types.)
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import os
import random
import threading
import time
import uuid

from dotenv import load_dotenv

from agent import probes as probes_mod
from agent import prompts
from agent.brain import Brain, BrainError
from agent.ledger import Ledger
from contracts.types import EvalReport, Program, QueryBatch

load_dotenv()

ROLLOUT_SEEDS = list(range(0, 8))
HOLDOUT_SEEDS = list(range(100, 108))
EVAL_SEEDS = list(range(200, 220))
FRAME_EVERY = 3
TARGET_AGREEMENT = 0.97
FALLBACK_CODE = "def policy(obs, P):\n    return 0\n"


class _Emitter:
    def __init__(self, run_id: str, emit, store, t0: float) -> None:
        self.run_id, self.emit, self.store, self.t0 = run_id, emit, store, t0
        self._lock = threading.Lock()

    def __call__(self, type_: str, **data) -> dict:
        from store.store import to_jsonable

        ev = {"type": type_, "run_id": self.run_id, "t": round(time.time() - self.t0, 3), "data": to_jsonable(data)}
        with self._lock:
            if self.store is not None:
                try:
                    self.store.save_event(self.run_id, ev)
                except Exception as e:  # noqa: BLE001
                    print(f"[loop] store.save_event failed: {e}")
            if self.emit is not None:
                try:
                    self.emit(ev)
                except Exception as e:  # noqa: BLE001
                    print(f"[loop] emit failed: {e}")
        return ev


def _episode_batch(ep, qid: str, source: str) -> QueryBatch:
    return QueryBatch(query_id=qid, obs=[list(map(float, o)) for o in ep.obs], actions=list(ep.actions), source=source)


def _changed_lines(old: str | None, new: str) -> list[int]:
    """1-based line numbers in `new` that are added or changed vs `old`."""
    if not old:
        return list(range(1, len(new.splitlines()) + 1))
    out = []
    sm = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out += list(range(j1 + 1, j2 + 1))
    return out


def _same(a, b) -> bool:
    try:
        return int(a) == int(b)
    except Exception:  # noqa: BLE001 - continuous
        return a == b


def _end_info(terminated: bool, truncated: bool, rewards: list[float], steps: int) -> dict:
    return {"terminated": bool(terminated), "truncated": bool(truncated), "steps": int(steps),
            "last_reward": float(rewards[-1]) if rewards else None,
            "total_reward": float(sum(rewards)) if rewards else 0.0}


def render_pair(bb, env_factory, fn, params, demo_seed: int, frame_every: int = FRAME_EVERY,
                max_steps: int = 1000, bb_only: bool = False) -> dict:
    """Black-box clip (cached by the black box) + clone clip from a local env with the same seed.

    disagree[i] belongs to clone frame i: at the exact state shown in that frame, the robot (asked with one
    batched /act call) would choose a different action than the copy did. Same length as `clone`, so the
    UI sets the red border from the same index, in the same tick, as the clone image."""
    from contracts.mocks import encode_frame

    ep = bb.episode(demo_seed, render=True, frame_every=frame_every)
    out = {"seed": demo_seed, "fps": round(50 / frame_every), "bb": ep.frames or [],
           "bb_end": _end_info(ep.terminated, ep.truncated, ep.rewards, len(ep.actions)),
           "clone": [], "disagree": [], "clone_end": None}
    if bb_only:
        return out
    clone_frames, clone_obs, clone_acts, rewards = [], [], [], []
    term = trunc = False
    steps = 0
    try:
        env, obs = env_factory(demo_seed)
        try:
            for step in range(max_steps):
                o = list(map(float, obs))
                a = fn(o, params)
                if step % frame_every == 0:
                    rgb = env.render()
                    if rgb is None:
                        break
                    clone_frames.append(encode_frame(rgb))
                    clone_obs.append(o)
                    clone_acts.append(a)
                obs, r, term, trunc, _ = env.step(int(a) if not isinstance(a, list) else a)
                rewards.append(float(r))
                steps = step + 1
                if term or trunc:
                    break
        finally:
            env.close()
    except Exception as e:  # noqa: BLE001 - a missing clone clip never stops the run
        print(f"[loop] clone clip failed: {type(e).__name__}: {e}")
    disagree = [False] * len(clone_frames)
    if clone_obs:
        try:
            qb = bb.act(clone_obs, source="probe")
            disagree = [not _same(ca, ba) for ca, ba in zip(clone_acts, qb.actions)]
        except Exception as e:  # noqa: BLE001
            print(f"[loop] disagree check failed: {type(e).__name__}: {e}")
    out.update(clone=clone_frames, disagree=disagree[:len(clone_frames)],
               clone_end=_end_info(term, trunc, rewards, steps))
    return out


def run(bb, brain: str, *, env_factory, numerics, emit, store, run_id: str,
        budget: int, max_turns: int, live: bool, timeout_s: float | None,
        policy_label: str, demo_seed: int = 0) -> dict:
    """bb: contracts.types.BlackBox. env_factory(seed) -> (env, first_obs).
    numerics: module with the contracts/numerics_api.py functions (real or stub).
    emit(event_dict) is called from this thread; must be thread-safe on the caller side.
    Returns {"program": Program, "ledger": [LedgerEntry], "eval": EvalReport, "events_path": str}."""
    t0 = time.time()
    if store is None:
        from store.store import get_store

        store = get_store()
    E = _Emitter(run_id, emit, store, t0)
    use_snap = os.getenv("SNAP", "1") == "1"
    full_ctx = os.getenv("CONTEXT_MODE", "compact") == "full"
    fit_seconds = 4.0 if live else 8.0
    prober_max = 200 if live else 400

    def over_time() -> bool:
        return timeout_s is not None and time.time() - t0 > timeout_s

    def remaining() -> int:
        return max(0, budget - bb.query_count())

    try:
        brain_client = Brain(brain, run_id=run_id, store=store, live=live)
    except Exception as e:  # noqa: BLE001
        E("error", message=f"brain unavailable: {e}")
        raise
    E("run_started", policy_label=policy_label, brain=brain, model=brain_client.model, budget=budget, live=live)

    # 1. data ------------------------------------------------------------------
    spec = bb.spec()
    E("status", text=f"Watching the robot: {len(ROLLOUT_SEEDS)} rollouts + {len(HOLDOUT_SEEDS)} holdout episodes")
    batches: list[QueryBatch] = []
    for s in ROLLOUT_SEEDS:
        batches.append(_episode_batch(bb.episode(s), f"roll-s{s}", "rollout"))
    holdout = [_episode_batch(bb.episode(s), f"hold-s{s}", "holdout") for s in HOLDOUT_SEEDS]
    for b in batches + holdout:
        store.save_queries(run_id, b)
    HX = [o for b in holdout for o in b.obs]
    Hy = [a for b in holdout for a in b.actions]
    X = [o for b in batches for o in b.obs]
    y = [a for b in batches for a in b.actions]
    rows = [(b.query_id, o, a, b.source) for b in batches for o, a in zip(b.obs, b.actions)]
    rng = random.Random(0)
    seed_obs = X if len(X) <= 2000 else rng.sample(X, 2000)
    E("query_count", n=bb.query_count())

    # the robot's own demo-seed clip goes up first, before any copy exists
    try:
        E("frames", version=0, **render_pair(bb, env_factory, None, None, demo_seed, bb_only=True))
    except Exception as e:  # noqa: BLE001
        E("status", text=f"robot clip skipped ({type(e).__name__})")

    # 2. hints -------------------------------------------------------------------
    hints = numerics.pattern_hints(X, y, spec)

    ledger = Ledger()
    version = 0
    code: str | None = None
    fn = None
    params: dict = {}
    agreement: float | None = None
    best = None                                     # (agreement, version, code, params, fn)
    best_frames_agr = -1.0
    history: list[float] = []
    last_error = ""
    probe_results: list[dict] = []
    cex: list = []
    turns_done = 0

    # 3. turns -------------------------------------------------------------------
    longest_turn_s, turn_t0 = 0.0, None
    stopped_for_time = False
    for turn in range(1, max_turns + 1):
        if turn_t0 is not None:
            longest_turn_s = max(longest_turn_s, time.time() - turn_t0)
        # live: don't start a turn that can't finish before the time limit (the app gives up then)
        if over_time() or (timeout_s is not None and time.time() - t0 + longest_turn_s > timeout_s):
            E("status", text="Live time limit reached")
            stopped_for_time = True
            break
        turn_t0 = time.time()
        turns_done = turn

        scan = None
        if use_snap:
            try:
                from agent import snap

                E("status", text=f"SubQ Snap reading all {len(rows):,} raw rows")
                scan = snap.scan(spec, rows, cex, ledger.snapshot(), run_id, store=store)
                E("snap", turn=turn, n_rows=scan["n_rows"], input_tokens=scan["input_tokens"],
                  feature_relevance=scan["feature_relevance"], anomaly_feature=scan["anomaly_feature"],
                  next_probe=scan["next_probe"], rule_support=scan["rule_support"])
            except Exception as e:  # noqa: BLE001 - never block the loop
                scan = None
                E("status", text=f"SubQ scan skipped ({type(e).__name__})")

        msgs = [{"role": "system", "content": prompts.system_prompt()},
                {"role": "user", "content": _turn_context(spec, bb.query_count(), budget, turn, max_turns, version,
                                                          agreement, params, code, ledger, len(X), hints, scan,
                                                          use_snap, probe_results, cex, last_error,
                                                          batches if full_ctx else None)}]
        E("status", text=f"Turn {turn}: Muse is thinking")
        try:
            out = brain_client.turn(msgs, action_names=spec.action_names)
        except BrainError as e:
            last_error = f"brain call failed: {e}"
            E("error", message=last_error[:300])
            continue
        last_error = ""
        E("status", text=out["summary"][:200])

        ledger.apply(out.get("ledger") or [], version + (1 if out.get("program") else 0))
        E("ledger", entries=ledger.snapshot())
        store.save_ledger(run_id, ledger.snapshot())

        # program --------------------------------------------------------------
        if out.get("program"):
            new_code, init = out["program"]["code"], out["program"]["params"]
            try:
                new_fn = numerics.load_program(new_code, spec)
            except Exception as e:  # noqa: BLE001 - ProgramError text goes back to the brain verbatim
                last_error = f"program rejected: {e}"
                E("status", text=last_error[:200])
            else:
                E("status", text=f"Fitting {len(init)} parameters on {len(X):,} labeled states")
                try:
                    new_params, _ = numerics.fit_params(new_fn, init, X, y, spec, max_seconds=fit_seconds)
                    new_agr = numerics.agreement(new_fn, new_params, HX, Hy, spec)
                except Exception as e:  # noqa: BLE001
                    last_error = f"program failed while fitting: {type(e).__name__}: {e}"
                    E("status", text=last_error[:200])
                else:
                    version += 1
                    changed = _changed_lines(code, new_code)
                    code, fn, params, agreement = new_code, new_fn, new_params, new_agr
                    E("program", version=version, code=code, params=params, changed_lines=changed,
                      agreement=agreement)
                    E("metric", queries=bb.query_count(), agreement=agreement)
                    store.save_program(run_id, {"version": version, "code": code, "params": params,
                                                "agreement": agreement})
                    if best is None or agreement >= best[0]:
                        best = (agreement, version, code, params, fn)
                    if agreement > best_frames_agr:
                        best_frames_agr = agreement
                        try:
                            clip = render_pair(bb, env_factory, fn, params, demo_seed)
                            E("frames", version=version, **clip)
                        except Exception as e:  # noqa: BLE001
                            E("status", text=f"clip skipped ({type(e).__name__})")

        # probes ---------------------------------------------------------------
        probe_results = []
        for p in out.get("probes") or []:
            try:
                # `cex` = last turn's counterexamples, the ones Muse just saw: sweep from those states
                n_pts = len(probes_mod.expand(p, spec, X, cex))
            except probes_mod.ProbeError as e:
                probe_results.append({"rule_id": p.get("rule_id") or "new", "description": "invalid probe",
                                      "n_in_region": 0, "fraction": 0.0, "status": "skipped",
                                      "query_id": "-", "note": str(e)})
                continue
            if n_pts > remaining():
                E("status", text="Budget too low for the next probe")
                break
            rid = ledger.ensure_rule(p.get("rule_id"), probes_mod.rule_text(p, spec))
            E("status", text=f"Probing {rid}: {probes_mod.describe(p)}"[:200])
            res = probes_mod.run_probe(p, rid, bb, spec, X, cex)
            b = res.pop("batch")
            store.save_queries(run_id, b)
            batches.append(b)
            X += b.obs
            y += b.actions
            rows += [(b.query_id, o, a, "probe") for o, a in zip(b.obs, b.actions)]
            conf = res["fraction"] if res["n_in_region"] else ledger.get(rid).confidence
            ledger.verdict(rid, res["status"], conf, b.query_id, extra_evidence=res["base_evidence"])
            probe_results.append(res)
        if probe_results:
            E("ledger", entries=ledger.snapshot())
            store.save_ledger(run_id, ledger.snapshot())
            E("query_count", n=bb.query_count())

        # prober ---------------------------------------------------------------
        cex = []
        if fn is not None and remaining() > 0 and not over_time():
            E("status", text="Hunting for states where the clone disagrees with the robot")
            try:
                cex = numerics.find_disagreements(fn, params, bb, seed_obs, spec, k=8,
                                                  max_queries=min(prober_max, remaining()))
            except Exception as e:  # noqa: BLE001
                E("status", text=f"prober failed ({type(e).__name__}: {e})"[:200])
            for c in cex:
                X.append(c.obs)
                y.append(c.bb_action)
                rows.append((c.query_id, c.obs, c.bb_action, "prober"))
            E("query_count", n=bb.query_count())

        if turn % 2 == 0:
            hints = numerics.pattern_hints(X, y, spec)

        # stop? ----------------------------------------------------------------
        if agreement is not None:
            history.append(agreement)
        prober_quiet = fn is not None and len(cex) < 2
        if agreement is not None and agreement >= TARGET_AGREEMENT and prober_quiet:
            E("status", text=f"Target reached: {agreement:.1%} holdout match and the prober found nothing new")
            break
        if len(history) >= 3 and max(history[-2:]) < max(history[:-2]) + 0.01 and prober_quiet:
            E("status", text="Agreement stopped improving")
            break
        if out.get("done") and turn > 1:
            E("status", text="Muse says it is done")
            break
        if remaining() <= 0:
            E("status", text="Query budget spent")
            break

    # 4. final evaluation --------------------------------------------------------
    if live and timeout_s is not None and (stopped_for_time or over_time()) and (
            best is None or best[0] < TARGET_AGREEMENT):
        # the live time limit stopped the run before the copy converged: let the app switch to
        # the cached run of this exact configuration (it says so on screen)
        raise TimeoutError("live time limit reached before the copy converged")
    if best is None:
        fn_final = numerics.load_program(FALLBACK_CODE, spec)
        best = (0.0, 0, FALLBACK_CODE, {}, fn_final)
        E("error", message="no valid program was produced; evaluating a constant fallback")
    agr_b, ver_b, code_b, params_b, fn_b = best
    E("status", text=f"Evaluating program v{ver_b}")
    try:
        report = numerics.evaluate(fn_b, params_b, bb, spec, holdout, env_factory,
                                   EVAL_SEEDS[:5] if live else EVAL_SEEDS)
    except Exception as e:  # noqa: BLE001
        E("error", message=f"evaluate failed: {type(e).__name__}: {e}"[:300])
        report = EvalReport(open_loop_agreement=agr_b, n_holdout_obs=len(HX))
    E("run_finished", eval=dataclasses.asdict(report), total_queries=bb.query_count(), turns=turns_done,
      seconds=round(time.time() - t0, 1), cached=False)
    store.save_eval(run_id, dataclasses.asdict(report))
    return {"program": Program(version=ver_b, code=code_b, params=params_b, agreement=agr_b),
            "ledger": ledger.entries_list(), "eval": report, "events_path": store.events_path(run_id)}


def _turn_context(spec, queries_used, budget, turn, max_turns, version, agreement, params, code, ledger,
                  n_labeled, hints, scan, use_snap, probe_results, cex, error, all_batches) -> str:
    if scan:
        a = scan["anomaly_feature"]
        p = (a.get("probabilities") or {}).get(a.get("choice"), 0.0)
        subq = prompts.SUBQ_LINE.format(
            n_rows=f"{scan['n_rows']:,}",
            relevance={k: round(v, 2) for k, v in scan["feature_relevance"].items()},
            anomaly_feature=a.get("choice"), p=f"{float(p):.2f}",
            next_probe=scan["next_probe"].get("choice"),
            rule_support={k: round(v, 2) for k, v in scan["rule_support"].items()} or "(no rules yet)")
    elif use_snap:
        subq = "SUBQ EVIDENCE SCAN: unavailable this turn"
    else:
        subq = ""
    ctx = prompts.TURN.format(
        spec=prompts.spec_text(spec), queries_used=queries_used, budget=budget, turn=turn, max_turns=max_turns,
        version=version, agreement="n/a" if agreement is None else f"{agreement:.1%}",
        params={k: round(v, 3) for k, v in (params or {}).items()} or "{}",
        code=code or "(none yet: write the first version)",
        ledger_table=prompts.ledger_table(ledger.snapshot()), n=f"{n_labeled:,}", hints="\n" + hints,
        subq_line=subq, probe_table=prompts.probe_table(probe_results),
        counterexample_table=prompts.counterexample_table(cex, spec), error=error or "none")
    if all_batches:
        lines = ["query_id,source," + ",".join(spec.obs_names) + ",action"]
        for b in all_batches:
            for o, a in zip(b.obs, b.actions):
                lines.append(f"{b.query_id},{b.source}," + ",".join(f"{v:.3f}" for v in o) + f",{a}")
        ctx += "\nALL QUERIES (CSV):\n" + "\n".join(lines)
    return ctx


# ------------------------------------------------------------------ CLI
def _print_event(ev: dict) -> None:
    t, d = ev["t"], ev["data"]
    kind = ev["type"]
    if kind == "status":
        print(f"[{t:6.1f}s] {d['text']}")
    elif kind == "program":
        print(f"[{t:6.1f}s] PROGRAM v{d['version']}: holdout agreement {d['agreement']:.1%}  params {d['params']}")
    elif kind == "ledger":
        print(f"[{t:6.1f}s] LEDGER " + "  ".join(f"{e['rule_id']}:{e['status']}" for e in d["entries"]))
    elif kind == "snap":
        a = d["anomaly_feature"]
        print(f"[{t:6.1f}s] SNAP read {d['n_rows']:,} rows ({d['input_tokens']} tokens): anomaly "
              f"{a['choice']} (p={(a['probabilities'] or {}).get(a['choice'], 0):.2f}), next probe "
              f"{d['next_probe']['choice']}")
    elif kind == "frames":
        print(f"[{t:6.1f}s] FRAMES v{d['version']}: {len(d['bb'])} bb / {len(d['clone'])} clone frames, "
              f"{sum(d['disagree'])} disagree")
    elif kind == "error":
        print(f"[{t:6.1f}s] ERROR {d['message']}")
    elif kind == "run_finished":
        print(f"[{t:6.1f}s] FINISHED in {d['seconds']}s, {d['turns']} turns, {d['total_queries']} queries; "
              f"eval {json.dumps(d['eval'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Unbox headless.")
    ap.add_argument("--bb", default="mock", help="mock, or a black-box id such as bb-01")
    ap.add_argument("--env", default=None, help="gymnasium env id for the clone (real black box only)")
    ap.add_argument("--brain", default=os.getenv("BRAIN", "muse"))
    ap.add_argument("--twist", default="t-cp-1", help="mock only: t-cp-1 or none. Real twists are set in the app.")
    ap.add_argument("--budget", type=int, default=int(os.getenv("QUERY_BUDGET", "5000")))
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--demo-seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    use_mocks = args.bb == "mock" or os.getenv("USE_MOCKS", "1") == "1"
    if use_mocks:
        from contracts.mocks import MockBlackBox, mock_env_factory, numerics_stub

        bb = MockBlackBox(twist=args.twist != "none")
        numerics, env_factory, label = numerics_stub, mock_env_factory, "Mock CartPole"
    else:
        import gymnasium as gym

        import numerics
        from blackbox.client import HttpBlackBox

        if not args.env:
            ap.error("--env is required with a real black box (e.g. --env CartPole-v1)")
        if args.twist:
            print("note: the agent cannot set twists; set the active flaw in the app before running")

        def env_factory(seed: int):
            env = gym.make(args.env, render_mode="rgb_array")
            obs, _ = env.reset(seed=seed)
            return env, [float(v) for v in obs]

        bb = HttpBlackBox(args.bb, os.getenv("BLACKBOX_URL", "http://localhost:8001"))
        label = args.bb
    from store.store import get_store

    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{args.bb}-{uuid.uuid4().hex[:4]}"
    print(f"run {run_id}: bb={args.bb} brain={args.brain} mocks={use_mocks} SNAP={os.getenv('SNAP', '1')}")
    out = run(bb, args.brain, env_factory=env_factory, numerics=numerics,
              emit=None if args.quiet else _print_event, store=get_store(), run_id=run_id,
              budget=args.budget, max_turns=args.max_turns, live=args.live,
              timeout_s=float(os.getenv("LIVE_TIMEOUT_S", "75")) if args.live else None,
              policy_label=label, demo_seed=args.demo_seed)
    prog = out["program"]
    print("\n=== FINAL PROGRAM v%d (holdout agreement %.1f%%) ===" % (prog.version, 100 * (prog.agreement or 0)))
    print(prog.code)
    print("params:", {k: round(v, 4) for k, v in prog.params.items()})
    print("\n=== LEDGER ===")
    for e in out["ledger"]:
        print(f"  {e.rule_id:<4} {e.status:<10} {e.confidence:.2f}  {e.text}")
    print("\nevents:", out["events_path"])


if __name__ == "__main__":
    main()
