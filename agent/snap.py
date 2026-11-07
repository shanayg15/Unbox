"""Lane B, task B2b: SubQ Snap evidence scan (Decisions API, NOT a chat model).

Every turn, Snap reads the *entire raw query log* and returns calibrated probabilities: which features matter,
which feature separates the counterexamples, what to probe next, and how well each ledger rule fits.
Advisory only (rule 4): confirmed/refuted still come only from Python probe verdicts.

    python -m agent.snap --ping
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

from agent.calllog import log_call

load_dotenv()

TIMEOUT_S = 20.0
MAX_STATE_CHARS = 1_200_000          # ≈ 300K tokens, under Snap's 384K window
MAX_QUESTIONS = 128


class SnapError(RuntimeError):
    pass


def _cfg() -> tuple[str, str, str]:
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        raise SnapError("SUBQ_API_KEY is not set in .env")
    base = (os.getenv("SUBQ_BASE_URL") or "https://api.subq.ai/v1").rstrip("/")
    return base, key, os.getenv("SUBQ_MODEL") or "subq-snap-1.1"


def decide(state: str, questions: dict, *, run_id: str = "adhoc", store=None, timeout: float = TIMEOUT_S) -> dict:
    """POST /systemone. Returns the raw response JSON. Logs every call (provider='subq')."""
    import httpx

    base, key, model = _cfg()
    t0 = time.time()
    rec = {"provider": "subq", "model": model, "n_questions": len(questions), "state_chars": len(state)}
    try:
        r = httpx.post(f"{base}/systemone", json={"model": model, "state": state, "questions": questions},
                       headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        latency = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            detail = r.text[:300]
            log_call(run_id, {**rec, "latency_ms": latency, "input_tokens": None, "ok": False,
                              "error": f"HTTP {r.status_code}: {detail}"}, store)
            raise SnapError(f"HTTP {r.status_code}: {detail}")
        body = r.json()
        tokens = (body.get("usage") or {}).get("input_tokens")
        log_call(run_id, {**rec, "latency_ms": latency, "input_tokens": tokens, "prompt_tokens": tokens,
                          "completion_tokens": 0, "ok": True, "error": None}, store)
        return body
    except SnapError:
        raise
    except Exception as e:  # noqa: BLE001
        log_call(run_id, {**rec, "latency_ms": int((time.time() - t0) * 1000), "input_tokens": None,
                          "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}, store)
        raise SnapError(f"{type(e).__name__}: {e}") from e


# ------------------------------------------------------------------ state + questions
def _fmt_row(qid: str, obs, action, spec) -> str:
    feats = " ".join(f"{n}={float(v):.3f}" for n, v in zip(spec.obs_names, obs))
    if spec.action_type == "discrete":
        act = spec.action_names[int(action)]
    else:
        act = "[" + ", ".join(f"{float(x):.3f}" for x in action) + "]"
    return f"{qid} | {feats} -> {act}"


def build_state(spec, rows, counterexamples, ledger) -> tuple[str, int]:
    """rows: list of (query_id, obs, action, source). Returns (state text, n_rows actually sent).

    If the text is over MAX_STATE_CHARS, the oldest rollout rows are dropped first."""
    head = ["SENSOR SPEC: " + ", ".join(f"{n} in [{lo:.3f}, {hi:.3f}]" for n, lo, hi in
                                        zip(spec.obs_names, spec.obs_low, spec.obs_high)) +
            ("; actions: " + ", ".join(spec.action_names)),
            "QUERY LOG (every state the robot was asked about, with its action):"]
    cex_lines = ["COUNTEREXAMPLES (the current clone program disagrees with the robot here):"]
    for i, c in enumerate(counterexamples, 1):
        clone = "none" if c.clone_action is None else (
            spec.action_names[int(c.clone_action)] if spec.action_type == "discrete" else str(c.clone_action))
        cex_lines.append(f"[C{i}] " + _fmt_row(c.query_id, c.obs, c.bb_action, spec) + f" (clone said {clone})")
    if not counterexamples:
        cex_lines.append("(none yet)")
    rule_lines = ["ACTIVE RULES (hypotheses about the robot):"]
    active = [e for e in ledger if _get(e, "status") != "refuted"]
    for e in active:
        rule_lines.append(f"[{_get(e, 'rule_id')}] {_get(e, 'text')}")
    if not active:
        rule_lines.append("(none yet)")

    row_lines = [(src, _fmt_row(q, o, a, spec)) for q, o, a, src in rows]
    fixed = sum(len(s) + 1 for s in head + cex_lines + rule_lines)
    total = fixed + sum(len(s) + 1 for _, s in row_lines)
    if total > MAX_STATE_CHARS:                        # drop the oldest rollout rows first
        keep = []
        drop = total - MAX_STATE_CHARS
        for src, s in row_lines:
            if drop > 0 and src == "rollout":
                drop -= len(s) + 1
                continue
            keep.append((src, s))
        row_lines = keep
        while row_lines and fixed + sum(len(s) + 1 for _, s in row_lines) > MAX_STATE_CHARS:
            row_lines.pop(0)
    text = "\n".join(head + [s for _, s in row_lines] + cex_lines + rule_lines)
    return text, len(row_lines)


def _get(e, k):
    return e[k] if isinstance(e, dict) else getattr(e, k)


def build_questions(spec, ledger) -> dict:
    feats = list(spec.obs_names)
    q: dict = {}
    for f in feats:
        q[f"rel_{f}"] = {"type": "noul", "instructions":
                         f"Across all rows, does the robot's action depend on {f}, i.e. do states that differ "
                         f"mainly in {f} get different actions?"}
    q["anomaly_feature"] = {"type": "choice", "instructions":
                            "Which feature best separates the counterexamples [C*] from the other rows?",
                            "criteria": {**{f: None for f in feats},
                                         "none": "There are no counterexamples, or no single feature separates them"}}
    q["next_probe"] = {"type": "choice", "instructions":
                       "Probing which feature next is most likely to reveal behavior the current rules do not explain?",
                       "criteria": {f: None for f in feats}}
    for e in ledger:
        if _get(e, "status") == "refuted" or len(q) >= MAX_QUESTIONS:
            continue
        rid = _get(e, "rule_id")
        q[f"rule_{rid}"] = {"type": "noul", "instructions":
                            f"Is this rule consistent with all rows: {_get(e, 'text')}?"}
    return q


def _choice(ans: dict | None) -> dict:
    ans = ans or {}
    return {"choice": ans.get("choice"), "probabilities": ans.get("probabilities", {}),
            "confidence": ans.get("confidence")}


def scan(spec, rows, counterexamples, ledger, run_id: str, *, store=None, timeout: float = TIMEOUT_S) -> dict:
    """-> {feature_relevance, anomaly_feature, next_probe, rule_support, n_rows, input_tokens, seconds}.
    Raises SnapError on any failure; the loop catches it, emits status "SubQ scan skipped", and continues."""
    t0 = time.time()
    state, n_rows = build_state(spec, rows, counterexamples, ledger)
    questions = build_questions(spec, ledger)
    body = decide(state, questions, run_id=run_id, store=store, timeout=timeout)
    ans = body.get("answers") or {}
    rel = {f: float((ans.get(f"rel_{f}") or {}).get("noul", 0.0)) for f in spec.obs_names}
    rules = {k[len("rule_"):]: float(v.get("noul", 0.0)) for k, v in ans.items() if k.startswith("rule_")}
    return {
        "feature_relevance": rel,
        "anomaly_feature": _choice(ans.get("anomaly_feature")),
        "next_probe": _choice(ans.get("next_probe")),
        "rule_support": rules,
        "n_rows": n_rows,
        "input_tokens": (body.get("usage") or {}).get("input_tokens"),
        "seconds": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    ap.parse_args()
    state = ("q1 | speed=0.10 -> brake_off\nq2 | speed=0.95 -> brake_on\nq3 | speed=0.20 -> brake_off\n"
             "q4 | speed=0.90 -> brake_on\nq5 | temp=0.9 speed=0.15 -> brake_off")
    questions = {
        "depends_on_speed": {"type": "noul",
                             "instructions": "Does the action depend on speed (fast rows brake, slow rows do not)?"},
        "key_feature": {"type": "choice", "instructions": "Which feature decides the action?",
                        "criteria": {"speed": None, "temp": None, "none": "No feature decides it"}},
    }
    t0 = time.time()
    body = decide(state, questions, run_id="ping")
    print(json.dumps(body, indent=2))
    print(f"ok in {time.time() - t0:.1f}s; logged to runs/ping/brain_calls.jsonl")


if __name__ == "__main__":
    main()
