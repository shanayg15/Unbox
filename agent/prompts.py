"""Lane B, task B3: prompts. Rule 2: the brain only ever sees the sensor spec, evidence, the current program,
the ledger and the budget. Never an env id, policy label or twist.
"""
from __future__ import annotations

import json

SYSTEM = """You are Unbox, a reverse-engineering agent. A robot controller (a neural network) is hidden behind an API.
You only see what it does. Write a short, readable Python program that makes the same decisions, and keep
a ledger of the rules you believe it follows, each backed by evidence.

Respond with ONLY one JSON object matching this schema: {schema}

Program rules:
- def policy(obs, P): obs is a list of floats in sensor-spec order. Return an int action id (discrete) or a
  list of floats (continuous).
- Start by naming features: x = obs[0], etc. Then if/elif/else rules. Arithmetic, comparisons, math.*, np.* only.
  No loops, no imports, nothing starting with "_". At most 40 lines.
- Every threshold or gain is a parameter in P with an initial guess. Never hard-code a number you guessed.
  An optimizer fits P to the data after you answer.
- Prefer a few rules that explain most decisions. Add a rule only when counterexamples demand it.

Ledger rules:
- One entry per rule, plain English, citing query ids as evidence. Ledger rule text: 12 words max.
- Status: observed (directly visible in data), inferred (your hypothesis), probing (you are testing it now),
  refuted (evidence contradicts it). You can never mark a rule confirmed; the system does that from probes.

Probes (max 3 per turn): design experiments that could REFUTE your hypothesis, not just support it.
If a rule only applies in a specific situation (e.g. while firing an engine), set base to a counterexample state, not the median.
Set done=true when agreement stops improving."""

TURN = """SENSOR SPEC: {spec}
BUDGET: {queries_used}/{budget} queries, turn {turn}/{max_turns}
CURRENT PROGRAM (v{version}, holdout agreement {agreement}, fitted params {params}):
{code}
LEDGER: {ledger_table}
PATTERN HINTS (decision tree on {n} labeled states): {hints}
{subq_line}
PROBE RESULTS since last turn: {probe_table}
COUNTEREXAMPLES (clone disagrees with robot): {counterexample_table}
LAST ERROR (if any): {error}"""

SUBQ_LINE = ("SUBQ EVIDENCE SCAN (SubQ Snap read all {n_rows} raw rows): feature relevance {relevance}; "
             "counterexamples best separated by {anomaly_feature} (p={p}); suggested next probe: {next_probe}; "
             "rule support {rule_support}")

FIX = """Here is the recovered program and ledger. Engineers report this unwanted behavior: "{reveal_label}".
Return JSON {{"program": {{"code", "params"}}, "changed_rule_ids": [...], "explanation": "..."}}.
Change as few lines as possible. Every other behavior must stay identical."""

ZERO_QUERY = """Task: {task_name}
SENSOR SPEC: {spec}
You get no data from the robot. Write a controller that performs this task well.
Return ONLY JSON {{"program": {{"code": "...", "params": {{...}}}}}}.
Program rules:
- def policy(obs, P): obs is a list of floats in sensor-spec order. Return an int action id (discrete) or a
  list of floats (continuous).
- Start by naming features: x = obs[0], etc. Then if/elif/else rules. Arithmetic, comparisons, math.*, np.* only.
  No loops, no imports, nothing starting with "_". At most 40 lines.
- Every threshold or gain is a parameter in P with an initial guess."""


def brain_schema() -> str:
    from agent.protocol import BrainTurn

    return json.dumps(BrainTurn.model_json_schema(), separators=(",", ":"))


def system_prompt() -> str:
    return SYSTEM.format(schema=brain_schema()) + "\n\n" + BREVITY


# ------------------------------------------------------------------ compact formatting helpers
def r3(v) -> str:
    return f"{float(v):.3f}"


def spec_text(spec) -> str:
    feats = ", ".join(f"obs[{i}] {n} in [{r3(lo)}, {r3(hi)}]"
                      for i, (n, lo, hi) in enumerate(zip(spec.obs_names, spec.obs_low, spec.obs_high)))
    if spec.action_type == "discrete":
        acts = ", ".join(f"{i}={n}" for i, n in enumerate(spec.action_names))
        return f"features: {feats}. discrete actions: {acts}"
    return f"features: {feats}. continuous actions: {spec.action_names} in [{spec.action_low}, {spec.action_high}]"


def action_text(a, spec) -> str:
    if spec.action_type == "discrete" and a is not None:
        return f"{int(a)}:{spec.action_names[int(a)]}"
    return "None" if a is None else "[" + ", ".join(r3(x) for x in a) + "]"


def ledger_table(entries) -> str:
    if not entries:
        return "(empty)"
    rows = ["id  | status    | conf  | rule | evidence"]
    for e in entries:
        d = e if isinstance(e, dict) else e.__dict__
        rows.append(f"{d['rule_id']:<3} | {d['status']:<9} | {float(d['confidence']):.2f}  | {d['text']} | "
                    f"{','.join(d.get('evidence', [])[-4:])}")
    return "\n" + "\n".join(rows)


def counterexample_table(cex, spec, cap: int = 12) -> str:
    if not cex:
        return "(none found)"
    names = spec.obs_names
    rows = [" ".join(f"{n:>10}" for n in names) + " | robot      | clone      | query"]
    for c in cex[:cap]:
        rows.append(" ".join(f"{float(v):>10.3f}" for v in c.obs) +
                    f" | {action_text(c.bb_action, spec):<10} | {action_text(c.clone_action, spec):<10} | {c.query_id}")
    return "\n" + "\n".join(rows)


def probe_table(results, cap: int = 20) -> str:
    if not results:
        return "(none)"
    rows = []
    for r in results[:cap]:
        rows.append(f"{r['rule_id']}: {r['description']} -> {r['n_in_region']} points in region, "
                    f"{r['fraction']:.0%} matched expected action -> {r['status']} ({r['query_id']})"
                    + (f"; {r['note']}" if r.get("note") else ""))
    return "\n" + "\n".join(rows)

BREVITY = ("Keep your reasoning short: a few sentences of thinking, then output the JSON object. "
           "Long deliberation gets cut off and wastes the turn.")
