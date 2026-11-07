"""Session owner: B-Brain. FIX prompt wrapper: after the reveal, Muse changes as few lines as possible."""
from __future__ import annotations

import difflib

from agent import prompts
from agent.brain import Brain
from agent.protocol import FixOut

FIX_SYSTEM = """You are Unbox. You recovered a robot controller as a short Python program. Respond with ONLY one
JSON object. Keep the program rules: def policy(obs, P), no loops, no imports, nothing starting with "_",
every threshold or gain a named parameter in P."""

FIX_GUIDANCE = """How to fix it:
- The unwanted behavior is a hidden special case. Find the one rule (usually an override branch that applies only
  in some region of the inputs) that produces it, and remove or neutralize just that branch.
- Do NOT retune the general control law. Keep every other parameter at exactly the value given above.
- changed_rule_ids lists only the ledger rules your edit removes or changes."""


def _ledger_with_highlights(ledger: list[dict]) -> str:
    rows = [dict(e, text=("[FLAGGED BY ENGINEERS] " if e.get("highlight") else "") + e["text"]) for e in ledger]
    return prompts.ledger_table(rows)


def _diff(old_code: str, new_code: str, old_params: dict, new_params: dict) -> str:
    out = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines(),
                                    "recovered", "fixed", lineterm=""))
    p_lines = []
    for k in sorted(set(old_params) | set(new_params)):
        a, b = old_params.get(k), new_params.get(k)
        if a is None:
            p_lines.append(f"+P[{k!r}] = {b:.4g}")
        elif b is None or k not in new_params:
            p_lines.append(f"-P[{k!r}] = {a:.4g}")
        elif abs(a - b) > 1e-9:
            p_lines.append(f" P[{k!r}]: {a:.4g} -> {b:.4g}")
    if p_lines:
        out += ["@@ params @@"] + p_lines
    return "\n".join(out)


def fix(program: dict, ledger: list[dict], reveal_label: str, brain: str, run_id: str, store=None) -> dict:
    """FIX prompt → {"code", "params", "diff", "changed_rule_ids", "explanation"}.

    Ledger entries with "highlight": True (rules the app matched to the reveal) are flagged to the brain."""
    old_code = program["code"]
    old_params = dict(program.get("params") or {})
    user = (prompts.FIX.format(reveal_label=reveal_label) +
            f"\n\nRECOVERED PROGRAM (fitted params {old_params}):\n{old_code}\n"
            f"LEDGER: {_ledger_with_highlights(ledger)}\n\n{FIX_GUIDANCE}")
    b = Brain(brain, run_id=run_id, store=store)
    out = b.complete_json([{"role": "system", "content": FIX_SYSTEM}, {"role": "user", "content": user}], FixOut)
    new_code = out["program"]["code"]
    used = set(__import__("re").findall(r"P\[\s*['\"]([^'\"]+)['\"]\s*\]", new_code))
    params = {k: v for k, v in {**old_params, **out["program"]["params"]}.items() if k in used or not used}
    return {"code": new_code, "params": params, "diff": _diff(old_code, new_code, old_params, params),
            "changed_rule_ids": out.get("changed_rule_ids", []), "explanation": out.get("explanation", "")}
