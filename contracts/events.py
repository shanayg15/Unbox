# Envelope: {"type": str, "run_id": str, "t": float (seconds since run start), "data": {...}}

EVENT_TYPES = {
    "run_started":  "{policy_label, brain, model, budget, live: bool}",
    "status":       "{text}                                  # one-line narration, e.g. 'Probing r3: sweep x 0.3→0.8'",
    "query_count":  "{n}",
    "ledger":       "{entries: [LedgerEntry as dict]}        # full snapshot every time",
    "program":      "{version, code, params, changed_lines: [int], agreement}",
    "metric":       "{queries, agreement}                    # one point on the match-vs-queries curve",
    "frames":       "{version, seed, fps, bb: [b64], clone: [b64], disagree: [bool]}  # whole clip",
    "reveal":       "{label, feature, highlight_rule_ids: [str]}",
    "fix":          "{code, params, diff: str}",
    "shield":       "{on: bool, bb: [b64] | null}            # shielded black-box clip when on",
    "run_finished": "{eval: EvalReport as dict, total_queries, turns, seconds, cached: bool}",
    "error":        "{message}",
}
