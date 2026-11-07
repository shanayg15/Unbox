"""Lane B, task B3: ledger. Python assigns rule ids, and only Python verdicts (agent/probes.py) set
confirmed / refuted (rule 4). The brain can propose, revise and refute rules, never confirm them."""
from __future__ import annotations

import dataclasses
import re

from contracts.types import LedgerEntry

_RID = re.compile(r"r(\d+)")


class Ledger:
    def __init__(self) -> None:
        self.entries: dict[str, LedgerEntry] = {}
        self._next = 1
        self._python_verdict: set[str] = set()        # rule ids whose status came from a probe verdict

    def _new_id(self, wanted: str | None = None) -> str:
        m = _RID.fullmatch(wanted or "")
        if m and wanted not in self.entries and int(m[1]) >= self._next:
            self._next = int(m[1]) + 1                 # keep the brain's own numbering when it is new and in order
            return wanted
        rid = f"r{self._next}"
        self._next += 1
        return rid

    def apply(self, updates: list[dict], program_version: int | None) -> dict[str, str]:
        """Apply brain ledger updates. Returns {brain_rule_id_or_text: python_rule_id} for this turn."""
        mapping: dict[str, str] = {}
        for u in updates:
            status = u.get("status", "inferred")
            if status == "confirmed":                  # only probes confirm
                status = "inferred"
            rid = u.get("rule_id")
            evidence = [str(x) for x in u.get("evidence") or []]
            if rid and rid in self.entries:
                e = self.entries[rid]
                text_changed = u.get("text", e.text).strip() != e.text.strip()
                if rid in self._python_verdict and not text_changed and status != "refuted":
                    status = e.status                  # keep the Python verdict for an unchanged rule
                elif text_changed:
                    self._python_verdict.discard(rid)
                e.text = u.get("text", e.text)
                e.status = status
                if rid not in self._python_verdict:
                    e.confidence = float(u.get("confidence", e.confidence))
                e.evidence = list(dict.fromkeys(e.evidence + evidence))
                e.program_version = program_version
            else:
                new = self._new_id(rid)
                self.entries[new] = LedgerEntry(rule_id=new, text=u.get("text", ""), status=status,
                                                confidence=float(u.get("confidence", 0.5)),
                                                evidence=evidence, program_version=program_version)
                rid = rid or new
                mapping[rid] = new
            mapping.setdefault(rid, rid)
            mapping[u.get("text", "")] = mapping[rid]
        return mapping

    def ensure_rule(self, rule_id: str | None, text: str) -> str:
        """Rule a probe tests: an existing rule, or a new 'probing' entry described by the probe."""
        if rule_id and rule_id in self.entries:
            return rule_id
        new = self._new_id(rule_id)
        self.entries[new] = LedgerEntry(rule_id=new, text=text, status="probing", confidence=0.5)
        return new

    def verdict(self, rule_id: str, status: str, confidence: float, query_id: str,
                extra_evidence: list[str] | tuple = ()) -> None:
        """Python probe verdict. `extra_evidence`: e.g. the counterexample states a probe started from."""
        e = self.entries[rule_id]
        e.status = status
        e.confidence = round(float(confidence), 3)
        for qid in [query_id, *extra_evidence]:
            if qid and qid not in e.evidence:
                e.evidence.append(qid)
        if status in ("confirmed", "refuted"):
            self._python_verdict.add(rule_id)

    def get(self, rule_id: str) -> LedgerEntry | None:
        return self.entries.get(rule_id)

    def snapshot(self) -> list[dict]:
        return [dataclasses.asdict(e) for e in self.entries.values()]

    def entries_list(self) -> list[LedgerEntry]:
        return list(self.entries.values())
