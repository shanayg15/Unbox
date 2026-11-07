"""Lane B, task B3: brain turn protocol (pydantic). The brain's JSON reply must validate as BrainTurn."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Sweep(BaseModel):
    feature: str
    start: float
    stop: float
    n: int = Field(default=20, ge=2, le=60)


class Expect(BaseModel):
    feature: str
    op: Literal[">", "<"]
    value: float
    action: int | list[float]            # predicted black-box action inside this region


class ProbeSpec(BaseModel):
    rule_id: str | None = None           # None = new rule
    base: Literal["median"] | list[float] = "median"
    sweep: Sweep
    sweep2: Sweep | None = None          # 2-D grid; n ≤ 15 each
    expect: Expect


class LedgerUpdate(BaseModel):
    rule_id: str | None = None           # None = new rule (Python assigns r1, r2...)
    text: str
    status: Literal["observed", "inferred", "probing", "refuted"]   # never "confirmed"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = []


class ProgramOut(BaseModel):
    code: str
    params: dict[str, float]


class BrainTurn(BaseModel):
    summary: str                         # one sentence, shown as the status line
    ledger: list[LedgerUpdate] = []
    program: ProgramOut | None = None
    probes: list[ProbeSpec] = Field(default=[], max_length=3)
    done: bool = False


class FixOut(BaseModel):
    program: ProgramOut
    changed_rule_ids: list[str] = []
    explanation: str = ""


def normalize_turn(obj: dict, action_names: list[str] | None = None) -> dict:
    """Forgive small, unambiguous slips before validation (the rules themselves stay strict).

    - status "confirmed" -> "inferred" (only agent/probes.py may confirm)
    - more than 3 probes -> first 3
    - action given by name ("push_left") -> its id
    - confidence given as a percentage (85) -> 0.85
    """
    if not isinstance(obj, dict):
        return obj
    for e in obj.get("ledger") or []:
        if isinstance(e, dict):
            if e.get("status") == "confirmed":
                e["status"] = "inferred"
            c = e.get("confidence")
            if isinstance(c, (int, float)) and 1 < c <= 100:
                e["confidence"] = c / 100
            ev = e.get("evidence")
            if isinstance(ev, str):
                e["evidence"] = [ev]
            elif isinstance(ev, list):
                e["evidence"] = [str(x) for x in ev]
    probes = obj.get("probes")
    if isinstance(probes, list):
        obj["probes"] = probes[:3]
        for p in obj["probes"]:
            exp = p.get("expect") if isinstance(p, dict) else None
            if isinstance(exp, dict) and isinstance(exp.get("action"), str) and action_names:
                a = exp["action"].strip()
                if a in action_names:
                    exp["action"] = action_names.index(a)
                elif a.isdigit():
                    exp["action"] = int(a)
    prog = obj.get("program")
    if isinstance(prog, dict) and isinstance(prog.get("params"), dict):
        prog["params"] = {k: float(v) for k, v in prog["params"].items() if isinstance(v, (int, float))}
    return obj
