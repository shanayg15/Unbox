"""Lane B, task B4: store (JSON and Mongo backends, same interface).

JsonStore writes runs/<run_id>/*.jsonl and always works offline.
MongoStore writes to Atlas *and* JSON (so runs replay even if Atlas is down). get_store() picks one from STORE
and falls back to JSON automatically if the Atlas ping fails. The database never hangs a live run.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def to_jsonable(obj: Any) -> Any:
    """Dataclasses / pydantic models / numpy scalars -> plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):      # numpy scalar
        return obj.item()
    return obj


class Store:
    def save_event(self, run_id: str, event: dict) -> None:
        raise NotImplementedError

    def save_brain_call(self, run_id: str, record: dict) -> None:
        raise NotImplementedError

    def save_program(self, run_id: str, program: dict) -> None:
        raise NotImplementedError

    def save_ledger(self, run_id: str, entries: list) -> None:
        raise NotImplementedError

    def save_eval(self, run_id: str, report: dict) -> None:
        raise NotImplementedError

    def save_queries(self, run_id: str, batch: dict) -> None:
        raise NotImplementedError

    def list_runs(self) -> list:
        raise NotImplementedError

    def load_events(self, run_id: str) -> list:
        raise NotImplementedError

    def events_path(self, run_id: str) -> str:
        return str(RUNS_DIR / run_id / "events.jsonl")


class JsonStore(Store):
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else RUNS_DIR
        self._lock = threading.Lock()

    def _dir(self, run_id: str) -> Path:
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append(self, run_id: str, name: str, obj: Any) -> None:
        line = json.dumps(to_jsonable(obj), separators=(",", ":"))
        with self._lock, open(self._dir(run_id) / name, "a") as f:
            f.write(line + "\n")

    def save_event(self, run_id: str, event: dict) -> None:
        self._append(run_id, "events.jsonl", event)

    def save_brain_call(self, run_id: str, record: dict) -> None:
        self._append(run_id, "brain_calls.jsonl", record)

    def save_program(self, run_id: str, program: dict) -> None:
        self._append(run_id, "programs.jsonl", program)

    def save_ledger(self, run_id: str, entries: list) -> None:
        self._append(run_id, "ledger.jsonl", {"t": time.time(), "entries": entries})

    def save_eval(self, run_id: str, report: dict) -> None:
        with self._lock:
            (self._dir(run_id) / "eval.json").write_text(json.dumps(to_jsonable(report), indent=2))

    def save_queries(self, run_id: str, batch: dict) -> None:
        self._append(run_id, "queries.jsonl", batch)

    def list_runs(self) -> list:
        if not self.root.exists():
            return []
        runs = [d for d in self.root.iterdir() if (d / "events.jsonl").exists()]
        return [d.name for d in sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)]

    def load_events(self, run_id: str) -> list:
        p = self.root / run_id / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def events_path(self, run_id: str) -> str:
        return str(self.root / run_id / "events.jsonl")


class MongoStore(Store):
    """Atlas + JSON. Construct via get_store(); raises if the ping fails so get_store can fall back."""

    def __init__(self, uri: str, db_name: str = "unbox", json_store: JsonStore | None = None) -> None:
        from pymongo import ASCENDING, MongoClient

        self.json = json_store or JsonStore()
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=8000)
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.db.events.create_index([("run_id", ASCENDING), ("t", ASCENDING)])
        self.db.queries.create_index([("run_id", ASCENDING)])
        self._warned = False

    def _safe(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - never let the database break a run
            if not self._warned:
                print(f"[store] MongoDB write failed, continuing on JSON only: {type(e).__name__}: {e}")
                self._warned = True

    def save_event(self, run_id: str, event: dict) -> None:
        self.json.save_event(run_id, event)
        doc = to_jsonable(event)
        if doc.get("type") == "run_started":
            self._safe(lambda: self.db.runs.update_one({"run_id": run_id},
                                                       {"$set": {"run_id": run_id, **doc.get("data", {})}},
                                                       upsert=True))
        if doc.get("type") == "frames":                       # base64 clips stay in JSON only (size)
            doc = {**doc, "data": {k: v for k, v in doc["data"].items() if k not in ("bb", "clone")}}
        self._safe(lambda: self.db.events.insert_one({**doc, "run_id": run_id}))

    def save_brain_call(self, run_id: str, record: dict) -> None:
        self.json.save_brain_call(run_id, record)
        self._safe(lambda: self.db.brain_calls.insert_one({**to_jsonable(record), "run_id": run_id}))

    def save_program(self, run_id: str, program: dict) -> None:
        self.json.save_program(run_id, program)
        self._safe(lambda: self.db.programs.insert_one({**to_jsonable(program), "run_id": run_id}))

    def save_ledger(self, run_id: str, entries: list) -> None:
        self.json.save_ledger(run_id, entries)
        self._safe(lambda: self.db.ledger.insert_one({"run_id": run_id, "t": time.time(),
                                                      "entries": to_jsonable(entries)}))

    def save_eval(self, run_id: str, report: dict) -> None:
        self.json.save_eval(run_id, report)
        self._safe(lambda: self.db.runs.update_one({"run_id": run_id}, {"$set": {"eval": to_jsonable(report)}},
                                                   upsert=True))

    def save_queries(self, run_id: str, batch: dict) -> None:
        self.json.save_queries(run_id, batch)
        b = to_jsonable(batch)
        rows = [{"run_id": run_id, "query_id": b.get("query_id"), "source": b.get("source"),
                 "rule_id": b.get("rule_id"), "obs": o, "action": a}
                for o, a in zip(b.get("obs", []), b.get("actions", []))]
        if rows:
            self._safe(lambda: self.db.queries.insert_many(rows, ordered=False))

    def list_runs(self) -> list:
        return self.json.list_runs()

    def load_events(self, run_id: str) -> list:
        return self.json.load_events(run_id)     # JSON is the complete copy (frames included)

    def events_path(self, run_id: str) -> str:
        return self.json.events_path(run_id)


_STORE: Store | None = None
_STORE_LOCK = threading.Lock()


def get_store(kind: str | None = None) -> Store:
    """STORE=json (default) or mongo. Mongo falls back to JSON with one warning if Atlas is unreachable."""
    global _STORE
    with _STORE_LOCK:
        kind = (kind or os.getenv("STORE", "json")).lower()
        if _STORE is not None and (kind != "mongo") == isinstance(_STORE, JsonStore):
            return _STORE
        if kind == "mongo" and os.getenv("MONGODB_URI"):
            try:
                _STORE = MongoStore(os.environ["MONGODB_URI"], os.getenv("MONGODB_DB", "unbox"))
                return _STORE
            except Exception as e:  # noqa: BLE001
                print(f"[store] MongoDB unavailable ({type(e).__name__}: {str(e)[:120]}); falling back to JSON")
        _STORE = JsonStore()
        return _STORE


def replay(run_id: str, emit: Callable[[dict], None], speed: float = 0.0, store: Store | None = None) -> int:
    """Re-emit a stored run's events in order. speed=0 -> as fast as possible; else original timing / speed."""
    events = (store or get_store()).load_events(run_id)
    t_prev = 0.0
    for ev in events:
        if speed > 0:
            dt = (float(ev.get("t", 0.0)) - t_prev) / speed
            if dt > 0:
                time.sleep(min(dt, 5.0))
            t_prev = float(ev.get("t", 0.0))
        emit(ev)
    return len(events)
