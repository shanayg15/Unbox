"""Rule 5: log every brain / SubQ call to runs/<run_id>/brain_calls.jsonl (via the store when one is given)."""
from __future__ import annotations

import time


def log_call(run_id: str, record: dict, store=None) -> None:
    rec = {"ts": time.time(), **record}
    if store is None:
        from store.store import JsonStore

        store = JsonStore()
    try:
        store.save_brain_call(run_id, rec)
    except Exception as e:  # noqa: BLE001 - logging must never break a run
        print(f"[calllog] could not log call: {e}")
