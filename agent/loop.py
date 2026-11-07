"""Lane B, task B3: agent loop"""


def run(bb, brain: str, *, env_factory, numerics, emit, store, run_id: str,
        budget: int, max_turns: int, live: bool, timeout_s: float | None,
        policy_label: str, demo_seed: int = 0) -> dict:
    """bb: contracts.types.BlackBox. env_factory(seed) -> (env, first_obs).
    numerics: module with the contracts/numerics_api.py functions (real or stub).
    emit(event_dict) is called from this thread; must be thread-safe on the caller side.
    Returns {"program": Program, "ledger": [LedgerEntry], "eval": EvalReport, "events_path": str}."""
    raise NotImplementedError


def render_pair(fn, params, demo_seed):
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
