"""Lane B, task B1: smoke check. Pings the brain(s), the black box (real or mock), numerics and the store.

    USE_MOCKS=1 python -m scripts.smoke
"""
from __future__ import annotations

import os
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _check_muse() -> str:
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY missing in .env")
    import openai

    client = openai.OpenAI(base_url=os.getenv("META_BASE_URL"), api_key=key, timeout=60, max_retries=1)
    model = os.getenv("META_MODEL", "muse-spark-1.1")
    t0 = time.time()
    r = client.chat.completions.create(model=model, max_tokens=4000,
                                       messages=[{"role": "user", "content": "Reply with the single word OK."}])
    text = (r.choices[0].message.content or "").strip()
    return f"{model}: {text[:20]!r} in {time.time() - t0:.1f}s"


def _check_subq() -> str:
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        raise RuntimeError("SUBQ_API_KEY missing in .env")
    import httpx

    base = os.getenv("SUBQ_BASE_URL", "https://api.subq.ai/v1").rstrip("/")
    model = os.getenv("SUBQ_MODEL", "subq-snap-1.1")
    body = {"model": model, "state": "The sky is blue on a clear day.",
            "questions": {"blue": {"type": "noul", "instructions": "Does the text say the sky is blue?"}}}
    t0 = time.time()
    r = httpx.post(f"{base}/systemone", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    p = r.json()["answers"]["blue"]["noul"]
    return f"{model}: P(yes)={p:.2f} in {time.time() - t0:.1f}s"


def _mocks() -> bool:
    return os.getenv("USE_MOCKS", "1") == "1"


def _check_blackbox() -> str:
    if _mocks():
        from contracts.mocks import MockBlackBox

        bb = MockBlackBox()
        label = "mock"
    else:
        from blackbox.client import HttpBlackBox

        bb = HttpBlackBox("bb-01", os.getenv("BLACKBOX_URL", "http://localhost:8001"))
        label = "bb-01 @ " + os.getenv("BLACKBOX_URL", "")
    spec = bb.spec()
    q = bb.act([[0.0] * len(spec.obs_names)] * 4, source="probe")
    ep = bb.episode(0)
    assert len(q.actions) == 4, "act() did not return one action per obs"
    return f"{label}: {len(spec.obs_names)} obs, act ok ({q.query_id}), episode reward {ep.total_reward:.0f}"


def _check_numerics() -> str:
    from contracts.mocks import CARTPOLE_SPEC

    if _mocks():
        from contracts.mocks import numerics_stub as nm

        label = "stub"
    else:
        import numerics as nm

        label = "numerics"
    code = "def policy(obs, P):\n    return 1 if obs[2] + P['k'] * obs[3] > 0 else 0\n"
    fn = nm.load_program(code, CARTPOLE_SPEC)
    X = [[0, 0, 0.05, 0.1], [0, 0, -0.05, -0.1], [0, 0, 0.01, -0.1], [0, 0, -0.01, 0.1]]
    y = [1, 0, 0, 1]
    params, agr = nm.fit_params(fn, {"k": 0.5}, X, y, CARTPOLE_SPEC, max_seconds=1.0)
    return f"{label}: load_program + fit_params ok (agreement {agr:.2f})"


def _check_store() -> str:
    from store.store import get_store

    st = get_store()
    st.save_event("smoke", {"type": "status", "run_id": "smoke", "t": 0.0, "data": {"text": "smoke"}})
    assert st.load_events("smoke"), "event did not round-trip"
    return f"{type(st).__name__}: write + reload ok"


CHECKS = [
    ("Muse (Meta Model API)", _check_muse),
    ("SubQ Snap (Decisions API)", _check_subq),
    ("Black box", _check_blackbox),
    ("Numerics", _check_numerics),
    ("Store", _check_store),
]


def main() -> int:
    print(f"USE_MOCKS={'1' if _mocks() else '0'}  BRAIN={os.getenv('BRAIN', 'muse')}  STORE={os.getenv('STORE', 'json')}")
    failed = 0
    for name, check in CHECKS:
        try:
            detail = check()
            mark = f"{GREEN}  OK  {RESET}"
        except NotImplementedError:
            detail, mark = "not built yet (NotImplementedError)", f"{YELLOW} TODO {RESET}"
            failed += 1
        except Exception as e:  # noqa: BLE001
            detail, mark = f"{type(e).__name__}: {e}", f"{RED} FAIL {RESET}"
            failed += 1
            if os.getenv("SMOKE_DEBUG"):
                traceback.print_exc()
        print(f"[{mark}] {name:28s} {detail}")
    print("all green" if not failed else f"{failed} check(s) not green")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
