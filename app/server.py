"""Lane B, task B5: app server (FastAPI :8000).

REST + WebSocket bridge between run threads and the browser:

* A run (live agent, replay, or fallback replay) owns a ``Bridge``: an append-only event
  buffer plus the set of connected sockets. ``emit`` is called from any thread; it
  appends to the buffer and hands the event to each socket's asyncio queue with
  ``loop.call_soon_threadsafe``. A socket that connects late gets the whole buffer
  first, so a reload mid-demo catches up.
* ``/api/replay`` re-emits any stored ``.jsonl`` (fixture, ``runs/<id>``, or
  ``cache/<key>``) with the original spacing / speed, through the same bridge.
* Live fallback: if the agent is not built yet (``NotImplementedError``), fails, or
  overruns ``LIVE_TIMEOUT_S``, the run switches to the cached run of the same
  configuration, and says so on screen. Never silent.

Only ``app/`` holds the admin token. ``app/`` imports nothing from ``blackbox/`` except
``blackbox.client`` (tests/test_isolation.py).
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001 - dotenv is optional
    pass

# The live run renders clone clips (and the shield clip) from worker threads. On macOS the
# default Cocoa SDL driver aborts the whole process off the main thread; use the dummy driver
# (same guard as blackbox/policies.py).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "static"
FIXTURES_DIR = ROOT / "fixtures"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
DEFAULT_FIXTURE = "demo_events"

FALLBACK_TIMEOUT_TEXT = ("The live run could not finish within the time limit. Showing our earlier "
                         "run of this exact configuration.")
FALLBACK_REPLAY_MAX_S = 20.0       # a cached fallback replays in at most this many seconds

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def use_mocks() -> bool:
    return _env_flag("USE_MOCKS", "1")


def live_timeout_s() -> float:
    return float(os.getenv("LIVE_TIMEOUT_S", "75"))


def fallback_speed() -> float:
    """Replay speed for the live fallback (1.0 = real time). Tests turn it up."""
    return float(os.getenv("FALLBACK_SPEED", "1.0"))


def live_grace_s() -> float:
    """Extra seconds past LIVE_TIMEOUT_S before the app gives up on the agent thread."""
    return float(os.getenv("LIVE_GRACE_S", "3"))


def best_program(bridge: "Bridge") -> dict | None:
    """The program the UI shows everywhere: highest held-out match so far (latest on a tie)."""
    best = None
    with bridge.lock:
        for ev in bridge.buffer:
            if ev["type"] == "program":
                d = ev["data"]
                if best is None or (d.get("agreement") or 0) >= (best.get("agreement") or 0):
                    best = d
    return best


def blackbox_url() -> str:
    return os.getenv("BLACKBOX_URL", "http://localhost:8001").rstrip("/")


def admin_headers() -> dict:
    return {"X-Admin-Token": os.getenv("BLACKBOX_ADMIN_TOKEN", "change-me")}


MOCK_POLICIES = [{"bb": "mock", "label": "CartPole-PPO (mock)", "env_id": "CartPole-v1"}]
MOCK_TWISTS = [{"twist_id": "t-cp-1", "hidden_label": "Hidden flaw #1"}]
MOCK_REVEAL = {"twist_id": "t-cp-1", "reveal_label": "Over-reacts when moving fast",
               "feature": "cart_v", "rule_text": "cart_v > 0.5 -> push_left"}
BRAINS = [{"id": "muse", "label": "Muse"}]

# Plain-English ways a ledger rule can mention a feature (reveal highlighting).
FEATURE_WORDS = {
    "cart_x": ["cart_x", "cart position", "cart's position", "on the track", "track position"],
    "cart_v": ["cart_v", "cart speed", "cart's speed", "cart-speed", "cart velocity",
               "cart moving", "moving fast"],
    "pole_angle": ["pole_angle", "pole's angle", "pole angle", "leans", "lean"],
    "pole_ang_v": ["pole_ang_v", "turn rate", "angular velocity", "pole turns"],
    "x": ["horizontal position", "right side", "left side", " x "],
    "y": ["altitude", "height", "up high", " y "],
    "vx": ["horizontal speed", "sideways speed", "vx"],
    "vy": ["vertical speed", "descent", "falling", "vy"],
    "angle": ["tilt", "angle"],
    "ang_v": ["spin", "angular velocity", "ang_v"],
    "leg_left": ["left leg", "leg_left"],
    "leg_right": ["right leg", "leg_right"],
}


def rules_mentioning(entries: list[dict], feature: str) -> list[str]:
    words = [w.lower() for w in FEATURE_WORDS.get(feature, [])] + [feature.lower()]
    out = []
    for e in entries:
        text = f" {str(e.get('text', '')).lower()} "
        if any(w in text for w in words):
            out.append(e.get("rule_id"))
    return [r for r in out if r]


# ---------------------------------------------------------------------------
# the bridge: thread -> asyncio
# ---------------------------------------------------------------------------
class Bridge:
    def __init__(self, run_id: str, *, kind: str, config: dict | None = None) -> None:
        self.run_id = run_id
        self.kind = kind                      # "live" | "replay"
        self.config = config or {}
        self.t0 = time.time()
        self.buffer: list[dict] = []
        self.sockets: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self.lock = threading.Lock()
        self.finished = False
        self.superseded = False               # live agent dropped after a fallback switch
        self.source: dict[str, dict] = {}     # last reveal/fix/shield seen in a replayed source
        self.ctx: dict[str, Any] = {}         # bb, spec, env_id, demo_seed, ... for reveal/fix/shield

    # -- emit ------------------------------------------------------------
    def emit(self, event: dict, *, from_agent: bool = False) -> dict | None:
        """Append + fan out. Returns the stored envelope (None if dropped)."""
        if from_agent and self.superseded:
            return None
        ev = self._envelope(event)
        with self.lock:
            self.buffer.append(ev)
            if ev["type"] == "run_finished":
                self.finished = True
            targets = list(self.sockets)
        for loop, q in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:              # loop closed: socket is gone
                pass
        return ev

    def _envelope(self, event: dict) -> dict:
        if "type" not in event:
            raise ValueError(f"event without a type: {event!r}")
        data = event.get("data")
        if data is None:                      # tolerate flat {type, ...fields}
            data = {k: v for k, v in event.items() if k not in ("type", "run_id", "t")}
        t = event.get("t")
        if t is None:
            t = time.time() - self.t0
        return {"type": event["type"], "run_id": self.run_id, "t": round(float(t), 3),
                "data": _jsonable(data)}

    def now(self) -> float:
        return round(time.time() - self.t0, 3)

    # -- sockets -----------------------------------------------------------
    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[list[dict], asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue()
        with self.lock:
            snapshot = list(self.buffer)
            self.sockets.append((loop, q))
        return snapshot, q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self.lock:
            self.sockets = [(lp, qq) for lp, qq in self.sockets if qq is not q]

    # -- state helpers ---------------------------------------------------
    def last(self, etype: str) -> dict | None:
        with self.lock:
            for ev in reversed(self.buffer):
                if ev["type"] == etype:
                    return ev["data"]
        return None


RUNS: dict[str, Bridge] = {}
RUNS_LOCK = threading.Lock()


def new_bridge(kind: str, config: dict | None = None, run_id: str | None = None) -> Bridge:
    rid = run_id or f"{kind}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    b = Bridge(rid, kind=kind, config=config)
    with RUNS_LOCK:
        RUNS[rid] = b
    return b


def get_bridge(run_id: str) -> Bridge:
    with RUNS_LOCK:
        b = RUNS.get(run_id)
    if b is None:
        raise HTTPException(404, f"unknown run_id {run_id!r}")
    return b


def _jsonable(obj: Any) -> Any:
    try:
        from store.store import to_jsonable

        return to_jsonable(obj)
    except Exception:  # noqa: BLE001
        return json.loads(json.dumps(obj, default=str))


def _store():
    try:
        from store.store import get_store

        return get_store()
    except Exception:  # noqa: BLE001
        return None


def _persist(run_id: str, ev: dict | None) -> None:
    """App-originated events on a live run (reveal/fix/shield/fallback) go to the store."""
    st = _store()
    if st is None or ev is None:
        return
    try:
        st.save_event(run_id, ev)
    except Exception:  # noqa: BLE001 - never let storage break the demo
        pass


# ---------------------------------------------------------------------------
# event sources for replay
# ---------------------------------------------------------------------------
def _check_name(name: str, what: str) -> str:
    if not name or not _SAFE_NAME.match(name) or ".." in name:
        raise HTTPException(400, f"bad {what} {name!r}")
    return name


def fixture_path(name: str) -> Path:
    name = _check_name(name.removesuffix(".jsonl"), "fixture")
    return FIXTURES_DIR / f"{name}.jsonl"


def run_path(run_id: str) -> Path:
    return RUNS_DIR / _check_name(run_id, "run_id") / "events.jsonl"


def cache_path(key: str) -> Path:
    return CACHE_DIR / _check_name(key, "cache_key") / "events.jsonl"


def cache_key(bb: str, twist_id: str | None, brain: str) -> str:
    return f"{bb}__{twist_id or 'none'}__{brain}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ensure_fixture() -> Path:
    p = fixture_path(DEFAULT_FIXTURE)
    if not p.exists():
        from scripts import make_fixture

        make_fixture.main(["--quiet", "--out", str(p)])
    return p


def replay_into(bridge: Bridge, events: list[dict], speed: float, *, t_offset: float = 0.0,
                mark_cached: bool = True, stop: threading.Event | None = None) -> int:
    """Re-emit `events` into `bridge` with the original spacing / speed (speed<=0: instant)."""
    start = time.time()
    t_first = float(events[0].get("t", 0.0)) if events else 0.0
    n = 0
    for ev in events:
        if stop is not None and stop.is_set():
            break
        t_ev = float(ev.get("t", 0.0)) - t_first
        if speed > 0:
            wait = t_ev / speed - (time.time() - start)
            if wait > 0:
                time.sleep(min(wait, 10.0))
        data = dict(ev.get("data") or {})
        etype = ev.get("type")
        if etype in ("reveal", "fix", "shield"):
            bridge.source[etype] = data
        if etype == "run_finished" and mark_cached:
            data["cached"] = True
        bridge.emit({"type": etype, "t": t_offset + (t_ev / speed if speed > 0 else t_ev),
                     "data": data})
        n += 1
    return n


def _replay_source_reveal_only(events: list[dict]) -> tuple[list[dict], dict]:
    """Split a source stream: reveal/fix/shield wait for the buttons, the rest plays."""
    held: dict[str, dict] = {}
    play: list[dict] = []
    for ev in events:
        if ev.get("type") in ("reveal", "fix", "shield"):
            held[ev["type"]] = ev.get("data") or {}
        else:
            play.append(ev)
    return play, held


# ---------------------------------------------------------------------------
# black box access (admin calls live here and only here)
# ---------------------------------------------------------------------------
def admin_get(path: str, **params) -> Any:
    r = httpx.get(f"{blackbox_url()}{path}", params=params, headers=admin_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def admin_post(path: str, body: dict) -> Any:
    r = httpx.post(f"{blackbox_url()}{path}", json=body, headers=admin_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def list_policies() -> tuple[list[dict], str | None]:
    if use_mocks():
        return MOCK_POLICIES, None
    try:
        return admin_get("/admin/policies"), None
    except Exception as e:  # noqa: BLE001
        return MOCK_POLICIES, f"black box unreachable at {blackbox_url()}: {type(e).__name__}"


def policy_info(bb: str) -> dict:
    pols, _ = list_policies()
    for p in pols:
        if p.get("bb") == bb:
            return p
    return {"bb": bb, "label": bb, "env_id": "CartPole-v1"}


def make_blackbox(bb: str):
    if use_mocks() or bb == "mock":
        from contracts.mocks import MockBlackBox

        return MockBlackBox()
    from blackbox.client import HttpBlackBox

    return HttpBlackBox(bb, blackbox_url())


def make_numerics():
    if use_mocks():
        from contracts.mocks import numerics_stub

        return numerics_stub
    import numerics

    return numerics


def make_env_factory(env_id: str, render: bool = False):
    def env_factory(seed: int):
        import gymnasium as gym

        env = gym.make(env_id, render_mode="rgb_array" if render else None)
        obs, _ = env.reset(seed=int(seed))
        return env, [float(v) for v in obs]

    return env_factory


def demo_seed_for(bb: str) -> int:
    return int(os.getenv(f"DEMO_SEED_{bb.replace('-', '_').upper()}", os.getenv("DEMO_SEED", "0")))


# ---------------------------------------------------------------------------
# live runs
# ---------------------------------------------------------------------------
def _fallback(bridge: Bridge, reason: str, cfg: dict) -> None:
    """Switch a live run to the cached run of this configuration (or the fixture)."""
    bridge.superseded = True
    key = cache_key(cfg["bb"], cfg.get("twist_id"), cfg["brain"])
    try:
        events = load_jsonl(cache_path(key))
        banner = FALLBACK_TIMEOUT_TEXT if reason == "timeout" else (
            f"{reason} Showing our earlier run of this exact configuration.")
        source = f"cache/{key}"
    except FileNotFoundError:
        events = load_jsonl(ensure_fixture())
        banner = (f"{reason} No cached run exists for {key}; replaying the UI fixture "
                  f"(fixtures/{DEFAULT_FIXTURE}.jsonl), not this configuration.")
        source = f"fixtures/{DEFAULT_FIXTURE}.jsonl"
    ev = {"type": "status", "t": bridge.now(),
          "data": {"text": banner, "banner": "fallback", "source": source}}
    _persist(bridge.run_id, bridge.emit(ev))
    bridge.ctx["fallback_source"] = source
    play, held = _replay_source_reveal_only(events)
    bridge.source.update(held)
    span = (float(play[-1].get("t", 0)) - float(play[0].get("t", 0))) if play else 0.0
    speed = fallback_speed()
    if speed > 0:                                  # 0 = instant (tests); else cap the replay length
        speed = max(speed, span / FALLBACK_REPLAY_MAX_S)
    replay_into(bridge, play, speed=speed, t_offset=bridge.now())


def _run_thread(bridge: Bridge, cfg: dict) -> None:
    import agent.loop as agent_loop

    info = policy_info(cfg["bb"])
    live = bool(cfg.get("live", True))
    timeout = live_timeout_s() if live else None
    result: dict[str, Any] = {}
    failure: dict[str, BaseException] = {}

    try:
        bb = make_blackbox(cfg["bb"])
        numerics = make_numerics()
    except NotImplementedError as e:
        _error(bridge, f"Black box client not built yet ({type(e).__name__}).")
        _fallback(bridge, "Black box client not built yet.", cfg)
        return
    except Exception as e:  # noqa: BLE001
        _error(bridge, f"Could not reach the black box: {type(e).__name__}: {e}")
        _fallback(bridge, "Black box unavailable.", cfg)
        return

    bridge.ctx.update({"bb": bb, "numerics": numerics, "env_id": info.get("env_id"),
                       "demo_seed": cfg["demo_seed"], "policy_label": info.get("label")})

    def emit(event: dict) -> None:
        bridge.emit(event, from_agent=True)

    def target() -> None:
        try:
            result.update(agent_loop.run(
                bb, cfg["brain"],
                # render=True: render_pair draws the clone clip with env.render()
                env_factory=make_env_factory(info.get("env_id", "CartPole-v1"), render=True),
                numerics=numerics, emit=emit, store=_store(), run_id=bridge.run_id,
                budget=int(os.getenv("QUERY_BUDGET", "5000")),
                max_turns=int(os.getenv("LIVE_MAX_TURNS", "4")),
                live=live, timeout_s=timeout, policy_label=info.get("label", cfg["bb"]),
                demo_seed=cfg["demo_seed"],
            ) or {})
        except BaseException as e:  # noqa: BLE001 - report everything, never crash the server
            failure["e"] = e

    t = threading.Thread(target=target, name=f"agent-{bridge.run_id}", daemon=True)
    t.start()
    t.join(timeout=(timeout + live_grace_s()) if timeout else None)

    if t.is_alive():
        _fallback(bridge, "timeout", cfg)
        return
    if "e" in failure:
        e = failure["e"]
        if isinstance(e, TimeoutError):
            _fallback(bridge, "timeout", cfg)
        elif isinstance(e, NotImplementedError):
            _error(bridge, "agent.loop.run is not built yet (NotImplementedError).")
            _fallback(bridge, "Agent loop not built yet.", cfg)
        else:
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            _error(bridge, f"Agent run failed: {tb}")
            _fallback(bridge, "Live run failed.", cfg)
        return

    bridge.ctx["result"] = result
    st = _store()
    if st is not None:                        # make sure the run replays later
        try:
            if not st.load_events(bridge.run_id):
                for ev in list(bridge.buffer):
                    st.save_event(bridge.run_id, ev)
        except Exception:  # noqa: BLE001
            pass
    if not bridge.finished:
        bridge.emit({"type": "run_finished", "data": {
            "eval": _jsonable(result.get("eval")), "total_queries": None, "turns": None,
            "seconds": bridge.now(), "cached": False}})


def _error(bridge: Bridge, message: str) -> None:
    ev = {"type": "error", "t": bridge.now(), "data": {"message": message}}
    _persist(bridge.run_id, bridge.emit(ev))


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
app = FastAPI(title="Unbox")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "app"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"use_mocks": use_mocks(), "snap": _env_flag("SNAP", "1"),
            "live_timeout_s": live_timeout_s(), "brains": BRAINS,
            "default_fixture": DEFAULT_FIXTURE}


@app.get("/api/policies")
def api_policies() -> dict:
    pols, warning = list_policies()
    return {"policies": pols, "warning": warning}


@app.get("/api/twists")
def api_twists(bb: str) -> dict:
    if use_mocks() or bb == "mock":
        return {"twists": MOCK_TWISTS}
    try:
        tw = admin_get("/admin/twists", bb=bb)
    except Exception as e:  # noqa: BLE001
        return {"twists": [], "warning": f"black box unreachable: {type(e).__name__}"}
    # hidden labels only; never forward anything that would reveal the flaw
    return {"twists": [{"twist_id": t["twist_id"], "hidden_label": t["hidden_label"]} for t in tw]}


@app.get("/api/sources")
def api_sources() -> dict:
    """Everything /api/replay can play."""
    fixtures = sorted(p.stem for p in FIXTURES_DIR.glob("*.jsonl"))
    runs = sorted((p.parent.name for p in RUNS_DIR.glob("*/events.jsonl")), reverse=True)
    caches = sorted(p.parent.name for p in CACHE_DIR.glob("*/events.jsonl"))
    return {"fixtures": fixtures, "runs": runs, "cache": caches}


class RunReq(BaseModel):
    bb: str
    twist_id: str | None = None
    brain: str = "muse"
    live: bool = True
    demo_seed: int | None = None


@app.post("/api/run")
def api_run(req: RunReq) -> dict:
    cfg = req.model_dump()
    cfg["demo_seed"] = req.demo_seed if req.demo_seed is not None else demo_seed_for(req.bb)
    if not (use_mocks() or req.bb == "mock"):
        try:
            admin_post("/admin/config", {"bb": req.bb, "twist_id": req.twist_id})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not set the hidden flaw via admin: {type(e).__name__}: {e}")
    bridge = new_bridge("live", cfg)
    threading.Thread(target=_run_thread, args=(bridge, cfg), daemon=True,
                     name=f"run-{bridge.run_id}").start()
    return {"run_id": bridge.run_id}


class ReplayReq(BaseModel):
    run_id: str | None = None
    cache_key: str | None = None
    fixture: str | None = None
    speed: float = 1.0
    hold_reveal: bool = True       # keep reveal/fix/shield for the buttons, like a live run
    max_seconds: float | None = None   # speed up so the replay takes at most this long


@app.post("/api/replay")
def api_replay(req: ReplayReq) -> dict:
    if req.run_id:
        path, label = run_path(req.run_id), f"runs/{req.run_id}"
    elif req.cache_key:
        path, label = cache_path(req.cache_key), f"cache/{req.cache_key}"
    else:
        name = req.fixture or DEFAULT_FIXTURE
        path = ensure_fixture() if name == DEFAULT_FIXTURE else fixture_path(name)
        label = f"fixtures/{path.name}"
    try:
        events = load_jsonl(path)
    except FileNotFoundError:
        raise HTTPException(404, f"no event log at {label}")
    if not events:
        raise HTTPException(404, f"{label} is empty")

    speed = req.speed
    if req.max_seconds and speed > 0 and len(events) > 1:
        span = float(events[-1].get("t", 0)) - float(events[0].get("t", 0))
        speed = max(speed, span / req.max_seconds)
    cfg: dict[str, Any] = {"source": label, "speed": speed}
    m = re.match(r"^(bb-\d+)__(.+?)__(\w+)$", req.cache_key or "")
    if m:                                  # a cached run of a real config: Reveal/Fix/Guard work live
        cfg.update(bb=m.group(1), twist_id=None if m.group(2) == "none" else m.group(2), brain=m.group(3))
    bridge = new_bridge("replay", cfg)
    bridge.ctx["fallback_source"] = label
    if m and not use_mocks():
        try:
            admin_post("/admin/config", {"bb": cfg["bb"], "twist_id": cfg["twist_id"]})
            seed = next((int(e["data"]["seed"]) for e in events
                         if e.get("type") == "frames" and e.get("data", {}).get("seed") is not None),
                        demo_seed_for(cfg["bb"]))
            info = policy_info(cfg["bb"])
            bridge.ctx.update({"bb": make_blackbox(cfg["bb"]), "numerics": make_numerics(),
                               "env_id": info.get("env_id"), "demo_seed": seed,
                               "policy_label": info.get("label")})
        except Exception as e:  # noqa: BLE001 - the replay still plays; reveal may fail later
            print(f"[replay] could not set up live buttons for {label}: {type(e).__name__}: {e}")
    play, held = _replay_source_reveal_only(events) if req.hold_reveal else (events, {})
    bridge.source.update(held)
    threading.Thread(target=replay_into, args=(bridge, play, speed), daemon=True,
                     name=f"replay-{bridge.run_id}").start()
    return {"run_id": bridge.run_id, "source": label, "events": len(events),
            "held": sorted(held)}


class RunRef(BaseModel):
    run_id: str


@app.post("/api/reveal")
def api_reveal(req: RunRef) -> dict:
    bridge = get_bridge(req.run_id)
    if bridge.ctx.get("reveal") is not None:      # idempotent: one reveal per run
        return bridge.ctx["reveal"]
    ledger = (bridge.last("ledger") or {}).get("entries", [])
    held = bridge.source.get("reveal")
    if held is not None and (bridge.kind == "replay" or bridge.superseded):
        data = dict(held)
    else:
        cfg = bridge.config
        if use_mocks() or cfg.get("bb") == "mock":
            info = MOCK_REVEAL
        else:
            try:
                info = admin_get("/admin/reveal", bb=cfg["bb"])
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"admin reveal failed: {type(e).__name__}: {e}")
        data = {"label": info["reveal_label"], "feature": info["feature"],
                "highlight_rule_ids": rules_mentioning(ledger, info["feature"])}
    if not data.get("highlight_rule_ids"):
        data["highlight_rule_ids"] = rules_mentioning(ledger, data.get("feature", ""))
    bridge.ctx["reveal"] = data
    ev = bridge.emit({"type": "reveal", "t": bridge.now(), "data": data})
    if bridge.kind == "live":
        _persist(bridge.run_id, ev)
    return data


@app.post("/api/fix")
def api_fix(req: RunRef) -> dict:
    bridge = get_bridge(req.run_id)
    if bridge.ctx.get("fix") is not None:         # idempotent: one fix per run
        return {"ok": True, "diff": bridge.ctx["fix"].get("diff", "")}
    program = best_program(bridge)
    reveal = bridge.ctx.get("reveal") or bridge.last("reveal")
    if program is None:
        raise HTTPException(409, "no program recovered yet")
    if reveal is None:
        raise HTTPException(409, "reveal the flaw first")

    data: dict | None = None
    held = bridge.source.get("fix")
    if held is not None and (bridge.kind == "replay" or bridge.superseded):
        data = dict(held)
    else:
        try:
            import agent.fixer as fixer

            out = fixer.fix(program, (bridge.last("ledger") or {}).get("entries", []),
                            reveal["label"], bridge.config.get("brain", "muse"), bridge.run_id)
            data = {"code": out["code"], "params": out.get("params", {}),
                    "diff": out.get("diff") or "".join(difflib.unified_diff(
                        program["code"].splitlines(keepends=True),
                        out["code"].splitlines(keepends=True),
                        fromfile="recovered.py", tofile="fixed.py", n=2)),
                    "changed_rule_ids": out.get("changed_rule_ids", []),
                    "explanation": out.get("explanation", "")}
        except NotImplementedError:
            if held is not None:
                data = dict(held)
            else:
                _error(bridge, "agent.fixer.fix is not built yet (NotImplementedError).")
                raise HTTPException(501, "agent.fixer.fix is not built yet")
        except Exception as e:  # noqa: BLE001
            _error(bridge, f"Fix failed: {type(e).__name__}: {e}")
            raise HTTPException(500, f"fix failed: {type(e).__name__}: {e}")

    bridge.ctx["fix"] = data
    ev = bridge.emit({"type": "fix", "t": bridge.now(), "data": data})
    if bridge.kind == "live":
        _persist(bridge.run_id, ev)
    return {"ok": True, "diff": data.get("diff", "")}


class ShieldReq(BaseModel):
    run_id: str
    on: bool


def shield_clip(bridge: Bridge, *, max_steps: int = 1000, frame_every: int = 3) -> list[str]:
    """Run the demo seed locally; each step: bb action from /act, then shielded_action."""
    import base64
    import io

    from PIL import Image

    ctx = bridge.ctx
    bb, numerics = ctx.get("bb"), ctx.get("numerics")
    fix, program = ctx.get("fix") or bridge.last("fix"), best_program(bridge)
    if bb is None or numerics is None or fix is None or program is None:
        raise RuntimeError("shield needs a live run with a recovered program and a fix")
    spec = bb.spec()
    rec = numerics.load_program(program["code"], spec)
    fixed = numerics.load_program(fix["code"], spec)
    env, obs = make_env_factory(ctx.get("env_id") or "CartPole-v1", render=True)(ctx["demo_seed"])
    frames: list[str] = []
    rewards: list[float] = []
    term = trunc = False
    try:
        for step in range(max_steps):
            if step % frame_every == 0:
                im = Image.fromarray(env.render())
                im = im.resize((360, round(360 * im.height / im.width)))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=60)
                frames.append(base64.b64encode(buf.getvalue()).decode())
            bb_action = bb.act([obs], source="probe").actions[0]
            a = numerics.shielded_action(rec, program["params"], fixed, fix["params"], obs, bb_action)
            o, r, term, trunc, _ = env.step(a)
            rewards.append(float(r))
            obs = [float(v) for v in o]
            if term or trunc:
                break
    finally:
        env.close()
    bridge.ctx["shield_end"] = {"terminated": bool(term), "truncated": bool(trunc), "steps": len(rewards),
                                "last_reward": rewards[-1] if rewards else None,
                                "total_reward": float(sum(rewards))}
    return frames


@app.post("/api/shield")
def api_shield(req: ShieldReq) -> dict:
    bridge = get_bridge(req.run_id)
    if not req.on:
        data = {"on": False, "bb": None}
    else:
        held = bridge.source.get("shield")
        cached = bridge.ctx.get("shield_frames")
        if held is not None and (bridge.kind == "replay" or bridge.superseded):
            data = {"on": True, "bb": held.get("bb")}
        elif cached:
            data = {"on": True, "bb": cached, "bb_end": bridge.ctx.get("shield_end")}
        else:
            try:
                frames = shield_clip(bridge)
            except NotImplementedError:
                if held is None:
                    _error(bridge, "Shield needs numerics.shielded_action (not built yet).")
                    raise HTTPException(501, "numerics.shielded_action is not built yet")
                frames = held.get("bb")
            except Exception as e:  # noqa: BLE001
                if held is None:
                    _error(bridge, f"Shield failed: {type(e).__name__}: {e}")
                    raise HTTPException(409, f"shield failed: {type(e).__name__}: {e}")
                frames = held.get("bb")
            bridge.ctx["shield_frames"] = frames
            data = {"on": True, "bb": frames, "bb_end": bridge.ctx.get("shield_end")}
    bridge.emit({"type": "shield", "t": bridge.now(), "data": data})
    return {"ok": True, "on": data["on"], "frames": len(data["bb"] or [])}


@app.get("/api/results")
def api_results(source: str = "results") -> dict:
    """results/*.json (A6 batch). source=fixture serves fixtures/results/ for UI work."""
    folder = FIXTURES_DIR / "results" if source == "fixture" else RESULTS_DIR
    out: dict[str, Any] = {}
    for p in sorted(folder.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            out[p.stem] = {"error": f"unreadable: {type(e).__name__}"}
    return {"available": bool(out), "results": out, "source": source if source == "fixture" else "results"}


@app.get("/api/runs/{run_id}")
def api_run_state(run_id: str) -> dict:
    b = get_bridge(run_id)
    return {"run_id": b.run_id, "kind": b.kind, "events": len(b.buffer),
            "finished": b.finished, "superseded": b.superseded,
            "source": b.ctx.get("fallback_source"), "held": sorted(b.source),
            "config": {k: b.config.get(k) for k in ("bb", "twist_id", "brain")}}


@app.websocket("/ws/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    with RUNS_LOCK:
        bridge = RUNS.get(run_id)
    if bridge is None:
        await websocket.send_json({"type": "error", "run_id": run_id, "t": 0.0,
                                   "data": {"message": f"unknown run_id {run_id!r}"}})
        await websocket.close()
        return
    snapshot, q = bridge.subscribe(asyncio.get_running_loop())
    try:
        for ev in snapshot:                  # catch-up first: late joiners and reloads
            await websocket.send_json(ev)
        while True:
            ev = await q.get()
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bridge.unsubscribe(q)


from app.report import router as report_router  # noqa: E402

app.include_router(report_router)
