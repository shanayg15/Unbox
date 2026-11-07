"""Lane A, tasks A1-A3: black-box server. Run: uvicorn blackbox.server:app --port 8001

Public responses never name the env, checkpoint, algorithm or active twist (CLAUDE.md hard
rule 1). Only /admin/* does, behind the admin token that app/ holds.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import asdict

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from blackbox.policies import DEFAULT_FRAME_EVERY, REGISTRY, get_policy, rollout, sensor_spec
from blackbox.twists import TWISTS, Twist, twists_for

load_dotenv()

app = FastAPI(title="Unbox black box")

_lock = threading.Lock()
_queries: dict[str, int] = {}
_batches: dict[str, int] = {}
_episodes: dict[tuple, dict] = {}
_active: dict[str, Twist] = {}


class ActRequest(BaseModel):
    bb: str
    obs: list[list[float]]
    source: str = "probe"
    rule_id: str | None = None


class EpisodeRequest(BaseModel):
    bb: str
    seed: int
    render: bool = False
    frame_every: int = DEFAULT_FRAME_EVERY


class ConfigRequest(BaseModel):
    bb: str
    twist_id: str | None = None


def _require_known(bb: str) -> None:
    if bb not in REGISTRY:
        raise HTTPException(404, f"unknown black box id: {bb}")


def _require_admin(x_admin_token: str = Header(default="")) -> None:
    expected = os.environ.get("BLACKBOX_ADMIN_TOKEN", "")
    if not expected or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "admin token required")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "blackbox"}


@app.get("/spec")
def spec(bb: str) -> dict:
    _require_known(bb)
    return asdict(sensor_spec(bb))


@app.post("/act")
def act(req: ActRequest) -> dict:
    _require_known(req.bb)
    width = len(REGISTRY[req.bb].obs_names)
    bad = next((i for i, row in enumerate(req.obs) if len(row) != width), None)
    if bad is not None:
        raise HTTPException(422, f"obs[{bad}] has {len(req.obs[bad])} values, expected {width}")
    batch = np.asarray(req.obs, dtype=np.float32).reshape(-1, width)
    if not np.isfinite(batch).all():
        raise HTTPException(422, "obs must be finite 32-bit floats")

    policy = get_policy(req.bb)
    started = time.perf_counter()
    actions = policy.predict(batch).tolist() if len(batch) else []
    elapsed = time.perf_counter() - started

    with _lock:
        twist = _active.get(req.bb)
    if twist is not None:
        actions = [twist.apply(row, action) for row, action in zip(req.obs, actions)]

    with _lock:
        _queries[req.bb] = _queries.get(req.bb, 0) + len(req.obs)
        _batches[req.bb] = _batches.get(req.bb, 0) + 1
        query_id = f"q{_batches[req.bb]}"
    return {
        "query_id": query_id,
        "actions": actions,
        "inference_us": elapsed * 1e6 / max(len(req.obs), 1),
    }


@app.post("/episode")
def episode(req: EpisodeRequest) -> dict:
    _require_known(req.bb)
    if req.frame_every < 1:
        raise HTTPException(422, "frame_every must be at least 1")
    with _lock:
        twist = _active.get(req.bb)
        # frame_every only changes the payload of a rendered episode, so it joins the key only then
        key = (req.bb, twist.twist_id if twist else None, req.seed, req.render,
               req.frame_every if req.render else None)
        cached = _episodes.get(key)
    if cached is not None:
        return cached
    result = asdict(rollout(req.bb, req.seed, twist=twist, render=req.render, frame_every=req.frame_every))
    with _lock:
        _episodes[key] = result
    return result


@app.get("/count")
def count(bb: str) -> dict:
    _require_known(bb)
    with _lock:
        return {"queries": _queries.get(bb, 0)}


@app.get("/admin/policies", dependencies=[Depends(_require_admin)])
def admin_policies() -> list[dict]:
    return [{"bb": d.bb, "label": d.label, "env_id": d.env_id} for d in REGISTRY.values()]


@app.get("/admin/twists", dependencies=[Depends(_require_admin)])
def admin_twists(bb: str) -> list[dict]:
    _require_known(bb)
    return [{"twist_id": t.twist_id, "hidden_label": t.hidden_label} for t in twists_for(bb)]


@app.post("/admin/config", dependencies=[Depends(_require_admin)])
def admin_config(req: ConfigRequest) -> dict:
    _require_known(req.bb)
    twist = None
    if req.twist_id is not None:
        twist = TWISTS.get(req.twist_id)
        if twist is None or twist not in twists_for(req.bb):
            raise HTTPException(422, f"twist {req.twist_id} does not apply to {req.bb}")
    with _lock:
        if twist is None:
            _active.pop(req.bb, None)
        else:
            _active[req.bb] = twist
        _queries[req.bb] = 0
        _batches[req.bb] = 0
        for key in [k for k in _episodes if k[0] == req.bb]:
            del _episodes[key]
    return {"bb": req.bb, "twist_id": req.twist_id}


@app.get("/admin/reveal", dependencies=[Depends(_require_admin)])
def admin_reveal(bb: str) -> dict:
    _require_known(bb)
    with _lock:
        twist = _active.get(bb)
    if twist is None:
        return {"twist_id": None, "reveal_label": None, "feature": None, "rule_text": None}
    return {
        "twist_id": twist.twist_id,
        "reveal_label": twist.reveal_label,
        "feature": twist.feature,
        "rule_text": twist.rule_text,
    }
