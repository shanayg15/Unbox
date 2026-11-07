"""Session owner: B-Show.

Build ``fixtures/demo_events.jsonl``: a fake but realistic run, so the whole UI can be
built and tested with no agent, no brain, no black-box server and no network.

The fixture is a full event stream in ``contracts/events.py`` envelope form, one JSON
object per line, with realistic timestamps spread over ~60 seconds:

  run_started -> (4 turns of: snap, status, ledger, program, metric, frames, probes,
                  query_count, counterexample mining) -> reveal -> fix -> shield
              -> run_finished

The "black box" is a hand-written CartPole policy plus the ``t-cp-1`` hidden flaw
(``cart_v > limit -> push_left``; limit 0.2 in the fixture), so the demo feature is ``cart_v``. Clips are really
rendered from gymnasium, so the UI gets real pixels.

Usage::

    python -m scripts.make_fixture                  # -> fixtures/demo_events.jsonl
    python -m scripts.make_fixture --max-frames 24  # smaller file
    python -m scripts.make_fixture --list-seeds     # show the demo-seed scan
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "fixtures"
DEFAULT_OUT = FIXTURE_DIR / "demo_events.jsonl"

RUN_ID = "fixture-demo-cp1"
POLICY_LABEL = "CartPole-PPO"
BRAIN = "muse"
MODEL = "muse-spark-1.1"
BUDGET = 5000

OBS_NAMES = ["cart_x", "cart_v", "pole_angle", "pole_ang_v"]
ACTION_NAMES = ["push_left", "push_right"]
PUSH_LEFT, PUSH_RIGHT = 0, 1

# The hidden flaw being recovered: t-cp-1 ("cart_v > limit -> push_left").
# The real twist uses 0.5, but this hand balancer never moves the cart that fast,
# so the fixture uses 0.2: it fires on ~15-30% of steps, which is what the UI needs.
TWIST_FEATURE = "cart_v"
TWIST_THRESHOLD = 0.2
REVEAL_LABEL = "Over-reacts when moving fast"

FRAME_WIDTH = 360
FRAME_QUALITY = 60
FRAME_EVERY = 3
DEFAULT_MAX_FRAMES = 32
CLIP_FPS = 10


# ---------------------------------------------------------------------------
# policies: the black box, three clone versions, the fix, the shield
# ---------------------------------------------------------------------------
def bb_policy(obs: list[float]) -> int:
    """The hidden neural net, as a stand-in: a decent balancer plus the t-cp-1 flaw."""
    if obs[1] > TWIST_THRESHOLD:          # cart_v -- the hidden flaw, fires late
        return PUSH_LEFT
    return PUSH_RIGHT if obs[2] + 0.5 * obs[3] > 0 else PUSH_LEFT


def clone_v1(obs: list[float]) -> int:
    """v1: angle only. Visibly worse -- it oscillates and drops the pole."""
    return PUSH_RIGHT if obs[2] > 0 else PUSH_LEFT


def clone_v2(obs: list[float]) -> int:
    """v2: angle plus some angular velocity. Better, still misses the flaw."""
    return PUSH_RIGHT if obs[2] + 0.3 * obs[3] > 0 else PUSH_LEFT


def clone_v3(obs: list[float]) -> int:
    """v3: the recovered program -- reproduces the flaw too. That is the proof."""
    if obs[1] > TWIST_THRESHOLD:
        return PUSH_LEFT
    return PUSH_RIGHT if obs[2] + 0.5 * obs[3] > 0 else PUSH_LEFT


def fixed_policy(obs: list[float]) -> int:
    """After the FIX prompt: the cart_v override is gone, everything else identical."""
    return PUSH_RIGHT if obs[2] + 0.5 * obs[3] > 0 else PUSH_LEFT


def shielded_policy(obs: list[float]) -> int:
    """numerics.shielded_action(recovered=v3, fixed=fix, obs, bb_action)."""
    bb_action = bb_policy(obs)
    rec, fix = clone_v3(obs), fixed_policy(obs)
    return fix if rec != fix else bb_action


CLONES = {1: clone_v1, 2: clone_v2, 3: clone_v3}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _jpeg_b64(frame, width: int = FRAME_WIDTH, quality: int = FRAME_QUALITY) -> str:
    from PIL import Image

    im = Image.fromarray(frame)
    im = im.resize((width, max(1, round(width * im.height / im.width))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def rollout(policy, seed: int, *, render: bool, max_frames: int,
            frame_every: int = FRAME_EVERY) -> dict:
    """Drive a local CartPole with `policy`. Returns obs/actions and (optionally) a clip."""
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    try:
        obs, _ = env.reset(seed=seed)
        obs_log: list[list[float]] = []
        act_log: list[int] = []
        frames: list[str] = []
        frame_steps: list[int] = []
        total_reward = 0.0
        max_steps = max_frames * frame_every
        terminated = truncated = False

        for step in range(max_steps):
            o = [float(v) for v in obs]
            a = int(policy(o))
            obs_log.append(o)
            act_log.append(a)
            if render and step % frame_every == 0 and len(frames) < max_frames:
                frames.append(_jpeg_b64(env.render()))
                frame_steps.append(step)
            obs, reward, terminated, truncated, _ = env.step(a)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "obs": obs_log,
        "actions": act_log,
        "frames": frames,
        "frame_steps": frame_steps,
        "steps": len(act_log),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _pad(frames: list[str], n: int) -> list[str]:
    """Hold the last frame so both clips in a pair have the same length.

    A clone that drops the pole early simply freezes on the fallen pole, which is
    exactly what should stay on screen next to the still-balancing black box.
    """
    if not frames:
        return []
    out = list(frames[:n])
    while len(out) < n:
        out.append(frames[-1])
    return out


def render_pair(clone, demo_seed: int, max_frames: int) -> dict:
    """Mirror of agent.loop.render_pair: bb clip, clone clip, per-frame disagreement."""
    bb = rollout(bb_policy, demo_seed, render=True, max_frames=max_frames)
    cl = rollout(clone, demo_seed, render=True, max_frames=max_frames)
    n = max(len(bb["frames"]), len(cl["frames"]))
    # disagree is measured on the BLACK BOX's trajectory; a frame is red when the
    # clone disagrees on any of the FRAME_EVERY steps that frame stands for
    disagree: list[bool] = []
    for i in range(n):
        lo, hi = i * FRAME_EVERY, min((i + 1) * FRAME_EVERY, len(bb["obs"]))
        if lo < len(bb["obs"]):
            disagree.append(any(clone(bb["obs"][t]) != bb["actions"][t] for t in range(lo, hi)))
        else:
            disagree.append(False)
    return {
        "bb": _pad(bb["frames"], n),
        "clone": _pad(cl["frames"], n),
        "disagree": disagree,
        "bb_steps": bb["steps"],
        "clone_steps": cl["steps"],
        "bb_reward": bb["total_reward"],
        "clone_reward": cl["total_reward"],
    }


def render_shield(demo_seed: int, max_frames: int) -> dict:
    return rollout(shielded_policy, demo_seed, render=True, max_frames=max_frames)


# ---------------------------------------------------------------------------
# demo seed: the twist must fire on screen, and v1 must visibly fail
# ---------------------------------------------------------------------------
def score_seed(seed: int, max_frames: int) -> dict:
    window = max_frames * FRAME_EVERY
    bb = rollout(bb_policy, seed, render=False, max_frames=max_frames)
    v1 = rollout(clone_v1, seed, render=False, max_frames=max_frames)
    fires = sum(1 for o in bb["obs"] if o[1] > TWIST_THRESHOLD)
    first_fire = next((i for i, o in enumerate(bb["obs"]) if o[1] > TWIST_THRESHOLD), None)
    disagree_v1 = sum(1 for o, a in zip(bb["obs"], bb["actions"]) if clone_v1(o) != a)
    return {
        "seed": seed,
        "bb_steps": bb["steps"],
        "v1_steps": v1["steps"],
        "fire_rate": fires / max(1, bb["steps"]),
        "first_fire": first_fire,
        "v1_disagree_rate": disagree_v1 / max(1, bb["steps"]),
        # good demo: bb fills the clip, the flaw fires in the first half, v1 falls early
        "ok": (
            bb["steps"] >= window
            and first_fire is not None
            and first_fire < window // 2
            and v1["steps"] < window
        ),
    }


def pick_demo_seed(max_frames: int, n_seeds: int = 60) -> tuple[int, list[dict]]:
    scans = [score_seed(s, max_frames) for s in range(n_seeds)]
    good = [s for s in scans if s["ok"]]
    if good:
        good.sort(key=lambda s: (-s["fire_rate"], s["first_fire"]))
        return good[0]["seed"], scans
    # fall back to the seed where the flaw fires earliest
    fallback = sorted(
        [s for s in scans if s["first_fire"] is not None] or scans,
        key=lambda s: (s["first_fire"] if s["first_fire"] is not None else 10**6, -s["bb_steps"]),
    )
    return fallback[0]["seed"], scans


# ---------------------------------------------------------------------------
# program versions
# ---------------------------------------------------------------------------
PROGRAM_V1 = '''def policy(obs, P):
    cart_x, cart_v, pole_angle, pole_ang_v = obs[0], obs[1], obs[2], obs[3]
    # r1: it pushes toward the side the pole is leaning
    if pole_angle > P["angle_thresh"]:
        return 1
    return 0
'''

PROGRAM_V2 = '''def policy(obs, P):
    cart_x, cart_v, pole_angle, pole_ang_v = obs[0], obs[1], obs[2], obs[3]
    # r1 + r5: the lean is judged together with how fast the pole is turning
    lean = pole_angle + P["ang_v_gain"] * pole_ang_v
    if lean > P["lean_thresh"]:
        return 1
    return 0
'''

PROGRAM_V3 = '''def policy(obs, P):
    cart_x, cart_v, pole_angle, pole_ang_v = obs[0], obs[1], obs[2], obs[3]
    # r4: above a cart-speed limit it always pushes left, whatever the pole is doing
    if cart_v > P["cart_v_limit"]:
        return 0
    # r1 + r5: otherwise the lean is judged with the pole's turn rate
    lean = pole_angle + P["ang_v_gain"] * pole_ang_v
    if lean > P["lean_thresh"]:
        return 1
    return 0
'''

PROGRAM_FIXED = '''def policy(obs, P):
    cart_x, cart_v, pole_angle, pole_ang_v = obs[0], obs[1], obs[2], obs[3]
    # r4 removed: the cart-speed override was the reported flaw
    # r1 + r5: the lean is judged with the pole's turn rate
    lean = pole_angle + P["ang_v_gain"] * pole_ang_v
    if lean > P["lean_thresh"]:
        return 1
    return 0
'''

PARAMS_V1 = {"angle_thresh": 0.004}
PARAMS_V2 = {"ang_v_gain": 0.41, "lean_thresh": 0.002}
PARAMS_V3 = {"cart_v_limit": 0.197, "ang_v_gain": 0.503, "lean_thresh": 0.001}
PARAMS_FIXED = {"ang_v_gain": 0.503, "lean_thresh": 0.001}

PROGRAMS = {1: (PROGRAM_V1, PARAMS_V1), 2: (PROGRAM_V2, PARAMS_V2), 3: (PROGRAM_V3, PARAMS_V3)}


def changed_lines(old: str, new: str) -> list[int]:
    """1-based line numbers in `new` that differ from `old` (what the UI glows)."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    out: list[int] = []
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out.extend(range(j1 + 1, j2 + 1))
    return out


def unified_diff(old: str, new: str, a: str = "recovered.py", b: str = "fixed.py") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=a, tofile=b, n=2,
        )
    )


# ---------------------------------------------------------------------------
# ledger snapshots (full snapshot every time, per contracts/events.py)
# ---------------------------------------------------------------------------
def entry(rule_id, text, status, confidence, evidence, version=None) -> dict:
    return {
        "rule_id": rule_id,
        "text": text,
        "status": status,
        "confidence": round(confidence, 3),
        "evidence": list(evidence),
        "program_version": version,
    }


R1 = "Pushes right while the pole leans right, left while it leans left"
R2 = "Ignores where the cart sits on the track"
R3 = "Decides on the pole's angle alone, ignoring how fast the pole turns"
R4 = "Above a cart-speed limit it pushes left no matter what the pole is doing"
R5 = "Weighs the pole's turn rate about half as much as its angle"

LEDGERS: dict[str, list[dict]] = {
    # turn 1: first hypotheses off the rollout data
    "t1": [
        entry("r1", R1, "observed", 0.92, ["q1", "q2", "q3"], 1),
        entry("r2", R2, "inferred", 0.61, ["q1", "q4"], 1),
        entry("r3", R3, "inferred", 0.55, ["q2", "q5"], 1),
    ],
    # turn 1 probes come back: r1 confirmed, r3 refuted by the sweep
    "t1_probed": [
        entry("r1", R1, "confirmed", 0.99, ["q1", "q2", "q3", "q6"], 1),
        entry("r2", R2, "inferred", 0.61, ["q1", "q4"], 1),
        entry("r3", R3, "refuted", 0.18, ["q2", "q5", "q7"], 1),
    ],
    # turn 2: the turn-rate rule replaces r3; the cart_v anomaly shows up as r4
    "t2": [
        entry("r1", R1, "confirmed", 0.99, ["q1", "q2", "q3", "q6"], 2),
        entry("r2", R2, "inferred", 0.68, ["q1", "q4"], 2),
        entry("r3", R3, "refuted", 0.18, ["q2", "q5", "q7"], 1),
        entry("r4", R4, "inferred", 0.44, ["q8", "q9"], 2),
        entry("r5", R5, "inferred", 0.71, ["q7", "q8"], 2),
    ],
    "t2_probed": [
        entry("r1", R1, "confirmed", 0.99, ["q1", "q2", "q3", "q6"], 2),
        entry("r2", R2, "confirmed", 0.95, ["q1", "q4", "q10"], 2),
        entry("r3", R3, "refuted", 0.18, ["q2", "q5", "q7"], 1),
        entry("r4", R4, "probing", 0.63, ["q8", "q9", "q11"], 2),
        entry("r5", R5, "confirmed", 0.97, ["q7", "q8", "q12"], 2),
    ],
    # turn 3: the 2-D sweep pins the cart-speed limit -> r4 confirmed
    "t3": [
        entry("r1", R1, "confirmed", 0.99, ["q1", "q2", "q3", "q6"], 3),
        entry("r2", R2, "confirmed", 0.95, ["q1", "q4", "q10"], 3),
        entry("r3", R3, "refuted", 0.18, ["q2", "q5", "q7"], 1),
        entry("r4", R4, "probing", 0.79, ["q8", "q9", "q11", "q13"], 3),
        entry("r5", R5, "confirmed", 0.97, ["q7", "q8", "q12"], 3),
    ],
    "t3_probed": [
        entry("r1", R1, "confirmed", 0.99, ["q1", "q2", "q3", "q6"], 3),
        entry("r2", R2, "confirmed", 0.95, ["q1", "q4", "q10"], 3),
        entry("r3", R3, "refuted", 0.18, ["q2", "q5", "q7"], 1),
        entry("r4", R4, "confirmed", 0.96, ["q8", "q9", "q11", "q13", "q14"], 3),
        entry("r5", R5, "confirmed", 0.97, ["q7", "q8", "q12"], 3),
    ],
}
LEDGERS["t4"] = LEDGERS["t3_probed"]


# ---------------------------------------------------------------------------
# snap events: SubQ Snap reads the whole raw query log, every turn
# ---------------------------------------------------------------------------
def snap_data(turn: int, n_rows: int, relevance: dict, anomaly: dict,
              next_probe: dict, rule_support: dict) -> dict:
    return {
        "turn": turn,
        "n_rows": n_rows,
        "input_tokens": int(n_rows * 54.5),
        "feature_relevance": {k: round(v, 3) for k, v in relevance.items()},
        "anomaly_feature": anomaly,
        "next_probe": next_probe,
        "rule_support": {k: round(v, 3) for k, v in rule_support.items()},
    }


def _choice(choice: str, probs: dict) -> dict:
    probs = {k: round(v, 3) for k, v in probs.items()}
    return {"choice": choice, "probabilities": probs, "confidence": probs[choice]}


# cart_v (the flaw's feature) climbs 0.21 -> 0.48 -> 0.79 -> 0.94 across the turns
SNAPS = [
    snap_data(
        1, 2416,
        {"cart_x": 0.112, "cart_v": 0.341, "pole_angle": 0.962, "pole_ang_v": 0.788},
        _choice("pole_ang_v", {"cart_x": 0.08, "cart_v": 0.21, "pole_angle": 0.17,
                               "pole_ang_v": 0.42, "none": 0.12}),
        _choice("pole_ang_v", {"cart_x": 0.11, "cart_v": 0.24, "pole_angle": 0.09,
                               "pole_ang_v": 0.56}),
        {"r1": 0.94, "r2": 0.72, "r3": 0.49},
    ),
    snap_data(
        2, 2968,
        {"cart_x": 0.126, "cart_v": 0.713, "pole_angle": 0.971, "pole_ang_v": 0.934},
        _choice("cart_v", {"cart_x": 0.06, "cart_v": 0.48, "pole_angle": 0.12,
                           "pole_ang_v": 0.29, "none": 0.05}),
        _choice("cart_v", {"cart_x": 0.08, "cart_v": 0.67, "pole_angle": 0.07,
                           "pole_ang_v": 0.18}),
        {"r1": 0.97, "r2": 0.81, "r3": 0.14, "r4": 0.58, "r5": 0.88},
    ),
    snap_data(
        3, 3518,
        {"cart_x": 0.119, "cart_v": 0.881, "pole_angle": 0.976, "pole_ang_v": 0.941},
        _choice("cart_v", {"cart_x": 0.04, "cart_v": 0.79, "pole_angle": 0.06,
                           "pole_ang_v": 0.09, "none": 0.02}),
        _choice("cart_v", {"cart_x": 0.07, "cart_v": 0.81, "pole_angle": 0.05,
                           "pole_ang_v": 0.07}),
        {"r1": 0.98, "r2": 0.92, "r3": 0.09, "r4": 0.84, "r5": 0.95},
    ),
    snap_data(
        4, 4012,
        {"cart_x": 0.108, "cart_v": 0.963, "pole_angle": 0.979, "pole_ang_v": 0.948},
        _choice("cart_v", {"cart_x": 0.02, "cart_v": 0.94, "pole_angle": 0.02,
                           "pole_ang_v": 0.01, "none": 0.01}),
        _choice("cart_x", {"cart_x": 0.52, "cart_v": 0.21, "pole_angle": 0.14,
                           "pole_ang_v": 0.13}),
        {"r1": 0.99, "r2": 0.94, "r3": 0.07, "r4": 0.96, "r5": 0.96},
    ),
]


# ---------------------------------------------------------------------------
# the event stream
# ---------------------------------------------------------------------------
def build_events(demo_seed: int, max_frames: int, *, with_frames: bool = True,
                 log=print) -> list[dict]:
    events: list[dict] = []

    def ev(etype: str, t: float, data: dict) -> None:
        events.append({"type": etype, "run_id": RUN_ID, "t": round(float(t), 3), "data": data})

    clips: dict[int, dict] = {}
    shield_clip: list[str] = []
    if with_frames:
        for version, clone in CLONES.items():
            log(f"  rendering clip for clone v{version} (seed {demo_seed}) ...")
            clips[version] = render_pair(clone, demo_seed, max_frames)
            c = clips[version]
            log(f"    bb {c['bb_steps']} steps / clone {c['clone_steps']} steps, "
                f"{sum(c['disagree'])}/{len(c['disagree'])} frames disagree")
        log("  rendering shielded black-box clip ...")
        shield_clip = render_shield(demo_seed, max_frames)["frames"]

    def frames_event(t: float, version: int) -> None:
        if not with_frames:
            return
        c = clips[version]
        ev("frames", t, {"version": version, "seed": demo_seed, "fps": CLIP_FPS,
                         "bb": c["bb"], "clone": c["clone"], "disagree": c["disagree"]})

    # --- run start -----------------------------------------------------------
    ev("run_started", 0.0, {"policy_label": POLICY_LABEL, "brain": BRAIN, "model": MODEL,
                            "budget": BUDGET, "live": True})
    ev("status", 0.4, {"text": "Rolling out the robot on seeds 0-7 to watch what it does"})
    ev("query_count", 3.1, {"n": 1248})
    ev("status", 3.4, {"text": "Holding out seeds 100-107 to score the clone honestly"})
    ev("query_count", 5.6, {"n": 2416})
    ev("status", 5.9, {"text": "Fitting a decision tree to 2,416 labelled states for pattern hints"})

    # --- turn 1 --------------------------------------------------------------
    ev("snap", 7.2, SNAPS[0])
    ev("status", 7.6, {"text": "SubQ Snap read 2,416 raw rows: the pole's angle and turn rate matter most"})
    ev("status", 9.8, {"text": "Muse: the pole's lean seems to set the push; proposing a first program"})
    ev("ledger", 10.4, {"entries": LEDGERS["t1"]})
    code1, params1 = PROGRAMS[1]
    ev("program", 11.9, {"version": 1, "code": code1, "params": params1,
                         "changed_lines": list(range(1, len(code1.splitlines()) + 1)),
                         "agreement": 0.404})
    ev("metric", 12.0, {"queries": 2416, "agreement": 0.404})
    frames_event(13.4, 1)
    ev("status", 14.1, {"text": "Probing r3: sweep pole_ang_v -1.2 -> 1.2 at pole_angle = 0"})
    ev("query_count", 16.3, {"n": 2568})
    ev("ledger", 16.6, {"entries": LEDGERS["t1_probed"]})
    ev("status", 16.9, {"text": "Probe verdict: r3 refuted -- the turn rate changes the push on its own"})
    ev("status", 17.4, {"text": "Hunting counterexamples around the states we have seen"})
    ev("query_count", 20.2, {"n": 2968})
    ev("status", 20.5, {"text": "8 counterexamples found; all of them have the cart moving right fast"})

    # --- turn 2 --------------------------------------------------------------
    ev("snap", 21.8, SNAPS[1])
    ev("status", 22.2, {"text": "SubQ Snap: counterexamples best separated by cart_v (p 0.48); probe cart_v next"})
    ev("status", 24.6, {"text": "Muse: adding the pole's turn rate, and a new rule r4 for the cart's speed"})
    ev("ledger", 25.1, {"entries": LEDGERS["t2"]})
    code2, params2 = PROGRAMS[2]
    ev("program", 26.7, {"version": 2, "code": code2, "params": params2,
                         "changed_lines": changed_lines(code1, code2), "agreement": 0.679})
    ev("metric", 26.8, {"queries": 2968, "agreement": 0.679})
    frames_event(28.1, 2)
    ev("status", 28.8, {"text": "Probing r2 and r5: sweep cart_x -2.0 -> 2.0, then pole_ang_v"})
    ev("query_count", 31.2, {"n": 3118})
    ev("ledger", 31.5, {"entries": LEDGERS["t2_probed"]})
    ev("status", 31.8, {"text": "Probe verdict: r2 confirmed, r5 confirmed, r4 still probing (0.63)"})
    ev("query_count", 34.4, {"n": 3518})

    # --- turn 3 --------------------------------------------------------------
    ev("snap", 35.6, SNAPS[2])
    ev("status", 36.0, {"text": "SubQ Snap: cart_v now p 0.79 for the anomaly; relevance 0.88"})
    ev("status", 38.3, {"text": "Muse: writing the cart-speed override into the program as r4"})
    ev("ledger", 38.8, {"entries": LEDGERS["t3"]})
    code3, params3 = PROGRAMS[3]
    ev("program", 40.5, {"version": 3, "code": code3, "params": params3,
                         "changed_lines": changed_lines(code2, code3), "agreement": 0.931})
    ev("metric", 40.6, {"queries": 3518, "agreement": 0.931})
    frames_event(41.9, 3)
    ev("status", 42.6, {"text": "Probing r4: 2-D sweep cart_v 0.0 -> 0.6 x pole_angle -0.2 -> 0.2"})
    ev("query_count", 45.1, {"n": 3652})
    ev("ledger", 45.4, {"entries": LEDGERS["t3_probed"]})
    ev("status", 45.7, {"text": "Probe verdict: r4 CONFIRMED -- the push flips left above cart_v 0.197"})
    ev("query_count", 48.0, {"n": 4012})

    # --- turn 4: nothing left to explain ------------------------------------
    ev("snap", 49.1, SNAPS[3])
    ev("status", 49.5, {"text": "SubQ Snap: every rule supported, cart_v anomaly p 0.94; nothing new to probe"})
    ev("status", 51.2, {"text": "Muse: done -- agreement stopped improving and all five rules are settled"})
    ev("ledger", 51.6, {"entries": LEDGERS["t4"]})
    ev("status", 52.0, {"text": "Scoring the clone closed-loop on 20 unseen seeds"})
    ev("metric", 54.8, {"queries": 4184, "agreement": 0.938})
    ev("query_count", 54.9, {"n": 4184})

    # --- reveal / fix / shield ----------------------------------------------
    ev("reveal", 56.2, {"label": REVEAL_LABEL, "feature": TWIST_FEATURE,
                        "highlight_rule_ids": ["r4"]})
    ev("status", 56.5, {"text": f"Engineers report: \"{REVEAL_LABEL}\" -- that is r4, already confirmed"})
    ev("fix", 58.0, {"code": PROGRAM_FIXED, "params": PARAMS_FIXED,
                     "diff": unified_diff(PROGRAM_V3, PROGRAM_FIXED)})
    ev("status", 58.3, {"text": "Fix: one rule removed, four lines changed, every other rule untouched"})
    if with_frames:
        ev("shield", 59.4, {"on": True, "bb": shield_clip})
        ev("status", 59.7, {"text": "Shield on: the robot's own output, corrected only where r4 fired"})

    ev("run_finished", 60.4, {
        "eval": {
            "open_loop_agreement": 0.938,
            "n_holdout_obs": 1168,
            "clone_mean_reward": 476.2,
            "bb_mean_reward": 491.8,
            "n_episodes": 20,
            "n_failures": 0,
            "failure_rate_bound_95": 0.15,
            "us_per_action_clone": 1.9,
            "us_per_action_bb": 214.0,
        },
        "total_queries": 4184,
        "turns": 4,
        "seconds": 60.4,
        "cached": False,
    })
    return events


# ---------------------------------------------------------------------------
def write_events(events: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the UI fixture event stream.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=None, help="demo seed (default: scan for one)")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument("--no-frames", action="store_true", help="skip rendering (tiny fixture)")
    p.add_argument("--list-seeds", action="store_true", help="print the demo-seed scan and exit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    log = (lambda *a, **k: None) if args.quiet else print

    if args.list_seeds:
        _, scans = pick_demo_seed(args.max_frames)
        print(f"{'seed':>5} {'bb_steps':>9} {'v1_steps':>9} {'fire%':>7} {'1st_fire':>9} {'v1_dis%':>8}  ok")
        for s in scans:
            ff = "-" if s["first_fire"] is None else s["first_fire"]
            print(f"{s['seed']:>5} {s['bb_steps']:>9} {s['v1_steps']:>9} "
                  f"{100*s['fire_rate']:>6.1f}% {ff:>9} {100*s['v1_disagree_rate']:>7.1f}%  "
                  f"{'yes' if s['ok'] else ''}")
        return 0

    if args.seed is None:
        log("scanning for a demo seed (flaw must fire on screen, v1 must visibly fail) ...")
        seed, _ = pick_demo_seed(args.max_frames)
        log(f"  demo seed = {seed}")
    else:
        seed = args.seed

    events = build_events(seed, args.max_frames, with_frames=not args.no_frames, log=log)
    out = write_events(events, args.out)
    size_mb = out.stat().st_size / 1e6
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    log(f"\nwrote {out.relative_to(ROOT)}  ({len(events)} events, {size_mb:.2f} MB, "
        f"{events[-1]['t']:.1f}s of run time, demo seed {seed})")
    log("  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
