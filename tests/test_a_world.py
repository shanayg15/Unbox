"""A-World: black-box server, twists and rendering, exercised over real HTTP.

The server runs as `uvicorn blackbox.server:app`, the same command the demo uses,
on an ephemeral port so it never collides with a dev server on :8001.
"""
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from PIL import Image

from blackbox.client import HttpBlackBox
from blackbox.policies import REGISTRY, rollout
from blackbox.twists import TWISTS, find_demo_seeds, report_rows, twists_for
from contracts.types import Episode, QueryBatch, SensorSpec

ROOT = Path(__file__).resolve().parent.parent
ADMIN_TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": ADMIN_TOKEN}
CARTPOLE_OBS = [0.01, 0.02, 0.03, 0.04]
ENV_IDS = ["CartPole", "LunarLander", "MountainCar", "Acrobot", "PPO", "DQN", "sb3/"]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def url():
    port = free_port()
    env = {**os.environ, "BLACKBOX_ADMIN_TOKEN": ADMIN_TOKEN}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "blackbox.server:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.25)
        else:
            raise RuntimeError("black box server never came up")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def http(url):
    with httpx.Client(base_url=url, timeout=120) as client:
        yield client


# --- A1: server skeleton + CartPole -----------------------------------------


def test_act_returns_one_action_per_obs(http):
    resp = http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS] * 256, "source": "probe"})
    resp.raise_for_status()
    body = resp.json()
    print(f"\n/act 256 obs -> {len(body['actions'])} actions, {body['query_id']}, {body['inference_us']:.1f} us/obs")
    assert len(body["actions"]) == 256
    assert set(body["actions"]) <= {0, 1}
    assert body["query_id"].startswith("q")
    assert body["inference_us"] > 0


def test_act_increments_counter_by_batch_size(http):
    before = http.get("/count", params={"bb": "bb-01"}).json()["queries"]
    http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS] * 7}).raise_for_status()
    assert http.get("/count", params={"bb": "bb-01"}).json()["queries"] == before + 7


def test_act_query_ids_are_distinct(http):
    first = http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS]}).json()["query_id"]
    second = http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS]}).json()["query_id"]
    assert first != second


def test_act_rejects_wrong_obs_width_without_leaking_env(http):
    resp = http.post("/act", json={"bb": "bb-01", "obs": [[1.0, 2.0]]})
    assert resp.status_code == 422
    assert "4" in resp.text
    assert not any(name in resp.text for name in ENV_IDS)


def test_unknown_bb_is_404(http):
    assert http.get("/spec", params={"bb": "bb-99"}).status_code == 404
    assert http.post("/act", json={"bb": "bb-99", "obs": [CARTPOLE_OBS]}).status_code == 404


def test_episode_on_cartpole_seed_0_scores_about_500(http):
    resp = http.post("/episode", json={"bb": "bb-01", "seed": 0})
    resp.raise_for_status()
    body = resp.json()
    print(f"\n/episode bb-01 seed 0 -> reward {body['total_reward']}, {len(body['actions'])} steps")
    assert body["total_reward"] == pytest.approx(500, abs=5)
    assert len(body["obs"]) == len(body["actions"]) == len(body["rewards"])
    assert body["frames"] is None


def test_rendered_frames_decode_as_jpegs(http):
    resp = http.post("/episode", json={"bb": "bb-01", "seed": 0, "render": True})
    resp.raise_for_status()
    body = resp.json()
    images = [Image.open(io.BytesIO(base64.b64decode(frame))) for frame in body["frames"]]
    for image in images:
        image.verify()
    print(f"\n/episode render -> {len(images)} frames, {images[0].format} {images[0].size}")
    assert images[0].format == "JPEG"
    assert images[0].size[0] == 360
    assert body["frame_every"] == 3
    assert len(images) == (len(body["obs"]) + 2) // 3


def test_rendered_and_plain_episodes_match(http):
    plain = http.post("/episode", json={"bb": "bb-01", "seed": 1}).json()
    rendered = http.post("/episode", json={"bb": "bb-01", "seed": 1, "render": True}).json()
    assert plain["actions"] == rendered["actions"]
    assert plain["total_reward"] == rendered["total_reward"]


def test_explicit_frame_every_is_honored_after_a_cached_default(http):
    default = http.post("/episode", json={"bb": "bb-01", "seed": 2, "render": True}).json()
    explicit = http.post("/episode", json={"bb": "bb-01", "seed": 2, "render": True, "frame_every": 10}).json()
    assert default["frame_every"] == 3
    assert explicit["frame_every"] == 10
    assert len(explicit["frames"]) == (len(explicit["obs"]) + 9) // 10


def test_spec_is_finite_named_and_opaque(http):
    resp = http.get("/spec", params={"bb": "bb-01"})
    resp.raise_for_status()
    body = resp.json()
    assert body["obs_names"] == ["cart_x", "cart_v", "pole_angle", "pole_ang_v"]
    assert body["action_names"] == ["push_left", "push_right"]
    assert body["action_type"] == "discrete"
    assert all(abs(v) < 100 for v in body["obs_low"] + body["obs_high"])
    assert all(lo < hi for lo, hi in zip(body["obs_low"], body["obs_high"]))
    assert not any(name in json.dumps(body) for name in ENV_IDS)


def test_http_client_returns_contract_types(url):
    bb = HttpBlackBox("bb-01", url)
    spec = bb.spec()
    assert isinstance(spec, SensorSpec)
    before = bb.query_count()
    batch = bb.act([CARTPOLE_OBS] * 3, source="prober", rule_id="r2")
    assert isinstance(batch, QueryBatch)
    assert batch.obs == [CARTPOLE_OBS] * 3
    assert batch.source == "prober"
    assert batch.rule_id == "r2"
    assert len(batch.actions) == 3
    assert bb.query_count() == before + 3
    episode = bb.episode(0, render=True)
    assert isinstance(episode, Episode)
    assert episode.total_reward == pytest.approx(500, abs=5)
    assert episode.frames


def test_client_module_has_no_admin_calls_or_server_imports():
    source = (ROOT / "blackbox" / "client.py").read_text()
    assert "/admin" not in source
    assert "policies" not in source
    assert "twists" not in source


# --- A2: hidden flaws + admin endpoints -------------------------------------


@pytest.fixture()
def untwisted(http):
    """Leave every policy clean for the next test, whatever this one configured."""
    yield
    for bb in ("bb-01", "bb-02", "bb-03", "bb-04", "bb-05", "bb-06"):
        http.post("/admin/config", json={"bb": bb, "twist_id": None}, headers=ADMIN).raise_for_status()


def configure(http, bb, twist_id):
    resp = http.post("/admin/config", json={"bb": bb, "twist_id": twist_id}, headers=ADMIN)
    resp.raise_for_status()


def test_admin_endpoints_reject_a_missing_or_wrong_token(http):
    calls = [
        ("GET", "/admin/policies", None),
        ("GET", "/admin/twists?bb=bb-01", None),
        ("POST", "/admin/config", {"bb": "bb-01", "twist_id": None}),
        ("GET", "/admin/reveal?bb=bb-01", None),
    ]
    for method, path, body in calls:
        assert http.request(method, path, json=body).status_code == 401, path
        assert http.request(method, path, json=body, headers={"X-Admin-Token": "nope"}).status_code == 401, path


def test_admin_policies_lists_every_black_box(http):
    resp = http.get("/admin/policies", headers=ADMIN)
    resp.raise_for_status()
    rows = {row["bb"]: row for row in resp.json()}
    assert sorted(rows) == ["bb-01", "bb-02", "bb-03", "bb-04", "bb-05", "bb-06"]
    assert rows["bb-05"] == {"bb": "bb-05", "label": "LunarLander-PPO", "env_id": "LunarLander-v3"}


def test_admin_twists_returns_hidden_labels_only(http):
    resp = http.get("/admin/twists", params={"bb": "bb-01"}, headers=ADMIN)
    resp.raise_for_status()
    assert resp.json() == [
        {"twist_id": "t-cp-1", "hidden_label": "Hidden flaw #1"},
        {"twist_id": "t-cp-2", "hidden_label": "Hidden flaw #2"},
    ]


def test_config_rejects_a_twist_from_another_env(http):
    assert http.post("/admin/config", json={"bb": "bb-01", "twist_id": "t-ll-1"}, headers=ADMIN).status_code == 422


def test_config_resets_the_counter_and_query_ids(http, untwisted):
    http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS] * 5}).raise_for_status()
    configure(http, "bb-01", "t-cp-1")
    assert http.get("/count", params={"bb": "bb-01"}).json()["queries"] == 0
    assert http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS]}).json()["query_id"] == "q1"


def test_active_twist_overrides_actions_only_where_its_condition_holds(http, untwisted):
    twist = TWISTS["t-cp-1"]
    grid = [[0.0, v / 10, angle / 100, ang_v / 10] for v in range(-20, 21) for angle in (-5, 5) for ang_v in (-5, 5)]
    configure(http, "bb-01", None)
    clean = http.post("/act", json={"bb": "bb-01", "obs": grid}).json()["actions"]
    configure(http, "bb-01", "t-cp-1")
    twisted = http.post("/act", json={"bb": "bb-01", "obs": grid}).json()["actions"]
    changed = 0
    for row, before, after in zip(grid, clean, twisted):
        if twist.condition(row):
            assert after == 0
            changed += before != after
        else:
            assert after == before
    assert changed > 0


def test_episodes_change_under_a_twist_and_cache_per_twist(http, untwisted):
    configure(http, "bb-01", None)
    clean = http.post("/episode", json={"bb": "bb-01", "seed": 1}).json()
    configure(http, "bb-01", "t-cp-1")
    twisted = http.post("/episode", json={"bb": "bb-01", "seed": 1}).json()
    assert twisted["actions"] != clean["actions"]
    assert twisted["total_reward"] < clean["total_reward"]
    configure(http, "bb-01", None)
    assert http.post("/episode", json={"bb": "bb-01", "seed": 1}).json()["actions"] == clean["actions"]


def test_reveal_describes_the_active_twist(http, untwisted):
    configure(http, "bb-05", "t-ll-1")
    resp = http.get("/admin/reveal", params={"bb": "bb-05"}, headers=ADMIN)
    resp.raise_for_status()
    body = resp.json()
    assert body["twist_id"] == "t-ll-1"
    assert body["reveal_label"] == "Skips its safety action on the right side"
    assert body["feature"] == "x"
    assert "x >" in body["rule_text"]
    configure(http, "bb-05", None)
    assert http.get("/admin/reveal", params={"bb": "bb-05"}, headers=ADMIN).json()["twist_id"] is None


def test_public_responses_never_mention_the_active_twist(http, untwisted):
    configure(http, "bb-01", "t-cp-1")
    payloads = [
        http.get("/spec", params={"bb": "bb-01"}).text,
        http.post("/act", json={"bb": "bb-01", "obs": [CARTPOLE_OBS]}).text,
        http.get("/count", params={"bb": "bb-01"}).text,
    ]
    for text in payloads:
        assert "t-cp" not in text
        assert "flaw" not in text.lower()


@pytest.fixture(scope="module")
def report():
    return report_rows()


# No threshold puts these in the band; each still fires and costs reward. Everything else must hit it.
#   t-cp-1/bb-01: PPO already pushes left in most of that region, so the override rarely changes it
#   t-cp-1/bb-02: same threshold, but the DQN sways far more, so it fires more
KNOWN_OFF_TARGET = {("t-cp-1", "bb-01"), ("t-cp-1", "bb-02"),
                    ("t-mc-1", "bb-03")}   # strengthened on purpose: the robot must visibly get stuck


def test_twists_fire_on_5_to_30_percent_of_steps(report):
    off_target = {(r.twist_id, r.bb) for r in report if not 0.05 <= r.fire_rate <= 0.30}
    assert off_target == KNOWN_OFF_TARGET
    assert all(r.fire_rate > 0.01 for r in report)


def test_hero_twist_hits_the_fire_target_on_the_demo_policy(report):
    hero = next(r for r in report if (r.twist_id, r.bb) == ("t-ll-1", "bb-05"))
    assert 0.05 <= hero.fire_rate <= 0.30


def test_twists_cost_visible_reward(report):
    no_drop = [f"{r.twist_id}/{r.bb}: {r.clean_reward:.0f} -> {r.twisted_reward:.0f}"
               for r in report if r.twisted_reward >= r.clean_reward - 5]
    assert not no_drop, no_drop


def test_actions_differ_from_clean_only_where_the_condition_holds(report):
    assert all(r.violations == 0 for r in report)
    assert all(r.compared > 0 for r in report)


# --- A3: remaining policies + rendering -------------------------------------

ALL_BBS = sorted(REGISTRY)


@pytest.mark.parametrize("bb", ALL_BBS)
def test_every_policy_serves_spec_act_and_a_rendered_episode(http, untwisted, bb):
    defn = REGISTRY[bb]
    spec = http.get("/spec", params={"bb": bb}).raise_for_status().json()
    assert spec["obs_names"] == list(defn.obs_names)
    assert spec["action_names"] == list(defn.action_names)
    assert not any(name in json.dumps(spec) for name in ENV_IDS)

    midpoint = [(lo + hi) / 2 for lo, hi in zip(spec["obs_low"], spec["obs_high"])]
    actions = http.post("/act", json={"bb": bb, "obs": [midpoint] * 16}).raise_for_status().json()["actions"]
    assert len(actions) == 16
    assert all(0 <= a < len(defn.action_names) for a in actions)

    started = time.perf_counter()
    body = http.post("/episode", json={"bb": bb, "seed": 0, "render": True}).raise_for_status().json()
    elapsed = time.perf_counter() - started
    assert len(body["obs"][0]) == len(defn.obs_names)
    images = [Image.open(io.BytesIO(base64.b64decode(frame))) for frame in body["frames"]]
    assert len(images) == (len(body["obs"]) + 2) // 3
    assert all(image.format == "JPEG" and image.size[0] == 360 for image in images)
    kb = sum(len(frame) for frame in body["frames"]) / 1024
    print(f"\n{bb} {defn.label:16} reward {body['total_reward']:7.1f}  {len(body['obs']):4d} steps  "
          f"{len(images):3d} frames {images[0].size}  {kb:6.0f} KB  {elapsed:.2f}s")


@pytest.mark.parametrize("bb", ALL_BBS)
def test_twists_for_each_env_apply_over_http(http, untwisted, bb):
    states = [row for seed in range(3) for row in rollout(bb, seed).obs][:800]
    configure(http, bb, None)
    clean = http.post("/act", json={"bb": bb, "obs": states}).raise_for_status().json()["actions"]
    for twist in twists_for(bb):
        configure(http, bb, twist.twist_id)
        twisted = http.post("/act", json={"bb": bb, "obs": states}).raise_for_status().json()["actions"]
        for row, before, after in zip(states, clean, twisted):
            assert after == twist.apply(row, before)


def test_only_acrobot_has_no_twists():
    assert [bb for bb in ALL_BBS if not twists_for(bb)] == ["bb-04"]


@pytest.mark.parametrize("bb", ALL_BBS)
def test_mean_reward_is_near_the_model_card(bb):
    defn = REGISTRY[bb]
    mean = sum(rollout(bb, seed).total_reward for seed in range(5)) / 5
    assert abs(mean - defn.card_reward) <= max(2 * defn.card_std, 0.02 * abs(defn.card_reward))


@pytest.mark.parametrize("bb,twist_id", [("bb-05", None), ("bb-05", "t-ll-1"), ("bb-06", None), ("bb-03", None), ("bb-04", None)])
def test_rendering_does_not_change_the_trajectory(bb, twist_id):
    twist = TWISTS[twist_id] if twist_id else None
    for seed in range(3):
        plain = rollout(bb, seed, twist=twist)
        rendered = rollout(bb, seed, twist=twist, render=True)
        assert plain.actions == rendered.actions
        assert plain.total_reward == rendered.total_reward


def test_concurrent_requests_stay_deterministic(http, url, untwisted):
    def fetch(job):
        bb, seed = job
        with httpx.Client(base_url=url, timeout=120) as client:
            body = client.post("/episode", json={"bb": bb, "seed": seed + 500, "render": True}).raise_for_status().json()
            client.post("/act", json={"bb": bb, "obs": body["obs"][:64]}).raise_for_status()
            return job, body["actions"]

    jobs = [(bb, seed) for bb in ALL_BBS for seed in range(2)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, jobs))
    for (bb, seed), actions in results:
        assert actions == rollout(bb, seed + 500).actions


# --- Demo-seed finder -------------------------------------------------------


@pytest.fixture(scope="module")
def hero_seeds():
    return find_demo_seeds("bb-05", "t-ll-1", seeds=range(60))


def test_demo_seeds_meet_the_brief_when_replayed_independently(hero_seeds):
    assert hero_seeds
    twist = TWISTS["t-ll-1"]
    for found in hero_seeds:
        clean = rollout("bb-05", found.seed)
        twisted = rollout("bb-05", found.seed, twist=twist)
        assert clean.total_reward > 200 and clean.rewards[-1] == 100, "clean policy must land"
        assert twisted.rewards[-1] == -100, "twisted policy must crash"
        first_change = next(i for i, (a, b) in enumerate(zip(clean.actions, twisted.actions)) if a != b)
        assert first_change == found.first_fire
        assert first_change < len(twisted.actions) / 2, "twist must fire in the first half"
        assert (found.clean_reward, found.twisted_reward) == (clean.total_reward, twisted.total_reward)


def test_demo_seeds_are_ranked_right_side_crashes_first_then_biggest_gap(hero_seeds):
    keys = [(not s.in_zone, -(s.clean_reward - s.twisted_reward)) for s in hero_seeds]
    assert keys == sorted(keys)


def test_best_demo_seed_replays_identically_over_http_with_frames(http, untwisted, hero_seeds):
    best = hero_seeds[0]
    configure(http, "bb-05", None)
    clean = http.post("/episode", json={"bb": "bb-05", "seed": best.seed, "render": True}).raise_for_status().json()
    configure(http, "bb-05", "t-ll-1")
    twisted = http.post("/episode", json={"bb": "bb-05", "seed": best.seed, "render": True}).raise_for_status().json()
    assert clean["total_reward"] == best.clean_reward
    assert twisted["total_reward"] == best.twisted_reward
    assert clean["frames"] and twisted["frames"]


def test_demo_seed_finder_rejects_a_twist_from_another_env():
    with pytest.raises(ValueError):
        find_demo_seeds("bb-05", "t-cp-1", seeds=range(1))
