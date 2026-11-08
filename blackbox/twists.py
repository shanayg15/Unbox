"""Lane A, task A2: hidden flaws (twists), applied server-side after the policy acts.

`python -m blackbox.twists --report` prints each twist's fire rate and reward cost on seeds 0-9.
`python -m blackbox.twists --demo-seed bb-05 t-ll-1` finds the seeds the demo can use.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

from contracts.types import Episode

from blackbox.policies import REGISTRY, get_policy, rollout

CARTPOLE = "CartPole-v1"
LUNARLANDER = "LunarLander-v3"
MOUNTAINCAR = "MountainCar-v0"
REPORT_SEEDS = range(10)
FIRE_TARGET = (0.05, 0.30)
DEMO_SEEDS = range(201)
LANDED, CRASHED = 100, -100   # LunarLander's final-step reward for each ending
SUCCESS_REWARD = {LUNARLANDER: 200, CARTPOLE: 475, MOUNTAINCAR: -199}

Obs = list[float]


@dataclass(frozen=True)
class Twist:
    twist_id: str
    env: str
    hidden_label: str         # all the audience sees before the reveal
    reveal_label: str         # factory language, shown on reveal
    feature: str              # sensor name the twist turns on
    threshold: float
    rule_text: str
    condition: Callable[[Obs], bool]
    override: Callable[[Obs, int], int]

    def apply(self, obs: Obs, action: int) -> int:
        return self.override(obs, action) if self.condition(obs) else action


def _make() -> dict[str, Twist]:
    # Thresholds tuned on seeds 0-9 with --report. These policies hold their set-point tightly
    # (x stays near 0, cart_v under 0.3), so the numbers are smaller than first guesses.
    # LunarLander: x=0 y=1 vy=3 angle=4 | fire: noop=0 left=1 main=2 right=3
    t_ll_1, t_ll_2, t_ll_3 = 0.04, 0.04, 1.0
    t_ll_3_vy = -0.3
    # CartPole: cart_x=0 cart_v=1 | push_left=0 push_right=1
    t_cp_1, t_cp_2 = 0.03, -0.08
    # MountainCar: position=0 | noop=1
    t_mc_1 = -0.5    # strong enough that the robot visibly fails (stuck in the left corner)
    twists = [
        Twist("t-ll-1", LUNARLANDER, "Hidden flaw #1", "Skips its safety action on the right side",
              # Every action goes to noop past the line, whatever the policy chose: the engines cut out
              "x", t_ll_1, f"x > {t_ll_1} -> noop",
              lambda o: o[0] > t_ll_1, lambda o, a: 0),
        Twist("t-ll-2", LUNARLANDER, "Hidden flaw #2", "Over-corrects when it tilts",
              "angle", t_ll_2, f"angle > {t_ll_2} -> fire_left",
              lambda o: o[4] > t_ll_2, lambda o, a: 1),
        Twist("t-ll-3", LUNARLANDER, "Hidden flaw #3", "Ignores its speed limit up high",
              "y", t_ll_3, f"y > {t_ll_3} and vy < {t_ll_3_vy} -> noop",
              lambda o: o[1] > t_ll_3 and o[3] < t_ll_3_vy, lambda o, a: 0),
        Twist("t-cp-1", CARTPOLE, "Hidden flaw #1", "Over-reacts when moving fast",
              "cart_v", t_cp_1, f"cart_v > {t_cp_1} -> push_left",
              lambda o: o[1] > t_cp_1, lambda o, a: 0),
        Twist("t-cp-2", CARTPOLE, "Hidden flaw #2", "Drifts in one zone of the track",
              "cart_x", t_cp_2, f"cart_x < {t_cp_2} -> push_right",
              lambda o: o[0] < t_cp_2, lambda o, a: 1),
        Twist("t-mc-1", MOUNTAINCAR, "Hidden flaw #1", "Freezes in one corner of its workspace",
              "position", t_mc_1, f"position < {t_mc_1} -> noop",
              lambda o: o[0] < t_mc_1, lambda o, a: 1),
    ]
    return {t.twist_id: t for t in twists}


TWISTS: dict[str, Twist] = _make()


def twists_for(bb: str) -> list[Twist]:
    env = REGISTRY[bb].env_id
    return [t for t in TWISTS.values() if t.env == env]


@dataclass(frozen=True)
class ReportRow:
    twist_id: str
    bb: str
    rule_text: str
    fire_rate: float          # share of steps where the twist changed the action
    cond_rate: float          # share of steps where its condition held
    clean_reward: float
    twisted_reward: float
    compared: int             # steps checked against the clean policy
    violations: int           # steps where the action changed although the condition was false


def _clean_actions(bb: str, episode: Episode) -> list[int]:
    """What the clean policy does in each state of an episode.

    One row at a time, as the rollout does: on a near-tie a DQN's argmax can flip with batch
    size (rounding only, about 1 step in 5,000), which is not the twist's doing.
    """
    policy = get_policy(bb)
    return [int(policy.predict(np.asarray(obs, dtype=np.float32)[None])[0]) for obs in episode.obs]


def _report_row(twist: Twist, bb: str, seeds: range) -> ReportRow:
    clean_rewards, twisted_rewards = [], []
    steps = fired = held = violations = 0
    for seed in seeds:
        clean_rewards.append(rollout(bb, seed).total_reward)
        episode = rollout(bb, seed, twist=twist)
        twisted_rewards.append(episode.total_reward)
        for obs, twisted_action, clean_action in zip(episode.obs, episode.actions, _clean_actions(bb, episode)):
            steps += 1
            holds = twist.condition(obs)
            held += holds
            if twisted_action != clean_action:
                fired += 1
                violations += not holds
    return ReportRow(twist.twist_id, bb, twist.rule_text, fired / steps, held / steps,
                     float(np.mean(clean_rewards)), float(np.mean(twisted_rewards)), steps, violations)


def report_rows(seeds: range = REPORT_SEEDS) -> list[ReportRow]:
    return [
        _report_row(twist, bb, seeds)
        for twist in TWISTS.values()
        for bb, defn in REGISTRY.items()
        if defn.env_id == twist.env
    ]


def _target(fire_rate: float) -> str:
    low, high = FIRE_TARGET
    return "LOW" if fire_rate < low else "HIGH" if fire_rate > high else "ok"


def print_report(rows: list[ReportRow]) -> None:
    print(f"{'twist':7} {'bb':6} {'fire':>6} {'target':>6} {'cond':>6} {'clean':>8} {'twisted':>8} {'drop':>8} "
          f"{'steps':>6} {'viol':>5}  rule")
    for r in rows:
        print(f"{r.twist_id:7} {r.bb:6} {r.fire_rate:6.1%} {_target(r.fire_rate):>6} {r.cond_rate:6.1%} "
              f"{r.clean_reward:8.1f} {r.twisted_reward:8.1f} {r.clean_reward - r.twisted_reward:8.1f} "
              f"{r.compared:6d} {r.violations:5d}  {r.rule_text}")


@dataclass(frozen=True)
class DemoSeed:
    seed: int
    clean_reward: float
    clean_steps: int
    twisted_reward: float
    twisted_steps: int
    first_fire: int           # first step where the twist changed the action
    fires: int
    end_obs: list[float]      # where the twisted episode ended
    in_zone: bool             # it ended inside the twist's own region (t-ll-1: on the right side)


def _clean_succeeds(env: str, episode: Episode) -> bool:
    if episode.total_reward <= SUCCESS_REWARD[env]:
        return False
    return env != LUNARLANDER or episode.rewards[-1] == LANDED


def _twisted_fails(env: str, episode: Episode) -> bool:
    if env == LUNARLANDER:
        return episode.rewards[-1] == CRASHED
    return episode.total_reward <= SUCCESS_REWARD[env]


def find_demo_seeds(bb: str, twist_id: str, seeds: range = DEMO_SEEDS) -> list[DemoSeed]:
    """Seeds where the clean policy succeeds, the twisted one fails, and the twist fires in the
    first half of the episode. Ranked: failures inside the twist's region first, then biggest gap."""
    twist = TWISTS[twist_id]
    env = REGISTRY[bb].env_id
    if twist.env != env:
        raise ValueError(f"twist {twist_id} does not apply to {bb}")
    found = []
    for seed in seeds:
        clean = rollout(bb, seed)
        if not _clean_succeeds(env, clean):
            continue
        twisted = rollout(bb, seed, twist=twist)
        if not _twisted_fails(env, twisted):
            continue
        fired = [t for t, (a, b) in enumerate(zip(twisted.actions, _clean_actions(bb, twisted))) if a != b]
        if not fired or fired[0] >= len(twisted.actions) / 2:
            continue
        found.append(DemoSeed(seed, clean.total_reward, len(clean.actions), twisted.total_reward,
                              len(twisted.actions), fired[0], len(fired), twisted.obs[-1],
                              twist.condition(twisted.obs[-1])))
    return sorted(found, key=lambda s: (not s.in_zone, s.twisted_reward - s.clean_reward))


def print_demo_seeds(bb: str, twist_id: str, top: int) -> None:
    found = find_demo_seeds(bb, twist_id)
    twist = TWISTS[twist_id]
    feature = REGISTRY[bb].obs_names.index(twist.feature)
    print(f"{bb} + {twist_id} ({twist.rule_text}): {len(found)} of {len(DEMO_SEEDS)} seeds qualify, best {top}:")
    print(f"{'seed':>5} {'clean':>8} {'steps':>6} {'twisted':>8} {'steps':>6} {'gap':>7} {'1st fire':>8} "
          f"{'fires':>6} {'end ' + twist.feature:>10} {'in zone':>8}")
    for s in found[:top]:
        print(f"{s.seed:5d} {s.clean_reward:8.1f} {s.clean_steps:6d} {s.twisted_reward:8.1f} {s.twisted_steps:6d} "
              f"{s.clean_reward - s.twisted_reward:7.1f} {s.first_fire:8d} {s.fires:6d} "
              f"{s.end_obs[feature]:10.3f} {'yes' if s.in_zone else 'no':>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hidden-flaw tooling")
    parser.add_argument("--report", action="store_true", help="fire rate and reward cost per twist, seeds 0-9")
    parser.add_argument("--demo-seed", nargs=2, metavar=("BB", "TWIST"), help="best demo seeds among 0-200")
    parser.add_argument("--top", type=int, default=5, help="how many demo seeds to print")
    args = parser.parse_args()
    if args.report:
        print_report(report_rows())
    elif args.demo_seed:
        print_demo_seeds(args.demo_seed[0], args.demo_seed[1], args.top)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
