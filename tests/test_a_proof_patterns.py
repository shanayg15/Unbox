"""A4: pattern_hints turns labelled states into text the brain can read."""
import random
import re

from contracts.types import SensorSpec
from numerics.patterns import pattern_hints
from tests.test_a_proof_helpers import ARM_SPEC, CARTPOLE_SPEC, LUNAR_SPEC


def cart_data(n=1000):
    rng = random.Random(3)
    X = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(n)]
    return X, [1 if o[3] > 0.25 else 0 for o in X]


def leaf_counts(text: str) -> list[int]:
    return [int(v) for v in re.findall(r"n=(\d+)", text)]


def test_tree_finds_the_split_in_named_terms():
    X, y = cart_data()
    text = pattern_hints(X, y, CARTPOLE_SPEC)
    assert "pole_ang_v" in text and "push_right" in text and "push_left" in text
    split = float(re.search(r"pole_ang_v <= (-?[\d.]+)", text).group(1))
    assert abs(split - 0.25) < 0.05


def test_every_leaf_reports_count_and_accuracy_and_counts_add_up():
    X, y = cart_data()
    text = pattern_hints(X, y, CARTPOLE_SPEC)
    assert sum(leaf_counts(text)) == 1000
    assert "acc=100%" in text


def test_impure_leaf_reports_its_real_accuracy():
    # 100 identical states, 70/30 labels: no split possible, a single 70% leaf.
    X, y = [[0.0, 0.0, 0.0, 0.0]] * 100, [1] * 70 + [0] * 30
    text = pattern_hints(X, y, CARTPOLE_SPEC)
    assert "push_right" in text and "n=100" in text and "acc=70%" in text


def test_action_names_stay_right_when_some_actions_never_appear():
    # Only actions 0 and 2 occur; a tree's class index 1 must still read as fire_main, not fire_left.
    rng = random.Random(5)
    X = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(400)]
    y = [2 if o[1] > 0 else 0 for o in X]
    text = pattern_hints(X, y, LUNAR_SPEC)
    assert "fire_main" in text and "noop" in text
    assert "fire_left" not in text and "fire_right" not in text


def test_max_depth_is_respected():
    rng = random.Random(6)
    X = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(2000)]
    y = [int(sum(o) * 7) % 2 for o in X]
    text = pattern_hints(X, y, CARTPOLE_SPEC, max_depth=2)
    depths = [line.count("|") for line in text.splitlines() if "n=" in line]
    assert depths and max(depths) <= 3


def test_too_little_data_for_a_split_still_reports():
    assert "n=5" in pattern_hints([[0.1, 0.2, 0.3, 0.4]] * 5, [0] * 5, CARTPOLE_SPEC)


def test_no_data_says_so():
    assert "no" in pattern_hints([], [], CARTPOLE_SPEC).lower()


def test_continuous_gets_one_tree_per_action_dim():
    rng = random.Random(7)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(500)]
    y = [[1.5 if o[0] > 0 else -1.5, 0.8 if o[1] > 0 else 0.1] for o in X]
    text = pattern_hints(X, y, ARM_SPEC)
    assert "torque" in text and "brake" in text
    assert "error <=" in text and "error_rate <=" in text
    assert "mean=1.5" in text and "mean=-1.5" in text
