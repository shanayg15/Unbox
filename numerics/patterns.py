"""Lane A, task A4: decision-tree pattern hints"""
import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from contracts.types import Action, Obs, SensorSpec

MIN_SAMPLES_LEAF = 20
MAX_REGRESSOR_DEPTH = 3


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.1f}".rstrip("0").rstrip(".") + "%"


def _render(tree, obs_names: list[str], leaf_text) -> list[str]:
    """Same layout as sklearn's export_text, with our own leaf text carrying counts and accuracy."""
    t = tree.tree_
    lines: list[str] = []

    def walk(node: int, depth: int) -> None:
        indent = "|   " * depth + "|--- "
        if t.children_left[node] == t.children_right[node]:     # sklearn's marker for a leaf
            lines.append(indent + leaf_text(node))
            return
        name, threshold = obs_names[t.feature[node]], round(float(t.threshold[node]), 3)
        lines.append(f"{indent}{name} <= {threshold}")
        walk(t.children_left[node], depth + 1)
        lines.append(f"{indent}{name} >  {threshold}")
        walk(t.children_right[node], depth + 1)

    walk(0, 0)
    return lines


def _classifier_hints(X: np.ndarray, y: list[Action], spec: SensorSpec, max_depth: int) -> str:
    labels = np.asarray(y, dtype=int)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0)
    clf.fit(X, labels)
    t = clf.tree_

    def leaf_text(node: int) -> str:
        # tree_.value holds fractions on some sklearn versions and counts on others; the ratio is the same.
        weights = t.value[node][0]
        # clf.classes_ holds only the actions that occurred, so an index is not an action id until mapped.
        action = int(clf.classes_[int(weights.argmax())])
        name = spec.action_names[action] if 0 <= action < len(spec.action_names) else str(action)
        return f"{name} (n={int(t.n_node_samples[node])}, acc={_percent(weights.max() / weights.sum())})"

    header = f"Decision tree on {len(labels)} states, overall accuracy {_percent(clf.score(X, labels))}:"
    return "\n".join([header, *_render(clf, spec.obs_names, leaf_text)])


def _regressor_hints(X: np.ndarray, y: list[Action], spec: SensorSpec, max_depth: int) -> str:
    targets = np.asarray(y, dtype=float)
    sections = []
    for dim, action_name in enumerate(spec.action_names):
        reg = DecisionTreeRegressor(max_depth=min(max_depth, MAX_REGRESSOR_DEPTH),
                                    min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0)
        reg.fit(X, targets[:, dim])
        t = reg.tree_

        def leaf_text(node: int) -> str:
            mean, std = float(t.value[node][0][0]), math.sqrt(max(float(t.impurity[node]), 0.0))
            return f"mean={mean:.3g} (n={int(t.n_node_samples[node])}, std={std:.3g})"

        sections.append(f"Decision tree for {action_name} on {len(targets)} states:")
        sections.extend(_render(reg, spec.obs_names, leaf_text))
    return "\n".join(sections)


def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str:
    """decision-tree splits as readable text, with per-leaf sample counts and accuracy"""
    if not X:
        return "No labelled states yet."
    data = np.asarray(X, dtype=float)
    if spec.action_type == "discrete":
        return _classifier_hints(data, y, spec, max_depth)
    return _regressor_hints(data, y, spec, max_depth)
