"""Lane A, task A4: decision-tree pattern hints"""
from contracts.types import Action, Obs, SensorSpec


def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str:
    """decision-tree splits as readable text, with per-leaf sample counts and accuracy"""
    raise NotImplementedError
