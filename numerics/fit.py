"""Lane A, task A4: agreement and parameter fitting"""
from contracts.types import Action, Obs, Params, ProgramFn, SensorSpec


def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float:
    """discrete = exact-match fraction; continuous = fraction with L2 error <= 0.1 x action range."""
    raise NotImplementedError


def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
               max_seconds: float = 8.0) -> tuple[Params, float]:
    """returns (best_params, agreement_on_X)"""
    raise NotImplementedError
