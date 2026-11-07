"""Lane A, task A5: evaluator"""
from typing import Callable

from contracts.types import BlackBox, EvalReport, Params, ProgramFn, QueryBatch, SensorSpec


def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
             holdout: list[QueryBatch], env_factory: Callable[[int], object],
             seeds: list[int]) -> EvalReport:
    """env_factory(seed) -> gymnasium env already reset with that seed; returns (env, first_obs)"""
    raise NotImplementedError
