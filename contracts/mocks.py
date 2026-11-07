"""Owned by Lane B, task B1"""
from typing import Callable

from contracts.numerics_api import ProgramError as _ProgramError
from contracts.types import (
    Action,
    BlackBox,
    Counterexample,
    Episode,
    EvalReport,
    Obs,
    Params,
    ProgramFn,
    QueryBatch,
    SensorSpec,
)


class MockBlackBox:
    def spec(self) -> SensorSpec:
        raise NotImplementedError

    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch:
        raise NotImplementedError

    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode:
        raise NotImplementedError

    def query_count(self) -> int:
        raise NotImplementedError


class numerics_stub:
    """Same surface as contracts/numerics_api.py."""

    ProgramError = _ProgramError

    @staticmethod
    def load_program(code: str, spec: SensorSpec) -> ProgramFn:
        raise NotImplementedError

    @staticmethod
    def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float:
        raise NotImplementedError

    @staticmethod
    def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
                   max_seconds: float = 8.0) -> tuple[Params, float]:
        raise NotImplementedError

    @staticmethod
    def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str:
        raise NotImplementedError

    @staticmethod
    def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                           spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]:
        raise NotImplementedError

    @staticmethod
    def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
                 holdout: list[QueryBatch], env_factory: Callable[[int], object],
                 seeds: list[int]) -> EvalReport:
        raise NotImplementedError

    @staticmethod
    def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                        obs: Obs, bb_action: Action) -> Action:
        raise NotImplementedError
