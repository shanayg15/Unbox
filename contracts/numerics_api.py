from contracts.types import *


class ProgramError(Exception):
    """Bad program. str(e) is fed back to the brain verbatim, so make it helpful."""


def load_program(code: str, spec: SensorSpec) -> ProgramFn:
    """AST-validates, compiles, test-runs on 50 sample obs, returns fn(obs, P) -> Action."""
    raise NotImplementedError


def agreement(fn: ProgramFn, params: Params, X: list[Obs], y: list[Action], spec: SensorSpec) -> float:
    """discrete = exact-match fraction; continuous = fraction with L2 error <= 0.1 x action range."""
    raise NotImplementedError


def fit_params(fn: ProgramFn, init: Params, X: list[Obs], y: list[Action], spec: SensorSpec,
               max_seconds: float = 8.0) -> tuple[Params, float]:
    """returns (best_params, agreement_on_X)"""
    raise NotImplementedError


def pattern_hints(X: list[Obs], y: list[Action], spec: SensorSpec, max_depth: int = 4) -> str:
    """decision-tree splits as readable text, with per-leaf sample counts and accuracy"""
    raise NotImplementedError


def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                       spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]:
    """returns the k most diverse disagreements; respects max_queries exactly"""
    raise NotImplementedError


def evaluate(fn: ProgramFn, params: Params, bb: BlackBox, spec: SensorSpec,
             holdout: list[QueryBatch], env_factory: Callable[[int], object],
             seeds: list[int]) -> EvalReport:
    """env_factory(seed) -> gymnasium env already reset with that seed; returns (env, first_obs)"""
    raise NotImplementedError


def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                    obs: Obs, bb_action: Action) -> Action:
    """if recovered(obs) != fixed(obs): return fixed(obs), else return bb_action"""
    raise NotImplementedError
