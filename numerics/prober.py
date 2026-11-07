"""Lane A, task A5: prober"""
from contracts.types import BlackBox, Counterexample, Obs, Params, ProgramFn, SensorSpec


def find_disagreements(fn: ProgramFn, params: Params, bb: BlackBox, seed_obs: list[Obs],
                       spec: SensorSpec, k: int = 8, max_queries: int = 400) -> list[Counterexample]:
    """returns the k most diverse disagreements; respects max_queries exactly"""
    raise NotImplementedError
