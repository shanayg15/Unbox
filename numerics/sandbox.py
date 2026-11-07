"""Lane A, task A4: sandbox"""
from contracts.numerics_api import ProgramError
from contracts.types import ProgramFn, SensorSpec


def load_program(code: str, spec: SensorSpec) -> ProgramFn:
    """AST-validates, compiles, test-runs on 50 sample obs, returns fn(obs, P) -> Action."""
    raise NotImplementedError
