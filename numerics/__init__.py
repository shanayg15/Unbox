"""Lane A, tasks A4-A5: numerics core, re-exports everything in contracts/numerics_api.py"""
# The evaluate function replaces the numerics.evaluate submodule as an attribute of this package.
# `from numerics.evaluate import evaluate` still works; `import numerics.evaluate as m` gets the function.
from numerics.evaluate import evaluate
from numerics.fit import agreement, fit_params
from numerics.patterns import pattern_hints
from numerics.prober import find_disagreements
from numerics.sandbox import ProgramError, load_program
from numerics.shield import shielded_action

__all__ = [
    "ProgramError",
    "agreement",
    "evaluate",
    "find_disagreements",
    "fit_params",
    "load_program",
    "pattern_hints",
    "shielded_action",
]
