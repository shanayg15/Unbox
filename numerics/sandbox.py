"""Lane A, task A4: sandbox.

An AST whitelist for programs the brain writes. It stops accidents and overreach from an LLM (imports,
file access, loops, writing into obs or P). It is not a boundary for hostile code: `10**10**10` would
still hang, because the threat here is a confused model, not an attacker.
"""
import ast
import math
import random

import numpy as np

from contracts.numerics_api import ProgramError
from contracts.types import Action, Obs, ProgramFn, SensorSpec

SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "float": float, "int": int,
                 "round": round, "len": len, "bool": bool}
FILENAME = "<program>"
N_TEST_OBS = 50

ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.If,
    ast.IfExp, ast.Expr, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Store, ast.Constant, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.keyword, ast.Attribute,
    ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
)
# np.* is allowed by name, so anything in it that touches files or global state has to be listed here.
NP_BLOCKED = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
              "fromregex", "memmap", "DataSource", "seterr", "seterrcall", "seterrobj", "setbufsize",
              "set_printoptions", "ctypeslib", "testing", "lib", "f2py", "distutils", "random"}
RESERVED = {"obs", "P", "math", "np", "policy", *SAFE_BUILTINS}
FRIENDLY = {
    ast.Import: "import", ast.ImportFrom: "import", ast.While: "a while loop", ast.For: "a for loop",
    ast.Lambda: "lambda", ast.Global: "global", ast.Nonlocal: "nonlocal", ast.With: "with",
    ast.Try: "try/except", ast.ListComp: "a comprehension", ast.DictComp: "a comprehension",
    ast.SetComp: "a comprehension", ast.GeneratorExp: "a comprehension", ast.ClassDef: "a class",
}
ALLOWED_SUMMARY = ("Programs may only use assignments, if/elif/else, return, arithmetic, comparisons, boolean "
                   "ops, and calls to abs, min, max, float, int, round, len, bool, math.*, np.*.")
SHAPE = 'A program must be exactly one function, `def policy(obs, P):`, and nothing else at the top level.'


def _fail(node: ast.AST, message: str) -> None:
    raise ProgramError(f"line {getattr(node, 'lineno', '?')}: {message}")


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return node.id if isinstance(node, ast.Name) else "<expression>"


def _check_whitelist(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            what = FRIENDLY.get(type(node), type(node).__name__)
            _fail(node, f"{what} is not allowed. {ALLOWED_SUMMARY}")


def _check_shape(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "policy":
        raise ProgramError(SHAPE)
    fn = tree.body[0]
    args = fn.args
    plain = ([a.arg for a in args.args] == ["obs", "P"] and not args.posonlyargs and not args.kwonlyargs
             and not args.vararg and not args.kwarg and not args.defaults and not args.kw_defaults)
    if not plain or fn.decorator_list or fn.returns or any(a.annotation for a in args.args):
        raise ProgramError(SHAPE)
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node is not fn:
            _fail(node, f"nested functions are not allowed. {SHAPE}")
    return fn


def _check_target(node: ast.AST, target: ast.AST) -> None:
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _check_target(node, item)
    elif not isinstance(target, ast.Name):
        _fail(target, "you can only assign to plain variable names, not into obs, P or a list. "
                      "Make a new variable instead.")
    elif target.id in RESERVED:
        _fail(target, f"you can't assign to '{target.id}', that name is reserved.")


def _check_assignments(fn: ast.FunctionDef) -> None:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _check_target(node, target)
        elif isinstance(node, ast.AugAssign):
            _check_target(node, node.target)


def _check_recursion(fn: ast.FunctionDef) -> None:
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "policy":
            _fail(node, "recursion is not allowed, policy can't refer to itself.")


def _read_params(fn: ast.FunctionDef) -> list[str]:
    """Every P["name"] read, in order. Any other use of P is an error."""
    message = 'P may only be read as P["name"], with a literal name.'
    names: list[str] = []
    allowed: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "P":
            key = node.slice
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                _fail(node, message)
            if key.value.startswith("_"):
                _fail(node, f"P[{key.value!r}]: names starting with '_' are not allowed.")
            allowed.add(id(node.value))
            if key.value not in names:
                names.append(key.value)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "P" and id(node) not in allowed:
            _fail(node, message)
    return names


def _check_attribute(node: ast.Attribute) -> None:
    dotted = _dotted(node)
    if not (isinstance(node.value, ast.Name) and node.value.id in ("math", "np")):
        _fail(node, f"attribute access '{dotted}' is not allowed, only one level of math.* and np.*.")
    if node.attr.startswith("_"):
        _fail(node, f"'{dotted}': names starting with '_' are not allowed.")
    module = math if node.value.id == "math" else np
    if node.value.id == "np" and node.attr in NP_BLOCKED:
        _fail(node, f"'{dotted}' is not allowed, programs only compute numbers.")
    if not hasattr(module, node.attr):
        _fail(node, f"'{dotted}' does not exist.")


def _check_call(node: ast.Call) -> None:
    func = node.func
    named_builtin = isinstance(func, ast.Name) and func.id in SAFE_BUILTINS
    named_module = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
        and func.value.id in ("math", "np")
    if not (named_builtin or named_module):
        _fail(node, f"calling '{_dotted(func)}' is not allowed. {ALLOWED_SUMMARY}")
    if any(kw.arg is None for kw in node.keywords):
        _fail(node, "**kwargs is not allowed.")


def _check_details(fn: ast.FunctionDef, spec: SensorSpec) -> None:
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    known = RESERVED | assigned
    docstring = None
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        docstring = fn.body[0]
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and node is not docstring:
            _fail(node, "a bare expression does nothing here. Assign it to a variable or return it.")
        elif isinstance(node, ast.Call):
            _check_call(node)
        elif isinstance(node, ast.Attribute):
            _check_attribute(node)
        elif isinstance(node, ast.Name):
            if node.id.startswith("_"):
                _fail(node, f"'{node.id}': names starting with '_' are not allowed.")
            if node.id not in known:
                _fail(node, f"'{node.id}' is not defined. {ALLOWED_SUMMARY}")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "obs" \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            n = len(spec.obs_names)
            if not -n <= node.slice.value < n:
                _fail(node, f"obs[{node.slice.value}] is out of range, obs has {n} features: "
                            f"{', '.join(spec.obs_names)}.")


def _explain(exc: Exception) -> str:
    """Turn an exception raised inside a program into a message that names the program's line."""
    line = "?"
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == FILENAME:
            line = tb.tb_lineno
        tb = tb.tb_next
    if isinstance(exc, KeyError):
        return f'line {line}: the program reads P["{exc.args[0]}"] but params has no value for it.'
    return f"line {line}: {type(exc).__name__}: {exc}"


def _discrete_coercer(spec: SensorSpec):
    n = len(spec.action_names)
    names = ", ".join(f"{i}={name}" for i, name in enumerate(spec.action_names))

    def coerce(action) -> int:
        if type(action) is int and 0 <= action < n:
            return action
        if isinstance(action, (bool, np.bool_, np.integer)):
            action = int(action)
            if 0 <= action < n:
                return action
        raise ProgramError(f"policy returned {action!r}. It must return an int action id: {names}.")
    return coerce


def _continuous_coercer(spec: SensorSpec):
    n = len(spec.action_names)
    wanted = f"a list of {n} floats ({', '.join(spec.action_names)})"

    def coerce(action) -> list[float]:
        if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != n:
            raise ProgramError(f"policy returned {action!r}. It must return {wanted}.")
        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            raise ProgramError(f"policy returned {action!r}. It must return {wanted}.") from exc
        if not all(math.isfinite(v) for v in values):
            raise ProgramError(f"policy returned {action!r}, which is not finite. It must return {wanted}.")
        return values
    return coerce


def sample_obs(spec: SensorSpec, n: int) -> list[Obs]:
    rng = random.Random(0)
    bounds = [(lo if math.isfinite(lo) else -10.0, hi if math.isfinite(hi) else 10.0)
              for lo, hi in zip(spec.obs_low, spec.obs_high)]
    return [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n)]


def load_program(code: str, spec: SensorSpec) -> ProgramFn:
    """AST-validates, compiles, test-runs on 50 sample obs, returns fn(obs, P) -> Action."""
    try:
        tree = ast.parse(code, filename=FILENAME)
    except SyntaxError as exc:
        raise ProgramError(f"line {exc.lineno}: syntax error: {exc.msg}") from exc

    _check_whitelist(tree)
    function_def = _check_shape(tree)
    _check_assignments(function_def)
    _check_recursion(function_def)
    param_names = _read_params(function_def)
    _check_details(function_def, spec)

    namespace = {"__builtins__": SAFE_BUILTINS, "math": math, "np": np}
    exec(compile(tree, FILENAME, "exec"), namespace)
    raw = namespace["policy"]
    coerce = _discrete_coercer(spec) if spec.action_type == "discrete" else _continuous_coercer(spec)

    def fn(obs: Obs, P: dict[str, float]) -> Action:
        try:
            action = raw(obs, P)
        except Exception as exc:
            raise ProgramError(_explain(exc)) from exc
        return coerce(action)

    fn.param_names = tuple(param_names)

    # load_program never sees the brain's params, so the test-run uses stand-ins. They are distinct and
    # non-zero so `x / P["a"]` and `x / (P["a"] - P["b"])` don't fail on the stand-in alone.
    stand_ins = {name: 0.5 + 0.1 * i for i, name in enumerate(param_names)}
    for obs in sample_obs(spec, N_TEST_OBS):
        try:
            fn(obs, stand_ins)
        except ProgramError as exc:
            shown = [round(v, 3) for v in obs]
            raise ProgramError(f"test run on obs {shown}: {exc}") from exc
    return fn
