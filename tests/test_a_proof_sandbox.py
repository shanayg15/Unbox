"""A4: load_program accepts the programs the brain is told to write and rejects everything else."""
import numpy as np
import pytest

from contracts.numerics_api import ProgramError
from numerics.sandbox import load_program
from tests.test_a_proof_helpers import ARM_SPEC, CARTPOLE_SPEC, LUNAR_SPEC

GOOD = '''
def policy(obs, P):
    """Lean into the fall."""
    angle = obs[2]
    ang_v = obs[3]
    score = P["k_a"] * angle + P["k_w"] * ang_v
    if abs(score) < P["dead"]:
        return 1 if ang_v > 0 else 0
    elif score > 0:
        return 1
    else:
        return 0
'''


def wrap(body: str) -> str:
    return "def policy(obs, P):\n" + "".join(f"    {line}\n" for line in body.splitlines())


def test_good_program_runs_and_uses_params():
    fn = load_program(GOOD, CARTPOLE_SPEC)
    assert fn([0.0, 0.0, 0.2, 0.1], {"k_a": 1.0, "k_w": 1.0, "dead": 0.01}) == 1
    assert fn([0.0, 0.0, -0.2, -0.1], {"k_a": 1.0, "k_w": 1.0, "dead": 0.01}) == 0


@pytest.mark.parametrize("line", [
    "x = math.sqrt(abs(obs[0])) + np.clip(obs[1], -1.0, 1.0)",
    "x, y = obs[0], obs[1]",
    "x = min(obs[0], max(obs[1], 0.5))",
    "x = float(len(obs)) + round(obs[0], 2) + int(obs[1] > 0) + bool(obs[2])",
    "x = -obs[0] ** 2 % 3 // 1",
    "x = obs[0]\nx += 1.0",
    "x = sum_ = obs[:2][0]",
    "x = np.dot(obs, [1.0, 2.0, 3.0, 4.0])",
    "x = 1.0 if obs[0] > 0 and not obs[1] < 0 else 0.0",
])
def test_allowed_constructs_load(line):
    load_program(wrap(line + "\nreturn 0"), CARTPOLE_SPEC)


@pytest.mark.parametrize("code, fragment", [
    ("import os\ndef policy(obs, P):\n    return 0\n", "import"),
    ("from os import path\ndef policy(obs, P):\n    return 0\n", "import"),
    (wrap("import os\nreturn 0"), "import"),
    (wrap("while True:\n    x = 1\nreturn 0"), "while"),
    (wrap("for i in obs:\n    x = i\nreturn 0"), "for"),
    (wrap("f = lambda a: a\nreturn 0"), "lambda"),
    (wrap("global x\nreturn 0"), "global"),
    (wrap("with obs as o:\n    x = 1\nreturn 0"), "with"),
    (wrap("try:\n    x = 1\nexcept Exception:\n    x = 2\nreturn 0"), "try"),
    (wrap("x = [i for i in obs]\nreturn 0"), "comprehension"),
    (wrap("return policy(obs, P)"), "recursi"),
    (wrap("x = obs.__class__\nreturn 0"), "__class__"),
    (wrap("x = obs.count(1.0)\nreturn 0"), "obs.count"),
    (wrap("_x = 1\nreturn 0"), "_x"),
    (wrap("x = math.__loader__\nreturn 0"), "__loader__"),
    (wrap("x = np.linalg.norm(obs)\nreturn 0"), "np.linalg"),
    (wrap("x = eval('1')\nreturn 0"), "eval"),
    (wrap("x = open('f')\nreturn 0"), "open"),
    (wrap("x = __import__('os')\nreturn 0"), "__import__"),
    (wrap("x = P\nreturn 0"), 'P["name"]'),
    (wrap("x = P.get('a')\nreturn 0"), 'P["name"]'),
    (wrap("x = P[obs[0]]\nreturn 0"), 'P["name"]'),
    (wrap("P['a'] = 1.0\nreturn 0"), "assign"),
    (wrap("obs[0] = 1.0\nreturn 0"), "assign"),
    (wrap("np.pi = 3.0\nreturn 0"), "assign"),
    (wrap("abs = 1.0\nreturn 0"), "abs"),
    (wrap("x = undefined_thing\nreturn 0"), "undefined_thing"),
    (wrap("x = np.not_a_function(1)\nreturn 0"), "np.not_a_function"),
    (wrap("x = obs[9]\nreturn 0"), "obs[9]"),
    (wrap("x = 1.0\nx\nreturn 0"), "line 3"),
    ("def controller(obs, P):\n    return 0\n", "def policy(obs, P)"),
    ("def policy(obs):\n    return 0\n", "def policy(obs, P)"),
    ("def policy(obs, P):\n    return 0\ndef other(obs, P):\n    return 0\n", "exactly one"),
    ("x = 1\ndef policy(obs, P):\n    return 0\n", "exactly one"),
    ("def policy(obs, P)\n    return 0\n", "line 1"),
])
def test_rejected_with_specific_message(code, fragment):
    with pytest.raises(ProgramError) as exc:
        load_program(code, CARTPOLE_SPEC)
    assert fragment in str(exc.value)


@pytest.mark.parametrize("name", ["load", "save", "fromfile", "memmap", "loadtxt", "savetxt", "genfromtxt"])
def test_numpy_file_io_blocked_even_when_passed_as_a_key(name):
    with pytest.raises(ProgramError):
        load_program(wrap(f"x = max(['f'], key=np.{name})\nreturn 0"), CARTPOLE_SPEC)


def test_error_names_the_line_of_the_bad_node():
    with pytest.raises(ProgramError, match="line 3"):
        load_program("def policy(obs, P):\n    x = 1\n    import os\n    return 0\n", CARTPOLE_SPEC)


@pytest.mark.parametrize("body, fragment", [
    ("return 2", "2"),                   # out of range for a 2-action spec
    ("return -1", "-1"),
    ("return 0.5", "0.5"),               # float is not an action id
    ('return "left"', "left"),
    ("x = 1", "None"),                   # no return at all
    ("return obs[0] / 0.0", "ZeroDivisionError"),
    ("return math.log(-1.0 - abs(obs[0]))", "line 2"),
])
def test_bad_return_or_crash_found_by_test_run(body, fragment):
    with pytest.raises(ProgramError) as exc:
        load_program(wrap(body), CARTPOLE_SPEC)
    assert fragment in str(exc.value)


def test_test_run_reaches_a_branch_only_some_obs_take():
    # Only samples with a positive cart_x hit the invalid action; a 50-sample uniform draw must find it.
    with pytest.raises(ProgramError, match="5"):
        load_program(wrap("return 5 if obs[0] > 0 else 0"), CARTPOLE_SPEC)


@pytest.mark.parametrize("body, want, kind", [
    ("return np.int64(1)", 1, int),
    ("return np.bool_(True)", 1, int),
    ("return True", 1, int),
    ("return bool(False)", 0, int),
])
def test_discrete_returns_coerced_to_plain_int(body, want, kind):
    fn = load_program(wrap(body), CARTPOLE_SPEC)
    got = fn([0.0, 0.0, 0.0, 0.0], {})
    assert got == want and type(got) is kind


def test_unbounded_spec_does_not_break_the_test_run():
    spec = CARTPOLE_SPEC.__class__(**{**CARTPOLE_SPEC.__dict__, "obs_low": [-np.inf] * 4, "obs_high": [np.inf] * 4})
    assert load_program(wrap("return 1 if obs[0] > 0 else 0"), spec)([1.0, 0, 0, 0], {}) == 1


def test_continuous_return_coerced_to_float_list():
    fn = load_program(wrap("return np.array([obs[0], 0.5])"), ARM_SPEC)
    got = fn([0.25, 0.0], {})
    assert got == [0.25, 0.5] and type(got) is list and all(type(v) is float for v in got)


@pytest.mark.parametrize("body", ["return [obs[0]]", "return 1", "return [obs[0], float('nan')]"])
def test_continuous_wrong_shape_or_nan_rejected(body):
    with pytest.raises(ProgramError):
        load_program(wrap(body), ARM_SPEC)


def test_runtime_crash_surfaces_as_program_error_with_the_line():
    fn = load_program(wrap("x = 1.0 / (obs[0] - 7.0)\nreturn 0"), CARTPOLE_SPEC)  # never 7 in the test-run bounds
    with pytest.raises(ProgramError, match="line 2"):
        fn([7.0, 0.0, 0.0, 0.0], {})


def test_missing_param_reported_by_name_at_runtime():
    fn = load_program(wrap('return 1 if obs[0] > P["limit"] else 0'), CARTPOLE_SPEC)
    with pytest.raises(ProgramError, match="limit"):
        fn([0.0, 0.0, 0.0, 0.0], {})


def test_loaded_fn_lists_the_params_it_reads():
    fn = load_program(GOOD, CARTPOLE_SPEC)
    assert set(fn.param_names) == {"k_a", "k_w", "dead"}


def test_program_cannot_mutate_its_observation():
    # Aliasing obs then writing through the alias would corrupt the fit data.
    with pytest.raises(ProgramError):
        load_program(wrap("o = obs\no[0] = 9.0\nreturn 0"), CARTPOLE_SPEC)


def test_four_action_spec_accepts_id_three():
    assert load_program(wrap("return 3"), LUNAR_SPEC)([0.0] * 8, {}) == 3
