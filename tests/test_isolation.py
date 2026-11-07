"""Whoever creates the repo: isolation rules from CLAUDE.md section 1, plus an import check"""
import ast
import importlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED = {"stable_baselines3", "huggingface_sb3", "rl_zoo3", "torch"}
SKIP_DIRS = {"__pycache__", "runs", "cache", "results"}


def py_files(*top_level_dirs: str) -> list[Path]:
    """Every .py file under the given top-level dirs, or the whole repo if none are given."""
    starts = [ROOT / d for d in top_level_dirs] or [ROOT]
    found = []
    for start in starts:
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
            found += [Path(dirpath) / name for name in filenames if name.endswith(".py")]
    return sorted(found)


def imported_names(path: Path):
    """Yield each dotted name a file imports: nested, relative, and literal dynamic imports included."""
    package = path.relative_to(ROOT).parent.parts
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = tuple(node.module.split(".")) if node.module else ()
            base = package[: max(0, len(package) - node.level + 1)] if node.level else ()
            yield from (".".join(base + module + (alias.name,)) for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            called = getattr(node.func, "attr", getattr(node.func, "id", None))
            first = node.args[0]
            if called in {"import_module", "__import__"} and isinstance(first, ast.Constant):
                yield str(first.value)


def is_forbidden(name: str) -> bool:
    top = name.split(".")[0]
    if top in BANNED:
        return True
    return top == "blackbox" and name != "blackbox.client" and not name.startswith("blackbox.client.")


def test_agent_and_app_import_nothing_forbidden():
    files = py_files("agent", "app")
    assert files, "no files scanned under agent/ or app/"
    bad = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in files
        for name in imported_names(path)
        if is_forbidden(name)
    ]
    assert not bad, f"forbidden imports: {bad}"


def test_agent_never_mentions_admin_endpoints():
    files = py_files("agent")
    assert files, "no files scanned under agent/"
    bad = [str(path.relative_to(ROOT)) for path in files if "/admin" in path.read_text()]
    assert not bad, f"'/admin' found in: {bad}"


def test_every_module_imports():
    files = py_files()
    assert files, "no modules found"
    failures = {}
    for path in files:
        parts = path.relative_to(ROOT).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures[name] = repr(exc)
    assert not failures, f"modules that failed to import: {failures}"
