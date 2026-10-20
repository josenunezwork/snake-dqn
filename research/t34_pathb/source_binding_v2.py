"""Fail-closed source closure and frozen-worktree fence for T34 retention.

The closure is intentionally derived from Python syntax instead of a broad
directory glob.  It binds only the executable import graph rooted at the eight
registered entry points, their package initializers, and the static repository
data inputs those programs read.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Final


class SourceBindingError(RuntimeError):
    """The registered executable closure or frozen worktree is unsafe."""


ENTRYPOINTS: Final = (
    "research.t34_pathb.retention_v2",
    "research.t34_pathb.analyze_v2",
    "research.t34_pathb.timing_v2",
    "research.t34_pathb.host_guard_v2",
    "research.t34_pathb.registration_retention_v2",
    "research.t34_pathb.retention_oc_v2",
    "research.t34_pathb.unpaced_colab",
    "research.t34_pathb.make_unpaced_manifest",
)

# These are repository inputs read by the retained execution path.  Runtime
# receipts, output directories, and external historical artifacts are not
# source inputs and therefore must be bound separately by their registrations.
DECLARED_DATA_FILES: Final = frozenset(
    {
        "research/t34_pathb/PROTOCOL_V2.md",
        "research/t34_pathb/PROTOCOL_V2_BUILD_RULES.md",
        "research/t34_pathb/RETENTION_PROFILE_V2.json",
        "research/t34_pathb/PREREGISTRATION_RETENTION_V2.md",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/t34_pathb/SEED_AUDIT_RETENTION_V2.json",
        "research/t34_round0/BANK_ROUND0.json",
        "research/t13b_six_snake_20261017/results/episodes.json.gz",
        "research/apex_safety_20260926/deployment.yaml",
        "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
        "web/backend/served_checkpoint_pins.json",
        "research/t34_pathb/unpaced_colab.sh",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "research/colab_cli/QUALIFICATION.md",
    }
)
DECLARED_DATA_GLOBS: Final = ("configs/**/*.yaml", "configs/**/*.yml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_file(repo: Path, module: str) -> Path | None:
    """Return a repo-local Python module path, preferring module over package."""
    if not module or not module.startswith(("src", "research", "web")):
        return None
    base = repo.joinpath(*module.split("."))
    file = base.with_suffix(".py")
    if file.is_file():
        return file
    init = base / "__init__.py"
    if init.is_file():
        return init
    return None


def _package_initializers(repo: Path, module: str) -> set[Path]:
    """Return existing ``__init__`` files for a repository-local module."""
    parts = module.split(".")
    result: set[Path] = set()
    for count in range(1, len(parts)):
        init = repo.joinpath(*parts[:count], "__init__.py")
        if init.is_file():
            result.add(init)
    return result


def _module_name(repo: Path, path: Path) -> str:
    relative = path.relative_to(repo)
    if relative.name == "__init__.py":
        relative = relative.parent
    else:
        relative = relative.with_suffix("")
    return ".".join(relative.parts)


def _relative_module(current: str, level: int, name: str | None, is_package: bool) -> str:
    """Resolve ``from ... import`` names using the importing module package."""
    if not level:
        return name or ""
    package = current.split(".") if is_package else current.split(".")[:-1]
    if level:
        if level > len(package):
            raise SourceBindingError(f"relative import escapes package: {current}")
        package = package[: len(package) - level + 1]
    parts = [*package]
    if name:
        parts.extend(name.split("."))
    return ".".join(parts)


def _literal_import(call: ast.Call) -> str | None:
    """Return a literal dynamic module import, or reject dynamic expressions."""
    func = call.func
    importlib_call = isinstance(func, ast.Attribute) and func.attr == "import_module"
    builtin_call = isinstance(func, ast.Name) and func.id == "__import__"
    if not importlib_call and not builtin_call:
        return None
    if (
        not call.args
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ):
        raise SourceBindingError("dynamic import cannot be statically source-bound")
    return call.args[0].value


def _imports(repo: Path, path: Path) -> set[str]:
    """Return local imports from every AST scope, rejecting dynamic imports."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise SourceBindingError(f"cannot parse source input {path}") from exc
    current = _module_name(repo, path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _relative_module(current, node.level, node.module, path.name == "__init__.py")
            if base and _module_file(repo, base) is not None:
                imports.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base else alias.name
                if alias.name != "*" and _module_file(repo, candidate) is not None:
                    imports.add(candidate)
        elif isinstance(node, ast.Call):
            literal = _literal_import(node)
            if literal:
                imports.add(literal)
    return {name for name in imports if name.startswith(("src", "research", "web"))}


def _resolve_import(repo: Path, name: str) -> set[Path]:
    """Resolve a local import and its package initialization chain."""
    path = _module_file(repo, name)
    if path is not None:
        return {path, *_package_initializers(repo, name)}
    raise SourceBindingError(f"unresolved repository-local import: {name}")


def source_paths(repo: Path) -> tuple[str, ...]:
    """Compute the deterministic registered Python closure plus declared data."""
    repo = repo.resolve()
    pending = list(ENTRYPOINTS)
    seen_modules: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in seen_modules:
            continue
        seen_modules.add(module)
        for path in _resolve_import(repo, module):
            if path in paths:
                continue
            paths.add(path)
            pending.extend(_imports(repo, path))
    for name in DECLARED_DATA_FILES:
        path = repo / name
        if not path.is_file():
            raise SourceBindingError(f"declared source data file is absent: {name}")
        paths.add(path)
    for pattern in DECLARED_DATA_GLOBS:
        paths.update(path for path in repo.glob(pattern) if path.is_file())
    return tuple(sorted(path.relative_to(repo).as_posix() for path in paths))


def source_inventory(repo: Path) -> dict[str, str]:
    """Return sorted relative path to SHA-256 bindings for the actual closure."""
    return {name: _sha256(repo / name) for name in source_paths(repo)}


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceBindingError(f"git verification failed: {' '.join(args)}") from exc


def _is_detached(repo: Path) -> bool:
    """Return whether Git has no checked-out branch, preserving Git failures."""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceBindingError(
            "git verification failed: symbolic-ref --quiet --short HEAD"
        ) from exc
    if result.returncode == 1:
        return True
    if result.returncode:
        raise SourceBindingError("git verification failed: symbolic-ref --quiet --short HEAD")
    return False


def verify_frozen_worktree(repo: Path, ratified_commit: str) -> None:
    """Require detached exact HEAD and no tracked or untracked worktree changes."""
    repo = repo.resolve()
    if not isinstance(ratified_commit, str) or len(ratified_commit) != 40:
        raise SourceBindingError("ratified commit must be a full 40-character SHA")
    head = _git(repo, "rev-parse", "HEAD")
    if head != ratified_commit:
        raise SourceBindingError("frozen worktree HEAD differs from ratified commit")
    if not _is_detached(repo):
        raise SourceBindingError("frozen worktree must use a detached HEAD")
    dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise SourceBindingError("frozen worktree has tracked or untracked changes")


class FrozenWorktreeGuard:
    """Background worktree monitor whose per-tick checks never run subprocesses."""

    def __init__(self, repo: Path, ratified_commit: str, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.repo = repo.resolve()
        self.ratified_commit = ratified_commit
        self.interval_seconds = interval_seconds
        self._failure: SourceBindingError | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        verify_frozen_worktree(self.repo, self.ratified_commit)

    def start(self) -> "FrozenWorktreeGuard":
        """Start periodic validation; safe to call once or repeatedly."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._poll, daemon=True, name="t34-source-fence")
            self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                verify_frozen_worktree(self.repo, self.ratified_commit)
            except SourceBindingError as exc:
                with self._lock:
                    self._failure = exc
                return

    def check(self) -> None:
        """Raise a cached monitor failure without blocking on Git."""
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def final_check(self) -> None:
        """Synchronously confirm the worktree remained frozen through analysis."""
        self.check()
        verify_frozen_worktree(self.repo, self.ratified_commit)
        self.check()

    def close(self) -> None:
        """Stop the monitor without suppressing its recorded failure."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)
            if self._thread.is_alive():
                raise SourceBindingError("source monitor did not close within its timeout")
        self.check()

    def __enter__(self) -> "FrozenWorktreeGuard":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        try:
            self.final_check()
        finally:
            self.close()
