"""Check a pinned sparse job's imports in a fresh declared-dependencies-only venv.

No project install, ambient site packages, PYTHONPATH, input payloads or job execution.
Call ``check`` before publishing a pack, or use the CLI for another pack family.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(
        argv, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT, timeout=600
    )


def module_path(root: Path, module: str) -> Path | None:
    """Resolve only source in the pinned sparse tree, without importing it."""
    path = root.joinpath(*module.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def import_statements(root: Path, module: str) -> list[str]:
    """Include function-local imports; callable selectors bound an explicit seam."""
    module, _, function = module.partition(":")
    path = module_path(root, module)
    if path is None:
        raise ValueError(f"runtime root absent from pinned source: {module}")
    tree = ast.parse(path.read_text())
    if function:
        tree = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function
        )
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    statements = []
    nodes = []

    class RuntimeImports(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            condition = node.test
            typing_only = (isinstance(condition, ast.Name) and condition.id == "TYPE_CHECKING") or (
                isinstance(condition, ast.Attribute) and condition.attr == "TYPE_CHECKING"
            )
            if typing_only:
                for child in node.orelse:
                    self.visit(child)
            else:
                self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            nodes.append(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            nodes.append(node)

    RuntimeImports().visit(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            statements.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            if node.level:
                node.module = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                node.level = 0
            statements.append(ast.unparse(node))
    return sorted(set(statements))


def import_closure(root: Path, modules: list[str]) -> tuple[list[str], list[str]]:
    """Recursively force static imports, including lazy transitive and package imports.

    This is deliberately conservative: optional branches also require declarations.
    Runtime-computed import strings need an explicit additional root.
    """
    pending = list(modules)
    visited: set[str] = set()
    statements: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        module = name.partition(":")[0]
        for end in range(1, len(module.split("."))):
            parent = ".".join(module.split(".")[:end])
            if module_path(root, parent):
                pending.append(parent)
        for statement in import_statements(root, name):
            statements.add(statement)
            node = ast.parse(statement).body[0]
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            else:
                targets = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            pending.extend(t for t in targets if module_path(root, t))
    return sorted(visited), sorted(statements)


def check(
    repo: Path,
    commit: str,
    source_paths: list[str],
    requirements: str,
    modules: list[str],
    *,
    uv: str = "uv",
) -> dict[str, Any]:
    """Fail closed before pack creation; probe exactly the pinned sparse source.

    ``modules`` names import-only runtime roots, including lazy replay/backends.
    Every static import in the recursively reachable local modules, including
    function-local imports and package initializers, is executed. Runtime-computed
    import strings must be included as additional roots by each pack author.
    CUDA execution remains the responsibility of the real same-pin family canary.
    """
    if not modules:
        raise ValueError("at least one runtime import root is required")
    repo = repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repo, text=True
    ).strip()
    # Never pass through Python/uv/pip configuration from the developer's venv.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        SNAKE_DQN_DEVICE="cpu",
        UV_NO_CONFIG="1",
    )
    with tempfile.TemporaryDirectory(prefix="job-import-preflight-") as temporary:
        work = Path(temporary).resolve()
        source = work / "source"
        source.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", commit, "--", *source_paths], cwd=repo, timeout=60
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(source, filter="data")
        req = source / requirements
        if not req.is_file() or not req.resolve().is_relative_to(source):
            raise ValueError("declared requirements absent from pinned source")
        # Fresh path for each call; no --system-site-packages, project, or seed deps.
        _run(
            [uv, "--no-config", "venv", "--python", "3.12", "--managed-python", str(work / "venv")],
            work,
            env,
        )
        python = work / "venv/bin/python"
        install = _run(
            [uv, "--no-config", "pip", "install", "--python", str(python), "-r", str(req)],
            source,
            env,
        )
        closure, statements = import_closure(source, modules)
        probe = (
            "import importlib,json,site,sys; "
            "from pathlib import Path; "
            "assert sys.prefix != sys.base_prefix; "
            "assert not site.ENABLE_USER_SITE; "
            f"sys.path.insert(0, {str(source)!r}); "
            f"modules={modules!r}; closure={closure!r}; statements={statements!r}; "
            '[importlib.import_module(m.partition(":")[0]) for m in closure]; '
            "[exec(s, {}) for s in statements]; "
            "local={m:str(Path(v.__file__).relative_to(sys.path[0])) "
            "for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) "
            "and Path(v.__file__).is_relative_to(sys.path[0])}; "
            "print(json.dumps(dict(status='PASS',python=sys.version.split()[0],"
            "user_site=site.ENABLE_USER_SITE,system_site_packages=False,"
            "runtime_roots=modules,closure=closure,import_statements=len(statements),"
            "local_modules=local),sort_keys=True))"
        )
        output = _run([str(python), "-I", "-c", probe], source, env)
        result = json.loads(output.splitlines()[-1])
        result.update(
            schema="job-clean-import-preflight/v1",
            checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            commit=commit,
            requirements=requirements,
            requirements_sha256=hashlib.sha256(req.read_bytes()).hexdigest(),
            source_archive_sha256=hashlib.sha256(archive).hexdigest(),
            installed=json.loads(
                _run(
                    [
                        uv,
                        "--no-config",
                        "--quiet",
                        "pip",
                        "list",
                        "--python",
                        str(python),
                        "--format",
                        "json",
                    ],
                    source,
                    env,
                )
            ),
            install_log=install,
        )
        print(
            f"CLEAN_VENV_IMPORT_PREFLIGHT PASS python={result['python']} "
            f"roots={len(modules)} statements={len(statements)} "
            f"local_modules={len(result['local_modules'])} "
            "system_site_packages=false user_site=false",
            flush=True,
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.repo, args.commit, args.source_path, args.requirements, args.module)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
