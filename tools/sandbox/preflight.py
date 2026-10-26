"""Offline preflight for a committed Standing A100 Development Sandbox manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .manifest import PROTECTED_TOKENS, ManifestError, load_manifest

TRANSPORT_TREE = "5833e54e9c5396121cbbf99a5cd4e52006e564fd"
REPO_ROOT = Path(__file__).resolve().parents[2]


class PreflightError(RuntimeError):
    """An automated sandbox admission check failed."""


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreflightError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def _assert_safe_path(value: str | Path, name: str) -> Path:
    path = Path(value)
    rendered = path.as_posix().lower()
    if any(token in rendered for token in PROTECTED_TOKENS):
        raise PreflightError(f"{name} names a protected F5/study path")
    return path


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PreflightError(f"path must be inside repository: {path}") from exc


def verify_committed_manifest(path: Path, *, require_committed: bool = True) -> None:
    """Require a tracked manifest whose staged HEAD blob matches working bytes."""
    relative = _repo_relative(path)
    if not require_committed:
        return
    tracked = _run(["git", "ls-files", "--error-unmatch", "--", relative])
    if tracked != relative:
        raise PreflightError("manifest must be tracked")
    head_bytes = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    if head_bytes.returncode or head_bytes.stdout != path.read_bytes():
        raise PreflightError("manifest working bytes must exactly match its committed HEAD blob")


def verify_transport_tree() -> None:
    """Confirm the qualified transport subtree is bit-identical and untouched."""
    tree = _run(["git", "rev-parse", "HEAD:research/colab_cli"])
    if tree != TRANSPORT_TREE:
        raise PreflightError("qualified research/colab_cli tree differs from 5833e54e")
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "research/colab_cli"]
    )
    if dirty:
        raise PreflightError("qualified research/colab_cli tree has dirty or untracked files")


def _validate_registry_inputs(payload: dict[str, Any], registry_path: Path) -> None:
    _assert_safe_path(registry_path, "registry")
    from .registry import load_registry, validate_input, verify_sources

    registry = load_registry(registry_path)
    verify_sources(registry)
    for record in payload["inputs"]:
        # manifest validation has already rejected protected strings, before any
        # record path is opened by the registry loader/validator.
        validate_input(record, registry)


def _verify_broker_binding(payload: dict[str, Any]) -> None:
    """Bind the ten-line record to the packet that an operator will dry-run."""
    broker = payload["outputs"].get("broker_manifest")
    if broker is None:
        return
    if not isinstance(broker, dict) or set(broker) != {"path", "sha256"}:
        raise PreflightError("outputs.broker_manifest must have path and sha256")
    path = _assert_safe_path(broker["path"], "broker manifest")
    _repo_relative(path)
    if not path.is_file():
        raise PreflightError("broker manifest does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != broker["sha256"]:
        raise PreflightError("broker manifest SHA-256 does not match sandbox manifest binding")


def run_focused_tests() -> None:
    """Run the local sandbox tests with the owner-required low-impact settings."""
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    command = [
        "nice",
        "-n",
        "15",
        str(REPO_ROOT / "venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tools/sandbox/tests",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, env=env, check=False)
    if completed.returncode:
        raise PreflightError("focused sandbox tests failed")


def preflight(
    manifest_path: Path,
    *,
    registry_path: Path | None = None,
    require_committed: bool = True,
    run_tests: bool = True,
) -> dict[str, Any]:
    """Run all offline preflight checks without invoking any Colab/dispatch code."""
    manifest_path = _assert_safe_path(manifest_path, "manifest")
    _repo_relative(manifest_path)
    verify_committed_manifest(manifest_path, require_committed=require_committed)
    payload = load_manifest(manifest_path)
    _verify_broker_binding(payload)
    resolved_registry = registry_path or REPO_ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    _validate_registry_inputs(payload, resolved_registry)
    verify_transport_tree()
    if run_tests:
        run_focused_tests()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--skip-tests", action="store_true", help="test helper only; not an admission PASS"
    )
    args = parser.parse_args(argv)
    try:
        preflight(args.manifest, registry_path=args.registry, run_tests=not args.skip_tests)
    except (ManifestError, PreflightError, ValueError) as exc:
        print(f"PREFLIGHT REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        "PREFLIGHT PASS: schema, registry, committed manifest, qualified transport, focused tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
