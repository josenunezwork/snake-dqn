"""Offline preflight for a committed Standing A100 Development Sandbox manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .manifest import PROTECTED_TOKENS, ManifestError, _forbidden, load_manifest

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

    verify_committed_manifest(registry_path)
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
    _assert_safe_path(path.resolve(), "resolved broker manifest")
    if not path.is_file():
        raise PreflightError("broker manifest does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != broker["sha256"]:
        raise PreflightError("broker manifest SHA-256 does not match sandbox manifest binding")
    packet = json.loads(path.read_text())
    _forbidden(
        {key: packet.get(key) for key in ("script", "source_paths", "env", "output_dir")}, "broker"
    )
    script = packet.get("script", "")
    if not isinstance(script, str) or not script.startswith(
        ("research/sandbox_b_heads/", "research/sandbox_a_resident/")
    ):
        raise PreflightError("broker script is outside sandbox packages")
    allowed = {
        (record.get("sha256"), str(Path(record["path"]).resolve()))
        for record in payload["inputs"]
        if "path" in record
    }
    shipped = set()
    for record in packet.get("inputs", []):
        source = _assert_safe_path(record.get("path", ""), "broker input").resolve()
        _assert_safe_path(source, "resolved broker input")
        _assert_safe_path(record.get("dest", ""), "broker input destination")
        pair = (record.get("sha256"), str(source))
        if pair not in allowed:
            raise PreflightError("broker uploads an input absent from sandbox admission")
        shipped.add(pair)
    if shipped != allowed:
        raise PreflightError("broker input set differs from sandbox admission")
    seconds = packet.get("timeout_minutes", 0) * 60
    if isinstance(seconds, bool) or not math.isclose(
        seconds, payload["timeout"]["seconds"], abs_tol=1e-8
    ):
        raise PreflightError("broker timeout differs from sandbox timeout")
    if packet.get("requirements", {}).get("gpu") != "A100":
        raise PreflightError("sandbox broker must request A100")
    commit = packet.get("commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PreflightError("broker runtime pin must be a full commit SHA")
    pinned_tree = _run(["git", "rev-parse", f"{commit}:research/colab_cli"])
    if pinned_tree != TRANSPORT_TREE:
        raise PreflightError("broker source pin changes qualified transport")
    for source in packet.get("source_paths", []):
        _assert_safe_path(source, "broker source")
        relative = Path(source).as_posix()
        pinned = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False
        )
        if pinned.returncode or pinned.stdout != (REPO_ROOT / source).read_bytes():
            raise PreflightError("broker source differs from committed runtime pin: " + source)


def run_focused_tests() -> None:
    """Run the local sandbox tests with the owner-required low-impact settings."""
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "SNAKE_DQN_DEVICE": "cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
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
        "-p",
        "no:xdist",
        "tools/sandbox/tests",
        "research/sandbox_b_heads/tests",
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
    if args.skip_tests:
        print("PREFLIGHT CHECK ONLY: tests skipped; no admission PASS")
        return 3
    print(
        "PREFLIGHT PASS: schema, registry, committed manifest, qualified transport, focused tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
