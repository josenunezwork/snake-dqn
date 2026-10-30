"""Offline preflight for a committed Standing A100 Development Sandbox manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .manifest import ManifestError, _forbidden, load_manifest, protected_name

TRANSPORT_TREE = "3a7b2188e3afd47a57ed87cff46d95deb49146aa"
QUALIFIED_TRANSPORT_TREES = frozenset({TRANSPORT_TREE})
REPO_ROOT = Path(__file__).resolve().parents[2]


# A sandbox B ledger is only portable if the recipe is the one the pinned
# runtime will construct.  Keep the version boundary explicit: a new study
# cannot silently inherit an old namespace, ledger schema, or update count.
_B_HEADS_RUNTIME = {
    "sandbox-b-heads-v1": {
        "resume_module": "research.sandbox_b_heads.resume",
        "core_module": None,
        "schema": "sandbox-b-verified-ledger/v3",
        "namespace": "a100-dev/sandbox-b/heads-v1",
        "allocation": "split-v1",
    },
    "sandbox-b-heads-v2": {
        "resume_module": "research.sandbox_b_heads.resume_b2",
        "core_module": "research.sandbox_b_heads.core_b2",
        "schema": "sandbox-b-verified-ledger/v4",
        "namespace": "a100-dev/sandbox-b/heads-v2",
        "allocation": "split-v1",
    },
    "sandbox-b-heads-v3": {
        "resume_module": "research.sandbox_b_heads.resume_b3",
        "core_module": "research.sandbox_b_heads.core_b3",
        "schema": "sandbox-b-verified-ledger/v5",
        "namespace": "a100-dev/sandbox-b/heads-v3",
        "allocation": "split-v1",
    },
}


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
    if protected_name(rendered) or protected_name(path.resolve().as_posix()):
        raise PreflightError(f"{name} names a protected F5/study path")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PreflightError(f"{name} must not traverse symlinks")
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
        raise PreflightError("qualified research/colab_cli tree differs from 3a7b2188")
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "research/colab_cli"]
    )
    if dirty:
        raise PreflightError("qualified research/colab_cli tree has dirty or untracked files")


def _runtime_updates(resume: Any, core_module: str | None, recipe: dict[str, Any]) -> int:
    """Return the total finite update budget declared by the runtime source."""
    if core_module is None:
        configs = resume.CONFIG_IDS
    else:
        configs = importlib.import_module(core_module).CONFIGS
    return sum(getattr(config, "updates", recipe["updates"]) for config in configs)


def _validate_b_heads_runtime_recipe(
    payload: dict[str, Any],
    packet: dict[str, Any],
    ledger: dict[str, Any],
    rows: list[dict[str, Any]],
    b8: list[dict[str, Any]],
) -> None:
    """Require a ledger recipe to equal the recipe produced by the shipped runtime."""
    contract = _B_HEADS_RUNTIME.get(payload["study"])
    if contract is None:
        return
    env = packet.get("env")
    if not isinstance(env, dict) or env.get("SANDBOX_B_SALT") != payload["salt"]:
        raise PreflightError("broker runtime salt differs from sandbox manifest")
    resume = importlib.import_module(contract["resume_module"])
    recipe = ledger.get("recipe")
    expected = resume.recipe(Path(rows[0]["path"]), Path(b8[0]["path"]), salt=payload["salt"])
    if recipe != expected:
        raise PreflightError("resume recipe differs from full shipped runtime recipe")
    if (
        ledger.get("schema") != contract["schema"]
        or ledger.get("namespace") != contract["namespace"]
        or ledger.get("allocation") != contract["allocation"]
        or ledger.get("fingerprint") != resume.fingerprint(expected)
    ):
        raise PreflightError("foreign resume ledger identity")
    if payload["endpoint"]["kind"] != "updates" or payload["endpoint"]["value"] != _runtime_updates(
        resume, contract["core_module"], expected
    ):
        raise PreflightError("sandbox endpoint differs from full runtime recipe")


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
    if payload["study"] in {"sandbox-a-train01", "t34-cf-1028", "t34-ct-h5000-1029"} or any(
        r["kind"] == "frozen-opponent" for r in payload["inputs"]
    ):
        from .registry import TRAINING_OPPONENTS

        opponents = [r for r in payload["inputs"] if r["kind"] == "frozen-opponent"]
        if payload["study"] not in {"sandbox-a-train01", "t34-cf-1028", "t34-ct-h5000-1029"} or (
            len(opponents) != 2 or {r["sha256"] for r in opponents} != TRAINING_OPPONENTS
        ):
            raise PreflightError("TRAIN-01 requires exactly the two canonical fixed opponents")
        export_kinds = (
            {
                "training-export",
                "training-export-sidecar",
                "training-export-derivative",
            }
            if payload["study"] == "sandbox-a-train01"
            else {
                "ct-export",
                "ct-export-sidecar",
                "ct-export-derivative",
            }
        )
        kinds = {r["kind"] for r in payload["inputs"]}
        if kinds - export_kinds != {
            "worlds",
            "frozen-checkpoint",
            "frozen-opponent",
            "verified-ledger",
            "registry",
        }:
            raise PreflightError("TRAIN-01 accepts fresh worlds and fixed models only")
        if any(
            sum(r["kind"] == kind for r in payload["inputs"]) != 1
            for kind in ("frozen-checkpoint", "verified-ledger", "registry")
        ):
            raise PreflightError("TRAIN-01 requires exactly one B8, ledger and registry")
    # Resume payloads are admitted only by the hash-bound verified ledger. Never
    # open their checkpoint/report contents during Mac admission.
    unit_records = [r for r in payload["inputs"] if r["kind"] == "verified-unit"]
    ledger_records = [r for r in payload["inputs"] if r["kind"] == "verified-ledger"]
    if payload["study"] in _B_HEADS_RUNTIME:
        contract = _B_HEADS_RUNTIME[payload["study"]]
        resume = importlib.import_module(contract["resume_module"])
        if len(ledger_records) != 1:
            raise PreflightError("Job B requires one verified ledger")
        ledger = json.loads(Path(ledger_records[0]["path"]).read_text())
        rows = [r for r in payload["inputs"] if r["kind"] == "closed-f0"]
        b8 = [r for r in payload["inputs"] if r["kind"] == "frozen-checkpoint"]
        if len(rows) != 1 or len(b8) != 1:
            raise PreflightError("Job B requires one closed-F0 input and one frozen B8 checkpoint")
        packet_path = Path(payload["outputs"]["broker_manifest"]["path"])
        packet = json.loads(packet_path.read_text())
        _validate_b_heads_runtime_recipe(payload, packet, ledger, rows, b8)
        uploads = {r["dest"]: r for r in packet["inputs"]}
        admitted_units = {(r["sha256"], str(Path(r["path"]).resolve())) for r in unit_records}
        referenced = set()
        indices = set()
        for unit in ledger["units"]:
            expected = resume.identity(
                unit.get("allocation_index"),
                contract["namespace"],
                contract["allocation"],
                ledger["fingerprint"],
            )
            if any(unit.get(key) != value for key, value in expected.items()):
                raise PreflightError("foreign resume unit identity")
            index = unit["allocation_index"]
            if index in indices:
                raise PreflightError("duplicate resume unit")
            indices.add(index)
            artifact = unit["bundle"]
            upload = uploads.get(artifact["path"], {})
            source = Path(upload.get("path", ""))
            pair = (artifact["sha256"], str(source.resolve()))
            if pair not in admitted_units or source.stat().st_size != artifact["size"]:
                raise PreflightError("resume bundle is not an admitted upload")
            referenced.add(pair)
        if referenced != admitted_units:
            raise PreflightError("unreferenced resume bundle input")


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
    _assert_safe_path(Path(packet.get("output_dir", "")).resolve(), "broker output")
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
    req = packet.get("requirements", {})
    if req.get("gpu") not in {"A100", "H100"} or req.get("allowed_gpus", [req["gpu"]]) != [
        req["gpu"]
    ]:
        raise PreflightError("sandbox broker must request explicit H100 or A100")
    if "prefer" in req:
        from research.compute.gpu_preference import variants

        variants(packet)
    commit = packet.get("commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PreflightError("broker runtime pin must be a full commit SHA")
    pinned_tree = _run(["git", "rev-parse", f"{commit}:research/colab_cli"])
    if pinned_tree not in QUALIFIED_TRANSPORT_TREES:
        raise PreflightError("broker source pin changes qualified transport")
    for source in packet.get("source_paths", []):
        _assert_safe_path(source, "broker source")
        relative = Path(source).as_posix()
        pinned = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False
        )
        current_bytes = (REPO_ROOT / source).read_bytes()
        if not pinned.returncode and pinned.stdout == current_bytes:
            continue
        if not pinned.returncode and payload.get("study") == "t34-cf-1028":
            from .ct_mac_compatibility import permits_mac_source

            if permits_mac_source(packet, relative, pinned.stdout, current_bytes):
                continue
        raise PreflightError("broker source differs from committed runtime pin: " + source)


def run_focused_tests() -> None:
    """Run the local sandbox tests with the owner-required low-impact settings."""
    env = os.environ.copy()
    env.pop("SNAKE_SANDBOX_TEST_RECEIPT", None)
    env.pop("SNAKE_SANDBOX_ALLOCATION_LEASE", None)
    env.pop("SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT", None)
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
    python_bin = (
        REPO_ROOT / "venv/bin/python"
        if (REPO_ROOT / "venv/bin/python").is_file()
        else Path(sys.executable)
    )
    command = [
        "nice",
        "-n",
        "15",
        str(python_bin),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:xdist",
        "tools/sandbox/tests",
        "research/sandbox_b_heads/tests",
    ]
    from tools.ops.processes import run
    from tools.sandbox.admission import TEST_SECONDS

    completed = run(command, cwd=REPO_ROOT, text=True, env=env, timeout=TEST_SECONDS)
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
    except (ManifestError, PreflightError, ValueError, OSError, KeyError, TypeError) as exc:
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
