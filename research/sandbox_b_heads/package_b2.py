"""Build a source-bound Job B-2 dispatch packet without dispatching it."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from research.bl_residual.data import validate_rows as validate_projection
from research.sandbox_b_heads.core_b2 import CONFIGS
from research.sandbox_b_heads.job import SALT, parent_split, sha256
from research.sandbox_b_heads.resume_b2 import recipe, verify_ledger

RUNAWAY_BASE_MINUTES = 60.0
RUNAWAY_TIMEOUT_MINUTES = 120.0


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _repository_path(path: Path) -> str:
    """Return a source-root-relative packet path when the target is in this checkout."""
    root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _write_ten_line_sandbox_manifest(
    path: Path,
    data: Path,
    registry: Path,
    b8: Path,
    ledger: Path,
    broker: Path,
    base_timeout_minutes: float,
    prior_bundles: list[dict] | None = None,
) -> None:
    """Emit the human-reviewable ten-physical-line sandbox manifest."""
    payload = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": "sandbox-b-heads-v2",
        "inputs": [
            {
                "kind": "closed-f0",
                "dose": "D1024",
                "path": str(data.resolve()),
                "sha256": sha256(data),
            },
            {"kind": "frozen-checkpoint", "path": str(b8.resolve()), "sha256": sha256(b8)},
            {"kind": "verified-ledger", "path": str(ledger.resolve()), "sha256": sha256(ledger)},
            {"kind": "registry", "path": str(registry.resolve()), "sha256": sha256(registry)},
        ],
        "salt": SALT,
        "endpoint": {
            "kind": "updates",
            "value": sum(config.updates for config in CONFIGS),
            "broker_manifest": _repository_path(broker),
        },
        "timeout": {
            "base_seconds": base_timeout_minutes * 60,
            "seconds": base_timeout_minutes * 120,
            "purpose": "runaway-guard",
        },
        "excluded_studies": ["F5", "F4-fold-0", "R30", "strict", "probe", "serving"],
        "owner_rule": (
            "docs/research/governance_a100_sandbox_20261026/ADOPTED.md"
            "#2-governance-adopt-this-rule-text"
        ),
        "outputs": {
            "label": "simulator-D",
            "paths": ["checkpoints", "results.json", "timing.json"],
            "broker_manifest": {"path": str(broker.resolve()), "sha256": sha256(broker)},
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    payload["inputs"].extend(prior_bundles or [])
    groups = [
        ("schema", "study"),
        ("inputs",),
        ("salt",),
        ("endpoint",),
        ("timeout",),
        ("excluded_studies",),
        ("owner_rule", "outputs"),
        ("crossing",),
    ]
    lines = ["{"]
    for index, group in enumerate(groups):
        fragment = ", ".join(
            f"{json.dumps(key)}: {json.dumps(payload[key], sort_keys=True)}" for key in group
        )
        lines.append(fragment + ("," if index + 1 < len(groups) else ""))
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def build(
    data: Path,
    registry: Path,
    b8: Path,
    ledger: Path,
    manifest_output: Path,
    output_dir: Path,
    base_timeout_minutes: float,
    sandbox_manifest: Path | None = None,
) -> Path:
    """Validate a filtered input against the canonical registry and write a packet."""
    # Registry admission is intentionally before loading any projected outcome arrays.
    from tools.sandbox.registry import load_registry, validate_npz, validate_rows

    admitted = load_registry(registry)
    # This verifies the registered file hash and exact row/life/parent admission
    # before this package opens any numerical or outcome-bearing arrays.
    validate_npz(data, registry=admitted, dose="D1024")
    with np.load(data, allow_pickle=False) as archive:
        identity = {key: archive[key] for key in ("row_id", "life_id", "parent_world")}
        table = {key: archive[key] for key in archive.files}
    validate_rows(
        [
            {"row_id": str(row), "life_id": str(life), "parent_world": str(parent)}
            for row, life, parent in zip(
                identity["row_id"], identity["life_id"], identity["parent_world"]
            )
        ],
        registry=admitted,
        dose="D1024",
    )
    validate_projection(table)
    validation = parent_split(table["parent_world"])
    if not validation.any() or validation.all():
        raise ValueError("filtered D1024 input does not support the committed parent split")
    if base_timeout_minutes != RUNAWAY_BASE_MINUTES or not ledger.is_file():
        raise ValueError("sandbox B requires the owner-adopted 60-minute runaway-guard base")
    if not output_dir.is_absolute() or output_dir.exists():
        raise ValueError("output must be a fresh absolute output directory path")
    targets = [
        manifest_output,
        manifest_output.with_suffix(manifest_output.suffix + ".sha256"),
        manifest_output.with_name(manifest_output.stem + "-runtime-ledger.json"),
    ]
    if sandbox_manifest is not None:
        targets.append(sandbox_manifest)
    if any(target.exists() for target in targets):
        raise ValueError("packet outputs must be fresh; preserve prior immutable packets")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    verified = verify_ledger(ledger, recipe(data, b8))
    runtime_ledger = manifest_output.with_name(manifest_output.stem + "-runtime-ledger.json")
    prior_inputs, sandbox_prior = [], []
    for unit in verified["units"]:
        original = Path(unit["bundle"]["path"])
        if not original.is_absolute():
            original = ledger.parent / original
        destination = f"resume-{unit['allocation_index']:02d}-{unit['id']}.tar"
        prior_inputs.append(
            {
                "path": str(original.resolve()),
                "dest": destination,
                "sha256": unit["bundle"]["sha256"],
            }
        )
        sandbox_prior.append(
            {
                "kind": "verified-unit",
                "path": str(original.resolve()),
                "sha256": unit["bundle"]["sha256"],
            }
        )
        unit["bundle"]["path"] = destination
    runtime_ledger.write_text(json.dumps(verified, sort_keys=True, indent=2) + "\n")
    manifest = {
        "job_id": "sandbox-b-heads-v2",
        "commit": _git_commit(),
        "script": "research/sandbox_b_heads/run_b2.sh",
        "source_paths": [
            "research/sandbox_b_heads/core_b2.py",
            "research/sandbox_b_heads/model_b2.py",
            "research/sandbox_b_heads/job_b2.py",
            "research/sandbox_b_heads/resume_b2.py",
            "research/sandbox_b_heads/run_b2.sh",
            "research/sandbox_b_heads/__init__.py",
            "research/sandbox_b_heads/model.py",
            "research/sandbox_b_heads/job.py",
            "research/sandbox_b_heads/resume.py",
            "research/sandbox_b_heads/run_job.sh",
            "research/sandbox_b_heads/requirements.txt",
            "research/a100_a2/model.py",
            "research/bl_residual/data.py",
            "src/__init__.py",
            "src/model/__init__.py",
            "src/model/base_network.py",
            "src/model/apex_network.py",
            "src/model/checkpoint_manager.py",
            "src/model/checkpoint_io.py",
            "src/model/ego2s_network.py",
            "src/model/obs_spec.py",
            "src/core/runtime_contract.py",
            "src/core/__init__.py",
            "src/core/config_loader.py",
            "src/core/game_config.py",
            "src/core/device_manager.py",
            "tools/sandbox/__init__.py",
            "tools/sandbox/registry.py",
        ],
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 4,
            # Compatibility estimate required by the qualified supervisor.  It
            # deliberately names the fixed technical guard rather than a CPU
            # smoke extrapolation or an expected GPU duration.
            "est_minutes": base_timeout_minutes,
        },
        "timeout_minutes": RUNAWAY_TIMEOUT_MINUTES,
        "output_dir": str(output_dir),
        "inputs": [
            {
                "path": str(data.resolve()),
                "dest": "d1024-admitted.npz",
                "sha256": sha256(data),
            },
            {"path": str(b8.resolve()), "dest": "b8.pth", "sha256": sha256(b8)},
            {
                "path": str(registry.resolve()),
                "dest": "SANDBOX_INPUT_REGISTRY.json",
                "sha256": sha256(registry),
            },
            {
                "path": str(runtime_ledger.resolve()),
                "dest": "verified-ledger.json",
                "sha256": sha256(runtime_ledger),
            },
        ],
        "required_exports": [
            "checkpoints.tar",
            "results.json",
            "results.csv",
            "pareto.json",
            "verified-ledger.json",
            "timing.json",
        ],
        "resume": {
            "unit_kind": "head-loss-config",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "env": {
            "SANDBOX_B_SALT": SALT,
            "SNAKE_DQN_DEVICE": "cuda",
            "COLAB_CLI_REQUIREMENTS": "research/sandbox_b_heads/requirements.txt",
        },
        "sandbox_execution": (
            "entire finite development sweep; no timing canary under the adopted sandbox rule"
        ),
        "requirements_file": "research/sandbox_b_heads/requirements.txt",
        # The qualified supervisor requires unit_cap.  This is the doubled wall-time
        # runaway guard at a prospective A100 pricing ceiling, not an economic allocation cap.
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": RUNAWAY_TIMEOUT_MINUTES * 8.0 / 60,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "technical_envelope": {
            "base_timeout_minutes": base_timeout_minutes,
            "effective_timeout_minutes": RUNAWAY_TIMEOUT_MINUTES,
            "includes_setup_and_export": True,
            "purpose": "runaway-guard",
        },
    }
    manifest["inputs"].extend(prior_inputs)
    if sandbox_manifest is not None:
        manifest["sandbox_admission"] = {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str(sandbox_manifest.resolve()),
        }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_output.with_suffix(manifest_output.suffix + ".sha256").write_text(
        f"{sha256(manifest_output)}  {manifest_output.name}\n"
    )
    if sandbox_manifest is not None:
        _write_ten_line_sandbox_manifest(
            sandbox_manifest,
            data,
            registry,
            b8,
            runtime_ledger,
            manifest_output,
            base_timeout_minutes,
            sandbox_prior,
        )
    return manifest_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--registry", default=Path("tools/sandbox/SANDBOX_INPUT_REGISTRY.json"), type=Path
    )
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sandbox-manifest", type=Path)
    parser.add_argument(
        "--runaway-base-minutes",
        type=float,
        default=RUNAWAY_BASE_MINUTES,
        help="fixed owner-adopted technical runaway-guard base (must be 60)",
    )
    args = parser.parse_args()
    print(
        build(
            args.data,
            args.registry,
            args.b8,
            args.ledger,
            args.output,
            args.output_dir,
            args.runaway_base_minutes,
            args.sandbox_manifest,
        )
    )


if __name__ == "__main__":
    main()
