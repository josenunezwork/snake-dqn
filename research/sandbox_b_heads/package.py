"""Build a source-bound Job B dispatch packet without dispatching it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from research.bl_residual.data import validate_rows as validate_projection
from research.sandbox_b_heads.job import SALT, parent_split, sha256


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_ten_line_sandbox_manifest(
    path: Path,
    data: Path,
    registry: Path,
    b8: Path,
    ledger: Path,
    broker: Path,
    estimated_minutes: float,
) -> None:
    """Emit the human-reviewable ten-physical-line sandbox manifest."""
    payload = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": "sandbox-b-heads-v1",
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
            "value": 15000,
            "broker_manifest": "research/sandbox_b_heads/dispatch-manifest.json",
        },
        "timeout": {
            "measured_estimate_seconds": estimated_minutes * 60,
            "seconds": estimated_minutes * 120,
        },
        "excluded_studies": ["F5", "F4-fold-0", "R30", "strict", "probe", "serving"],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md#2-governance-adopt-this-rule-text",
        "outputs": {
            "label": "simulator-D",
            "paths": ["checkpoints", "results.json", "timing.json"],
            "broker_manifest": {"path": str(broker.resolve()), "sha256": sha256(broker)},
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
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
    output: Path,
    estimated_minutes: float,
    sandbox_manifest: Path | None = None,
) -> Path:
    """Validate a filtered input against the canonical registry and write a packet."""
    # Registry admission is intentionally before loading any projected outcome arrays.
    from tools.sandbox.registry import load_registry, validate_rows

    admitted = load_registry(registry)
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
    if estimated_minutes <= 0 or not ledger.is_file():
        raise ValueError("measured estimate and an initial verified ledger are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job_id": "sandbox-b-heads-v1",
        "commit": _git_commit(),
        "script": "research/sandbox_b_heads/run_job.sh",
        "source_paths": [
            "research/sandbox_b_heads/__init__.py",
            "research/sandbox_b_heads/model.py",
            "research/sandbox_b_heads/job.py",
            "research/sandbox_b_heads/run_job.sh",
            "research/a100_a2/model.py",
            "research/bl_residual/data.py",
        ],
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 4,
            "est_minutes": estimated_minutes,
        },
        "timeout_minutes": estimated_minutes * 2,
        "output_dir": "REPLACE_WITH_FRESH_ABSOLUTE_OUTPUT_DIR",
        "inputs": [
            {
                "path": str(data.resolve()),
                "dest": "inputs/d1024-admitted.npz",
                "sha256": sha256(data),
            },
            {"path": str(b8.resolve()), "dest": "inputs/b8.pth", "sha256": sha256(b8)},
            {
                "path": str(registry.resolve()),
                "dest": "inputs/SANDBOX_INPUT_REGISTRY.json",
                "sha256": sha256(registry),
            },
            {
                "path": str(ledger.resolve()),
                "dest": "inputs/verified-ledger.json",
                "sha256": sha256(ledger),
            },
        ],
        "required_exports": ["results.json", "results.json.sha256", "verified-ledger.json"],
        "resume": {
            "unit_kind": "head-loss-config",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "env": {"SANDBOX_B_SALT": SALT, "SNAKE_DQN_DEVICE": "cuda"},
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    if sandbox_manifest is not None:
        _write_ten_line_sandbox_manifest(
            sandbox_manifest, data, registry, b8, ledger, output, estimated_minutes
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--registry", default=Path("tools/sandbox/SANDBOX_INPUT_REGISTRY.json"), type=Path
    )
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sandbox-manifest", type=Path)
    parser.add_argument("--estimated-minutes", required=True, type=float)
    args = parser.parse_args()
    print(
        build(
            args.data,
            args.registry,
            args.b8,
            args.ledger,
            args.output,
            args.estimated_minutes,
            args.sandbox_manifest,
        )
    )


if __name__ == "__main__":
    main()
