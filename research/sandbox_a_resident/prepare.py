"""Prepare Job A only: committed source pin, positive registry checks, no dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from research.compute.gpu_preference import prepare_preference
from research.sandbox_a_resident.core import seed64
from research.sandbox_a_resident.units import (
    allocation,
    cell_specs,
    digest,
    load_ledger,
    unit_id,
    unit_stem,
)

ROOT = Path(__file__).resolve().parents[2]
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def runtime_source_paths(commit: str, *, repo_root: Path = ROOT) -> list[str]:
    """Ship the resident runtime and its audited CPU/CUDA dependency superset.

    Admission binds every shipped byte. Excluding unrelated research fixtures and
    benchmarks cuts repeated Git blob checks without skipping any admission check.
    """
    gpu_names = {
        "__init__.py",
        "device_sim.py",
        "stage_timing.py",
        "r2_cache.py",
        "device_policy.py",
        "anchors_device.py",
        "ego_device.py",
        "masks_device.py",
        "v8_device.py",
        "vector_device.py",
        *(
            f"{stem}.{suffix}"
            for stem in ("flood_cuda", "move_cuda", "event_cuda", "vector_geometry", "v8_indexed")
            for suffix in ("py", "cpp", "cu")
        ),
    }
    resident_names = {
        "__init__.py",
        "device_sim.py",
        "run.py",
        "core.py",
        "proof.py",
        "runtime_tools.py",
        "tensor_blocks.py",
        "observation_tensor.py",
        "units.py",
        "job.sh",
        "requirements.txt",
    }
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src", "research"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    return [
        name
        for name in tracked
        if (
            name in {"src/__init__.py", "research/__init__.py"}
            or (
                name.startswith(("src/core/", "src/model/", "src/simd_env/", "src/evaluation/"))
                and name.endswith(".py")
            )
            or (
                Path(name).parent.as_posix() == "research/gpu_sim_spike"
                and Path(name).name in gpu_names
            )
            or (
                Path(name).parent.as_posix() == "research/sandbox_a_resident"
                and Path(name).name in resident_names
            )
        )
    ]


def seed_allocations(control: dict) -> list[dict]:
    episodes = 1 + (control["cycles"] + control["warmups"]) * control["horizon"]
    return [
        {"offset": offset, "count": e * episodes, "worlds": e, "phase": p, "episode_limit": limit}
        for e, p, limit, offset in cell_specs(control)
    ]


def registry_check(control: dict[str, Any], registry_path: Path, checkpoint: Path) -> dict:
    """Check every reset allocation, including low63 engine aliases, before data."""
    from tools.sandbox.registry import load_registry, validate_input, verify_sources, world_seed

    registry = load_registry(registry_path)
    verify_sources(registry)
    validate_input(
        {"kind": "frozen-checkpoint", "path": str(checkpoint), "sha256": digest(checkpoint)},
        registry,
    )
    excluded = {int(x) & 0x7FFF_FFFF_FFFF_FFFF for x in registry["excluded_seeds"]}
    allocations = [
        *seed_allocations(control),
        {
            "offset": control.get("parity_seed_offset", 9_000_000),
            "count": control.get("parity_seed_count", 48),
        },
    ]
    if any(
        type(row.get("offset")) is not int
        or row["offset"] < 0
        or type(row.get("count")) is not int
        or row["count"] < 1
        for row in allocations
    ):
        raise ValueError("frozen seed allocations overlap or are nonpositive")
    ordered = sorted(allocations, key=lambda row: row["offset"])
    if any(
        left["offset"] + left["count"] > right["offset"]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("frozen seed allocations overlap or are nonpositive")
    count = sum(row["count"] for row in allocations)
    # Keep all-root checks, but avoid millions of Python set entries and repeated
    # validation of the same literal study slug.  The independent canonical SHA
    # stream is checked against the registry helper at its allocation boundaries.
    prefix = hashlib.sha256(b"a100-dev/sandbox-a/")
    values = np.empty(count, dtype=np.uint64)
    cursor = 0
    for row in allocations:
        for index in range(row["offset"], row["offset"] + row["count"]):
            canonical = prefix.copy()
            canonical.update(str(index).encode())
            values[cursor] = int.from_bytes(canonical.digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
            cursor += 1
        for index in (row["offset"], row["offset"] + row["count"] - 1):
            canonical = prefix.copy()
            canonical.update(str(index).encode())
            expected = int.from_bytes(canonical.digest()[:8], "big")
            if (
                seed64("a100-dev/sandbox-a", index) != expected
                or world_seed("sandbox-a", index) != expected
            ):
                raise ValueError("registry canonical seed derivation drift")
    unique = np.unique(values)
    excluded_values = np.fromiter(excluded, dtype=np.uint64, count=len(excluded))
    # Both arrays contain exact uint64 values; unique is sorted. Binary search
    # every excluded seed rather than asking isin to sort the full bank again.
    positions = np.searchsorted(unique, excluded_values)
    in_range = positions < len(unique)
    intersects = (unique[positions[in_range]] == excluded_values[in_range]).any()
    if unique[0] == 0 or len(unique) != count or intersects:
        raise ValueError("excluded/aliased development seed")
    return {
        "status": "PASS",
        "allocated_uint64_seeds": count,
        "effective_seed_bits": 63,
        "excluded_intersection": 0,
        "aliases": 0,
        "registry_sha256": digest(registry_path),
    }


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(
    packet: Path,
    commit: str,
    checkpoint: Path,
    registry: Path,
    estimate_seconds: float,
    *,
    variant: str = "a3-03",
) -> None:
    """Write the full broker packet plus exactly ten physical sandbox JSON lines."""
    if not math.isfinite(2 * estimate_seconds) or estimate_seconds <= 0:
        raise ValueError("positive finite engineering forecast required")
    variants = {
        "a3-eager-01": (
            "none",
            240_000_000,
            320_000_000,
            "a3-eager-attempt-01",
            "sandbox-a-resident-a3-eager-20261026-01",
        ),
        "a3-03": (
            "default",
            360_000_000,
            440_000_000,
            "a3-attempt-03",
            "sandbox-a-resident-a3-20261026-03",
        ),
    }
    if variant not in variants:
        raise ValueError("unsupported packet variant")
    compile_mode, seed_base, proof_base, attempt, job_id = variants[variant]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if digest(checkpoint) != B8_SHA:
        raise ValueError("published B8 s3 hash differs")
    transport = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=ROOT, text=True
    ).strip()
    if transport != "5833e54e9c5396121cbbf99a5cd4e52006e564fd":
        raise ValueError("qualified transport changed")
    packet.mkdir(parents=True, exist_ok=False)
    control = {
        "source_commit": commit,
        "checkpoint_sha256": B8_SHA,
        "recipe_parameters": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip": 0.2,
            "learning_rate": 0.0003,
            "stagger_initial_age": True,
        },
        "worlds": [128, 1024, 2048, 4096],
        "horizon": 128,
        "minibatch": 2048,
        "epochs": 1,
        "cycles": 3,
        "warmups": 1,
        "episode_limits": [64, 512],
        "recipe_version": "A-3",
        "compile_mode": compile_mode,
        "seed_offset_base": seed_base,
        "allocation_stride": 3_000_000,
        "parity_seed_offset": proof_base,
        "parity_seed_count": 54,
        "tiny": False,
        "timeout_seconds": estimate_seconds * 2,
    }
    check = registry_check(control, registry, checkpoint)
    write_json(packet / "control.json", control)
    write_json(packet / "registry-check.json", check)
    ledger = {
        "schema": "sandbox-a-ledger/v1",
        "namespace": "a100-dev/sandbox-a",
        "allocation": allocation(control),
        "units": [],
    }
    write_json(packet / "verified-ledger.json", ledger)
    verified = load_ledger(packet / "verified-ledger.json", control)
    required = ["assay/summary.json", "assay/parity.json"]
    for worlds, phase, limit, _ in cell_specs(control):
        if unit_id(worlds, phase, limit) not in verified:
            stem = unit_stem(worlds, phase, limit)
            required += [f"assay/{stem}.{suffix}" for suffix in ("tar", "json")]
    output = (
        Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident")
        / attempt
        / "output"
    )
    manifest = {
        "job_id": job_id,
        "commit": commit,
        "script": "research/sandbox_a_resident/job.sh",
        "device": "cuda",
        "source_paths": runtime_source_paths(commit),
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": estimate_seconds / 60,
        },
        "timeout_minutes": estimate_seconds / 30,
        "output_dir": str(output),
        "max_export_bytes": 2 * 1024**3,
        "inputs": [
            {"path": str(checkpoint.resolve()), "dest": "b8-s3.pth", "sha256": B8_SHA},
            {"path": str(registry.resolve()), "dest": "registry.json", "sha256": digest(registry)},
            *[
                {
                    "path": str((packet / name).resolve()),
                    "dest": name,
                    "sha256": digest(packet / name),
                }
                for name in ("verified-ledger.json",)
            ],
        ],
        "env": {
            "SANDBOX_A_CONTROL_JSON": json.dumps(control, sort_keys=True),
            "COLAB_CLI_REQUIREMENTS": "research/sandbox_a_resident/requirements.txt",
            "SANDBOX_A_SOURCE_COMMIT": commit,
            "SNAKE_DQN_DEVICE": "cuda",
            "MAX_JOBS": "1",
        },
        "required_exports": required,
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": 2 * estimate_seconds * 8.0 / 3600,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "resident-phase-full-state",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str((packet / "sandbox.json").resolve()),
        },
        "sandbox_execution": "finite resident timing/parity assay; adopted sandbox admission",
        "requirements_file": "research/sandbox_a_resident/requirements.txt",
        "sandbox_a": {
            "seed_allocations": seed_allocations(control),
            "parity_seed_allocation": {
                "offset": control.get("parity_seed_offset", 9_000_000),
                "count": control.get("parity_seed_count", 48),
            },
            "numerical_outputs": "timing/parity only",
        },
        "technical_envelope": {
            "base_seconds": estimate_seconds,
            "effective_seconds": 2 * estimate_seconds,
            "endpoint": "3 cycles per phase, shape and episode limit; one excluded warmup",
        },
    }
    prepare_preference(manifest)
    write_json(packet / "dispatch-manifest.json", manifest)
    world_roots = [
        i
        for row in seed_allocations(control)
        for i in range(row["offset"], row["offset"] + row["worlds"])
    ]
    payload = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": "sandbox-a",
        "inputs": [
            {
                "kind": "worlds",
                "study": "sandbox-a",
                "indices": world_roots,
                "seeds": [seed64("a100-dev/sandbox-a", i) for i in world_roots],
            },
            {"kind": "frozen-checkpoint", "path": str(checkpoint.resolve()), "sha256": B8_SHA},
            {
                "kind": "verified-ledger",
                "path": str((packet / "verified-ledger.json").resolve()),
                "sha256": digest(packet / "verified-ledger.json"),
            },
            {"kind": "registry", "path": str(registry.resolve()), "sha256": digest(registry)},
        ],
        "salt": "a100-dev/sandbox-a/",
        "endpoint": {
            "kind": "steps",
            "value": control["cycles"]
            * control["horizon"]
            * sum(control["worlds"])
            * 3
            * len(control["episode_limits"]),
        },
        "timeout": {
            "base_seconds": estimate_seconds,
            "seconds": 2 * estimate_seconds,
            "purpose": "runaway-guard",
        },
        "excluded_studies": [
            "F5",
            "F4-fold-0",
            "29-census-parents",
            "R30",
            "strict",
            "probe",
            "serving",
            "D4096",
            "t34-round0-study-01",
        ],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": [str(output)],
            "broker_manifest": {
                "path": str((packet / "dispatch-manifest.json").resolve()),
                "sha256": digest(packet / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    pairs = list(payload.items())
    # Ten keys, ten lines: opening brace shares first key; closing shares last.
    lines = [
        ("{" if i == 0 else " ")
        + json.dumps(k)
        + ": "
        + json.dumps(v, separators=(",", ":"))
        + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(pairs)
    ]
    (packet / "sandbox.json").write_text("\n".join(lines) + "\n")
    print(f"PREPARED {packet} seeds={check['allocated_uint64_seeds']} allocations=0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--variant", choices=("a3-eager-01", "a3-03"), default="a3-03")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    )
    parser.add_argument(
        "--base-seconds", "--estimate-seconds", dest="estimate_seconds", type=float, required=True
    )
    args = parser.parse_args()
    prepare(
        args.packet.resolve(),
        args.commit,
        args.checkpoint,
        args.registry,
        args.estimate_seconds,
        variant=args.variant,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
