"""Seal the independent H5000 natural-support study; never allocate a runtime."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.sandbox_a_resident import ct_json as json
from research.sandbox_a_resident.ct_h5000.export_projection import project_stage_bytes
from research.sandbox_a_resident.ct_h5000.task import (
    CT_DOMAIN_PREFIX,
    CT_STUDY_SLUG,
    CTControl,
    ct_seed_allocations,
    train01_protected_eval_seeds,
)
from research.sandbox_a_resident.ct_h5000.units import empty_ledger
from research.sandbox_a_resident.ct_topology import TOPOLOGY_SPEC
from research.sandbox_a_resident.prepare import runtime_source_paths
from tools.sandbox.registry import (
    TRAINING_OPPONENTS,
    digest,
    load_registry,
    validate_input,
    verify_sources,
    world_seed,
)

ROOT = Path(__file__).resolve().parents[3]
STUDY = CT_STUDY_SLUG
ARTIFACT_ROOT = ROOT.parent / "snake-dqn-artifacts/t34-ct-h5000-1029/attempt-01"


def write_once(path: Path, value: dict[str, Any]) -> None:
    """Create JSON without replacing prior evidence."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ct_endpoint_steps(control: CTControl | Mapping[str, Any]) -> int:
    """Return the finite collection-only endpoint; no branches or fits are dispatched."""
    ctrl = control if isinstance(control, CTControl) else CTControl.from_dict(control)
    collection_steps = int(ctrl.worlds) * int(ctrl.collection_ticks)
    return int(collection_steps)


def source_paths(commit: str) -> list[str]:
    """Return sparse runtime source closure for C/T (excluding offline seal/test tools)."""
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src", "research/sandbox_a_resident"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    excluded_resident = {
        "training_prepare.py",
        "training_preflight.py",
        "ct_prepare.py",
        "ct_preflight.py",
        "ct_timing_probe.py",
    }
    additions = [
        name
        for name in tracked
        if (name.startswith("src/") and name.endswith(".py"))
        or (
            Path(name).parent.as_posix() == "research/sandbox_a_resident"
            and (
                Path(name).name in {"train.py", "train_job.sh", "ct_job.sh", "stall_watchdog.py"}
                or (
                    (Path(name).name.startswith("training_") or Path(name).name.startswith("ct_"))
                    and Path(name).name.endswith(".py")
                    and Path(name).name not in excluded_resident
                )
            )
        )
    ]
    successor = [
        name
        for name in tracked
        if name.startswith("research/sandbox_a_resident/ct_h5000/")
        and name.endswith((".py", ".sh"))
        and not Path(name).name.startswith("test_")
        and Path(name).name not in {"prepare.py", "preflight.py"}
    ]
    return sorted(
        set(runtime_source_paths(commit))
        | set(additions)
        | set(successor)
        | {
            "research/sandbox_a_resident/ct_h5000/PREREGISTRATION.md",
            "research/sandbox_a_resident/ct_h5000/PREREGISTRATION_FIXES_20261029.md",
        }
    )


def audit_ct_seeds(allocations: list[dict[str, Any]], registry_path: Path) -> dict[str, Any]:
    """Exhaustively audit 63-bit aliases, protected banks, and TRAIN-01 monitor/nomination."""
    registry = load_registry(registry_path)
    verify_sources(registry)
    ordered = sorted(allocations, key=lambda row: row["offset"])
    if {row.get("phase") for row in ordered} != {"training", "validation", "assessment"}:
        raise ValueError("C/T requires distinct training, validation, and assessment seed domains")
    for row in ordered:
        if type(row["offset"]) is not int or type(row["count"]) is not int:
            raise ValueError("integer seed allocations required")
        if row["offset"] < 0 or row["count"] < 1:
            raise ValueError("positive finite seed allocation required")
    if any(a["offset"] + a["count"] > b["offset"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("overlapping C/T seed allocations")

    count = sum(row["count"] for row in ordered)
    prefix = hashlib.sha256(f"{CT_DOMAIN_PREFIX}/".encode("utf-8"))
    seed_hash = hashlib.sha256()
    train01_protected = train01_protected_eval_seeds()
    values = np.empty(count, dtype=np.uint64)
    cursor = 0
    for row in ordered:
        for index in range(row["offset"], row["offset"] + row["count"]):
            key = prefix.copy()
            key.update(str(index).encode("utf-8"))
            raw = key.digest()[:8]
            seed_hash.update(raw)
            values[cursor] = int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF
            cursor += 1
    values.sort()
    if values[0] == 0:
        raise ValueError("zero effective seed in C/T domain")
    for start in range(0, count - 1, 1_000_000):
        end = min(count - 1, start + 1_000_000)
        if np.any(values[start:end] == values[start + 1 : end + 1]):
            raise ValueError("effective 63-bit seed alias in C/T domain")
    for seed in registry["excluded_seeds"]:
        effective = int(seed) & 0x7FFF_FFFF_FFFF_FFFF
        idx = int(np.searchsorted(values, effective))
        if idx < count and int(values[idx]) == effective:
            raise ValueError("protected registry seed intersection in C/T domain")
    for seed in train01_protected:
        effective = int(seed) & 0x7FFF_FFFF_FFFF_FFFF
        idx = int(np.searchsorted(values, effective))
        if idx < count and int(values[idx]) == effective:
            raise ValueError("TRAIN-01 monitor/nomination seed intersection in C/T domain")
    predecessor_intersection = 0
    predecessor = hashlib.sha256(b"a100-dev/t34-cf-1028/")
    for offset, n in ((0, 2100020), (10000000, 1024), (20000000, 522444)):
        block = np.empty(n, dtype=np.uint64)
        for j in range(n):
            h = predecessor.copy()
            h.update(str(offset + j).encode())
            block[j] = int.from_bytes(h.digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
        indices = np.searchsorted(values, block)
        mask = indices < count
        predecessor_intersection += int(np.count_nonzero(values[indices[mask]] == block[mask]))
    if predecessor_intersection:
        raise ValueError("predecessor C/T effective seed intersection")
    del values

    return {
        "schema": "t34-cf-seed-audit/v1",
        "status": "PASS",
        "study": STUDY,
        "namespace": f"{CT_DOMAIN_PREFIX}/",
        "allocations": allocations,
        "count": count,
        "canonical_uint64_sha256": seed_hash.hexdigest(),
        "registry_sha256": digest(registry_path),
        "effective_seed_bits": 63,
        "aliases": 0,
        "excluded_intersection": 0,
        "train01_protected_count": len(train01_protected),
        "train01_excluded_intersection": 0,
        "predecessor_count": 2623488,
        "predecessor_intersection": predecessor_intersection,
    }


def prepare(
    packet: Path,
    commit: str,
    checkpoints: dict[str, Path],
    registry: Path,
    *,
    seed_audit: Path | None = None,
    allocation_index: int = 0,
    seed_offset: int = 0,
    job_id: str = "sandbox-ct-h5000-20261029-01",
    output_dir: Path | None = None,
    est_minutes: int = 180,
    timeout_minutes: int = 360,
    unit_cap: float = 72.0,
    sandbox_source_delivery: str | None = "background-sparse-v1",
    stall_timeout_seconds: int | None = 900,
    prefer_h100: bool = True,
) -> None:
    """Create a finite C/T packet and ten-line adopted sandbox manifest."""
    if (
        type(est_minutes) is not int
        or est_minutes < 1
        or type(timeout_minutes) is not int
        or timeout_minutes != 2 * est_minutes
    ):
        raise ValueError("timeout_minutes must equal 2x integer est_minutes")
    base_seconds = est_minutes * 60
    effective_seconds = timeout_minutes * 60
    resolved_output = ARTIFACT_ROOT / "output" if output_dir is None else output_dir
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()

    ctrl_obj = CTControl(
        study=STUDY,
        source_commit=commit,
        allocation_index=allocation_index,
        seed_offset=seed_offset,
        tiny=False,
    )
    control = ctrl_obj.to_dict()
    endpoint_steps = ct_endpoint_steps(ctrl_obj)

    for name, path in checkpoints.items():
        validate_input(
            {
                "kind": "frozen-checkpoint" if name == "b8-s3" else "frozen-opponent",
                "path": str(path),
                "sha256": digest(path),
            }
        )
    if {digest(p) for name, p in checkpoints.items() if name != "b8-s3"} != TRAINING_OPPONENTS:
        raise ValueError("both canonical opponent checkpoints required")

    allocations = ct_seed_allocations(ctrl_obj)
    check = (
        audit_ct_seeds(allocations, registry)
        if seed_audit is None
        else json.loads(seed_audit.read_text())
    )
    if (
        check.get("schema") != "t34-cf-seed-audit/v1"
        or check.get("status") != "PASS"
        or check.get("study") != STUDY
        or check.get("allocations") != allocations
        or check.get("registry_sha256") != digest(registry)
        or check.get("aliases") != 0
        or check.get("excluded_intersection") != 0
        or check.get("train01_protected_count") != len(train01_protected_eval_seeds())
        or check.get("train01_excluded_intersection") != 0
        or check.get("predecessor_count") != 2623488
        or check.get("predecessor_intersection") != 0
    ):
        raise ValueError("C/T seed audit does not bind this exact allocation and registry")

    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", control)
    write_once(packet / "export-projection.json", project_stage_bytes())
    write_once(packet / "registry-check.json", check)
    write_once(packet / "topology-manifest.json", dict(TOPOLOGY_SPEC))
    write_once(packet / "verified-ledger.json", empty_ledger(control))

    uploads = [
        {"path": str(path.resolve()), "dest": name + ".pth", "sha256": digest(path)}
        for name, path in checkpoints.items()
    ] + [
        {"path": str(registry.resolve()), "dest": "registry.json", "sha256": digest(registry)},
        {
            "path": str((packet / "verified-ledger.json").resolve()),
            "dest": "verified-ledger.json",
            "sha256": digest(packet / "verified-ledger.json"),
        },
    ]
    env = {
        "SANDBOX_CT_CONTROL_JSON": json.dumps(control, sort_keys=True),
        "SANDBOX_A_SOURCE_COMMIT": commit,
        "SNAKE_DQN_DEVICE": "cuda",
        "MAX_JOBS": "1",
        "COLAB_CLI_REQUIREMENTS": "research/sandbox_a_resident/requirements.txt",
    }
    if stall_timeout_seconds is not None:
        if type(stall_timeout_seconds) is not int or not 0 < stall_timeout_seconds <= 900:
            raise ValueError("stall_timeout_seconds must be an integer in 1..900")
        env["SANDBOX_A_STALL_TIMEOUT_SECONDS"] = str(stall_timeout_seconds)

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": commit,
        "script": "research/sandbox_a_resident/ct_h5000/job.sh",
        "device": "cuda",
        "source_paths": source_paths(commit),
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": est_minutes,
        },
        "timeout_minutes": timeout_minutes,
        "output_dir": str(resolved_output),
        "max_export_bytes": 1536 * 1024**2,
        "inputs": uploads,
        "env": env,
        "required_exports": [
            "ct/summary.json",
            "ct/deaths.jsonl.gz",
            "ct/length-histograms.jsonl.gz",
            "ct/replay-ties.jsonl.gz",
        ],
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": float(unit_cap),
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "ct-stage-snapshot",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str((packet / "sandbox.json").resolve()),
        },
        "sandbox_execution": (
            "finite H5000 natural collection/support successor only; " "no Watch v2 transfer claim"
        ),
        "requirements_file": "research/sandbox_a_resident/requirements.txt",
        "sandbox_a": {
            "seed_allocations": allocations,
            "seed_audit_sha256": digest(packet / "registry-check.json"),
            "topology_manifest_sha256": digest(packet / "topology-manifest.json"),
        },
        "technical_envelope": {
            "base_seconds": base_seconds,
            "effective_seconds": effective_seconds,
            "endpoint": (f"{endpoint_steps} collection rows; fixed H5000 or 13200s admission cap"),
        },
    }
    if sandbox_source_delivery is not None:
        if sandbox_source_delivery != "background-sparse-v1":
            raise ValueError("sandbox_source_delivery must be background-sparse-v1")
        manifest["sandbox_source_delivery"] = sandbox_source_delivery
    if prefer_h100:
        from research.compute.gpu_preference import prepare_preference

        prepare_preference(manifest)
    write_once(packet / "dispatch-manifest.json", manifest)

    roots = [row["offset"] for row in allocations]
    inputs: list[dict[str, Any]] = [
        {
            "kind": "worlds",
            "study": STUDY,
            "indices": roots,
            "seeds": [world_seed(STUDY, i) for i in roots],
        }
    ]
    for row in uploads:
        dest = row["dest"]
        kind = (
            "frozen-checkpoint"
            if dest == "b8-s3.pth"
            else (
                "registry"
                if dest == "registry.json"
                else "verified-ledger" if dest == "verified-ledger.json" else "frozen-opponent"
            )
        )
        inputs.append({"kind": kind, "path": row["path"], "sha256": row["sha256"]})

    sandbox = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": inputs,
        "salt": f"{CT_DOMAIN_PREFIX}/",
        "endpoint": {"kind": "steps", "value": endpoint_steps},
        "timeout": {
            "base_seconds": base_seconds,
            "seconds": effective_seconds,
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
            "sandbox-a-train01-monitor",
            "sandbox-a-train01-nomination",
        ],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": [manifest["output_dir"]],
            "broker_manifest": {
                "path": str((packet / "dispatch-manifest.json").resolve()),
                "sha256": digest(packet / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    lines = [
        ("{" if i == 0 else " ") + json.dumps(k) + ": " + json.dumps(v) + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(sandbox.items())
    ]
    with (packet / "sandbox.json").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    from research.compute.gpu_preference import variants

    write_once(packet / "a100-fallback-preview.json", variants(manifest)[1])
    components = (
        "export-projection.json",
        "control.json",
        "registry-check.json",
        "topology-manifest.json",
        "verified-ledger.json",
        "dispatch-manifest.json",
        "sandbox.json",
        "a100-fallback-preview.json",
    )
    write_once(
        packet / "SEAL.json",
        {
            "schema": "t34-ct-h5000-prep-seal/v1",
            "source_commit": commit,
            "components": {name: digest(packet / name) for name in components},
            "source_sha256": {
                name: hashlib.sha256(
                    subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
                ).hexdigest()
                for name in manifest["source_paths"]
            },
        },
    )
    print(f"PREPARED_CTH5000 pin={commit} seeds={check['count']} allocations=0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--allocation-index", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--job-id", default="sandbox-ct-h5000-20261029-01")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--est-minutes", type=int, default=180)
    parser.add_argument("--timeout-minutes", type=int, default=360)
    parser.add_argument("--unit-cap", type=float, default=72.0)
    parser.add_argument("--sandbox-source-delivery", default="background-sparse-v1")
    parser.add_argument("--stall-timeout-seconds", type=int, default=900)
    parser.add_argument("--prefer-h100", action="store_true", default=True)
    args = parser.parse_args()
    prepare(
        args.packet.resolve(),
        args.commit,
        {
            "b8-s3": args.checkpoint_dir / "b8_s3_t7_20261016.pth",
            "frp3-s12": args.checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth",
            "champion": args.checkpoint_dir / "champion_a5_freespace_20260621.pth",
        },
        ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
        seed_audit=args.seed_audit,
        allocation_index=args.allocation_index,
        seed_offset=args.seed_offset,
        job_id=args.job_id,
        output_dir=None if args.output_dir is None else args.output_dir.resolve(),
        est_minutes=args.est_minutes,
        timeout_minutes=args.timeout_minutes,
        unit_cap=args.unit_cap,
        sandbox_source_delivery=args.sandbox_source_delivery,
        stall_timeout_seconds=args.stall_timeout_seconds,
        prefer_h100=args.prefer_h100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
