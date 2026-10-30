"""Seal the spatial pilot packet offline; never allocate or submit Colab work.

The 15-minute integrated ruler is also the genuine target-platform family
canary. Both modes upload exactly the same input bytes and run ``job.sh``.
The pilot stays unqualified until both hardware receipts are bound and pass
the unchanged supervisor canary gate. No stand-in receipt is generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from research.colab_cli.canary import SCHEMA, family
from research.compute.gpu_preference import prepare_preference, variants

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/spatial_student_pilot"
SCRIPT = f"{PACKAGE}/job.sh"
REQUIREMENTS = f"{PACKAGE}/requirements.txt"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
OPPONENT_SHAS = {
    "frp3-s12.pth": "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
    "champion.pth": "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
}
CHECKPOINT_NAMES = {
    "b8-s3.pth": "b8_s3_t7_20261016.pth",
    "frp3-s12.pth": "frp3_m3_s12_u60000_20261005.pth",
    "champion.pth": "champion_a5_freespace_20260621.pth",
}


def digest(path: Path) -> str:
    """Hash bytes without decoding a study outcome."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, payload: dict[str, Any]) -> None:
    """Create evidence without overwriting prior preparation."""
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def runtime_source_paths(commit: str) -> list[str]:
    """Select pinned source, avoiding unrelated research artifacts and fixtures.

    The simulator's lazy CPU paths are included for parity/import qualification;
    the source archive contains code only, never a historical outcome bank.
    """
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True, timeout=30
    ).splitlines()
    trees = (
        "src/",
        "research/gpu_sim_spike/",
        "research/sandbox_a_resident/",
        "research/colab_cli/",
        PACKAGE + "/",
    )
    return [
        name
        for name in names
        if name in {"pyproject.toml", "research/__init__.py"}
        or (
            name.startswith(trees)
            and Path(name).suffix in {".py", ".sh", ".cu", ".cpp", ".h", ".txt"}
            and not Path(name).name.startswith("test_")
        )
    ]


def binding(path: Path, dest: str | None = None) -> dict[str, str]:
    """Bind an absolute local input or a hardware canary receipt."""
    result = {"path": str(path.resolve(strict=True)), "sha256": digest(path)}
    if dest is not None:
        result["dest"] = dest
    return result


def build_manifest(
    *,
    commit: str,
    source_paths: list[str],
    inputs: list[dict[str, str]],
    output: Path,
    job_id: str,
    mode: str,
    registration_sha256: str,
    hardware_canaries: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Freeze separate 2x session envelopes without changing scientific N."""
    if mode not in {"ruler", "pilot"}:
        raise ValueError("mode must be ruler or pilot")
    if not output.is_absolute() or output.exists():
        raise ValueError("fresh absolute output directory required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,43}", job_id):
        raise ValueError("job-id must leave room for the hardware fallback identity")
    estimate, timeout = (15, 30) if mode == "ruler" else (360, 720)
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": commit,
        "script": SCRIPT,
        "device": "cuda",
        "source_paths": source_paths,
        "requirements": {"min_vcpu": 2, "est_minutes": estimate},
        "timeout_minutes": timeout,
        "output_dir": str(output),
        "max_export_bytes": 2 * 1024**3,
        "inputs": inputs,
        "required_exports": ["spatial/summary.json"],
        "env": {
            "SPATIAL_MODE": mode,
            "SPATIAL_SOURCE_COMMIT": commit,
            "SPATIAL_CONFIG": "inputs/config.json",
            "SPATIAL_LEDGER": "inputs/verified-ledger.json",
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
            "SNAKE_DQN_DEVICE": "cuda",
        },
        "budget": {
            "reserve_units": timeout * 12 / 60 / 6,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "spatial-pilot-immutable-unit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "registration_sha256": registration_sha256,
        "technical_envelope": {
            "purpose": "whole-session runaway guard; units are informational",
            "useful_compute_estimate_minutes": estimate,
            "ruler_workload_seconds": 900,
            "series_max_original_plus_completions": 3,
        },
        "hardware_envelopes": {
            gpu: {
                "base_timeout_minutes": timeout / 2,
                "effective_timeout_minutes": timeout,
                "base_unit_cap": timeout / 2 * rate / 60,
                "effective_unit_cap": timeout * rate / 60,
                "base_reserve_units": timeout / 2 * rate / 60 / 6,
                "effective_reserve_units": timeout * rate / 60 / 6,
                "rate_units_per_hour": rate,
            }
            for gpu, rate in (("H100", 12), ("A100", 8))
        },
    }
    prepare_preference(manifest)
    if mode == "ruler":
        manifest["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
        manifest["canary"] = {"schema": SCHEMA, "mode": "probe", "family": family(manifest)}
    elif hardware_canaries is not None:
        if set(hardware_canaries) != {"H100", "A100"}:
            raise ValueError("both H100 and A100 same-pin family receipts required")
        manifest["requirements"]["hardware_canaries"] = hardware_canaries
    return manifest


def prepare(
    packet: Path,
    commit: str,
    checkpoint_dir: Path,
    artifacts_root: Path,
    *,
    config_path: Path | None = None,
    seed_audit: Path | None = None,
    hardware_canaries: dict[str, Path] | None = None,
    job_id: str = "t34-spatial-pilot-01",
) -> dict[str, Any]:
    """Run clean import admission, then create a source/input-bound packet."""
    from research.spatial_student_pilot.config import PilotConfig, audit_seeds
    from research.spatial_student_pilot.resume import empty_ledger
    from tools.ops.import_preflight import check

    if not packet.is_absolute() or not artifacts_root.is_absolute():
        raise ValueError("packet and artifacts-root must be absolute")
    if packet.exists():
        raise FileExistsError(packet)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    config = (
        PilotConfig.from_dict(json.loads(config_path.read_text()))
        if config_path is not None
        else PilotConfig()
    )
    source_paths = runtime_source_paths(commit)
    for required in (SCRIPT, REQUIREMENTS, f"{PACKAGE}/run.py"):
        if required not in source_paths:
            raise ValueError(f"runtime source is absent from the committed pin: {required}")
    registration = ROOT / PACKAGE / "REGISTRATION.md"
    registration_bytes = subprocess.check_output(
        ["git", "show", f"{commit}:{PACKAGE}/REGISTRATION.md"], cwd=ROOT
    )
    if registration.read_bytes() != registration_bytes:
        raise ValueError("registration differs from the committed pin")
    checkpoints = {dest: checkpoint_dir / name for dest, name in CHECKPOINT_NAMES.items()}
    if digest(checkpoints["b8-s3.pth"]) != B8_SHA:
        raise ValueError("B8 s3 checkpoint hash differs")
    for dest, expected in OPPONENT_SHAS.items():
        if digest(checkpoints[dest]) != expected:
            raise ValueError(f"canonical opponent checkpoint hash differs: {dest}")
    checkpoint_inputs = [binding(path, dest) for dest, path in checkpoints.items()]
    seed_receipt = audit_seeds(ROOT, config)
    if seed_receipt.get("collision_count") != 0:
        raise ValueError("fresh world-disjoint seed audit did not pass")
    if seed_audit is not None and json.loads(seed_audit.read_text()) != seed_receipt:
        raise ValueError("supplied seed audit differs from the computed audit")
    # Import-only clean Python 3.12 environment: no collection, fit or data look.
    import_receipt = check(
        ROOT, commit, source_paths, REQUIREMENTS, [PACKAGE.replace("/", ".") + ".run"]
    )
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "config.json", config.to_dict())
    write_once(packet / "verified-ledger.json", empty_ledger(config))
    (packet / "REGISTRATION.md").write_bytes(registration_bytes)
    write_once(packet / "import-preflight.json", import_receipt)
    write_once(packet / "seed-audit.json", seed_receipt)
    inputs = checkpoint_inputs + [
        binding(packet / name, name)
        for name in ("config.json", "verified-ledger.json", "REGISTRATION.md", "seed-audit.json")
    ]
    proofs = (
        {gpu: binding(path) for gpu, path in hardware_canaries.items()}
        if hardware_canaries is not None
        else None
    )
    manifests = {}
    for mode in ("ruler", "pilot"):
        manifest = build_manifest(
            commit=commit,
            source_paths=source_paths,
            inputs=inputs,
            output=artifacts_root / job_id / (mode + "-output"),
            job_id=job_id + ("-ruler" if mode == "ruler" else ""),
            mode=mode,
            registration_sha256=hashlib.sha256(registration_bytes).hexdigest(),
            hardware_canaries=proofs if mode == "pilot" else None,
        )
        write_once(packet / f"{mode}.json", manifest)
        manifests[mode] = manifest
        # Persist wrapper-derived variants for review; dispatch still derives
        # and seals its own original/sibling before any hardware operation.
        for variant in variants(manifest):
            # Hardware-only ruler previews can be dispatched independently to
            # obtain a missing platform receipt. An A100 sibling cannot enter
            # the H100-only preference parser again.
            variant["requirements"].pop("prefer", None)
            write_once(packet / f"{mode}-{variant['requirements']['gpu'].lower()}.json", variant)
    commands = []
    for mode in ("ruler", "pilot"):
        argv = [
            str(ROOT / "tools/ops/dispatch.sh"),
            job_id + "-" + mode,
            str(packet / f"{mode}.json"),
            "--broker",
            str(ROOT),
            "--cwd",
            str(ROOT),
            "--state-dir",
            str(artifacts_root / job_id / (mode + "-state")),
            "--dispatch-txt",
            str(packet / "DISPATCH.txt"),
        ]
        commands.extend((shlex.join([argv[0], "--dry-run", *argv[1:]]), shlex.join(argv)))
        if mode == "ruler":
            a100_argv = [
                argv[0],
                job_id + "-ruler-a100",
                str(packet / "ruler-a100.json"),
                "--broker",
                str(ROOT),
                "--cwd",
                str(ROOT),
                "--state-dir",
                str(artifacts_root / job_id / "ruler-a100-state"),
                "--dispatch-txt",
                str(packet / "DISPATCH.txt"),
            ]
            commands.extend(
                (shlex.join([a100_argv[0], "--dry-run", *a100_argv[1:]]), shlex.join(a100_argv))
            )
    (packet / "DISPATCH.txt").write_text(
        "Prepared only. Operator submits ruler first. Pilot requires passing same-pin "
        "H100 and A100 family receipts, verified exports and stop verification.\n"
        + "\n".join(commands)
        + "\n"
    )
    receipt = {
        "schema": "spatial-pilot-preparation/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "commit": commit,
        "registration_sha256": digest(packet / "REGISTRATION.md"),
        "import_preflight": binding(packet / "import-preflight.json"),
        "manifests": {mode: binding(packet / f"{mode}.json") for mode in manifests},
        "hardware_canaries_bound": proofs is not None,
        "main_prerequisites": "same-pin H100 and A100 passing family canaries",
        "reserve_note": (
            "Wrapper preserves H100 reserve for fallback; conservative and informational"
        ),
        "allocations": 0,
    }
    write_once(packet / "PREPARATION.json", receipt)
    for command in commands:
        print(command)
    return receipt


def main() -> None:
    """Prepare once under an explicit committed source pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--h100-canary", type=Path)
    parser.add_argument("--a100-canary", type=Path)
    parser.add_argument("--job-id", default="t34-spatial-pilot-01")
    args = parser.parse_args()
    if bool(args.h100_canary) != bool(args.a100_canary):
        parser.error("bind both hardware canaries or neither")
    prepare(
        args.packet,
        args.commit,
        args.checkpoint_dir,
        args.artifacts_root,
        config_path=args.config,
        seed_audit=args.seed_audit,
        hardware_canaries=(
            {"H100": args.h100_canary, "A100": args.a100_canary} if args.h100_canary else None
        ),
        job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
