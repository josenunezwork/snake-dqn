"""Seal the spatial pilot packet offline; never allocate or submit Colab work.

The 15-minute integrated ruler is also the genuine target-platform family
canary. Both modes upload exactly the same input bytes and run ``job.sh``.
The pilot stays unqualified until both hardware receipts are bound and pass
the unchanged supervisor canary gate. No stand-in receipt is generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from research.colab_cli.canary import SCHEMA, family
from research.compute.gpu_preference import prepare_preference, variants
from research.spatial_student_pilot.ruler_fixtures import THREAD_ENV

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/spatial_student_pilot"
SCRIPT = f"{PACKAGE}/job.sh"
REQUIREMENTS = f"{PACKAGE}/requirements.txt"
RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/spatial-latency-ruling/RULING.md"
)
RULING_SHA256 = "887fba0553098638081382fea1e0fd5c6d1fc30b9f8fc8443f73479d82636e97"
GOVERNANCE_FILES = (
    "REGISTRATION.md",
    "REGISTRATION_CLARIFICATION_01.md",
    "REGISTRATION_CLARIFICATION_02.md",
    "REGISTRATION_CLARIFICATION_03.md",
    "REGISTRATION_AMENDMENT_04.md",
    "REGISTRATION_CLARIFICATION_05.md",
    "REGISTRATION_CLARIFICATION_06.md",
)
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
    selected = [
        name
        for name in names
        if name in {"pyproject.toml", "research/__init__.py"}
        or (
            name.startswith(trees)
            and Path(name).suffix in {".py", ".sh", ".cu", ".cpp", ".h", ".txt"}
            and not Path(name).name.startswith("test_")
        )
    ]
    # C/T's parity helpers have function-local imports reaching the production
    # policy adapters. Resolve those imports from the PIN, never the ambient
    # checkout; include code only, not historical research banks or outcomes.
    from tools.ops.import_preflight import import_closure, module_path

    python_paths = [name for name in names if name.endswith(".py")]
    archive = subprocess.check_output(
        ["git", "archive", commit, "--", *python_paths], cwd=ROOT, timeout=60
    )
    with tempfile.TemporaryDirectory(prefix="spatial-source-closure-") as directory:
        frozen = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(frozen, filter="data")
        closure, _ = import_closure(frozen, ["research.spatial_student_pilot.run"])
        selected.extend(str(module_path(frozen, module).relative_to(frozen)) for module in closure)
    return sorted(set(selected))


def binding(path: Path, dest: str | None = None) -> dict[str, str]:
    """Bind an absolute local input or a hardware canary receipt."""
    result = {"path": str(path.resolve(strict=True)), "sha256": digest(path)}
    if dest is not None:
        result["dest"] = dest
    return result


def governance_bytes(commit: str, ruling: Path) -> dict[str, bytes]:
    """Refuse drift in the ruling or any source-pinned governance sibling."""
    if ruling.is_symlink() or digest(ruling) != RULING_SHA256:
        raise ValueError("binding latency ruling digest differs")
    result = {"LATENCY_RULING.md": ruling.read_bytes()}
    for name in GOVERNANCE_FILES:
        pinned = subprocess.check_output(["git", "show", f"{commit}:{PACKAGE}/{name}"], cwd=ROOT)
        if (ROOT / PACKAGE / name).read_bytes() != pinned:
            raise ValueError(f"governance differs from committed pin: {name}")
        result[name] = pinned
    if RULING_SHA256.encode() not in result["REGISTRATION_AMENDMENT_04.md"]:
        raise ValueError("amendment does not bind latency ruling")
    return result


def verify_ruler_receipt(path: Path, commit: str) -> None:
    """Require real, hash-bound integrated rate evidence before main packing."""
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "PASS" or receipt.get("stop_verified") is not True:
        raise ValueError("ruler requires a passing stopped supervisor receipt")
    if receipt.get("commit") != commit:
        raise ValueError("ruler receipt source pin differs")
    summary = path.parent / "exports/spatial/summary.json"
    matches = [r for r in receipt.get("exports", []) if r["path"] == "spatial/summary.json"]
    if (
        len(matches) != 1
        or not summary.is_file()
        or digest(summary) != matches[0]["sha256"]
        or summary.stat().st_size != matches[0]["size"]
    ):
        raise ValueError("ruler summary is not bound to supervisor exports")
    report = json.loads(summary.read_text())
    if (
        report.get("status") != "RULER_COMPLETE"
        or report.get("forecast_qualified") is not True
        or report.get("within_six_hours") is not True
        or report.get("support_projection", {}).get("qualified") is not True
        or report.get("export_projection", {}).get("qualified") is not True
        or report.get("rate_qualified") is not True
    ):
        raise ValueError("integrated ruler did not qualify mature support and six-hour forecast")
    proof = report.get("phases", {}).get("mapped_parity", {})
    if (
        proof.get("exact") is not True
        or proof.get("exact_actions") is not True
        or proof.get("reference_worlds") != 128
        or proof.get("candidate_worlds") != 1024
        or proof.get("ticks", 0) < 3
    ):
        raise ValueError("ruler lacks exact mapped E128/E1024 action/state parity")
    peak = report.get("peak_vram_bytes")
    if not isinstance(peak, int) or peak <= 0 or peak > 34 * 1024**3:
        raise ValueError("ruler lacks bounded measured CUDA peak memory")
    threads = report.get("threads", {})
    if (
        threads.get("environment") != {key: "1" for key in THREAD_ENV}
        or threads.get("torch_threads") != 1
        or threads.get("torch_interop_threads") != 1
    ):
        raise ValueError("ruler lacks verified one-thread environment")


def build_manifest(
    *,
    commit: str,
    source_paths: list[str],
    inputs: list[dict[str, str]],
    output: Path,
    job_id: str,
    mode: str,
    registration_sha256: str,
    ruling_sha256: str,
    amendment_sha256: str,
    clarification_sha256: str,
    clarification_06_sha256: str,
    hardware_canaries: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Freeze separate 2x session envelopes without changing scientific N."""
    if mode not in {"ruler", "pilot"}:
        raise ValueError("mode must be ruler or pilot")
    if mode == "pilot" and not hardware_canaries:
        raise ValueError("main manifest requires bound H100 and A100 ruler receipts")
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
            **{key: "1" for key in THREAD_ENV},
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
        "latency_ruling_sha256": ruling_sha256,
        "registration_amendment_04_sha256": amendment_sha256,
        "registration_clarification_05_sha256": clarification_sha256,
        "registration_clarification_06_sha256": clarification_06_sha256,
        "technical_envelope": {
            "purpose": "whole-session runaway guard; units are informational",
            "useful_compute_estimate_minutes": estimate,
            "ruler_workload_seconds": 900,
            "production_collection_worlds": 1024,
            "production_branch_jobs": 1024,
            "paired_gameplay_worlds": 384,
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
        manifest.pop("resume")
        manifest["resume_exemption"] = {
            "kind": "short-canary",
            "reason": "15-minute integrated development ruler; no scientific units imported",
        }
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
    ledger_path: Path | None = None,
    resume_roots: list[Path] | None = None,
    seed_audit: Path | None = None,
    hardware_canaries: dict[str, Path] | None = None,
    job_id: str = "t34-spatial-pilot-01",
    ruling: Path = RULING,
    native_cache_receipts: list[Path] | None = None,
    wheelhouse: Path | None = None,
) -> dict[str, Any]:
    """Run clean import admission, then create a source/input-bound packet."""
    from research.spatial_student_pilot.config import PilotConfig, audit_seeds
    from research.spatial_student_pilot.resume import empty_ledger, upload_inputs, validate_ledger
    from research.spatial_student_pilot.ruler_fixtures import FIXTURE_NAME, make_fixtures
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
    ledger = json.loads(ledger_path.read_text()) if ledger_path else empty_ledger(config)
    validate_ledger(ledger, config)
    resumed_inputs = upload_inputs(ledger, resume_roots or [], config)
    if hardware_canaries:
        for receipt in hardware_canaries.values():
            verify_ruler_receipt(receipt, commit)
    source_paths = runtime_source_paths(commit)
    for required in (SCRIPT, REQUIREMENTS, f"{PACKAGE}/run.py"):
        if required not in source_paths:
            raise ValueError(f"runtime source is absent from the committed pin: {required}")
    governance = governance_bytes(commit, ruling)
    registration_bytes = governance["REGISTRATION.md"]
    amendment_hash = hashlib.sha256(governance["REGISTRATION_AMENDMENT_04.md"]).hexdigest()
    clarification_hash = hashlib.sha256(governance["REGISTRATION_CLARIFICATION_05.md"]).hexdigest()
    clarification_06_hash = hashlib.sha256(
        governance["REGISTRATION_CLARIFICATION_06.md"]
    ).hexdigest()
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
    write_once(packet / "verified-ledger.json", ledger)
    # Only ruler() decodes this input; identical main uploads preserve the
    # supervisor's unchanged family/input identity contract.
    import torch

    with (packet / FIXTURE_NAME).open("xb") as stream:
        torch.save(make_fixtures(), stream)
    for name, content in governance.items():
        (packet / name).write_bytes(content)
    write_once(packet / "import-preflight.json", import_receipt)
    write_once(packet / "seed-audit.json", seed_receipt)
    inputs = (
        checkpoint_inputs
        + resumed_inputs
        + [
            binding(packet / name, name)
            for name in (
                "config.json",
                "verified-ledger.json",
                "seed-audit.json",
                FIXTURE_NAME,
            )
        ]
    )
    inputs.extend(binding(packet / name, name) for name in governance)
    cache_hashes = {}
    if native_cache_receipts:
        from research.spatial_student_pilot.native_cache import upload_inputs

        for receipt in native_cache_receipts:
            cached = upload_inputs(receipt, commit)
            report = json.loads(Path(cached[0]["path"]).read_text())
            arch = report["architecture"]
            if arch in cache_hashes:
                raise ValueError("duplicate native cache architecture")
            cache_hashes[arch] = cached[0]["sha256"]
            for row in cached:
                row["dest"] = row["dest"].replace("native-cache/", f"native-cache/{arch}/", 1)
            inputs.extend(cached)
        if set(cache_hashes) != {"8.0", "9.0"}:
            raise ValueError("bind both H100 and A100 native caches for the same input family")
    if wheelhouse is not None:
        wheels = sorted(wheelhouse.glob("*.whl"))
        if (
            not wheels
            or sum(w.stat().st_size for w in wheels) > 128 * 1024**2
            or any(
                w.is_symlink()
                or not w.is_file()
                or not any(tag in w.name for tag in ("manylinux", "linux_x86_64", "none-any"))
                for w in wheels
            )
        ):
            raise ValueError("wheelhouse must be bounded Linux/any wheels, at most128MiB")
        inputs.extend(binding(w, "wheelhouse/" + w.name) for w in wheels)
    proofs = (
        {gpu: binding(path) for gpu, path in hardware_canaries.items()}
        if hardware_canaries is not None
        else None
    )
    manifests = {}
    for mode in ("ruler", "pilot"):
        if mode == "pilot" and proofs is None:
            write_once(
                packet / "PILOT_NOT_PREPARED.json",
                {
                    "reason": "main requires genuine same-pin H100 and A100 ruler receipts",
                    "dispatchable": False,
                },
            )
            continue
        manifest = build_manifest(
            commit=commit,
            source_paths=source_paths,
            inputs=inputs,
            output=artifacts_root / job_id / (mode + "-output"),
            job_id=job_id + ("-ruler" if mode == "ruler" else ""),
            mode=mode,
            registration_sha256=hashlib.sha256(registration_bytes).hexdigest(),
            ruling_sha256=RULING_SHA256,
            amendment_sha256=amendment_hash,
            clarification_sha256=clarification_hash,
            clarification_06_sha256=clarification_06_hash,
            hardware_canaries=proofs if mode == "pilot" else None,
        )
        if cache_hashes:
            manifest["env"]["SPATIAL_NATIVE_CACHE_SHA256"] = json.dumps(
                cache_hashes, sort_keys=True
            )
        if wheelhouse is not None:
            manifest["env"]["COLAB_CLI_WHEELHOUSE"] = "inputs/wheelhouse"
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
    for mode in manifests:
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
        "latency_ruling_sha256": RULING_SHA256,
        "registration_amendment_04_sha256": amendment_hash,
        "registration_clarification_05_sha256": clarification_hash,
        "registration_clarification_06_sha256": clarification_06_hash,
        "native_cache_bound": bool(cache_hashes),
        "native_cache_provenance": [binding(p) for p in (native_cache_receipts or [])],
        "wheelhouse_bound": wheelhouse is not None,
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
    parser.add_argument("--verified-ledger", type=Path)
    parser.add_argument("--resume-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--h100-canary", type=Path)
    parser.add_argument("--a100-canary", type=Path)
    parser.add_argument("--job-id", default="t34-spatial-pilot-01")
    parser.add_argument("--native-cache-receipts", nargs="*", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--ruling", type=Path, default=RULING)
    args = parser.parse_args()
    if bool(args.h100_canary) != bool(args.a100_canary):
        parser.error("bind both hardware canaries or neither")
    prepare(
        args.packet,
        args.commit,
        args.checkpoint_dir,
        args.artifacts_root,
        config_path=args.config,
        ledger_path=args.verified_ledger,
        resume_roots=args.resume_roots,
        seed_audit=args.seed_audit,
        hardware_canaries=(
            {"H100": args.h100_canary, "A100": args.a100_canary} if args.h100_canary else None
        ),
        job_id=args.job_id,
        ruling=args.ruling,
        native_cache_receipts=args.native_cache_receipts,
        wheelhouse=args.wheelhouse,
    )


if __name__ == "__main__":
    main()
