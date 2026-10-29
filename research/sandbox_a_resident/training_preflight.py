"""Offline TRAIN-01 schema, source, registry and executable-test admission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.training_prepare import STUDY, seed_allocations
from research.sandbox_a_resident.training_resume import (
    HOST_ENOSPC_RECOVERY_SCHEMA,
    verify_host_enospc_recovery,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.training_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    load_ledger,
    validate_compact_resume,
)
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[2]


def _regular_packet_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked continuation proof: {path}")


def _validate_continuation_stop(packet: Path, broker: dict) -> None:
    """Reopen the committed seal and its underlying stop proof on every admission."""
    seal_file = packet / "SEAL.json"
    recovery_file = packet / "recovery-evidence.json"
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "SEAL.json"],
            cwd=packet,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not (
        tracked
        or "completion_attempt" in broker
        or seal_file.exists()
        or seal_file.is_symlink()
        or recovery_file.exists()
        or recovery_file.is_symlink()
    ):
        return
    _regular_packet_file(seal_file)
    verify_committed_manifest(seal_file)
    seal = json.loads(seal_file.read_text())
    if (
        "completion_attempt" not in broker
        and seal.get("schema") == "sandbox-a-train01-resident-prep-seal/v1"
        and not (recovery_file.exists() or recovery_file.is_symlink())
    ):
        return
    if seal.get("schema") != "sandbox-a-train01-continuation-seal/v1":
        raise ValueError("unsupported continuation SEAL.json schema")
    kind = seal.get("prior_stop_evidence_kind")
    # 'supervisor_receipt' is the existing builder's canonical spelling.
    if kind not in {"receipt", "supervisor_receipt", "host_enospc_recovery_certificate"}:
        raise ValueError("SEAL.json has unsupported prior_stop_evidence_kind")
    prior_name = seal.get("prior_packet")
    if (
        not isinstance(prior_name, str)
        or not prior_name
        or prior_name in {".", ".."}
        or Path(prior_name).name != prior_name
    ):
        raise ValueError("SEAL.json prior_packet must be a packet name")
    packet_dir = packet.resolve()
    checkout = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=packet_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    checkout_root = (
        Path(checkout.stdout.strip()).resolve() if checkout.returncode == 0 else packet_dir.parent
    )
    raw_source = seal.get("prior_packet_path", str(packet_dir.parent / prior_name))
    if not isinstance(raw_source, str) or not raw_source or not Path(raw_source).is_absolute():
        raise ValueError("SEAL.json prior_packet_path must be an absolute path string")
    source_packet = Path(raw_source)
    try:
        source_resolved = source_packet.resolve()
        source_resolved.relative_to(checkout_root)
        if source_packet.name != prior_name or ".." in source_packet.parts:
            raise ValueError("prior packet path differs")
        cursor = source_packet
        while cursor.resolve() != checkout_root:
            if cursor.is_symlink() or cursor == cursor.parent:
                raise ValueError("symlink below checkout root")
            cursor = cursor.parent
        if tracked and source_resolved != packet_dir.parent / prior_name:
            raise ValueError("tracked prior packet differs")
        source_packet = source_resolved
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("SEAL.json prior_packet_path differs from prior_packet") from error
    source_packet_manifest = source_packet / "dispatch-manifest.json"
    for path in (source_packet_manifest, source_packet / "control.json"):
        _regular_packet_file(path)
        verify_committed_manifest(path)
    source_frozen = json.loads(source_packet_manifest.read_text())
    if kind == "host_enospc_recovery_certificate":
        if not recovery_file.is_file() or recovery_file.is_symlink():
            raise ValueError(
                "SEAL.json requires recovery-evidence.json; admission proof is missing"
            )
        verify_committed_manifest(recovery_file)
        recovery = json.loads(recovery_file.read_text())
        if (
            recovery.get("schema") != HOST_ENOSPC_RECOVERY_SCHEMA
            or seal.get("prior_recovery_evidence_schema") != HOST_ENOSPC_RECOVERY_SCHEMA
            or seal.get("prior_receipt_present") is not False
            or seal.get("prior_receipt_sha256") is not None
            or seal.get("prior_recovery_evidence_sha256") != digest(recovery_file)
        ):
            raise ValueError("SEAL.json does not bind committed recovery-evidence.json")
        identities = recovery["identities"]
        if identities["source_packet_id"] != prior_name:
            raise ValueError("recovery evidence source packet differs from SEAL.json")
        source_output = Path(identities["output_dir"])
    else:
        if recovery_file.exists() or recovery_file.is_symlink():
            raise ValueError("receipt stop proof conflicts with recovery-evidence.json")
        source_output = Path(seal.get("prior_output_dir", source_frozen["output_dir"]))
    if not source_output.is_absolute() or any(
        p.is_symlink() for p in (source_output, *source_output.parents)
    ):
        raise ValueError("prior output root must be an absolute path without symlinks")
    source_manifest_path = source_output / "manifest.json"
    _regular_packet_file(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    if (
        seal.get("prior_executed_manifest_sha256") != digest(source_manifest_path)
        or seal.get("prior_job_id") != source_manifest.get("job_id")
        or broker.get("completion_attempt", {}).get("prior_job_id") != seal.get("prior_job_id")
    ):
        raise ValueError("prior executed manifest identity differs from SEAL.json")
    if kind == "host_enospc_recovery_certificate":
        verify_host_enospc_recovery(
            recovery_file,
            source_manifest,
            digest(source_manifest_path),
            output_root=source_output,
            source_control=json.loads((source_packet / "control.json").read_text()),
            source_packet_id=source_packet.name,
            source_packet_manifest_sha256=digest(source_packet_manifest),
        )
    else:
        receipt_file = source_output / "receipt.json"
        _regular_packet_file(receipt_file)
        if seal.get("prior_receipt_present") is not True or seal.get(
            "prior_receipt_sha256"
        ) != digest(receipt_file):
            raise ValueError("SEAL.json does not bind prior receipt")
        log = seal.get("prior_supervisor_log")
        log_path = None
        if log is not None:
            log_path = Path(log["path"])
            _regular_packet_file(log_path)
            if log.get("size") != log_path.stat().st_size or log.get("sha256") != digest(log_path):
                raise ValueError("SEAL.json prior supervisor log binding differs")
        verify_mechanical_stop(
            json.loads(receipt_file.read_text()),
            source_manifest,
            digest(source_manifest_path),
            output_root=source_output,
            supervisor_log_path=log_path,
        )


def _verify_mac_compact_sidecars(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    complete = Path(f"{path}.complete")
    if not (path.is_file() and sidecar.is_file() and complete.is_file()):
        raise ValueError(f"incomplete Mac compact export or sidecars: {path.name}")
    if any(item.is_symlink() for item in (path, sidecar, complete)):
        raise ValueError(f"symlinked Mac compact export or sidecars: {path.name}")
    if sidecar.read_text().strip().split()[0] != digest(path):
        raise ValueError(f"corrupt Mac compact export SHA-256 sidecar: {path.name}")


def validate_uploaded_ledger(
    packet: Path,
    broker: dict,
    control: dict,
    *,
    allowlist_path: Path | None = None,
) -> None:
    """Verify relative runtime ledger paths against the actual Mac upload set.

    With an allowlisted ``compact_resume`` binding, the full-state ``.pt`` is not
    uploaded: its derivative and manifest are, as exact hash-bound siblings of
    the Mac original (plus the original's completion sidecars when a raw ``.pt``
    exists on Mac, or decoded and verified on Mac from the compact-at-source
    ``.pt.xz`` + ``.pt.xz.manifest`` export when the runtime exported compactly).
    """
    from research.sandbox_a_resident.training_compact import (
        CompactResumeRefused,
        decode_stream,
        load_manifest,
    )

    resolved_allowlist = allowlist_path
    if resolved_allowlist is None and (packet.parent / "compact_allowlist.json").is_file():
        resolved_allowlist = packet.parent / "compact_allowlist.json"
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    compact = validate_compact_resume(ledger, allowlist_path=resolved_allowlist)
    uploads = {row["dest"]: row for row in broker["inputs"]}
    if len(uploads) != len(broker["inputs"]):
        raise ValueError("duplicate runtime input destination")
    referenced = set()
    pending_compact_decodes: list[tuple[dict[str, Any], Path, str]] = []
    for unit in ledger.get("units", []):
        expected = [
            f"resume/iteration-{unit['iteration']:04d}{suffix}"
            for suffix in (".pt", ".json", "-actor.pth")
        ]
        if [row["path"] for row in unit["exports"]] != expected:
            raise ValueError("complete canonical snapshot triple required")
        for row in unit["exports"]:
            destination = row["path"]
            if compact is not None and destination == compact["original"]["path"]:
                if destination in uploads:
                    raise ValueError("compact resume must not also upload the full-state payload")
                sources = []
                for role, suffix in (
                    ("derivative", COMPACT_DERIVATIVE_SUFFIX),
                    ("manifest", COMPACT_MANIFEST_SUFFIX),
                ):
                    item = uploads.get(destination + suffix)
                    bound = compact[role]
                    if (
                        item is None
                        or bound["path"] != destination + suffix
                        or item["sha256"] != bound["sha256"]
                        or not item["path"].endswith(suffix)
                        or digest(Path(item["path"])) != bound["sha256"]
                        or Path(item["path"]).stat().st_size != bound["size"]
                    ):
                        raise ValueError(f"compact resume {role} is not a hash-bound upload")
                    sources.append(Path(item["path"][: -len(suffix)]))
                    referenced.add(destination + suffix)
                if sources[0] != sources[1]:
                    raise ValueError("compact derivative and manifest must be original siblings")
                source = sources[0]
                if source.is_symlink():
                    raise ValueError("symlinked Mac full-state snapshot")
                if source.is_file():
                    for suffix in (".sha256", ".complete"):
                        item = uploads.get(destination + suffix)
                        if item is None or Path(item["path"]) != Path(str(source) + suffix):
                            raise ValueError("resume sidecars must be exact uploaded siblings")
                        referenced.add(destination + suffix)
                    row["path"] = str(source)
                else:
                    if source.exists():
                        raise ValueError("non-regular Mac full-state snapshot")
                    for suffix in (".sha256", ".complete"):
                        raw_sidecar = Path(str(source) + suffix)
                        if raw_sidecar.exists() or raw_sidecar.is_symlink():
                            raise ValueError(
                                "orphan raw full-state sidecar on compact-at-source snapshot"
                            )
                        if (destination + suffix) in uploads:
                            raise ValueError(
                                "compact-at-source resume must not upload raw .pt sidecars"
                            )
                    derivative_file = Path(str(source) + COMPACT_DERIVATIVE_SUFFIX)
                    manifest_file = Path(str(source) + COMPACT_MANIFEST_SUFFIX)
                    _verify_mac_compact_sidecars(derivative_file)
                    _verify_mac_compact_sidecars(manifest_file)
                    try:
                        load_manifest(manifest_file, compact)
                    except CompactResumeRefused as error:
                        raise ValueError(
                            f"compact-at-source manifest verification failed: {error}"
                        ) from error
                    pending_compact_decodes.append((row, derivative_file, Path(destination).name))
                continue
            upload = uploads.get(destination)
            if upload is None or row["sha256"] != upload["sha256"]:
                raise ValueError("resume payload is not a hash-bound uploaded input")
            source = Path(upload["path"])
            for suffix in ("", ".sha256", ".complete"):
                name = destination + suffix
                item = uploads.get(name)
                if item is None or Path(item["path"]) != Path(str(source) + suffix):
                    raise ValueError("resume sidecars must be exact uploaded siblings")
                referenced.add(name)
            row["path"] = str(source)
    if referenced != {name for name in uploads if name.startswith("resume/")}:
        raise ValueError("unreferenced resume upload")
    # The compact binding was validated against relative paths above; the Mac
    # copy below verifies the local original (or stream-decoded compact-at-source
    # original) by path/size/hash without loading Torch payloads.
    ledger.pop(COMPACT_RESUME_FIELD, None)
    with tempfile.TemporaryDirectory(prefix="train01-verify-ledger-") as directory:
        temp_root = Path(directory)
        for target_row, derivative_file, filename in pending_compact_decodes:
            assert compact is not None
            decoded_pt = temp_root / filename
            try:
                with decoded_pt.open("xb") as stream:
                    decode_stream(
                        derivative_file,
                        stream.write,
                        expected_size=compact["original"]["size"],
                        expected_sha256=compact["original"]["sha256"],
                    )
            except CompactResumeRefused as error:
                raise ValueError(
                    f"compact-at-source stream verification failed: {error}"
                ) from error
            Path(f"{decoded_pt}.sha256").write_text(compact["original"]["sha256"] + "\n")
            Path(f"{decoded_pt}.complete").write_text("complete\n")
            target_row["path"] = str(decoded_pt)
        path = temp_root / "ledger.json"
        path.write_text(json.dumps(ledger))
        load_ledger(path, control, allowlist_path=resolved_allowlist)


def validate_packet(packet: Path) -> dict[str, Any]:
    """Bind the actual runtime controls to the exhaustive committed seed audit."""
    from research.sandbox_a_resident.train import TrainControl

    payload = preflight(packet / "sandbox.json", run_tests=False)
    broker = json.loads((packet / "dispatch-manifest.json").read_text())
    control = json.loads((packet / "control.json").read_text())
    allocation_index = control.get("allocation_index", 0)
    train_seed_offset = control.get("train_seed_offset", 0)
    if (
        type(allocation_index) is not int
        or allocation_index < 0
        or type(train_seed_offset) is not int
        or train_seed_offset < 0
    ):
        raise ValueError("TRAIN-01 seed domain requires nonnegative integer allocation/offset")
    default = asdict(
        TrainControl(
            study=STUDY,
            source_commit=broker["commit"],
            allocation_index=allocation_index,
            train_seed_offset=train_seed_offset,
        )
    )
    if control != default or control != json.loads(broker["env"]["SANDBOX_A_CONTROL_JSON"]):
        raise ValueError("TRAIN-01 control differs from frozen production recipe")
    if payload["study"] != STUDY or broker["script"] != "research/sandbox_a_resident/train_job.sh":
        raise ValueError("wrong training study/entrypoint")
    if broker["env"]["SANDBOX_A_SOURCE_COMMIT"] != broker["commit"]:
        raise ValueError("runtime source pin differs")
    if payload["endpoint"] != {"kind": "steps", "value": 100_663_296}:
        raise ValueError("wrong finite actor endpoint")
    for name in ("control.json", "registry-check.json", "verified-ledger.json"):
        verify_committed_manifest(packet / name)
    _validate_continuation_stop(packet, broker)
    check = json.loads((packet / "registry-check.json").read_text())
    expected = seed_allocations(control)
    registry = next(r for r in payload["inputs"] if r["kind"] == "registry")
    if (
        check.get("schema") != "sandbox-a-train-seed-audit/v1"
        or check.get("status") != "PASS"
        or check.get("study") != STUDY
        or check.get("allocations") != expected
        or check.get("count") != sum(row["count"] for row in expected)
        or check.get("registry_sha256") != registry["sha256"]
        or check.get("effective_seed_bits") != 63
        or check.get("aliases") != 0
        or check.get("excluded_intersection") != 0
        or broker["sandbox_a"].get("seed_allocations") != expected
        or broker["sandbox_a"].get("seed_audit_sha256") != digest(packet / "registry-check.json")
    ):
        raise ValueError("exhaustive seed audit does not bind this allocation/registry")
    validate_uploaded_ledger(packet, broker, control)
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument(
        "--cuda-semantics",
        action="store_true",
        help="run strict CUDA API-contract mocks without a GPU",
    )
    parser.add_argument(
        "--admission",
        action="store_true",
        help="run bounded runtime safety tests; full engineering suite remains the default",
    )
    args = parser.parse_args()
    check = validate_packet(args.packet.resolve())
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        env[key] = "1"
    env["SNAKE_DQN_DEVICE"] = "cpu"
    cuda_contract_tests = [
        "research/sandbox_a_resident/test_training_cuda_semantics.py",
        "research/sandbox_a_resident/test_training_learner.py"
        "::test_critic_quality_baseline_has_cuda_like_same_device_semantics",
        "research/sandbox_a_resident/test_training_cpu_reference.py"
        "::test_cpu_reference_can_route_frozen_inference_to_cuda_semantics",
        "research/sandbox_a_resident/test_training_rosters.py"
        "::test_runtime_resume_normalizes_relocated_row_identity_to_cpu",
    ]
    admission_tests = [
        "research/sandbox_a_resident/test_training_prepare.py",
        "research/sandbox_a_resident/test_training_units.py",
        "research/sandbox_a_resident/test_training_runner.py",
        "research/sandbox_a_resident/test_training_compact.py",
        "research/sandbox_a_resident/test_training_sigusr1.py",
        "research/sandbox_a_resident/test_training_learner.py",
        "tools/sandbox/tests/test_source_delivery.py",
    ]
    python_exe = ROOT / "venv/bin/python"
    subprocess.run(
        [
            "nice",
            "-n",
            "15",
            str(python_exe if python_exe.exists() else Path(sys.executable)),
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:xdist",
            *(
                cuda_contract_tests
                if args.cuda_semantics
                else (
                    admission_tests
                    if args.admission
                    else ["research/sandbox_a_resident", "tools/sandbox/tests/test_source_delivery.py"]
                )
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    label = "PASS_TRAIN01_CUDA_SEMANTICS" if args.cuda_semantics else "PASS_TRAIN01_PREFLIGHT"
    print(f"{label} seeds={check['count']} schema=PASS registry=PASS tests=PASS allocations=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
