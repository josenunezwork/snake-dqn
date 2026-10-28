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
from research.sandbox_a_resident.training_resume import HOST_ENOSPC_RECOVERY_SCHEMA
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
    recovery_file = packet / "recovery-evidence.json"
    if recovery_file.exists() or recovery_file.is_symlink():
        verify_committed_manifest(recovery_file)
        recovery_payload = json.loads(recovery_file.read_text())
        if recovery_payload.get("schema") != HOST_ENOSPC_RECOVERY_SCHEMA:
            raise ValueError("recovery-evidence.json has unsupported schema")
        seal_file = packet / "SEAL.json"
        if not seal_file.is_file():
            raise ValueError("recovery-evidence.json requires committed SEAL.json binding")
        verify_committed_manifest(seal_file)
        seal_payload = json.loads(seal_file.read_text())
        if (
            seal_payload.get("prior_stop_evidence_kind") != "host_enospc_recovery_certificate"
            or seal_payload.get("prior_recovery_evidence_schema") != HOST_ENOSPC_RECOVERY_SCHEMA
            or seal_payload.get("prior_receipt_present") is not False
            or seal_payload.get("prior_receipt_sha256") is not None
            or seal_payload.get("prior_recovery_evidence_sha256") != digest(recovery_file)
        ):
            raise ValueError("SEAL.json does not bind committed recovery-evidence.json")
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
                else ["research/sandbox_a_resident", "tools/sandbox/tests/test_source_delivery.py"]
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
