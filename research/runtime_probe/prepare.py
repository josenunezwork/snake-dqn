"""Manifest builders and input staging for CPU, T4, L4, and A100 runtime probes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import urlopen

from research.runtime_probe.bench import (
    CONTRACT_SCHEMA,
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_SEED,
    LEDGER_SCHEMA,
    ROOT,
    UNIT_ID,
)
from research.t34_exit_b8 import public_inputs
from research.watch6_l1_confirm import protocol as l1_protocol
from research.watch6_l1_confirm.uploads import _materialize_local_or_remote_checkpoints
from research.watch_replay.product import stage_release_evidence
from research.watch_replay.uploads import (
    CHUNK_BYTES,
    INDEX,
    MAX_METADATA_BYTES,
    METADATA,
    _assert_safe_new_target,
    _verify_checkpoint_pins,
)
from tools.ops.package_spec import digest, publish_immutable, publish_json, verify_immutable

SCRIPT_PATH = "research/runtime_probe/job.sh"
CANARY_PACKAGE = "research/runtime_probe"
DEFAULT_NAMESPACE = "runtime-probe-v1"
MAX_EXPORT_BYTES = 10 * 1024 * 1024

TARGETS: tuple[str, ...] = ("CPU", "T4", "L4", "A100")

RUNTIME_SPECS: dict[str, dict[str, Any]] = {
    "cpu": {
        "target": "CPU",
        "gpu": "cpu",
        "allowed_gpus": ["cpu"],
        "min_vcpu": 1,
        "est_minutes": 7.5,
        "timeout_minutes": 15.0,
        "rate_units_per_hour": 0.08,
        "unit_cap": 0.5,
        "reserve_units": 0.1,
    },
    "T4": {
        "target": "T4",
        "gpu": "T4",
        "allowed_gpus": ["T4"],
        "min_vcpu": 1,
        "est_minutes": 7.5,
        "timeout_minutes": 15.0,
        "rate_units_per_hour": 1.54,
        "unit_cap": 0.5,
        "reserve_units": 0.1,
    },
    "L4": {
        "target": "L4",
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 7.5,
        "timeout_minutes": 15.0,
        "rate_units_per_hour": 1.94,
        "unit_cap": 0.5,
        "reserve_units": 0.1,
    },
    "A100": {
        "target": "A100",
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 7.5,
        "timeout_minutes": 15.0,
        "rate_units_per_hour": 6.73,
        "unit_cap": 0.5,
        "reserve_units": 0.1,
    },
}


def source_paths() -> list[str]:
    """Sparse-checkout closure required by colab_cli/source.py on Colab runtimes."""
    return sorted({*l1_protocol.source_paths(), "research/runtime_probe"})


SMALL_PAYLOADS: tuple[str, ...] = (
    "probe-contract.json",
    "verified-ledger.json",
    "release-evidence.tar.gz",
    "public-inputs.json",
)


def normalize_gpu_target(gpu: str) -> str:
    """Normalize target name in {CPU, T4, L4, A100} to supervisor requirement key."""
    if not isinstance(gpu, str):
        raise ValueError("gpu target must be a string")
    token = gpu.strip().upper()
    if token == "CPU":
        return "cpu"
    if token in ("T4", "L4", "A100"):
        return token
    raise ValueError(f"unsupported gpu target {gpu!r}; expected one of {TARGETS}")


def metadata_names() -> set[str]:
    """Exact small immutable payloads and .sha256/.complete sidecars bundled into METADATA."""
    return {f"{name}{suffix}" for name in SMALL_PAYLOADS for suffix in ("", ".sha256", ".complete")}


def empty_ledger(namespace: str, allocation: str) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": str(namespace),
        "allocation": str(allocation),
        "verified_units": [],
        "unit_records": {},
    }


def stage_release_evidence_archive(dest: Path) -> dict[str, Any]:
    """Create a deterministic gzipped tar of the 6 strict release receipt files."""
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text("utf-8"))
    members: list[tuple[str, bytes]] = []
    for name, pin in sorted(pins["checkpoints"].items()):
        receipt = pin["strict_receipt"]
        src_root = Path(receipt["root"])
        for key, ref in sorted(receipt.items()):
            if not isinstance(ref, dict):
                continue
            src = src_root / ref["path"]
            data = src.read_bytes()
            if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                raise ValueError(f"strict release pin mismatch for {name}:{key}")
            members.append((f"{name}/{ref['path']}", data))

    buf = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for rel, data in members:
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return publish_immutable(dest, buf.getvalue())


def _stage_small_item(path: Path) -> list[dict[str, Any]]:
    verified = verify_immutable(path)
    rows = [
        {
            "path": str(path),
            "dest": path.name,
            "sha256": verified["sha256"],
            "size_bytes": verified["size"],
        }
    ]
    for suffix in (".sha256", ".complete"):
        sidecar = Path(f"{path}{suffix}")
        rows.append(
            {
                "path": str(sidecar),
                "dest": sidecar.name,
                "sha256": digest(sidecar),
                "size_bytes": sidecar.stat().st_size,
            }
        )
    return rows


def pack_probe_inputs(
    rows: list[dict[str, Any]],
    archive_path: Path,
    *,
    verify_pins: bool = True,
) -> list[dict[str, Any]]:
    """Bundle small immutable payloads and sidecars into one <=2 MiB METADATA archive."""
    expected_small = metadata_names()
    if {row["dest"] for row in rows} != expected_small or len(rows) != len(expected_small):
        raise ValueError("probe metadata inventory differs from SMALL_PAYLOADS schema")
    by_dest = {row["dest"]: row for row in rows}
    for name in SMALL_PAYLOADS:
        verified = verify_immutable(Path(by_dest[name]["path"]))
        if verified["size"] != int(by_dest[name]["size_bytes"]) or verified["sha256"] != str(
            by_dest[name]["sha256"]
        ):
            raise ValueError(f"staged payload metadata mismatch: {name}")
    if verify_pins:
        _verify_checkpoint_pins(Path(by_dest["public-inputs.json"]["path"]))

    inventory = {
        row["dest"]: {"sha256": row["sha256"], "size": int(row["size_bytes"])} for row in rows
    }
    payloads = {row["dest"]: Path(row["path"]).read_bytes() for row in rows}
    payloads[INDEX] = json.dumps(inventory, sort_keys=True).encode("utf-8")
    if sum(map(len, payloads.values())) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 4 MiB unpacked size bound")

    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(payloads.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = 0
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, io.BytesIO(data))

    archive_size = archive_path.stat().st_size
    if archive_size > CHUNK_BYTES:
        raise ValueError("compressed metadata archive exceeds 2 MiB single-chunk bound")
    return [
        {
            "path": str(archive_path),
            "dest": METADATA,
            "sha256": digest(archive_path),
            "size_bytes": archive_size,
        }
    ]


def stage_probe_metadata(
    root: Path,
    expected_sha256: str,
    *,
    verify_pins: bool = True,
    materialize_checkpoints: bool = True,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Verify METADATA archive, unpack create-only (.complete last), and materialize checkpoints."""
    root = root.resolve(strict=True)
    path = root / METADATA
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > CHUNK_BYTES
        or digest(path) != expected_sha256
    ):
        raise ValueError("metadata archive SHA256 or size mismatch")

    expected_small = metadata_names()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or set(names) != expected_small | {INDEX}
            or any(not member.isfile() or member.size < 0 for member in members)
            or sum(member.size for member in members) > MAX_METADATA_BYTES
        ):
            raise ValueError("unsafe or unexpected metadata archive inventory")
        payloads: dict[str, bytes] = {}
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"missing tar member stream: {member.name}")
            payloads[member.name] = extracted.read()

    inventory = json.loads(payloads.pop(INDEX).decode("utf-8"))
    if set(inventory) != expected_small:
        raise ValueError("metadata index inventory differs")

    for name, data in payloads.items():
        if inventory[name] != {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}:
            raise ValueError(f"metadata member digest/size mismatch: {name}")
        _assert_safe_new_target(root, name)

    with tempfile.TemporaryDirectory(prefix=".probe-metadata-", dir=root) as temporary:
        ordered = sorted(payloads, key=lambda item: (item.endswith(".complete"), item))
        for index, name in enumerate(ordered):
            staged = Path(temporary) / str(index)
            staged.write_bytes(payloads[name])
            staged.chmod(0o444)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, target)

    for payload_name in SMALL_PAYLOADS:
        verify_immutable(root / payload_name)
    if verify_pins:
        _verify_checkpoint_pins(root / "public-inputs.json")
    if materialize_checkpoints:
        _materialize_local_or_remote_checkpoints(
            root,
            verify_pins=verify_pins,
            opener=opener,
        )


def build_manifest(
    gpu: str,
    *,
    commit: str,
    metadata_input: Mapping[str, Any],
    output_dir: Path,
    job_id: str | None = None,
    attempt: int = 1,
    other_assignments: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build a research/colab_cli manifest for gpu in {CPU, T4, L4, A100}."""
    hw = normalize_gpu_target(gpu)
    spec = RUNTIME_SPECS[hw]
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    resolved_job_id = job_id or f"runtime-probe-{hw.lower()}-a{attempt}"
    return {
        "job_id": resolved_job_id,
        "commit": str(commit),
        "script": SCRIPT_PATH,
        "source_paths": source_paths(),
        "args": [],
        "device": "cpu",
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "TORCH_NUM_THREADS": "1",
            "SNAKE_DQN_DEVICE": "cpu",
            "SNAKE_SERVE_PATHB": "off",
            "SNAKE_MECHANICS_V2": "1",
            "SNAKE_WATCH_SOURCE": "live",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "RUNTIME_PROBE_TARGET_GPU": hw,
            "RUNTIME_PROBE_METADATA_SHA256": str(metadata_input["sha256"]),
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": CANARY_PACKAGE,
        },
        "inputs": [dict(metadata_input)],
        "output_dir": str(output_dir.resolve()),
        "required_exports": ["probe.json"],
        "max_export_bytes": MAX_EXPORT_BYTES,
        "timeout_minutes": float(spec["timeout_minutes"]),
        "requirements": {
            "gpu": hw,
            "allowed_gpus": list(spec["allowed_gpus"]),
            "min_vcpu": int(spec["min_vcpu"]),
            "est_minutes": float(spec["est_minutes"]),
        },
        "budget": {
            "rate_units_per_hour": float(spec["rate_units_per_hour"]),
            "unit_cap": float(spec["unit_cap"]),
            "reserve_units": float(spec["reserve_units"]),
            "other_assignments": dict(other_assignments or {}),
        },
        "resume": {
            "unit_kind": "runtime-probe",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": SCRIPT_PATH,
        },
    }


def prepare_probe_package(
    pkg_dir: Path,
    *,
    commit: str,
    gpus: tuple[str, ...] = TARGETS,
    attempt: int = 1,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
    interval: int = DEFAULT_INTERVAL,
    namespace: str = DEFAULT_NAMESPACE,
    other_assignments: Mapping[str, float] | None = None,
    release_evidence_bytes: bytes | None = None,
    verify_pins: bool = True,
) -> dict[str, Path]:
    """Stage immutable probe inputs and write manifests for each requested GPU target."""
    if pkg_dir.exists():
        raise ValueError(f"package directory must be fresh: {pkg_dir}")
    pkg_dir.mkdir(parents=True, exist_ok=False)
    manifests_dir = pkg_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=False)

    written: dict[str, Path] = {}
    for target in gpus:
        hw = normalize_gpu_target(target)
        target_slug = hw.lower()
        target_dir = pkg_dir / target_slug
        inputs_dir = target_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=False)

        allocation = f"runtime-probe-{target_slug}"
        contract = {
            "schema": CONTRACT_SCHEMA,
            "namespace": namespace,
            "allocation": allocation,
            "target_gpu": hw,
            "unit_id": UNIT_ID,
            "seed": int(seed),
            "horizon": int(horizon),
            "interval": int(interval),
        }
        publish_json(inputs_dir / "probe-contract.json", contract)
        publish_json(inputs_dir / "verified-ledger.json", empty_ledger(namespace, allocation))
        if release_evidence_bytes is not None:
            publish_immutable(inputs_dir / "release-evidence.tar.gz", release_evidence_bytes)
        else:
            stage_release_evidence_archive(inputs_dir / "release-evidence.tar.gz")
        publish_immutable(
            inputs_dir / "public-inputs.json",
            (ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_bytes(),
        )

        small_rows: list[dict[str, Any]] = []
        for payload_name in SMALL_PAYLOADS:
            small_rows.extend(_stage_small_item(inputs_dir / payload_name))
        packed = pack_probe_inputs(
            small_rows,
            target_dir / METADATA,
            verify_pins=verify_pins,
        )[0]

        manifest = build_manifest(
            target,
            commit=commit,
            metadata_input=packed,
            output_dir=pkg_dir / "outputs" / f"{target_slug}-attempt-{attempt}",
            attempt=attempt,
            other_assignments=other_assignments,
        )
        manifest_path = manifests_dir / f"probe-{target_slug}-attempt-{attempt}.json"
        publish_json(manifest_path, manifest)
        written[hw] = manifest_path
    return written


def check_stack_support() -> dict[str, dict[str, Any]]:
    """Report exact support status and gaps in research/colab_cli and tools/ops for each target."""
    return {
        "cpu": {
            "requirements_gpu_value": "cpu",
            "supervisor_validate_supported": True,
            "supervisor_alloc_cmd": ["colab", "--auth", "oauth2", "new", "-s", "<session>"],
            "supervisor_admission_supported": True,
            "dispatch_live_session_supported": True,
            "dispatch_registered_rate_units_per_hour": 0.08,
            "unsupported_pieces": [
                "research/colab_cli/supervisor.py:181 requires lowercase 'cpu' in "
                "requirements.gpu and requirements.allowed_gpus (uppercase 'CPU' is refused).",
                "tools/ops/dispatch.py:2254 (allocation_unavailable) only matches "
                "'variant=GPU&accelerator=<acc>' URLs, so 503 capacity errors on CPU-only "
                "sessions (no --gpu flag) are not matched by allocation_unavailable / "
                "reconcile_gpu_pending.py.",
                "tools/ops/ready_queue.py defines ACCOUNT_ASSIGNMENT_CAP=6 and "
                "ACCOUNT_L4_CAP=3, with no separate CPU sub-cap.",
            ],
        },
        "T4": {
            "requirements_gpu_value": "T4",
            "supervisor_validate_supported": True,
            "supervisor_alloc_cmd": [
                "colab",
                "--auth",
                "oauth2",
                "new",
                "-s",
                "<session>",
                "--gpu",
                "T4",
            ],
            "supervisor_admission_supported": True,
            "dispatch_live_session_supported": False,
            "dispatch_registered_rate_units_per_hour": None,
            "unsupported_pieces": [
                "tools/ops/dispatch.py:1349 (refresh_assignments) checks "
                "`if hardware.upper() not in ('L4', 'CPU', 'A100', 'H100'): raise "
                "Refused(f'unregistered live hardware: {hardware}')`. A single T4 job can be "
                "dispatched when no T4 session is already live, but while a T4 session is live "
                "in `colab sessions`, any subsequent dispatch.sh call fails pre-allocation with "
                "`Refused('unregistered live hardware: T4')`.",
                "tools/ops/dispatch.py:1341 has no registered fallback rate for T4 in "
                "`{'L4': 1.54, 'CPU': 0.08}`.",
                "tools/ops/ready_queue.py has no T4 sub-cap constant.",
            ],
        },
        "L4": {
            "requirements_gpu_value": "L4",
            "supervisor_validate_supported": True,
            "supervisor_alloc_cmd": [
                "colab",
                "--auth",
                "oauth2",
                "new",
                "-s",
                "<session>",
                "--gpu",
                "L4",
            ],
            "supervisor_admission_supported": True,
            "dispatch_live_session_supported": True,
            "dispatch_registered_rate_units_per_hour": 1.54,
            "unsupported_pieces": [],
        },
        "A100": {
            "requirements_gpu_value": "A100",
            "supervisor_validate_supported": True,
            "supervisor_alloc_cmd": [
                "colab",
                "--auth",
                "oauth2",
                "new",
                "-s",
                "<session>",
                "--gpu",
                "A100",
            ],
            "supervisor_admission_supported": True,
            "dispatch_live_session_supported": True,
            "dispatch_registered_rate_units_per_hour": None,
            "unsupported_pieces": [],
        },
    }


def stage_vm_inputs() -> None:
    """VM entrypoint helper: unpack input-metadata.tar.gz, release evidence, and checkpoints."""
    input_root = Path(os.environ["COLAB_INPUT_DIR"])
    expected_sha = os.environ["RUNTIME_PROBE_METADATA_SHA256"]
    try:
        stage_probe_metadata(
            input_root,
            expected_sha,
            verify_pins=True,
            materialize_checkpoints=True,
        )
        stage_release_evidence(input_root)
    except public_inputs.PublicInputError as exc:
        print(
            f"RUNTIME_PROBE_PUBLIC_INPUT_FETCH_FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(74)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        timeout=10,
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-vm-inputs",
        action="store_true",
        help="Unpack input-metadata.tar.gz and materialize checkpoints inside COLAB_INPUT_DIR",
    )
    parser.add_argument(
        "--pkg-dir",
        type=Path,
        help="Fresh package directory on the Mac for staged inputs and manifests",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        dest="gpus",
        help="Target runtime(s) in {CPU, T4, L4, A100}; defaults to all four",
    )
    parser.add_argument("--commit", default=None, help="Pinned 40-char git commit SHA")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    args = parser.parse_args(argv)

    if args.stage_vm_inputs:
        stage_vm_inputs()
        return 0

    if args.pkg_dir is None:
        raise ValueError("--pkg-dir is required unless --stage-vm-inputs is passed")

    commit = args.commit or _git_head()
    gpus = tuple(args.gpus) if args.gpus else TARGETS
    written = prepare_probe_package(
        args.pkg_dir.resolve(),
        commit=commit,
        gpus=gpus,
        attempt=args.attempt,
        seed=args.seed,
        horizon=args.horizon,
        interval=args.interval,
        namespace=args.namespace,
    )
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "commit": commit,
                "manifests": {k: str(v) for k, v in written.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
