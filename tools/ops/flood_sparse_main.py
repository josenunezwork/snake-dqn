"""Prepare/check the frozen FLOOD sparse main; provider dispatch is explicit only.

The reviewed canary's upload bytes and runtime source closure are reused exactly.
Supervisor receipts have no sidecars; exported payloads and package controls do.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
PIN = "b7dfa1fc35321a388813122b3acfe033d1367782"
CANARY = BASE / (
    "snake-dqn-artifacts/big-lever/exp/gpu-sparse-metadata-20261024-02/packet/canary-output"
)
ARTIFACT = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-flood-sparse-main-20261024-03"
BROKER = BASE / "snake-dqn-sparse-metadata-qualified-02.noindex"
PYTHON = BASE / "snake-dqn/venv/bin/python"
JOB = "sparse-bench-a100-main-b7dfa1fc-20261024-03"
PROTOCOL = "research/gpu_sim_spike/GPU_FLOOD_V4_INTENT_20261023.md"
PROTOCOL_SHA = "29fef399660cce2aa5c1a9d5243955fb90b717a8c1ceef1b443238576160e5ab"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verified(path: Path) -> dict[str, Any]:
    """Package controls/exports use complete triples, unlike broker receipts."""
    if Path(str(path) + ".sha256").read_text().split() != [sha(path), path.name]:
        raise ValueError(f"digest/name differs: {path}")
    if Path(str(path) + ".complete").read_bytes() != b"complete\n":
        raise ValueError(f"completion marker differs: {path}")
    return load(path)


def publish(path: Path, value: Any) -> None:
    from research.gpu_sim_spike.a1_prof_units import _atomic_create

    data = (
        value
        if isinstance(value, bytes)
        else (
            value.encode()
            if isinstance(value, str)
            else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".sha256", ".complete"):
        if Path(str(path) + suffix).exists():
            raise FileExistsError(path)
    _atomic_create(path, data)
    _atomic_create(Path(str(path) + ".sha256"), f"{sha(path)}  {path.name}\n".encode())
    _atomic_create(Path(str(path) + ".complete"), b"complete\n")


def implementation() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in (
            Path(__file__),
            ROOT / "tools/ops/dispatch.sh",
            ROOT / "tools/ops/dispatch.py",
            ROOT / "tools/ops/repack_gpu_upload.py",
            ROOT / "research/gpu_sim_spike/native_compile.py",
            ROOT / "tests/test_flood_sparse_main.py",
        )
    }


def prior_spend() -> dict[str, Any]:
    """Count every retained sparse/compile attempt conservatively within Q2.

    A pre-allocation UPLOAD_CAP has no rate accounting; accept zero only when
    no endpoint/session/generation/CLI call or elapsed time exists. Other
    missing accounting or uncertain release is a refusal, never zero cost.
    """
    from tools.ops.spend import verify_receipts

    exp = BASE / "snake-dqn-artifacts/big-lever/exp"
    paths = sorted(
        set(exp.glob("gpu-engine-sprint*/sparse*/**/receipt.json"))
        | set(exp.glob("gpu-sparse-metadata*/**/receipt.json"))
        | set(exp.glob("gpu-flood-sparse-main*/main-output/receipt.json"))
    )
    if not paths:
        raise ValueError("missing prior sparse-series accounting")
    charged, zero = [], []
    for path in paths:
        value = load(path)
        if value.get("accounting") is not None:
            if value.get("stop_verified") is not True:
                raise ValueError("prior sparse attempt lacks verified release")
            charged.append(path)
        elif (
            value.get("elapsed_seconds") == 0.0
            and value.get("failure_class") == "UPLOAD_CAP"
            and value.get("cli_calls") == []
            and all(value.get(key) is None for key in ("endpoint", "session", "generation"))
        ):
            zero.append(
                {
                    "path": str(path),
                    "sha256": sha(path),
                    "units": 0,
                    "elapsed_seconds": 0,
                    "reason": "refused before allocation",
                }
            )
        else:
            raise ValueError("prior sparse attempt has unknown spend")
    rows = verify_receipts(charged)
    wall = sum(row.elapsed_seconds for row in rows) / 60
    units = sum(row.units for row in rows)
    if wall + 160 > 216 or units + 80 / 3 > 36:
        raise ValueError("remaining sparse-series envelope cannot fund this main cap")
    return {
        "receipts": [
            {
                "path": str(row.path),
                "sha256": row.sha256,
                "units": row.units,
                "elapsed_seconds": row.elapsed_seconds,
            }
            for row in rows
        ]
        + zero,
        "elapsed_minutes": wall,
        "units": units,
        "main_wall_minutes": 160,
        "main_unit_cap": 80 / 3,
        "committed_wall_minutes": wall + 160,
        "committed_units": units + 80 / 3,
        "scope": "All retained sparse canaries and L4 native compile attempts, conservatively "
        "charged to sparse's 216 minute /36 unit sub-envelope. Operator must also "
        "reconcile any external liabilities and the larger Q2/global/calendar limits.",
    }


def reviewed_canary(packet: Path, broker: Path) -> dict[str, Any]:
    """Authenticate the actual saved receipt layout and Mac review scope."""
    from research.gpu_sim_spike.native_compile import binding

    review = verified(packet / "CODEX_V_MAC_REVIEW.json")
    receipt_path, manifest_path = packet / "receipt.json", packet / "manifest.json"
    receipt, probe = load(receipt_path), load(manifest_path)
    if (
        review.get("verdict") != "PASS"
        or review.get("verdict_scope") != "CANARY_MAC_VERIFICATION_ONLY"
        or review.get("route_decision") != "INCOMPLETE"
        or review.get("measured_pin") != PIN
        or review.get("receipt", {}).get("sha256") != sha(receipt_path)
        or review.get("manifest", {}).get("sha256") != sha(manifest_path)
        or probe.get("commit") != PIN
        or receipt.get("commit") != PIN
        or receipt.get("manifest_sha256") != sha(manifest_path)
        or receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or any(receipt.get(key) for key in ("error", "cleanup_error", "failure_class"))
        or receipt.get("terminal", {}).get("returncode") != 0
        or receipt.get("job_id") != probe.get("job_id")
        or receipt.get("terminal", {}).get("job_id") != probe.get("job_id")
        or probe.get("env", {}).get("SPARSE_BENCH_CANARY") != "1"
    ):
        raise ValueError("same-pin reviewed canary/receipt/manifest identity differs")
    rows = receipt.get("exports", [])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("duplicate or absent canary exports")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("foreign canary export path")
        path = packet / "exports" / relative
        # NPZ payloads are binary; validate triples before parsing JSON separately.
        if (
            path.is_symlink()
            or sha(path) != row["sha256"]
            or path.stat().st_size != row["size"]
            or Path(str(path) + ".sha256").read_text().split() != [sha(path), path.name]
            or Path(str(path) + ".complete").read_bytes() != b"complete\n"
        ):
            raise ValueError("canary export triple differs")
    proof = probe["native_cuda_compile"]["receipt"]
    if binding(Path(proof["path"]), PIN, broker) != proof:
        raise ValueError("native compile binding differs")
    protocol = subprocess.check_output(["git", "show", f"{PIN}:{PROTOCOL}"], cwd=broker)
    if hashlib.sha256(protocol).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen FLOOD protocol differs")
    return probe


def main_manifest(packet: Path, artifact: Path, broker: Path) -> dict[str, Any]:
    """Use the frozen builder's main sizing without regenerating canary inputs."""
    from research.gpu_sim_spike import sparse_units
    from research.metadata_archive import unpack

    probe = reviewed_canary(packet, broker)
    job = copy.deepcopy(probe)
    job.update(job_id=JOB, output_dir=str(artifact / "main-output"), timeout_minutes=160)
    job["requirements"]["est_minutes"] = 80
    job["budget"].update(unit_cap=80 / 3, reserve_units=16 / 3, other_assignments={})
    env = job["env"]
    for key in ("SNAKE_COLAB_CANARY", "SNAKE_COLAB_CANARY_PACKAGE"):
        env.pop(key, None)
    env.update(
        SPARSE_BENCH_CANARY="0",
        SPARSE_BENCH_SECONDS="4800",
        GPU_SIM_SPIKE_SECONDS="4800",
        SPARSE_BENCH_LEDGER_PATH="inputs/verified-ledger.json",
    )
    job["resume"]["ledger_path"] = "inputs/verified-ledger.json"
    job["canary"] = {
        "receipt": {"path": str(packet / "receipt.json"), "sha256": sha(packet / "receipt.json")}
    }
    # Byte-identical inputs are essential: the native family canary verifies
    # exact hashes/file destinations, not merely an equivalent ledger.
    if {row["dest"] for row in job["inputs"]} != {"metadata.tar", "public-inputs.json"}:
        raise ValueError("unexpected bundled input inventory")
    for row in job["inputs"]:
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("reviewed upload bytes differ")
    metadata = next(row for row in job["inputs"] if row["dest"] == "metadata.tar")
    with tempfile.TemporaryDirectory(prefix="flood-main-ledger-") as temporary:
        root = Path(temporary).resolve()
        unpack(Path(metadata["path"]), root, metadata["sha256"])
        for canary, prefix in ((False, "inputs"), (True, "canary-inputs")):
            ledger = root / prefix / "verified-ledger.json"
            if (
                sparse_units.load_ledger(
                    ledger,
                    sparse_units.namespace(PIN, canary=canary),
                    sparse_units.allocation(canary=canary),
                )
                or load(ledger)["units"] != []
            ):
                raise ValueError("first main package needs empty verified ledgers")
    return job


def prepare(packet: Path, artifact: Path, broker: Path) -> None:
    from tools.ops.repack_gpu_upload import check_upload
    from tools.ops.spend import read_spend

    job = main_manifest(packet, artifact, broker)
    accounting = prior_spend()
    if artifact.exists():
        raise FileExistsError(artifact)
    artifact.mkdir(parents=True)
    manifest_path = artifact / "sparse-bench-main.json"
    publish(manifest_path, job)
    upload = check_upload(job, broker, artifact / "upload-audit/staging")
    publish(artifact / "UPLOAD_AUDIT.json", upload)
    command = shlex.join(
        [
            str(PYTHON),
            "-I",
            str(Path(__file__).resolve()),
            "dispatch",
            "--artifact-dir",
            str(artifact),
            "--packet",
            str(packet),
            "--broker",
            str(broker),
        ]
    )
    publish(
        artifact / "dispatch-main.sh",
        f'#!/usr/bin/env bash\nset -euo pipefail\nexec {command} "$@"\n',
    )
    publish(
        artifact / "DISPATCH.txt",
        "# FLOOD sparse main; prepared for operator admission.\n"
        + f"nice -n 15 bash {shlex.quote(str(artifact / 'dispatch-main.sh'))}\n",
    )
    spent = read_spend(packet / "receipt.json")
    publish(
        artifact / "PREPARATION.json",
        {
            "schema": "flood-sparse-main-preparation/v1",
            "status": "PREPARED_NOT_DISPATCHED",
            "runtime_pin": PIN,
            "implementation": implementation(),
            "manifest_sha256": sha(manifest_path),
            "dispatch_sha256": sha(artifact / "dispatch-main.sh"),
            "dispatch_txt_sha256": sha(artifact / "DISPATCH.txt"),
            "upload_audit_sha256": sha(artifact / "UPLOAD_AUDIT.json"),
            "canary_review_sha256": sha(packet / "CODEX_V_MAC_REVIEW.json"),
            "protocol_sha256": PROTOCOL_SHA,
            "canary_spend": {
                "elapsed_seconds": spent.elapsed_seconds,
                "units": spent.units,
                "receipt_sha256": spent.sha256,
            },
            "prior_spend": accounting,
            "series": {
                "wall_minutes": 216,
                "units": 36,
                "included_reserve_units": 6,
                "canary_reserved_minutes": 24,
                "canary_reserved_units": 4,
                "main_reserved_minutes": 160,
                "main_reserved_units": 80 / 3,
                "original_main_ceiling_minutes": 192,
                "original_main_ceiling_units": 32,
            },
            "matrix_scope": "Sparse: E1024 scripted/frozen/mixed, 3 matched repeats, 3 warmups. "
            "A1-PROF's E64/E256/E1024 nine diagnostics are a separate allocation; "
            "no lower-E sparse cells exist at this immutable pin.",
            "admission": "Operator must reconcile prior FLOOD/Q2 attempts, compile costs, sealed "
            "round-28 T0/deadlines, remaining lane/global caps and balance/liabilities. "
            "First cap wins; no new entitlement or automatic completion attempt.",
        },
    )
    print(json.dumps({"status": "PREPARED_NOT_DISPATCHED", "artifact": str(artifact)}))


def dispatch(packet: Path, artifact: Path, broker: Path, dry: bool) -> None:
    from tools.ops.repack_gpu_upload import check_upload

    prep = verified(artifact / "PREPARATION.json")
    path = artifact / "sparse-bench-main.json"
    job = verified(path)
    if (
        prep["implementation"] != implementation()
        or prep["manifest_sha256"] != sha(path)
        or prep["dispatch_sha256"] != sha(artifact / "dispatch-main.sh")
        or prep["dispatch_txt_sha256"] != sha(artifact / "DISPATCH.txt")
        or prep["upload_audit_sha256"] != sha(artifact / "UPLOAD_AUDIT.json")
        or prep["canary_review_sha256"] != sha(packet / "CODEX_V_MAC_REVIEW.json")
        or prep["prior_spend"] != prior_spend()
        or job != main_manifest(packet, artifact, broker)
    ):
        raise ValueError("frozen main preparation/source/manifest differs")
    if not dry:
        review = verified(artifact / "CODEX_V_LIGHT.json")
        if (
            review.get("status") != "ACCEPT_OFFLINE"
            or review.get("preparation_sha256") != sha(artifact / "PREPARATION.json")
            or review.get("implementation") != implementation()
        ):
            raise ValueError("main package lacks exact Codex-V light acceptance")
    with tempfile.TemporaryDirectory(prefix="flood-main-upload-") as temporary:
        upload = check_upload(job, broker, Path(temporary) / "staging")
    command = [
        "bash",
        str(ROOT / "tools/ops/dispatch.sh"),
        job["job_id"],
        str(path),
        "--broker",
        str(broker),
        "--cwd",
        str(broker),
        "--state-dir",
        str(artifact / "state-main"),
        "--dispatch-txt",
        str(artifact / "DISPATCH.txt"),
    ]
    if dry:
        command.append("--dry-run")
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    environment["PATH"] = str(PYTHON.parent) + os.pathsep + environment.get("PATH", "")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(
        json.dumps(
            {
                "status": "PASS_DRY_RUN" if dry else "DISPATCHED",
                "allocations": 0 if dry else "operator_requested",
                "largest_upload_bytes": upload["largest_upload_bytes"],
                "live_accounting": "NOT_CHECKED" if dry else "ops_admission",
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dispatch"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    parser.add_argument("--packet", type=Path, default=CANARY)
    parser.add_argument("--broker", type=Path, default=BROKER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    packet, artifact, broker = (
        args.packet.resolve(),
        args.artifact_dir.resolve(),
        args.broker.resolve(),
    )
    if args.action == "prepare":
        prepare(packet, artifact, broker)
    else:
        dispatch(packet, artifact, broker, args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
