"""Round-16 timing/identity admission. Never deserialize a trajectory's outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_served_search import b2, b2_allocate
from research.bl_served_search.exports import digest, export_json, verify_export
from research.colab_cli import canary

CEILING_SECONDS = 7.125 * 60
QUALIFIED_TREE = "553af3f692515b2bf66173356fe4d099acc96942"
QUALIFIED_RECORD = b2.ROOT.parent / "snake-dqn-artifacts/ops/colab-cli-qualified.json"
RESCALE_CLAIMS = b2.ROOT.parent / "snake-dqn-artifacts/big-lever/exp/bl-b2/rescale-claims"


def finite(value: Any, *, positive: bool = False) -> float:
    """Reject missing, boolean, nonfinite and negative timing observations."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError("NOT_READY invalid measured timing")
    return float(value)


def decide(spec: dict, operational: dict, whole_seconds: float) -> dict:
    """Pure rule; takes no eligibility, yield, H300 or alternative-outcome inputs."""
    whole = finite(whole_seconds, positive=True)
    expected_id = b2.units.unit_id(spec["namespace"] + "/excluded-canary", 0)
    if (
        operational.get("schema") != b2.SCHEMA + "/canary-admission"
        or operational.get("allocation_digest") != spec["allocation_digest"]
        or operational.get("unit_id") != expected_id
        or operational.get("seed") != spec["canary_seed"]
        or operational.get("trajectory_frames") != spec["trajectory_frames"]
        or operational.get("workers") != 6
        or type(operational.get("serial_parallel_root_checks")) is not int
        or not 1 <= operational["serial_parallel_root_checks"] <= 4
        or operational.get("serial_parallel_equal") is not True
    ):
        raise ValueError("NOT_READY B2 identity or exercised serial/parallel proof differs")
    native = operational.get("native_v8")
    threads = operational.get("worker_threads")
    if not isinstance(native, list) or not isinstance(threads, list) or not threads:
        raise ValueError("NOT_READY missing native/worker attestation")
    if len(native) != len(threads) + 1 or len(threads) < operational["serial_parallel_root_checks"]:
        raise ValueError("NOT_READY incomplete branch attestation")
    for row in native:
        b2.engine.attest(row)
        if type(row.get("eligible_opponent_rows")) is not int:
            raise ValueError("NOT_READY missing native opportunity reconciliation")
    if any(row.get("torch") != 1 or row.get("numba") != 1 for row in threads):
        raise ValueError("NOT_READY worker threads must be 1/1")
    phases = operational.get("phase_timings")
    serial, parent = 0.0, None
    if phases is not None:
        serial = finite(phases.get("serial_replay_seconds"))
        collect = finite(phases.get("collect_seconds"), positive=True)
        parent = finite(phases.get("parent_seconds"), positive=True)
        if (
            phases.get("parent_frames") != spec["trajectory_frames"]
            or parent + serial > collect + 1e-6
            or collect > whole + 1e-6
        ):
            raise ValueError("NOT_READY phase timings do not reconcile")
    estimate = whole - serial
    if estimate <= 0:
        raise ValueError("NOT_READY nonpositive production estimate")
    result = {
        "schema": b2.SCHEMA + "/admission",
        "allocation_digest": spec["allocation_digest"],
        "whole_session_seconds": whole,
        "serial_replay_seconds": serial,
        "estimate_minutes": estimate / 60,
        "timing_ceiling_minutes": CEILING_SECONDS / 60,
        "outcomes_opened": False,
        "status": "ADMIT_UNCHANGED" if estimate <= CEILING_SECONDS else "NOT_READY_JUDGE",
    }
    if (
        estimate > CEILING_SECONDS
        and parent is not None
        and spec["trajectory_frames"] == 14400
        and "mechanical_rescale" not in spec
    ):
        projected = estimate - 4800 * parent / spec["trajectory_frames"]
        result["rescaled_estimate_minutes"] = projected / 60
        if 0 < projected <= CEILING_SECONDS:
            result["status"] = "RESCALE_REQUIRED"
    return result


def evaluate(spec_path: Path, receipt_path: Path) -> dict:
    """Re-derive qualified family proof, hash/backup verification and admission."""
    spec = b2.validate_spec(b2.read_export(spec_path))
    manifest_path = receipt_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("manifest_sha256") != digest(manifest_path)
        or manifest.get("script") != "research/bl_served_search/run_b2.sh"
        or manifest.get("env", {}).get("BL_B2_CANARY") != "1"
        or manifest.get("env", {}).get("BL_B2_INDEX") != "0"
        or manifest.get("bl_b2", {}).get("allocation_digest") != spec["allocation_digest"]
    ):
        raise ValueError("NOT_READY canary manifest identity differs")
    requested = json.loads(json.dumps(manifest))
    requested["env"].pop("SNAKE_COLAB_CANARY", None)
    requested["env"].pop("SNAKE_COLAB_CANARY_PACKAGE", None)
    requested["env"]["BL_B2_CANARY"] = "0"
    requested["canary"] = {
        "receipt": {"path": str(receipt_path.resolve()), "sha256": digest(receipt_path)}
    }
    canary.check(requested, b2.ROOT)
    # A family proof binds upload bytes. Also tie its spec to this exact request.
    if not any(
        row["dest"] == "b2-design.json" and row["sha256"] == digest(spec_path)
        for row in manifest["inputs"]
    ):
        raise ValueError("NOT_READY canary binds another design")
    backup = json.loads((receipt_path.parent / "backup.json").read_text())
    base = b2.drive_backup_root(Path(manifest["backup_dir"]))
    names = (receipt["job_id"], receipt["generation"])
    if any(not isinstance(n, str) or Path(n).name != n or n in {"", ".", ".."} for n in names):
        raise ValueError("NOT_READY invalid backup identity")
    destination = (base / names[0] / names[1]).resolve()
    if (
        backup.get("hash_verified") is not True
        or Path(backup["destination"]).resolve() != destination
    ):
        raise ValueError("NOT_READY backup binding differs")
    for row in receipt["exports"]:
        path = destination / row["path"]
        if not path.is_relative_to(destination) or path.resolve() != path or not path.is_file():
            raise ValueError("NOT_READY unsafe backup export")
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("NOT_READY backup export bytes differ")
    operational_path = receipt_path.parent / "exports/canary-admission.json"
    verify_export(operational_path)
    if not any(
        row["path"] == "canary-admission.json" and row["sha256"] == digest(operational_path)
        for row in receipt["exports"]
    ):
        raise ValueError("NOT_READY operational export is absent from broker inventory")
    result = decide(spec, json.loads(operational_path.read_text()), receipt["elapsed_seconds"])
    result.update(
        spec_sha256=digest(spec_path),
        receipt_path=str(receipt_path.resolve()),
        receipt_sha256=digest(receipt_path),
        commit=manifest["commit"],
        operational_sha256=digest(operational_path),
    )
    return result


def rescale(spec_path: Path, report_path: Path, out: Path) -> dict:
    """One timing-triggered fresh whole round; a new canary is always required."""
    spec = b2.validate_spec(b2.read_export(spec_path))
    report = b2.read_export(report_path)
    if (
        report != evaluate(spec_path, Path(report["receipt_path"]))
        or report["status"] != "RESCALE_REQUIRED"
    ):
        raise ValueError("NOT_READY no verified timing-only rescale authorization")
    if spec["trajectory_frames"] != 14400 or "mechanical_rescale" in spec:
        raise ValueError("NOT_READY the single rescale was already consumed")
    ruling_path = spec_path.parent.parent / "B2_PARAMETER_RULING.json"
    ruling, _ = b2_allocate.read_ruling(ruling_path)
    if digest(ruling_path) != spec["ruling"]["sha256"]:
        raise ValueError("NOT_READY rescale original ruling differs")
    if out.exists():
        raise FileExistsError("rescale output must be fresh")
    # Shared across worktrees and copies of the original registration. Claim
    # before auditing/allocating: a failed rescale preparation goes to the judge.
    claim_path = RESCALE_CLAIMS / (spec["allocation_digest"] + ".json")
    claim = {
        "original_spec_sha256": digest(spec_path),
        "admission_sha256": verify_export(report_path),
        "rescale_out": str(out.resolve()),
        "count": 1,
    }
    export_json(claim_path, claim)
    ruling = {
        **ruling,
        "trajectory_frames": 9600,
        "mechanical_rescale": {
            "original_spec_sha256": digest(spec_path),
            "admission_sha256": verify_export(report_path),
            "original_round_index": spec["round_index"],
            "count": 1,
            "claim_path": str(claim_path),
            "claim_sha256": verify_export(claim_path),
            "authority": (
                "round-16 JUDGMENT.md section (e)3; timing only; new excluded canary required"
            ),
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    export_json(out / "B2_PARAMETER_RULING.json", ruling)
    return b2_allocate.allocate(
        out / "B2_PARAMETER_RULING.json", out / "registration", start_round=spec["round_index"] + 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rescale-out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.spec, args.canary_receipt)
    export_json(args.out, result)
    if args.rescale_out:
        rescale(args.spec, args.out, args.rescale_out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
