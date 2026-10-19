"""Mac-side A3 re-qualification of a preserved packet; never launches compute."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from research.q0q1_20261017 import owner_admission, q0b_measure, q1_parity, qualification
from research.q0q1_20261017.make_q1_jobs import canonical, pretty_json, sha256_file
from research.q0q1_20261017.q0b_queue import bound_input, observed_pricing

SCHEMA = "q0q1-a3-requalification/v1"
PRESERVED_ATTEMPTS = {
    "A100": ("q0b-capacity-a100-20261018r6", "mature/n1"),
    "L4": ("q0b-capacity-l4-20261018r6", "mature/n3"),
}
# Preserve admission of the immutable A100 A3 archive produced before L4 support.
LEGACY_A100_TOOL_SHA256 = "44f7d8376e3ee4edc7827415a43bb665e173c5406322d865c32ff2c8f99e98fe"
R6_COMMIT = "98cb709b6febba82aa424b51796e1730e893ca77"
# A3 evidence is immutable.  These are the bytes that governed the preserved
# r6 packets, rather than the append-only A4 registration and current validator.
A3_REGISTRATION_SHA256 = "0b056540089473549493fc1d706796c912739e97d095bcc67843fa6c29309461"
A3_QUALIFICATION_SHA256 = "fee21f4d3208b19888c6253a4482bebba551cae04c7486118fc46e242a24b44a"
HISTORICAL_A3_TOOL_SHA256 = "e40f686bd9d5187eb87603d5ffac1388eb3c4610f1c3da81286044424b0e9b4e"
ROOT = Path(__file__).resolve().parents[2]


def preserved_attempt(original: dict) -> tuple[str, str]:
    """Select only the two disclosed pre-A3 r6 hardware/failure identities."""
    hardware = original.get("hardware")
    if hardware not in PRESERVED_ATTEMPTS:
        raise ValueError("A3 preserved-packet repair is limited to A100 r6 and L4 r6")
    job_id, cell = PRESERVED_ATTEMPTS[hardware]
    if original.get("qualification", {}).get("errors") != [
        f"cell {cell} has an unexplained policy-lag increase over its matched control"
    ]:
        raise ValueError("original packet did not fail solely on the disclosed lag check")
    return hardware, job_id


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _a3_registration_prefix() -> bytes:
    """Read the immutable A3 registration bytes from the recorded commit."""
    completed = subprocess.run(
        ["git", "show", "17a927c9:research/q0q1_20261017/PREREGISTRATION.md"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if (
        completed.returncode
        or hashlib.sha256(completed.stdout).hexdigest() != A3_REGISTRATION_SHA256
    ):
        raise ValueError("cannot recover the exact immutable A3 registration prefix")
    return completed.stdout


def _current_registration_has_a3_prefix() -> bool:
    """A4 is append-only, so preserved A3 provenance needs its original prefix intact."""
    return owner_admission.REGISTRATION.read_bytes().startswith(_a3_registration_prefix())


def archive_objects(archive_path: Path) -> dict[str, dict]:
    """Read only named JSON, never extract executable or checkpoint paths."""
    fields = archive_path.with_name(archive_path.name + ".sha256").read_text().split()
    if not fields or fields[0] != sha256_file(archive_path):
        raise ValueError("original archive SHA sidecar mismatch")
    names = (
        "attempt/measurement/receipt.json",
        "attempt/measurement/runtime-parity.json",
        "attempt/t22-cuda.json",
        "attempt/t24-cuda.json",
        "attempt/receipt.json",
        "attempt/queue-attempt-receipt.json",
    )
    objects = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in names:
            matches = [member for member in archive.getmembers() if member.name == name]
            limit = 128 * 1024**2 if name.endswith("runtime-parity.json") else 5_000_000
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > limit:
                raise ValueError(f"missing, duplicate or unsafe archive evidence: {name}")
            raw = archive.extractfile(matches[0]).read()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"archive evidence must be an object: {name}")
            objects[name] = {"value": value, "raw": raw}
    return objects


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("queue timing must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("queue timing must be finite and nonnegative")
    return float(value)


def qualify(
    *, archive_path: Path, claim_path: Path, terminal_path: Path, drive_root: Path
) -> tuple[dict, dict[str, bytes]]:
    objects = archive_objects(archive_path)
    original = objects["attempt/measurement/receipt.json"]["value"]
    hardware, job_id = preserved_attempt(original)
    if original.get("source") != q0b_measure.measurement_source_manifest():
        raise ValueError("preserved measurement source differs from the current pinned recipe")
    result = qualification.validate_packet(original)
    if result["status"] != "QUALIFIED":
        raise ValueError(f"A3 retained measurement is not qualified: {result['errors']}")
    claim = json.loads(claim_path.read_bytes())
    terminal = json.loads(terminal_path.read_bytes())
    for field in ("id", "generation", "token", "worker_id"):
        if claim.get(field) is None or claim[field] != terminal.get(field):
            raise ValueError(f"queue claim/terminal mismatch: {field}")
    if (
        claim["id"] != job_id
        or terminal.get("status") != "failed"
        or terminal.get("exit_code") != 64
        or terminal.get("gpu") != hardware
        or terminal.get("verified_commit") != R6_COMMIT
        or terminal.get("commit") != R6_COMMIT
        or claim.get("capabilities", {}).get("gpu") != hardware
        or objects["attempt/queue-attempt-receipt.json"]["value"].get("job_id") != job_id
    ):
        raise ValueError("queue evidence does not identify the preserved failed r6 attempt")
    claimed = dt.datetime.fromisoformat(claim["created_at"].replace("Z", "+00:00"))
    if claimed.tzinfo is None:
        raise ValueError("queue claim time lacks timezone")
    started, run_started, finished, elapsed, run_seconds = (
        _seconds(terminal.get(field))
        for field in (
            "started_at",
            "run_started_at",
            "finished_at",
            "elapsed_seconds",
            "run_seconds",
        )
    )
    if not claimed.timestamp() <= started <= run_started <= finished:
        raise ValueError("queue claim/start/run/finish times are unordered")
    if abs(finished - started - elapsed) > 0.01 or abs(finished - run_started - run_seconds) > 0.01:
        raise ValueError("queue receipt elapsed/run times do not reconcile")
    inclusive = finished - claimed.timestamp()
    measured = _seconds(original["packet_seconds"])
    if inclusive < measured or inclusive > qualification.PACKET_SECONDS:
        raise ValueError("inclusive claim-to-terminal time cannot cover measurement within cap")
    pricing_path, pricing = bound_input(drive_root, "q0q1-20261017/admission/observed-pricing.json")
    reference_path, reference = bound_input(
        drive_root, "q0q1-20261017/admission/mac-parity-reference.json"
    )
    rate, cpu_rate = observed_pricing(pricing, hardware)
    if (
        original.get("unit_rate_per_hour") != rate
        or original.get("observed_cpu_unit_rate_per_hour") != cpu_rate
    ):
        raise ValueError("preserved packet debit rates disagree with bound pricing")
    runtime = objects["attempt/measurement/runtime-parity.json"]
    if hashlib.sha256(runtime["raw"]).hexdigest() != original.get("runtime_parity_receipt_sha256"):
        raise ValueError("runtime parity digest mismatch")
    if runtime["value"].get("reference") != reference:
        raise ValueError("runtime parity does not embed bound Mac reference")
    from research.q0q1_20261017 import make_q1_jobs

    make_q1_jobs._validate_mac_reference(reference_path)
    q1_parity.validate_runtime_receipt(
        runtime["value"],
        bank_sha256=reference["bank_sha256"],
        b8_sha256=reference["b8_sha256"],
        source_manifest=make_q1_jobs.q1_evaluate.source_manifest(),
    )
    for name in ("t22", "t24"):
        raw = objects[f"attempt/{name}-cuda.json"]["raw"]
        if hashlib.sha256(raw).hexdigest() != original["hashes"][f"{name}_receipt"]:
            raise ValueError(f"{name} fixture digest mismatch")
    receipt = copy.deepcopy(original)
    receipt["packet_seconds"] = inclusive
    receipt["stages"]["setup_seconds"] += inclusive - measured
    receipt["observed_gpu_hours"] = inclusive / 3600
    receipt["observed_units"] = rate * inclusive / 3600
    receipt["observed_cpu_hours"] = 0.0
    receipt["hashes"].update(
        observed_pricing=sha256_file(pricing_path), mac_reference=sha256_file(reference_path)
    )
    overheads = [
        max(
            0.0,
            cell["observed_launcher_seconds"]
            - cell["warmup_seconds"]
            - sum(w["seconds"] for w in cell["scored_windows"]),
        )
        for cell in receipt["cells"]
    ]
    receipt["q1_attempt_overhead_seconds"] = (
        inclusive - measured + 4 * receipt["cpu_seconds_per_episode"] + max(overheads) + 60
    )
    receipt["queue_setup_includes_arbitration"] = True
    receipt["packet_seconds_includes_actual_export_cleanup"] = True
    receipt["stages"]["export_reserve_seconds"] = 0.0
    receipt["requalification"] = {
        "schema": SCHEMA,
        "amendment": "A3",
        "status": "QUALIFIED_FROM_PRESERVED_MEASUREMENT",
        "job_id": job_id,
        "original_queue_status": "failed",
        "original_archive_sha256": sha256_file(archive_path),
        "original_measurement_sha256": hashlib.sha256(
            objects["attempt/measurement/receipt.json"]["raw"]
        ).hexdigest(),
        "claim_sha256": sha256_file(claim_path),
        "terminal_sha256": sha256_file(terminal_path),
        "claim_raw": claim_path.read_text(),
        "terminal_raw": terminal_path.read_text(),
        "original_measurement_raw": objects["attempt/measurement/receipt.json"]["raw"].decode(),
        "claim": claim,
        "terminal": terminal,
        "registration_sha256": A3_REGISTRATION_SHA256,
        "validator_sha256": A3_QUALIFICATION_SHA256,
        "tool_sha256": sha256_file(Path(__file__)),
        "measurement_content_sha256": digest(
            {k: receipt[k] for k in ("source", "config", "cells", "parity")}
        ),
        "inclusive_claim_to_terminal_seconds": inclusive,
        "q1_prelude_upper_bound_seconds": inclusive - measured,
        "accounting_note": (
            "Actual terminal time includes export/cleanup; no additional GPU attempt or CPU debit. "
            "Q1 prelude uses all outside-measurement time as conservative upper bound."
        ),
    }
    receipt["qualification"] = qualification.validate_packet(receipt)
    if receipt["qualification"]["status"] != "QUALIFIED":
        raise ValueError(f"inclusive A3 packet failed: {receipt['qualification']['errors']}")
    payload = {
        name: data["raw"] for name, data in objects.items() if name != "attempt/receipt.json"
    }
    payload["attempt/original-wrapper-receipt.json"] = objects["attempt/receipt.json"]["raw"]
    payload["attempt/receipt.json"] = pretty_json(receipt)
    payload["attempt/claim.json"] = claim_path.read_bytes()
    payload["attempt/terminal.json"] = terminal_path.read_bytes()
    return receipt, payload


def validate_receipt(receipt: dict) -> str:
    """Verify the A3 repair provenance before omitting its failed queue dependency."""
    proof = receipt.get("requalification", {})
    hardware = receipt.get("hardware")
    attempt = PRESERVED_ATTEMPTS.get(hardware)
    allowed_tools = {sha256_file(Path(__file__)), HISTORICAL_A3_TOOL_SHA256}
    if hardware == "A100":
        allowed_tools.add(LEGACY_A100_TOOL_SHA256)
    if (
        proof.get("schema") != SCHEMA
        or proof.get("status") != "QUALIFIED_FROM_PRESERVED_MEASUREMENT"
        or attempt is None
        or proof.get("job_id") != attempt[0]
        or proof.get("registration_sha256") != A3_REGISTRATION_SHA256
        or proof.get("validator_sha256") != A3_QUALIFICATION_SHA256
        or not _current_registration_has_a3_prefix()
        or proof.get("tool_sha256") not in allowed_tools
        or proof.get("measurement_content_sha256")
        != digest({k: receipt.get(k) for k in ("source", "config", "cells", "parity")})
    ):
        raise ValueError("invalid or stale A3 re-qualification provenance")
    for key in (
        "original_archive_sha256",
        "original_measurement_sha256",
        "claim_sha256",
        "terminal_sha256",
    ):
        value = proof.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("missing re-qualification evidence digest")
    raw_values = {}
    for name, binding in (
        ("claim", "claim_sha256"),
        ("terminal", "terminal_sha256"),
        ("original_measurement", "original_measurement_sha256"),
    ):
        raw = proof.get(f"{name}_raw")
        if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != proof[binding]:
            raise ValueError("re-qualification embedded evidence hash mismatch")
        raw_values[name] = json.loads(raw)
    claim, terminal, original = (
        raw_values[k] for k in ("claim", "terminal", "original_measurement")
    )
    if preserved_attempt(original) != (hardware, attempt[0]):
        raise ValueError("re-qualification original hardware mismatch")
    if original.get("source") != q0b_measure.measurement_source_manifest():
        raise ValueError("preserved measurement source differs from the current pinned recipe")
    if proof.get("claim") != claim or proof.get("terminal") != terminal:
        raise ValueError("re-qualification embedded queue evidence mismatch")
    if proof.get("amendment") != "A3" or proof.get("original_queue_status") != "failed":
        raise ValueError("re-qualification amendment or queue status mismatch")
    for field in ("id", "generation", "token", "worker_id"):
        if claim.get(field) is None or claim[field] != terminal.get(field):
            raise ValueError("re-qualification queue identity mismatch")
    if (
        claim["id"] != attempt[0]
        or terminal.get("status") != "failed"
        or terminal.get("exit_code") != 64
    ):
        raise ValueError("re-qualification queue status mismatch")
    claimed = dt.datetime.fromisoformat(claim["created_at"].replace("Z", "+00:00"))
    if claimed.tzinfo is None:
        raise ValueError("re-qualification claim timezone missing")
    started, run_started, finished, elapsed, run_seconds = (
        _seconds(terminal.get(k))
        for k in ("started_at", "run_started_at", "finished_at", "elapsed_seconds", "run_seconds")
    )
    inclusive = finished - claimed.timestamp()
    if (
        not claimed.timestamp() <= started <= run_started <= finished
        or abs(finished - started - elapsed) > 0.01
        or abs(finished - run_started - run_seconds) > 0.01
        or not 0 < inclusive <= qualification.PACKET_SECONDS
    ):
        raise ValueError("re-qualification queue time mismatch")
    if (
        terminal.get("gpu") != hardware
        or claim.get("capabilities", {}).get("gpu") != hardware
        or terminal.get("verified_commit") != R6_COMMIT
        or terminal.get("commit") != R6_COMMIT
    ):
        raise ValueError("re-qualification hardware/commit mismatch")
    for field in (
        "source",
        "config",
        "cells",
        "parity",
        "runtime",
        "environment",
        "runtime_parity_receipt_sha256",
        "unit_rate_per_hour",
        "observed_cpu_unit_rate_per_hour",
        "cpu_seconds_per_episode",
        "cpu_worker_efficiency",
    ):
        if receipt.get(field) != original.get(field):
            raise ValueError(f"re-qualification changed preserved measurement: {field}")
    outside = inclusive - _seconds(original["packet_seconds"])
    if outside < 0:
        raise ValueError("re-qualification time undercounts measurement")
    cell_overhead = max(
        max(
            0.0,
            c["observed_launcher_seconds"]
            - c["warmup_seconds"]
            - sum(w["seconds"] for w in c["scored_windows"]),
        )
        for c in receipt["cells"]
    )
    expected = {
        "packet_seconds": inclusive,
        "observed_gpu_hours": inclusive / 3600,
        "observed_cpu_hours": 0.0,
        "observed_units": receipt["unit_rate_per_hour"] * inclusive / 3600,
        "q1_attempt_overhead_seconds": outside
        + 4 * receipt["cpu_seconds_per_episode"]
        + cell_overhead
        + 60,
    }
    for field, value in expected.items():
        if _seconds(receipt.get(field)) != value:
            raise ValueError(f"re-qualification accounting mismatch: {field}")
    if (
        proof.get("inclusive_claim_to_terminal_seconds") != inclusive
        or proof.get("q1_prelude_upper_bound_seconds") != outside
        or receipt["stages"]["setup_seconds"] != original["stages"]["setup_seconds"] + outside
        or receipt["stages"]["cleanup_seconds"] != original["stages"]["cleanup_seconds"]
    ):
        raise ValueError("re-qualification inclusive stage mismatch")
    if qualification.validate_packet(receipt)["status"] != "QUALIFIED":
        raise ValueError("re-qualification packet no longer qualifies")
    return attempt[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # Publish only a reviewable receipt from committed source, never dirty code.
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT).returncode:
        parser.error("commit A3 before re-qualifying preserved data")
    if args.out_dir.exists():
        parser.error("refusing to replace an existing re-qualification directory")
    receipt, payload = qualify(
        archive_path=args.archive,
        claim_path=args.claim,
        terminal_path=args.terminal,
        drive_root=args.drive_root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    archive_path = args.out_dir / "result.tar.gz"
    with archive_path.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in sorted(payload.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    archive_path.with_name(archive_path.name + ".sha256").write_text(
        f"{sha256_file(archive_path)}  result.tar.gz\n"
    )
    print(
        json.dumps(
            {
                "status": receipt["qualification"]["status"],
                "archive": str(archive_path),
                "sha256": sha256_file(archive_path),
                "packet_seconds": receipt["packet_seconds"],
                "observed_units": receipt["observed_units"],
                "q1_attempt_overhead_seconds": receipt["q1_attempt_overhead_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
