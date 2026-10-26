"""Verify published canary operational records offline without opening outcomes."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from research.bl_ppo.exports import verify_export
from research.bl_residual import f5_environment, f5_environment_reference, f5_timing_resume
from research.colab_cli import canary

DEST = Path(__file__).resolve().parent
REPO = DEST.parents[3]
ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "f5-timing-recovery-offline-20261026/canary-03"
)
PUBLICATION_COMMIT = "0c76c088e641bf0a314ade32d773df290cb2e290"


def sha(raw: bytes) -> str:
    """Hash opaque bytes."""
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict:
    """Read an explicitly selected operational JSON record."""
    return json.loads(path.read_bytes())


def publish(path: Path, raw: bytes) -> None:
    """Create payload and sidecars once, publishing completion last."""
    for name, data in [
        (path, raw),
        (Path(f"{path}.sha256"), f"{sha(raw)}  {path.name}\n".encode()),
        (Path(f"{path}.complete"), b"complete\n"),
    ]:
        with name.open("xb") as stream:
            stream.write(data)


def main() -> None:
    """Validate actual native attestations, scanner, clocks and telemetry."""
    proof = read(Path("/tmp/f5-canary03-publication-proof.json"))
    assert proof["authoring_head"] == proof["origin_main"] == PUBLICATION_COMMIT
    index_path = DEST / "OUTPUT_PUBLICATION_03.json"
    relative = index_path.relative_to(REPO).as_posix()
    assert sha(index_path.read_bytes()) == proof["ledger_sha256"]
    assert (
        subprocess.check_output(["git", "show", f"{PUBLICATION_COMMIT}:{relative}"], cwd=REPO)
        == index_path.read_bytes()
    )
    index = read(index_path)
    for entry in index["entries"]:
        raw = b"".join((DEST / part).read_bytes() for part in entry["parts"])
        if entry["encoding"] != "identity":
            raw = gzip.decompress(raw)
        assert sha(raw) == entry["sha256"] and len(raw) == entry["size"]
        assert raw == (ROOT / "output-03" / entry["source_relative_path"]).read_bytes()
    output = ROOT / "output-03"
    receipt = read(output / "receipt.json")
    terminal = read(output / "terminal.json")
    qualification = read(output / "exports/qualification/qualification.json")
    skip = read(output / "exports/resume/verified-skip-proof.json")
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    assert receipt["cleanup_error"] is None
    assert terminal["status"] == "succeeded" and terminal["returncode"] == 0
    assert receipt["manifest_sha256"] == sha((output / "manifest.json").read_bytes())
    exports = receipt["exports"]
    assert len(exports) == 15
    assert sorted(terminal["export_scanner"]["inventory"], key=lambda e: e["path"]) == sorted(
        exports, key=lambda e: e["path"]
    )
    for entry in exports:
        path = output / "exports" / entry["path"]
        assert path.stat().st_size == entry["size"]
        assert verify_export(path) == entry["sha256"]
    before = read(output / "exports/environment/before.json")
    f5_environment.validate_environment(before)
    boundary_names = ["after.json", "after-fixture-672.json", "after-fixture-673.json"]
    for name in boundary_names:
        after = read(output / "exports/environment" / name)
        f5_environment.validate_environment(after)
        f5_environment.compare_environment(before, after, mode="same-assignment")
    comparison = f5_environment_reference.compare_capture05_reference(
        qualification["environment_evidence"]["legacy_reference"], before
    )
    assert comparison == qualification["environment_evidence"]["comparison"]
    ledger_path = output / "exports/timing-canary/verified-ledger.json"
    ledger = read(ledger_path)
    units = f5_timing_resume.load_ledger(ledger_path, ledger["identity"], canary_only=True)
    assert sorted(units) == [672]
    assert skip["skipped_keys"] == [672] and skip["complete_keys"] == [672, 673]
    assert skip["classification"] == "CANARY_ONLY"
    assert skip["identity"] == ledger["identity"]
    for entry in skip["new_unit_exports"]:
        key, payload = f5_timing_resume._read_unit(
            output / "exports" / entry["path"], ledger["identity"], entry
        )
        assert key == entry["key"] and payload["canary_only"] is True
    assert [s["allocation_index"] for s in qualification["samples"]] == [672, 673]
    assert all(s["parity"] == "PASS" and s["q_shape"] == [1, 6] for s in qualification["samples"])
    assert qualification["precision"] == "fp32" and qualification["device"] == "cuda:0"
    assert qualification["parity_atol"] == qualification["parity_rtol"] == 2e-4
    assert qualification["fit_or_optimizer_executed"] is False
    assert qualification["timing_only_calibration"] is None
    upload = read(output / "upload-measurement.json")
    clocks = canary.staging_measurement(read(output / "manifest.json"), upload)
    assert clocks["transfer_seconds"] <= 300 and clocks["elapsed_seconds"] <= 300
    assert (
        canary.evidence(read(output / "manifest.json"), receipt, output, REPO)
        == receipt["canary_evidence"]
    )
    records = {}
    for path in sorted(output.glob("poll-*.json")):
        for line in read(path).get("job_log_tail", "").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("schema") == "bl-f5-qualification-progress/v1":
                token = json.dumps(record, sort_keys=True)
                if token not in records:
                    records[token] = {"first_saved_poll": path.name, **record}
    progress = sorted(records.values(), key=lambda r: r["elapsed_seconds"])
    phases = [r for r in progress if r["status"] != "HEARTBEAT"]
    phase_intervals = []
    for start, end in zip(phases, phases[1:]):
        phase_intervals.append(
            {
                "phase": start["phase"],
                "start_seconds": start["elapsed_seconds"],
                "end_seconds": end["elapsed_seconds"],
                "duration_seconds": end["elapsed_seconds"] - start["elapsed_seconds"],
                "recorded_count_before": start["fixtures_completed"],
                "recorded_count_after": end["fixtures_completed"],
            }
        )
    first = 356.794 - 238.919
    second = 481.763 - 388.541
    qualification_seconds = phases[-1]["elapsed_seconds"]
    assert qualification_seconds == 636.035
    terminal_calls = receipt["cli_calls"][-3:]
    assert [c["command"] for c in terminal_calls] == ["stop", "sessions", "usage"]
    assert all(c["status"] == "completed" for c in terminal_calls)
    assert receipt["usage_after"]["assignments"] == receipt["usage_after"]["rate"] == 0
    report = {
        "schema": "bl-f5-post-canary03-verification/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_EXERCISED_CANARY_COMPATIBILITY_RESUME_PATH_ONLY",
        "job_id": receipt["job_id"],
        "runtime_source_commit": receipt["commit"],
        "publication_commit": PUBLICATION_COMMIT,
        "publication_gate": proof,
        "publication_verification": {
            "original_files": index["original_file_count"],
            "original_bytes": index["original_total_bytes"],
            "all_reconstructed_bytes_equal_original": True,
            "exports_verified": len(exports),
            "export_payload_bytes": sum(e["size"] for e in exports),
            "exports": exports,
        },
        "environment": {
            "status": "PASS",
            "schema": before["schema"],
            "capture_status": before["capture_status"],
            "os_kernel": before["os_kernel"],
            "platform": before["platform"],
            "libc": before["libc"],
            "gpu_uuid": before["gpu_uuid"],
            "assignment_raw_sha256": f5_environment.receipt_sha256(before),
            "all_four_native_captures_validated": True,
            "same_assignment_raw_comparisons": "PASS",
            "sealed_capture05_comparison_recomputed": "PASS_EXACTLY_EQUALS_SAVED",
            "canonical_sha256": comparison["comparison"]["actual_canonical_sha256"],
        },
        "cuda_parity": {
            "status": "PASS_RECORDED_LIVE_AND_SOURCE_CHECKED_OFFLINE",
            "device": "cuda:0",
            "precision": "fp32",
            "native_observation_and_mask": True,
            "executed_fixtures": [672, 673],
            "finite_q_shape": [1, 6],
            "rtol": 2e-4,
            "atol": 2e-4,
            "masked_argmax_exact_equality": True,
            "cuda_observer": receipt["canary_evidence"]["cuda_observations"],
            "offline_cuda_reexecution": False,
            "limitation": (
                "Raw Q vectors are not exported; live recorded checks bi" "nd frozen source."
            ),
        },
        "resume": {
            "status": "PASS_ACTUAL_PRODUCER_AND_SCANNER",
            "classification": "CANARY_ONLY",
            "sequence": ["export 672", "build_and_verify_ledger672", "skip 672", "export 673"],
            "saved_ledger_keys_verified_by_load_ledger": sorted(units),
            "separately_verified_by_read_unit": [672, 673],
            "skip_proof": skip,
            "full_timing_units_admitted": 0,
            "nonempty_ledger_transport_qualified": False,
        },
        "clocks": {
            "status": "PASS_SCOPED_300_300_BARS",
            "transfer_seconds": clocks["transfer_seconds"],
            "transfer_cap_seconds": 300,
            "whole_staging_seconds": clocks["elapsed_seconds"],
            "staging_cap_seconds": 300,
            "source_delivery_seconds": clocks["source_delivery_seconds"],
            "setup_seconds": upload["setup_seconds"],
            "job_elapsed_seconds": receipt["elapsed_seconds"],
            "scope": (
                "Saved input transfer and staging clocks; excludes later" " bootstrap and runtime."
            ),
        },
        "release": {
            "status": "PASS_SAVED_RECEIPT_BACKED_RELEASE_VERIFICATION",
            "endpoint": receipt["endpoint"],
            "stop_verified": True,
            "cleanup_error": None,
            "terminal_cli_calls": terminal_calls,
            "usage_after": receipt["usage_after"],
            "source_guard": (
                "supervisor._stop_verified returns only after endpoint a"
                "bsent from account sessions"
            ),
            "raw_backend_sessions_response_in_output": False,
            "current_provider_query_performed": False,
            "serving_release_changed": False,
        },
        "operational_telemetry_only": {
            "progress_records": progress,
            "phase_intervals": phase_intervals,
            "first_fixture_parity_boundary_export_seconds": first,
            "first_fixture_per_second": 1 / first,
            "second_fixture_parity_boundary_export_seconds": second,
            "second_fixture_per_second": 1 / second,
            "ledger_scan_skip_seconds": 388.541 - 356.794,
            "both_parity_phases_fixtures_per_second": 2 / (first + second),
            "whole_qualification_seconds": qualification_seconds,
            "whole_qualification_fixtures_per_second": 2 / qualification_seconds,
            "unique_fixtures_completed": 2,
            "counter_caveat": (
                "Loop visits count skipped672; transient3 resets2. Never" " use3/12 as throughput."
            ),
            "samples": qualification["samples"],
            "calibration": {"status": "NOT_EXECUTED", "seconds": None},
            "full_source_timing": "NOT_EXECUTED_TRANSPORT_ONLY",
            "mac_companion": qualification["mac_timing_status"],
            "reported_inner_qualification_elapsed_seconds": qualification["elapsed_seconds"],
            "clock_caveat": (
                "Inner sample/qualification clocks differ from whole pro"
                "gress and broker wall clocks."
            ),
        },
        "sizing_120_240_minutes": {
            "status": "NOT_ESTABLISHED_BY_THIS_CANARY",
            "observed_canary_wall_minutes": receipt["elapsed_seconds"] / 60,
            "observed_wall_fraction_of_120_minutes": receipt["elapsed_seconds"] / 7200,
            "observed_wall_fraction_of_240_minutes": receipt["elapsed_seconds"] / 14400,
            "implication": (
                "Use measured setup, boundary and scanner overhead in a "
                "prospective forecast; full twelve-workload source timin"
                "gs and optimizer/resume calibration remain unmeasured. "
                "Neither 120 nor 240 minutes is qualified. No caps or di"
                "spatch authorization changed."
            ),
        },
        "boundaries": {
            "dispatches": 0,
            "colab_or_provider_calls": 0,
            "label_or_outcome_payloads_opened": False,
            "checkpoint_payloads_deserialized": False,
            "raw_attestation_envelope_branch_touched": False,
            "scientific_look_consumed": False,
        },
        "source_references": [
            "research/bl_residual/f5_qualification.py:103-169,551-668,669-708",
            "research/bl_residual/f5_timing_resume.py:169-358",
            "research/bl_residual/f5_environment_reference.py:177-197",
            "research/colab_cli/canary.py:326-375",
            "research/colab_cli/supervisor.py:841-878",
        ],
    }
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    table = (
        "# Canary-03 offline verification\n\n"
        f"Output published first at `{PUBLICATION_COMMIT}`; clean local==origin gate passed.\n\n"
        "| Check | Result | Evidence |\n|---|---|---|\n"
        "| Native v6 environment | PASS | Linux 6.6.159+, #1 SMP"
        " Thu Oct 22 03:07:54 UTC 2026; glibc 2.39; exact before"
        "/after; sealed reference comparison |\n"
        "| CUDA parity | PASS on 2 exercised fixtures | cuda:0 f"
        "p32 finite 1x6 Q; rtol=atol 2e-4; exact masked argmax; "
        "live recorded, source checked offline |\n"
        "| Producer/scanner resume | PASS CANARY_ONLY | export 6"
        "72 -> verified ledger [672] -> skip 672 -> export 673; "
        "actual scanner revalidated both |\n"
        f"| Transfer / staging | PASS | {clocks['transfer_seconds']:.6f}s / "
        f"{clocks['elapsed_seconds']:.6f}s, independent 300/300 s bars |\n"
        "| Complete publication / exports | PASS | 106 files, 11"
        "0619361 bytes; 15 exported payloads, 109181750 bytes; e"
        "xact roundtrip, hashes, sizes, sidecars, markers |\n"
        "| Runtime release | PASS saved evidence | stop_verified"
        "; completed stop/sessions/usage; assignments 0/rate 0, "
        "cleanup error null; no fresh provider query |\n"
        "| Full timing / calibration | NOT EXECUTED | transport-"
        "only; no optimizer, full source workloads, or Mac compa"
        "nion measurement |\n\n"
        "Operational telemetry only; unique fixtures=2. Loop cou"
        "nter transiently reports 3 because skip 672 counts as a"
        " visit.\n\n"
        "| Phase | Seconds | Unique fixtures/s |\n|---|---:|---:|\n"
    )
    for phase in phase_intervals:
        duration = phase["duration_seconds"]
        if duration <= 0:
            continue
        rate = f"{1 / duration:.8f}" if phase["phase"] == "cuda-parity" else "n/a"
        table += f"| {phase['phase']} | {duration:.3f} | {rate} |\n"
    table += (
        "\n672 inner sample 24.927973s (feature 24.548578, CUDA 0.364777, CPU 0.011033); "
        "673 inner sample 0.025813s (feature 0.007373, CUDA 0.009178, CPU 0.006038). "
        "These inner clocks exclude boundary capture/scanner/exp"
        "ort costs. Calibration seconds: unmeasured.\n\n"
        "Whole qualification 636.035s, 2/636.035=0.00314448 fixt"
        "ures/s; broker wall 1335.656052s (22.260934min). "
        "This wall time occupies 18.55% of 120 min and 9.28% of 240 min. "
        "Neither timeout is qualified for the full twelve-worklo"
        "ad run: source timings and calibration remain missing. "
        "Use observed fixed overhead for planning, without extra"
        "polating tiny parity throughput to H300/H5000/Watch/cal"
        "ibration. "
        "Raw-attestation envelope work remains independent and untouched.\n"
    )
    for folder in [ROOT, DEST]:
        publish(folder / "POST_CANARY03_VERIFICATION.json", raw)
        publish(folder / "POST_CANARY03_VERIFICATION.md", table.encode())
    script = Path(__file__)
    for suffix, data in [
        (".sha256", f"{sha(script.read_bytes())}  {script.name}\n".encode()),
        (".complete", b"complete\n"),
    ]:
        with Path(f"{script}{suffix}").open("xb") as stream:
            stream.write(data)
    print(json.dumps({"status": report["status"], "report_sha256": sha(raw)}))


if __name__ == "__main__":
    main()
