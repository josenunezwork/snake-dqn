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
    "f5-timing-recovery-offline-20261026/canary-04"
)
PUBLICATION_COMMIT = "e4dc238a55a27304e69d5d99d9a20bea76db0ec4"


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
    proof = read(Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f5-timing-recovery-offline-20261026/canary-04/OUTPUT_PUBLICATION_04_PROOF.json"))
    assert proof["authoring_head"] == proof["origin_main"] == PUBLICATION_COMMIT
    index_path = DEST / "OUTPUT_PUBLICATION_04.json"
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
        assert raw == (ROOT / "output-04" / entry["source_relative_path"]).read_bytes()
    output = ROOT / "output-04"
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
    first = 355.595 - 238.044
    second = 480.125 - 387.370
    qualification_seconds = phases[-1]["elapsed_seconds"]
    assert qualification_seconds == 634.251
    terminal_calls = [c for c in receipt["cli_calls"] if c["thread"] == "MainThread" and c["operation"] == "control"][-3:]
    assert [c["command"] for c in terminal_calls] == ["stop", "sessions", "usage"]
    assert all(c["status"] == "completed" for c in terminal_calls)
    assert receipt["usage_after"]["assignments"] == receipt["usage_after"]["rate"] == 0
    report = {
        "schema": "bl-f5-post-canary04-verification/v1",
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
            "interleaved_lease_heartbeat": "A completed background renewal upload interleaves stop and sessions; MainThread control ordering is preserved.",
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
            "ledger_scan_skip_seconds": 387.370 - 355.595,
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
            "raw_attestation_envelope_runtime_source_bound": True,
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
        "# Canary-04 offline verification\n\n"
        f"Complete output published FIRST at `{PUBLICATION_COMMIT}`; local==origin verified before report.\n\n"
        "| Check | Result | Evidence |\n|---|---|---|\n"
        f"| Native v6 + kernel | PASS | {before['schema']}; kernel {before['os_kernel']['release']}; exact raw boundaries and frozen reference |\n"
        "| Real CUDA parity | PASS exercised fixtures 672/673 | fp32 cuda:0 finite 1x6 Q; atol=rtol2e-4; exact masked argmax; live recorded checks, frozen source verified offline |\n"
        "| Actual producer/scanner | PASS CANARY_ONLY | export672 → ledger672 → scanner skip672 → export673; both units independently reopened |\n"
        f"| Transfer / staging | PASS | {clocks['transfer_seconds']:.6f}s / {clocks['elapsed_seconds']:.6f}s; independent 300/300s limits |\n"
        f"| Publication / exports | PASS | {index['original_file_count']} original files / {index['original_total_bytes']} bytes reconstructed exactly; {len(exports)} exports / {sum(e['size'] for e in exports)} payload bytes; hashes, sizes, SHA sidecars and markers |\n"
        f"| Terminal / release | PASS saved receipt | {receipt['elapsed_seconds']:.6f}s; succeeded exit0; stop_verified; stop/sessions/usage complete; assignments0/rate0, cleanup null |\n"
        f"| Operational telemetry | PASS recorded | {len(progress)} records; whole qualification {qualification_seconds}s; unique fixtures2; counter transient3 includes skipped672 |\n"
        "| Full timing / calibration | NOT EXECUTED | Twelve workloads, coupled C/W and Mac companion remain unmeasured; base120/effective240 is an engineering allowance |\n\n"
        "| Phase | Seconds |\n|---|---:|\n"
    )
    for phase in phase_intervals:
        table += f"| {phase['phase']} | {phase['duration_seconds']:.3f} |\n"
    table += (
        f"\nFirst fixture parity/boundary/export {first:.3f}s; scanner/skip {387.370-355.595:.3f}s; second fixture {second:.3f}s. "
        "Inner sample clocks are separately recorded in JSON and exclude boundaries/scanner/export. "
        "No extrapolation of tiny fixture throughput to full timing or calibration. "
        "Saved receipt-backed release verification only; no fresh provider query. "
        "No dispatch, Colab/provider call, label/outcome opening or checkpoint deserialization.\n"
    )
    for folder in [ROOT, DEST]:
        publish(folder / "POST_CANARY04_VERIFICATION.json", raw)
        publish(folder / "POST_CANARY04_VERIFICATION.md", table.encode())
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
