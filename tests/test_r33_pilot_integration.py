"""Finite integration checks for the R33 packet, without launching test subprocesses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.r33_verification import contract, inventory, pilot, pilot_runtime, verify
from tools.ops import package_spec as spec


def test_required_missing_skip_and_xfail_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="INCOMPLETE"):
        inventory.validate_collection(["expected"], [], 0)
    with pytest.raises(ValueError):
        inventory.validate_collection(["expected"], ["expected", "foreign"], 0)
    for node in ("skipped", "failure", "error"):
        report = tmp_path / f"{node}.xml"
        report.write_text(
            f"<testsuites><testsuite><testcase><{node}/></testcase></testsuite></testsuites>"
        )
        with pytest.raises(ValueError, match="INCOMPLETE"):
            inventory.validate_reports(report, 1, 0)
    report = tmp_path / "valid.xml"
    report.write_text("<testsuites><testsuite><testcase/></testsuite></testsuites>")
    inventory.validate_reports(report, 1, 0)
    with pytest.raises(ValueError):
        inventory.validate_reports(report, 2, 0)


def _env(monkeypatch: pytest.MonkeyPatch, inputs: Path) -> dict[str, str]:
    identity = {
        "pin": "b" * 40,
        "environment_sha256": contract.environment_identity(pilot_runtime.ROOT),
        "packet": spec.digest(inventory.INVENTORY),
    }
    relative = "tools/ops/package_spec.py"
    path = pilot_runtime.ROOT / relative
    files = {relative: {"sha256": spec.digest(path), "mode": path.stat().st_mode & 0o777}}
    identity["source_closure_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    spec.publish_json(inputs / "source-inventory.json", {"commit": identity["pin"], "files": files})
    monkeypatch.setenv("COLAB_INPUT_DIR", str(inputs))
    for key, value in {
        "R33_NAMESPACE": contract.NAMESPACE,
        "R33_ALLOCATION": contract.ALLOCATION,
        "R33_PIN": identity["pin"],
        "R33_ENVIRONMENT_SHA256": identity["environment_sha256"],
        "R33_PACKET": identity["packet"],
        "R33_SOURCE_CLOSURE": identity["source_closure_sha256"],
    }.items():
        monkeypatch.setenv(key, value)
    return identity


def test_runtime_interruption_resume_exact_bytes_and_foreign_refusal(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _env(monkeypatch, tmp_path / "inputs")
    ns, allocation = contract.NAMESPACE, contract.ALLOCATION
    ledger = tmp_path / "empty.json"
    spec.publish_json(ledger, spec.empty_ledger(ns, allocation, identity))
    count = 0

    def interrupted(nodes):
        nonlocal count
        count += 1
        if count == 2:
            raise InterruptedError("synthetic excluded interruption")

    monkeypatch.setattr(inventory, "run_checks", interrupted)
    first = tmp_path / "first"
    with pytest.raises(InterruptedError):
        pilot_runtime.run(ledger, first)
    expected = {spec.unit_id(ns, allocation, index) for index in range(3)}
    partial_ledger = tmp_path / "partial.json"
    result = verify.union([first], partial_ledger, ns, allocation, identity, expected)
    assert [row["index"] for row in result["units"]] == [0]
    calls = []
    monkeypatch.setattr(inventory, "run_checks", lambda nodes: calls.append(nodes))
    resumed = tmp_path / "resumed"
    receipt = pilot_runtime.run(partial_ledger, resumed)
    assert receipt["skipped_verified_indices"] == [0]
    assert receipt["new_indices"] == [1, 2] and len(calls) == 2
    union = tmp_path / "union.json"
    verify.union([first, resumed], union, ns, allocation, identity, expected)
    assert (
        verify.decide(
            [first, resumed], union, tmp_path / "decision.json", ns, allocation, identity
        )["verified_units"]
        == 3
    )
    plain = tmp_path / "plain"
    pilot_runtime.run(ledger, plain)
    for index, attempt in ((0, first), (1, resumed), (2, resumed)):
        assert spec.digest(attempt / "shards" / f"shard-{index:06d}.json") == spec.digest(
            plain / "shards" / f"shard-{index:06d}.json"
        )
    with pytest.raises(spec.PackageRefused, match="duplicate"):
        verify.union(
            [plain, resumed], tmp_path / "duplicate.json", ns, allocation, identity, expected
        )
    foreign = tmp_path / "foreign.json"
    spec.publish_json(foreign, spec.empty_ledger(ns, allocation, {**identity, "pin": "c" * 40}))
    with pytest.raises(spec.PackageRefused):
        pilot_runtime.run(foreign, tmp_path / "foreign-output")


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_payload_sidecar_marker_interruptions_are_excluded(
    tmp_path: Path, monkeypatch, boundary: int
) -> None:
    original = spec._publish
    count = 0

    def interrupted(path, value):
        nonlocal count
        if count == boundary:
            raise InterruptedError("excluded publication interruption")
        count += 1
        original(path, value)

    monkeypatch.setattr(spec, "_publish", interrupted)
    with pytest.raises(InterruptedError):
        spec.publish_json(tmp_path / "attempt/shards/shard-000000.json", {"synthetic": True})
    monkeypatch.setattr(spec, "_publish", original)
    ledger = verify.union(
        [tmp_path / "attempt"],
        tmp_path / "ledger.json",
        "ns",
        "allocation",
        {"pin": "a" * 40},
        {"ns/allocation/000000"},
    )
    assert ledger["units"] == []


def test_missing_ledger_and_bad_dest_refuse_before_allocation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spec, "require_published_commit", lambda *_args: "b" * 40)
    with pytest.raises(spec.PackageRefused, match="uploaded verified-ledger"):
        spec.build_cpu_manifest(
            repo=tmp_path,
            commit="b" * 40,
            output_dir=tmp_path / "out",
            staged_inputs=[],
            script="job.sh",
            namespace="ns",
            allocation="al",
            authority_receipts=[{"sha256": "a" * 64}],
        )
    source = tmp_path / "source"
    source.write_text("synthetic")
    for bad in ("../escape", "/absolute", "a/../b", "a//b"):
        with pytest.raises(spec.PackageRefused):
            spec.stage_inputs(tmp_path / "stage", [spec.Input(source, bad)])
    (tmp_path / "stage").mkdir(exist_ok=True)
    (tmp_path / "stage/link").symlink_to(tmp_path / "outside", target_is_directory=True)
    (tmp_path / "outside").mkdir()
    with pytest.raises(spec.PackageRefused):
        spec.stage_inputs(tmp_path / "stage", [spec.Input(source, "link/escaped")])
    assert not (tmp_path / "outside/escaped").exists()


def test_stale_procedure_canary_and_missing_snapshot_refused(tmp_path: Path) -> None:
    # This packet has no trajectory/checkpoint. A required snapshot of any future
    # package remains an immutable-artifact obligation, with no weights-only waiver.
    with pytest.raises(spec.PackageRefused):
        spec.verify_immutable(tmp_path / "required-full-state-snapshot.json")
    identity = {"pin": "a" * 40, "environment_sha256": "b" * 64, "packet": "c" * 64}
    base = {
        "schema": "r33-mac-preflight/v1",
        "status": "PASS",
        "identity": identity,
        "units": 3,
        "scrubbed": True,
        "excluded": True,
        "allocation_attempted": False,
        "effective_nice": 15,
        "commit": "a" * 40,
        "source_closure_sha256": "d" * 64,
        "entrypoint": ["/bin/bash", "changed-procedure.sh"],
    }
    preflight, independent = tmp_path / "preflight.json", tmp_path / "v.json"
    spec.publish_json(preflight, base)
    spec.publish_json(
        independent,
        {
            "status": "PASS_IMPLEMENTATION_LIGHT_VERIFICATION",
            "commit": "a" * 40,
            "source_closure_sha256": "d" * 64,
        },
    )
    with pytest.raises(spec.PackageRefused, match="stale"):
        pilot.validate_evidence(preflight, independent, "a" * 40, identity, "d" * 64)


def test_cap_exhaustion_and_timeout_after_success_never_authorize_completion(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    spec.publish_json(
        receipt,
        {
            "stop_verified": True,
            "status": "INCOMPLETE",
            "failure_class": "RPC_EXEC_POLL_TIMEOUT",
            "terminal": {"returncode": 0},
        },
    )
    with pytest.raises(spec.PackageRefused, match="verified stop"):
        verify.completion_admission(
            [receipt],
            tmp_path / "ledger.json",
            attempts_used=1,
            series_minutes=240,
            series_units=0.32,
        )
    for attempts in (0, 3):
        with pytest.raises(spec.PackageRefused):
            verify.completion_admission(
                [receipt],
                tmp_path / "ledger.json",
                attempts_used=attempts,
                series_minutes=289,
                series_units=0.385,
            )


def _replace_test_receipt(path: Path, value: dict) -> None:
    # Every mutation is a new synthetic fixture, never production evidence.
    for suffix in ("", ".sha256", ".complete"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    spec.publish_json(path, value)


def test_completion_frozen_envelope_and_contiguous_receipts(tmp_path: Path) -> None:
    identity = {"pin": "b" * 40, "environment_sha256": "c" * 64, "packet": "d" * 64}
    manifest = {
        "commit": identity["pin"],
        "job_id": "original",
        "timeout_minutes": 240,
        "budget": {"unit_cap": 0.32, "rate_units_per_hour": 0.08},
        "r33": {"identity": identity, "namespace": "ns", "allocation": "al", "attempt": 0},
    }
    original = tmp_path / "manifest.json"
    original.write_text(json.dumps(manifest))
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "job_id": "original",
        "commit": identity["pin"],
        "manifest_sha256": spec.digest(original),
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "generation": "g",
        "terminal": {"returncode": -15, "generation": "g"},
        "elapsed_seconds": 60,
        "accounting": {"priced_rate_units_per_hour": 0.08},
        "exports": [],
    }
    _replace_test_receipt(receipt_path, receipt)
    ledger = tmp_path / "ledger.json"
    spec.publish_json(ledger, spec.empty_ledger("ns", "al", identity))
    kwargs = {
        "attempts_used": 1,
        "series_minutes": 1,
        "series_units": 0.08 / 60,
        "manifest": manifest,
    }
    assert verify.completion_admission([receipt_path], ledger, **kwargs)["missing_indices"] == [
        0,
        1,
        2,
    ]
    receipt["elapsed_seconds"] = 240 * 60 + 1
    _replace_test_receipt(receipt_path, receipt)
    with pytest.raises(spec.PackageRefused, match="per-attempt"):
        verify.completion_admission([receipt_path], ledger, **kwargs)
    receipt["elapsed_seconds"] = 60
    _replace_test_receipt(receipt_path, receipt)
    manifest["r33"]["attempt"] = 1
    original.write_text(json.dumps(manifest))
    receipt["manifest_sha256"] = spec.digest(original)
    _replace_test_receipt(receipt_path, receipt)
    with pytest.raises(spec.PackageRefused, match="noncontiguous"):
        verify.completion_admission([receipt_path], ledger, **kwargs)


def test_sparse_pass_receipt_cannot_establish_pilot_acceptance(tmp_path: Path) -> None:
    receipt, decision, release = (
        tmp_path / "receipt.json",
        tmp_path / "decision.json",
        tmp_path / "release.json",
    )
    spec.publish_json(receipt, {"status": "PASS", "stop_verified": True})
    spec.publish_json(decision, {"status": "PASS_LINUX_CONTRACTS"})
    spec.publish_json(release, {"endpoint": "missing"})
    with pytest.raises(spec.PackageRefused, match="manifest"):
        verify.accept_supervisor({}, receipt, decision, tmp_path / "accepted.json", release)


def test_pilot_acceptance_requires_exact_manifest_exports_and_sealed_receipts(
    tmp_path: Path,
) -> None:
    identity = {"pin": "a" * 40, "packet": "b" * 64, "environment_sha256": "c" * 64}
    manifest = {
        "commit": identity["pin"],
        "job_id": "r33-cpu-pilot-01",
        "r33": {"identity": identity, "attempt": 0, "prior_receipt_sha256": []},
        "budget": {"unit_cap": 0.32},
    }
    snapshot = tmp_path / "manifest.json"
    snapshot.write_text(json.dumps(manifest))
    terminal = tmp_path / "exports/terminal-receipt.json"
    spec.publish_json(terminal, {"status": "ATTEMPT_COMPLETE"})
    decision = tmp_path / "decision.json"
    spec.publish_json(
        decision,
        {
            "status": "PASS_LINUX_CONTRACTS",
            "identity": identity,
            "verified_units": 3,
            "native_pending": ["native host qualification"],
        },
    )
    receipt = tmp_path / "receipt.json"
    spec.publish_json(
        receipt,
        {
            "status": "PASS",
            "stop_verified": True,
            "commit": identity["pin"],
            "job_id": manifest["job_id"],
            "manifest_sha256": spec.digest(snapshot),
            "session": "unique",
            "generation": "g",
            "endpoint": "e",
            "elapsed_seconds": 60,
            "terminal": {"generation": "g", "returncode": 0, "status": "completed"},
            "accounting": {
                "allocated_endpoint": "e",
                "priced_rate_units_per_hour": 0.08,
                "observed_rate_units_per_hour": 0.08,
                "unaccounted_endpoints": [],
                "calibration": {"status": "PASS"},
                "last_crosscheck": {"balance_crosscheck_status": "OK"},
            },
            "exports": [
                {
                    "path": "terminal-receipt.json",
                    "sha256": spec.digest(terminal),
                    "size": terminal.stat().st_size,
                }
            ],
        },
    )
    release = tmp_path / "release.json"
    spec.publish_json(
        release,
        {
            "schema": "r33-release-observation/v1",
            "endpoint": "e",
            "sessions": {},
            "supervisor_receipt_sha256": spec.digest(receipt),
        },
    )
    assert (
        verify.accept_supervisor(manifest, receipt, decision, tmp_path / "accepted.json", release)[
            "status"
        ]
        == "PASS"
    )
    changed = {**manifest, "budget": {"unit_cap": 0.64}}
    with pytest.raises(spec.PackageRefused, match="manifest changed"):
        verify.accept_supervisor(changed, receipt, decision, tmp_path / "bad.json", release)
    Path(f"{receipt}.complete").unlink()
    with pytest.raises(spec.PackageRefused, match="immutable export"):
        verify.accept_supervisor(manifest, receipt, decision, tmp_path / "bad.json", release)
