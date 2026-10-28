"""Generated admission matrices: filenames never decide identity or byte safety."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_resume import (
    ResumeRefused,
    _check_output_root_fault_evidence,
    _scan_retained_text,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.units import digest, publish

FILENAMES = (
    "diagnostic",
    "supervisor.log",
    "job.log.partial",
    "status.json",
    "poll-1.json.partial",
    "unknown.json.tmp-a",
    "proxy-1.jsonl",
    "usage-1.json",
    "receipt.json",
    "nested/any/file.bin",
)


@pytest.mark.parametrize("filename", FILENAMES)
def test_generated_record_admission_matrix(tmp_path: Path, filename: str) -> None:
    """Sweep filename x alias x value case x depth x identity/failure conflict."""
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = 0
    for alias, case, depth, defect in itertools.product(
        ("job", "job_id", "JOB", "Job_ID"),
        (str.lower, str.upper, str.title),
        (0, 1, 4, 12),
        (
            "foreign",
            "conflict",
            "generation",
            "failure_class",
            "incomplete",
            "terminated",
            "no_identity",
        ),
    ):
        record = {alias: "train-01", "generation": "g-1"}
        if defect == "foreign":
            record[alias] = "foreign"
        elif defect == "conflict":
            record["job_id" if alias.lower() == "job" else "job"] = "foreign"
        elif defect == "generation":
            record["generation"] = "foreign"
        elif defect == "failure_class":
            record["failure_class"] = case("CHILD_NONZERO_EXIT")
        elif defect == "no_identity":
            record = {"status": case("cancelled")}
        else:
            record["status"] = " " + case(defect) + " "
        value = record
        for _ in range(depth):
            value = {"nested": [value]}
        path.write_text(json.dumps(value) + "\n")
        with pytest.raises(ResumeRefused):
            _scan_retained_text(
                path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
            )
        cases += 1
    assert cases == 336


@pytest.mark.parametrize(
    "status",
    [
        "killed",
        "canceled",
        "cancelled",
        "timed_out",
        "timeout",
        "diverged",
        "halted",
        "invalid",
        "refused",
        "failed",
        "crashed",
        "aborted",
        "incomplete",
        "stopped",
        "terminated",
    ],
)
def test_generated_failure_states_never_depend_on_identity(tmp_path: Path, status: str) -> None:
    for case in (str.lower, str.upper, str.title):
        for identity in ({}, {"job": "train-01", "generation": "g-1"}):
            path = tmp_path / "arbitrary"
            path.write_text(json.dumps({"record": {**identity, "status": case(status)}}))
            with pytest.raises(ResumeRefused):
                _scan_retained_text(
                    path,
                    output_root=tmp_path,
                    expected_job_id="train-01",
                    expected_generation="g-1",
                )


@pytest.mark.parametrize(
    "key", ["status", "reason", "failure_class", "returncode", "job", "generation"]
)
@pytest.mark.parametrize("value", [[], {}, 123, True])
def test_malformed_signals_refuse_without_crash(tmp_path: Path, key: str, value: object) -> None:
    path = tmp_path / "unusual"
    path.write_text(json.dumps({key: value}))
    with pytest.raises(ResumeRefused):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


def test_mixed_log_and_validated_receipt_aliases_are_checked(tmp_path: Path) -> None:
    from research.sandbox_a_resident.test_training_resume import manifest, receipt

    path = tmp_path / "strange.log"
    path.write_text('progress\n{"job":"foreign"}\n')
    with pytest.raises(ResumeRefused):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )
    path.unlink()
    for extra in ({"job": "foreign"}, {"diagnostic": {"failure_class": "SUPERVISOR"}}):
        rec = {**receipt(), **extra}
        (tmp_path / "receipt.json").write_text(json.dumps(rec))
        with pytest.raises(ResumeRefused):
            verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)


STAGED_PAYLOADS = (
    "iteration-0128.pt.xz",
    "iteration-0128.pt.xz.partial",
    "iteration-0128.pt.xz.part",
    "iteration-0128.pt.xz.tmp",
    "iteration-0128.pt.xz.tmp-a.b",
    ".transfer-abcd",
    ".transfer-abcd.0",
    ".transfer-stage/data",
    ".transfer-stage/nested/.transfer-abcd",
)


@pytest.mark.parametrize("name", STAGED_PAYLOADS)
@pytest.mark.parametrize("binding", ["publication", "poll"])
def test_generated_transport_names_require_byte_proof(
    tmp_path: Path, name: str, binding: str
) -> None:
    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    canonical = training / "iteration-0128.pt.xz"
    content = b"\xfd7zXZ\x00" + b"bound checkpoint state" * 8
    if binding == "publication":
        publish(canonical, lambda p: p.write_bytes(content))
    else:
        (tmp_path / "poll-1.json").write_text(
            json.dumps(
                {
                    "job": "train-01",
                    "generation": "g-1",
                    "exports": [
                        {
                            "path": "training/iteration-0128.pt.xz",
                            "size": len(content),
                            "sha256": __import__("hashlib").sha256(content).hexdigest(),
                        }
                    ],
                }
            )
        )
    path = training / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path != canonical or binding != "publication":
        path.write_bytes(content[:40] if binding == "publication" else content)
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    path.write_bytes(b"\xfd7zXZ\x00FloatingPointError renamed diagnostic")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("name", STAGED_PAYLOADS)
def test_generated_unbound_transport_names_refuse(tmp_path: Path, name: str) -> None:
    path = tmp_path / "exports/training" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xfd7zXZ\x00unbound checkpoint")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("suffix", ["", ".partial", ".part", ".tmp", ".tmp12", ".tmp-a.b"])
@pytest.mark.parametrize("kind", ["sha256", "complete"])
@pytest.mark.parametrize("length", [0, 1, 4, 8, 12, 63, 64, 65])
def test_generated_truncated_sidecars(tmp_path: Path, suffix: str, kind: str, length: int) -> None:
    payload = tmp_path / "proxy-1.json"
    payload.write_text('{"status":"PASS"}')
    expected = (digest(payload) + "\n").encode() if kind == "sha256" else b"complete\n"
    sidecar = Path(str(payload) + "." + kind + suffix)
    # Without canonical completion, every prefix is recognizable but never finalizes.
    sidecar.write_bytes(expected[:length])
    if kind == "complete" and not suffix and length >= 9:
        Path(str(payload) + ".sha256").write_text(digest(payload) + "\n")
    _check_output_root_fault_evidence(tmp_path)
    sidecar.write_bytes(b"FloatingPointError")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("kind", ["sha256", "complete"])
def test_supervisor_transfer_directory_sidecars_are_bound(tmp_path: Path, kind: str) -> None:
    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    payload = training / "iteration-0128.pt.xz"
    publish(payload, lambda p: p.write_bytes(b"\xfd7zXZ\x00bound state"))
    directory = training / ".transfer-stage"
    directory.mkdir()
    (directory / "data").write_bytes(payload.read_bytes())
    sidecar = directory / kind
    sidecar.write_bytes(digest(payload).encode()[:10] if kind == "sha256" else b"comp")
    _check_output_root_fault_evidence(tmp_path)
    (directory / "data").write_bytes(b"unbound")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("alias", ["job", "job_id"])
@pytest.mark.parametrize("conflict", ["foreign", "alias", "generation", "failure_class"])
def test_poll_binding_must_pass_the_same_record_validator(
    tmp_path: Path, alias: str, conflict: str
) -> None:
    import hashlib

    data = b"\xfd7zXZ\x00bound-looking fault bytes"
    path = tmp_path / "exports/training/.transfer-stage/data"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    record = {
        alias: "train-01",
        "generation": "g-1",
        "exports": [
            {
                "path": "training/iteration-0128.pt.xz",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    if conflict == "foreign":
        record[alias] = "foreign"
    elif conflict == "alias":
        record["job_id" if alias == "job" else "job"] = "foreign"
    elif conflict == "generation":
        record["generation"] = "foreign"
    else:
        record["failure_class"] = "CHILD_NONZERO_EXIT"
    (tmp_path / "poll-1.json").write_text(json.dumps(record))
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("raw", [None, 7, [], {}, "relative/prior"])
def test_prior_packet_path_malformed_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: object
) -> None:
    from research.sandbox_a_resident import training_preflight

    (tmp_path / "SEAL.json").write_text(
        json.dumps(
            {
                "schema": "sandbox-a-train01-continuation-seal/v1",
                "prior_stop_evidence_kind": "receipt",
                "prior_packet": "prior",
                "prior_packet_path": raw,
            }
        )
    )
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _: None)
    with pytest.raises(ValueError, match="absolute path string"):
        training_preflight._validate_continuation_stop(tmp_path, {"completion_attempt": {}})


def test_symlink_above_checkout_is_not_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.sandbox_a_resident import training_preflight

    repo = tmp_path / "real/checkout"
    packet = repo / "research/package/packet"
    packet.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (packet / "SEAL.json").write_text(
        json.dumps(
            {
                "schema": "sandbox-a-train01-continuation-seal/v1",
                "prior_stop_evidence_kind": "receipt",
                "prior_packet": "prior",
            }
        )
    )
    link = tmp_path / "mount"
    link.symlink_to(repo.parent, target_is_directory=True)
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _: None)
    original = training_preflight._regular_packet_file

    def probe(path: Path) -> None:
        if path.name == "dispatch-manifest.json":
            raise RuntimeError("reached prior manifest")
        original(path)

    monkeypatch.setattr(training_preflight, "_regular_packet_file", probe)
    with pytest.raises(RuntimeError, match="reached prior manifest"):
        training_preflight._validate_continuation_stop(
            link / "checkout/research/package/packet", {"completion_attempt": {}}
        )


@pytest.mark.parametrize(
    "name", [".transfer-sidecar", ".transfer-dir/.transfer-sidecar", ".snake-transfer-x.assembling"]
)
@pytest.mark.parametrize("data", [b"", b"a" * 32, b"b" * 64 + b"\n", b"comp", b"complete\n"])
def test_anonymous_sidecar_transfers_require_recognizable_bytes(
    tmp_path: Path, name: str, data: bytes
) -> None:
    path = tmp_path / "exports/training" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _check_output_root_fault_evidence(tmp_path)
    path.write_bytes(b"INCOMPLETE failure_class=CHILD_NONZERO_EXIT")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(tmp_path)


def test_hash_bound_unfinalized_canonical_is_not_a_resume_unit(tmp_path: Path) -> None:
    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    payload = training / "iteration-0128.pt.xz"
    payload.write_bytes(b"\xfd7zXZ\x00hash bound but incomplete")
    Path(str(payload) + ".sha256").write_text(digest(payload) + "\n")
    _check_output_root_fault_evidence(tmp_path)
    assert not Path(str(payload) + ".complete").exists()
    payload.write_bytes(b"\xfd7zXZ\x00changed bytes")
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(tmp_path)


def test_deep_record_validation_is_iterative(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _scan_json_faults

    record = {"job": "foreign"}
    for _ in range(2000):
        record = {"nested": record}
    with pytest.raises(ResumeRefused):
        _scan_json_faults(
            record, label="deep", expected_job_id="train-01", expected_generation="g-1"
        )
    path = tmp_path / "too-deep.json"
    path.write_text('{"nested":' * 2000 + '{"job":"foreign"}' + "}" * 2000)
    with pytest.raises(ResumeRefused):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )
