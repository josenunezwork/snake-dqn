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


JOB_ALIASES = ("job", "job_id", "JOB", "Job_ID", "jobId", "Job ID")


def _write_and_scan(path: Path, root: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")
    _scan_retained_text(
        path, output_root=root, expected_job_id="train-01", expected_generation="g-1"
    )


def _nested_record(record: dict, depth: int) -> dict:
    value = record
    for _ in range(depth):
        value = {"nested": [value]}
    return value


@pytest.mark.parametrize("filename", FILENAMES)
def test_generated_record_admission_matrix(tmp_path: Path, filename: str) -> None:
    """Every filename/alias/case/depth cell admits its twin before its one defect."""
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    defects = {
        "foreign": "foreign job_id",
        "conflict": "conflicting .* identities",
        "generation": "foreign generation",
        "failure_class": "non-receipt failure_class",
        "incomplete": "unknown execution status|failed status",
        "terminated": "unknown execution status|failed status",
        "no_identity": "unknown execution status|failed status",
    }
    cases = 0
    for alias, case, depth, defect in itertools.product(
        JOB_ALIASES, (str.lower, str.upper, str.title), (0, 1, 4, 12), defects
    ):
        record = {alias: "train-01", "generation": "g-1"}
        if defect == "no_identity":
            record = {"status": case("running")}
        _write_and_scan(path, tmp_path, _nested_record(record, depth))
        if defect == "foreign":
            record[alias] = "foreign"
        elif defect == "conflict":
            record["job_id" if alias.lower() == "job" else "job"] = "foreign"
        elif defect == "generation":
            record["generation"] = "foreign"
        elif defect == "failure_class":
            record["failure_class"] = case("CHILD_NONZERO_EXIT")
        elif defect == "no_identity":
            record["status"] = case("cancelled")
        else:
            record["status"] = " " + case(defect) + " "
        with pytest.raises(ResumeRefused, match=defects[defect]):
            _write_and_scan(path, tmp_path, _nested_record(record, depth))
        cases += 1
    assert cases == len(JOB_ALIASES) * 3 * 4 * len(defects)


@pytest.mark.parametrize("alias", ["returncode", "exit_code", "exitCode", "returnCode"])
@pytest.mark.parametrize("depth", [0, 1, 4, 12])
@pytest.mark.parametrize("defect", ["nonzero", "watchdog", "malformed"])
def test_generated_returncode_alias_twins(
    tmp_path: Path, alias: str, depth: int, defect: str
) -> None:
    path = tmp_path / "arbitrary.json"
    record = {"jobId": "train-01", "generation": "g-1", alias: 0}
    _write_and_scan(path, tmp_path, _nested_record(record, depth))
    values = {"nonzero": 1, "watchdog": 124, "malformed": "1"}
    reasons = {
        "nonzero": "unexplained/non-mechanical",
        "watchdog": "watchdog timeout",
        "malformed": "malformed returncode",
    }
    record[alias] = values[defect]
    with pytest.raises(ResumeRefused, match=reasons[defect]):
        _write_and_scan(path, tmp_path, _nested_record(record, depth))


@pytest.mark.parametrize("alias", ["failure_class", "failureClass", "Failure Class"])
@pytest.mark.parametrize("depth", [0, 1, 4, 12])
def test_generated_failure_class_alias_twins(tmp_path: Path, alias: str, depth: int) -> None:
    path = tmp_path / "arbitrary.json"
    record = {"job": "train-01", "generation": "g-1", alias: ""}
    _write_and_scan(path, tmp_path, _nested_record(record, depth))
    record[alias] = "CHILD_NONZERO_EXIT"
    with pytest.raises(ResumeRefused, match="non-receipt failure_class"):
        _write_and_scan(path, tmp_path, _nested_record(record, depth))


@pytest.mark.parametrize("context", ["metrics", "metadata"])
def test_scientific_data_context_accepts_producer_vocabulary(tmp_path: Path, context: str) -> None:
    path = tmp_path / "exports/training/proxy-0128.json"
    path.parent.mkdir(parents=True)
    data = {"status": "measured", "generation": "Ampere"}
    _write_and_scan(path, tmp_path, {context: data})
    # The same values are execution signals when moved out of the explicit data scope.
    with pytest.raises(ResumeRefused, match="foreign generation"):
        _write_and_scan(path, tmp_path, {"diagnostic": data})


@pytest.mark.parametrize("context", ["metrics", "metadata"])
@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("failureClass", "CHILD_NONZERO_EXIT", "non-receipt failure_class"),
        ("exitCode", 1, "unexplained/non-mechanical"),
        ("jobId", "foreign", "foreign job_id"),
    ],
)
def test_scientific_data_context_never_hides_execution_faults(
    tmp_path: Path, context: str, key: str, value: object, reason: str
) -> None:
    path = tmp_path / "exports/training/proxy-0128.json"
    path.parent.mkdir(parents=True)
    record = {"status": "measured", "generation": "Ampere"}
    _write_and_scan(path, tmp_path, {context: record})
    record["diagnostic"] = {key: value}
    with pytest.raises(ResumeRefused, match=reason):
        _write_and_scan(path, tmp_path, {context: record})


@pytest.mark.parametrize("context", ["metrics", "metadata"])
@pytest.mark.parametrize("execution_scope", ["state", "terminal"])
@pytest.mark.parametrize("defect", ["status", "generation"])
def test_explicit_execution_scope_resets_scientific_vocabulary(
    tmp_path: Path, context: str, execution_scope: str, defect: str
) -> None:
    path = tmp_path / "exports/training/proxy-0128.json"
    path.parent.mkdir(parents=True)
    record = {"job_id": "train-01", "generation": "g-1", "status": "running"}
    _write_and_scan(path, tmp_path, {context: {execution_scope: record}})
    record[defect] = "measured" if defect == "status" else "Ampere"
    reason = (
        "unknown execution status|failed status" if defect == "status" else "foreign generation"
    )
    with pytest.raises(ResumeRefused, match=reason):
        _write_and_scan(path, tmp_path, {context: {execution_scope: record}})


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
            _write_and_scan(path, tmp_path, {"record": {**identity, "status": case("running")}})
            path.write_text(json.dumps({"record": {**identity, "status": case(status)}}))
            with pytest.raises(ResumeRefused, match="unknown execution status|failed status"):
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
    valid = {
        "status": "running",
        "reason": "timeout",
        "failure_class": "",
        "returncode": 0,
        "job": "train-01",
        "generation": "g-1",
    }
    baseline = {"job": "train-01", "generation": "g-1", key: valid[key]}
    _write_and_scan(path, tmp_path, baseline)
    baseline[key] = value
    path.write_text(json.dumps(baseline))
    reasons = {
        "status": "malformed execution status",
        "reason": "malformed execution reason",
        "failure_class": "non-receipt failure_class",
        "returncode": "malformed returncode",
        "job": "foreign job_id",
        "generation": "foreign generation",
    }
    if key == "returncode" and type(value) is int:
        reasons[key] = "unexplained/non-mechanical"
    with pytest.raises(ResumeRefused, match=reasons[key]):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


def test_mixed_log_and_validated_receipt_aliases_are_checked(tmp_path: Path) -> None:
    from research.sandbox_a_resident.test_training_resume import manifest, receipt

    path = tmp_path / "strange.log"
    path.write_text('progress\n{"job":"train-01"}\n')
    _scan_retained_text(
        path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    path.write_text('progress\n{"job":"foreign"}\n')
    with pytest.raises(ResumeRefused, match="foreign job_id"):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )
    path.unlink()
    for extra, reason in (
        ({"job": "foreign"}, "conflicting .* identities"),
        ({"diagnostic": {"failure_class": "SUPERVISOR"}}, "non-receipt failure_class"),
    ):
        baseline = receipt()
        (tmp_path / "receipt.json").write_text(json.dumps(baseline))
        verify_mechanical_stop(baseline, manifest(), "b" * 64, output_root=tmp_path)
        rec = {**baseline, **extra}
        (tmp_path / "receipt.json").write_text(json.dumps(rec))
        with pytest.raises(ResumeRefused, match=reason):
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
@pytest.mark.parametrize("binding", ["publication", "poll", "qualified-poll"])
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
        identity = {"job": "train-01", "generation": "g-1", "status": "running"}
        record = identity if binding == "poll" else {"state": identity, "terminal": None}
        record["exports"] = [
            {
                "path": "training/iteration-0128.pt.xz",
                "size": len(content),
                "sha256": __import__("hashlib").sha256(content).hexdigest(),
            }
        ]
        (tmp_path / "poll-1.json").write_text(json.dumps(record))
    path = training / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path != canonical or binding != "publication":
        path.write_bytes(content[:40] if binding == "publication" else content)
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    path.write_bytes(b"\xfd7zXZ\x00FloatingPointError renamed diagnostic")
    with pytest.raises(
        ResumeRefused, match="transport partial|corrupt retained publication sidecar"
    ):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("name", STAGED_PAYLOADS)
def test_generated_unbound_transport_names_refuse(tmp_path: Path, name: str) -> None:
    path = tmp_path / "exports/training" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"\xfd7zXZ\x00unbound checkpoint"
    canonical = tmp_path / "exports/training/iteration-0128.pt.xz"
    publish(canonical, lambda p: p.write_bytes(content))
    path.write_bytes(content)
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    Path(str(canonical) + ".complete").unlink()
    Path(str(canonical) + ".sha256").unlink()
    if path != canonical:
        canonical.unlink()
    with pytest.raises(ResumeRefused, match="unverifiable retained model transport partial"):
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
    with pytest.raises(
        ResumeRefused, match="malformed retained .*sidecar|malformed retained completion"
    ):
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
    with pytest.raises(ResumeRefused, match="transport partial differs from verified payload"):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("alias", JOB_ALIASES)
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
    poll_path = tmp_path / "poll-1.json"
    poll_path.write_text(json.dumps(record))
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    if conflict == "foreign":
        record[alias] = "foreign"
    elif conflict == "alias":
        record["job_id" if alias.lower() == "job" else "job"] = "foreign"
    elif conflict == "generation":
        record["generation"] = "foreign"
    else:
        record["failure_class"] = "CHILD_NONZERO_EXIT"
    poll_path.write_text(json.dumps(record))
    reasons = {
        "foreign": "foreign job_id",
        "alias": "conflicting .* identities",
        "generation": "foreign generation",
        "failure_class": "non-receipt failure_class",
    }
    with pytest.raises(ResumeRefused, match=reasons[conflict]):
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


@pytest.mark.parametrize("explicit", [False, True])
def test_symlink_above_checkout_is_not_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
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
    if explicit:
        seal = json.loads((packet / "SEAL.json").read_text())
        seal["prior_packet_path"] = str(link / "checkout/research/package/prior")
        (packet / "SEAL.json").write_text(json.dumps(seal))
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
    with pytest.raises(ResumeRefused, match="unverifiable retained model transport partial"):
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
    with pytest.raises(ResumeRefused, match="corrupt retained publication sidecar"):
        _check_output_root_fault_evidence(tmp_path)


def test_deep_record_validation_is_iterative(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _scan_json_faults

    baseline = {"job": "train-01"}
    record = {"job": "foreign"}
    for _ in range(2000):
        baseline = {"nested": baseline}
        record = {"nested": record}
    _scan_json_faults(baseline, label="deep", expected_job_id="train-01", expected_generation="g-1")
    with pytest.raises(ResumeRefused, match="foreign job_id"):
        _scan_json_faults(
            record, label="deep", expected_job_id="train-01", expected_generation="g-1"
        )
    path = tmp_path / "too-deep.json"
    path.write_text('{"nested":' * 2000 + '{"job":"foreign"}' + "}" * 2000)
    # Decoder depth support varies by Python: either a parse refusal or the
    # foreign identity itself must be reported; no unrelated refusal is allowed.
    with pytest.raises(
        ResumeRefused, match="foreign job_id|unreadable or partial retained JSON evidence"
    ):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("conflict", ["job", "job_id", "generation", "failure_class", "nested"])
def test_qualified_poll_state_never_excuses_other_scope_faults(
    tmp_path: Path, conflict: str
) -> None:
    import hashlib

    data = b"\xfd7zXZ\x00bound checkpoint"
    path = tmp_path / "exports/training/.transfer-stage/data"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    record = {
        "state": {"job_id": "train-01", "generation": "g-1", "status": "running"},
        "exports": [
            {
                "path": "training/iteration-0128.pt.xz",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    poll_path = tmp_path / "poll-1.json"
    poll_path.write_text(json.dumps(record))
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    if conflict == "nested":
        record["diagnostic"] = {"failure_class": "SUPERVISOR"}
    else:
        record[conflict] = "SUPERVISOR" if conflict == "failure_class" else "foreign"
    poll_path.write_text(json.dumps(record))
    reasons = {
        "job": "foreign job_id",
        "job_id": "foreign job_id",
        "generation": "foreign generation",
        "failure_class": "non-receipt failure_class",
        "nested": "non-receipt failure_class",
    }
    with pytest.raises(ResumeRefused, match=reasons[conflict]):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("binding", ["publication", "qualified-poll"])
def test_transport_json_is_bound_and_still_scanned(
    tmp_path: Path, failure: bool, binding: str
) -> None:
    training = tmp_path / "exports/training"
    directory = training / ".transfer-stage"
    directory.mkdir(parents=True)
    data = directory / "data"

    def retain(content: bytes) -> None:
        data.write_bytes(content)
        if binding == "publication":
            for sidecar in (
                training / "proxy-0128.json.sha256",
                training / "proxy-0128.json.complete",
            ):
                sidecar.unlink(missing_ok=True)
            (training / "proxy-0128.json").unlink(missing_ok=True)
            publish(training / "proxy-0128.json", lambda p: p.write_bytes(content))
        else:
            (tmp_path / "poll-1.json").write_text(
                json.dumps(
                    {
                        "state": {"job_id": "train-01", "generation": "g-1", "status": "running"},
                        "exports": [
                            {
                                "path": "training/proxy-0128.json",
                                "size": len(content),
                                "sha256": digest(data),
                            }
                        ],
                    }
                )
            )

    retain(json.dumps({"status": "PASS"}).encode())
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    if failure:
        retain(json.dumps({"failure_class": "CHILD_NONZERO_EXIT"}).encode())
        with pytest.raises(ResumeRefused, match="non-receipt failure_class"):
            _check_output_root_fault_evidence(
                tmp_path, expected_job_id="train-01", expected_generation="g-1"
            )
