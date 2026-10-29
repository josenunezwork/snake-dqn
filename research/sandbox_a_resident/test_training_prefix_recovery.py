"""Receipt-less c2 positive paths and fail-closed hash/cutoff/fault twins."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident import training_prefix_recovery as prefix
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_c2_readiness3 import refuses
from research.sandbox_a_resident.test_training_compact import sample_state
from research.sandbox_a_resident.training_compact import publish_compact_snapshot
from research.sandbox_a_resident.training_units import allocation
from research.sandbox_a_resident.units import digest, publish


def fixture(tmp_path: Path) -> dict:
    output = tmp_path / "attempt/output"
    exports = output / "exports/training"
    exports.mkdir(parents=True)
    control = {
        "study": "sandbox-a-train01",
        "source_commit": "a" * 40,
        "outer_iterations": 384,
        "snapshot_interval": 16,
    }
    state = sample_state()
    state["iteration"] = 96
    publish_compact_snapshot(
        exports,
        96,
        lambda dst: torch.save(state, dst),
        metadata={
            "allocation": allocation(control),
            "source_commit": "a" * 40,
            "prepare_phase": "ready",
        },
        source_packet_id="packet-train01-15-c1-i0064",
    )
    publish(exports / "iteration-0096-actor.pth", lambda dst: dst.write_bytes(b"actor"))
    manifest = {"job_id": "train01-r6", "commit": "a" * 40, "output_dir": str(output), "inputs": []}
    (output / "manifest.json").write_text(json.dumps(manifest))
    generation = "b" * 32
    inventory = [
        {"path": "training/" + path.name, "size": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(exports.iterdir())
        if not path.name.endswith((".sha256", ".complete"))
    ]
    poll = {
        "state": {"job_id": manifest["job_id"], "generation": generation, "status": "running"},
        "terminal": None,
        "exports": inventory,
        "job_log_tail": 'TRAIN01_PROGRESS {"stage":"iteration-complete","iteration":96}\n',
    }
    (output / "poll-000000.json").write_text(json.dumps({**poll, "exports": []}))
    (output / "poll-000001.json").write_text(json.dumps(poll))
    log = tmp_path / "supervisor.log"
    event = {
        "job": manifest["job_id"],
        "elapsed": 100.0,
        "status": "running",
        "terminal": False,
        "exports": len(inventory),
        "job_log_tail": poll["job_log_tail"],
    }
    log.write_text("PREFLIGHT_OK\n" + json.dumps(event) + "\n")
    return {
        "output": output,
        "exports": exports,
        "control": control,
        "manifest": manifest,
        "log": log,
        "event": event,
        "poll": poll,
    }


def build(fx: dict, path: Path, kind: str = "host_loss") -> dict:
    return prefix.build_prefix_recovery_certificate(
        fx["output"],
        fx["control"],
        fx["log"],
        path,
        stop_kind=kind,
        operator_confirmed_stopped=True,
        source_packet_id="packet-train01-15-c1-i0064",
        source_packet_manifest_sha256="c" * 64,
    )


def verify(fx: dict, path: Path) -> dict:
    return r.verify_recovery_evidence(
        path,
        fx["manifest"],
        digest(fx["output"] / "manifest.json"),
        output_root=fx["output"],
        source_control=fx["control"],
        source_packet_id="packet-train01-15-c1-i0064",
        source_packet_manifest_sha256="c" * 64,
    )


def rewrite(path: Path, value: dict) -> None:
    for end in ("", ".sha256", ".complete"):
        Path(str(path) + end).unlink()
    publish(path, lambda dst: dst.write_text(json.dumps(value)))


@pytest.mark.parametrize("kind", sorted(prefix.STOP_KINDS))
def test_receiptless_stops_admit_c2_and_fault_twins(tmp_path: Path, kind: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "certificate.json"
    summary = build(fx, cert, kind)
    assert summary["receipt_less"] is True
    assert summary["post_hoc_release_verified"] is False
    assert summary["last_verified_poll"] == 1
    result = r.completion_packet(
        fx["manifest"],
        None,
        fx["exports"],
        fx["control"],
        completion_index=2,
        job_id="train01-c2",
        output_dir=tmp_path / "c2/output",
        original_manifest_sha256=digest(fx["output"] / "manifest.json"),
        output_root=fx["output"],
        recovery_evidence_path=cert,
    )
    assert result.latest_iteration == 96
    assert result.manifest["completion_attempt"]["admission"]["receipt_less"] is True
    assert not (fx["output"] / "receipt.json").exists()
    # No suffix bytes, later receipt, later corrupt poll or unfinished publication
    # can support admission. They are outside this explicitly labelled evidence cut.
    with fx["log"].open("a") as stream:
        stream.write("NovelDiagnosticException('after cutoff; untrusted')\n")
    (fx["output"] / "poll-000002.json").write_text("corrupt after cutoff")
    (fx["output"] / "receipt.json").write_text("untrusted after cutoff")
    (fx["exports"] / "progress.json.partial").write_text("untrusted after cutoff")
    assert verify(fx, cert)["after_last_verified_export_trusted"] is False
    # A retained prefix fault refuses even after rebinding every affected hash.
    poll = fx["output"] / "poll-000000.json"
    value = json.loads(poll.read_text())
    value["job_log_tail"] = "NovelDiagnosticException('before cutoff')"
    poll.write_text(json.dumps(value))
    certificate = json.loads(cert.read_text())
    certificate["retained_files"][poll.name].update(size=poll.stat().st_size, sha256=digest(poll))
    rewrite(cert, certificate)
    refuses(
        lambda: verify(fx, cert),
        "prior evidence records explicit scientific/non-mechanical fault "
        "in retained_poll:0.job_log_tail",
    )


@pytest.mark.parametrize(
    "fault,message",
    [
        ("hash", "retained_prefix:poll-000000.json size or SHA-256 differs from bound descriptor"),
        ("gap", "retained-prefix file set differs from verified snapshots and poll prefix"),
        ("foreign", "retained-prefix certificate identity differs from frozen source"),
        ("unconfirmed", "retained-prefix recovery requires an operator-confirmed eligible stop"),
        ("cause", "retained-prefix recovery requires an operator-confirmed eligible stop"),
        ("cutoff", "retained-prefix supervisor log hash or boundary differs"),
        ("inventory", "retained-prefix cutoff does not bind all compact snapshot exports"),
        (
            "log-fault",
            "prior evidence records explicit scientific/non-mechanical fault in supervisor_log:1",
        ),
    ],
)
def test_receiptless_exact_message_fault_twins(tmp_path: Path, fault: str, message: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "certificate.json"
    build(fx, cert)
    verify(fx, cert)
    value = json.loads(cert.read_text())
    if fault == "hash":
        path = fx["output"] / "poll-000000.json"
        path.write_text(path.read_text() + " ")
        message += f": {path}"
    elif fault == "gap":
        del value["retained_files"]["poll-000000.json"]
    elif fault == "foreign":
        value["identities"]["generation"] = "foreign"
    elif fault == "unconfirmed":
        value["operator_confirmed_stopped"] = False
    elif fault == "cause":
        value["stop_kind"] = "scientific_failure"
    elif fault == "cutoff":
        value["supervisor_log_prefix"]["size"] -= 1
    elif fault == "inventory":
        path = fx["output"] / "poll-000001.json"
        poll = json.loads(path.read_text())
        poll["exports"] = []
        path.write_text(json.dumps(poll))
        value["retained_files"][path.name].update(size=path.stat().st_size, sha256=digest(path))
    elif fault == "log-fault":
        fx["log"].write_text("NovelDiagnosticException('x')\n" + json.dumps(fx["event"]) + "\n")
        value["supervisor_log_prefix"].update(
            size=fx["log"].stat().st_size, sha256=digest(fx["log"])
        )
    rewrite(cert, value)
    refuses(lambda: verify(fx, cert), message)


def test_certificate_capture_is_create_only_and_requires_complete_compact_exports(
    tmp_path: Path,
) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "certificate.json"
    build(fx, cert)
    before = cert.read_bytes()
    with pytest.raises(FileExistsError):
        build(fx, cert)
    assert cert.read_bytes() == before
    path = fx["exports"] / "iteration-0096.pt.xz.complete"
    path.unlink()
    refuses(lambda: verify(fx, cert), "no complete full-trajectory snapshot triples")


def test_receiptless_rejects_older_cut_and_rehashed_snapshot_fault(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "certificate.json"
    build(fx, cert)
    # Once the same exported snapshot is already verified in an earlier poll,
    # later history cannot be laundered into a different cutoff.
    poll = fx["output"] / "poll-000000.json"
    poll.write_text(json.dumps(fx["poll"]))
    value = json.loads(cert.read_text())
    value["retained_files"][poll.name].update(size=poll.stat().st_size, sha256=digest(poll))
    rewrite(cert, value)
    refuses(lambda: verify(fx, cert), "retained-prefix cutoff extends past last verified export")


def test_c1_cannot_use_receiptless_c2_authority(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "certificate.json"
    build(fx, cert)
    refuses(
        lambda: r.completion_packet(
            fx["manifest"],
            None,
            fx["exports"],
            fx["control"],
            completion_index=1,
            job_id="new-c1",
            output_dir=tmp_path / "new-c1",
            output_root=fx["output"],
            original_manifest_sha256=digest(fx["output"] / "manifest.json"),
            recovery_evidence_path=cert,
        ),
        "retained-export-prefix admission is restricted to c2",
    )


@pytest.mark.parametrize("fault", ["missing", "label", "index", "drift"])
def test_packet_preflight_reopens_receiptless_c2_proof_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import shutil
    import subprocess

    from research.sandbox_a_resident import training_preflight as preflight

    fx = fixture(tmp_path)
    packet = tmp_path / "c2-packet"
    packet.mkdir()
    prior = tmp_path / "packet-train01-15-c1-i0064"
    prior.mkdir()
    (prior / "control.json").write_text(json.dumps(fx["control"]))
    (prior / "dispatch-manifest.json").write_text(json.dumps(fx["manifest"]))
    # Bind the real frozen-source-manifest digest for the preflight boundary.
    cert = tmp_path / "certificate.json"
    prefix.build_prefix_recovery_certificate(
        fx["output"],
        fx["control"],
        fx["log"],
        cert,
        stop_kind="host_loss",
        operator_confirmed_stopped=True,
        source_packet_id=prior.name,
        source_packet_manifest_sha256=digest(prior / "dispatch-manifest.json"),
    )
    recovery = packet / "recovery-evidence.json"
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(Path(str(cert) + suffix), Path(str(recovery) + suffix))
    summary = r.verify_recovery_evidence(
        recovery,
        fx["manifest"],
        digest(fx["output"] / "manifest.json"),
        output_root=fx["output"],
        source_control=fx["control"],
        source_packet_id=prior.name,
        source_packet_manifest_sha256=digest(prior / "dispatch-manifest.json"),
    )
    seal = {
        "schema": "sandbox-a-train01-continuation-seal/v1",
        "prior_stop_evidence_kind": "receiptless_retained_export_prefix",
        "prior_packet": prior.name,
        "prior_packet_path": str(prior),
        "prior_recovery_evidence_schema": prefix.SCHEMA,
        "prior_receipt_present": None,
        "prior_receipt_sha256": None,
        "prior_recovery_evidence_sha256": digest(recovery),
        "prior_executed_manifest_sha256": digest(fx["output"] / "manifest.json"),
        "prior_job_id": fx["manifest"]["job_id"],
        "completion_attempt_index": 2,
        "prior_release_proof": summary,
    }
    (packet / "SEAL.json").write_text(json.dumps(seal))
    broker = {
        "completion_attempt": {
            "index": 2,
            "prior_job_id": fx["manifest"]["job_id"],
            "admission": summary,
        }
    }

    # Only Git transport is stubbed. The entire hash/poll/log/snapshot verifier runs.
    def git(path: Path, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(arguments, 1, "", "")

    monkeypatch.setattr(preflight, "_packet_git", git)
    monkeypatch.setattr(preflight, "verify_committed_manifest", lambda path: None)
    preflight._validate_continuation_stop(packet, broker)
    if fault == "missing":
        recovery.unlink()
        message = "SEAL.json requires recovery-evidence.json; admission proof is missing"
    elif fault == "label":
        broker = copy.deepcopy(broker)
        broker["completion_attempt"]["admission"]["post_hoc_release_verified"] = True
        message = "receipt-less c2 packet labels differ from verified retained prefix"
    elif fault == "index":
        broker["completion_attempt"]["index"] = 1
        message = "receipt-less c2 packet labels differ from verified retained prefix"
    else:
        poll = fx["output"] / "poll-000000.json"
        poll.write_text(poll.read_text() + " ")
        message = (
            f"retained_prefix:{poll.name} size or SHA-256 differs from bound descriptor: {poll}"
        )
    with pytest.raises(ValueError) as caught:
        preflight._validate_continuation_stop(packet, broker)
    assert str(caught.value) == message


def test_decoded_raw_copy_cannot_bypass_compact_header_binding(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_compact import decode_stream

    fx = fixture(tmp_path)
    info_path = fx["exports"] / "iteration-0096.json"
    info = json.loads(info_path.read_text())
    original = info["snapshot"]

    def decode(dst: Path) -> None:
        with dst.open("wb") as stream:
            decode_stream(
                fx["exports"] / "iteration-0096.pt.xz",
                stream.write,
                expected_size=original["size"],
                expected_sha256=original["sha256"],
            )

    publish(fx["exports"] / "iteration-0096.pt", decode)
    build(fx, tmp_path / "healthy.json")
    # Rebind a raw-compatible but non-compact header and its poll inventory;
    # admission must still demand the original compact-at-source contract.
    del info["compact_snapshot"]
    rewrite(info_path, info)
    poll_path = fx["output"] / "poll-000001.json"
    poll = json.loads(poll_path.read_text())
    row = next(e for e in poll["exports"] if e["path"].endswith("iteration-0096.json"))
    row.update(size=info_path.stat().st_size, sha256=digest(info_path))
    poll_path.write_text(json.dumps(poll))
    refuses(
        lambda: build(fx, tmp_path / "bad.json"),
        "retained-prefix snapshot lacks compact-at-source metadata",
    )
    assert not (tmp_path / "bad.json").exists()
