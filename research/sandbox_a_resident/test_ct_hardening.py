"""CT4 adversarial regressions for retained diagnostics and continuation lineage."""

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_resume, ct_run
from research.sandbox_a_resident.ct_task import CTControl


@pytest.mark.parametrize(
    "name",
    [
        "other.json",
        "other.json.partial",
        "other.json.tmp",
        ".state.json.tmp.9",
        "events.jsonl",
        "events.jsonl.partial",
        "events.jsonl.tmp.9",
    ],
)
@pytest.mark.parametrize(
    "record",
    [
        {"status": "failed", "returncode": 7},
        {"returncode": 124},
        {"status": "completed", "returncode": 0},
        {"job_id": "foreign"},
        {"result": {"status": "failed"}},
        {"cleanup_error": "stop failed"},
        {"error_message": "opaque failure"},
    ],
)
def test_ct4_json_variants_refuse(tmp_path: Path, name: str, record: dict) -> None:
    path = tmp_path / name
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(
            tmp_path, expected_job_id="ct0", expected_generation="g0"
        )


@pytest.mark.parametrize("name", ["receipt.json", ".receipt.json.tmp", "receipt.json.partial"])
@pytest.mark.parametrize(
    "extra",
    [
        [],
        None,
        {"error": "opaque failure"},
        {"failure_class": "CHILD_NONZERO_EXIT"},
        {"returncode": 7},
        {"cleanup_error": "vm still active"},
        {"error": "schema mismatch"},
    ],
)
def test_ct4_unselected_receipts_refuse(tmp_path: Path, name: str, extra: object) -> None:
    record = extra
    if isinstance(extra, dict):
        record = {"status": "INCOMPLETE", "stop_verified": True, **extra}
    (tmp_path / name).write_text(json.dumps(record))
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize(
    "name,data,match",
    [
        ("job.log.gz", b"compressed diagnostic", "archive"),
        ("job.log", b"x" * 9000 + b"\x00opaque", "diagnostic"),
        ("job.log", b"FloatingPointError: non-finite \xe2\x80", "scientific"),
        ("other.json.partial", b'{"error":"FloatingPointError: non-finite \xe2\x80', "scientific"),
        ("job.log", b"x" * 9000 + "FloatingPointError".encode("utf-16le"), "scientific"),
    ],
)
def test_ct4_binary_and_truncated_diagnostics_refuse(
    tmp_path: Path, name: str, data: bytes, match: str
) -> None:
    (tmp_path / name).write_bytes(data)
    with pytest.raises(ct_resume.CTResumeRefused, match=match):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_ct4_ancestor_symlink_refuses(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    (actual / "nested").mkdir(parents=True)
    link = tmp_path / "alias"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ct_resume.CTResumeRefused, match="symlink"):
        ct_resume.retained_inventory(link / "nested")
    with pytest.raises(ct_resume.CTResumeRefused, match="symlink"):
        ct_resume._check_attempt_base_fault_evidence(link / "nested")


def test_ct4_fault_uses_terminal_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctrl = CTControl(
        source_commit="a" * 40,
        tiny=True,
        max_stage_export_bytes=16 * 1024 * 1024,
        max_total_export_bytes=16 * 1024 * 1024,
    )
    output = tmp_path / "out"

    def explode(_device: object) -> None:
        from research.sandbox_a_resident.ct_units import recursive_export_bytes

        quota = ctrl.max_total_export_bytes - ct_run._summary_tail_reserve_bytes(ctrl)
        (output / "prior-stage.pt").write_bytes(b"x" * (quota - recursive_export_bytes(output)))
        raise FloatingPointError("finite check failed")

    monkeypatch.setattr(ct_run, "forced_state_determinism_suite", explode)
    with pytest.raises(FloatingPointError):
        ct_run.run_ct_experiment(
            ctrl,
            b8_checkpoint=tmp_path / "unused",
            frp3_checkpoint=tmp_path / "unused",
            champion_checkpoint=tmp_path / "unused",
            output=output,
            device="cpu",
        )
    record = json.loads((output / "fault.json").read_text())
    assert (output / "fault.json.complete").is_file()
    assert record["diagnostic_text_truncated"] is False


def test_ct4_one_level_parent_scanned(tmp_path: Path) -> None:
    exports = tmp_path / "ct"
    exports.mkdir()
    (tmp_path / "state.json").write_text('{"status":"failed","returncode":7}')
    with pytest.raises(ct_resume.CTResumeRefused, match="unbound retained evidence scan root"):
        ct_resume._check_single_root_fault_evidence(exports)

    @ct_resume._evidence_session
    def scan_bound() -> None:
        ct_resume._EVIDENCE_CACHE.get()["bound_output_dirs"] = {str(tmp_path)}
        ct_resume._check_single_root_fault_evidence(exports)

    with pytest.raises(ct_resume.CTResumeRefused):
        scan_bound()


def _continuation_fixture(tmp_path: Path) -> tuple[Path, dict, dict, Path, dict]:
    from research.sandbox_a_resident.ct_units import allocation, publish_stage_snapshot
    from research.sandbox_a_resident.test_ct_runner_and_resume import (
        _clean_supervisor_log,
        _save_tiny_checkpoints,
        _write_tiny_initial_packet,
    )

    ctrl = CTControl(source_commit="a" * 40, worlds=6, collection_ticks=12, tiny=True)
    initial, successor = tmp_path / "p0", tmp_path / "c1"
    output = tmp_path / "a0"
    # Canonical output root has 0 stages; payloads are in a separately retained export root.
    exports = output / "external-exports"
    _write_tiny_initial_packet(initial, ctrl, _save_tiny_checkpoints(tmp_path), output)
    (output / "exports/ct").mkdir(parents=True)
    (output / "state.json").write_text(
        json.dumps({"job_id": "tiny-ct-00", "generation": "g0", "status": "running"})
    )
    publish_stage_snapshot(
        exports,
        1,
        lambda p: p.write_bytes(b"verified stage bytes"),
        metadata={"allocation": allocation(ctrl.to_dict()), "source_commit": ctrl.source_commit},
    )
    receipt = {
        "job_id": "tiny-ct-00",
        "generation": "g0",
        "commit": ctrl.source_commit,
        "manifest_sha256": ct_resume.digest(initial / "dispatch-manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "whole-session wall cap reached",
        "cleanup_error": None,
        "stop_verified": True,
    }
    kwargs = dict(
        prior_packet_dir=initial,
        terminal_receipt=receipt,
        export_root=exports,
        completion_index=1,
        job_id="ct-c1",
        output_dir=tmp_path / "a1",
        use_compact_xz_for_stages=None,
        validate_registry_inputs=False,
        supervisor_log_path=_clean_supervisor_log(tmp_path),
    )
    ct_resume.write_continuation_packet(successor, **kwargs)
    broker = json.loads((successor / "dispatch-manifest.json").read_text())
    return successor, broker, ctrl.to_dict(), exports, kwargs


@pytest.mark.parametrize(
    "attack",
    [
        "shadow",
        "missing_root_seal",
        "missing_log",
        "bad_log_schema",
        "relative_log",
        "ancestor_log_link",
        "log_fault",
        "log_failed_json",
        "omit_canonical_export",
    ],
)
def test_ct4_lineage_refuses(tmp_path: Path, attack: str) -> None:
    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    packet, broker, control, exports, kwargs = _continuation_fixture(tmp_path)
    validate_uploaded_ledger(packet, broker, control, require_committed=False)
    proof_path = packet / "provenance.json"
    proof = json.loads(proof_path.read_text())
    if attack == "shadow":
        shadow = kwargs["prior_packet_dir"] / "shadow.json"
        shadow.write_text((kwargs["prior_packet_dir"] / "dispatch-manifest.json").read_text())
        broker["completion_attempt"]["prior_manifest_path"] = str(shadow)
        proof["prior_manifest_path"] = str(shadow)
    elif attack == "missing_root_seal":
        (kwargs["prior_packet_dir"] / "SEAL.json").unlink()
    elif attack == "missing_log":
        proof["supervisor_log"] = None
    elif attack == "bad_log_schema":
        proof["supervisor_log"] = "ignored"
    elif attack == "relative_log":
        proof["supervisor_log"]["path"] = "supervisor.log"
    elif attack == "ancestor_log_link":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        proof["supervisor_log"]["path"] = str(alias / "supervisor.log")
    elif attack in {"log_fault", "log_failed_json"}:
        log_path = Path(proof["supervisor_log"]["path"])
        log_path.write_text(
            "FloatingPointError: non-finite"
            if attack == "log_fault"
            else '{"status":"failed","returncode":7}\n'
        )
        proof["supervisor_log"].update(
            size=log_path.stat().st_size, sha256=ct_resume.digest(log_path)
        )
    else:
        broker["completion_attempt"]["prior_export_roots"] = [str(exports)]
        proof["retained"].pop(str(tmp_path / "a0/exports/ct"))
    proof_path.write_text(json.dumps(proof))
    broker["completion_attempt"]["provenance_sha256"] = ct_resume.digest(proof_path)
    (packet / "dispatch-manifest.json").write_text(json.dumps(broker))
    seal = json.loads((packet / "SEAL.json").read_text())
    seal.update(
        manifest_sha256=ct_resume.digest(packet / "dispatch-manifest.json"),
        provenance_sha256=ct_resume.digest(proof_path),
    )
    (packet / "SEAL.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        validate_uploaded_ledger(packet, broker, control, require_committed=False)


def test_ct4_inherited_zero_stage_output_c2_and_scan_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger
    from research.sandbox_a_resident.ct_units import allocation, publish_stage_snapshot

    packet, broker, control, exports, kwargs = _continuation_fixture(tmp_path)
    # A1 has its own telemetry; A0's empty canonical export and host state remain inherited.
    new_exports = tmp_path / "a1/exports/ct"
    publish_stage_snapshot(
        new_exports,
        2,
        lambda p: p.write_bytes(b"verified second stage"),
        metadata={"allocation": allocation(control), "source_commit": control["source_commit"]},
    )
    (tmp_path / "a1/state.json").write_text(
        json.dumps({"job_id": "ct-c1", "generation": "g1", "status": "running"})
    )
    receipt = {
        **kwargs["terminal_receipt"],
        "job_id": "ct-c1",
        "generation": "g1",
        "manifest_sha256": ct_resume.digest(packet / "dispatch-manifest.json"),
    }
    next_packet = tmp_path / "c2"
    ct_resume.write_continuation_packet(
        next_packet,
        prior_packet_dir=packet,
        terminal_receipt=receipt,
        export_root=[exports, new_exports],
        completion_index=2,
        job_id="ct-c2",
        output_dir=tmp_path / "a2",
        use_compact_xz_for_stages=None,
        validate_registry_inputs=False,
        supervisor_log_path=kwargs["supervisor_log_path"],
    )
    new_broker = json.loads((next_packet / "dispatch-manifest.json").read_text())
    calls = []
    original = ct_resume.digest

    def counted(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(ct_resume, "digest", counted)
    validate_uploaded_ledger(next_packet, new_broker, control, require_committed=False)
    assert calls.count(exports / "stage-0001.pt") == 1
    # A second admission must reopen changed evidence, never reuse a global cache.
    (exports / "extra.json").write_text('{"error_message":"opaque fault"}')
    with pytest.raises(ValueError):
        validate_uploaded_ledger(next_packet, new_broker, control, require_committed=False)


def test_ct4_no_stage_c1_refuses_before_missing_exports_can_be_omitted(tmp_path: Path) -> None:
    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    packet, broker, control, exports, kwargs = _continuation_fixture(tmp_path)
    ledger_path = packet / "verified-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["units"] = []
    ledger_path.write_text(json.dumps(ledger))
    broker["completion_attempt"]["verified_stages"] = []
    broker["completion_attempt"]["ledger_sha256"] = ct_resume.digest(ledger_path)
    next(u for u in broker["inputs"] if u["dest"] == "verified-ledger.json")["sha256"] = (
        ct_resume.digest(ledger_path)
    )
    (packet / "dispatch-manifest.json").write_text(json.dumps(broker))
    seal = json.loads((packet / "SEAL.json").read_text())
    seal["manifest_sha256"] = ct_resume.digest(packet / "dispatch-manifest.json")
    (packet / "SEAL.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="at least one verified stage"):
        validate_uploaded_ledger(packet, broker, control, require_committed=False)
    # The writer also refuses a true zero-stage stop, preserving the existing semantics.
    empty = Path(kwargs["export_root"]).parent / "empty-export"
    empty.mkdir()
    with pytest.raises(ValueError, match="no complete verified stage"):
        ct_resume.write_continuation_packet(
            tmp_path / "empty-c1", **{**kwargs, "export_root": empty}
        )


@pytest.mark.parametrize(
    "name", [".summary.json.tmp", ".fault.json.tmp", "faulthandler-stacks.log.gz"]
)
def test_ct4_hidden_fault_names_refuse(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("{}")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_ct4_supervisor_structured_records_and_qualified_child_receipt(tmp_path: Path) -> None:
    log = tmp_path / "supervisor.log"
    log.write_text(
        json.dumps(
            {
                "job": "ct0",
                "generation": "g0",
                "status": "running",
                "terminal": False,
                "exports": [{"path": "canary.json", "size": 1, "sha256": "a" * 64}],
            }
        )
        + "\n"
    )
    ct_resume._scan_retained_text(
        log, output_root=tmp_path, expected_job_id="ct0", expected_generation="g0"
    )
    log.write_text('{"job":"foreign","generation":"g0","status":"running"}\n')
    with pytest.raises(ValueError, match="foreign"):
        ct_resume._scan_retained_text(
            log, output_root=tmp_path, expected_job_id="ct0", expected_generation="g0"
        )
    record = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "job_id": "ct0",
        "generation": "g0",
        "error": (
            "remote child exited 138: STALL_WATCHDOG_DIAGNOSTICS "
            "sealed_dir=/content/out/ct/stall-watchdog"
        ),
        "terminal": {
            "job_id": "ct0",
            "generation": "g0",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(record))
    ct_resume._scan_retained_text(
        receipt, output_root=tmp_path, expected_job_id="ct0", expected_generation="g0"
    )


@pytest.mark.parametrize("name", ["supervisor.log.tmp", ".supervisor.log", "state.json.partial"])
def test_ct4_clean_incomplete_publications_refuse(tmp_path: Path, name: str) -> None:
    record = {"job_id": "ct0", "generation": "g0", "status": "running"}
    (tmp_path / name).write_text(json.dumps(record) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused, match="incomplete"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize("reason", ["timeout", "lease_expired", "release_credential_expiring"])
@pytest.mark.parametrize(
    "line",
    [None, "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/session/output/ct/stall-watchdog"],
)
def test_ct4_real_qualified_child_receipt_is_admitted(
    tmp_path: Path, reason: str, line: str | None
) -> None:
    from research.colab_cli.failures import terminal_failure

    terminal = {
        "job_id": "ct0",
        "generation": "g0",
        "status": "failed",
        "reason": reason,
        "returncode": 138,
    }
    if line is not None:
        terminal["last_traceback_line"] = line
    failure = terminal_failure(terminal, "g0")
    assert failure is not None
    record = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": "CHILD_NONZERO_EXIT",
        "job_id": "ct0",
        "generation": "g0",
        "terminal": terminal,
        "error": repr(failure),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(record))
    ct_resume._scan_retained_text(
        path, output_root=tmp_path, expected_job_id="ct0", expected_generation="g0"
    )
    record["error"] += "; opaque failure"
    path.write_text(json.dumps(record))
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_retained_text(
            path, output_root=tmp_path, expected_job_id="ct0", expected_generation="g0"
        )


@pytest.mark.parametrize("suffix", ["", "/"])
def test_scan_root_binding_uses_normpath_and_never_skips_parent(
    tmp_path: Path, suffix: str
) -> None:
    exports = tmp_path / "exports/ct"
    exports.mkdir(parents=True)
    (tmp_path / "state.json").write_text('{"status":"failed","returncode":7}')

    @ct_resume._evidence_session
    def scan() -> None:
        ct_resume._EVIDENCE_CACHE.get()["bound_output_dirs"] = {str(tmp_path) + suffix}
        ct_resume._check_single_root_fault_evidence(exports)

    with pytest.raises(ct_resume.CTResumeRefused, match="non-mechanical|lacks matching job_id"):
        scan()
    with pytest.raises(ct_resume.CTResumeRefused, match="unbound retained evidence scan root"):
        ct_resume._check_single_root_fault_evidence(exports)


@pytest.mark.parametrize("failure", ["missing", "oversized", "sha", "retired"])
def test_lineage_refuses_unverified_or_retired_parent(tmp_path: Path, failure: str) -> None:
    import hashlib

    from research.sandbox_a_resident import ct_preflight

    packet, manifest, control, _, _ = _continuation_fixture(tmp_path)
    comp = manifest["completion_attempt"]
    parent = Path(comp["prior_manifest_path"])
    if failure == "missing":
        parent.unlink()
        match = "unreadable lineage manifest"
    elif failure == "oversized":
        parent.write_bytes(b" " * (1024 * 1024 + 1))
        match = "oversized lineage manifest"
    elif failure == "sha":
        comp["prior_manifest_sha256"] = "0" * 64
        match = "prior manifest SHA-256 mismatch"
    else:
        row = json.loads(parent.read_text())
        row["job_id"] = "sandbox-ct-cf-20261028-05"
        parent.write_text(json.dumps(row))
        comp["prior_manifest_sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
        match = "packet superseded"
    (packet / "dispatch-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        ct_preflight._check_lineage_graph(packet, manifest, control)


def test_retired_root_writer_refuses_before_creating_packet(tmp_path: Path) -> None:
    packet, manifest, _, _, _ = _continuation_fixture(tmp_path)
    parent = Path(manifest["completion_attempt"]["prior_manifest_path"])
    root = json.loads(parent.read_text())
    root["job_id"] = "sandbox-ct-cf-20261028-05"
    parent.write_text(json.dumps(root))
    target = tmp_path / "retired-successor"
    with pytest.raises(ValueError, match="packet superseded"):
        ct_resume.write_continuation_packet(
            target,
            prior_packet_dir=parent.parent,
            terminal_receipt={},
            export_root=tmp_path,
            completion_index=1,
            job_id="fresh-id",
            output_dir=tmp_path / "out",
            supervisor_log_path=parent,
        )
    assert not target.exists()


@pytest.mark.parametrize("row", [{}, {"packet_dir": "old"}, {"job_ids": "old"}, None])
def test_malformed_supersession_rows_refuse_cleanly(tmp_path: Path, row: object) -> None:
    from tools.ops.packet_supersession import require_current_packet

    registry = tmp_path / "tools/ops/packet-supersessions.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"schema": "snake-packet-supersessions/v1", "superseded": [row]})
    )
    with pytest.raises(ValueError, match="malformed packet supersession registry row"):
        require_current_packet({}, tmp_path / "manifest.json", root=tmp_path, original_bytes=b"{}")


def test_supersession_uses_original_loaded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.ops.packet_supersession import require_current_packet

    registry = tmp_path / "tools/ops/packet-supersessions.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema": "snake-packet-supersessions/v1",
                "superseded": [
                    {
                        "packet_dir": "old",
                        "successor": "new",
                        "reason": "retired",
                        "job_ids": [],
                        "manifest_sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("manifest reread"))
    require_current_packet({}, tmp_path / "missing.json", root=tmp_path, original_bytes=b"{}")


def test_missing_scan_root_refuses(tmp_path: Path) -> None:
    with pytest.raises(ct_resume.CTResumeRefused, match="missing retained evidence scan root"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path / "missing")


def test_exact_generation_bound_signal_footer_without_caller_identity(tmp_path: Path) -> None:
    generation = "a" * 32
    footer = (
        f"STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/snake-cli/{generation}"
        "/output/ct/stall-watchdog"
    )
    terminal = {
        "job_id": "ct0",
        "generation": generation,
        "status": "failed",
        "reason": "timeout",
        "returncode": 138,
        "last_traceback_line": footer,
        "job_log_tail": footer + "\n",
    }
    record = {
        "job_id": "ct0",
        "generation": generation,
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "cleanup_error": None,
        "terminal": terminal,
        "error": f"ClassifiedFailure('remote child exited 138: {footer}')",
    }
    (tmp_path / "receipt.json").write_text(json.dumps(record))
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    terminal["last_traceback_line"] += "; LookupError: concealed science"
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize("failure_class", ["SUPERVISOR", "CHILD_NONZERO_EXIT"])
def test_structured_mechanical_reason_never_overrides_non_signal_exit(failure_class: str) -> None:
    record = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "failure_class": failure_class,
        "terminal": {"reason": "timeout", "status": "failed", "returncode": 7},
    }
    with pytest.raises(ct_resume.CTResumeRefused, match="non-signal exit"):
        ct_resume._check_retained_receipt(record, label="receipt")
    with pytest.raises(ct_resume.CTResumeRefused, match="non-signal exit"):
        ct_resume._scan_json_faults(
            record["terminal"], label="poll.state", supervisor_transport=True
        )
