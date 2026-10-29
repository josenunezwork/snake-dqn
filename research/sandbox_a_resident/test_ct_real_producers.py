"""Packet 06 admission against committed, unedited qualified-producer goldens."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_preflight, ct_resume
from research.sandbox_a_resident.test_ct_hardening import _continuation_fixture
from research.sandbox_a_resident.test_ct_runner_and_resume import _rebind_forged_packet
from tools.sandbox import preflight as sandbox_preflight

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests/fixtures/ct-supervisor-06-exports"
MECHANICAL = tuple(
    path.name
    for path in sorted(GOLDENS.iterdir() if GOLDENS.is_dir() else [])
    if (path / "output/receipt.json").is_file()
    and path.name not in {"clean-completion", "child-nonmechanical-exit"}
)


assert len(MECHANICAL) == 13, "real-producer goldens missing or incomplete"


@pytest.fixture(autouse=True)
def historical_fixture_paths_only_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ct_resume._observed_signal_line
    observed = {
        "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/private/tmp/"
        f"astra-impl-ct6-producers-{name}/child-ct-entrypoint/"
        "runtime-producer/output/ct/stall-watchdog"
        for name in ("entrypoint-fixed", "entrypoint-main")
    }
    monkeypatch.setattr(
        ct_resume,
        "_observed_signal_line",
        lambda value, generation=None: original(value, generation)
        or isinstance(value, str)
        and value in observed,
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_binding(output: Path, control: dict) -> dict:
    archive = output / "staging/source.tar.gz"
    return {
        "source_tree": _read(output / "staging/source-plan.json")["tree"],
        "source_commit": control["source_commit"],
        "source_files": {
            k: v["sha256"] for k, v in _read(output / "staging/source-plan.json")["files"].items()
        },
        "source_archive": {
            "size": archive.stat().st_size,
            "sha256": ct_resume.digest(archive),
            "plan_sha256": ct_resume.digest(output / "staging/source-plan.json"),
        },
    }


@pytest.mark.parametrize("scenario", MECHANICAL)
@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("corpus", ["ct-supervisor-06-exports", "ct-supervisor-06-main"])
def test_real_supervisor_mechanical_formats_and_compound_twins(
    tmp_path: Path,
    scenario: str,
    fault: bool,
    corpus: str,
) -> None:
    goldens = ROOT / "tests/fixtures" / corpus
    capture = goldens / scenario
    shutil.copytree(capture / "output", tmp_path / "output")
    output = tmp_path / "output"
    log = tmp_path / "supervisor.log"
    shutil.copy2(capture / "supervisor.log", log)
    receipt, manifest = _read(output / "receipt.json"), _read(output / "manifest.json")
    # An offline path relocation does not rewrite any golden fixture. The
    # original producer manifest/receipt binding remains byte-for-byte intact.
    control = _read(goldens / "control.json")

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = _source_binding(output, control)
        ct_resume._EVIDENCE_CACHE.get().setdefault("bound_output_dirs", set()).add(str(output))
        ct_resume.verify_mechanical_stop(
            receipt,
            manifest,
            ct_resume.digest(output / "manifest.json"),
            export_root=output,
            supervisor_log_path=log,
        )

    if fault:
        receipt["error"] += "; FloatingPointError: non-finite appended fault"
        with pytest.raises(
            ct_resume.CTResumeRefused,
            match=(
                "not a qualified mechanical trigger"
                if receipt["failure_class"] == "RPC_EXEC_POLL_TIMEOUT"
                else "scientific/non-mechanical"
            ),
        ):
            admit()
    elif scenario in {
        "lease-renewal-timeout",
        "rpc-exec-poll-timeout",
        "rpc-busy-timeout",
        "unit-runaway-guard",
    }:
        with pytest.raises(
            ct_resume.CTResumeRefused,
            match=(
                "not a qualified mechanical trigger"
                if receipt["failure_class"] == "RPC_EXEC_POLL_TIMEOUT"
                else "scientific/non-mechanical exception"
            ),
        ):
            admit()
    else:
        admit()


@pytest.mark.parametrize("scenario", ["clean-completion", "child-nonmechanical-exit"])
def test_real_completed_or_unexplained_child_cannot_continue(scenario: str) -> None:
    output = GOLDENS / scenario / "output"

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = _source_binding(
            output, _read(GOLDENS / "control.json")
        )
        ct_resume.verify_mechanical_stop(
            _read(output / "receipt.json"),
            _read(output / "manifest.json"),
            ct_resume.digest(output / "manifest.json"),
            export_root=output,
        )

    match = (
        "not a qualified mechanical trigger"
        if scenario == "clean-completion"
        else "unexplained retained receipt error"
    )
    with pytest.raises(ct_resume.CTResumeRefused, match=match):
        admit()


@pytest.mark.parametrize(
    "record",
    [
        {"detail": "schema mismatch in runtime"},
        {"nested": [{"detail": "corrupt input"}]},
        {"nested": {"detail": "unexpected exception in runtime"}},
    ],
)
def test_log_denylist_applies_to_all_json_leaves(tmp_path: Path, record: dict) -> None:
    log = tmp_path / "supervisor.log"
    log.write_text(json.dumps(record) + "\n")
    with pytest.raises(ct_resume.CTResumeRefused, match="supervisor"):
        ct_resume._scan_supervisor_log(log, job_id=None, generation=None)


def test_nested_qualified_mechanical_error_survives(tmp_path: Path) -> None:
    log = tmp_path / "supervisor.log"
    log.write_text(json.dumps({"nested": [{"error": "Refused('whole-session wall cap reached')"}]}))
    ct_resume._scan_supervisor_log(log, job_id=None, generation=None)


@pytest.mark.parametrize("mutation", ["append", "different-path", "different-cause"])
def test_cli_footer_is_an_exact_bound_cause(tmp_path: Path, mutation: str) -> None:
    capture = ROOT / "tests/fixtures/ct-supervisor-06-main/lease-renewal-timeout"
    receipt = _read(capture / "output/receipt.json")
    manifest = _read(capture / "output/manifest.json")
    text = (capture / "supervisor.log").read_text()
    if mutation == "append":
        text = text.rstrip("\n") + "; FloatingPointError appended fault\n"
    elif mutation == "different-path":
        text = text.replace(manifest["output_dir"], str(tmp_path / "foreign"))
    else:
        text = text.replace("fake request deadline exceeded", "a different cause")
    log = tmp_path / "supervisor.log"
    log.write_text(text)
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_supervisor_log(
            log,
            job_id=receipt["job_id"],
            generation=receipt["generation"],
            receipt=receipt,
            receipt_path=Path(manifest["output_dir"]) / "receipt.json",
        )


@pytest.mark.parametrize("name", ["events.json", "events.jsonl", "supervisor.log"])
def test_duplicate_keys_cannot_hide_a_fault(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text('{"error":"FloatingPointError","error":null}\n')
    with pytest.raises(ct_resume.CTResumeRefused, match="duplicate|corrupt"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_jsonl_unicode_separators_stream_without_read_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"detail": "a\u2028b\u2029c\u0085d"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"detail": "next record"})
        + "\n"
    )
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("JSONL buffered"))
    ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_publication_sidecars_share_the_runtime_parser(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text('{"detail":"okay"}')
    Path(str(path) + ".sha256").write_text(ct_resume.digest(path) + "  value.json\n")
    Path(str(path) + ".complete").write_text("complete\n")
    ct_resume._descriptor(path, runtime_path=path.name)
    ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_dotdot_path_is_not_canonical(tmp_path: Path) -> None:
    with pytest.raises(ct_resume.CTResumeRefused, match="absolute"):
        ct_resume._no_symlink_path(tmp_path / "inside/../inside/dispatch-manifest.json")


def test_scan_cache_dies_between_sessions_same_inode_and_size(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    clean = '{"detail":"everything fine"}'
    fault = '{"detail":"non-finite data"}'
    assert len(clean) == len(fault)
    path.write_text(clean)
    before = path.stat()
    scan = ct_resume._evidence_session(ct_resume._check_attempt_base_fault_evidence)
    scan(tmp_path)
    with path.open("r+") as stream:
        stream.write(fault)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_ino == before.st_ino and path.stat().st_size == before.st_size
    with pytest.raises(ct_resume.CTResumeRefused, match="non-finite"):
        scan(tmp_path)


def test_real_watchdog_sidecars_do_not_mask_a_nonempty_stack(tmp_path: Path) -> None:
    capture = (
        ROOT / "tests/fixtures/ct-supervisor-06-entrypoint/child-ct-entrypoint/output/exports/ct"
    )
    exports = tmp_path / "ct-output"
    shutil.copytree(capture, exports)
    ct_resume._check_attempt_base_fault_evidence(exports)
    stack = exports / "stall-watchdog/faulthandler-stacks.log"
    stack.chmod(0o600)
    stack.write_text("watchdog diagnostic stacks\n")
    sha = Path(str(stack) + ".sha256")
    sha.chmod(0o600)
    sha.write_text(ct_resume.digest(stack) + "  faulthandler-stacks.log\n")
    with pytest.raises(ct_resume.CTResumeRefused, match="^non-empty faulthandler stack dump:"):
        ct_resume._check_attempt_base_fault_evidence(exports)


@pytest.mark.parametrize("attack", ["c3", "self-cycle", "two-cycle", "index-gap"])
def test_lineage_graph_refuses_boundedly(tmp_path: Path, attack: str) -> None:
    packet, broker, control, _, _ = _continuation_fixture(tmp_path)
    if attack == "c3":
        broker["completion_attempt"]["index"] = 3
        match = "^continuation lineage exceeds two completion attempts$"
    elif attack == "self-cycle":
        broker["completion_attempt"]["prior_manifest_path"] = str(packet / "dispatch-manifest.json")
        match = "^continuation lineage cycle detected$"
    elif attack == "two-cycle":
        other = tmp_path / "other"
        shutil.copytree(packet, other)
        other_broker = copy.deepcopy(broker)
        other_broker["completion_attempt"]["prior_manifest_path"] = str(
            packet / "dispatch-manifest.json"
        )
        _rebind_forged_packet(other, other_broker)
        broker["completion_attempt"]["prior_manifest_path"] = str(other / "dispatch-manifest.json")
        broker["completion_attempt"]["prior_manifest_sha256"] = ct_resume.digest(
            other / "dispatch-manifest.json"
        )
        match = "^continuation lineage cycle detected$"
    else:
        broker["completion_attempt"]["index"] = 2
        match = r"^continuation index must equal prior index \+ 1$"
    _rebind_forged_packet(packet, broker)
    with pytest.raises(ValueError, match=match):
        ct_preflight.validate_uploaded_ledger(packet, broker, control, require_committed=False)


@pytest.mark.parametrize("attack", [None, "stale-commit", "archive-sha", "archive-size"])
def test_source_archive_uses_sealed_bytes_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str | None,
) -> None:
    output = tmp_path / "output"
    shutil.copytree(GOLDENS / "wall-cap/output", output)
    binding = _source_binding(output, _read(GOLDENS / "control.json"))
    if attack == "stale-commit":
        for path in (output / "manifest.json", output / "staging/source-plan.json"):
            value = _read(path)
            value["commit"] = "1" * 40
            _write(path, value)
    elif attack == "archive-sha":
        binding["source_files"][next(iter(binding["source_files"]))] = "0" * 64
    elif attack == "archive-size":
        plan = _read(output / "staging/source-plan.json")
        plan["fallback_size"] += 1
        _write(output / "staging/source-plan.json", plan)
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("archive invoked Git"))
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_k: pytest.fail("archive invoked Git")
    )

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = binding
        ct_resume._verify_staged_source_archive(output / "staging/source.tar.gz", output)

    if attack:
        with pytest.raises(ct_resume.CTResumeRefused, match="sealed source_commit/files/size/SHA"):
            admit()
    else:
        admit()


@pytest.mark.parametrize("number", range(1, 6))
def test_superseded_packet_refuses_in_preflight_and_dispatch(tmp_path: Path, number: int) -> None:
    from types import SimpleNamespace

    from tools.ops import dispatch
    from tools.ops.packet_supersession import require_current_packet

    packet = tmp_path / f"packet-ct-1028-{number:02d}"
    packet.mkdir()
    manifest = {"job_id": f"sandbox-ct-cf-20261028-{number:02d}"}
    _write(packet / "dispatch-manifest.json", manifest)
    with pytest.raises(ValueError, match="packet superseded"):
        ct_preflight.validate_packet(packet)
    # Shared check is the first manifest check in dispatch, before state/log/account I/O.
    with pytest.raises(ValueError, match="packet superseded"):
        require_current_packet(
            manifest,
            packet / "dispatch-manifest.json",
            root=ROOT,
            original_bytes=(packet / "dispatch-manifest.json").read_bytes(),
        )
    with pytest.raises(ValueError, match="packet superseded"):
        dispatch.dispatch(
            SimpleNamespace(
                name=manifest["job_id"],
                manifest=packet / "dispatch-manifest.json",
                broker=ROOT,
                cwd=ROOT,
                state_dir=tmp_path / "unused-state",
                dispatch_txt=None,
            )
        )
    assert not (tmp_path / "unused-state").exists()


@pytest.mark.parametrize(
    "scenario", ["child-timeout", "wall-cap", "child-ct-entrypoint", "cli-entrypoint"]
)
def test_actual_ct_tree_writes_and_commits_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    """Real transferred C/T tree -> writer -> complete committed packet admission."""
    corpus = (
        "ct-supervisor-06-entrypoint-main"
        if scenario == "cli-entrypoint"
        else (
            "ct-supervisor-06-entrypoint"
            if scenario == "child-ct-entrypoint"
            else "ct-supervisor-06-ct-bound"
        )
    )
    capture = (
        ROOT
        / "tests/fixtures"
        / corpus
        / ("child-ct-entrypoint" if scenario == "cli-entrypoint" else scenario)
    )
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(repo)], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "sparse-checkout",
            "set",
            "--no-cone",
            "/tools/",
            "/research/colab_cli/",
            "/research/sandbox_a_resident/ct_run.py",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "HEAD"], check=True)
    packet, output = repo / "test-packets/p0", tmp_path / "output"
    shutil.copytree(capture / "initial-packet", packet)
    shutil.copytree(capture / "output", output)
    old_manifest = _read(output / "manifest.json")
    # Preserve the historical runtime source pin while exercising today's
    # admission code. ct_run now also retains progress history, so checking
    # today's ct_run bytes against a historical pin is not a valid fixture.
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "restore",
            "--source",
            old_manifest["commit"],
            "--staged",
            "--worktree",
            "research/sandbox_a_resident/ct_run.py",
        ],
        check=True,
    )
    old_packet = str(
        Path(
            next(
                row["path"]
                for row in old_manifest["inputs"]
                if row["dest"] == "verified-ledger.json"
            )
        ).parent
    )
    replacements = {old_manifest["output_dir"]: str(output), old_packet: str(packet)}
    for base in (packet, output):
        for path in base.rglob("*.json"):
            path.chmod(0o600)
            text = path.read_text()
            for old, new in replacements.items():
                text = text.replace(old, new)
            path.write_text(text)
    receipt = _read(output / "receipt.json")
    receipt["manifest_sha256"] = ct_resume.digest(output / "manifest.json")
    _write(output / "receipt.json", receipt)
    (packet / "dispatch-manifest.json").write_bytes((output / "manifest.json").read_bytes())
    control = _read(packet / "control.json")
    # Rebind path relocation and the observed normalized producer manifest;
    # all C/T stage bytes, logs and numerical controls remain the golden bytes.
    sandbox = _read(packet / "sandbox.json")
    sandbox["timeout"] = {"base_seconds": 90, "seconds": 180, "purpose": "runaway-guard"}
    sandbox["outputs"]["paths"] = [str(output)]
    sandbox["outputs"]["broker_manifest"].update(
        path=str(packet / "dispatch-manifest.json"),
        sha256=ct_resume.digest(output / "manifest.json"),
    )
    for row in sandbox["inputs"]:
        if row.get("kind") == "verified-ledger":
            row["path"] = str(packet / "verified-ledger.json")

    def ten_lines(value: dict) -> str:
        return (
            "\n".join(
                ("{" if i == 0 else " ")
                + json.dumps(k)
                + ": "
                + json.dumps(v)
                + ("," if i < 9 else "}")
                for i, (k, v) in enumerate(value.items())
            )
            + "\n"
        )

    (packet / "sandbox.json").write_text(ten_lines(sandbox))
    seal = _read(packet / "SEAL.json")
    seal.update(
        manifest_sha256=ct_resume.digest(packet / "dispatch-manifest.json"),
        sandbox_sha256=ct_resume.digest(packet / "sandbox.json"),
        source_archive=_source_binding(output, control)["source_archive"],
        source_sha256=_source_binding(output, control)["source_files"],
    )
    _write(packet / "SEAL.json", seal)
    log = tmp_path / "supervisor.log"
    log_text = (capture / "supervisor.log").read_text()
    for old, new in replacements.items():
        log_text = log_text.replace(old, new)
    log.write_text(log_text)
    successor = repo / "test-packets/c1"
    ct_resume.write_continuation_packet(
        successor,
        prior_packet_dir=packet,
        terminal_receipt=receipt,
        export_root=output / "exports/ct",
        completion_index=1,
        job_id="ct-real-c1",
        output_dir=tmp_path / "next-output",
        use_compact_xz_for_stages=None,
        supervisor_log_path=log,
    )
    subprocess.run(["git", "-C", str(repo), "add", "--sparse", "test-packets"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=CT fixture",
            "-c",
            "user.email=ct@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "seal relocated real producer continuation",
        ],
        check=True,
    )
    monkeypatch.setattr(sandbox_preflight, "REPO_ROOT", repo)
    # _run's default cwd was captured when the real module was imported. Route
    # its real Git checks to this private clone along with REPO_ROOT.
    original_run = sandbox_preflight._run
    monkeypatch.setattr(sandbox_preflight, "_run", lambda command: original_run(command, cwd=repo))
    monkeypatch.setattr(ct_preflight, "ROOT", repo)
    ct_preflight.validate_packet(successor, require_committed=True, allow_tiny=True)
    with log.open("a") as stream:
        stream.write('{"nested":{"detail":"schema mismatch appended fault"}}\n')
    with pytest.raises(ValueError, match="supervisor|inventory"):
        ct_preflight.validate_packet(successor, require_committed=True, allow_tiny=True)


@pytest.mark.parametrize("scenario", MECHANICAL)
@pytest.mark.parametrize("corpus", ["ct-supervisor-06-exports", "ct-supervisor-06-main"])
@pytest.mark.parametrize(
    "surface",
    [
        "last_error_line",
        "last_traceback_line",
        "poll-transport",
        "job_log_tail",
        "transport_summary",
    ],
)
def test_every_mechanical_capture_fault_twin_on_every_exempt_surface(
    tmp_path: Path,
    scenario: str,
    corpus: str,
    surface: str,
) -> None:
    """Run the actual scanner directly so an earlier receipt refusal cannot mask a gap."""
    output = tmp_path / "output"
    shutil.copytree(ROOT / "tests/fixtures" / corpus / scenario / "output", output)
    receipt = _read(output / "receipt.json")
    fault = "LookupError: injected scientific invariant"
    assert not any(marker in fault.lower() for marker in ct_resume.EXPLICIT_FAULT_EVIDENCE_MARKERS)
    if surface in {"last_error_line", "last_traceback_line"}:
        record = receipt.get("terminal") or {
            "job_id": receipt["job_id"],
            "generation": receipt["generation"],
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        }
        record[surface] = fault
        name = "terminal.json"
    elif surface == "poll-transport":
        record = copy.deepcopy(receipt["cli_calls"][0])
        record["last_error_line"] = fault
        name = "poll-999999.json"
    elif surface == "transport_summary":
        record = {
            "job": receipt["job_id"],
            "elapsed": 1.0,
            "status": "failed",
            "terminal": False,
            "exports": 0,
            "job_log_tail": fault,
        }
        name = "poll-999999.json"
    else:
        record = {"job_log_tail": fault}
        name = "poll-999999.json"
    path = output / name
    _write(path, record)
    with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical|job_log_tail"):
        ct_resume._scan_retained_text(
            path,
            output_root=output,
            expected_job_id=receipt["job_id"],
            expected_generation=receipt["generation"],
        )


@pytest.mark.parametrize("sidecar", ["sha256", "complete"])
@pytest.mark.parametrize("fault", ["LookupError: unexpected science", "opaque appended bytes"])
def test_sidecar_trailing_content_refuses(tmp_path: Path, sidecar: str, fault: str) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}")
    Path(str(path) + ".sha256").write_text(ct_resume.digest(path) + "  value.json\n")
    Path(str(path) + ".complete").write_text("complete\n")
    with Path(str(path) + "." + sidecar).open("a") as stream:
        stream.write(fault)
    with pytest.raises(ct_resume.CTResumeRefused, match="malformed"):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


def test_regenerated_source_archive_metadata_is_not_identity(tmp_path: Path) -> None:
    import gzip
    import hashlib

    output = tmp_path / "output"
    shutil.copytree(GOLDENS / "wall-cap/output", output)
    binding = _source_binding(output, _read(GOLDENS / "control.json"))
    archive = output / "staging/source.tar.gz"
    original = archive.read_bytes()
    archive.chmod(0o600)
    archive.write_bytes(gzip.compress(gzip.decompress(original), mtime=123456))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() != hashlib.sha256(original).hexdigest()
    plan = _read(output / "staging/source-plan.json")
    plan.update(fallback_size=archive.stat().st_size, fallback_sha256=ct_resume.digest(archive))
    _write(output / "staging/source-plan.json", plan)

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = binding
        ct_resume._verify_staged_source_archive(archive, output)

    admit()


@pytest.mark.parametrize(
    "cause",
    [
        "deadline reached during telemetry recovery",
        "telemetry recovery exhausted",
        "kernel remains busy/unknown after telemetry interruption",
        "kernel remains busy/unknown after telemetry hiccup",
        "ClassifiedFailure('whole-session wall cap reached')",
        "colab upload failed (1): provider changed its stderr",
        "colab exec failed (1): provider changed its stderr",
    ],
)
def test_unregistered_and_provider_reported_causes_are_not_mechanical(cause: str) -> None:
    assert not ct_resume._mechanical_supervisor_error(cause)


@pytest.mark.parametrize("surface", ["plan-field", "member-field", "paths", "unpacked-size"])
def test_source_plan_exemption_has_a_closed_producer_schema(tmp_path: Path, surface: str) -> None:
    output = tmp_path / "output"
    shutil.copytree(GOLDENS / "wall-cap/output", output)
    binding = _source_binding(output, _read(GOLDENS / "control.json"))
    path = output / "staging/source-plan.json"
    plan = _read(path)
    if surface == "plan-field":
        plan["error"] = "LookupError: hidden diagnostic"
    elif surface == "member-field":
        next(iter(plan["files"].values()))["error"] = "LookupError: hidden diagnostic"
    elif surface == "paths":
        plan["paths"] = ["LookupError: hidden diagnostic"]
    else:
        plan["fallback_unpacked_size"] += 1
    _write(path, plan)

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = binding
        ct_resume._verify_staged_source_archive(output / "staging/source.tar.gz", output)

    with pytest.raises(
        ct_resume.CTResumeRefused, match="unqualified fields|unpacked-size mismatch"
    ):
        admit()
