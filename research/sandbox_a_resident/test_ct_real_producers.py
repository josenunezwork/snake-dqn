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
    for path in sorted(GOLDENS.iterdir())
    if (path / "output/receipt.json").is_file()
    and path.name not in {"clean-completion", "child-nonmechanical-exit"}
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
        "source_commit": control["source_commit"],
        "source_archive": {
            "size": archive.stat().st_size,
            "sha256": ct_resume.digest(archive),
            "plan_sha256": ct_resume.digest(output / "staging/source-plan.json"),
        },
    }


@pytest.mark.parametrize("scenario", MECHANICAL)
@pytest.mark.parametrize("fault", [False, True])
def test_real_supervisor_mechanical_formats_and_compound_twins(
    tmp_path: Path,
    scenario: str,
    fault: bool,
) -> None:
    capture = GOLDENS / scenario
    shutil.copytree(capture / "output", tmp_path / "output")
    output = tmp_path / "output"
    log = tmp_path / "supervisor.log"
    shutil.copy2(capture / "supervisor.log", log)
    receipt, manifest = _read(output / "receipt.json"), _read(output / "manifest.json")
    # An offline path relocation does not rewrite any golden fixture. The
    # original producer manifest/receipt binding remains byte-for-byte intact.
    control = _read(GOLDENS / "control.json")

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = _source_binding(output, control)
        ct_resume.verify_mechanical_stop(
            receipt,
            manifest,
            ct_resume.digest(output / "manifest.json"),
            export_root=output,
            supervisor_log_path=log,
        )

    if fault:
        receipt["error"] += "; FloatingPointError: non-finite appended fault"
        with pytest.raises(ct_resume.CTResumeRefused, match="scientific/non-mechanical"):
            admit()
    else:
        admit()


@pytest.mark.parametrize("scenario", ["clean-completion", "child-nonmechanical-exit"])
def test_real_completed_or_unexplained_child_cannot_continue(scenario: str) -> None:
    output = GOLDENS / scenario / "output"
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume.verify_mechanical_stop(
            _read(output / "receipt.json"),
            _read(output / "manifest.json"),
            ct_resume.digest(output / "manifest.json"),
            export_root=output,
        )


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
    Path(str(path) + ".complete").touch()
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
        binding["source_archive"]["sha256"] = "0" * 64
    elif attack == "archive-size":
        binding["source_archive"]["size"] += 1
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("archive invoked Git"))
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_a, **_k: pytest.fail("archive invoked Git")
    )

    @ct_resume._evidence_session
    def admit() -> None:
        ct_resume._EVIDENCE_CACHE.get()["source_archive_binding"] = binding
        ct_resume._verify_staged_source_archive(output / "staging/source.tar.gz", output)

    if attack:
        with pytest.raises(ct_resume.CTResumeRefused, match="sealed source_commit/size/SHA"):
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
        require_current_packet(manifest, packet / "dispatch-manifest.json", root=ROOT)
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


@pytest.mark.parametrize("scenario", ["child-timeout", "wall-cap"])
def test_actual_ct_tree_writes_and_commits_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    """Real transferred C/T tree -> writer -> complete committed packet admission."""
    capture = ROOT / "tests/fixtures/ct-supervisor-06-ct-bound" / scenario
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
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "HEAD"], check=True)
    packet, output = repo / "test-packets/p0", tmp_path / "output"
    shutil.copytree(capture / "initial-packet", packet)
    shutil.copytree(capture / "output", output)
    old_manifest = _read(output / "manifest.json")
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
    )
    _write(packet / "SEAL.json", seal)
    log = tmp_path / "supervisor.log"
    shutil.copy2(capture / "supervisor.log", log)
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
    monkeypatch.setattr(ct_preflight, "ROOT", repo)
    ct_preflight.validate_packet(successor, require_committed=True, allow_tiny=True)
    with log.open("a") as stream:
        stream.write('{"nested":{"detail":"schema mismatch appended fault"}}\n')
    with pytest.raises(ValueError, match="supervisor|inventory"):
        ct_preflight.validate_packet(successor, require_committed=True, allow_tiny=True)
