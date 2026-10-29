"""Readiness-5: the reviewer's attacks must poison every c2 recovery boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_prefix_recovery as p
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_training_prefix_recovery import (
    build,
    fixture,
    mechanical_stop,
    rewrite,
    verify,
)
from research.sandbox_a_resident.units import digest


def completion(fx: dict, cert: Path) -> r.ResumePacket:
    return r.completion_packet(
        fx["manifest"],
        None,
        fx["exports"],
        fx["control"],
        completion_index=2,
        job_id="train01-c2",
        output_dir=cert.parent / "c2-output",
        original_manifest_sha256=digest(fx["output"] / "manifest.json"),
        output_root=fx["output"],
        recovery_evidence_path=cert,
    )


def rebind(fx: dict, cert: Path) -> None:
    value = json.loads(cert.read_text())
    value["export_directory_inventory"] = p._export_inventory(fx["output"] / "exports")
    for name, row in value["retained_files"].items():
        path = fx["output"] / name
        row.update(size=path.stat().st_size, sha256=digest(path))
    rewrite(cert, value)


@pytest.mark.parametrize(
    "attack",
    [
        "receipt-nonfinite",
        "receipt-child",
        "receipt-unknown",
        "receipt-truncated",
        "receipt-cleanup",
        "receipt-mechanical-wrong-error",
        "terminal-runtime-error",
        "terminal-false-finite",
        "terminal-completed",
        "terminal-exit124",
        "terminal-truncated",
        "poll-runtime-error",
        "poll-nonfinite",
        "poll-truncated-fault",
        "log-traceback",
        "log-exit",
        "log-novel",
        "log-truncated-fault",
        "distribution",
        "stacks",
        "summary",
        "progress-stage",
        "watchdog-stage",
        "unknown-root-fault",
        "recombined-review-attack",
    ],
)
def test_review_attacks_refuse_builder_verifier_and_completion(tmp_path: Path, attack: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    terminal = {
        "job_id": fx["manifest"]["job_id"],
        "generation": "b" * 32,
        "status": "failed",
        "reason": "runtime_error",
        "returncode": 1,
    }
    if attack.startswith("receipt") or attack == "recombined-review-attack":
        mechanical_stop(fx, "runtime_timeout", receipt=True)
        value = json.loads((fx["output"] / "receipt.json").read_text())
        if attack in {"receipt-nonfinite", "recombined-review-attack"}:
            value.update(
                failure_class="CHILD_NONZERO_EXIT",
                error="FloatingPointError: non-finite value_loss",
            )
        elif attack == "receipt-child":
            value.update(
                failure_class="CHILD_NONZERO_EXIT",
                error="ClassifiedFailure('child exited with status 1')",
            )
        elif attack == "receipt-unknown":
            value["extra"] = "NovelScientificException('trip')"
        elif attack == "receipt-cleanup":
            value["cleanup_error"] = "cleanup did not complete"
        elif attack == "receipt-mechanical-wrong-error":
            value["error"] = (
                "AccountingError('supervised assignment disappeared from sessions; "
                "NovelException()')"
            )
        (fx["output"] / "receipt.json").write_text(
            json.dumps(value) if attack != "receipt-truncated" else '{"status":'
        )
    if attack.startswith("terminal"):
        if attack == "terminal-false-finite":
            terminal.update(reason="timeout", returncode=-15, finite=False)
        elif attack == "terminal-completed":
            terminal.update(reason=None, status="succeeded", returncode=0)
        elif attack == "terminal-exit124":
            terminal.update(reason="timeout", returncode=124)
        (fx["output"] / "terminal.json").write_text(
            json.dumps(terminal) if attack != "terminal-truncated" else '{"reason":'
        )
    if attack.startswith("poll") or attack == "recombined-review-attack":
        value = {**fx["poll"], "terminal": terminal}
        if attack in {"poll-nonfinite", "recombined-review-attack"}:
            value["finite"] = False
        (fx["output"] / "poll-000002.json").write_text(
            json.dumps(value) if attack != "poll-truncated-fault" else '{"finite": false, "data":'
        )
    if attack.startswith("log") or attack == "recombined-review-attack":
        text = {
            "log-traceback": (
                "Traceback (most recent call last):\n"
                "FloatingPointError: non-finite value_loss\nEXIT=1\n"
            ),
            "log-exit": "EXIT=1\n",
            "log-novel": "NovelScientificException('not in denylist')\n",
            "log-truncated-fault": '{"stage": "update-failed", "data":',
            "recombined-review-attack": "FloatingPointError: non-finite value_loss\nEXIT=1\n",
        }[attack]
        with fx["log"].open("a") as stream:
            stream.write(text)
    paths = {
        "distribution": ("distribution-screen.json", '{"material_gaps": ["missing mix"]}'),
        "stacks": ("stall-watchdog/faulthandler-stacks.log", "Thread 0x1234 stopped in learner"),
        "summary": ("summary.json", "{}"),
        "progress-stage": ("progress.json", '{"stage": "update-failed"}'),
        "watchdog-stage": ("stall-watchdog/stall-watchdog.json", '{"stage": "update-failed"}'),
        "unknown-root-fault": ("../../unexpected-diagnostic.json", '{"finite": false}'),
    }
    if attack in paths:
        rel, text = paths[attack]
        path = fx["exports"] / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    # Rebinding is deliberate: refusal must come from facts, not stale hashes.
    rebind(fx, cert)
    for action in (
        lambda: build(fx, tmp_path / "bad.json"),
        lambda: verify(fx, cert),
        lambda: completion(fx, cert),
    ):
        with pytest.raises(r.ResumeRefused):
            action()
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("kind", ["runtime_timeout", "lease_expired"])
@pytest.mark.parametrize("with_receipt", [False, True])
def test_known_self_release_and_suffix_truncation_admit(
    tmp_path: Path, kind: str, with_receipt: bool
) -> None:
    fx = fixture(tmp_path)
    mechanical_stop(fx, kind, receipt=with_receipt)
    # A genuinely EOF-truncated file after cutoff may be retained but cannot bind payloads.
    (fx["output"] / "poll-000002.json").write_text('{"state": {"job_id": "train01-r6", "progress":')
    (fx["exports"] / "progress.json.partial").write_text('{"stage": "update-running", "data":')
    with fx["log"].open("a") as stream:
        stream.write('{"progress":')
    cert = tmp_path / "good.json"
    result = build(fx, cert, kind)
    assert result["post_hoc_release_verified"] is True
    assert result["original_receipt_present"] is with_receipt
    assert result["suffix_scanned_for_poison"] is True
    assert completion(fx, cert).latest_iteration == 96


@pytest.mark.parametrize(
    "attack",
    [
        "missing-capture",
        "busy-sessions",
        "nonzero-rate",
        "pending",
        "stale",
        "future",
        "timeout-hour3-of46",
        "lease-no-terminal",
        "host-loss-conflicts",
    ],
)
def test_release_and_cause_evidence_fault_twins(tmp_path: Path, attack: str) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    value = json.loads(cert.read_text())
    kind = "host_loss"
    if attack == "missing-capture":
        value.pop("account_status_capture")
    elif attack in {"busy-sessions", "nonzero-rate", "pending", "stale", "future"}:
        capture = json.loads(fx["capture"].read_text())
        if attack == "busy-sessions":
            capture["status"]["sessions"] = {"r6": {"endpoint": "live"}}
        elif attack == "nonzero-rate":
            capture["status"]["usage"]["rate"] = 8.4
        elif attack == "pending":
            capture["status"]["unresolved"] = ["active-r6"]
        else:
            capture["captured_unix_seconds"] += -90000 if attack == "stale" else 3600
        rewrite(fx["capture"], capture)
        value["account_status_capture"].update(
            size=fx["capture"].stat().st_size,
            sha256=digest(fx["capture"]),
            captured_unix_seconds=capture["captured_unix_seconds"],
        )
    elif attack == "timeout-hour3-of46":
        mechanical_stop(fx, "runtime_timeout")
        fx["manifest"]["timeout_minutes"] = 46 * 60
        (fx["output"] / "manifest.json").write_text(json.dumps(fx["manifest"]))
        value["identities"]["manifest_sha256"] = digest(fx["output"] / "manifest.json")
        row = value["retained_files"]["manifest.json"]
        row.update(
            size=(fx["output"] / "manifest.json").stat().st_size,
            sha256=digest(fx["output"] / "manifest.json"),
        )
        kind = "runtime_timeout"
    elif attack == "lease-no-terminal":
        kind = "lease_expired"
    else:
        mechanical_stop(fx, "runtime_timeout")
    value["stop_kind"] = kind
    rewrite(cert, value)
    for action in (lambda: verify(fx, cert), lambda: completion(fx, cert)):
        with pytest.raises(r.ResumeRefused):
            action()


@pytest.mark.parametrize(
    "fragment",
    [
        '"stage": "update-failed"}',
        '"phase": "custom-failed"}',
        '"prepare_phase": "collection-failed"}',
    ],
)
def test_first_line_failed_stage_fragment_refuses(fragment: str) -> None:
    with pytest.raises(r.ResumeRefused, match="failed stage fragment"):
        r._scan_diagnostic_lines(
            fragment, label="tail", expected_job_id="train01-r6", expected_generation="b" * 32
        )


@pytest.mark.parametrize("value", ["mode: nan-guard", "mode: inf-guard", "mode: Infinity-check"])
def test_nonfinite_hyphenated_word_healthy_fault_twins(value: str) -> None:
    r._refuse_if_fault_text(value, field_name="tail")
    for bad in ["value_loss=nan", "loss: inf", "loss=Infinity"]:
        with pytest.raises(r.ResumeRefused, match="non-finite numeric text"):
            r._refuse_if_fault_text(bad, field_name="tail")


def test_every_prefix_event_is_bound_not_only_the_last(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    data = fx["log"].read_text().replace('"exports": 0', '"exports": 99')
    fx["log"].write_text(data)
    value = json.loads(cert.read_text())
    value["supervisor_log_prefix"].update(size=fx["log"].stat().st_size, sha256=digest(fx["log"]))
    rewrite(cert, value)
    with pytest.raises(r.ResumeRefused, match="ordered poll prefix"):
        verify(fx, cert)
    with pytest.raises(r.ResumeRefused, match="ordered poll prefix"):
        build(fx, tmp_path / "bad.json")


def test_full_export_listing_and_newer_deleted_snapshot_poison(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    cert = tmp_path / "good.json"
    build(fx, cert)
    # Listing binds unfinished files as well as ledger units.
    (fx["exports"] / "pending.json.partial").write_text('{"data":')
    with pytest.raises(r.ResumeRefused, match="full export directory inventory differs"):
        verify(fx, cert)
    (fx["exports"] / "pending.json.partial").unlink()
    poll = {
        **fx["poll"],
        "exports": [
            *fx["poll"]["exports"],
            {"path": "training/iteration-0112.pt.xz", "size": 4, "sha256": "d" * 64},
        ],
    }
    (fx["output"] / "poll-000002.json").write_text(json.dumps(poll))
    with pytest.raises(r.ResumeRefused, match="newer snapshot than recovery cutoff"):
        build(fx, tmp_path / "bad.json")


def test_suffix_export_inventory_never_supports_prefix(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    (fx["output"] / "poll-000001.json").write_text(json.dumps({**fx["poll"], "exports": []}))
    (fx["output"] / "poll-000002.json").write_text(json.dumps(fx["poll"]))
    # There is no matching log event for this extra poll; a suffix must not backfill it.
    with pytest.raises(r.ResumeRefused, match="ordered poll prefix|no supervisor event"):
        build(fx, tmp_path / "bad.json")


def test_fresh_r6_scratch_copy_admits_every_log_line_and_poll(tmp_path: Path) -> None:
    """Immutable real live capture is diagnostic liveness, never release admission."""
    import hashlib
    import tarfile

    corpus = Path(__file__).parent / "fixtures/r6_readiness5_liveness"
    with tarfile.open(corpus / "evidence.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    inventory = json.loads((corpus / "inventory.json").read_text())
    assert set(inventory) == {
        str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()
    }
    for name, row in inventory.items():
        path = tmp_path / name
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    generation = json.loads((tmp_path / "output/poll-000000.json").read_text())["state"][
        "generation"
    ]
    result = r._check_supervisor_log_fault_evidence(
        tmp_path / "supervisor.log",
        expected_job_id=manifest["job_id"],
        expected_generation=generation,
    )
    assert result["line_count"] == 1296
    assert result["matched_job_records"] >= 1088
    assert r._check_output_root_fault_evidence(
        tmp_path / "output", expected_job_id=manifest["job_id"], expected_generation=generation
    ) == (1088, 1088)


@pytest.mark.parametrize(
    "text",
    [
        '{"value_loss": 1e309, "data":',
        '{"material_gaps": ["missing mix"], "data":',
        '{"state": {"generation": "foreign"}, "data":',
        '{"error": "NovelScientificException()", "data":',
    ],
)
@pytest.mark.parametrize("surface", ["poll", "log", "publication"])
def test_truncated_suffix_keeps_every_negative_fact(
    tmp_path: Path, text: str, surface: str
) -> None:
    fx = fixture(tmp_path)
    if surface == "log":
        with fx["log"].open("a") as stream:
            stream.write(text)
    else:
        path = (
            fx["output"] / "poll-000002.json"
            if surface == "poll"
            else fx["exports"] / "progress.json.partial"
        )
        path.write_text(text)
    with pytest.raises(r.ResumeRefused):
        build(fx, tmp_path / "bad.json")


@pytest.mark.parametrize("kind", ["runtime_timeout", "lease_expired"])
def test_native_terminal_event_and_exact_receipt_footer_twins(tmp_path: Path, kind: str) -> None:
    fx = fixture(tmp_path)
    mechanical_stop(fx, kind, receipt=True)
    receipt_path = fx["output"] / "receipt.json"
    value = json.loads(receipt_path.read_text())
    value["terminal"]["exports"] = fx["poll"]["exports"]
    value["exports"] = fx["poll"]["exports"]
    receipt_path.write_text(json.dumps(value))
    (fx["output"] / "terminal.json").write_text(json.dumps(value["terminal"]))
    lines = fx["log"].read_text().splitlines()
    lines[-1] = json.dumps({**fx["event"], "terminal": True, "status": "failed", "elapsed": 3600.0})
    # Exact qualified wrapper footer is allowed only for the verified mechanical receipt.
    lines.extend(
        [
            json.dumps(value),
            f"INCOMPLETE: {p.SELF_RELEASE_ERROR}; receipt: {receipt_path}",
            "EXIT=1",
        ]
    )
    fx["log"].write_text("\n".join(lines) + "\n")
    cert = tmp_path / "good.json"
    build(fx, cert, kind)
    assert completion(fx, cert).latest_iteration == 96
    with fx["log"].open("a") as stream:
        stream.write("NovelScientificException('after known footer')\n")
    with pytest.raises(r.ResumeRefused):
        verify(fx, cert)
