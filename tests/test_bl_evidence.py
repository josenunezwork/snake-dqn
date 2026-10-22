"""M3 loader conformance controls; generated files never constitute measured qualification."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from research.bl_evidence.loader import CLOCK_SCHEMA, EXPECTED, clock_problems, digest_file
from research.bl_evidence.strict_pins import strict_evidence_problems
from research.bl_paced_identity.controller import PacedController
from research.bl_paced_identity.events import sha256
from research.bl_paced_identity.fixtures import (
    FixtureClock,
    fixture_identity,
    fixture_request,
    fixture_result,
    request_context,
)
from research.bl_paced_identity.record import Journal, export_json, read_journal


def ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": digest_file(path)}


def clock_corpus(root: Path, *, platform: str = "Darwin", synthetic: bool = False) -> dict:
    """Build a tiny protocol-conformance corpus under a pytest temporary root."""
    root.mkdir(parents=True)
    source = root / "scheduler.py"
    source.write_text("# fixture scheduler bytes\n")
    sources = {str(source): digest_file(source)}
    binding = sha256(sources)
    typed_identity = replace(
        fixture_identity(),
        warmup={"scope": "production-clock-seam"},
        load_envelope={"scope": "declared-production-load"},
    )
    identity = typed_identity.as_dict()
    candidate = sha256(identity)
    execution = {
        "schema": "bl-evidence-execution/v1",
        "candidate_sha256": candidate,
        "source_binding_sha256": binding,
        "platform": platform,
        "environment": "local-mac" if platform == "Darwin" else "colab",
        "scope": "production-clock-seam",
        "synthetic": synthetic,
        "real_clock": not synthetic,
        "real_transport": not synthetic,
    }
    execution_path = root / "execution.json"
    export_json(execution_path, execution)
    execution_sha = digest_file(execution_path)
    cases = {}
    for name, visible, consume, frame in (
        ("on-time", 79, 100, 5),
        ("publication-late", 80, 100, 5),
        ("consumption-late", 79, 110, 5),
        ("wrong-frame", 79, 100, 6),
    ):
        clock = FixtureClock(1)
        journal = Journal(
            root / name,
            identity,
            {
                "synthetic": synthetic,
                "scope": "production-clock-seam",
                "execution_sha256": execution_sha,
            },
            binding_sha256=binding,
        )
        controller = PacedController(typed_identity, journal, clock=clock, synthetic=False)
        request = fixture_request(publish_cutoff_ns=80, consume_cutoff_ns=110)
        assert controller.submit(request, {"root": "fixture"})
        assert controller.dispatch(request.request_id)
        clock.now = 70
        controller.worker_complete(request.request_id, 70)
        clock.now = visible
        controller.publish(request.request_id, fixture_result(request), 70)
        clock.now = 100
        if (
            controller.begin_poll({**request_context(request), "frame": frame}, 0, [True] * 6)
            is None
        ):
            controller.validation_complete(request.request_id, True)
            clock.now = consume
            controller.consume(request.request_id)
        journal.seal()
        _, events, _ = read_journal(journal.directory)
        decision = next(event.data for event in events if event.kind == "decision")
        accepted, reason = EXPECTED[name]
        assert decision["accepted"] is accepted and decision["reason"] == reason
        seal = journal.directory / "seal.json"
        latch_path = journal.directory / "latch.json"
        export_json(
            latch_path,
            {
                "schema": "bl-evidence-game-latch/v1",
                "candidate_sha256": candidate,
                "execution_sha256": execution_sha,
                "journal_seal_sha256": digest_file(seal),
                "request_id": decision["request_id"],
                "action": decision["action"],
                "applied_ns": clock.now,
            },
        )
        cases[name] = {
            "accepted": accepted,
            "reason": reason,
            "scheduler_verdict": "PASS",
            "journal_seal": ref(root, seal),
            "latch": ref(root, latch_path),
        }
    return {
        "schema": CLOCK_SCHEMA,
        "verdict": "PASS",
        "root": str(root),
        "candidate_sha256": candidate,
        "source_binding_sha256": binding,
        "source_sha256": sources,
        "execution": ref(root, execution_path),
        "cases": cases,
    }


def problems(report: dict) -> list[str]:
    return clock_problems(report, report["source_binding_sha256"], report["candidate_sha256"])


def test_file_backed_four_case_protocol_conformance(tmp_path):
    assert problems(clock_corpus(tmp_path / "corpus")) == []


def test_negative_missing_files_and_field_only_summary(tmp_path):
    report = clock_corpus(tmp_path / "corpus")
    (Path(report["root"]) / "execution.json").unlink()
    assert problems(report)
    field_only = {
        "schema": "bl-paced-identity-clock-seam/v1",
        "verdict": "PASS",
        "source_binding_sha256": "a" * 64,
        "cases": {
            name: {"accepted": accepted, "reason": reason, "scheduler_verdict": "PASS"}
            for name, (accepted, reason) in EXPECTED.items()
        },
    }
    assert clock_problems(field_only, "a" * 64, "b" * 64)


def test_negative_copied_b8_strict_pins():
    from web.backend.served_checkpoint import load_pins

    pin = load_pins()["checkpoints"]["b8-s3"]
    assert strict_evidence_problems("b8-s3", pin, candidate_sha256="f" * 64)


def test_negative_colab_as_mac(tmp_path, monkeypatch):
    # Validator host is irrelevant: all supplied execution evidence remains Colab.
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert problems(clock_corpus(tmp_path / "corpus", platform="Linux"))


def test_negative_changed_scheduler(tmp_path):
    report = clock_corpus(tmp_path / "corpus")
    source = Path(next(iter(report["source_sha256"])))
    source.write_text("# changed scheduler\n")
    assert problems(report)
    # Rehashing the modified closure cannot satisfy the externally frozen digest.
    changed = copy.deepcopy(report)
    changed["source_sha256"][str(source)] = digest_file(source)
    changed["source_binding_sha256"] = sha256(changed["source_sha256"])
    assert clock_problems(changed, report["source_binding_sha256"], report["candidate_sha256"])
    # An internally consistent foreign C also cannot satisfy the consumer's C.
    assert clock_problems(report, report["source_binding_sha256"], "e" * 64)


def test_negative_synthetic_clock_as_production(tmp_path):
    assert problems(clock_corpus(tmp_path / "corpus", synthetic=True))


@pytest.mark.parametrize("component", ["header.json", "seal.json.complete", "chunk-00000000.json"])
def test_missing_or_changed_journal_refused(tmp_path, component):
    report = clock_corpus(tmp_path / "corpus")
    path = Path(report["root"]) / "on-time" / component
    path.unlink()
    assert problems(report)


def test_tampered_summary_does_not_override_journal(tmp_path):
    report = clock_corpus(tmp_path / "corpus")
    report["cases"]["on-time"]["accepted"] = False
    assert problems(report)


def test_latch_after_cutoff_refused_even_with_valid_scheduler(tmp_path):
    report = clock_corpus(tmp_path / "corpus")
    path = Path(report["root"]) / "on-time/latch.json"
    value = json.loads(path.read_text())
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    value["applied_ns"] = 110
    export_json(path, value)
    report["cases"]["on-time"]["latch"] = ref(Path(report["root"]), path)
    assert problems(report)


def test_actual_legacy_release_pins_still_validate():
    from web.backend.served_checkpoint import load_pins

    for name, pin in load_pins()["checkpoints"].items():
        if not all(
            (Path(pin["strict_receipt"]["root"]) / item["path"]).is_file()
            for item in pin["strict_receipt"].values()
            if isinstance(item, dict)
        ):
            pytest.skip("immutable release artifacts are not installed on this host")
        assert strict_evidence_problems(name, pin) == []


def test_strict_missing_evidence_even_with_genuine_release_hashes(tmp_path):
    from web.backend.served_checkpoint import load_pins

    pin = load_pins()["checkpoints"]["b8-s3"]
    pin["strict_receipt"]["root"] = str(tmp_path)
    assert strict_evidence_problems("b8-s3", pin)


@pytest.mark.parametrize("relative", ["../decision.json", "/absolute/decision.json"])
def test_strict_path_escape_refused(relative):
    from web.backend.served_checkpoint import load_pins

    pin = load_pins()["checkpoints"]["b8-s3"]
    pin["strict_receipt"]["decision"]["path"] = relative
    assert strict_evidence_problems("b8-s3", pin)


def test_source_symlink_refused(tmp_path):
    report = clock_corpus(tmp_path / "corpus")
    source = Path(next(iter(report["source_sha256"])))
    moved = source.with_suffix(".real")
    source.rename(moved)
    source.symlink_to(moved)
    assert problems(report)


def test_export_replacement_between_hash_and_parse_is_refused(tmp_path, monkeypatch):
    from research.bl_evidence.loader import EvidenceError, FileRef

    path = tmp_path / "export.json"
    export_json(path, {"version": "original"})
    pinned = FileRef.parse(ref(tmp_path, path))
    original_resolve = FileRef.resolve

    def replace_after_resolve(self, root):
        resolved = original_resolve(self, root)
        for suffix in ("", ".sha256", ".complete"):
            Path(str(resolved) + suffix).unlink()
        export_json(resolved, {"version": "replacement"})
        return resolved

    monkeypatch.setattr(FileRef, "resolve", replace_after_resolve)
    with pytest.raises(EvidenceError, match="changed during read"):
        pinned.read(tmp_path)


def test_clock_cli_completed_report_and_create_only_receipt(tmp_path, monkeypatch):
    from research.bl_evidence.__main__ import main
    from research.bl_paced_identity.record import read_export

    report = clock_corpus(tmp_path / "corpus")
    path = tmp_path / "clock.json"
    export_json(path, report)
    receipt = tmp_path / "check.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m3",
            "--report",
            str(path),
            "--report-sha256",
            digest_file(path),
            "--source-binding-sha256",
            report["source_binding_sha256"],
            "--candidate-sha256",
            report["candidate_sha256"],
            "--receipt",
            str(receipt),
        ],
    )
    assert main() == 0
    assert read_export(receipt)["verdict"] == "PASS"
    with pytest.raises(FileExistsError):
        main()


def test_clock_cli_missing_report_is_not_ready(tmp_path, monkeypatch):
    from research.bl_evidence.__main__ import main
    from research.bl_paced_identity.record import read_export

    receipt = tmp_path / "check.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m3",
            "--report",
            str(tmp_path / "missing.json"),
            "--report-sha256",
            "a" * 64,
            "--source-binding-sha256",
            "b" * 64,
            "--candidate-sha256",
            "c" * 64,
            "--receipt",
            str(receipt),
        ],
    )
    assert main() == 2
    assert read_export(receipt)["verdict"] == "NOT_READY"


def test_public_strict_validator_rejects_field_only_pin(tmp_path):
    from web.backend import served_checkpoint as registry

    pins = registry.load_pins()
    pins["checkpoints"]["b8-s3"]["strict_receipt"]["root"] = str(tmp_path)
    assert registry.strict_pin_problems("b8-s3", pins)


def test_public_strict_validator_refuses_copied_b8_for_assisted_c():
    from web.backend import served_checkpoint as registry

    assert registry.strict_pin_problems("b8-s3", candidate_sha256="d" * 64)


def test_public_strict_validator_accepts_actual_released_evidence():
    from web.backend import served_checkpoint as registry

    for name in ("frp3-s12", "b8-s3"):
        pin = registry.load_pins()["checkpoints"][name]["strict_receipt"]
        if not all(
            (Path(pin["root"]) / item["path"]).is_file()
            for item in pin.values()
            if isinstance(item, dict)
        ):
            pytest.skip("immutable release artifacts are not installed on this host")
        assert registry.strict_pin_problems(name) == []


@pytest.mark.parametrize(
    "document", [[], {"schema": "served-checkpoint-strict-pins/v1", "checkpoints": []}]
)
def test_public_strict_validator_malformed_documents_fail_closed(document):
    from web.backend import served_checkpoint as registry

    assert registry.strict_pin_problems("b8-s3", document)


def test_public_clock_validator_requires_external_c_and_file_backed_proof(tmp_path):
    from research.bl_paced_identity.clock_seam import clock_seam_passed

    report = clock_corpus(tmp_path / "corpus")
    source, candidate = report["source_binding_sha256"], report["candidate_sha256"]
    assert not clock_seam_passed(report, source)
    assert clock_seam_passed(report, source, candidate_sha256=candidate)
    assert not clock_seam_passed(report, source, candidate_sha256="f" * 64)
    (Path(report["root"]) / "on-time/seal.json.complete").unlink()
    assert not clock_seam_passed(report, source, candidate_sha256=candidate)


@pytest.mark.parametrize("mutation", ["missing", "colab", "scheduler", "synthetic"])
def test_public_clock_validator_negative_controls(tmp_path, mutation):
    from research.bl_paced_identity.clock_seam import clock_seam_passed

    report = clock_corpus(
        tmp_path / "corpus",
        platform="Linux" if mutation == "colab" else "Darwin",
        synthetic=mutation == "synthetic",
    )
    if mutation == "missing":
        (Path(report["root"]) / "execution.json").unlink()
    if mutation == "scheduler":
        Path(next(iter(report["source_sha256"]))).write_text("# changed scheduler\n")
    assert not clock_seam_passed(
        report, report["source_binding_sha256"], candidate_sha256=report["candidate_sha256"]
    )
