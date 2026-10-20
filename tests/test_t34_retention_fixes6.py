"""FIX-FIRST #6: fixed phase roots, published history, and abort verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb import retention_history_v2 as history
from research.t34_pathb import retention_v2 as runner
from research.t34_pathb.host_guard_v2 import HostContentionError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _authoring_repo(tmp_path: Path) -> Path:
    """Build an actual local authoring checkout plus bare origin/main."""
    remote = tmp_path / "origin.git"
    author = tmp_path / "author"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(tmp_path, "init", str(author))
    _git(author, "config", "user.email", "retention@example.test")
    _git(author, "config", "user.name", "Retention test")
    _git(author, "checkout", "-b", "main")
    _git(author, "remote", "add", "origin", str(remote))
    (author / "README").write_text("fixture\n")
    _git(author, "add", "README")
    _git(author, "commit", "-m", "initial fixture")
    _git(author, "push", "-u", "origin", "main")
    return author


def _registration(tmp_path: Path, author: Path) -> dict[str, object]:
    study = tmp_path / "study"
    return {
        "study_root": str(study),
        "timing_attempt_parent": str(study / "timing"),
        "timing_max_attempts": 3,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=author, text=True
        ).strip(),
        "history_ledger": {
            "repo": str(author.resolve()),
            "path": f"research/t34_pathb/ledger/{study.name}.jsonl",
        },
    }


def _publish(author: Path, registration: dict[str, object]) -> None:
    _, ledger_name = history.ledger_policy(registration)
    _git(author, "add", ledger_name)
    _git(author, "commit", "-m", "publish terminal history")
    _git(author, "push", "origin", "main")


def _reserve_and_finalize(
    registration: dict[str, object], phase: str, *, receipt: str = "receipt.json"
) -> Path:
    out = Path(str(registration["study_root"])) / phase
    runner._reserve_phase(out, registration, phase)
    (out / receipt).write_text(json.dumps({"phase": phase, "receipt": receipt}) + "\n")
    # A terminal receipt must include the source-fence certificate before its
    # phase seal is generated, so later additions are detected as tampering.
    (out / "source_validated.json").write_text(json.dumps({"status": "PASS"}) + "\n")
    history.finalize_phase(out, registration)
    return out


def test_fixed_phase_reservation_is_create_only_and_checks_history(
    monkeypatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner, "require_published_history", lambda value: calls.append(dict(value))
    )
    precheck = Path(str(registration["study_root"])) / "precheck"

    runner._reserve_phase(precheck, registration, "precheck")
    assert calls == [registration]
    assert json.loads((precheck / "phase_reservation.json").read_text())["phase"] == "precheck"
    with pytest.raises(runner.RegistrationError, match="fresh"):
        runner._reserve_phase(precheck, registration, "precheck")
    with pytest.raises(runner.RegistrationError, match="fixed phase directory"):
        runner._reserve_phase(tmp_path / "elsewhere", registration, "efficacy")


@pytest.mark.parametrize(
    "phase,relative",
    [("precheck", "precheck"), ("timing", "timing/attempt-01")],
)
def test_remote_admission_claim_blocks_deleted_phase_reroll(
    tmp_path: Path, phase: str, relative: str
) -> None:
    """An erased local directory cannot make a claimed phase eligible again."""
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    out = Path(str(registration["study_root"])) / relative
    if phase == "precheck":
        runner._reserve_phase(out, registration, phase)
        (out / "phase_reservation.json").unlink()
    else:
        runner._reserve_timing_attempt(out, registration)
    out.rmdir()

    with pytest.raises(history.RegistrationError, match="remote phase admission claim exists"):
        if phase == "precheck":
            runner._reserve_phase(out, registration, phase)
        else:
            runner._reserve_timing_attempt(out, registration)


def test_efficacy_accepts_only_original_fixed_precheck(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    precheck = Path(str(registration["study_root"])) / "precheck"
    precheck.mkdir(parents=True)
    original = {"source_validation_path": str(precheck / "source_validated.json"), "value": 1}
    (precheck / "precheck.json").write_text(json.dumps(original))

    runner._require_fixed_precheck(original, registration)
    copied = dict(original)
    copied["source_validation_path"] = str(tmp_path / "copied" / "source_validated.json")
    with pytest.raises(runner.RegistrationError, match="fixed precheck"):
        runner._require_fixed_precheck(copied, registration)
    (precheck / "precheck.json").write_text(json.dumps({"value": 2}))
    with pytest.raises(runner.RegistrationError, match="fixed precheck"):
        runner._require_fixed_precheck(original, registration)


def test_precheck_source_certificate_rejects_interruption_or_kill_decision(tmp_path: Path) -> None:
    """A late-aborted precheck cannot be reused to admit an efficacy phase."""
    precheck = tmp_path / "precheck"
    precheck.mkdir()
    certificate_path = precheck / "source_validated.json"
    receipt = {"source_commit": "a" * 40, "source_validation_path": str(certificate_path)}
    certificate_path.write_text(
        json.dumps(
            {
                "schema": "t34-pathb-v2-source-validation/v1",
                "status": "PASS",
                "source_commit": receipt["source_commit"],
                "receipts": {"precheck.json": runner._canonical_sha(receipt)},
            }
        )
    )
    (precheck / "decision.json").write_text(json.dumps({"verdict": "GO"}))
    runner._require_phase_source(receipt, "precheck.json")

    (precheck / "interruption.json").write_text(json.dumps({"claim_b": "KILL"}))
    with pytest.raises(runner.RegistrationError, match="interruption"):
        runner._require_phase_source(receipt, "precheck.json")
    (precheck / "interruption.json").unlink()
    (precheck / "decision.json").write_text(json.dumps({"verdict": "KILL"}))
    with pytest.raises(runner.RegistrationError, match="terminal failure"):
        runner._require_phase_source(receipt, "precheck.json")


def test_analyze_cli_reports_interruption_as_effective_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Descriptive analysis cannot revive a decision invalidated by interruption."""
    runner._write_new(tmp_path / "decision.json", {"verdict": "GO"})
    runner._write_new(tmp_path / "interruption.json", {"claim_b": "KILL"})
    monkeypatch.setattr(runner, "analyze_path", lambda *_args, **_kwargs: {"metric": 1})
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["retention_v2.py", "analyze", "--out", str(tmp_path)])

    assert runner.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["official_decision"]["verdict"] == "KILL"
    assert output["preserved_decision"] == {"verdict": "GO"}
    assert output["source_failure"] == {"claim_b": "KILL"}
    assert output["descriptive_statistics"] == {"metric": 1}


def test_terminal_ledger_must_be_committed_pushed_and_byte_exact(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    precheck = _reserve_and_finalize(registration, "precheck")
    seal = json.loads((precheck / "phase_terminal.json").read_text())
    assert set(seal["files"]) == {"phase_reservation.json", "receipt.json", "source_validated.json"}
    with pytest.raises(history.RegistrationError, match="committed and pushed"):
        history.require_published_history(registration)

    _publish(author, registration)
    history.require_published_history(registration)

    # A changed or removed evidence receipt cannot be hidden behind a published seal.
    (precheck / "receipt.json").write_text('{"changed": true}\n')
    with pytest.raises(history.RegistrationError, match="terminal history"):
        history.require_published_history(registration)
    (precheck / "receipt.json").unlink()
    with pytest.raises(history.RegistrationError, match="terminal history"):
        history.require_published_history(registration)


def test_reordered_or_unpublished_terminal_history_fails_closed(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    _reserve_and_finalize(registration, "precheck")
    _publish(author, registration)
    _reserve_and_finalize(registration, "efficacy")
    # The local append is intentionally not pushed, so the next admission refuses it.
    with pytest.raises(history.RegistrationError, match="committed and pushed"):
        history.require_published_history(registration)

    _, ledger_name = history.ledger_policy(registration)
    ledger = author / ledger_name
    rows = ledger.read_text().splitlines()
    ledger.write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(history.RegistrationError, match="committed and pushed|differs"):
        history.require_published_history(registration)


@pytest.mark.parametrize("operation", ["truncate", "rewrite", "delete"])
def test_published_ledger_rewrite_or_deletion_is_permanently_refused(
    tmp_path: Path, operation: str
) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    _reserve_and_finalize(registration, "precheck")
    _publish(author, registration)
    _, ledger_name = history.ledger_policy(registration)
    ledger = author / ledger_name
    if operation == "truncate":
        ledger.write_text("")
        _git(author, "add", ledger_name)
    elif operation == "rewrite":
        row = json.loads(ledger.read_text())
        row["terminal_sha256"] = "0" * 64
        ledger.write_text(json.dumps(row) + "\n")
        _git(author, "add", ledger_name)
    else:
        _git(author, "rm", ledger_name)
    _git(author, "commit", "-m", f"{operation} published ledger")
    _git(author, "push", "origin", "main")
    with pytest.raises(history.RegistrationError, match="not append-only|was removed"):
        history.require_published_history(registration)


def test_finalizer_rejects_a_second_seal_or_nonlatest_phase(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    precheck = _reserve_and_finalize(registration, "precheck")
    with pytest.raises(history.RegistrationError, match="prefix differs|already exists"):
        history.finalize_phase(precheck, registration)


def test_measured_host_contention_is_the_only_excused_interruption() -> None:
    record = {"contended": True, "reasons": ["thermal_not_ok"]}
    exc = HostContentionError("measured thermal contention")
    exc.measured_records = [record]  # type: ignore[attr-defined]
    assert (
        runner._interruption_verdict(exc, SimpleNamespace(records=[record], _sample_error=None))
        == "INCONCLUSIVE"
    )


def test_precheck_operator_interrupt_records_kill_without_opening_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner itself turns an operator abort into a durable KILL receipt."""
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    registration.update({"ratified": False, "load_profile": {}})
    monkeypatch.setattr(runner, "require_timing_gate", lambda *_args: None)
    monkeypatch.setattr(runner, "require_registration", lambda *_args: None)
    monkeypatch.setattr(runner, "_require_committed_pushed_registration", lambda *_args: None)
    monkeypatch.setattr(runner, "require_published_history", lambda *_args: None)
    monkeypatch.setattr(runner, "finalize_phase", lambda *_args: None)

    class AbortingGuard:
        records: list[dict[str, object]] = []

        def admit(self) -> None:
            raise KeyboardInterrupt("operator abort")

        def summary(self, **_kwargs: object) -> dict[str, object]:
            return {}

        def close(self) -> None:
            pass

    with pytest.raises(KeyboardInterrupt, match="operator abort"):
        runner.run_precheck(
            [],
            host_guard=AbortingGuard(),
            timing_gate={},
            registration=registration,
            registration_path=Path(__file__),
            out=Path(str(registration["study_root"])) / "precheck",
        )
    decision = json.loads(
        (Path(str(registration["study_root"])) / "precheck" / "decision.json").read_text()
    )
    assert decision["verdict"] == "KILL"


def test_late_operator_interrupt_invalidates_saved_go_and_seals_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A signal after a GO receipt cannot leave a promotable terminal phase."""
    from research.t34_pathb import source_binding_v2

    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    registration.update({"ratified": True})

    class Fence:
        def start(self) -> "Fence":
            return self

        def final_check(self) -> None:
            raise KeyboardInterrupt("late operator abort")

        def close(self) -> None:
            pass

    monkeypatch.setattr(source_binding_v2, "FrozenWorktreeGuard", lambda *_args: Fence())
    out = Path(str(registration["study_root"])) / "precheck"

    @runner._frozen_execution
    def producer(*, out: Path, registration: dict[str, object]) -> None:
        out.mkdir(parents=True)
        runner._TIMING_RESERVATION.get()[0] = True
        runner._write_new(out / "decision.json", {"verdict": "GO"})

    with pytest.raises(KeyboardInterrupt, match="late operator abort"):
        producer(out=out, registration=registration)
    interruption = json.loads((out / "interruption.json").read_text())
    assert interruption["claim_b"] == interruption["status"] == "KILL"
    assert interruption["invalidates_decision"] is True
    assert json.loads((out / "decision.json").read_text())["verdict"] == "GO"
    seal = json.loads((out / "phase_terminal.json").read_text())
    assert {"decision.json", "interruption.json"} <= set(seal["files"])
    _, ledger_name = history.ledger_policy(registration)
    assert len((author / ledger_name).read_text().splitlines()) == 1


def test_measured_wrapper_interruption_preserves_inconclusive_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Measured host interruption retains its registered INCONCLUSIVE semantics."""
    from research.t34_pathb import source_binding_v2

    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    registration.update({"ratified": True})

    class Fence:
        def start(self) -> "Fence":
            return self

        def close(self) -> None:
            pass

    monkeypatch.setattr(source_binding_v2, "FrozenWorktreeGuard", lambda *_args: Fence())
    out = Path(str(registration["study_root"])) / "precheck"

    @runner._frozen_execution
    def producer(*, out: Path, registration: dict[str, object]) -> None:
        out.mkdir(parents=True)
        runner._TIMING_RESERVATION.get()[0] = True
        runner._write_new(out / "decision.json", {"verdict": "INCONCLUSIVE"})
        exc = HostContentionError("measured thermal contention")
        exc._retention_measured_interruption = True  # type: ignore[attr-defined]
        raise exc

    with pytest.raises(HostContentionError, match="measured thermal"):
        producer(out=out, registration=registration)
    assert not (out / "interruption.json").exists()
    assert json.loads((out / "decision.json").read_text())["verdict"] == "INCONCLUSIVE"
    assert (out / "phase_terminal.json").is_file()


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), RuntimeError("operator abort")])
def test_operator_or_unmeasured_interruptions_kill(exc: BaseException) -> None:
    record = {"contended": True, "reasons": ["thermal_not_ok"]}
    assert (
        runner._interruption_verdict(exc, SimpleNamespace(records=[record], _sample_error=None))
        == "KILL"
    )


@pytest.mark.parametrize(
    "record,measured,sample_error",
    [
        ({"contended": True, "reasons": ["unrecognized"]}, True, None),
        ({"contended": False, "reasons": ["thermal_not_ok"]}, True, None),
        ({"contended": True, "reasons": ["thermal_not_ok"]}, False, None),
        ({"contended": True, "reasons": ["thermal_not_ok"]}, True, "sampler failed"),
    ],
)
def test_bad_or_absent_measured_host_evidence_kills(record, measured, sample_error) -> None:
    exc = HostContentionError("interrupted")
    if measured:
        exc.measured_records = [record]  # type: ignore[attr-defined]
    assert (
        runner._interruption_verdict(
            exc, SimpleNamespace(records=[record], _sample_error=sample_error)
        )
        == "KILL"
    )
