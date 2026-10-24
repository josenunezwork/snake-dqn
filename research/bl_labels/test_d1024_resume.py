"""Synthetic continuation proofs; no real factory, labels, or scientific look."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels import d1024_resume as resume
from research.bl_labels import frozen_factory, mac_driver, mac_gates
from research.bl_labels.exports import _verify, load_ledger, publish_json, publish_session
from research.bl_labels.test_d256_resume import life
from research.bl_labels.test_mac_integration import _install_fake_factory, _patch_inline_runtime


@pytest.fixture(autouse=True)
def isolate_inline_worker_environment(monkeypatch):
    for key in ("F0_COMMIT", "F1_COMMIT", "BL_FROZEN_FACTORY_WORKER"):
        prior = os.environ.get(key)
        monkeypatch.setenv(key, prior or "")
        if prior is None:
            monkeypatch.delenv(key)


def pack_fixture(tmp_path: Path) -> tuple[Path, dict]:
    pack = tmp_path / "pack"
    control = {
        "namespace": "d1024-synthetic",
        "source_commit": resume.SCIENCE_PIN,
        "allocation": [
            {"allocation_id": f"p{i}", "seed": i, "stream": "watch6", "horizon": 2}
            for i in range(3)
        ],
        "mac_gate": {"stage": "d1024", "status": "HOLD", "source_commit": resume.SCIENCE_PIN},
    }
    publish_json(pack / "d1024-control.json", control)
    control = copy.deepcopy(control)
    control["mac_gate"].update(
        status="APPROVED",
        deadline_seconds=resume.CAP,
        package_root=str(pack),
        state_dir=str(pack / "d1024-state-mac-01"),
        original_control_path=str(pack / "d1024-control.json"),
        original_control_sha256=_verify(pack / "d1024-control.json"),
        collector_source_commit=frozen_factory.OLD_PIN,
        collector_root=str(tmp_path / "old"),
        w12_calibration_source_commit=frozen_factory.OLD_PIN,
        bindings={
            "speed_equivalence": {"status": "FAIL_OLD_FALLBACK", "path": str(tmp_path / "eq")},
            "calibration": {"path": str(tmp_path / "cal")},
            "f4_readiness": {
                "control_path": str(tmp_path / "f4/control.json"),
                "inputs_path": str(tmp_path / "f4"),
            },
        },
    )
    publish_json(pack / "d1024-admitted.json", control)
    return pack, control


def add_attempt(
    pack: Path,
    control: dict,
    index: int,
    *,
    elapsed: float = 100,
    prior: float = 0,
    terminal: bool = True,
    start_change: dict | None = None,
    terminal_change: dict | None = None,
    export_change: dict | None = None,
) -> Path:
    state = pack / "d1024-state-mac-01"
    output = pack / f"d1024-output-mac-{index:02d}"
    start_path = state / f"attempt-{index:02d}.start.json"
    start = {
        "schema": "bl-labels-f0-mac-attempt-start/v1",
        "namespace": control["namespace"],
        "source_commit": resume.SCIENCE_PIN,
        "workers": 6,
        "owner_w12_ok": False,
        "w12_calibration_receipt_sha256": None,
        "w12_calibration_source_commit": frozen_factory.OLD_PIN,
        "allocation_sha256": resume.canonical_sha(control["allocation"]),
        "control_sha256": resume.canonical_sha(control),
        "deadline_seconds": resume.CAP,
        "original_wall_cap_seconds": resume.CAP,
        "state_dir": str(state),
        "output": str(output),
        "operational_source_commit": resume.ORIGINAL_OPERATIONAL_PIN,
        **(start_change or {}),
    }
    publish_json(start_path, start)
    os.utime(start_path, ns=(index * 10**9, index * 10**9))
    output.mkdir(exist_ok=True)
    if not terminal:
        return output
    sealed = state / f"attempt-{index:02d}.rate-sealed.json"
    publish_json(sealed, {"events": []})
    result = {
        "schema": "bl-labels-f0-mac-driver/v1",
        "namespace": control["namespace"],
        "status": "PREEMPTED",
        "invalid_reason": None,
        "workers": 6,
        "owner_w12_ok": False,
        "w12_calibration_receipt_sha256": None,
        "w12_calibration_source_commit": frozen_factory.OLD_PIN,
        "collector_source_commit": frozen_factory.OLD_PIN,
        "operational_source_commit": resume.ORIGINAL_OPERATIONAL_PIN,
        "stop_reason": "operational guard: synthetic interruption",
        "registered_stop_reason": None,
        "output": str(output),
        "state_dir": str(state),
        "start_receipt": start_path.name,
        "start_sha256": _verify(start_path),
        "elapsed_seconds": elapsed,
        "prior_elapsed_seconds": prior,
        "cumulative_elapsed_seconds": prior + elapsed,
        "worker_seconds_denominator": 6 * elapsed,
        "prior_worker_seconds_denominator": 6 * prior,
        "cumulative_worker_seconds_denominator": 6 * prior + 6 * elapsed,
        "attempts_ledger": [f"attempt-{i:02d}.terminal.json" for i in range(1, index + 1)],
        "rate_sealed_path": str(sealed),
        **(terminal_change or {}),
    }
    publish_json(state / f"attempt-{index:02d}.terminal.json", result)
    publish_json(
        output / "runtime-result.json", {**result, "mac_lease": {}, **(export_change or {})}
    )
    return output


@pytest.mark.parametrize(
    "change",
    [
        {"workers": 12},
        {"deadline_seconds": 144001},
        {"original_wall_cap_seconds": 100},
        {"source_commit": "0" * 40},
        {"control_sha256": "0" * 64},
        {"allocation_sha256": "0" * 64},
        {"state_dir": "/foreign"},
        {"output": "/foreign"},
        {"operational_source_commit": "0" * 40},
        {"owner_w12_ok": True},
    ],
)
def test_changed_start_holds(tmp_path, change) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, start_change=change)
    with pytest.raises(ValueError, match="start identity.*HOLD"):
        resume.attempt_chain(pack, control)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "INVALID"},
        {"status": "COMPLETE"},
        {"invalid_reason": "scientific failure"},
        {"collector_source_commit": resume.SCIENCE_PIN},
        {"attempts_ledger": []},
        {"prior_elapsed_seconds": 1},
        {"prior_worker_seconds_denominator": 1},
        {"elapsed_seconds": 0},
        {"elapsed_seconds": True},
        {"worker_seconds_denominator": 601},
        {"cumulative_elapsed_seconds": 101},
        {"cumulative_worker_seconds_denominator": 601},
        {"registered_stop_reason": resume.REGISTERED_RATE_STOP},
        {"stop_reason": resume.REGISTERED_RATE_STOP},
        {"post_slowdown_stop_reason": resume.REGISTERED_RATE_STOP},
        {"rate_sealed_path": "/foreign"},
    ],
)
def test_bad_terminal_holds(tmp_path, change) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, terminal_change=change)
    with pytest.raises(ValueError, match="HOLD"):
        resume.attempt_chain(pack, control)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-terminal",
        "export-mismatch",
        "orphan",
        "foreign-output",
        "cap",
        "rate-hash",
        "terminal-sidecar",
    ],
)
def test_incomplete_or_uncharged_chain_holds(tmp_path, failure) -> None:
    pack, control = pack_fixture(tmp_path)
    output = add_attempt(
        pack,
        control,
        1,
        terminal=failure != "missing-terminal",
        elapsed=resume.CAP if failure == "cap" else 100,
        export_change={"workers": 12} if failure == "export-mismatch" else None,
    )
    state = pack / "d1024-state-mac-01"
    if failure == "orphan":
        publish_json(state / "attempt-foreign.terminal.json", {})
    if failure == "foreign-output":
        (pack / "d1024-output-mac-03").mkdir()
    if failure == "rate-hash":
        (state / "attempt-01.rate-sealed.json").write_text("tampered")
    if failure == "terminal-sidecar":
        (state / "attempt-01.terminal.json.complete").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        resume.publish_ledger(pack, [output], pack / "resume/verified-ledger.json")
    assert not (pack / "resume/verified-ledger.json").exists()


def test_union_preserves_partial_lives_and_sessions_without_aggregation(
    tmp_path, monkeypatch
) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    second = add_attempt(pack, control, 2, prior=100)
    life(first, control, 0, 0)
    life(first, control, 1, 0)
    life(second, control, 0, 1)
    publish_session(
        second,
        control["namespace"],
        control["allocation"],
        {
            "schema": "bl-labels-f0-session/v1",
            "status": "COMPLETE",
            "namespace": control["namespace"],
            "allocation_id": "p0",
            "allocation": control["allocation"][0],
            "counters": {"deaths": 2},
        },
    )
    (second / "lives/p2--life-000000.json").write_text("unsealed partial bytes")
    from research.bl_labels import exports

    monkeypatch.setattr(exports, "aggregate_attempts", lambda *_: pytest.fail("new look"))
    ledger = pack / "resume/verified-ledger.json"
    value = resume.publish_ledger(pack, [first, second], ledger)
    assert len(load_ledger(ledger, control["namespace"], control["allocation"])) == 4
    assert "mac_aggregate" not in value
    _, _, jobs = mac_driver._prepare_jobs(control, ledger)
    assert [(job.allocation["allocation_id"], job.skip_lives) for job in jobs] == [
        ("p1", {0}),
        ("p2", set()),
    ]
    resume.verify_union(ledger, control, [first, second])
    with pytest.raises(ValueError, match="attempt set"):
        resume.verify_union(ledger, control, [first])
    assert resume.attempt_chain(pack, control)[1] == 200


@pytest.mark.parametrize("failure", ["duplicate", "foreign", "omitted"])
def test_union_refuses_duplicate_foreign_or_omitted_units(tmp_path, failure) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    second = add_attempt(pack, control, 2, prior=100)
    life(first, control, 0, 0)
    if failure == "duplicate":
        life(second, control, 0, 0)
    if failure == "foreign":
        life(second, {**control, "namespace": "foreign"}, 1, 0)
    ledger = pack / "resume/verified-ledger.json"
    with pytest.raises(ValueError):
        resume.publish_ledger(pack, [first] if failure == "omitted" else [first, second], ledger)
    assert not Path(str(ledger) + ".complete").exists()


def launcher_fixture(tmp_path, monkeypatch):
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1)
    monkeypatch.setattr(resume, "PACK", pack)
    monkeypatch.setattr(resume.d1024_w12, "_prepare_science_root", lambda *_: None)
    monkeypatch.setattr(resume.d1024_w12, "_validate_inputs", lambda *_: (control, 6, [], True))
    monkeypatch.setattr(mac_gates, "require_admission", lambda *_: None)
    monkeypatch.setattr(frozen_factory, "_validate", lambda *_: None)
    monkeypatch.setattr(mac_driver, "_timing_armed", lambda _: False)
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _: {})
    args = (
        pack,
        tmp_path / "eq",
        tmp_path / "cal",
        tmp_path / "old",
        tmp_path / "f4/control.json",
        tmp_path / "f4",
        Path("b8"),
        Path("frp3"),
    )
    return pack, control, args


def test_launcher_reuses_union_after_check_then_advances_after_preemption(
    tmp_path, monkeypatch
) -> None:
    pack, control, args = launcher_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(resume.subprocess, "run", lambda cmd, **_: calls.append(cmd))
    monkeypatch.setattr(resume.os, "execv", lambda _py, cmd: calls.append(cmd))
    resume.launch(*args, check=True)
    ledger = pack / "d1024-resume-02/verified-ledger.json"
    before = ledger.read_bytes()
    resume.launch(*args)
    assert ledger.read_bytes() == before
    assert calls[-2][-1] == "--check" and "--check" not in calls[-1]
    add_attempt(pack, control, 2, prior=100)
    resume.launch(*args)
    command = calls[-1]
    assert command[command.index("--output") + 1] == str(pack / "d1024-output-mac-03")
    assert command[command.index("--state-dir") + 1] == str(pack / "d1024-state-mac-01")
    assert command[command.index("--control") + 1] == str(pack / "d1024-admitted.json")
    assert command[command.index("--deadline-seconds") + 1] == "144000"
    assert command[command.index("--workers") + 1] == "6"


def test_launcher_check_failure_never_executes(tmp_path, monkeypatch) -> None:
    pack, _, args = launcher_fixture(tmp_path, monkeypatch)

    def refuse(cmd, **_):
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(resume.subprocess, "run", refuse)
    monkeypatch.setattr(resume.os, "execv", lambda *_: pytest.fail("exec on check failure"))
    with pytest.raises(subprocess.CalledProcessError):
        resume.launch(*args)
    assert not (pack / "d1024-output-mac-02").exists()


def test_real_coordinator_two_continuations_skip_verified_lives_and_charge_original_cap(
    tmp_path, monkeypatch
) -> None:
    from research.bl_labels import factory

    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    old_life = life(first, control, 0, 0)
    before = old_life.read_bytes()
    ledger2 = pack / "resume2/verified-ledger.json"
    resume.publish_ledger(pack, [first], ledger2)
    _patch_inline_runtime(monkeypatch)
    monkeypatch.setenv("F0_COMMIT", resume.SCIENCE_PIN)
    monkeypatch.setenv("BL_FROZEN_FACTORY_WORKER", "1")
    monkeypatch.setattr(mac_gates, "require_admission", lambda *_: None)
    # Stand in only for frozen activation; coordinator/accounting/publication are real.
    monkeypatch.setattr(
        frozen_factory,
        "activate",
        lambda *_: SimpleNamespace(
            FactoryInvalid=factory.FactoryInvalid,
            FactoryPreempted=factory.FactoryPreempted,
            collect_unit=factory.collect_unit,
        ),
    )
    timing = tmp_path / "timing"
    _install_fake_factory(monkeypatch, timing)
    output2 = pack / "d1024-output-mac-02"
    second = mac_driver.run(
        control,
        ledger2,
        Path("b8"),
        Path("frp3"),
        output2,
        workers=6,
        deadline_seconds=resume.CAP,
        state_dir=pack / "d1024-state-mac-01",
        timing_armed=timing,
    )
    publish_json(output2 / "runtime-result.json", second)
    assert second["status"] == "PREEMPTED"
    assert second["prior_elapsed_seconds"] == 100
    assert second["prior_worker_seconds_denominator"] == 600
    assert not (output2 / "lives/p0--life-000000.json").exists()
    ledger3 = pack / "resume3/verified-ledger.json"
    resume.publish_ledger(pack, [first, output2], ledger3)
    timing.unlink()
    _install_fake_factory(monkeypatch)
    # Inline workers share this test process; production workers are isolated.
    monkeypatch.setenv("F0_COMMIT", resume.SCIENCE_PIN)
    output3 = pack / "d1024-output-mac-03"
    third = mac_driver.run(
        control,
        ledger3,
        Path("b8"),
        Path("frp3"),
        output3,
        workers=6,
        deadline_seconds=resume.CAP,
        state_dir=pack / "d1024-state-mac-01",
        timing_armed=timing,
    )
    publish_json(output3 / "runtime-result.json", third)
    assert third["status"] == "COMPLETE"
    assert third["prior_elapsed_seconds"] == 100 + second["elapsed_seconds"]
    assert third["prior_worker_seconds_denominator"] == 600 + 6 * second["elapsed_seconds"]
    assert third["cumulative_elapsed_seconds"] < resume.CAP
    assert old_life.read_bytes() == before
    with pytest.raises(ValueError, match="non-preempted"):
        resume.attempt_chain(pack, control)


def test_shell_entrypoint_help_and_original_checkpoint_default(tmp_path) -> None:
    script = Path(resume.__file__).with_name("RUN_D1024_RESUME.sh")
    result = subprocess.run(
        ["/bin/bash", str(script), "--help"],
        text=True,
        capture_output=True,
        env={**os.environ, "F4_CONTROL": "sealed", "F4_INPUTS": "sealed"},
        check=True,
    )
    assert "--check" in result.stdout and "--pack" in result.stdout
    assert "frp3_m3_s12_u60000_20261005.pth" in script.read_text()
