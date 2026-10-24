"""Synthetic, single-process continuation proof; never run the real factory."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from research.bl_labels import d256_resume as resume
from research.bl_labels import frozen_factory, mac_driver
from research.bl_labels.exports import (
    _verify,
    load_ledger,
    publish_json,
    publish_life,
    publish_session,
)
from research.bl_labels.test_mac_integration import (
    _install_fake_factory,
    _patch_inline_runtime,
)
from research.bl_labels.test_w12_calibration import _receipt


@pytest.fixture(autouse=True)
def isolated_worker_environment(monkeypatch):
    for key in ("F0_COMMIT", "F1_COMMIT", "BL_FROZEN_FACTORY_WORKER"):
        prior = os.environ.get(key)
        monkeypatch.setenv(key, prior or "")
        if prior is None:
            monkeypatch.delenv(key)


def pack_fixture(tmp_path: Path) -> tuple[Path, dict]:
    pack = tmp_path / "pack"
    control = {
        "namespace": "d256-synthetic",
        "source_commit": resume.OLD_PIN,
        "operational_canary_rate": 48.0,
        "allocation": [
            {"allocation_id": f"p{i}", "seed": i, "stream": "watch6", "horizon": 2}
            for i in range(3)
        ],
    }
    publish_json(pack / "d256-control.json", control)
    # Canary-B is deliberately foreign, to catch the original hardwired bug.
    publish_json(pack / "control.json", {"namespace": "foreign-canary", "allocation": []})
    admitted = copy.deepcopy(control)
    admitted["mac_gate"] = {
        "stage": "d256",
        "status": "APPROVED",
        "package_root": str(pack),
        "original_control_sha256": _verify(pack / "d256-control.json"),
    }
    publish_json(pack / "d256-admitted.json", admitted)
    return pack, admitted


def add_attempt(
    pack: Path,
    control: dict,
    index: int,
    *,
    elapsed: float = 100,
    prior: float = 0,
    workers: int = 6,
    prior_worker: float = 0,
    status: str = "PREEMPTED",
    terminal: bool = True,
    legacy: bool = False,
    stop_reason: str | None = None,
) -> Path:
    state = pack / "d256-state-mac-01"
    output = pack / f"d256-output-mac-{index:02d}"
    start_path = state / f"attempt-{index:02d}.start.json"
    start = {
        "namespace": control["namespace"],
        "workers": workers,
        "allocation_sha256": resume.canonical_sha(control["allocation"]),
        "control_sha256": resume.canonical_sha(control),
        "source_commit": resume.OLD_PIN,
        "deadline_seconds": resume.CAP,
        "original_wall_cap_seconds": resume.CAP,
        "output": str(output),
        "state_dir": str(state),
        "platform_attestation": {"driver_sha256": resume.OLD_DRIVER_SHA256},
    }
    if not legacy:
        start.update(owner_w12_ok=workers == 12, w12_calibration_receipt_sha256="c" * 64)
    publish_json(start_path, start)
    os.utime(start_path, ns=(index * 10**9, index * 10**9))
    output.mkdir(exist_ok=True)
    if not terminal:
        return output
    sealed = state / f"attempt-{index:02d}.rate-sealed.json"
    publish_json(sealed, {"events": []})
    result = {
        "namespace": control["namespace"],
        "workers": workers,
        "status": status,
        "invalid_reason": None,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "prior_elapsed_seconds": prior,
        "cumulative_elapsed_seconds": prior + elapsed + (0.000001 if legacy else 0),
        "worker_seconds_denominator": workers * elapsed + (0.00001 if legacy else 0),
        "prior_worker_seconds_denominator": prior_worker,
        "cumulative_worker_seconds_denominator": prior_worker + workers * elapsed,
        "output": str(output),
        "state_dir": str(state),
        "start_receipt": start_path.name,
        "start_sha256": _verify(start_path),
        "attempts_ledger": [f"attempt-{i:02d}.terminal.json" for i in range(1, index + 1)],
        "rate_sealed_path": str(sealed),
        "w12_calibration_receipt_sha256": "c" * 64,
        "first_hour_life_seconds": [2.0, 3.0],
    }
    publish_json(state / f"attempt-{index:02d}.terminal.json", result)
    publish_json(output / "runtime-result.json", {**result, "mac_lease": {"released": True}})
    return output


def life(output: Path, control: dict, parent: int, number: int) -> Path:
    item = control["allocation"][parent]
    return publish_life(
        output,
        control["namespace"],
        control["allocation"],
        {
            "schema": "bl-labels-f0-life/v1",
            "namespace": control["namespace"],
            "allocation": item,
            "allocation_id": item["allocation_id"],
            "life": number,
            "status": "COMPLETE",
            "rows": [{"opaque_outcome": "must never aggregate"}],
        },
    )


def test_union_of_two_attempts_is_d256_portable_and_skips_completed_units(
    tmp_path, monkeypatch
) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, legacy=True)
    second = add_attempt(pack, control, 2, prior=100, workers=12, prior_worker=600)
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
    # Unsealed interrupted output cannot enter the union.
    (second / "lives/p2--life-000000.json").write_text("incomplete bytes")
    ledger = tmp_path / "union/verified-ledger.json"
    value = resume.publish_ledger(pack, [first, second], ledger)
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    assert len(names) == 4
    assert value["namespace"] == control["namespace"]
    assert "mac_aggregate" not in value
    _, _, jobs = mac_driver._prepare_jobs(control, ledger)
    assert [job.allocation["allocation_id"] for job in jobs] == ["p1", "p2"]
    assert jobs[0].skip_lives == {0}
    assert jobs[1].skip_lives == set()
    # Execute the original frozen reader itself, without a worker or simulation.
    source = subprocess.check_output(
        ["git", "show", f"{resume.OLD_PIN}:research/bl_labels/mac_driver.py"], cwd=resume.ROOT
    )
    exports_source = subprocess.check_output(
        ["git", "show", f"{resume.OLD_PIN}:research/bl_labels/exports.py"], cwd=resume.ROOT
    )
    assert exports_source == (resume.ROOT / "research/bl_labels/exports.py").read_bytes()
    frozen = ModuleType("research.bl_labels._original_reader_test")
    frozen.__file__ = str(resume.ROOT / "research/bl_labels/mac_driver.py")
    monkeypatch.setitem(sys.modules, frozen.__name__, frozen)
    exec(compile(source, "original_mac_driver.py", "exec"), frozen.__dict__)
    _, _, original_jobs = frozen._prepare_jobs(control, ledger)
    assert [(j.allocation, j.skip_lives) for j in original_jobs] == [
        (j.allocation, j.skip_lives) for j in jobs
    ]


def test_synchronous_shell_entrypoint_publishes_synthetic_union(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, legacy=True)
    life(first, control, 0, 0)
    out = tmp_path / "entrypoint-union/verified-ledger.json"
    result = subprocess.run(
        [
            "/bin/bash",
            str(resume.ROOT / "research/bl_labels/RUN_D256_RESUME.sh"),
            "--pack",
            str(pack),
            "--ledger-out",
            str(out),
        ],
        cwd=resume.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"status": "PUBLISHED", "out": str(out)}
    assert len(load_ledger(out, control["namespace"], control["allocation"])) == 1
    assert not (pack / "d256-output-mac-02").exists()


@pytest.mark.parametrize("failure", ["missing-terminal", "INVALID", "duplicate", "foreign"])
def test_union_refusals_publish_no_ledger(tmp_path, failure) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    life(first, control, 0, 0)
    second = add_attempt(
        pack,
        control,
        2,
        prior=100,
        prior_worker=600,
        terminal=failure != "missing-terminal",
        status="INVALID" if failure == "INVALID" else "PREEMPTED",
    )
    if failure == "duplicate":
        life(second, control, 0, 0)
    if failure == "foreign":
        foreign = copy.deepcopy(control)
        foreign["namespace"] = "foreign"
        life(second, foreign, 1, 0)
    ledger = tmp_path / "union/verified-ledger.json"
    with pytest.raises(ValueError):
        resume.publish_ledger(pack, [first, second], ledger)
    assert not ledger.exists()
    assert not Path(str(ledger) + ".complete").exists()


def test_cap_exhaustion_holds_before_publication(tmp_path, monkeypatch) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, elapsed=resume.CAP)
    monkeypatch.setattr(resume, "PACK", pack)
    monkeypatch.setattr(resume, "ensure_frozen_tree", lambda _: pytest.fail("created frozen tree"))
    with pytest.raises(ValueError, match="exhausted.*HOLD"):
        resume.launch(
            pack, tmp_path / "science", Path("receipt"), Path("owner"), Path("b8"), Path("frp3")
        )
    assert not (pack / "d256-output-mac-02").exists()
    assert not (pack / "d256-resume-02").exists()


def test_registered_stop_cannot_be_relabelled_as_operational_preemption(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, stop_reason=resume.REGISTERED_RATE_STOP)
    with pytest.raises(ValueError, match="registered D256 stopping rule.*HOLD"):
        resume.attempt_chain(pack, control)


@pytest.mark.parametrize("status,ratio,width", [("PASS", 1.2, 12), ("FAIL", 1.4, 6)])
def test_sealed_worker_selection(tmp_path, monkeypatch, status, ratio, width) -> None:
    b8, frp3 = tmp_path / "b8", tmp_path / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    receipt = _receipt(
        tmp_path,
        monkeypatch,
        status=status,
        w12_w6_wall_ratio=ratio,
        b8_sha256=resume.w12_calibration.sha(b8),
        frp3_sha256=resume.w12_calibration.sha(frp3),
    )
    selected, flags = resume.select_workers(receipt, Path("owner"), b8, frp3)
    assert selected == width
    assert bool(flags) == (width == 12)
    b8.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoints differ"):
        resume.select_workers(receipt, Path("owner"), b8, frp3)


def test_missing_calibration_is_w6_but_unsealed_or_hold_is_refused(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing/receipt.json"
    assert resume.select_workers(missing, Path("owner"), Path("b8"), Path("frp3")) == (6, [])
    receipt = _receipt(tmp_path, monkeypatch, status="HOLD")
    with pytest.raises(ValueError, match="HOLD"):
        resume.select_workers(receipt, Path("owner"), Path("b8"), Path("frp3"))
    Path(str(receipt) + ".complete").unlink()
    with pytest.raises(ValueError, match="sidecars"):
        resume.select_workers(receipt, Path("owner"), Path("b8"), Path("frp3"))


def test_legacy_clock_tolerance_requires_original_driver_and_is_bounded() -> None:
    start = {
        "workers": 6,
        "source_commit": resume.OLD_PIN,
        "platform_attestation": {"driver_sha256": resume.OLD_DRIVER_SHA256},
    }
    resume.verify_worker_charge(start, {"worker_seconds_denominator": 600.00001}, 100)
    for modified, charge in [
        (start, 600.1),
        ({**start, "owner_w12_ok": False}, 600.00001),
        ({**start, "platform_attestation": {}}, 600.00001),
    ]:
        with pytest.raises(ValueError, match="worker-second"):
            resume.verify_worker_charge(modified, {"worker_seconds_denominator": charge}, 100)


def test_frozen_admission_uses_separate_original_process(tmp_path, monkeypatch) -> None:
    pack, control = pack_fixture(tmp_path)
    monkeypatch.setattr(resume, "PACK", pack)
    calls = []
    root = tmp_path / "frozen"
    monkeypatch.setattr(frozen_factory, "_validate", lambda r, p: calls.append((r, p)))
    monkeypatch.setattr(resume.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    resume.verify_frozen_admission(control, root)
    assert calls[0] == (root, resume.OLD_PIN)
    cmd, kwargs = calls[1]
    assert "require_admission" in cmd[-1]
    assert kwargs["cwd"] == root
    assert json.loads(kwargs["input"]) == control
    assert kwargs["env"]["PYTHONPATH"] == str(root)


def test_launcher_reuses_union_after_check_refusal_and_advances_after_preempt(
    tmp_path, monkeypatch
) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    life(first, control, 0, 0)
    monkeypatch.setattr(resume, "PACK", pack)
    monkeypatch.setenv("F0_COMMIT", resume.OLD_PIN)
    monkeypatch.setenv("F1_COMMIT", resume.OLD_PIN)
    monkeypatch.setattr(resume, "ensure_frozen_tree", lambda _: None)
    monkeypatch.setattr(resume, "verify_frozen_admission", lambda *_: None)
    monkeypatch.setattr(resume, "select_workers", lambda *_: (6, []))
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _: {})
    monkeypatch.setattr(mac_driver, "_timing_armed", lambda _: False)
    calls = []
    monkeypatch.setattr(resume.subprocess, "run", lambda cmd, **_: calls.append(cmd))
    monkeypatch.setattr(resume.os, "execv", lambda _py, cmd: calls.append(cmd))
    args = (pack, tmp_path / "science", Path("receipt"), Path("owner"), Path("b8"), Path("frp3"))
    resume.launch(*args, check=True)
    before = (pack / "d256-resume-02/verified-ledger.json").read_bytes()
    resume.launch(*args)
    assert (pack / "d256-resume-02/verified-ledger.json").read_bytes() == before
    assert calls[-2][-1] == "--check" and "--check" not in calls[-1]
    assert calls[-1][calls[-1].index("--output") + 1] == str(pack / "d256-output-mac-02")
    add_attempt(pack, control, 2, prior=100, prior_worker=600)
    resume.launch(*args)
    assert calls[-1][calls[-1].index("--output") + 1] == str(pack / "d256-output-mac-03")
    assert calls[-1][calls[-1].index("--state-dir") + 1] == str(pack / "d256-state-mac-01")


def test_original_driver_resume_skips_life_then_resumes_again_at_w6(tmp_path, monkeypatch) -> None:
    """Fake frozen collector, real coordinator: original W6 -> W12 -> W6."""
    from research.bl_labels import factory

    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, legacy=True)
    original = life(first, control, 0, 0)
    before = original.read_bytes()
    ledger = pack / "union2/verified-ledger.json"
    resume.publish_ledger(pack, [first], ledger)
    _patch_inline_runtime(monkeypatch)
    monkeypatch.setenv("F0_COMMIT", resume.OLD_PIN)
    monkeypatch.setenv("BL_FROZEN_FACTORY_WORKER", "1")
    monkeypatch.setattr(resume, "verify_frozen_admission", lambda *_: None)
    monkeypatch.setattr(mac_driver, "_require_w12_authorization", lambda *_: "c" * 64)
    monkeypatch.setattr(resume, "require_w12", lambda *_: None)
    timing = tmp_path / "timing"
    _install_fake_factory(monkeypatch, timing)
    # Stand in only for activation; all scheduling/publication/skip logic is real.
    monkeypatch.setattr(
        frozen_factory,
        "activate",
        lambda *_: SimpleNamespace(
            FactoryInvalid=factory.FactoryInvalid,
            FactoryPreempted=factory.FactoryPreempted,
            collect_unit=factory.collect_unit,
        ),
    )
    output2 = pack / "d256-output-mac-02"
    second = mac_driver.run(
        control,
        ledger,
        Path("b8"),
        Path("frp3"),
        output2,
        workers=12,
        owner_w12_ok=True,
        w12_calibration_receipt=Path("calibration"),
        deadline_seconds=resume.CAP,
        state_dir=pack / "d256-state-mac-01",
        timing_armed=timing,
        d256_frozen_root=tmp_path / "frozen",
    )
    publish_json(output2 / "runtime-result.json", second)
    assert not (output2 / "lives/p0--life-000000.json").exists()
    assert second["status"] == "PREEMPTED"
    assert second["prior_elapsed_seconds"] == 100
    assert second["prior_recorded_worker_seconds_denominator"] == 600.00001
    assert second["first_hour_life_seconds_by_workers"]["6"] == [2.0, 3.0]
    assert all(x < 2.0 for x in second["first_hour_life_seconds"])
    assert second["baseline_segment_elapsed_seconds"] == second["elapsed_seconds"]
    ledger3 = pack / "union3/verified-ledger.json"
    resume.publish_ledger(pack, [first, output2], ledger3)
    timing.unlink()
    _install_fake_factory(monkeypatch)
    output3 = pack / "d256-output-mac-03"
    third = mac_driver.run(
        control,
        ledger3,
        Path("b8"),
        Path("frp3"),
        output3,
        workers=6,
        deadline_seconds=resume.CAP,
        state_dir=pack / "d256-state-mac-01",
        timing_armed=timing,
        d256_frozen_root=tmp_path / "frozen",
    )
    publish_json(output3 / "runtime-result.json", third)
    assert third["status"] == "COMPLETE"
    assert third["prior_worker_seconds_denominator"] == 600 + 12 * second["elapsed_seconds"]
    assert all(x < 2.0 for x in third["first_hour_life_seconds"])
    assert third["baseline_segment_elapsed_seconds"] == third["elapsed_seconds"]
    assert original.read_bytes() == before
    resume.attempt_chain(pack, control)


def test_registered_rate_stop_survives_later_operational_guard(tmp_path, monkeypatch) -> None:
    """Real coordinator reaches its stop, then its host guard overwrites the reason."""
    from research.bl_labels import factory

    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, elapsed=21601, legacy=True)
    ledger = pack / "union/verified-ledger.json"
    resume.publish_ledger(pack, [first], ledger)
    _patch_inline_runtime(monkeypatch)
    _install_fake_factory(monkeypatch)
    monkeypatch.setenv("F0_COMMIT", resume.OLD_PIN)
    monkeypatch.setattr(resume, "verify_frozen_admission", lambda *_: None)
    monkeypatch.setattr(
        frozen_factory,
        "activate",
        lambda *_: SimpleNamespace(
            FactoryInvalid=factory.FactoryInvalid,
            FactoryPreempted=factory.FactoryPreempted,
            collect_unit=factory.collect_unit,
        ),
    )
    output = pack / "d256-output-mac-02"

    def guard(_output):
        if list((output / "lives").glob("*.json.complete")):
            raise mac_driver.MacGuardError("synthetic host failure after a completed life")
        return {"fixture": True}

    monkeypatch.setattr(mac_driver, "_guard_snapshot", guard)
    result = mac_driver.run(
        control,
        ledger,
        Path("b8"),
        Path("frp3"),
        output,
        workers=6,
        state_dir=pack / "d256-state-mac-01",
        deadline_seconds=resume.CAP,
        timing_armed=tmp_path / "timing",
        d256_frozen_root=tmp_path / "frozen",
    )
    publish_json(output / "runtime-result.json", result)
    assert result["status"] == "PREEMPTED"
    assert result["stop_reason"].startswith("operational guard:")
    assert result["registered_stop_reason"] == resume.REGISTERED_RATE_STOP
    with pytest.raises(ValueError, match="registered D256 stopping rule.*HOLD"):
        resume.attempt_chain(pack, control)
