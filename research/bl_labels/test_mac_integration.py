"""Independent, outcome-blind Mac factory integration checks.

The fixtures below stand in for the exact factory.  They exercise the
orchestrator's unit publication, resume identity, and preemption ordering
without opening a gameplay row or depending on Darwin-only host commands.
"""

from __future__ import annotations

import hashlib
import json
import queue
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_labels import mac_driver
from research.bl_labels.exports import (
    aggregate_attempts,
    empty_ledger,
    load_ledger,
    publish_json,
)
from research.bl_labels.f1_ruler import compare_mac_configs


def _allocation() -> list[dict[str, object]]:
    return [{"allocation_id": "watch6-000", "seed": 73, "stream": "watch6", "horizon": 10}]


def _allocations(count: int = 6) -> list[dict[str, object]]:
    return [
        {
            "allocation_id": f"watch6-{index:03d}",
            "seed": 73 + index,
            "stream": "watch6",
            "horizon": 10,
        }
        for index in range(count)
    ]


def _opaque_life(namespace: str, allocation: dict[str, object], life: int) -> dict[str, object]:
    """Make a deterministic payload whose semantic body is treated as opaque."""
    public_allocation = {
        key: value
        for key, value in allocation.items()
        if key not in {"publish_life", "progress", "skip_lives", "should_preempt"}
    }
    semantic = {
        "life": life,
        "allocation_id": public_allocation["allocation_id"],
        "decision_tape_digest": hashlib.sha256(f"{namespace}/{life}".encode()).hexdigest(),
    }
    return {
        "schema": "bl-labels-f0-life/v1",
        "status": "COMPLETE",
        "namespace": namespace,
        "allocation_id": allocation["allocation_id"],
        "allocation": public_allocation,
        "life": life,
        "rows": [],
        "semantic_label_sha256": hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "semantic": semantic,
    }


class _InlineProcess:
    _next_pid = 1000

    def __init__(self, target, args) -> None:
        self.target = target
        self.args = args
        self.pid = _InlineProcess._next_pid
        _InlineProcess._next_pid += 1
        self.exitcode = None
        self._finished_at: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.target(*self.args)
        except BaseException:
            self.exitcode = 1
            raise
        else:
            self.exitcode = 0
        finally:
            self._finished_at = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive() or (
            self._finished_at is not None and time.monotonic() - self._finished_at < 0.5
        )

    def join(self) -> None:
        self._thread.join()


class _InlineContext:
    def Queue(self):
        return queue.Queue()

    def Event(self):
        return SimpleNamespace(is_set=lambda: False, set=lambda: None)

    def Process(self, target, args):
        return _InlineProcess(target, args)


def _patch_inline_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mac_driver.os, "setpriority", lambda *_args: None)
    monkeypatch.setattr(mac_driver.os, "getpriority", lambda *_args: 10)
    monkeypatch.setattr(mac_driver.mp, "get_context", lambda _name: _InlineContext())
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _output: {"fixture": True})
    monkeypatch.setattr(mac_driver, "_rss_many", lambda _pids: {pid: 0 for pid in _pids})


def _install_fake_factory(
    monkeypatch: pytest.MonkeyPatch, arm_after_first: Path | None = None
) -> None:
    from research.bl_labels import factory

    class FakePreempted(RuntimeError):
        pass

    def collect_unit(allocation, _b8, _frp3, namespace):
        publish = allocation["publish_life"]
        skip = set(allocation.get("skip_lives", ()))
        for life in range(2):
            if life in skip:
                continue
            publish(_opaque_life(namespace, allocation, life))
            if arm_after_first is not None and life == 0:
                arm_after_first.write_text("timing\n")
            if allocation["should_preempt"]():
                raise FakePreempted("fixture preemption at life boundary")
        allocation["publish_life"](
            {
                "schema": "bl-labels-f0-session/v1",
                "status": "COMPLETE",
                "namespace": namespace,
                "allocation_id": allocation["allocation_id"],
                "allocation": {
                    key: value
                    for key, value in allocation.items()
                    if key not in {"publish_life", "progress", "skip_lives", "should_preempt"}
                },
                "rows": [],
                "counters": {"deaths": 2},
            }
        )

    monkeypatch.setattr(factory, "FactoryPreempted", FakePreempted)
    monkeypatch.setattr(factory, "collect_unit", collect_unit)


def _install_real_factory_fake_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``factory.collect_unit`` real while replacing only its simulator seam."""
    from types import SimpleNamespace

    from research.bl_labels import factory

    class FakeDriver:
        def __init__(self) -> None:
            self.decision = 0
            self.life = 0
            self.last_root_q = np.zeros(6, dtype=np.float32)
            self.sim = SimpleNamespace(
                alive=np.array([[True]], dtype=np.bool_),
                frame=np.array([0], dtype=np.int64),
                length=np.array([[9]], dtype=np.int64),
            )

        @classmethod
        def restore(cls, _payload: bytes) -> "FakeDriver":
            return cls()

        def prepare(self) -> None:
            self.sim.alive[0, 0] = True

        def legal_and_bare(self) -> tuple[tuple[int, ...], int]:
            return (0, 1), 0

        def snapshot(self) -> bytes:
            return b"real-collect-unit-fake-snapshot"

        def step(self, _forced: int | None = None) -> dict[str, object]:
            self.sim.alive[0, 0] = False
            self.decision = 1
            return {
                "decision": 0,
                "actions": [0],
                "hero_length": 9,
                "death_cause": "wall",
            }

    def branch(_snapshot: bytes, action: int, _depth: int, **_kwargs: object) -> dict[str, object]:
        survives = action == 1
        return {
            "survives": survives,
            "frames_executed": factory.HORIZON if survives else 1,
            "death_decision": None if survives else 0,
            "final_length": 9,
            "mass_integral": 9.0 if survives else 9 / factory.HORIZON,
            "extended": {"survives": survives},
        }

    monkeypatch.setattr(factory, "make_driver", lambda *_args: FakeDriver())
    monkeypatch.setattr(
        factory,
        "digest",
        lambda path: factory.ceiling.B8_SHA256 if "b8" in str(path) else factory.FRP3,
    )
    monkeypatch.setattr(factory, "_raw_observation", lambda _driver: {})
    monkeypatch.setattr(factory, "_env_digest", lambda _sim: "env-digest")
    monkeypatch.setattr(factory, "_digest_driver", lambda *_args: "d" * 64)
    monkeypatch.setattr(factory, "driver_stats", lambda _driver: {})
    monkeypatch.setattr(factory, "attest", lambda _stats: None)
    monkeypatch.setattr(factory, "_restore_root", lambda *_args, **_kwargs: FakeDriver())
    monkeypatch.setattr(factory, "_branch", branch)
    monkeypatch.setattr(factory, "_selection_hash", lambda *_parts: "0" * 64)


def _run_fixture(
    monkeypatch: pytest.MonkeyPatch,
    control,
    ledger,
    b8,
    frp3,
    output,
    *,
    timing,
    state,
    workers=1,
):
    return mac_driver.run(
        control,
        ledger,
        b8,
        frp3,
        output,
        workers=workers,
        timing_armed=timing,
        deadline_seconds=1200,
        state_dir=state,
    )


def test_preempted_life_is_immutable_and_resume_skips_verified_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A boundary preemption preserves a complete life and resumes by ledger ID."""
    _patch_inline_runtime(monkeypatch)
    timing = tmp_path / "mac-timing-armed"
    control = {"namespace": "f0-fixture", "allocation": _allocation()}
    output = tmp_path / "preempted"
    initial_ledger = tmp_path / "empty-ledger.json"
    b8, frp3 = tmp_path / "b8", tmp_path / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    publish_json(initial_ledger, empty_ledger(control["namespace"], control["allocation"]))

    _install_fake_factory(monkeypatch, timing)
    first = _run_fixture(
        monkeypatch,
        control,
        initial_ledger,
        b8,
        frp3,
        output,
        timing=timing,
        state=tmp_path / "state",
    )
    assert first["status"] == "PREEMPTED", first["telemetry_sealed_no_outcomes"]
    life_path = output / "lives/watch6-000--life-000000.json"
    assert life_path.exists(), first["telemetry_sealed_no_outcomes"]
    first_bytes = life_path.read_bytes()
    assert life_path.with_name(life_path.name + ".complete").read_text() == "complete\n"

    ledger_root = tmp_path / "verified"
    ledger = ledger_root / "verified-ledger.json"
    aggregate_attempts([output], ledger, control["namespace"], control["allocation"])
    assert load_ledger(ledger, control["namespace"], control["allocation"]) == {
        "watch6-000--life-000000"
    }

    timing.unlink()
    _install_fake_factory(monkeypatch)
    resumed_output = tmp_path / "resumed"
    resumed = _run_fixture(
        monkeypatch,
        control,
        ledger,
        b8,
        frp3,
        resumed_output,
        timing=timing,
        state=tmp_path / "state",
    )
    assert resumed["status"] == "COMPLETE"
    assert life_path.read_bytes() == first_bytes
    ledger_after_root = tmp_path / "verified-after-resume"
    ledger_after_resume = ledger_after_root / "verified-ledger.json"
    aggregate_attempts(
        [output, resumed_output],
        ledger_after_resume,
        control["namespace"],
        control["allocation"],
    )
    assert load_ledger(ledger_after_resume, control["namespace"], control["allocation"]) == {
        "watch6-000--life-000000",
        "watch6-000--life-000001",
        "watch6-000--session",
    }

    baseline = tmp_path / "baseline"
    baseline_ledger = tmp_path / "baseline-empty-ledger.json"
    publish_json(baseline_ledger, empty_ledger(control["namespace"], control["allocation"]))
    _install_fake_factory(monkeypatch)
    _run_fixture(
        monkeypatch,
        control,
        baseline_ledger,
        b8,
        frp3,
        baseline,
        timing=tmp_path / "never-armed",
        state=tmp_path / "baseline-state",
    )
    assert (
        json.loads(life_path.read_text())["semantic_label_sha256"]
        == json.loads((baseline / "lives/watch6-000--life-000000.json").read_text())[
            "semantic_label_sha256"
        ]
    )


def _mac_cells() -> tuple[list[dict[str, object]], list[str]]:
    roots = [hashlib.sha256(f"root-{index}".encode()).hexdigest() for index in range(64)]
    records = [
        {
            "root_id": root,
            "digest": hashlib.sha256(f"digest-{root}".encode()).hexdigest(),
            "label_sha256": hashlib.sha256(f"label-{root}".encode()).hexdigest(),
            "state_digest": hashlib.sha256(f"state-{root}".encode()).hexdigest(),
            "bare_audit_mismatches": 0,
        }
        for root in roots
    ]
    cells = []
    for label, workers in zip(("one", "half", "all"), (1, 3, 6)):
        cells.append(
            {
                "status": "COMPLETE",
                "platform": "mac",
                "transport_probe": False,
                "schedule_label": label,
                "workers": workers,
                "root_count": 64,
                "roots_per_hour": 1000.0 * workers,
                "records": records,
            }
        )
    return cells, roots


def test_tiny_deterministic_f1_fixture_requires_w1_w3_w6_label_hash_identity() -> None:
    cells, roots = _mac_cells()
    result = compare_mac_configs(cells, roots)
    assert result["status"] == "PASS"
    assert result["workers"] == [1, 3, 6]
    assert result["digest_and_label_identity"] is True
    assert result["state_digest_identity"] is True


def test_mac_driver_runs_six_units_at_w1_w3_w6_with_stable_semantic_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Worker invariance must hold across actual driver scheduling runs."""
    _patch_inline_runtime(monkeypatch)
    allocations = _allocations()
    control = {"namespace": "f0-driver-invariance", "allocation": allocations}
    b8, frp3 = tmp_path / "b8", tmp_path / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    _install_fake_factory(monkeypatch)

    hashes_by_workers: dict[int, dict[tuple[str, int], str]] = {}
    for workers in (1, 3, 6):
        output = tmp_path / f"w{workers}" / "output"
        ledger = tmp_path / f"w{workers}-empty-ledger.json"
        publish_json(ledger, empty_ledger(control["namespace"], allocations))
        result = _run_fixture(
            monkeypatch,
            control,
            ledger,
            b8,
            frp3,
            output,
            timing=tmp_path / f"w{workers}-timing-never-armed",
            state=tmp_path / f"w{workers}" / "state",
            workers=workers,
        )
        assert result["status"] == "COMPLETE"
        assert result["workers"] == workers
        assert set(result["completed_parents"]) == {row["allocation_id"] for row in allocations}
        assert (
            sum(event["event"] == "admit" for event in result["telemetry_sealed_no_outcomes"]) == 6
        )
        verified_root = tmp_path / f"w{workers}-verified"
        verified = verified_root / "verified-ledger.json"
        aggregate_attempts([output], verified, control["namespace"], allocations)
        names = load_ledger(verified, control["namespace"], allocations)
        assert len(names) == 18  # 6 parents x (2 immutable lives + 1 session)
        hashes_by_workers[workers] = {
            (path.stem.split("--life-", 1)[0], int(path.stem.rsplit("--life-", 1)[1])): json.loads(
                path.read_text()
            )["semantic_label_sha256"]
            for path in sorted((output / "lives").glob("*--life-*.json"))
        }

    assert hashes_by_workers[1] == hashes_by_workers[3] == hashes_by_workers[6]


def test_driver_preemption_uses_real_factory_collect_unit_fake_driver_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boundary callback is exercised through the real F0 collector."""
    _patch_inline_runtime(monkeypatch)
    _install_real_factory_fake_driver(monkeypatch)
    control = {"namespace": "f0-real-collector", "allocation": _allocation()}
    timing = tmp_path / "mac-timing-armed"
    output = tmp_path / "real-collector-output"
    ledger = tmp_path / "real-collector-empty-ledger.json"
    b8, frp3 = tmp_path / "b8", tmp_path / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    publish_json(ledger, empty_ledger(control["namespace"], control["allocation"]))

    real_publish_life = mac_driver.publish_life

    def publish_and_arm(root, namespace, allocation, value):
        path = real_publish_life(root, namespace, allocation, value)
        timing.write_text("timing\n")
        return path

    monkeypatch.setattr(mac_driver, "publish_life", publish_and_arm)
    result = _run_fixture(
        monkeypatch,
        control,
        ledger,
        b8,
        frp3,
        output,
        timing=timing,
        state=tmp_path / "real-collector-state",
    )
    assert result["status"] == "PREEMPTED"
    assert result["completed_parents"] == []
    life = output / "lives/watch6-000--life-000000.json"
    assert life.exists()
    assert life.with_name(life.name + ".complete").read_text() == "complete\n"
    assert json.loads(life.read_text())["schema"] == "bl-labels-f0-life/v1"


def test_timing_preemption_stops_owned_child_before_lease_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timing arm raises only after the child is stopped and before lease release."""
    events: list[str] = []
    timing = tmp_path / "mac-timing-armed"

    class Child:
        pid = 2718
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("child-stopped")
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    child = Child()

    @contextmanager
    def fake_held(*_args, **_kwargs):
        events.append("lease-acquired")
        try:
            yield SimpleNamespace(receipt={"fixture": True})
        finally:
            events.append("lease-released")

    monkeypatch.setattr(mac_driver, "held", fake_held)
    monkeypatch.setattr(mac_driver.subprocess, "Popen", lambda _command, **_kwargs: child)
    monkeypatch.setattr(mac_driver, "_stop_command_group", lambda owned: owned.terminate())
    monkeypatch.setattr(
        mac_driver.os,
        "killpg",
        lambda _pid, _signal: (
            events.append("child-stopped"),
            setattr(child, "returncode", -15),
        ),
    )
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _path: {"fixture": True})

    def arm_on_sleep(_seconds):
        events.append("worker-running")
        timing.write_text("timing\n")

    monkeypatch.setattr(mac_driver.time, "sleep", arm_on_sleep)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mac_driver.py",
            "guard-command",
            "--state-dir",
            str(tmp_path / "state"),
            "--seconds",
            "60",
            "--timing-armed",
            str(timing),
            "--",
            "fixture-worker",
        ],
    )
    with pytest.raises(mac_driver.MacGuardError, match="preempted by timing arm"):
        mac_driver.guard_command()
    assert events.index("child-stopped") < events.index("lease-released")
    assert events[-1] == "lease-released"


@pytest.mark.parametrize("workers", [1, 3, 6])
def test_f1_fixture_uses_registered_worker_counts(workers: int) -> None:
    cells, _roots = _mac_cells()
    assert cells[{1: 0, 3: 1, 6: 2}[workers]]["workers"] == workers
