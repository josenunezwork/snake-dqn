"""Excluded frozen-collector identity fixture; no real gameplay or production data.

The actual frozen collection/selection/export code runs against the established
one-transition fake simulator seam. Clock metadata is deterministic. Native
companion admission/ownership is tested separately by the supervisor tests.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from research.bl_labels import frozen_factory
from research.bl_labels.exports import (
    build_ledger,
    empty_ledger,
    load_ledger,
    publish_json,
    publish_life,
)

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _bounded_native_control(root: Path):
    """Run a real admitted cat waiting on an excluded FIFO, then prove cleanup."""
    from research.compute.factory_companion import DarwinProcessTable, FactoryCompanionSupervisor
    from research.compute.factory_companion_argv import native_argv
    from research.compute.factory_companion_commands import validate_command
    from research.compute.factory_companion_policy import POLICY_SHA256

    root.mkdir()
    fifo = root / "excluded-empty-control.pipe"
    os.mkfifo(fifo)

    def fixture_table() -> dict:
        # Native ownership/accounting of the excluded fixture only. Global
        # unmanaged-app admission uses independent synthetic process fixtures;
        # this maintenance test must not adopt or signal the coding clients.
        table = DarwinProcessTable.rows()
        registry = subject._read() if subject.state_path.exists() else {}
        groups = {
            r["pgid"]
            for r in registry.get("registrations", {}).values()
            if r.get("class") == "light" and "pgid" in r
        }
        return {
            pid: row
            for pid, row in table.items()
            if pid == os.getpid()
            or row["pgid"] in groups
            or (row["ppid"] == os.getpid() and row["pgid"] == pid)
        }

    subject = FactoryCompanionSupervisor(root, process_table=fixture_table)
    subject.claim_monitor()
    subject.update_factory_state(
        identity="excluded-identity",
        protected=True,
        streams={},
        owned=[{"pid": os.getpid(), "start": fixture_table()[os.getpid()]["start"]}],
    )
    command = ["cat", str(fifo)]
    results: list[object] = []

    def run() -> None:
        try:
            results.append(
                subject.launch(
                    "light",
                    policy_sha=POLICY_SHA256,
                    lease_identity="excluded-identity",
                    worktree=str(ROOT),
                    argv=command,
                    approved=validate_command("light", command, str(ROOT)),
                )
            )
        except BaseException as error:
            results.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    writer = None
    deadline = time.monotonic() + 3
    try:
        while time.monotonic() < deadline:
            state = subject._read()
            live = [r for r in state["registrations"].values() if r.get("class") == "light"]
            if live and live[0].get("status") == "running":
                pid = live[0]["pid"]
                if Path(native_argv(pid)[0]).name == "cat":
                    try:
                        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    except OSError:
                        pass
                    else:
                        assert os.getpriority(os.PRIO_PROCESS, pid) >= 15
                        break
            if results:
                pytest.fail(f"native companion ended before fixture: {results!r}")
            time.sleep(0.01)
        assert writer is not None, "native companion failed to reach its owned command"
        yield subject
    finally:
        if writer is not None:
            os.close(writer)
        thread.join(timeout=5)
        assert not thread.is_alive(), "excluded companion wrapper failed to reap"
    assert results == [0], results
    state = subject._read()
    record = next(r for r in state["registrations"].values() if r.get("class") == "light")
    assert record["wait4_reaped"] is True and record["descendants_gone"] is True
    assert record["status"] == "completed"


def _frozen_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{frozen_factory.OLD_PIN}:research/bl_labels/factory.py"], cwd=ROOT
    )
    module = ModuleType("research.bl_labels._concurrency_frozen_fixture")
    module.__file__ = "excluded-frozen-factory-fixture.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@pytest.mark.skipif(sys.platform != "darwin", reason="native companion identity proof")
def test_frozen_life_bytes_survive_companion_control_and_preempt_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real frozen selector/export bytes, including its life-boundary stop/skip."""
    from research import bl_labels
    from research.bl_labels.test_mac_integration import _install_real_factory_fake_driver

    factory = _frozen_module(monkeypatch)
    monkeypatch.setattr(bl_labels, "factory", factory, raising=False)
    _install_real_factory_fake_driver(monkeypatch)
    # The old collector consumes bare/legality directly, before prepared packets.
    monkeypatch.setattr(factory, "time", SimpleNamespace(monotonic=lambda: 1.0))
    monkeypatch.setenv("F0_COMMIT", frozen_factory.OLD_PIN)
    namespace = "excluded-concurrency-identity-61900af2"
    allocation = [
        {"allocation_id": "excluded-000", "seed": 7, "stream": "gate-frozen", "horizon": 1}
    ]

    def collect(output: Path, *, preempt: bool, skip: set[int]) -> None:
        def publish(value: dict) -> None:
            # Only label payloads are compared; terminal/session receipts are
            # intentionally different after a skipped prefix.
            if value["schema"] == "bl-labels-f0-life/v1":
                publish_life(output, namespace, allocation, value)

        factory.collect_unit(
            {
                **allocation[0],
                "publish_life": publish,
                "skip_lives": skip,
                "should_preempt": lambda: preempt,
            },
            Path("b8"),
            Path("frp3"),
            namespace,
        )

    baseline = tmp_path / "baseline"
    collect(baseline, preempt=False, skip=set())
    accompanied = tmp_path / "accompanied"
    # Native companion metadata stays in its operational registry. It cannot
    # enter the frozen allocation/label bytes. This is a maintenance fixture.
    with _bounded_native_control(tmp_path / "companion-operations"):
        with pytest.raises(factory.FactoryPreempted, match="life boundary"):
            collect(accompanied, preempt=True, skip=set())
    ledger = tmp_path / "verified-ledger.json"
    build_ledger(accompanied, ledger, namespace, allocation)
    assert load_ledger(ledger, namespace, allocation) == {"excluded-000--life-000000"}
    path = accompanied / "lives/excluded-000--life-000000.json"
    payload = path.read_bytes()
    with _bounded_native_control(tmp_path / "resumed-companion-operations"):
        collect(accompanied, preempt=False, skip={0})
    assert path.read_bytes() == payload
    original = (baseline / "lives" / path.name).read_bytes()
    assert payload == original
    assert hashlib.sha256(payload).digest() == hashlib.sha256(original).digest()
    # Resuming never overwrites the sealed prefix, including its SHA sidecar.
    assert (
        path.with_name(path.name + ".sha256").read_text().split()[0]
        == hashlib.sha256(original).hexdigest()
    )


def test_science_reference_remains_detached_clean_and_exactly_frozen() -> None:
    """Verify the documented D256 closure, without importing or mutating it."""
    root = ROOT.parent / "snake-dqn-d256-resume-science.noindex"
    if not root.exists():
        pytest.skip("operator science worktree unavailable; activation proof incomplete")
    assert frozen_factory._validate(root, frozen_factory.OLD_PIN) == root.resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", frozen_factory.OLD_PIN, "--"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode()
    # These immutable export bytes are shared by the frozen and operational
    # coordinators; the latter must retain the old ledger interpretation.
    expected = subprocess.check_output(
        ["git", "show", f"{frozen_factory.OLD_PIN}:research/bl_labels/exports.py"], cwd=ROOT
    )
    assert (ROOT / "research/bl_labels/exports.py").read_bytes() == expected
    assert frozen_factory.OLD_PIN == "61900af28932e47e22e332840fc4406559adb062"


def test_identity_ledger_namespace_refusal_remains_in_force(tmp_path: Path) -> None:
    allocation = [{"allocation_id": "excluded-000", "seed": 7}]
    ledger = tmp_path / "empty.json"
    publish_json(ledger, empty_ledger("excluded-identity", allocation))
    with pytest.raises(ValueError, match="namespace"):
        load_ledger(ledger, "foreign-companion-namespace", allocation)
