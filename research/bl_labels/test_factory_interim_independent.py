"""Independent Codex-V fixtures for interim conditions 1-3; no outcomes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels import mac_driver as md
from research.bl_labels import slowdown_resume as sr
from research.bl_labels.exports import publish_json
from research.bl_labels.factory_persistence import RULING_SHA256, PersistenceTracker
from research.bl_labels.test_slowdown_resume import _attempt, _trip
from research.compute import factory_interim_launcher as launcher

ROOT = Path(__file__).resolve().parents[2]
RULING = ROOT.parent / "snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_PERSISTENCE_20261023.md"
RULING_SHA = RULING_SHA256
BASE = "5dfd4a74cb7d304cc7b48a4f4fa0947c76f3c867"
SCIENCE = "61900af28932e47e22e332840fc4406559adb062"
AGENTS = [
    "python -m pytest tests/test_fixture.py",
    "pytest research/bl_labels/test_fixture.py",
    "python research/study/train.py",
    "/opt/homebrew/bin/codex exec task",
    "/bin/bash /Users/owner/codex-handoff.sh task",
]


def _pin() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


@pytest.fixture(autouse=True)
def _isolated_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "TEST_SLOT_PATH", tmp_path / "test-slot.lock")


def _pair(state):
    _attempt(state, "one", workers=6, records=[_trip("watch6", 6, 100, 1000)])
    _attempt(state, "two", workers=6, records=[_trip("watch6", 6, 200, 2000)])


def _persistence_clock(monkeypatch):
    """Mock Darwin identity and advance the admission cadence without sleeping."""
    from research.bl_labels import factory_persistence

    clock = [0.0]
    monkeypatch.setattr(
        factory_persistence,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    monkeypatch.setattr(
        factory_persistence,
        "_native_identity",
        lambda pid: {
            "birth": f"fixture-{pid}",
            "birth_precision": {"unit": "fixture"},
            "executable": "/usr/bin/python3",
        },
    )
    return clock


def _persistent_guard(monkeypatch, owned=None):
    clock = _persistence_clock(monkeypatch)
    tracker = PersistenceTracker("interim-guard-fixture")
    for index in range(7):
        clock[0] = index * 5.0
        md._exclusive_factory_guard(set() if owned is None else owned, tracker)


def _host(monkeypatch, table):
    pin = _pin()
    _persistence_clock(monkeypatch)
    monkeypatch.setattr(sr, "_operational_pin", lambda: pin)
    monkeypatch.setattr(md, "_command_output", lambda _: table)
    monkeypatch.setattr(
        md,
        "_guard_snapshot",
        lambda _: {
            "ac_power": True,
            "lid_open": True,
            "thermal": (
                "Thermal Warning Level = 0\nPerformance Warning Level = 0\n"
                "CPU_Speed_Limit = 100\n"
            ),
            "free_disk_bytes": 100 * 1024**3,
        },
    )
    monkeypatch.setattr(sr.subprocess, "check_output", lambda *args, **kwargs: "123 2.0 python\n")
    return pin


@pytest.mark.parametrize("argv", AGENTS)
@pytest.mark.parametrize("nice", [14, 15, 19])
@pytest.mark.parametrize("phase", ["runtime", "host-check"])
def test_foreign_priority_at_startup_runtime_and_recovery(tmp_path, monkeypatch, argv, nice, phase):
    _pair(tmp_path)
    _host(monkeypatch, f"123 1 {nice} {argv}\n")
    if nice < 15:
        with pytest.raises((md.MacGuardError, ValueError)):
            (
                _persistent_guard(monkeypatch)
                if phase == "runtime"
                else sr.record_host_check(tmp_path, checked_epoch=2001)
            )
        assert not (tmp_path / "slowdown-host-check-2001.json").exists()
    elif phase == "runtime":
        md._exclusive_factory_guard(set(), PersistenceTracker("safe-runtime"))
    else:
        sr.record_host_check(tmp_path, checked_epoch=2001)
        assert sr.resume_policy(6, tmp_path) == (6, [])


@pytest.mark.parametrize(
    "row",
    [
        "123 1 ? python -m pytest x\n",
        "123 1 python -m pytest x\n",
        "123 1 99 python -m pytest x\n",
        None,
    ],
)
@pytest.mark.parametrize("phase", ["runtime", "host-check"])
def test_unknown_or_unreadable_priority_refused(tmp_path, monkeypatch, row, phase):
    _pair(tmp_path)
    _host(monkeypatch, row)
    if row is None:

        def unreadable(_):
            raise subprocess.CalledProcessError(1, "ps")

        monkeypatch.setattr(md, "_command_output", unreadable)
    with pytest.raises((md.MacGuardError, ValueError, subprocess.SubprocessError)):
        (
            md._exclusive_factory_guard(set(), PersistenceTracker("invalid-metadata"))
            if phase == "runtime"
            else sr.record_host_check(tmp_path, checked_epoch=2001)
        )
    assert not (tmp_path / "slowdown-host-check-2001.json").exists()


@pytest.mark.parametrize(
    "child", ["python -c import_numpy", "/usr/bin/awk numeric_tool", "python -m pytest tests"]
)
@pytest.mark.parametrize("phase", ["runtime", "host-check"])
def test_foreign_descendant_requires_research_or_agent_match(tmp_path, monkeypatch, child, phase):
    _pair(tmp_path)
    _host(monkeypatch, f"123 1 15 codex exec task\n124 123 19 /bin/sh helper\n125 124 14 {child}\n")

    def check():
        (
            _persistent_guard(monkeypatch)
            if phase == "runtime"
            else sr.record_host_check(tmp_path, checked_epoch=2001)
        )

    if child == "python -m pytest tests":
        with pytest.raises((md.MacGuardError, ValueError)):
            check()
    else:
        check()


@pytest.mark.parametrize(
    "argv",
    [
        "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
        "CodexCLI.app/Contents/MacOS/codex exec task",
        "/Applications/ChatGPT.app/Contents/Frameworks/Codex (Service)",
        str(Path.home() / ".codex/computer-use/codex") + " exec task",
        str(Path.home() / "Library/Application Support/Codex/codex") + " exec task",
    ],
)
def test_desktop_helpers_remain_excluded(tmp_path, monkeypatch, argv):
    _pair(tmp_path)
    _host(monkeypatch, f"123 1 0 {argv}\n")
    md._exclusive_factory_guard(set(), PersistenceTracker("desktop-helper"))
    sr.record_host_check(tmp_path, checked_epoch=2001)


def test_owned_descendants_remain_owned(monkeypatch):
    monkeypatch.setattr(
        md,
        "_command_output",
        lambda _: "123 1 10 python research/factory.py\n124 123 0 python -c tool\n",
    )
    _persistent_guard(monkeypatch, {123})


def test_honest_receipt_binds_ruling_and_exact_operational_pin(tmp_path, monkeypatch):
    _pair(tmp_path)
    pin = _host(monkeypatch, "123 1 15 codex exec task\n")
    path = sr.record_host_check(tmp_path, checked_epoch=2001)
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == "bl-labels-f0-slowdown-host-check-colab-control/v4"
    assert receipt["policy"] == "ALLOW_PERSISTENCE_30S"
    assert receipt["policy_id"] == "factory-nice15-persistence30/v1"
    assert receipt["ruling_sha256"] == RULING_SHA
    assert receipt["control_policy_id"] == sr.CONTROL_POLICY_ID
    assert receipt["control_ruling_sha256"] == sr.CONTROL_RULING_SHA256
    assert receipt["process_window"]["observation_count"] >= 7
    assert receipt["operational_pin"] == pin
    assert "no_sol_codex" not in receipt
    assert receipt["no_disallowed_competing_compute"] is True
    assert receipt["serial_test_lane_stopped"] is True
    assert sr.resume_policy(6, tmp_path) == (6, [])
    triplet = [path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")]
    before = [part.read_bytes() for part in triplet]
    with pytest.raises(FileExistsError):
        sr.record_host_check(tmp_path, checked_epoch=2001)
    assert [part.read_bytes() for part in triplet] == before
    with pytest.raises(FileExistsError):
        publish_json(path, {**receipt, "operational_pin": "0" * 40})
    assert [part.read_bytes() for part in triplet] == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "bl-labels-f0-slowdown-host-check/v1"),
        ("schema", "foreign/v99"),
        ("policy", "ALLOW_BOUNDED_CONCURRENCY"),
        ("ruling_sha256", "0" * 64),
        ("operational_pin", "0" * 40),
        ("operational_pin", None),
        ("serial_test_lane_stopped", False),
        ("no_disallowed_competing_compute", False),
        ("no_sol_codex", True),
    ],
)
def test_otherwise_valid_foreign_receipt_refused(tmp_path, monkeypatch, field, value):
    _pair(tmp_path)
    pin = _pin()
    monkeypatch.setattr(sr, "_operational_pin", lambda: pin)
    receipt = {
        "schema": "bl-labels-f0-slowdown-host-check-nice-interim/v2",
        "policy": "ALLOW_NICE15_INTERIM",
        "ruling_sha256": RULING_SHA,
        "operational_pin": pin,
        "for_trip_epoch": 2000,
        "checked_epoch": 2001,
        "pmset_therm": "CPU_Speed_Limit = 100",
        "top_cpu_consumers": [],
        "host_health": {"ac_power": True, "lid_open": True},
        "serial_test_lane_stopped": True,
        "no_disallowed_competing_compute": True,
    }
    receipt[field] = value
    path = tmp_path / "slowdown-host-check-2001.json"
    publish_json(path, receipt)
    before = path.read_bytes()
    for width in (6, 12):
        with pytest.raises(ValueError, match="host check"):
            sr.resume_policy(width, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "failure,reason",
    [
        (None, None),
        ("battery", "battery"),
        ("lid", "lid closed"),
        ("thermal", "thermal"),
        ("disk", "free disk"),
    ],
)
def test_health_still_fails_closed_with_complete_observations(
    tmp_path, monkeypatch, failure, reason
):
    def read(command):
        if command[-1] == "batt":
            return "Battery Power" if failure == "battery" else "AC Power"
        if command[0].endswith("ioreg"):
            return (
                '"AppleClamshellState" = Yes' if failure == "lid" else '"AppleClamshellState" = No'
            )
        return (
            "Thermal Warning Level = 0\nPerformance Warning Level = 0\nCPU_Speed_Limit = "
            + ("50" if failure == "thermal" else "100")
            + "\n"
        )

    monkeypatch.setattr(md, "_command_output", read)
    monkeypatch.setattr(
        md.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=0 if failure == "disk" else 100 * 1024**3),
    )
    if failure:
        with pytest.raises(md.MacGuardError, match=reason):
            md._guard_snapshot(tmp_path)
    else:
        assert md._guard_snapshot(tmp_path)["ac_power"] is True


def test_worker_nice_ten_is_verified_before_science(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(md.os, "setpriority", lambda *args: calls.append(args))
    monkeypatch.setattr(md.os, "getpriority", lambda *args: 15)
    messages = queue.Queue()
    job = md._Job({"allocation_id": "synthetic"}, set(), False)
    md._worker(
        job,
        "synthetic",
        [],
        Path("b8"),
        Path("frp3"),
        tmp_path,
        tmp_path / "timing",
        None,
        None,
        None,
        messages,
    )
    assert calls == [(os.PRIO_PROCESS, 0, 10)]
    assert "priority is not nice 10" in messages.get()["error"]


def test_source_accounting_and_frozen_science_closure():
    assert hashlib.sha256(RULING.read_bytes()).hexdigest() == RULING_SHA
    assert sr.RULING_SHA256 == RULING_SHA

    def git_bytes(rev, path):
        return subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=ROOT)

    unchanged = [
        "research/bl_labels/slowdown_guard.py",
        "research/bl_labels/slowdown_state.py",
        "research/bl_labels/d256_resume.py",
        "research/bl_labels/frozen_factory.py",
        "research/bl_labels/d256_w12.py",
        "research/bl_labels/w12_runtime.py",
        "research/bl_labels/exports.py",
        "research/bl_labels/RUN_D256_RESUME.sh",
    ]
    for path in unchanged:
        assert (ROOT / path).read_bytes() == git_bytes(BASE, path), path

    driver = ROOT / "research/bl_labels/mac_driver.py"

    def accounting_assignments(source):
        run = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        names = ("elapsed", "worker", "deadline", "cap", "charge")
        return [
            ast.dump(node)
            for node in ast.walk(run)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and any(key in target.id for key in names)
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]

    assert accounting_assignments(driver.read_text()) == accounting_assignments(
        git_bytes(BASE, "research/bl_labels/mac_driver.py")
    )
    assert (ROOT / "research/bl_labels/exports.py").read_bytes() == git_bytes(
        SCIENCE, "research/bl_labels/exports.py"
    )
    science_tree = ROOT.parent / "snake-dqn-d256-resume-science.noindex"
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=science_tree, text=True).strip()
        == SCIENCE
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=science_tree, text=True
    ).strip()
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=science_tree, capture_output=True
        ).returncode
        != 0
    )
    assert (
        subprocess.run(
            ["git", "diff", "--exit-code", SCIENCE, "--"], cwd=science_tree, capture_output=True
        ).returncode
        == 0
    )
    run = next(
        node
        for node in ast.parse(driver.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_exclusive_factory_guard"
    ]
    assert {ast.unparse(node.args[0]) for node in calls} == {"set(active)"}
    assert all(len(node.args) == 2 for node in calls)
    admission = next(
        node
        for node in ast.parse(driver.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_factory_admission_window"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_exclusive_factory_guard"
        and ast.unparse(node.args[0]) == "set()"
        and len(node.args) == 2
        for node in ast.walk(admission)
    )


def test_real_launcher_obeys_the_outer_canonical_test_lock(tmp_path):
    """The real exact-pin review route owns the canonical test lock outside pytest."""
    source = ROOT / "research/bl_labels/test_factory_interim_independent.py"
    node = (
        "research/bl_labels/test_factory_interim_independent.py::"
        "test_source_accounting_and_frozen_science_closure"
    )
    manifest = tmp_path / "review.txt"
    manifest.write_text(
        "\n".join(
            [
                "schema=factory-companion-test-manifest/v2",
                "source_commit=" + _pin(),
                "fixture=research/bl_labels/test_factory_interim_independent.py|"
                + hashlib.sha256(source.read_bytes()).hexdigest(),
                "node=" + node,
            ]
        )
        + "\n"
    )
    command = [
        str(ROOT / "tools/ops/agent_nice.sh"),
        "--class",
        "test",
        "--review-manifest",
        str(manifest),
        "--review-sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "--",
        "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python",
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:xdist",
        "-p",
        "research.compute.factory_companion_pytest",
        node,
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "serial test slot is busy" in result.stderr
    assert "passed" not in result.stdout


def test_actual_operational_identity_ignores_the_science_env(monkeypatch):
    monkeypatch.setenv("F0_COMMIT", SCIENCE)
    expected = _pin()
    assert expected != SCIENCE
    assert sr._operational_pin() == expected
