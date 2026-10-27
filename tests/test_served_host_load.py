"""Classified host policy and interval evidence, without a Mac timing run."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from research.bl_served_search import host_load as load


@pytest.mark.parametrize(
    "command,kind",
    [
        ("/System/Library/Frameworks/CoreSpotlight.framework/corespotlightd", "macos_service"),
        ("/usr/libexec/fileproviderd", "macos_service"),
        ("/usr/libexec/cloudd", "macos_service"),
        ("/System/Library/PrivateFrameworks/SkyLight.framework/WindowServer", "macos_service"),
        ("/Applications/Codex.app/Contents/Frameworks/Codex Helper (Renderer)", "desktop_renderer"),
        ("/Applications/Chrome.app/Contents/Frameworks/Chrome Helper (GPU)", "desktop_renderer"),
        (
            "/Applications/Codex.app/Contents/Frameworks/Codex (Renderer).app/Contents/MacOS/"
            "Codex (Renderer)",
            "desktop_renderer",
        ),
        ("/Applications/Codex.app/Contents/Resources/codex", "agent_workload"),
        ("/usr/local/bin/agy", "agent_workload"),
        ("/System/Library/Python.app/Contents/MacOS/Python", "agent_workload"),
        ("/usr/local/bin/python3.12", "agent_workload"),
        ("/Users/josenunez/Projects/ml/snake-dqn/research/job", "agent_workload"),
        ("/Applications/Codex.app/Contents/Resources/node", "agent_workload"),
        ("/tmp/Codex Helper (Renderer)", "unknown"),
        ("/System/../tmp/corespotlightd", "unknown"),
        ("/Applications/Other.app/Contents/MacOS/Other", "unknown"),
        ("/bin/bash", "unknown"),
    ],
)
def test_executable_classifier(command, kind):
    assert load.process_kind(command) == kind


def inventory(at, rows):
    return {
        "time_ns": int(at * 1e9),
        "capture_start_ns": int(at * 1e9) - 100,
        "capture_end_ns": int(at * 1e9) + 100,
        "processes": [
            {"pid": pid, "ppid": ppid, "birth": birth, "command": command, "cpu_seconds": cpu}
            for pid, ppid, birth, command, cpu in rows
        ],
    }


def test_cpu_deltas_aggregate_background_and_exclude_exact_probe_ancestry():
    rows = [
        (10, 1, "a", "python", 1),
        (11, 10, "b", "python", 1),
        (12, 11, "c", "/bin/ps", 1),
        (20, 1, "d", "/usr/libexec/cloudd", 1),
        (21, 1, "e", "/usr/libexec/fileproviderd", 1),
        (22, 1, "f", "/usr/local/bin/python", 1),
    ]
    previous, current = inventory(1, rows), inventory(2, rows)
    for row in current["processes"]:
        row["cpu_seconds"] += 0.5
    sample = load.cpu_interval(previous, current, 10)
    assert sample["coverage_complete"]
    assert sample["background_cpu_percent"] == 100
    assert sample["workload_cpu_percent"] == 50
    assert [row["pid"] for row in sample["processes"]] == [20, 21, 22]
    with pytest.raises(RuntimeError, match="foreign workload"):
        load.LoadGuard().check(sample)


@pytest.mark.parametrize("change", ["birth", "exec", "exit", "new", "gap"])
def test_churn_reuse_exec_and_gaps_cannot_establish_low_coverage(change):
    rows = [(20, 1, "a", "/usr/libexec/cloudd", 1)]
    before, after = inventory(1, rows), inventory(2, rows)
    if change == "birth":
        after["processes"][0]["birth"] = "b"
    elif change == "exec":
        after["processes"][0]["command"] = "/usr/local/bin/python"
    elif change == "exit":
        after["processes"] = []
    elif change == "new":
        before["processes"] = []
    else:
        after["time_ns"] = 7_000_000_000
    assert not load.cpu_interval(before, after, 10)["coverage_complete"]


@pytest.mark.parametrize("command", ["python", "pytest", "codex-exec", "node", "/tmp/worker"])
def test_new_foreign_process_cpu_is_measured_and_aborts_before_exit(command):
    before = inventory(1, [])
    after = inventory(2, [(20, 1, "new", command, 0.8)])
    measured = load.cpu_interval(before, after, 10)
    assert not measured["coverage_complete"]
    assert measured["process_churn"][0]["pid"] == 20
    assert measured["processes"][0]["cpu_percent"] == pytest.approx(80)
    group = "unknown" if command == "/tmp/worker" else "workload"
    assert measured[f"{group}_cpu_percent"] == pytest.approx(80)
    with pytest.raises(RuntimeError, match="foreign workload"):
        load.LoadGuard().check(measured)


@pytest.mark.parametrize(
    "old_command,new_command,kind",
    [
        ("python", "/usr/libexec/cloudd", "agent_workload"),
        ("/usr/libexec/cloudd", "python", "agent_workload"),
        ("/tmp/worker", "/usr/libexec/cloudd", "unknown"),
        ("/usr/libexec/cloudd", "/tmp/worker", "unknown"),
        ("/tmp/worker", "python", "agent_workload"),
        ("python", "/tmp/worker", "agent_workload"),
    ],
)
def test_exec_cpu_delta_keeps_conservative_foreign_classification(old_command, new_command, kind):
    before = inventory(1, [(20, 1, "a", old_command, 10)])
    after = inventory(2, [(20, 1, "a", new_command, 10.8)])
    measured = load.cpu_interval(before, after, 10)
    assert not measured["coverage_complete"]
    assert len(measured["process_churn"]) == 1
    assert measured["processes"][0]["kind"] == kind
    assert measured["processes"][0]["cpu_percent"] == pytest.approx(80)
    group = "workload" if kind == "agent_workload" else "unknown"
    assert measured[f"{group}_cpu_percent"] == pytest.approx(80)
    assert measured["background_cpu_percent"] == 0
    with pytest.raises(RuntimeError, match="foreign workload"):
        load.LoadGuard().check(measured)


@pytest.mark.parametrize("churn", ["new", "exec"])
def test_background_churn_preserves_measured_cpu_and_sustained_ceiling(churn):
    guard = load.LoadGuard()
    for tick in range(61):
        before = inventory(tick, [(20, 1, "a", "/usr/libexec/cloudd", 10)])
        after = inventory(tick + 1, [(20, 1, "a", "/usr/libexec/cloudd", 11)])
        if churn == "new":
            after["processes"].append(
                inventory(0, [(30 + tick, 1, str(tick), "/usr/libexec/mdworker_shared", 0.6)])[
                    "processes"
                ][0]
            )
        else:
            before["processes"][0]["command"] = "/usr/libexec/fileproviderd"
            after["processes"][0]["cpu_seconds"] = 11.6
        measured = load.cpu_interval(before, after, 10)
        assert not measured["coverage_complete"]
        assert measured["process_churn"]
        assert measured["background_cpu_percent"] == pytest.approx(160)
        assert sum(row["cpu_percent"] for row in measured["processes"]) == pytest.approx(160)
        if tick == 60:
            with pytest.raises(RuntimeError, match=">150% for >60s"):
                guard.check(measured)
        else:
            guard.check(measured)
            assert guard.high_since_ns == 0


def sample(start, end, cpu=151):
    return {
        "start_ns": int(start * 1e9),
        "end_ns": int(end * 1e9),
        "background_cpu_percent": cpu,
        "processes": [],
        "coverage_complete": True,
    }


def test_sustained_background_ceiling_strict_boundaries_and_reset():
    guard = load.LoadGuard()
    for tick in range(60):
        guard.check(sample(tick, tick + 1))
    with pytest.raises(RuntimeError, match=">150% for >60s"):
        guard.check(sample(60, 61))
    guard = load.LoadGuard()
    for tick in range(120):
        guard.check(sample(tick, tick + 1, 150))
    assert guard.high_since_ns is None
    for tick in range(120, 170):
        guard.check(sample(tick, tick + 1))
    guard.check(sample(170, 171, 20))
    guard.check(sample(171, 172))
    assert guard.high_since_ns == 171_000_000_000


def test_sustained_guard_does_not_bridge_missing_or_long_intervals():
    guard = load.LoadGuard()
    for tick in range(50):
        guard.check(sample(tick, tick + 1))
    guard.check(sample(55, 56))
    assert guard.high_since_ns == 55_000_000_000
    guard.check(sample(56, 70))
    assert guard.high_since_ns is None


def test_parse_cumulative_inventory_retains_shared_clock_and_birth(monkeypatch):
    from unittest.mock import Mock

    capture = Mock(
        return_value=SimpleNamespace(
            stdout=(
                "PID PPID STARTED TIME COMM\n"
                "20 1 Tue Oct 27 08:00:00 2026 123:45.67 /usr/libexec/cloudd\n"
            )
        )
    )
    monkeypatch.setattr(load.subprocess, "run", capture)
    ticks = iter([100, 200])
    monkeypatch.setattr(load.time, "perf_counter_ns", lambda: next(ticks))
    result = load.capture_cpu()
    assert result["time_ns"] == 150
    assert result["processes"][0]["birth"] == "Tue Oct 27 08:00:00 2026"
    assert result["processes"][0]["cpu_seconds"] == 7425.67


def test_negative_counter_and_malformed_time_refuse_evidence():
    before = inventory(1, [(20, 1, "a", "python", 2)])
    after = deepcopy(before)
    after["time_ns"] += 1_000_000_000
    after["processes"][0]["cpu_seconds"] = 1
    with pytest.raises(ValueError, match="decreased"):
        load.cpu_interval(before, after, 10)
    with pytest.raises(ValueError):
        load._cpu_seconds("invalid")


class _DesktopClassifier:
    """Frozen-registry stand-in with terminal transition revocation."""

    def __init__(self):
        self.revoked = set()

    def invalidate_transition(self, row):
        self.revoked.add((row["pid"], row["birth"]))

    def classify(self, row):
        if (row.get("pid"), row.get("birth")) in self.revoked:
            return {"kind": "unknown", "identity_binding_changed": True}
        if (
            row.get("command") == "/Applications/Claude.app/Contents/MacOS/Claude"
            and row.get("uid") == 501
            and row.get("birth") == "claude"
        ):
            return {
                "kind": "desktop_app",
                "reason": "enrolled_verified_ui_role",
                "application": "Claude",
                "role": "main_ui",
            }
        if (
            row.get("command") == "/Applications/Outlook.app/Contents/MacOS/Outlook"
            and row.get("uid") == 501
            and row.get("birth") == "outlook"
        ):
            return {
                "kind": "desktop_app",
                "reason": "enrolled_verified_ui_role",
                "application": "Outlook",
                "role": "main_ui",
            }
        return {"kind": "unknown", "reason": "process_identity_not_revalidated"}


def test_verified_claude_gui_is_not_the_claude_cli_workload():
    classifier = _DesktopClassifier()
    assert (
        load.process_kind(
            "/Applications/Claude.app/Contents/MacOS/Claude",
            classifier,
            {"pid": 20, "uid": 501, "birth": "claude"},
        )
        == "desktop_app"
    )
    assert load.process_kind("/usr/local/bin/claude", classifier, {"pid": 21}) == "agent_workload"


def test_desktop_app_uses_shared_background_pool_but_workload_precedence_remains():
    before = inventory(
        1,
        [
            (20, 1, "outlook", "/Applications/Outlook.app/Contents/MacOS/Outlook", 1),
            (21, 1, "python", "/Applications/Outlook.app/Contents/MacOS/python", 1),
        ],
    )
    after = deepcopy(before)
    after["time_ns"] = 2_000_000_000
    for row in before["processes"] + after["processes"]:
        row["uid"] = 501
    for row in after["processes"]:
        row["cpu_seconds"] += 0.25
    sample = load.cpu_interval(before, after, 10, desktop_classifier=_DesktopClassifier())
    assert sample["desktop_app_cpu_percent"] == pytest.approx(25)
    assert sample["background_cpu_percent"] == pytest.approx(25)
    assert sample["workload_cpu_percent"] == pytest.approx(25)
    assert sample["processes"][0]["classification"]["role"] == "main_ui"
    with pytest.raises(RuntimeError, match="foreign workload"):
        load.LoadGuard(desktop_policy=True).check(sample)


def test_desktop_app_avoids_per_process_abort_but_shared_ceiling_still_applies():
    classifier = _DesktopClassifier()
    guard = load.LoadGuard(desktop_policy=True)
    for tick in range(60):
        before = inventory(
            tick,
            [(20, 1, "outlook", "/Applications/Outlook.app/Contents/MacOS/Outlook", 1)],
        )
        after = inventory(
            tick + 1,
            [(20, 1, "outlook", "/Applications/Outlook.app/Contents/MacOS/Outlook", 2.6)],
        )
        for row in before["processes"] + after["processes"]:
            row["uid"] = 501
        sample = load.cpu_interval(before, after, 10, desktop_classifier=classifier)
        assert sample["desktop_app_cpu_percent"] == pytest.approx(160)
        guard.check(sample)
    before = inventory(
        60, [(20, 1, "outlook", "/Applications/Outlook.app/Contents/MacOS/Outlook", 1)]
    )
    after = inventory(
        61, [(20, 1, "outlook", "/Applications/Outlook.app/Contents/MacOS/Outlook", 2.6)]
    )
    for row in before["processes"] + after["processes"]:
        row["uid"] = 501
    with pytest.raises(RuntimeError, match=">150% for >60s"):
        guard.check(load.cpu_interval(before, after, 10, desktop_classifier=classifier))


def test_identity_drift_aborts_below_twenty_but_initial_unknown_does_not():
    guard = load.LoadGuard(desktop_policy=True)
    ordinary_unknown = sample(0, 1, 0)
    ordinary_unknown["processes"] = [
        {"kind": "unknown", "cpu_percent": 1, "classification": {"reason": "uid_mismatch"}}
    ]
    guard.check(ordinary_unknown)
    changed = sample(1, 2, 0)
    changed["processes"] = [
        {
            "kind": "unknown",
            "cpu_percent": 1,
            "classification": {"reason": "uid_mismatch", "identity_binding_changed": True},
        }
    ]
    with pytest.raises(RuntimeError, match="identity changed"):
        guard.check(changed)


def test_unknown_cpu_is_not_double_counted_in_desktop_totals():
    before = inventory(0, [(20, 1, "unknown", "/tmp/unknown", 1)])
    after = inventory(1, [(20, 1, "unknown", "/tmp/unknown", 1.1)])
    sample = load.cpu_interval(before, after, 10, desktop_classifier=_DesktopClassifier())
    assert sample["unknown_cpu_percent"] == pytest.approx(10)
    assert sample["background_cpu_percent"] == 0


def test_desktop_interval_uid_change_is_churn_and_never_inherits_background():
    before = inventory(0, [(20, 1, "same", "/Applications/Outlook.app/Contents/MacOS/Outlook", 1)])
    after = inventory(1, [(20, 1, "same", "/Applications/Outlook.app/Contents/MacOS/Outlook", 1.5)])
    before["processes"][0]["uid"] = 0
    after["processes"][0]["uid"] = 501
    sample = load.cpu_interval(before, after, 10, desktop_classifier=_DesktopClassifier())
    assert not sample["coverage_complete"]
    assert sample["process_churn"][0]["reason"] == "uid_changed_no_delta"
    assert sample["processes"][0]["kind"] == "unknown"
    assert sample["background_cpu_percent"] == 0
    transition = sample["processes"][0]["interval_transition"]
    assert transition["reason"] == "uid_changed_no_delta"
    assert transition["previous"]["uid"] == 0
    assert transition["previous_classification"]["kind"] == "unknown"
    assert transition["conservative_kind"] == "unknown"


def test_interval_admits_new_desktop_process_before_its_first_classification():
    class AdmissionOnlyClassifier:
        def __init__(self):
            self.bound = set()

        def admit_process(self, row):
            self.bound.add((row["pid"], row["birth"]))
            return self.classify(row)

        def classify(self, row):
            if (row.get("pid"), row.get("birth")) in self.bound:
                return {"kind": "desktop_renderer", "reason": "enrolled_verified_ui_role"}
            return {"kind": "unknown", "reason": "process_identity_not_revalidated"}

    before = inventory(0, [])
    after = inventory(
        1,
        [
            (
                20,
                1,
                "renderer",
                "/Applications/Chrome.app/Contents/MacOS/Chrome Helper (Renderer)",
                0.3,
            )
        ],
    )
    after["processes"][0]["uid"] = 501
    sample = load.cpu_interval(before, after, 10, desktop_classifier=AdmissionOnlyClassifier())
    assert sample["processes"][0]["kind"] == "desktop_renderer"
    assert sample["background_cpu_percent"] == pytest.approx(30)
