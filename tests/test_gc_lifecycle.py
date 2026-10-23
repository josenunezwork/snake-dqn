"""Permanent subprocess d.3 retention check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(tmp_path: Path, policy: str) -> dict:
    path = tmp_path / f"lifecycle-{policy}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.bl_gc.lifecycle",
            "--policy",
            policy,
            "--out",
            str(path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, SNAKE_SERVE_GC_POLICY=policy, SNAKE_SERVE_PATHB="0"),
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    assert path.with_suffix(".json.sha256").is_file()
    assert path.with_suffix(".json.complete").read_text() == "complete\n"
    return json.loads(path.read_text())


@pytest.mark.parametrize("policy", ["default", "freeze_startup"])
def test_watch_session_replacements_release_under_both_gc_policies(
    tmp_path: Path, policy: str
) -> None:
    result = _run(tmp_path, policy)
    assert result["cycles"] == 20
    assert result["tracked_objects"]["cycle_20"] <= result["tracked_objects"]["cycle_5"] + 2000

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["weakrefs_cleared"] and result["teardown_weakrefs_cleared"]
    assert result["birth_census"]["total"] == 0
    assert result["teardown_census_pre_collect"]["total"] == 0
    assert result["freeze_calls_before_negative_control"] == int(policy == "freeze_startup")
    assert result["negative_control_retained_while_frozen"]
    assert result["negative_control_released_after_unfreeze"]
    assert result["gameplay_objects_released_by_teardown"] == 0
    assert len(result["cycle_checks"]) == 20
    for cycle, row in enumerate(result["cycle_checks"], 1):
        assert row["cycle"] == cycle
        assert all(
            row[key] for key in ("weakrefs_cleared", "freeze_non_increasing", "single_freeze")
        )
        assert all(value is not None for value in row.values())
    trajectory = result["freeze_count_trajectory"]
    assert len(trajectory) == 22  # receipt, after initialize, 20 boundaries
    assert all(b <= a <= result["startup_freeze_count"] for a, b in zip(trajectory, trajectory[1:]))
    assert result["freeze_count_total_decrease"] == trajectory[0] - trajectory[-1]
    assert (
        result["attestation"]["git_commit"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    assert type(result["attestation"]["git_dirty"]) is bool


def test_i1_real_frozen_decrease_passes_and_second_freeze_fails() -> None:
    code = """
import gc
from research.bl_gc.lifecycle import freeze_non_increasing
container = [object()]
gc.freeze()
baseline = gc.get_freeze_count()
del container
decreased = gc.get_freeze_count()
assert decreased < baseline
assert freeze_non_increasing([decreased], baseline)
young = [[] for _ in range(1000)]
gc.freeze()  # injected prohibited second freeze
increased = gc.get_freeze_count()
assert increased > decreased
assert not freeze_non_increasing([decreased, increased], baseline)
gc.unfreeze()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_census_detects_explicit_subclasses_and_module_prefixes() -> None:
    from research.bl_gc.lifecycle import gameplay_census

    class Gameplay:
        pass

    class Child(Gameplay):
        pass

    prefix_type = type("PrefixObject", (), {"__module__": "src.model.test_census"})
    objects = [Child(), prefix_type()]
    census = gameplay_census((Gameplay,))
    assert census["total"] >= len(objects)
    assert census["types"]["src.model.test_census.PrefixObject"] == 1


def test_failed_invariant_records_all_twenty_cycles_in_subprocess() -> None:
    code = """
import gc
from research.bl_gc import lifecycle as l
calls = 0
def cycle(seed):
    global calls
    calls += 1
    if calls == 1:
        gc.freeze()  # injected second freeze: I1/I2 FAIL; continue all cycles
    return (), {}
l._exercise_cycle = cycle
result = l.run("freeze_startup")
assert result["status"] == "FAIL", result
assert result["cycles"] == 20
assert result["checks"]["I1"] is False
assert result["checks"]["I2"] is False
assert result["freeze_calls_before_negative_control"] == 2
assert all(v is not None for row in result["cycle_checks"] for v in row.values())
assert result["negative_control_retained_while_frozen"]
assert result["negative_control_released_after_unfreeze"]
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_crash_preserves_reached_checks_and_nulls_only_unreached() -> None:
    code = """
import gc
from research.bl_gc import lifecycle as l
calls = 0
def cycle(seed):
    global calls
    calls += 1
    if calls == 2:
        raise RuntimeError("injected crash")
    return (), {}
l._exercise_cycle = cycle
r = l.run("default")
assert r["status"] == "FAIL" and r["cycles"] == 1
assert r["failure"]["message"] == "injected crash"
assert r["checks"]["I3"] is not None
assert r["checks"]["I1"] is not None
assert r["checks"]["weakrefs"] is True
assert all(v is not None for v in r["cycle_checks"][0].values())
assert all(v is None for k,v in r["cycle_checks"][1].items() if k != "cycle")
assert r["checks"]["I4"] is None
assert r["checks"]["negative_retained"] is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
