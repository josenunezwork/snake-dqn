"""Targeted V0 window, literal decision, lease and receipt guards."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from research.bl_served_search.exports import export_json
from research.bl_v0 import contract
from research.bl_v0.lease import acquire, require
from research.bl_v0.run import protected_jobs, verify_canary, verify_k1


def rows(c2: float = 0.5, c3: float = 1.0) -> list[dict]:
    return [
        {
            "fixture_id": fixture,
            "cell": cell,
            "repeat": repeat,
            "operator_wall_seconds": c2 if cell == "C2" else c3,
            "stage_seconds": {"sim": 0.1, "leaf_call_wall": 0.01},
            "identity_ok": True,
            "native_v8": {
                "mode": "kernel",
                "kernel_rows": 1,
                "rows": 1,
                "replayed_rows": 0,
                "stock_rows": 0,
                "fallback_rows": 0,
                "noncandidate_rows": 0,
            },
        }
        for fixture in contract.FIXTURES
        for repeat in range(30)
        for cell in contract.CELLS
    ]


def test_literal_rule_boundary_cold_and_dense_cap() -> None:
    population = rows()
    assert contract.decision(population)["status"] == "PASS"  # C3 == 2*C2
    assert contract.decision(rows(c3=0.999))["status"] == "NO_CONSUMER"
    assert contract.decision(rows(c3=1.458))["status"] == "PASS"
    assert contract.decision(rows(c3=1.458001))["status"] == "LEARNED_CUTOFF_CLOSED"
    for row in population:
        if row["repeat"] == 0:
            row["operator_wall_seconds"] = 500
    assert contract.decision(population)["status"] == "PASS"
    metric = contract.summarize(population)["frozen-800/C3"]["operator_wall_seconds"]
    assert metric["cold"] == 500 and metric["warm29"]["n"] == 29
    assert metric["warm29"]["p99"] == 1.0


@pytest.mark.parametrize(
    "defect",
    [
        "kernel_rows",
        "stock_rows",
        "fallback_rows",
        "noncandidate_rows",
        "replayed_rows",
        "identity",
        "duplicate",
    ],
)
def test_missing_execution_or_identity_never_passes(defect: str) -> None:
    population = rows()
    if defect == "identity":
        population[0]["identity_ok"] = False
    elif defect == "duplicate":
        population.append(population[0])
    else:
        population[0]["native_v8"][defect] = 0 if defect == "kernel_rows" else 1
    assert contract.decision(population)["status"] == "INVALID"


def test_no_scripted_counter_exemption_or_partial_pass() -> None:
    population = rows()
    assert contract.decision(population[:-1])["status"] == "INCOMPLETE"
    for row in population:
        if row["fixture_id"].startswith("scripted"):
            row["native_v8"]["kernel_rows"] = 0
    assert contract.decision(population)["status"] == "INVALID"


@pytest.mark.parametrize(
    "stamp,allowed",
    [
        ("2026-10-22T20:00:00", False),
        ("2026-10-23T09:00:00", False),
        ("2026-10-23T21:59:59", False),
        ("2026-10-23T22:00:00", True),
        ("2026-10-23T23:00:00", False),
        ("2026-10-24T08:00:00", True),
        ("2026-10-24T09:59:59", True),
        ("2026-10-24T10:00:00", False),
    ],
)
def test_window_boundaries(stamp: str, allowed: bool) -> None:
    now = dt.datetime.fromisoformat(stamp).replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    if allowed:
        assert 0 < contract.window(now) <= 7200
    else:
        with pytest.raises(RuntimeError):
            contract.window(now)
    if stamp.startswith("2026-10-22T20") or stamp.startswith("2026-10-23T09"):
        with pytest.raises(RuntimeError, match="protected"):
            contract.window(now, smoke=True)


def test_lease_exclusion_expiry_ownership_and_release(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    with acquire(path, "test", "canary", 30) as lease:
        require(path, lease)
        from research.bl_served_search import mac_lease

        assert mac_lease.require(path, "test", lease["id"]) == lease
        with pytest.raises(FileExistsError):
            with acquire(path, "other", "collision", 30):
                pytest.fail("lease collision admitted")
    assert not path.exists()
    with acquire(path, "test", "expired", -1) as lease:
        with pytest.raises(TimeoutError):
            require(path, lease)
        with pytest.raises(FileExistsError):
            with acquire(path, "other", "no stale stealing", 30):
                pytest.fail("expired lease stolen")
    with acquire(path, "test", "foreign replacement", 30) as lease:
        path.write_text(json.dumps({"id": "foreign"}))
        with pytest.raises(RuntimeError):
            require(path, lease)
    assert json.loads(path.read_text())["id"] == "foreign"


def test_admission_requires_real_matching_canary_and_measured_k1(tmp_path: Path) -> None:
    manifest = {"sources": {"instrument": "hash"}}
    canary = tmp_path / "smoke.json"
    export_json(
        canary,
        {
            "schema": "bl-v0-mac-receipt/v1",
            "status": "PRECHECK_ONLY",
            "sources": manifest["sources"],
            "completed_batches": 4,
            "canary_ok": False,
            "platform": {"system": "Darwin"},
        },
    )
    with pytest.raises(ValueError, match="canary"):
        verify_canary(canary, manifest)
    k1 = tmp_path / "timing.json"
    export_json(
        k1,
        {"schema": "bl-k1-mac-timing/v1", "status": "PREPARED_NOT_TIMED", "completed_batches": 0},
    )
    with pytest.raises(ValueError, match="K-1"):
        verify_k1(k1, [])


def test_invalidated_canary_is_refused_before_reading_any_success_metadata(tmp_path: Path) -> None:
    (tmp_path / "INVALIDATION.json").write_text("{}")
    with pytest.raises(ValueError, match="invalidated"):
        verify_canary(tmp_path / "smoke.json", {})


@pytest.mark.parametrize(
    "args",
    [
        "python -m research.bl_k1.audit --manifest a",
        "python -m research.bl_k1.timing --manifest a",
        "python research/bl_x5/timing.py --out a",
        "python -m research.bl_x5.timing --out a",
    ],
)
def test_protected_live_processes_are_never_overlapped(args: str) -> None:
    with pytest.raises(RuntimeError, match="protected"):
        protected_jobs(f"PID COMM ARGS\n100 Python {args}\n")
    with pytest.raises(RuntimeError, match="protected"):
        protected_jobs(
            "PID COMM ARGS\n100 /Library/Framewo "
            "/Library/Frameworks/Python.framework/Versions/3.12/Resources/"
            f"Python.app/Contents/MacOS/Python {args.removeprefix('python ')}\n"
        )
    protected_jobs(f"PID COMM ARGS\n100 codex codex --prompt {args}\n")


@pytest.mark.parametrize("bad_timer", [False, True])
def test_real_runner_export_loop_and_canary_validation_with_fake_batches(
    tmp_path, monkeypatch, bad_timer
) -> None:
    """Exercise CLI orchestration without simulation or a timing measurement."""
    from research.bl_v0 import operator, run

    fixture = {
        "fixture_id": "frozen-800",
        "root_sha256": "a" * 64,
        "mix": "frozen",
        "legal_actions": [0, 1],
    }
    manifest = {"fixtures": [fixture], "sources": {"synthetic_test_only": "a" * 64}}
    path = tmp_path / "manifest.json"
    export_json(path, manifest)
    monkeypatch.setattr(contract, "verify_manifest", lambda _: manifest)
    monkeypatch.setattr(contract, "published", lambda: {"synthetic_test_only": "not_science"})
    monkeypatch.setattr(contract, "window", lambda **_: 180)
    monkeypatch.setattr(run, "thread_guard", lambda: None)
    monkeypatch.setattr(run, "protected_jobs", lambda: None)
    monkeypatch.setattr(run, "DEFAULT_LEASE", tmp_path / "lease.json")
    monkeypatch.setattr(run, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(run, "check_guard", lambda: {"reasons": []})
    monkeypatch.setattr(run, "process_screen", lambda: {"other_heavy_processes": []})

    class FakeOperator:
        def batch(self, fixture, cell):
            result = rows()[0]
            if bad_timer:
                result["stage_seconds"]["sim"] = float("nan")
            return {
                **result,
                "cell": cell,
                "semantic": {
                    "chosen": 1,
                    "bare": 1,
                    "branches_evaluated": [0, 1],
                    "branches_skipped": [],
                    "branches": [
                        {"action": action, "state": "EVALUATED", "frames_executed": 1}
                        for action in (0, 1)
                    ],
                },
                "dummy_leaf": {
                    "calls": 1 if cell in ("C3", "C4") else 0,
                    "batch_size": 2,
                    "parameters": 12806,
                    "dtype": "float32",
                    "device": "cpu",
                    "torch_intraop_threads": 1,
                    "torch_interop_threads": 1,
                },
                "leaf_call_wall_seconds": 0.01,
            }

    monkeypatch.setattr(operator, "V0Operator", FakeOperator)
    out = tmp_path / "fake-run"
    args = SimpleNamespace(
        manifest=path, out=out, smoke=True, smoke_fixture="frozen-800", operator="test"
    )
    if bad_timer:
        with pytest.raises(ValueError, match="timer"):
            run.run(args)
        assert json.loads((out / "stopped.json").read_text())["status"] == "INVALID"
        assert json.loads((out / "invalid-batch.json").read_text())["invalid_timers"] == {
            "stage_seconds/sim": "nan"
        }
        assert not (tmp_path / "lease.json").exists()
        return
    run.run(args)
    receipt = json.loads((out / "smoke.json").read_text())
    assert receipt["canary_ok"] is True
    assert receipt["completed_batches"] == 4
    assert receipt["status"] == "PRECHECK_ONLY"
    assert len(receipt["batch_exports"]) == 4
    assert run.verify_canary(out / "smoke.json", manifest)
    assert not (tmp_path / "lease.json").exists()
