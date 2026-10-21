"""Hermetic tests for the speed-program S0 baseline harness (no checkpoint needed)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from research.speed_s0_baseline import harness as h

REPO = Path(__file__).resolve().parents[1]


def test_exclusive_stages_subtract_nested_children() -> None:
    stages = h.ExclusiveStages(("outer", "inner"))
    with stages.measure("outer"):
        time.sleep(0.02)
        with stages.measure("inner"):
            time.sleep(0.03)
    receipt = stages.receipt()
    assert receipt["inner"]["calls"] == 1
    assert receipt["outer"]["inclusive_seconds"] >= receipt["inner"]["inclusive_seconds"]
    assert receipt["outer"]["exclusive_seconds"] == pytest.approx(
        receipt["outer"]["inclusive_seconds"] - receipt["inner"]["inclusive_seconds"], abs=1e-6
    )


def test_schedule_serial_and_parallel() -> None:
    assert h.schedule(1, 3, 2) == [
        [["coverage", "plain", "instrumented", "plain", "instrumented", "plain"]]
    ]
    assert h.schedule(5, 3, 2) == [
        [["coverage"]],
        [["plain"], ["instrumented"], ["plain"], ["instrumented"], ["plain"]],
    ]
    with pytest.raises(ValueError):
        h.schedule(4, 3, 2)
    with pytest.raises(ValueError):
        h.schedule(1, 0, 0)


def test_compare_hashes_reports_missing_and_changed() -> None:
    result = h.compare_hashes({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"})
    assert result["identical"] is False
    assert result["files_compared"] == 3
    assert result["files_equal"] == 1
    assert result["differing"] == ["b", "c"]
    assert h.compare_hashes({"a": "1"}, {"a": "1"})["identical"] is True


def test_fixture_is_deterministic_and_covers_three_mixes_and_long_bodies() -> None:
    first, second = h.fixture_cases(), h.fixture_cases()
    assert [c["row"] for c in first] == [c["row"] for c in second]
    assert {(c["mix"], c["body"]) for c in first} == {(m, b) for m in h.MIXES for b in h.BODIES}
    assert len({c["row"]["world_id"] for c in first}) == 6
    assert max(h.LONG_LENGTHS) >= 800
    assert all(c["row"]["namespace"] == h.NAMESPACE for c in first)


def test_committed_reference_matches_fixture() -> None:
    reference = h.load_reference(h.DEFAULT_REFERENCE)
    assert reference["fixture"] == h.fixture_identity()
    assert len(reference["files"]) == 6 * len(h.EXPORT_SUFFIXES)
    assert all(len(v) == 64 for v in reference["files"].values())


def test_export_hashes_and_write_completed(tmp_path: Path) -> None:
    case = tmp_path / "short-scripted"
    case.mkdir()
    h.write_completed(case / "world-000000.json", {"x": 1})
    hashes = h.export_hashes(case, "short-scripted")
    assert sorted(hashes) == [
        "short-scripted/world-000000.json",
        "short-scripted/world-000000.json.complete",
        "short-scripted/world-000000.json.sha256",
    ]
    with pytest.raises(FileExistsError):
        h.write_completed(case / "world-000000.json", {"x": 2})


def _case(name: str, body: str, mix: str, wall: float, stages: dict[str, float]) -> dict:
    return {
        "case": name,
        "mix": mix,
        "body": body,
        "wall_seconds": wall,
        "decisions": 300,
        "active_row_steps": 900,
        "work": {},
        "shadow_searches": 1,
        "divergences": 0,
        "interventions": 1,
        "hero_survived_to_cap": True,
        "events": {"opponent_deaths": 1, "opponent_respawns": 1, "shadow_forks": 2},
        "exclusive_stages": {
            k: {"exclusive_seconds": v, "inclusive_seconds": v, "calls": 1}
            for k, v in stages.items()
        },
    }


def _round(kind: str, digest: float, v8: float) -> dict:
    cases = [
        _case(f"{body}-{mix}", body, mix, 10.0, {"digest": digest, "v8": v8})
        for body in h.BODIES
        for mix in h.MIXES
    ]
    return {"kind": kind, "wall_seconds": 60.0, "worlds": 6, "cases": cases, "hashes": {"a": "1"}}


@pytest.mark.parametrize(
    ("digest", "v8", "verdict"),
    [
        (3.0, 1.5, "S1_AND_S2_CONTINUE"),
        (3.0, 0.0, "S2_CUT_S1_CONTINUES"),
        (1.0, 1.0, "S2_CUT_S1_CUT"),
    ],
)
def test_summarize_whole_workload_and_d1(digest: float, v8: float, verdict: str) -> None:
    report = {
        "rounds": [
            _round("coverage", digest, v8),
            _round("plain", digest, v8),
            _round("instrumented", digest, v8),
        ]
    }
    summary = h.summarize([report], {"a": "1"})
    assert summary["identity"]["all_rounds_self_identical"] is True
    assert summary["identity"]["all_rounds_match_reference"] is True
    whole = summary["whole_workload"]
    assert whole["completed_worlds_per_hour"] == pytest.approx(6 * 3600 / 60.0)
    assert whole["active_row_steps_per_second"] == pytest.approx(5400 / 60.0)
    assert summary["d1"]["verdict_on_this_host"] == verdict
    shares = summary["pooled_exclusive_stage_share_median"]
    assert sum(shares.values()) == pytest.approx(1.0)
    assert summary["fixture_coverage"]["requirements_met"]["doom_search"] is True


def test_summarize_flags_reference_mismatch() -> None:
    report = {"rounds": [_round("coverage", 1, 1), _round("plain", 1, 1)]}
    summary = h.summarize([report], {"a": "2"})
    assert summary["identity"]["all_rounds_match_reference"] is False
    assert summary["d1"] is None


def test_colab_entrypoint_is_supervisor_only() -> None:
    script = REPO / "research/speed_s0_baseline/s0_colab.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {k: v for k, v in os.environ.items() if k != "COLAB_CLI_ROOT"}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "supervisor-only" in result.stderr


def test_engine_source_digest_is_stable() -> None:
    first, second = h.engine_source_digest(), h.engine_source_digest()
    assert first == second
    assert first["files"] > 10
    assert json.dumps(first)


def test_engine_runner_default_and_native_roots_stay_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.t34_exit_b8 import teacher_v2
    from research.t34_speed_s2 import native_v8

    config = {"device": "cpu"}
    calls = []

    def stock(value):
        calls.append(("stock", value))
        return {"complete": True}

    def collector(value):
        teacher = value["teacher"]
        assert not teacher.include_roots and not teacher.branch_only
        calls.append(("native", value))
        return {"complete": True}

    monkeypatch.setattr(teacher_v2, "run_worlds", stock)
    assert h.engine_runner()(config) == {"complete": True}
    monkeypatch.setattr(teacher_v2, "run_worlds", collector)
    result = h.engine_runner("native-v8")(config)
    assert result["native_v8"]["scope"] == "all non-root rows"
    assert [call[0] for call in calls] == ["stock", "native"]
    assert "teacher" not in config
    with pytest.raises(ValueError, match="unknown S0 engine"):
        h.engine_runner("typo")
    native_v8.disable()


def test_native_source_digest_extends_stock_closure() -> None:
    stock = h.engine_source_digest()
    native = h.engine_source_digest(engine="native-v8")
    assert native["roots"] == stock["roots"] + ["research/t34_speed_s2"]
    assert native["files"] > stock["files"]
    assert native["combined_sha256"] != stock["combined_sha256"]
