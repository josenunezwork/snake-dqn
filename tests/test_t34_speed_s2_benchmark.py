"""Verify the D3 matched workload's order, median timing and identity failure gate."""

from pathlib import Path

import pytest

from research.speed_s0_baseline import harness as h
from research.t34_speed_s2 import benchmark_s0 as b
from tests.test_speed_s0_baseline import _round


@pytest.mark.parametrize("bad_identity", [False, True])
def test_matched_benchmark_uses_medians_and_gates_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_identity: bool
) -> None:
    from collections import defaultdict

    calls = []
    indices = defaultdict(int)
    times = {"stock": [1, 100, 120, 110], "native-v8": [1, 80, 60, 70]}
    reference = {"checkpoint_sha256": "test", "fixture": h.fixture_identity(), "files": {"a": "1"}}
    monkeypatch.setattr(h, "load_reference", lambda _: reference)
    monkeypatch.setattr(h, "_sha256", lambda _: "test")
    monkeypatch.setattr(h, "warm_up", lambda *_, **__: 0)
    monkeypatch.setattr(h, "engine_source_digest", lambda **_: {})
    monkeypatch.setattr(h, "runtime_hardware", lambda: {})
    monkeypatch.setattr(h, "_git_revision", lambda: "source")

    def run(checkpoint, output, kind, *, engine):
        index = indices[engine]
        indices[engine] += 1
        calls.append((engine, kind))
        result = _round(kind, 0, 0)
        result["wall_seconds"] = times[engine][index]
        if bad_identity and engine == "native-v8" and index == 2:
            result["hashes"]["a"] = "changed"
        return result

    monkeypatch.setattr(h, "run_round", run)
    result = b.benchmark(tmp_path / "checkpoint", tmp_path, host_label="test-contended")
    assert calls == [
        ("stock", "coverage"),
        ("native-v8", "coverage"),
        ("stock", "plain"),
        ("native-v8", "plain"),
        ("native-v8", "plain"),
        ("stock", "plain"),
        ("stock", "plain"),
        ("native-v8", "plain"),
    ]
    assert result["median_wall_seconds"] == {"stock": 110, "native-v8": 70}
    assert result["whole_workload_speedup"] == pytest.approx(110 / 70)
    assert result["d3_timing_and_s0_identity_pass"] is not bad_identity


def test_identity_reports_median_instead_of_independent_minima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.t34_speed_s2 import identity

    times = {"stock": iter([1.0, 11.0, 10.0]), "native": iter([2.0, 5.0, 6.0])}

    def run(kind, *args, **kwargs):
        return {
            "wall_seconds": next(times[kind]),
            "process_cpu_seconds": 0,
            "host_stage_seconds": {},
            "deterministic": {"executed_row_steps": 10},
            "exports": {"file": "same"},
            "native_stats": {},
        }

    monkeypatch.setattr(identity, "_run", run)
    result = identity.run_fixture(
        tmp_path / "checkpoint", tmp_path, "frozen", "long", 1, repeats=3, q_ties=False, audit=False
    )
    assert result["speedup_min_wall"] == 0.5
    assert result["speedup_median_wall"] == 2.0
    assert result["ms_per_row_step"] == {"stock": 1000, "native": 500}
