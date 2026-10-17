"""Focused contract tests for the T12 B8 parity harness.

The real qualification worlds are intentionally not run in unit tests.  These
tests pin orchestration: P2 never batches worlds, and the serving probe records
only digest evidence while retaining the grid metric record.
"""

from __future__ import annotations

from research.t12_b8_serving_20261017 import parity


def test_assigned_rows_requires_the_registered_actual_seed_counts(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        parity,
        "_row_for_seed",
        lambda seed, mix: calls.append((seed, mix)) or {"world_seed": seed, "mix": mix},
    )

    assert [row["mix"] for row in parity._assigned_rows(range(9))] == [
        "frozen",
        "frozen",
        "frozen",
        "scripted",
        "scripted",
        "scripted",
        "mixed",
        "mixed",
        "mixed",
    ]
    assert calls[-3:] == [(6, "mixed"), (7, "mixed"), (8, "mixed")]


def test_b8_environment_selects_only_checkpoint_and_restores_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_WATCH_HERO", "0")
    monkeypatch.setenv("SNAKE_SERVE_VETO_PLAY_AI", "1")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")

    with parity._b8_environment():
        import os

        assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "b8-s3"
        assert "SNAKE_SERVE_VETO_WATCH_HERO" not in os.environ
        assert "SNAKE_SERVE_VETO_PLAY_AI" not in os.environ
        assert "SNAKE_SERVE_VETO_VARIANT" not in os.environ

    import os

    assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert os.environ["SNAKE_SERVE_VETO_WATCH_HERO"] == "0"
    assert os.environ["SNAKE_SERVE_VETO_PLAY_AI"] == "1"
    assert os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_run_p2_runs_one_live_and_grid_world_per_materialized_row(monkeypatch) -> None:
    rows = [
        {"mix": "frozen", "world_seed": 11, "world_identity": {"seed": 11}},
        {"mix": "scripted", "world_seed": 12, "world_identity": {"seed": 12}},
        {"mix": "mixed", "world_seed": 13, "world_identity": {"seed": 13}},
    ]
    calls = []
    monkeypatch.setattr(
        parity, "_resolve_b8", lambda: {"path": "/b8.pth", "sha256": parity.CHECKPOINT_SHA256}
    )
    monkeypatch.setattr(parity, "_context", lambda horizon: {"horizon": horizon})
    monkeypatch.setattr(parity, "_assigned_rows", lambda seeds: rows)

    def live(row, context, student, horizon):
        calls.append(("live", row["world_seed"], horizon, student))
        return {"record": {"seed": row["world_seed"]}}

    def sim(row, context, student, horizon):
        calls.append(("sim", row["world_seed"], horizon, student))
        return {"seed": row["world_seed"]}

    monkeypatch.setattr(parity, "_run_p2_live", live)
    monkeypatch.setattr(parity, "_run_p2_sim", sim)
    from research.redesign_m3_20261008 import student_record_parity as p2

    monkeypatch.setattr(p2, "compare", lambda live, sim: {"identical": True})
    result = parity.run_p2([1], 100)

    assert result["pass"] and result["compared"] == 3 and result["identical"] == 3
    assert calls == [
        ("live", 11, 100, "/b8.pth"),
        ("sim", 11, 100, "/b8.pth"),
        ("live", 12, 100, "/b8.pth"),
        ("sim", 12, 100, "/b8.pth"),
        ("live", 13, 100, "/b8.pth"),
        ("sim", 13, 100, "/b8.pth"),
    ]


def test_session_parity_reports_digest_only_trace_comparison(monkeypatch) -> None:
    monkeypatch.setattr(
        parity, "_resolve_b8", lambda: {"path": "/b8.pth", "sha256": parity.CHECKPOINT_SHA256}
    )
    monkeypatch.setattr(
        parity,
        "_row_for_seed",
        lambda seed, mix: {"mix": mix, "world_seed": seed, "world_identity": {"seed": seed}},
    )
    monkeypatch.setattr(
        parity,
        "_session_side",
        lambda *args: {"frame_digests": ["a", "b"], "served_checkpoint": {"name": "b8-s3"}},
    )
    monkeypatch.setattr(
        parity,
        "_simd_side",
        lambda *args: {"frame_digests": ["a", "b"], "record": {"mass_integral": 9.0}},
    )

    result = parity.run_session_parity(101, horizon=2, mix="mixed")

    assert result["pass"]
    assert result["frames_compared"] == 2
    assert result["first_differing_frame"] is None
    assert result["session"]["frame_digests"] == ["a", "b"]
    assert result["simd"]["frame_digests"] == ["a", "b"]
    assert result["simd"]["record"] == {"mass_integral": 9.0}


def test_session_parity_rejects_unknown_mix() -> None:
    import pytest

    with pytest.raises(ValueError, match="mix must be"):
        parity.run_session_parity(1, horizon=1, mix="wrong")
