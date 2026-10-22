"""Focused contracts for the P3 production-path identity harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_served_search import mac_identity


def test_derivation_has_required_five_h5000_mix_and_unique_seeds() -> None:
    rows = mac_identity.derive(0)
    assert [row["mix"] for row in rows] == list(mac_identity.MIXES)
    assert {row["horizon"] for row in rows} == {5000}
    assert {row["pool"] for row in rows} == {"selection"}
    assert len({row["world_seed"] for row in rows}) == len(rows) == 5
    assert rows == mac_identity.derive(0)
    assert not {row["world_seed"] for row in rows} & {
        row["world_seed"] for row in mac_identity.derive(1)
    }


def test_allocate_rejects_whole_round_without_seed_swaps(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[dict]] = []

    def audit(rows: list[dict]) -> dict:
        calls.append(rows)
        return {
            "status": "REJECTED" if len(calls) == 1 else "CLEAR",
            "history": {"never_opened": "t34-round0-study-01"},
            "coverage": "visible banks",
        }

    monkeypatch.setattr(mac_identity.bank, "audit", audit)
    monkeypatch.setattr(mac_identity, "materialize", lambda rows: rows)
    rows, report = mac_identity.allocate(tmp_path / "out")
    assert report["round_index"] == 1
    assert calls == [mac_identity.derive(0), mac_identity.derive(1)]
    assert rows == mac_identity.derive(1)
    assert (tmp_path / "out/seed-audit-round-00.json.complete").is_file()


def test_p3_materializer_never_calls_roster_planner_with_an_empty_pool() -> None:
    calls = []

    def build(context, seeds, mixes):
        assert context == "context"
        assert seeds and len(seeds) == len(mixes) == 5
        calls.append((seeds, mixes))
        return [{"seed": seed, "mix": mix} for seed, mix in zip(seeds, mixes)]

    rows = mac_identity.materialize(
        mac_identity.derive(0), context_factory=lambda threads: "context", roster_builder=build
    )
    assert len(calls) == 1
    assert [row["plan"] for row in rows] == [
        {"seed": row["world_seed"], "mix": row["mix"]} for row in rows
    ]


def test_compare_rows_ignores_only_execution_diagnostics() -> None:
    base = {
        "world_id": "dev/one",
        "search_mode": "all",
        "native_v8": {"kernel_rows": 10},
        "worker_threads": {"torch": 1},
        "dense": {
            "batch_diagnostics": [{"decision": 1, "chosen": 2}],
            "branches_evaluated": 6,
            "branches_skipped": 0,
            "wall_seconds": 1.0,
        },
        "trajectories": {"baseline_mass": [1, 0], "dense_mass": [2, 0]},
        "paired_delta": {"mass_integral": 0.5},
    }
    stopped = {
        **base,
        "search_mode": "exact_stop",
        "native_v8": {"kernel_rows": 3},
        "dense": {
            **base["dense"],
            "batch_diagnostics": [{"decision": 1, "chosen": 2}],
            "branches_evaluated": 3,
            "branches_skipped": 3,
            "wall_seconds": 9.0,
        },
    }
    mac_identity.compare_rows(base, stopped)
    stopped["trajectories"] = {"baseline_mass": [1, 0], "dense_mass": [3, 0]}
    with pytest.raises(ValueError, match="scientific row mismatch"):
        mac_identity.compare_rows(base, stopped)


def test_validate_row_uses_analyzer_for_requested_mode(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        mac_identity.analyze,
        "_row_error",
        lambda row, expected, **kwargs: seen.append((row, expected, kwargs)) or None,
    )
    expected = {"world_id": "x", "plan": {"seed": 1}}
    mac_identity.validate_row({"world_id": "x"}, expected, mode="all")
    assert seen == [({"world_id": "x"}, expected, {"search_mode": "all", "native_mode": "kernel"})]
