"""Focused contracts for the P3 production-path identity harness."""

from __future__ import annotations

import time
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


def test_validate_row_uses_analyzer_for_requested_stock_mode(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        mac_identity.analyze,
        "_row_error",
        lambda row, expected, **kwargs: seen.append((row, expected, kwargs)) or None,
    )
    expected = {"world_id": "x", "plan": {"seed": 1}}
    row = {"world_id": "x", "native_v8": {"mode": "stock", "calls": 0}}
    mac_identity.validate_row(row, expected, mode="all")
    assert seen == [(row, expected, {"search_mode": "all", "native_mode": "stock"})]


def test_validate_row_refuses_any_nonzero_veto_counter(monkeypatch) -> None:
    monkeypatch.setattr(mac_identity.analyze, "_row_error", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="nonzero veto counter"):
        mac_identity.validate_row(
            {"world_id": "x", "native_v8": {"mode": "stock", "calls": 1}},
            {"world_id": "x", "plan": {"seed": 1}},
            mode="all",
        )


def test_load_allocation_requires_audited_round_one_identity(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "p3-identity-02"
    out.mkdir()
    rows = [
        {
            **row,
            "plan": {
                "seed": row["world_seed"],
                "mix": row["mix"],
                "specs": [["x", "y"]] * 5,
                "member_sha256s": ["a" * 64] * 5,
            },
        }
        for row in mac_identity.derive(1)
    ]
    audit = {
        "schema": "bl-served-search-seed-audit/v1",
        "status": "CLEAR",
        "namespace": mac_identity.NAMESPACE,
        "round_index": 1,
        "candidate_count": 5,
        "unique_count": 5,
        "history": {"never_opened": mac_identity.bank.SEALED},
    }
    mac_identity.export_json(out / "seed-audit-round-01.json", audit)
    mac_identity.export_json(
        out / "allocation.json",
        {
            "schema": "bl-served-search-mac-identity-allocation/v1",
            "namespace": mac_identity.NAMESPACE,
            "round_index": 1,
            "development_only": True,
            "horizon": 5000,
            "mix_counts": {"frozen": 2, "mixed": 2, "scripted": 1},
            "seed_audit_sha256": mac_identity.digest(out / "seed-audit-round-01.json"),
            "rows": rows,
        },
    )
    monkeypatch.setattr(
        mac_identity, "P3_ALLOCATION_SHA256", mac_identity.digest(out / "allocation.json")
    )
    loaded, loaded_audit = mac_identity.load_allocation(out / "allocation.json")
    assert loaded == rows
    assert loaded_audit == audit


def test_mode_a_difference_leaves_valid_mode_b_ready(tmp_path: Path, monkeypatch) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"pinned")
    rows = [
        {
            "world_id": f"dev/{mix}",
            "mix": mix,
            "dense": {"batches": 1, "branches_evaluated": 1, "branches_skipped": 0},
            "native_v8": {
                "mode": "stock",
                "calls": 0,
                "kernel_rows": 0,
                "stock_rows": 0,
                "fallback_rows": 0,
                "noncandidate_rows": 0,
                "python_calls": 0,
                "root_stock_rows": 0,
                "rows": 0,
                "replayed_rows": 0,
                "tail_release_rows": 0,
                "audited_rows": 0,
                "overrides": 0,
                "v5_changes": 0,
                "v7_changes": 0,
                "head_changes": 0,
                "no_spacious": 0,
                "tie_rows": 0,
                "tie_overrides": 0,
                "boost_base_rows": 0,
                "long_rows": 0,
            },
        }
        for mix in ("frozen", "mixed", "scripted")
    ]
    monkeypatch.setattr(
        mac_identity.ceiling, "digest_file", lambda path: mac_identity.ceiling.B8_SHA256
    )
    monkeypatch.setattr(
        mac_identity,
        "load_allocation",
        lambda path: (
            rows,
            {
                "status": "CLEAR",
                "round_index": 1,
                "history": {"never_opened": mac_identity.bank.SEALED},
                "coverage": "test",
            },
        ),
    )
    allocation = tmp_path / "allocation.json"
    mac_identity.export_json(
        allocation.parent / "seed-audit-round-01.json",
        {
            "status": "CLEAR",
            "round_index": 1,
            "history": {"never_opened": mac_identity.bank.SEALED},
            "coverage": "test",
        },
    )

    def run_mode(rows, *, mode, **kwargs):
        if mode == "exact_stop":
            raise ValueError("chosen action mismatch: dev/frozen")
        return rows

    monkeypatch.setattr(mac_identity, "_run_mode", run_mode)
    receipt = mac_identity.run(
        tmp_path / "out",
        b8_path=b8,
        deadline=time.monotonic() + 10,
        allocation=allocation,
    )
    assert receipt["status"] == "READY_MODE_B"
    assert receipt["readiness"]["mode_b"]["status"] == "READY"
    assert receipt["readiness"]["mode_a"]["status"] == "NOT_READY"
    assert receipt["permitted_search_modes"] == ["all"]


def test_malformed_mode_b_writes_not_ready_receipt(tmp_path: Path, monkeypatch) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"pinned")
    allocation = tmp_path / "allocation.json"
    mac_identity.export_json(
        allocation.parent / "seed-audit-round-01.json",
        {
            "status": "CLEAR",
            "round_index": 1,
            "history": {"never_opened": mac_identity.bank.SEALED},
            "coverage": "test",
        },
    )
    monkeypatch.setattr(
        mac_identity.ceiling, "digest_file", lambda path: mac_identity.ceiling.B8_SHA256
    )
    monkeypatch.setattr(
        mac_identity,
        "load_allocation",
        lambda path: (
            [{"world_id": "dev/frozen"}],
            {
                "status": "CLEAR",
                "round_index": 1,
                "history": {"never_opened": mac_identity.bank.SEALED},
                "coverage": "test",
            },
        ),
    )
    monkeypatch.setattr(
        mac_identity,
        "_run_mode",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            mac_identity.ModeRunError(
                "malformed row",
                [
                    {
                        "world_id": "dev/frozen",
                        "mix": [],
                        "native_v8": {"mode": "stock", "calls": 0},
                    }
                ],
            )
        ),
    )
    with pytest.raises(RuntimeError, match="P3 NOT_READY"):
        mac_identity.run(
            tmp_path / "out",
            b8_path=b8,
            deadline=time.monotonic() + 10,
            allocation=allocation,
        )
    receipt = __import__("json").loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "NOT_READY"
    assert receipt["execution_counters"]["all"]["valid_rows"] == 0
    assert receipt["execution_counters"]["all"]["rejected_rows"]


def test_malformed_mode_a_preserves_valid_mode_b_receipt(tmp_path: Path, monkeypatch) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"pinned")
    allocation = tmp_path / "allocation.json"
    mac_identity.export_json(
        allocation.parent / "seed-audit-round-01.json",
        {
            "status": "CLEAR",
            "round_index": 1,
            "history": {"never_opened": mac_identity.bank.SEALED},
            "coverage": "test",
        },
    )
    valid = [
        {
            "world_id": f"dev/{mix}",
            "mix": mix,
            "dense": {"batches": 1, "branches_evaluated": 1, "branches_skipped": 0},
            "native_v8": {"mode": "stock", **dict.fromkeys(mac_identity._VETO_COUNTER_KEYS, 0)},
        }
        for mix in ("frozen", "mixed", "scripted")
    ]
    monkeypatch.setattr(
        mac_identity.ceiling, "digest_file", lambda path: mac_identity.ceiling.B8_SHA256
    )
    monkeypatch.setattr(
        mac_identity,
        "load_allocation",
        lambda path: (
            valid,
            {
                "status": "CLEAR",
                "round_index": 1,
                "history": {"never_opened": mac_identity.bank.SEALED},
                "coverage": "test",
            },
        ),
    )

    def run_mode(rows, *, mode, **kwargs):
        if mode == "exact_stop":
            raise mac_identity.ModeRunError(
                "malformed row",
                [{"world_id": "dev/frozen", "native_v8": {"mode": "stock", "calls": 0}}],
            )
        return rows

    monkeypatch.setattr(mac_identity, "_run_mode", run_mode)
    receipt = mac_identity.run(
        tmp_path / "out", b8_path=b8, deadline=time.monotonic() + 10, allocation=allocation
    )
    assert receipt["status"] == "READY_MODE_B"
    assert receipt["rows"]["exact_stop"] == [
        {"world_id": "dev/frozen", "native_v8": {"mode": "stock", "calls": 0}}
    ]
    assert receipt["execution_counters"]["exact_stop"]["valid_rows"] == 0
    assert receipt["execution_counters"]["exact_stop"]["rejected_rows"]
