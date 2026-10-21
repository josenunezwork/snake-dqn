"""Focused WP-2 coverage and fail-closed receipt/decision regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from research.bl_k1 import native
from research.bl_k1.audit import compare, stable, world_state
from research.bl_k1.timing import decision
from research.bl_served_search.exports import export_json, verify_export
from research.t34_speed_s2 import native_v8 as s2


def test_scope_stock_and_kernel_lifecycle() -> None:
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policy = SimpleNamespace(
        veto_variant="v8",
        runtime=object(),
        veto_spec=SimpleNamespace(lam=8.0, reference_lambda=None),
    )
    rows = np.array([[0, 1], [0, 2]])
    for _ in range(2):
        stats = native.RowStats()
        with patch.object(native, "_kernel_hook", return_value=np.array([1, 2])) as kernel:
            with native.configure("kernel", stats):
                with native.branch_scope("frozen"):
                    Vector61SimdPolicy._apply_veto_live_hook(policy, object(), rows)
            kernel.assert_called_once()
        s2.disable()


def test_kernel_guard_and_last_veto_without_replay() -> None:
    sim = SimpleNamespace(frame=np.array([12]), length=np.array([[900, 900]]))
    policy = SimpleNamespace(
        _rule_config_checked=sim,
        live_vetoes={},
        veto_spec=SimpleNamespace(lam=8.0),
        runtime=SimpleNamespace(featurizer=SimpleNamespace(gw=10, gh=10)),
    )
    counts = np.array([[10, 11, 12]])
    base = np.array([1])
    result = SimpleNamespace(
        action=np.array([2]), base=base, counts=counts, cap=np.array([160]), need=np.array([160])
    )
    args = (
        policy,
        sim,
        np.array([[0, 1]]),
        np.array([0]),
        counts,
        result.cap,
        result.need,
        np.ones((1, 3), bool),
        np.zeros((1, 6), np.float32),
        np.ones((1, 6), bool),
        base,
    )
    stats = native.RowStats()
    with (
        patch("src.simd_env.v8_nb.v8_decide", return_value=result),
        patch.object(s2, "_replay_row", side_effect=AssertionError("must skip replay")),
    ):
        assert native._kernel_hook(*args, stats=stats).tolist() == [2]
        assert stats.kernel_rows == stats.rows == 1
        assert stats.fallback_rows == stats.replayed_rows == 0
        assert not policy.live_vetoes
        assert policy.last_veto["base"].tolist() == [1]
        result.counts = counts + 1
        with pytest.raises(s2.NativeV8Mismatch, match="counts"):
            native._kernel_hook(*args)


def test_diagnostics_exclusion_is_mode_specific() -> None:
    reference = {"q": 1, "actions": 2, "carry": 3, "diagnostics": 4}
    other = {**reference, "diagnostics": 5}
    assert compare(reference, other, "kernel") == []
    assert compare(reference, other, "replay") == ["diagnostics"]
    other["carry"] = 6
    assert compare(reference, other, "kernel") == ["carry"]


def test_frozen_fivefold_bar_partial_and_boundary() -> None:
    assert decision([20.0] * 30, [4.0] * 30, True)["status"] == "K1_PASS"
    assert decision([20.0] * 30, [4.01] * 30, True)["wp4"] == "CANCELLED"
    assert decision([20.0] * 29, [4.0] * 30, True)["status"] == "NOT_READY"
    assert decision([20.0] * 30, [4.0] * 30, False)["status"] == "NOT_READY"
    with pytest.raises(ValueError):
        decision([float("nan")] * 30, [4.0] * 30, True)


def test_exports_create_only_and_tamper(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    export_json(path, {"status": "PRECHECK_ONLY"})
    assert verify_export(path)
    with pytest.raises(FileExistsError):
        export_json(path, {"status": "PASS"})
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        verify_export(path)


def test_rollout_isolates_policy_carry_and_leaves_root_stock() -> None:
    from research.t3_rollout_labels_20261015 import bank_engine as bank

    class World:
        mix = "frozen"
        sim = SimpleNamespace(alive=np.array([[True]]), length=np.array([[900]]))
        hero = SimpleNamespace(last_veto={"root": True})
        opponents = [SimpleNamespace(live_vetoes={"counter": 0})]

        def clone(self):
            other = copy.copy(self)
            other.hero = copy.copy(self.hero)
            other.opponents = [copy.copy(policy) for policy in self.opponents]
            return other

    root = World()
    observed = []

    def step(branch, action):
        branch.opponents[0].live_vetoes["counter"] += 1
        observed.append(branch.opponents[0].live_vetoes["counter"])
        assert native._LOCAL.active

    with native.configure("kernel"), patch.object(bank, "step_prepared", step):
        assert len(bank.rollout_all(root, [0, 1], depth=1)) == 2
    assert observed == [1, 1]
    assert root.opponents[0].live_vetoes == {"counter": 0}
    assert not native._LOCAL.active


def test_object_world_arrays_compare_contents_not_pointer_addresses() -> None:
    a, b = np.empty((1,), object), np.empty((1,), object)
    a[0], b[0] = [1, 2], [1, 2]
    assert a.tobytes() != b.tobytes()
    assert stable(a) == stable(b)
    b[0] = [1, 3]
    assert stable(a) != stable(b)


def test_timing_admission_rejects_partial_and_foreign_identity(tmp_path) -> None:
    import platform

    from research.bl_k1.audit import verify_identity

    value = {
        "status": "ACTION_IDENTITY_PASS",
        "manifest_sha256": "a" * 64,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "threads": {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "completed": 1,
        "cases": [],
    }
    path = tmp_path / "partial.json"
    export_json(path, value)
    with pytest.raises(ValueError, match="294"):
        verify_identity(path, {"fixtures": [], "fresh_roots": []}, "a" * 64)
    other = tmp_path / "foreign.json"
    value["platform"]["system"] = "Linux"
    export_json(other, value)
    with pytest.raises(ValueError, match="same Mac"):
        verify_identity(other, {"fixtures": [], "fresh_roots": []}, "a" * 64)


def test_world_trace_binds_authoritative_food_order() -> None:
    sim = SimpleNamespace(
        food_cells=[[(1, 2), (3, 4)]], corpse_cells=[{(1, 2)}], food_set=[{(1, 2), (3, 4)}]
    )
    first = stable(world_state(sim))
    sim.food_cells[0].reverse()
    assert first != stable(world_state(sim))
