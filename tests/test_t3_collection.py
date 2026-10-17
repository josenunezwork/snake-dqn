"""Focused contract tests for the Part-D dense label collector."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.t3_rollout_labels_20261015 import bank_engine as bank
from tests.t14_fixtures import evaluator_context, install_train_layout_context


def _parity_seeds(count: int) -> list[int]:
    """Dedicated fixture namespace; never consume the Part-D label namespace."""
    from research.apex_safety_20260926 import dev_screen

    return [
        int(seed)
        for seed in dev_screen.screen_seeds(
            count, "t3-runtime-parity-20261015", "t3-runtime-parity-20261015/v1"
        )
    ]


def _runtime_arrays(runtime: Any) -> dict[str, np.ndarray]:
    """Copy every mutable runtime array, including Vector61 trend memory."""
    arrays = {
        name: value.copy() for name, value in vars(runtime).items() if isinstance(value, np.ndarray)
    }
    if runtime.memory is not None:
        arrays.update(
            {
                f"memory.{name}": value.copy()
                for name, value in vars(runtime.memory).items()
                if isinstance(value, np.ndarray)
            }
        )
    return arrays


def _assert_runtime_arrays_unchanged(runtime: Any, expected: dict[str, np.ndarray]) -> None:
    """Assert the source clone did not share or mutate Vector61 carry state."""
    for name, before in expected.items():
        target = runtime.memory if name.startswith("memory.") else runtime
        attr = name.removeprefix("memory.")
        np.testing.assert_array_equal(getattr(target, attr), before)


def _row(decisive: bool, action: int = 1) -> dict:
    return {
        "local": np.zeros((7, 31, 31), np.uint8),
        "global": np.zeros((4, 37, 37), np.uint8),
        "scalars": np.zeros(60, np.float32),
        "mask": np.ones(6, bool),
        "surviving": np.array([False, True, False, False, False, False]),
        "action": action,
        "decisive": decisive,
        "decision": 2000,
        "frame": 2000,
    }


def test_cycle_has_exact_training_mix_and_all_finals() -> None:
    jobs = bank.cycle_jobs(1)
    assert len(jobs) == 200
    for arm in {job["arm"] for job in jobs}:
        mixes = [job["mix"] for job in jobs if job["arm"] == arm]
        assert mixes.count("scripted") == 27
        assert mixes.count("mixed") == 9
        assert mixes.count("frozen") == 4


def test_concrete_seed_scan_uses_exact_numeric_words(tmp_path: Path) -> None:
    root = tmp_path / "scan"
    root.mkdir()
    (root / "hits.txt").write_text("101 10011 303\n")
    assert bank.concrete_seed_hits(root, [101, 202, 303]) == {101, 303}


def test_seed_allocation_deterministically_rejects_preexisting_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    repo.mkdir()
    artifacts.mkdir()
    (repo / "prior.txt").write_text("1001 1002\n")
    (artifacts / "prior.txt").write_text("1003 1004\n")
    candidates = list(range(1001, 1013))
    monkeypatch.setattr(bank, "fresh_seeds", lambda count: candidates[:count])

    selected, receipt = bank.allocate_unseen_seeds(3, (repo, artifacts))

    assert selected == [1005, 1006, 1007]
    assert receipt["selected_seed_count"] == 3
    assert receipt["rejected_seeds"] == [
        {"seed": 1001, "roots": [str(repo.resolve())]},
        {"seed": 1002, "roots": [str(repo.resolve())]},
        {"seed": 1003, "roots": [str(artifacts.resolve())]},
        {"seed": 1004, "roots": [str(artifacts.resolve())]},
    ]
    assert receipt["candidate_batches"] == [
        {
            "candidate_prefix": 6,
            "new_candidates": 6,
            "matched_by_root": {
                str(repo.resolve()): [1001, 1002],
                str(artifacts.resolve()): [1003, 1004],
            },
        },
        {
            "candidate_prefix": 12,
            "new_candidates": 6,
            "matched_by_root": {str(repo.resolve()): [], str(artifacts.resolve()): []},
        },
    ]


def test_dense_trigger_is_anchored_then_gated_by_frame() -> None:
    from research.t3_rollout_labels_20261015.trigger import DenseTrigger

    trigger = DenseTrigger(700, 0, 5)
    assert not bank.dense_eligible(trigger, 10, 1999, 700, True)
    assert trigger.first_decision == 10
    assert bank.dense_eligible(trigger, 15, 2000, 700, True)
    assert not bank.dense_eligible(trigger, 16, 2001, 700, True)


def test_horizon_guard_rejects_truncated_d300_labels() -> None:
    assert bank.full_horizon(4701)
    assert not bank.full_horizon(4702)


def test_classification_requires_b8_death_for_decisive() -> None:
    outcomes = {0: {"survives": True}, 1: {"survives": False}, 3: {"survives": True}}
    decisive, surviving = bank.classify(outcomes, 1)
    assert decisive and surviving.tolist() == [True, False, False, True, False, False]
    decisive, surviving = bank.classify({1: {"survives": True}, 2: {"survives": False}}, 1)
    assert not decisive and surviving[1] and not surviving[2]


def test_counterweight_cap_is_global_and_deterministic() -> None:
    selected = bank.counterweight_trim(
        [_row(False), _row(True), _row(False), _row(True), _row(False)]
    )
    assert [bool(row["decisive"]) for row in selected] == [True, True, False, False]


def test_raw_shard_is_loader_compatible(tmp_path: Path) -> None:
    bank.write_rows(tmp_path / "bank.npz", [_row(True)])
    from research.t3_rollout_labels_20261015.bank import LabelBank

    loaded = LabelBank(tmp_path, nscal=60)
    assert loaded.size == 1
    assert loaded.surviving[0, 1]


def test_packed_shard_uses_the_trainer_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_train_layout_context(monkeypatch)
    bank.write_rows(tmp_path / "bank.npz", [_row(True)], packed=True)
    from research.redesign_m3c_20261008.obs_pack import train_layout
    from research.t3_rollout_labels_20261015.bank import LabelBank

    loaded = LabelBank(tmp_path, nscal=60, pack=train_layout())
    assert loaded.size == 1
    assert loaded.local.shape == (1, 7, 31, 31)


def test_world_shard_preserves_merge_class(tmp_path: Path) -> None:
    path = tmp_path / "world.npz"
    bank.write_world_rows(path, [_row(True), _row(False)])
    with np.load(path) as data:
        assert data["decisive"].tolist() == [True, False]


def test_empty_world_retains_metadata_schema(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    bank.write_world_rows(path, [])
    with np.load(path) as data:
        assert data["q"].shape == (0, 6)
        assert data["q"].dtype == np.float32
        assert all(data[key].shape == (0,) for key in ("length", "frame", "decision"))


def test_export_retains_calibration_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_train_layout_context(monkeypatch)
    row = _row(True) | {"q": np.arange(6, dtype=np.float32), "length": 1001}
    bank.write_rows(tmp_path / "bank.npz", [row], packed=True)
    with np.load(tmp_path / "bank.npz") as data:
        np.testing.assert_array_equal(data["q"], row["q"][None])
        assert data["length"].tolist() == [1001]


def test_resume_refuses_a_mismatched_receipt_before_world_build(tmp_path: Path) -> None:
    from research.t3_rollout_labels_20261015 import collect

    job = {"arm": "b8_s0", "mix": "scripted", "cycle": 0, "index": 0}
    plan = {"seed": 1234, "mix": "scripted", "world_index": 0, "specs": [], "member_sha256s": []}
    key = "b8_s0-scripted-c0-i0-s1234"
    rows = tmp_path / "worlds" / f"{key}.npz"
    bank.write_world_rows(rows, [])
    receipt = {
        "key": key,
        "arm": "b8_s0",
        "mix": "scripted",
        "cycle": 0,
        "seed": 1234,
        "roster": plan,
        "run_identity": {"wrong": True},
        "rows_sha256": bank.sha256(rows),
    }
    (tmp_path / "worlds" / f"{key}.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        collect.collect_world(job, plan, tmp_path, {"right": True})


def test_planned_rosters_preserve_ordered_pool_rotation(tmp_path: Path) -> None:
    """Regression: one-world materialization incorrectly made every index zero."""
    context, _ = evaluator_context(tmp_path)
    seeds = _parity_seeds(3)
    plans = bank.planned_rosters(context, seeds, ("frozen", "frozen", "frozen"))
    assert [plan["world_index"] for plan in plans] == [0, 1, 2]
    assert plans[0]["member_sha256s"] != plans[1]["member_sha256s"]
    assert all(plan["seed"] == seed for plan, seed in zip(plans, seeds))
    singleton_plans = [bank.planned_rosters(context, [seed], ("frozen",))[0] for seed in seeds]
    assert len({tuple(plan["member_sha256s"]) for plan in singleton_plans}) == 1


@pytest.mark.skipif(
    os.environ.get("T3_RUNTIME_PARITY") != "1",
    reason="bounded evaluator parity is run only after dense workers have exited",
)
@pytest.mark.parametrize("mix", ("scripted", "mixed", "frozen"))
def test_wrapper_matches_official_evaluator_for_bounded_trace(mix: str) -> None:
    """Compare actions and full-world digests against independent evaluator execution.

    Enable deliberately with ``T3_RUNTIME_PARITY=1`` only after the dense
    ceiling workers have exited.  This is an evaluator qualification, not a
    normal unit test: it builds one H5000-profile world per mix but steps 32
    frames only.
    """
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine as engine
    from src.simd_env import eval_engine as ee

    context = gid._context(1)
    seed = _parity_seeds(1)[0]
    plan = bank.planned_rosters(context, [seed], [mix])[0]
    official_actions, official_digests = [], []
    cls = ee._terminal_hero_sim_class("grid")
    original = cls.step_with_policy

    def traced(self: Any, selector: Any, *args: Any, **kwargs: Any) -> Any:
        def trace_selector(prepared: Any) -> np.ndarray:
            actions = selector(prepared)
            official_actions.append(np.asarray(actions[0], dtype=np.int64).copy())
            return actions

        result = original(self, trace_selector, *args, **kwargs)
        official_digests.append(gid.env_digest(self, 0))
        return result

    cls.step_with_policy = traced
    try:
        ee.run_simd_eval(
            context["lookup"][gid.FRP3_S12[1]],
            plan["specs"],
            32,
            [seed],
            profile=replace(context["profile"], scored_horizon=32),
            opponent_specs_by_world={seed: plan["specs"]},
            mix_id=mix,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            hero_ego2s=str(engine.census.b8_path("b8_s0")),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            vector61_backend="numba",
            scripted_anchors="fast",
        )
    finally:
        cls.step_with_policy = original
    world = engine.build_world("b8_s0", plan)
    actions, digests = [], []
    for _ in range(32):
        choice, _, _ = engine.choose_and_step(world, capture_obs=False)
        actions.append(choice[0].copy())
        digests.append(gid.env_digest(world.sim, 0))
    np.testing.assert_array_equal(np.asarray(actions), np.asarray(official_actions))
    assert digests == official_digests


@pytest.mark.skipif(
    os.environ.get("T3_RUNTIME_PARITY") != "1",
    reason="bounded evaluator parity is run only after dense workers have exited",
)
@pytest.mark.parametrize("mix", ("mixed", "frozen"))
def test_prepared_clone_rollouts_are_isolated(mix: str) -> None:
    """All D16 forced labels match independent official runs and preserve source state."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine as engine
    from src.simd_env import eval_engine as ee

    context = gid._context(1)
    seed = _parity_seeds(1)[0]
    plan = bank.planned_rosters(context, [seed], [mix])[0]
    world = engine.build_world("b8_s0", plan)
    for _ in range(8):
        engine.choose_and_step(world, capture_obs=False)
    snapshot = {}
    original_prepare = world.runtime.prepare

    def capture_prepared(sim: Any) -> None:
        original_prepare(sim)
        snapshot["branch"] = world.clone()

    world.runtime.prepare = capture_prepared
    try:
        _, advisory, _ = engine.choose_and_step(world, capture_obs=False)
    finally:
        world.runtime.prepare = original_prepare
    branch = snapshot["branch"]
    arrays = _runtime_arrays(branch.runtime)
    digest_before = gid.env_digest(branch.sim, 0)
    legal = [int(action) for action in np.flatnonzero(advisory)]
    outcomes = engine.rollout_all(branch, legal, depth=16, decision=8)
    assert set(outcomes) == set(legal)
    _assert_runtime_arrays_unchanged(branch.runtime, arrays)
    assert gid.env_digest(branch.sim, 0) == digest_before

    def official_survives(forced: int) -> bool:
        """A fresh evaluator applies one forced action at frame eight."""
        cls = ee._terminal_hero_sim_class("grid")
        original = cls.step_with_policy
        alive = []
        frame = 0

        def traced(self: Any, selector: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal frame

            def force_selector(prepared: Any) -> np.ndarray:
                actions = selector(prepared)
                if frame == 8:
                    actions = np.asarray(actions, dtype=np.int64).copy()
                    actions[0, 0] = forced
                return actions

            result = original(self, force_selector, *args, **kwargs)
            alive.append(bool(self.alive[0, 0]))
            frame += 1
            return result

        cls.step_with_policy = traced
        try:
            ee.run_simd_eval(
                context["lookup"][gid.FRP3_S12[1]],
                plan["specs"],
                24,
                [seed],
                profile=replace(context["profile"], scored_horizon=24),
                opponent_specs_by_world={seed: plan["specs"]},
                mix_id=mix,
                vector61=True,
                vector61_forward="rowwise",
                sim_engine="grid",
                hero_ego2s=str(engine.census.b8_path("b8_s0")),
                ego2s_forward="rowwise",
                ego2s_mask="advisory",
                vector61_backend="numba",
                scripted_anchors="fast",
            )
        finally:
            cls.step_with_policy = original
        return alive[23]

    for action in legal:
        assert outcomes[action]["survives"] == official_survives(action)


@pytest.mark.skipif(
    os.environ.get("T3_RUNTIME_PARITY") != "1",
    reason="late D300 evaluator parity is run only after dense workers have exited",
)
@pytest.mark.parametrize("mix", ("scripted", "mixed", "frozen"))
def test_late_prepared_d300_all_legal_matches_independent_evaluator(mix: str) -> None:
    """Qualify prepared runtime carry/respawn state at decision 2500 through D300.

    This deliberately does real long continuation work only when explicitly
    enabled.  It tries a fixed, separate parity namespace in order and requires
    an alive B8 state rather than silently skipping an unsuitable fixture.
    """
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine as engine
    from src.simd_env import eval_engine as ee

    decision, depth = 2500, 300
    context = gid._context(1)
    seeds = _parity_seeds(3)
    plans = bank.planned_rosters(context, seeds, [mix] * len(seeds))
    world = plan = None
    for candidate in plans:
        trial = engine.build_world("b8_s0", candidate)
        for _ in range(decision):
            engine.choose_and_step(trial, capture_obs=False)
            if not bool(trial.sim.alive[0, 0]):
                break
        if bool(trial.sim.alive[0, 0]):
            world, plan = trial, candidate
            break
    assert (
        world is not None and plan is not None
    ), "no fixed parity fixture survived to decision 2500"

    captured = {}
    original_prepare = world.runtime.prepare

    def capture_prepared(sim: Any) -> None:
        original_prepare(sim)
        captured["branch"] = world.clone()

    world.runtime.prepare = capture_prepared
    try:
        _, advisory, _ = engine.choose_and_step(world, capture_obs=False)
    finally:
        world.runtime.prepare = original_prepare
    branch = captured["branch"]
    legal = [int(action) for action in np.flatnonzero(advisory)]
    assert legal, "an alive prepared B8 state must retain an advisory legal action"
    digest_before = gid.env_digest(branch.sim, 0)
    arrays_before = _runtime_arrays(branch.runtime)
    outcomes = engine.rollout_all(branch, legal, depth=depth, decision=decision)
    assert set(outcomes) == set(legal)
    assert gid.env_digest(branch.sim, 0) == digest_before
    _assert_runtime_arrays_unchanged(branch.runtime, arrays_before)

    def official_survives(forced: int) -> bool:
        """Force the first action at decision 2500 in an independent H5000 run."""
        cls = ee._terminal_hero_sim_class("grid")
        original = cls.step_with_policy
        step, result = 0, {}

        class StopAfterDepth(Exception):
            pass

        def traced(self: Any, selector: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal step

            def force_selector(prepared: Any) -> np.ndarray:
                actions = selector(prepared)
                if step == decision:
                    actions = np.asarray(actions, dtype=np.int64).copy()
                    actions[0, 0] = forced
                return actions

            transition = original(self, force_selector, *args, **kwargs)
            if step == decision + depth - 1:
                result["survives"] = bool(self.alive[0, 0])
                raise StopAfterDepth
            step += 1
            return transition

        cls.step_with_policy = traced
        try:
            ee.run_simd_eval(
                context["lookup"][gid.FRP3_S12[1]],
                plan["specs"],
                engine.HORIZON,
                [plan["seed"]],
                profile=context["profile"],
                opponent_specs_by_world={plan["seed"]: plan["specs"]},
                mix_id=mix,
                vector61=True,
                vector61_forward="rowwise",
                sim_engine="grid",
                hero_ego2s=str(engine.census.b8_path("b8_s0")),
                ego2s_forward="rowwise",
                ego2s_mask="advisory",
                vector61_backend="numba",
                scripted_anchors="fast",
            )
        except StopAfterDepth:
            pass
        finally:
            cls.step_with_policy = original
        assert "survives" in result
        return bool(result["survives"])

    for action in legal:
        assert outcomes[action]["survives"] == official_survives(action)


def test_b8_controls_check_exact_d300_boundary_and_death_decision() -> None:
    assert bank.validate_b8_controls([(2000, True, 300)], 2300) == 1
    assert bank.validate_b8_controls([(2000, False, 300)], 2299) == 1
    assert bank.validate_b8_controls([(2000, True, 300)], None) == 1
    with pytest.raises(RuntimeError, match="survival"):
        bank.validate_b8_controls([(2000, True, 300)], 2299)
    with pytest.raises(RuntimeError, match="death frame"):
        bank.validate_b8_controls([(2000, False, 299)], 2299)
