"""Focused tests for the sandbox-local GPU training roster adapter."""

from __future__ import annotations

from collections import Counter

import torch

from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    MIX_IDS,
    roster_ids,
    training_rosters,
)


def test_balanced_training_rosters_retain_gate_member_identities() -> None:
    """Fresh rows use the gate source construction, never a saved gate bank."""
    rosters = training_rosters(10, namespace="a100-dev/test-training-rosters/v1")
    assert Counter(row.mix for row in rosters) == {"scripted": 4, "frozen": 3, "mixed": 3}
    references = {reference for row in rosters for _kind, reference in row.slots}
    assert FRP3_S12_SHA256 in references
    assert CHAMPION_SHA256 in references
    for row in rosters:
        assert len(row.slots) == 5
        assert row.veto == "v8"
        assert row.veto_lambda == 8.0


def test_local_descriptor_matches_cpu_gate_oracle_for_short_all_mix_prefix() -> None:
    """The protected source is a test oracle only, never a runtime dependency."""
    from research.t34_round0.bank import _rows

    local = training_rosters(6, namespace="a100-dev/test-training-roster-oracle/v1")
    oracle_rows = _rows(
        "a100-dev/test-training-roster-oracle/v1", {"scripted": 2, "frozen": 2, "mixed": 2}, 5000
    )
    expected = {mix: [] for mix in MIX_IDS}
    for row in oracle_rows:
        expected[row["mix"]].append(tuple(tuple(value) for value in row["opponents"]))
    ordinal = {mix: 0 for mix in MIX_IDS}
    for row in local:
        assert row.slots == expected[row.mix][ordinal[row.mix]]
        ordinal[row.mix] += 1


def test_roster_ids_are_stable_equal_mix_labels() -> None:
    rosters = training_rosters(3, namespace="a100-dev/test-training-roster-ids/v1")
    assert torch.equal(roster_ids(rosters, "cpu"), torch.tensor([MIX_IDS[x.mix] for x in rosters]))


def test_vector_normalizers_match_gate_and_ignore_ambient_config(monkeypatch) -> None:
    from research.sandbox_a_resident.training_rosters import vector61_params
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.vector61_featurizer import Vector61Params

    def refuse_global(*args, **kwargs):
        raise AssertionError("ambient configuration must not enter the frozen roster")

    monkeypatch.setattr(Vector61Params, "from_game_config", refuse_global)
    params = vector61_params(BatchSimConfig(num_envs=3, num_snakes=6))
    assert params.max_length == 150
    assert params.num_sectors == 16
    assert params.danger_max_distance == 30
    assert params.input_size == 61


def _stub_roster_runtime() -> tuple[object, torch.Tensor]:
    """Make a minimal runtime shell for mutable-state boundary tests."""
    from types import SimpleNamespace

    from research.sandbox_a_resident.training_rosters import DeviceRosterRuntime

    runtime = object.__new__(DeviceRosterRuntime)
    runtime.device = torch.device("cpu")
    vector_runtime = SimpleNamespace(
        carry=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        _carry_valid=torch.tensor([True, True]),
        sel_ready=torch.tensor([True, True]),
        sel_fresh=torch.tensor([True, True]),
        _alive_post=torch.tensor([[True, True], [True, True]]),
        prev_dist=torch.tensor([3.0, 5.0]),
        prev_id=torch.tensor([4, 6]),
    )
    policy = SimpleNamespace(vector_runtime=vector_runtime, vector_prepared=True)
    table = torch.arange(30_000, dtype=torch.long).reshape(2, 5000, 3)
    runtime._groups = [(torch.tensor([0, 1]), policy, {2: table})]
    return runtime, table


def test_runtime_snapshot_restores_random_safe_table_and_vector_carry() -> None:
    """Resume snapshots include the mutable random-safe tape, not just vector carry."""
    runtime, table = _stub_roster_runtime()
    state = runtime.state_dict()
    table.zero_()
    policy = runtime._groups[0][1]
    policy.vector_runtime.carry.zero_()
    policy.vector_runtime._carry_valid.fill_(False)
    policy.vector_prepared = False

    runtime.load_state_dict(state)

    assert torch.equal(table, state["groups"][0]["random_slots"][2])
    assert torch.equal(policy.vector_runtime.carry, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert bool(policy.vector_runtime._carry_valid[0])
    assert policy.vector_prepared


def test_runtime_resume_normalizes_relocated_row_identity_to_cpu() -> None:
    """Model a CUDA map_location tensor without requiring a CUDA runner."""
    runtime, _ = _stub_roster_runtime()
    state = runtime.state_dict()

    class RelocatedRows:
        def to(self, *, device, dtype):
            assert device == "cpu"
            assert dtype == torch.long
            return torch.tensor([0, 1], dtype=dtype)

    state["groups"][0]["rows"] = RelocatedRows()
    runtime.load_state_dict(state)


def test_runtime_reset_rekeys_only_completed_world_random_safe_tape() -> None:
    """A reset rekeys its table from the canonical world root without touching peers."""
    from research.gpu_sim_spike.anchors_device import random_table

    runtime, table = _stub_roster_runtime()
    original = table.clone()
    runtime.reset(torch.tensor([False, True]), seeds=[0x1234, 0x5678])

    assert torch.equal(table[0], original[0])
    assert torch.equal(table[1], random_table(0x5678, 2, torch.device("cpu")))
    assert not torch.equal(table[1], original[1])


def test_runtime_reset_defers_only_reset_row_vector_refresh() -> None:
    """A reset cannot mutate a carried peer or run prepare before task.observe()."""
    runtime, _ = _stub_roster_runtime()
    policy = runtime._groups[0][1]
    before = {
        name: getattr(policy.vector_runtime, name).clone()
        for name in ("carry", "_carry_valid", "sel_ready", "sel_fresh", "prev_dist", "prev_id")
    }
    runtime.reset(torch.tensor([False, True]), seeds={1: 0x5678})

    for name, value in before.items():
        assert torch.equal(getattr(policy.vector_runtime, name)[0], value[0])
    assert policy._training_pending_prepare.tolist() == [False, True]
    assert policy.vector_prepared
    restored = runtime.state_dict()
    assert restored["groups"][0]["pending_prepare"].tolist() == [False, True]


@torch.inference_mode()
def test_partial_reset_refreshes_real_vector_rows_after_task_observe(monkeypatch) -> None:
    """A reset row sees prepare-time food while its carried peer stays untouched."""
    from types import SimpleNamespace

    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from research.sandbox_a_resident.training_rosters import DeviceRosterRuntime, vector61_params
    from research.sandbox_a_resident.training_task import ResidentTrainingTask, TrainingTaskConfig

    task = ResidentTrainingTask(TrainingTaskConfig(worlds=3, tiny=True), device="cpu")
    vector = DeviceVectorRuntime(vector61_params(task.sim.cfg))
    controlled = torch.tensor([[False] + [True] * 5] * task.sim.E)
    vector.bind(task.sim, controlled)
    task.observe()
    vector.prepare(task.sim)
    peer_selection = vector.sel_state[1].clone()

    runtime = object.__new__(DeviceRosterRuntime)
    runtime.device = torch.device("cpu")
    policy = SimpleNamespace(vector_runtime=vector, vector_prepared=True)

    def policy_actions(sim, _actions):
        if not policy.vector_prepared:
            vector.prepare(sim)
            policy.vector_prepared = True
        return torch.ones((sim.E, sim.S), dtype=torch.long)

    policy.actions = policy_actions
    runtime._groups = [(torch.arange(task.sim.E), policy, {})]

    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original_prepare = vector.prepare

    def record_prepare(sim, rows_mask=None):
        calls.append(
            (
                torch.ones(sim.E, dtype=torch.bool) if rows_mask is None else rows_mask.clone(),
                sim.food_count.clone(),
                sim.alive.clone(),
            )
        )
        original_prepare(sim, rows_mask)

    monkeypatch.setattr(vector, "prepare", record_prepare)
    task.sim.alive[1, 1] = False
    task.sim.respawn_timer[1, 1] = 0
    reset = torch.tensor([True, False, False])
    task.reset(reset)
    runtime.reset(reset, seeds={0: 17})
    assert calls == []
    task.observe()
    assert task.sim.food_count[0].item() > 0
    assert task.sim.alive[1, 1]
    runtime.actions(task.sim, torch.ones(task.sim.E, dtype=torch.long))

    assert len(calls) == 1
    mask, food_count, alive = calls[0]
    assert mask.tolist() == [True, False, False]
    assert food_count[0].item() > 0
    assert alive[1, 1]
    assert torch.equal(vector.sel_state[1], peer_selection)

    # The normal after-step path remains one complete-group prepare; a pending
    # reset does not add a second prepare to that same decision seam.
    calls.clear()
    policy.vector_prepared = False
    policy._training_pending_prepare[:] = True
    runtime.actions(task.sim, torch.ones(task.sim.E, dtype=torch.long))
    assert len(calls) == 1
    assert calls[0][0].all()
    assert not policy._training_pending_prepare.any()
