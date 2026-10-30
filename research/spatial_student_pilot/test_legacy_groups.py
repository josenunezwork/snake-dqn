"""Legacy inference oracle independent of the candidate's group construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from research.gpu_sim_spike.device_policy import _masked_argmax
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.batching import BatchedOpponents, legacy_checkpoint_q


class ShapeSensitivePolicy(nn.Module):
    """Expose changed batch shapes and accidental single-row tie stabilization."""

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q = state[:, :6].clone()
        q[:, :2] = 10
        q[:, 1] += len(state) * 1e-7
        return q


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("real_network", [False, True])
@torch.inference_mode()
def test_checkpoint_groups_match_independent_legacy_q_and_actions(
    device: str, real_network: bool
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from research.sandbox_a_resident.ct_run import _load_vector61
    from research.spatial_student_pilot.prepare import CHECKPOINT_NAMES, OPPONENT_SHAS, ROOT

    table = training_rosters(1024)
    # Mixed rosters, repeated checkpoints in different slots, uneven last slab,
    # and shuffled world identities. Group boundaries follow resident positions.
    rosters = [table[(i * 17 + 9) % 1024] for i in range(263)]
    networks = {
        key: (
            _load_vector61(
                ROOT.parent / "snake-dqn/saved_snakes" / CHECKPOINT_NAMES[name],
                torch.device(device),
            )
            if real_network
            else ShapeSensitivePolicy().to(device)
        )
        for name, key in OPPONENT_SHAS.items()
    }
    runtime = BatchedOpponents.__new__(BatchedOpponents)
    runtime.device = torch.device(device)
    runtime.rosters = rosters
    runtime.networks = networks
    runtime.runtime = SimpleNamespace()
    runtime._dispatch()
    generator = torch.Generator().manual_seed(7821)
    states = torch.randn((len(rosters), 6, 61), generator=generator).to(device)
    masks = torch.rand((len(rosters), 6, 6), generator=generator).to(device) > 0.3
    expected = torch.zeros((len(rosters), 6, 6), device=device)
    # Reconstruct CTOpponentRuntime -> DevicePolicy nesting independently:
    # slab, roster identity in first-seen order, slot, ascending world row.
    for start in range(0, len(rosters), 128):
        by_roster: dict[Any, list[int]] = {}
        for row in range(start, min(start + 128, len(rosters))):
            by_roster.setdefault(rosters[row].identity, []).append(row)
        for rows in by_roster.values():
            for slot, (kind, ref) in enumerate(rosters[rows[0]].slots, 1):
                if kind == "checkpoint":
                    expected[rows, slot] = networks[ref](states[rows, slot]).float()
    for ref, rows in runtime.groups.items():
        er, sr = rows.unbind(1)
        actual = legacy_checkpoint_q(networks[ref], states[er, sr], runtime.reference_q_groups[ref])
        reference = expected[er, sr]
        assert actual.cpu().numpy().tobytes() == reference.cpu().numpy().tobytes()
        assert torch.equal(
            _masked_argmax(actual, masks[er, sr]), _masked_argmax(reference, masks[er, sr])
        )


def test_ruler_checkpoint_byte_guard_still_refuses_changed_q(monkeypatch: Any) -> None:
    from research.spatial_student_pilot import batching
    from research.spatial_student_pilot.config import PilotConfig
    from research.spatial_student_pilot.test_batching import simulator

    sim, opponents = simulator(PilotConfig().development("preflight"), list(range(6)))
    opponents.verify_q_bytes = True
    original = batching.legacy_checkpoint_q
    monkeypatch.setattr(batching, "legacy_checkpoint_q", lambda *args: original(*args) + 0.125)
    sim.prepare()
    with pytest.raises(ValueError, match="checkpoint Q bytes differ"):
        opponents.actions(sim, torch.ones(sim.E, dtype=torch.long))
