"""Actual RNG/optimizer/adapter restoration, with fake CPU-world storage."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.bl_ppo.exports import export_torch
from research.bl_ppo.recovery import capture_full_snapshot, restore_full_snapshot


class Policy(torch.nn.Linear):
    def checkpoint_dict(self):
        return {"state_dict": self.state_dict()}


class Learner:
    def __init__(self, policy):
        self.optimizer = torch.optim.Adam(policy.parameters())
        self.update_count = 4

    def learner_state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "update_count": self.update_count}

    def load_learner_state_dict(self, value):
        self.optimizer.load_state_dict(value["optimizer"])
        self.update_count = value["update_count"]


class Collector:
    def __init__(self):
        self.state = [{"world_seed": 123, "frame": 50}]

    def snapshot_state(self):
        return self.state

    def restore_state(self, value):
        self.state = value


def test_full_boundary_restores_next_rng_draw_and_state(tmp_path: Path) -> None:
    torch.manual_seed(8)
    policy = Policy(2, 1)
    learner, collector = Learner(policy), Collector()
    adapter = SimpleNamespace(
        policy=policy,
        device=torch.device("cpu"),
        records=[],
        hidden=torch.ones(1, 1, 256),
        prefix={"value": torch.ones(32)},
    )
    identity = {"arm": "R", "source_commit": "a" * 40, "target": 8000000}
    path = tmp_path / "snapshot.pt"
    export_torch(
        path,
        capture_full_snapshot(
            policy, learner, collector, adapter, {"hero_transitions": 100}, identity
        ),
    )
    expected = (torch.rand(3), np.random.rand(3), random.random())
    collector.state, learner.update_count = [], 99
    adapter.hidden.zero_()
    counters = restore_full_snapshot(path, policy, learner, collector, adapter, identity)
    assert counters == {"hero_transitions": 100}
    assert learner.update_count == 4 and collector.state[0]["frame"] == 50
    assert adapter.hidden.sum() == 256 and len(adapter.prefix["value"]) == 32
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]
    with pytest.raises(ValueError, match="identity"):
        restore_full_snapshot(path, policy, learner, collector, adapter, {"arm": "F"})


def test_partial_rollout_cannot_claim_exact_snapshot() -> None:
    adapter = SimpleNamespace(records=[{"action": 1}])
    with pytest.raises(ValueError, match="boundary"):
        capture_full_snapshot(None, None, None, adapter, {}, {})


def test_policy_temperature_is_part_of_resume_identity(tmp_path: Path) -> None:
    from research.bl_ppo.model import PPOPolicy
    from src.model.ego2s_network import Ego2sNet

    policy = PPOPolicy("F", None, 0.7, b8_model=Ego2sNet(7, 60))
    learner, collector = Learner(policy), Collector()
    adapter = SimpleNamespace(
        policy=policy, device=torch.device("cpu"), records=[], hidden=None, prefix=None
    )
    identity = {"arm": "F", "temperature": 0.7}
    path = tmp_path / "policy-snapshot.pt"
    export_torch(path, capture_full_snapshot(policy, learner, collector, adapter, {}, identity))
    policy.temperature = 0.8
    with pytest.raises(ValueError, match="temperature"):
        restore_full_snapshot(path, policy, learner, collector, adapter, identity)
    policy.temperature = 0.7
    assert restore_full_snapshot(path, policy, learner, collector, adapter, identity) == {}


def test_frozen_banks_export_once_and_resume_requires_verified_siblings(tmp_path: Path) -> None:
    policy = Policy(2, 1)
    learner, collector = Learner(policy), Collector()
    adapter = SimpleNamespace(
        policy=policy,
        device=torch.device("cpu"),
        records=[],
        critic_refit_receipt={"qualified": True, "observed_fit_transitions": 2_000_000},
        refit_bank={"exact_suffix": torch.tensor([1.0, 2.0])},
        ordinary_bank=SimpleNamespace(frozen=True, sha256="a" * 64, rows=[1, 2, 3]),
    )
    first = capture_full_snapshot(
        policy,
        learner,
        collector,
        adapter,
        {"actor_enabled": True},
        {"arm": "F"},
        immutable_dir=tmp_path,
    )
    second = capture_full_snapshot(
        policy,
        learner,
        collector,
        adapter,
        {"actor_enabled": True},
        {"arm": "F"},
        immutable_dir=tmp_path,
    )
    assert len(list(tmp_path.glob("immutable-*.pt"))) == 2
    assert first["adapter"]["ordinary_bank"] == second["adapter"]["ordinary_bank"]
    path = tmp_path / "snapshot.pt"
    export_torch(path, first)
    restore_full_snapshot(path, policy, learner, collector, adapter, {"arm": "F"})
    assert adapter.ordinary_bank.rows == [1, 2, 3]
    assert torch.equal(adapter.refit_bank["exact_suffix"], torch.tensor([1.0, 2.0]))
    immutable = next(tmp_path.glob("immutable-ordinary_bank-*.pt"))
    immutable.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        restore_full_snapshot(path, policy, learner, collector, adapter, {"arm": "F"})


def test_snapshot_bounds_externalize_fixture_bank_and_reject_oversize_existing_sidecar(
    tmp_path: Path,
) -> None:
    from research.bl_ppo.byte_bounds import derive_snapshot_byte_bounds

    policy = Policy(2, 1)
    learner, collector = Learner(policy), Collector()
    adapter = SimpleNamespace(
        policy=policy,
        device=torch.device("cpu"),
        records=[],
        refit_bank={"rows": torch.ones(2)},
        ordinary_bank={"rows": torch.ones(2)},
        fixture_late_body_bank={"rows": torch.ones(2)},
    )
    bounds = derive_snapshot_byte_bounds(
        worlds=1,
        processes=1,
        rollout_steps=1,
        max_refit_transitions=1,
        max_actor_transitions=1,
    )
    snapshot = capture_full_snapshot(
        policy,
        learner,
        collector,
        adapter,
        {},
        {"arm": "F"},
        immutable_dir=tmp_path,
        byte_bounds=bounds,
    )
    assert "__bl_ppo_immutable_state_ref__" in snapshot["adapter"]["fixture_late_body_bank"]
    fixture = next(tmp_path.glob("immutable-partial-fixture_late_body_bank-*.pt"))
    fixture.write_bytes(
        b"x" * (bounds["components"]["fixture_late_body_bank"]["effective_bytes"] + 1)
    )
    with pytest.raises(ValueError, match="predeclared export byte bound"):
        capture_full_snapshot(
            policy,
            learner,
            collector,
            adapter,
            {},
            {"arm": "F"},
            immutable_dir=tmp_path,
            byte_bounds=bounds,
        )
