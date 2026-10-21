"""Focused runner guards; simulator execution is covered by the explicit smoke command."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_e16 import runner


def test_train_requires_external_validated_plan() -> None:
    with pytest.raises(runner.AdmissionRequired, match="validated external"):
        runner.require_plan(None, "A")


def test_stub_arms_are_not_executable() -> None:
    parser = runner.build_parser()
    arguments = parser.parse_args(
        [
            "smoke",
            "--arm",
            "A",
            "--bank",
            "bank",
            "--seed-audit",
            "audit",
            "--b8-checkpoint",
            "b8",
            "--parent",
            "parent",
            "--output",
            "out",
        ]
    )
    arguments.arm = "R"
    with pytest.raises(runner.AdmissionRequired, match="UNRANKED"):
        runner.run(arguments)


def test_audit_requires_complete_sidecars(tmp_path: Path) -> None:
    bank_path, audit_path = tmp_path / "BANK.json", tmp_path / "SEED_AUDIT.json"
    bank_path.write_text("{}")
    audit_path.write_text("{}")
    with pytest.raises(ValueError, match="missing or unsafe"):
        runner.load_audited_bank(bank_path, audit_path)


def test_snapshot_captures_explicit_collector_and_rng_state() -> None:
    # The schema contract is intentionally inspectable without launching a simulator.
    assert runner.SCHEMA == "bl-e16-runner/v1"
    assert runner.CRITIC_TRANSITIONS == 2_000_000
    assert runner.ACTOR_TRANSITIONS == 4_000_000


def test_critic_terminal_boundary_is_complete_before_cap_snapshot(tmp_path, monkeypatch) -> None:
    from collections import deque
    from types import SimpleNamespace

    import torch

    from research.bl_e16.spool import StreamingLife

    class Driver:
        sim = None
        last_history_state = None

        def observe_hero(self):
            return {}

        def step(self, action):
            return {"mass": 1000.0, "done": True, "frame": 1}

    class Collector:
        pending_spool = None
        pending_spool_state = None
        history = deque()

        def ensure(self):
            return Driver()

        def terminal_reset(self):
            self.reset = True

    collector = Collector()
    record = {
        "local": torch.zeros(1, 7, 3, 3),
        "global_": torch.zeros(1, 4, 3, 3),
        "scalars": torch.zeros(1, 60),
        "mask": torch.ones(1, 6, dtype=torch.bool),
        "action": torch.zeros(1, dtype=torch.long),
        "logp": torch.zeros(1),
        "value": torch.zeros(1),
        "geometry": None,
        "history": None,
    }
    monkeypatch.setattr(runner, "_act", lambda *args: (0, record))

    def cap():
        assert collector.reset and collector.pending_spool.closed
        raise runner.AdmissionRequired("cap")

    with pytest.raises(runner.AdmissionRequired, match="cap"):
        runner._collect_critic_life(SimpleNamespace(), collector, tmp_path / "spool", "cpu", cap)
    restored = StreamingLife.restore(tmp_path / "restored", collector.pending_spool.snapshot())
    assert next(restored.batches()).critic_targets.item() == 1.0


def test_restored_numpy_grid_interiors_share_padded_storage() -> None:
    import pickle
    from types import SimpleNamespace

    import numpy as np

    sim = SimpleNamespace(_grid_pad=1, grid_h=2, grid_w=3)
    for name in ("owner", "slot", "food"):
        padded = np.arange(20).reshape(1, 4, 5)
        setattr(sim, f"_{name}_pad", padded)
        setattr(sim, f"_{name}", padded[:, 1:3, 1:4])
    state = pickle.loads(
        pickle.dumps(
            {
                "seed": 1,
                "horizon": 5000,
                "frame": 8,
                "sim": sim,
                "frozen": None,
                "kernel_module": None,
            }
        )
    )
    assert not np.shares_memory(state["sim"]._owner, state["sim"]._owner_pad)
    driver = runner._E16TerminalDriver.restore(state)
    for name in ("owner", "slot", "food"):
        interior, pad = getattr(driver.sim, f"_{name}"), getattr(driver.sim, f"_{name}_pad")
        assert np.shares_memory(interior, pad)
        interior[0, 0, 0] = 999
        assert pad[0, 1, 1] == 999
