"""Cross-module RETURN_C contracts for the planner addendum.

These tests stay small and synthetic, but cross the learner, bank, trainer, and
early-warning seams that the individual unit tests cannot cover together.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008.learner_m3c import return_regression_loss
from research.redesign_m3c_20261008.train_m3c import (
    arm_halt_reason,
    boost_bucket_counts,
    boost_bucket_rates,
    boost_halt_reason,
    run_return_early_warning,
    write_arm_halt_receipt,
)
from research.redesign_m3c_20261008.obs_pack import PackLayout
from research.t4d_return_regression_20261016.bank import ReturnBank


class _StateQ(torch.nn.Module):
    """Lookup network whose scalar zero identifies a synthetic state."""

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(values.clone())

    def forward(self, local, global_, scalars):
        return self.values[scalars[:, 0].long()]


def _survivor_rows(target_value: float) -> dict[str, torch.Tensor | list[int]]:
    """Two legal branches where only the survivor's bootstrap changes its target."""
    return {
        "local": torch.zeros((2, 7, 31, 31), dtype=torch.uint8),
        "global": torch.zeros((2, 4, 37, 37), dtype=torch.uint8),
        "scalars": torch.zeros((2, 1)),
        "action": torch.tensor([0, 1]),
        "T": torch.ones(2, dtype=torch.long),
        "died": torch.tensor([True, False]),
        "realised": torch.tensor([0.0, -1.0]),
        "terminal_local": torch.zeros((2, 7, 31, 31), dtype=torch.uint8),
        "terminal_global": torch.zeros((2, 4, 37, 37), dtype=torch.uint8),
        # The dead row's terminal state is lookup row 1; the survivor is row 2.
        "terminal_scalars": torch.tensor([[1.0], [2.0]]),
        "terminal_mask": torch.ones((2, 6), dtype=torch.bool),
        "weight": torch.ones(2),
        "group_sizes": [2],
        "selected": torch.ones(2, dtype=torch.bool),
        "target_value": target_value,
    }


def test_survivor_bootstrap_can_change_dynamic_target_reference() -> None:
    """Contrast anchoring follows argmax(y), including a survivor bootstrap."""
    online_values = torch.zeros((3, 6))
    # The target network's terminal row 2 has action 0 equal to target_value.
    target_values = torch.zeros((3, 6))
    online = _StateQ(online_values)

    def evaluate(target_value: float):
        target_values[2, 0] = target_value
        target = _StateQ(target_values)
        rows = _survivor_rows(target_value)
        rows.pop("target_value")
        return return_regression_loss(
            torch,
            online,
            target,
            rows,
            gamma=0.9,
            delta=100.0,
            mode="contrast",
            contrast_epsilon=0.0,
            details=True,
        )

    low_loss, low_y, low_telemetry = evaluate(0.0)
    high_loss, high_y, high_telemetry = evaluate(3.0)

    # At zero bootstrap y=[0,-1], so action 0 is a_ref.  At three, y=[0,1.7],
    # so the survivor action 1 becomes a_ref.  This is visible in the target
    # contrast sign and magnitude, and cannot be explained by the source Qs.
    torch.testing.assert_close(low_y, torch.tensor([0.0, -1.0]))
    torch.testing.assert_close(high_y, torch.tensor([0.0, 1.7]))
    assert low_telemetry["mean_abs_target_contrast"] == pytest.approx(1.0)
    assert high_telemetry["mean_abs_target_contrast"] == pytest.approx(1.7)
    assert float(high_loss.detach()) > float(low_loss.detach())
    assert low_telemetry["contrast_rows"] == high_telemetry["contrast_rows"] == 1


def _write_grouped_bank(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = PackLayout(rect_cells=1)
    n = 12  # two source states, every six legal actions
    manifest = {
        "schema_version": 2,
        "gamma": 0.998,
        "kappa": 1.0,
        "phi": "length/10,dead=0",
        "layout": layout.to_dict(),
        "packing_version": "ego2s-bv8/obs_pack-v1",
        "reward_semantics": "mass+pbrs+death_no_overlays_or_kill",
        "collection_schema": "t4f-return-bank/v2",
    }
    state_scalars = np.zeros((n, 60), np.float32)
    state_scalars[6:, 0] = 1.0
    shard = tmp_path / "rows.npz"
    np.savez(
        shard,
        manifest_json=np.array(json.dumps(manifest)),
        state_packed=np.zeros((n, layout.nbytes), np.uint8),
        state_scalars=state_scalars,
        action=np.tile(np.arange(6, dtype=np.int64), 2),
        T=np.ones(n, np.int32),
        died=np.zeros(n, np.bool_),
        S=np.ones(n, np.float64),
        L0=np.full(n, 100, np.int32),
        L_T=np.full(n, 110, np.int32),
        terminal_packed=np.zeros((n, layout.nbytes), np.uint8),
        terminal_scalars=np.zeros((n, 60), np.float32),
        terminal_mask=np.ones((n, 6), np.bool_),
        row_weight=np.ones(n, np.float32),
        family=np.zeros(n, np.uint8),
        # Mixed presence catches the real v2 exposure without making the sampling
        # contract depend on a particular B8 action for every source state.
        b8_action=np.repeat(np.array([0, 3], np.int64), 6),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({**manifest, "shards": {shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()}})
    )
    return shard


def test_grouped_sampling_keeps_128_selected_rows_and_level_rng(tmp_path: Path) -> None:
    bank = ReturnBank(_write_grouped_bank(tmp_path), 60)
    assert bank.b8_action is not None
    assert set(np.unique(bank.b8_action)) == {0, 3}

    level = bank.sample(np.random.default_rng(23), 128, kappa=1.0)
    grouped = bank.sample(np.random.default_rng(23), 128, kappa=1.0, grouped=True)

    assert len(grouped["action"]) == 128 * 6
    assert int(grouped["selected"].sum()) == 128
    assert np.all(grouped["group_sizes"] == 6)
    np.testing.assert_array_equal(grouped["action"][grouped["selected"]], level["action"])
    np.testing.assert_array_equal(grouped["scalars"][grouped["selected"]], level["scalars"])


def test_boost_bucket_boundaries_and_strict_overall_tripwires() -> None:
    lengths = np.array([399, 400, 699, 700, 999, 1000])
    actions = np.array([3, 2, 3, 2, 3, 3])
    decisions, boosts = boost_bucket_counts(lengths, actions)
    assert decisions.tolist() == [1, 2, 2, 1]
    assert boosts.tolist() == [1, 1, 1, 1]
    assert boost_bucket_rates(decisions, boosts) == {
        "lt_400": 1.0,
        "400_700": 0.5,
        "700_1000": 0.5,
        "ge_1000": 1.0,
    }

    args = SimpleNamespace(boost_halt_rate=0.02, boost_halt_low_length_rate=0.01)
    row = {
        "greedy_boost_rate": 0.02,
        "greedy_boost_rate_by_length": {
            "lt_400": 0.01,
            "400_700": 0.01,
            "700_1000": 0.0,
            "ge_1000": 0.0,
        },
        "greedy_boost_length_decisions": {
            "lt_400": 50,
            "400_700": 50,
            "700_1000": 1,
            "ge_1000": 1,
        },
    }
    assert boost_halt_reason(args, row, 100_000) is None
    row["greedy_boost_rate"] = 0.020001
    assert "overall greedy boost rate" in boost_halt_reason(args, row, 100_000)
    row["greedy_boost_rate"] = 0.0
    row["greedy_boost_rate_by_length"]["400_700"] = 0.010001
    assert "below length 700" in boost_halt_reason(args, row, 100_000)


def test_early_warning_nonpass_and_create_only_shared_receipt(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "warning.json"
    output_dir = tmp_path / "warning-output"
    config.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "holdout_bank": str(tmp_path / "holdout"),
                "outside_bank": str(tmp_path / "outside"),
                "initial_checkpoint": str(tmp_path / "initial.pth"),
            }
        )
    )
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output_path = Path(command[command.index("--out") + 1])
        output_path.parent.mkdir(parents=True)
        output_path.write_text(json.dumps({"status": "KILL", "reason": "outside drift"}))
        return SimpleNamespace(returncode=1)

    import research.redesign_m3c_20261008.train_m3c as trainer

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    result = run_return_early_warning(
        SimpleNamespace(
            return_early_warning_config=config,
            init=tmp_path / "initial.pth",
            out=tmp_path / "run",
        ),
        tmp_path / "ckpt_500k.pth",
        500_000,
        tmp_path / "log.jsonl",
    )
    assert result == {"status": "KILL", "reason": "outside drift"}
    assert len(commands) == 1
    assert commands[0][1:4] == ["-m", "research.t4f_return_20261016.early_warning", "--checkpoint"]
    assert "500000" in commands[0]

    receipt = tmp_path / "shared-halt.json"
    args = SimpleNamespace(arm_halt_file=receipt)
    write_arm_halt_receipt(args, "first reason", 500_000)
    write_arm_halt_receipt(args, "second reason", 600_000)
    assert json.loads(receipt.read_text()) == {"reason": "first reason", "transitions": 500_000}
    assert arm_halt_reason(args) == f"shared arm halt receipt exists: {receipt}"


def test_actual_learner_fast_and_reference_contrast_paths_match(tmp_path: Path) -> None:
    """The real Learner consumes grouped rows identically in both update paths."""
    from research.redesign_m3c_20261008 import learner_m3c as lm
    from tests import test_perf_learner as perf

    common = {
        "return_frac": 0.5,
        "return_weight": 1.0,
        "return_huber_delta": 100.0,
        "return_mode": "contrast",
        "return_contrast_epsilon": 2.0,
        "td_huber_delta": 1.0,
        "mass_kappa": 1.0,
    }
    reference_bank = ReturnBank(_write_grouped_bank(tmp_path / "reference"), 60)
    fast_bank = ReturnBank(_write_grouped_bank(tmp_path / "fast"), 60)
    reference = perf._run(
        "bv8",
        lm.LearnerOpts(fused_adam=False),
        n_updates=2,
        return_bank=reference_bank,
        **common,
    )
    fast = perf._run(
        "bv8",
        lm.LearnerOpts(fast=True, fused_adam=False),
        n_updates=2,
        return_bank=fast_bank,
        **common,
    )
    perf._assert_same(reference, fast, exact=True)
    for result in (reference, fast):
        state = result[2]
        assert state["return_rows"] == [4, 4]
        assert len(state["return_loss"]) == 2
        assert len(state["return_contrast_rows"]) == 2
        assert state["return_totals"]["rows"] == 8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_contrast_loss_one_update_mps_smoke() -> None:
    """Exercise the grouped contrast tensor/index operations on the MPS backend."""
    rows = _survivor_rows(3.0)
    rows.pop("target_value")
    device = torch.device("mps")
    for key, value in list(rows.items()):
        if isinstance(value, torch.Tensor):
            rows[key] = value.to(device)
    online = _StateQ(torch.zeros((3, 6), device=device))
    target = _StateQ(torch.zeros((3, 6), device=device))
    with torch.no_grad():
        target.values[2, 0] = 3.0
    loss, y, telemetry = return_regression_loss(
        torch,
        online,
        target,
        rows,
        gamma=0.9,
        delta=100.0,
        mode="contrast",
        contrast_epsilon=2.0,
        details=True,
    )
    assert torch.isfinite(loss).item()
    assert torch.isfinite(y).all().item()
    assert telemetry["contrast_rows"] == 0
