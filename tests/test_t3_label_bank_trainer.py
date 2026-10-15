"""T3 bank schema, exact loss, learner parity and bounded CPU CLI integration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008 import learner_m3c as lm
from research.redesign_m3c_20261008 import train_m3c as m3c
from research.t3_rollout_labels_20261015.bank import LabelBank, surviving_margin
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from tests.test_perf_learner import _args, _assert_same, _ingest, _st, _world


def _write_bank(path: Path, **extra) -> Path:
    n = 2
    payload = {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "mask": np.ones((n, 6), bool),
        "surviving": np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]], bool),
        "action": np.array([0, 1], np.int64),
        "weight": np.array([1.0, 2.0], np.float32),
    }
    payload.update(extra)
    np.savez(path / "shard.npz", **payload)
    return path


def test_bank_loads_and_samples_schema(tmp_path: Path) -> None:
    bank = LabelBank(_write_bank(tmp_path), 60)
    row = bank.sample(np.random.default_rng(7), 3)
    assert bank.size == 2
    assert row["local"].shape == (3, 7, 31, 31)
    assert row["surviving"].shape == (3, 6)


def test_bank_rejects_empty_surviving_set(tmp_path: Path) -> None:
    surviving = np.zeros((2, 6), bool)
    with pytest.raises(ValueError, match="surviving"):
        LabelBank(_write_bank(tmp_path, surviving=surviving), 60)


def test_surviving_margin_matches_hand_values() -> None:
    q = torch.tensor([[1.0, 3.0, 2.0, 0.0, 1.0, 1.0]], requires_grad=True)
    surviving = torch.tensor([[True, False, True, False, False, False]])
    weight = torch.tensor([2.0])
    # max(q + 2 for non-survivors)=5, max surviving=2, then row weight=2.
    loss = surviving_margin(torch, q, surviving, weight, 2.0)
    assert float(loss.detach()) == pytest.approx(6.0)
    loss.backward()
    torch.testing.assert_close(q.grad, torch.tensor([[0.0, 2.0, -2.0, 0.0, 0.0, 0.0]]))


def test_bank_margin_scale_and_fatal_b8_action_are_not_an_extra_target() -> None:
    q = torch.tensor([[1.0, 3.0, 2.0, 0.0, 1.0, 1.0]])
    surviving = torch.tensor([[True, False, True, False, False, False]])
    db = {
        "surviving": surviving,
        "weight": torch.tensor([2.0]),
        "a_v8": torch.tensor([1]),
        "mask_resolved": torch.ones((1, 6), dtype=torch.bool),
    }
    anchor, bank_loss, agree = lm._bank_anchor_terms(torch, q, db, 1, 1.0, 2.0)
    assert float(anchor) == float(bank_loss) == 6.0
    assert float(agree) == 0.0


def test_bank_packed_roundtrip(tmp_path: Path) -> None:
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    _write_bank(tmp_path)
    with np.load(tmp_path / "shard.npz") as archive:
        row = {k: archive[k] for k in archive.files}
    layout = train_layout()
    row["packed"] = pack_ref(row.pop("local"), row.pop("global"), layout)
    np.savez(tmp_path / "shard.npz", **row)
    bank = LabelBank(tmp_path, 60, pack=layout)
    assert bank.size == 2
    assert not bank.local.any() and not bank.global_.any()
    row["packed"][0, layout.off_g] = 2
    np.savez(tmp_path / "shard.npz", **row)
    with pytest.raises(ValueError):
        LabelBank(tmp_path, 60, pack=layout)


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_bank_reference_fast_updates_are_bitwise_equal(tmp_path: Path, fraction: float) -> None:
    """Eight updates consume 320 new synthetic transitions and both label sources."""
    torch.set_num_threads(1)
    bank = LabelBank(_write_bank(tmp_path), 60)

    def run(fast: bool):
        torch.manual_seed(3)
        nscal, multi, replay, labels, demos = _world("bv8")
        if fraction == 1.0:
            labels = None
        online, target = Ego2sNet(7, nscal), Ego2sNet(7, nscal)
        target.load_state_dict(online.state_dict())
        opts = lm.LearnerOpts(fast=fast, fused_adam=False)
        stats, rng = _st(), np.random.default_rng([7, 17])
        learner = lm.Learner(
            torch,
            online=online,
            target=target,
            opt=lm.make_optimizer(torch, online, opts, "cpu"),
            replay=replay,
            multi=multi,
            demos=demos,
            labels=labels,
            rng=rng,
            st=stats,
            args=_args(
                "bv8",
                label_bank_frac=fraction,
                label_frac=1.0 - fraction,
                label_bank_margin_scale=1.7,
                onpolicy_labels=fraction != 1.0,
            ),
            nscal=nscal,
            margin=0.3,
            device="cpu",
            opts=opts,
            agent_rows=24,
            anchor_rows=8,
            min_rows=4,
            bank=bank,
        )
        learner.target_every = 3
        feed = np.random.default_rng(99)
        for _ in range(8):
            learner.update(0.5)
            _ingest(replay, feed, 40)
        learner.flush()
        assert stats["updates"] == 8 and stats["halt"] is None
        assert stats["bank_rows"] == [int(8 * fraction)] * 8
        assert np.isfinite(stats["bank_margin_loss"]).all()
        assert all(0 <= value <= 1 for value in stats["bank_agree"])
        return online, target, stats, rng, multi

    reference, fast = run(False), run(True)
    _assert_same(reference, fast)
    for key in ("bank_rows", "bank_margin_loss", "bank_agree"):
        assert reference[2][key] == fast[2][key]


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--label-bank-margin-scale", "-1"],
        ["--label-frac", "nan"],
        ["--label-frac", "-1"],
        ["--obs-version", "b"],
    ],
)
def test_bad_bank_flags_fail_before_output(tmp_path: Path, bad_args: list[str]) -> None:
    args = [
        "--seed",
        "1",
        "--init",
        "missing.pth",
        "--demo-data",
        "missing",
        "--out",
        str(tmp_path / "out"),
        "--actors",
        "1",
        "--obs-version",
        "bv8",
        "--anchor-mode",
        "v8margin",
        "--label-bank",
        str(tmp_path),
        "--label-bank-frac",
        "1",
        "--label-frac",
        "0",
    ]
    with pytest.raises(SystemExit):
        m3c.main(args + bad_args)
    assert not (tmp_path / "out").exists()


def test_bank_only_cpu_cli_smoke(tmp_path: Path) -> None:
    """A 512-transition packed run exercises actor packets, bank-only loss and cleanup."""
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    init = tmp_path / "init.pth"
    torch.manual_seed(9)
    save_ego2s_checkpoint(init, Ego2sNet(7, 60))
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _write_bank(bank_dir)
    demos = tmp_path / "demos"
    held = demos / "round-12"
    held.mkdir(parents=True)
    (demos / "margin.json").write_text(json.dumps({"margin": 0.3}))
    np.savez(
        held / "chunk-0-part-0.npz",
        local=np.zeros((1, 7, 31, 31), np.uint8),
        **{"global": np.zeros((1, 4, 37, 37), np.uint8)},
        scalars=np.zeros((1, 30), np.float32),
        q_teacher=np.ones((1, 6), np.float32),
        a_v8=np.zeros(1, np.int64),
        mask_resolved=np.ones((1, 6), bool),
        world_seed=np.zeros(1, np.int64),
    )
    out = tmp_path / "smoke"
    command = [
        sys.executable,
        "research/redesign_m3c_20261008/train_m3c.py",
        "--seed",
        "9",
        "--init",
        str(init),
        "--demo-data",
        str(demos),
        "--out",
        str(out),
        "--actors",
        "1",
        "--worlds",
        "2",
        "--device",
        "cpu",
        "--transitions",
        "512",
        "--warmup",
        "64",
        "--replay",
        "512",
        "--segment-frames",
        "500",
        "--frozen-frac",
        "0",
        "--mixed-frac",
        "0",
        "--obs-version",
        "bv8",
        "--anchor-mode",
        "v8margin",
        "--label-bank",
        str(bank_dir),
        "--label-bank-frac",
        "1",
        "--label-frac",
        "0",
        "--learner-fast",
        "--obs-pack",
        "--v61-backend",
        "numba",
        "--no-guard",
        "--ring-dir",
        str(tmp_path),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-5000:] + result.stdout[-2000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["halt"] is None and summary["actor_exitcodes"] == [0]
    assert summary["transitions"] >= 512 and summary["updates"] > 0
    assert summary["label_buffer"] == 0
    assert summary["m3c"]["label_bank_frac"] == 1.0
    assert (out / "final.pth").exists()
