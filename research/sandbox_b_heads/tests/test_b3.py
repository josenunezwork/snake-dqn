"""B-3 policy, label, frontier and real export/resume checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.sandbox_b_heads.core_b3 import (
    CONFIGS,
    pareto_frontier,
    promising,
    survival_loss,
    veto_actions,
)


def test_hard_veto_is_independent_of_q_margin_and_preserves_base_at_boundary():
    table = {
        "bare": torch.tensor([0, 0, 0, 0]),
        "action_mask": torch.tensor([[True, True, False, False, False, False]] * 4),
        "q": torch.tensor([[100.0, -100.0, 0, 0, 0, 0]] * 4),
    }
    p = torch.tensor(
        [
            [0.1, 0.8, 1, 1, 1, 1],
            [0.5, 0.9, 1, 1, 1, 1],
            [0.9, 1, 1, 1, 1, 1],
            [0.1, 0.05, 1, 1, 1, 1],
        ]
    )
    assert veto_actions(p, table, 0.5).tolist() == [1, 0, 0, 1]
    assert veto_actions(p, table, 0.5, 0.1).tolist() == [1, 0, 0, 0]
    assert veto_actions(p, table, 0.0).tolist() == [0, 0, 0, 0]


def test_bce_excludes_unknown_and_illegal_instead_of_inventing_deaths():
    logits = torch.zeros((2, 6), requires_grad=True)
    table = {
        "status": torch.tensor([[True, True, False, True, False, False], [False] * 6]),
        "legal": torch.tensor([[True, True, True, False, True, True], [True] * 6]),
        "died": torch.tensor([[True, False, True, True, True, True], [True] * 6]),
    }
    survival_loss(logits, table).backward()
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] < 0
    assert torch.count_nonzero(logits.grad) == 2


def test_promising_and_frontier_use_flip_rate_with_count_harm():
    row = dict(
        rescue_retention=0.70,
        life_macro_rescue_retention=0.65,
        ordinary_flip_rate=0.01,
        ordinary_harm_count=1,
        ordinary_count=100,
    )
    assert promising(row)
    assert not promising({**row, "ordinary_flip_rate": 0.010001})
    assert not promising({**row, "ordinary_harm_count": 2})
    assert not promising({**row, "ordinary_count": 0})
    more_flips = {**row, "ordinary_flip_rate": 0.2}
    assert pareto_frontier([more_flips, row]) == [row]
    more_retention = {**more_flips, "rescue_retention": 0.8}
    assert pareto_frontier([more_retention, row]) == [more_retention, row]


def test_real_tiny_entrypoint_export_poll_and_complete_resume(tmp_path, monkeypatch, capsys):
    from research.colab_cli.runtime import _exports
    from research.sandbox_b_heads import job_b3 as job
    from research.sandbox_b_heads.resume_b3 import build_ledger, recipe, verify_ledger
    from src.model.ego2s_network import Ego2sNet

    candidates = np.array([f"synthetic-{i}" for i in range(50)])
    split = job.parent_split(candidates)
    parents = np.repeat([candidates[~split][0], candidates[split][0]], 2)
    n = 4
    table = {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "q": np.zeros((n, 6), np.float32),
        "legal": np.ones((n, 6), bool),
        "action_mask": np.ones((n, 6), bool),
        "bare": np.zeros(n, np.int64),
        "died": np.array(
            [[True, False, False, False, False, False], [False, True, False, False, False, False]]
            * 2
        ),
        "status": np.full((n, 6), "EXACT", dtype="U13"),
        "mass": np.ones((n, 6), np.float32),
        "ordinary": np.array([False, True] * 2),
        "natural_stream": np.array([False, True] * 2),
        "sampling_weight": np.ones(n, np.float32),
        "parent_world": parents,
        "life_id": np.array(["life-a", "life-a", "life-b", "life-b"]),
        "row_id": np.array(["row-a", "row-b", "row-c", "row-d"]),
    }
    data, b8, output = tmp_path / "synthetic.npz", tmp_path / "b8.pt", tmp_path / "out"
    np.savez(data, **table)
    torch.save(Ego2sNet(7, 60).state_dict(), b8)
    args = argparse.Namespace(
        input=str(data),
        b8_checkpoint=str(b8),
        output=str(output),
        namespace="synthetic",
        allocation="split-v1",
        salt=job.SALT,
        device="cpu",
        updates=1,
        batch_size=2,
        learning_rate=3e-3,
        seed=3,
        synthetic_fixture=True,
    )
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "output").symlink_to(output, target_is_directory=True)
    cache = {}
    publish = job._publish_bundle

    def publish_and_poll(*a, **kw):
        publish(*a, **kw)
        _exports(root, 1_000_000_000, cache)

    monkeypatch.setattr(job, "_publish_bundle", publish_and_poll)
    payload = json.loads(job.run(args).read_text())
    assert len(payload["configs"]) == len(CONFIGS) == 12
    assert len(payload["rows"]) == 12 * 12 * 3
    assert payload["criterion"]["requires_nonempty_decisive_and_ordinary_denominators"]
    assert all("ordinary_flip_rate" in row for row in payload["rows"])
    assert all(row["diagnostics"] is not None for row in payload["configs"][:6])
    frozen = recipe(data, b8, updates=1, batch_size=2, seed=3)
    ledger = tmp_path / "verified.json"
    built = build_ledger(output, ledger, frozen, namespace="synthetic")
    assert len(built["units"]) == 12
    assert verify_ledger(ledger, frozen, namespace="synthetic") == built
    inventory = _exports(root, 1_000_000_000, cache)
    assert len([item for item in inventory if item["path"].startswith("units/")]) == 12
    assert _exports(root, 1_000_000_000, cache) == inventory
    monkeypatch.setattr(
        job, "_cache_features", lambda *a: pytest.fail("complete resume recomputed features")
    )
    args.output, args.ledger = str(tmp_path / "resumed"), str(ledger)
    resumed = json.loads(job.run(args).read_text())
    assert resumed["rows"] == payload["rows"]
    foreign = {**built, "namespace": "foreign"}
    ledger.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="foreign"):
        verify_ledger(ledger, frozen, namespace="synthetic")
    ledger.write_text(json.dumps(built))
    bundle = Path(built["units"][0]["bundle"]["path"])
    bundle.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA"):
        verify_ledger(ledger, frozen, namespace="synthetic")
    telemetry = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert telemetry[0]["progress"] == "started"
