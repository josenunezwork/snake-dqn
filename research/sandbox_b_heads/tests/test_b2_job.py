import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.sandbox_b_heads.job import parent_split
from research.sandbox_b_heads.job_b2 import pareto_frontier, run, threshold_actions
from research.sandbox_b_heads.resume_b2 import (
    atomic_json,
    build_ledger,
    empty_ledger,
    recipe,
    verify_ledger,
)


def test_tiny_cpu_entrypoint_exports_all_config_hashes(tmp_path, monkeypatch, capsys):
    """Exercise the real finite sweep with synthetic, non-registry fixture data."""
    from src.model.ego2s_network import Ego2sNet

    candidate_parents = np.array([f"synthetic-{i}" for i in range(50)], dtype="U16")
    split = parent_split(candidate_parents)
    parents = np.array(
        [candidate_parents[np.where(~split)[0][0]], candidate_parents[np.where(split)[0][0]]],
        dtype="U16",
    )
    n = 2
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
        ),
        "status": np.full((n, 6), "EXACT", dtype="U13"),
        "mass": np.ones((n, 6), np.float32),
        "ordinary": np.array([False, True]),
        "natural_stream": np.array([False, True]),
        "sampling_weight": np.ones(n, np.float32),
        "parent_world": parents,
        "life_id": np.array(["life-a", "life-b"]),
        "row_id": np.array(["row-a", "row-b"]),
    }
    data, b8, output = tmp_path / "synthetic.npz", tmp_path / "b8.pt", tmp_path / "out"
    np.savez(data, **table)
    torch.save(Ego2sNet(7, 60).state_dict(), b8)
    args = argparse.Namespace(
        input=str(data),
        b8_checkpoint=str(b8),
        output=str(output),
        namespace="synthetic",
        salt="a100-dev/sandbox-b/split-v1",
        device="cpu",
        updates=1,
        batch_size=2,
        learning_rate=3e-3,
        seed=3,
        synthetic_fixture=True,
    )
    from research.colab_cli.runtime import _exports
    from research.sandbox_b_heads import job_b2 as job

    root = tmp_path / "runtime"
    root.mkdir()
    (root / "output").symlink_to(output, target_is_directory=True)
    cache = {}
    publish = job._publish_bundle

    def publish_then_poll(*a, **kw):
        publish(*a, **kw)
        _exports(root, 1_000_000_000, cache)

    monkeypatch.setattr(job, "_publish_bundle", publish_then_poll)
    result = run(args)
    payload = json.loads(result.read_text())
    telemetry = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert telemetry[0]["progress"] == "started"
    assert telemetry[0]["completed_configs"] == 0
    assert telemetry[0]["total_configs"] == 48
    assert telemetry[0]["elapsed_seconds"] >= 0
    assert telemetry[0]["eta_seconds"] is None
    exported = [line for line in telemetry if line["progress"] == "unit-exported"]
    assert len(exported) == 48
    assert exported[-1]["completed_configs"] == 48
    assert exported[-1]["eta_seconds"] == pytest.approx(0.0)
    assert len(payload["configs"]) == 48
    assert len(payload["rows"]) == 240
    assert {r["tau"] for r in payload["rows"]} == {0.0, 0.1, 0.2, 0.4, 0.8}
    assert json.loads((output / "pareto.json").read_text())["rows"] == payload["pareto"]
    assert len(list((output / "checkpoints").glob("*.pt.sha256"))) == 48
    assert (output / "checkpoints.tar").is_file()

    # Actual qualified immutable-export scanner: poll after all seals exist, then again
    # during fresh-directory restore. Already scanned payload signatures must stay fixed.
    inventory = _exports(root, 1_000_000_000, cache)
    assert len([item for item in inventory if item["path"].startswith("units/")]) == 48
    assert _exports(root, 1_000_000_000, cache) == inventory
    frozen = recipe(data, b8, updates=1, batch_size=2, seed=3)
    ledger = tmp_path / "mac-ledger.json"
    built = build_ledger(output, ledger, frozen, namespace="synthetic")
    assert len(built["units"]) == 48
    # Emulate uploading to a different VM: only basenames in ledger, inputs next to it.
    import shutil

    remote = tmp_path / "remote-inputs"
    remote.mkdir()
    portable = json.loads(json.dumps(built))
    for unit in portable["units"]:
        source = Path(unit["bundle"]["path"])
        shutil.copyfile(source, remote / source.name)
        unit["bundle"]["path"] = source.name
    remote_ledger = remote / "verified-ledger.json"
    remote_ledger.write_text(json.dumps(portable))
    monkeypatch.setattr(
        "research.sandbox_b_heads.job_b2._cache_features",
        lambda *a: pytest.fail("fully skipped resume extracted features"),
    )
    args.output = str(tmp_path / "fresh-resume")
    args.ledger = str(remote_ledger)
    resumed = json.loads(run(args).read_text())
    assert resumed["rows"] == payload["rows"]
    assert len(list((tmp_path / "fresh-resume" / "checkpoints").glob("*.pt"))) == 48
    assert _exports(root, 1_000_000_000, cache) == inventory
    # No outcomes are opened by the Mac ledger builder (tarfile open is forbidden).
    monkeypatch.setattr("tarfile.open", lambda *a, **k: pytest.fail("unsealed bundle"))
    build_ledger(output, tmp_path / "blind-ledger.json", frozen, namespace="synthetic")
    for key, value, match in (
        ("namespace", "foreign", "foreign"),
        ("allocation", "foreign", "foreign"),
        ("fingerprint", "0" * 64, "foreign"),
    ):
        changed = json.loads(ledger.read_text())
        changed[key] = value
        ledger.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=match):
            verify_ledger(ledger, frozen, "synthetic")
        ledger.write_text(json.dumps(built))
    changed = {**built, "units": built["units"] + built["units"][:1]}
    ledger.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="duplicate"):
        verify_ledger(ledger, frozen, "synthetic")
    changed = json.loads(json.dumps(built))
    changed["units"][0]["allocation_index"] = 1
    ledger.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="foreign"):
        verify_ledger(ledger, frozen, "synthetic")
    ledger.write_text(json.dumps(built))
    bundle = Path(built["units"][0]["bundle"]["path"])
    bundle.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA"):
        verify_ledger(ledger, frozen, "synthetic")


def test_recipe_binds_all_training_inputs_and_empty_ledger(tmp_path):
    data, b8 = tmp_path / "data", tmp_path / "b8"
    data.write_bytes(b"a")
    b8.write_bytes(b"b")
    baseline = recipe(data, b8)
    for key, value in (
        ("salt", "new"),
        ("seed", 2),
        ("updates", 2),
        ("learning_rate", 0.2),
        ("batch_size", 2),
        ("smoke_rows", 2),
    ):
        assert recipe(data, b8, **{key: value}) != baseline
    ledger = tmp_path / "empty.json"
    atomic_json(ledger, empty_ledger(baseline))
    with pytest.raises(ValueError, match="already exists"):
        atomic_json(ledger, empty_ledger(baseline))
    assert verify_ledger(ledger, baseline)["units"] == []
    data.write_bytes(b"c")
    with pytest.raises(ValueError, match="foreign"):
        verify_ledger(ledger, recipe(data, b8))


def test_tau_keeps_base_on_equal_or_insufficient_advantage():
    q = torch.tensor([[0.0, 0.1, 100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    table = {
        "bare": torch.tensor([0, 1]),
        "action_mask": torch.tensor([[True, True, False, False, False, False], [True] * 6]),
    }
    assert threshold_actions(q, table, 0.0).tolist() == [1, 1]
    assert threshold_actions(q, table, 0.1).tolist() == [0, 1]
    assert threshold_actions(q, table, 0.2).tolist() == [0, 1]


def test_frontier_preserves_micro_macro_tradeoffs_and_equal_points():
    def point(micro, macro, harm):
        return dict(
            rescue_retention=micro, life_macro_rescue_retention=macro, ordinary_harm_count=harm
        )

    a, b, c, d = point(0.8, 0.5, 1), point(0.7, 0.6, 1), point(0.6, 0.4, 2), point(0.8, 0.5, 1)
    assert pareto_frontier([a, b, c, d]) == [a, b, d]
    assert pareto_frontier([point(None, None, 0)]) == []
