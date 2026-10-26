import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.sandbox_b_heads.job import _served, metrics, parent_split, run, sha256
from research.sandbox_b_heads.model import make_head
from research.sandbox_b_heads.resume import (
    atomic_json,
    build_ledger,
    empty_ledger,
    recipe,
    verify_ledger,
)


def test_parent_split_is_stable_and_parent_scoped():
    parents = np.array(["a", "a", "b", "c"], dtype="U1")
    first, second = parent_split(parents), parent_split(parents)
    assert np.array_equal(first, second)
    assert first[0] == first[1]


def test_all_declared_heads_are_six_action_finite_residuals():
    features = torch.randn(4, 256)
    length, margin = torch.rand(4), torch.rand(4)
    for name in ("linear", "gated_mlp", "cosine"):
        result = make_head(name)(features, length, margin)
        assert result.shape == (4, 6)
        assert torch.isfinite(result).all()


def test_metrics_reports_rescue_and_ordinary_harm_proxy():
    table = {
        "q": torch.zeros(2, 6),
        "action_mask": torch.ones(2, 6, dtype=torch.bool),
        "bare": torch.tensor([0, 0]),
        "legal": torch.ones(2, 6, dtype=torch.bool),
        "died": torch.tensor(
            [[True, False, False, False, False, False], [False, True, False, False, False, False]]
        ),
        "ordinary": torch.tensor([False, True]),
        "status": torch.ones(2, 6, dtype=torch.bool),
        "life_id": torch.tensor([0, 1]),
    }
    corrected = torch.tensor([[0.0, 2.0, 0, 0, 0, 0], [0.0, 2.0, 0, 0, 0, 0]])
    report = metrics(corrected, table)
    assert report["rescue_retention"] == 1.0
    assert report["ordinary_harm_proxy"] == 1.0
    assert _served(corrected, table["action_mask"]).tolist() == [1, 1]


def test_tiny_cpu_entrypoint_exports_all_config_hashes(tmp_path, monkeypatch):
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
        learning_rate=1e-3,
        seed=3,
        synthetic_fixture=True,
    )
    from research.colab_cli.runtime import _exports
    from research.sandbox_b_heads import job

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
    assert len(payload["rows"]) == 60
    assert len(list((output / "checkpoints").glob("*.pt.sha256"))) == 60
    assert (output / "checkpoints.tar").is_file()

    # Actual qualified immutable-export scanner: poll after all seals exist, then again
    # during fresh-directory restore. Already scanned payload signatures must stay fixed.
    inventory = _exports(root, 1_000_000_000, cache)
    assert len([item for item in inventory if item["path"].startswith("units/")]) == 60
    assert _exports(root, 1_000_000_000, cache) == inventory
    frozen = recipe(data, b8, updates=1, batch_size=2, seed=3)
    ledger = tmp_path / "mac-ledger.json"
    built = build_ledger(output, ledger, frozen, namespace="synthetic")
    assert len(built["units"]) == 60
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
        "research.sandbox_b_heads.job._cache_features",
        lambda *a: pytest.fail("fully skipped resume extracted features"),
    )
    args.output = str(tmp_path / "fresh-resume")
    args.ledger = str(remote_ledger)
    resumed = json.loads(run(args).read_text())
    assert resumed["rows"] == payload["rows"]
    assert len(list((tmp_path / "fresh-resume" / "checkpoints").glob("*.pt"))) == 60
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
    assert verify_ledger(ledger, baseline)["units"] == []
    data.write_bytes(b"c")
    with pytest.raises(ValueError, match="foreign"):
        verify_ledger(ledger, recipe(data, b8))


def test_package_uploads_portable_verified_bundles_and_bootstrap_sources(tmp_path, monkeypatch):
    from research.sandbox_b_heads import job, package
    from research.sandbox_b_heads.resume import identity
    from tools.sandbox import registry

    # This test owns packet portability; registry admission is tested separately.
    monkeypatch.setattr(registry, "load_registry", lambda _: {})
    monkeypatch.setattr(registry, "validate_npz", lambda *a, **kw: None)
    monkeypatch.setattr(registry, "validate_rows", lambda *a, **kw: None)
    monkeypatch.setattr(package, "validate_projection", lambda _: None)
    monkeypatch.setattr(package, "_git_commit", lambda: "1" * 40)
    parents = np.array([f"synthetic-{i}" for i in range(20)])
    data, b8, registry_path = tmp_path / "rows.npz", tmp_path / "b8.pt", tmp_path / "registry.json"
    np.savez(data, parent_world=parents, row_id=parents, life_id=parents)
    b8.write_bytes(b"frozen")
    registry_path.write_text("{}")
    frozen = recipe(data, b8)
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"weights")
    unit_id = identity(
        0, "a100-dev/sandbox-b/heads-v1", "split-v1", empty_ledger(frozen)["fingerprint"]
    )
    bundle = tmp_path / "00-linear-kl1-survival1.tar"
    row = {"config": unit_id["id"], "status": "COMPLETE", "sha256": sha256(checkpoint)}
    job._publish_bundle(bundle, checkpoint, row, unit_id)
    ledger = tmp_path / "source-ledger.json"
    build_ledger(tmp_path, ledger, frozen)
    manifest, sandbox = tmp_path / "packet.json", tmp_path / "sandbox.json"
    package.build(data, registry_path, b8, ledger, manifest, tmp_path / "fresh-output", 3, sandbox)
    packet = json.loads(manifest.read_text())
    resume_input = next(item for item in packet["inputs"] if item["dest"].startswith("resume-"))
    assert resume_input["path"] == str(bundle.resolve())
    portable_input = next(
        item for item in packet["inputs"] if item["dest"] == "verified-ledger.json"
    )
    portable = json.loads(Path(portable_input["path"]).read_text())
    assert portable["units"][0]["bundle"]["path"] == resume_input["dest"]
    assert portable["units"][0]["bundle"]["sha256"] == resume_input["sha256"]
    assert len(sandbox.read_text().splitlines()) == 10
    assert any(
        item["kind"] == "verified-unit" for item in json.loads(sandbox.read_text())["inputs"]
    )
    assert packet["env"]["COLAB_CLI_REQUIREMENTS"] == "research/sandbox_b_heads/requirements.txt"
    assert "research/sandbox_b_heads/resume.py" in packet["source_paths"]
    assert "src/model/checkpoint_io.py" in packet["source_paths"]
    assert all(Path(name).is_file() for name in packet["source_paths"])
    assert not any(name.endswith(".sha256") for name in packet["required_exports"])


def test_changed_unknown_action_counts_as_ordinary_harm():
    table = {
        "q": torch.zeros(1, 6),
        "action_mask": torch.ones(1, 6, dtype=torch.bool),
        "bare": torch.tensor([0]),
        "legal": torch.ones(1, 6, dtype=torch.bool),
        "died": torch.zeros(1, 6, dtype=torch.bool),
        "ordinary": torch.tensor([True]),
        "status": torch.tensor([[True, False, True, True, True, True]]),
        "life_id": torch.tensor([0]),
    }
    report = metrics(torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]]), table)
    assert report["ordinary_harm_proxy"] == 1.0
    assert report["ordinary_harm_count"] == 1


def test_survivor_target_never_uses_unknown_or_dead_high_mass():
    from research.bl_residual.fit import FitRecipe, _preference_loss
    from research.sandbox_b_heads.job import _survival_pair_loss

    q = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    table = {
        "action_mask": torch.ones(1, 6, dtype=torch.bool),
        "legal": torch.ones(1, 6, dtype=torch.bool),
        "status": torch.tensor([[True, True, False, False, False, False]]),
        "died": torch.tensor([[True, False, False, False, False, False]]),
        "mass": torch.tensor([[999.0, 1.0, 10000.0, 10000.0, 10000.0, 10000.0]]),
    }
    loss = _survival_pair_loss(q, table)
    assert torch.equal(loss, _preference_loss(q, table, FitRecipe()))
    loss.backward()
    assert q.grad[0, 0] > 0 and q.grad[0, 1] < 0
    assert torch.equal(q.grad[0, 2:], torch.zeros(4))


def test_metrics_uses_real_precomputed_validation_actions():
    table = {
        "q": torch.zeros(1, 6),
        "action_mask": torch.ones(1, 6, dtype=torch.bool),
        "bare": torch.tensor([0]),
        "legal": torch.ones(1, 6, dtype=torch.bool),
        "died": torch.tensor([[True, False, False, False, False, False]]),
        "ordinary": torch.tensor([False]),
        "status": torch.ones(1, 6, dtype=torch.bool),
        "life_id": torch.tensor([0]),
    }
    q = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
    assert metrics(q, table) == metrics(q, table, _served(q, table["action_mask"]))
    assert metrics(q, table, torch.tensor([0]))["rescue_retention"] == 0.0
