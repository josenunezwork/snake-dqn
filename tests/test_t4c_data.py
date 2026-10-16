"""Focused immutable-bank lineage, weighting, and matching contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.t4c_20261016 import data


def test_distance_boundary_and_counter_invariance() -> None:
    decisive = np.array([True, True, True, False, False])
    actual = data.distance_weights(
        decisive, np.array([201, 200, 1, 400, 0]), np.array([300, 300, 300, -1, 300])
    )
    np.testing.assert_array_equal(actual, np.array([1, 0.3, 0.3, 1, 1], np.float32))
    for gap in (-1, 300):
        with pytest.raises(ValueError, match="strict"):
            data.distance_weights(np.array([True]), np.array([0]), np.array([gap]))


def _v2(tmp_path: Path) -> Path:
    source, bank = tmp_path / "original", tmp_path / "v2"
    source.mkdir()
    bank.mkdir()
    provenance = [
        {"row": i, "source_key": "world", "decision": decision, "decisive": i != 3}
        for i, decision in enumerate((201, 200, 1, 0))
    ]
    (source / "bank-00000.provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance)
    )
    (source / "worlds").mkdir()
    (source / "worlds/world.json").write_text(json.dumps({"death_decision": 300}))
    surviving = np.tile([False, True, False, False, False, False], (4, 1))
    surviving[-1, 0] = True
    arrays = {
        "action": np.zeros(4, np.int64),
        "weight": np.ones(4, np.float32),
        "surviving_original": surviving,
        "surviving": surviving,
        "scalars": np.arange(240, dtype=np.float32).reshape(4, 60),
    }
    np.savez_compressed(bank / "bank-00000.npz", **arrays)
    files = {
        str(path.relative_to(source)): data.sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    (bank / "manifest.json").write_text(
        json.dumps(
            {
                "source": {"path": str(source), "files": files},
                "shards": {"bank-00000.npz": data.sha256(bank / "bank-00000.npz")},
            }
        )
    )
    return bank


def test_derived_bank_is_reversible_and_bound(tmp_path: Path) -> None:
    bank = _v2(tmp_path)
    before = data.bound_files(bank)
    out = tmp_path / "v3"
    result = data.derive_v3(bank, out)
    old, new = data.load_arrays(bank / "bank-00000.npz"), data.load_arrays(out / "bank-00000.npz")
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    np.testing.assert_array_equal(new["row_weight"], np.array([1, 0.3, 0.3, 1], np.float32))
    np.testing.assert_array_equal(new["row_family"], [0, 0, 0, 1])
    assert data.bound_files(bank) == before
    assert result["source"]["files"] == before
    with pytest.raises(ValueError, match="empty"):
        data.derive_v3(bank, out)


def test_derived_bank_rejects_changed_source_provenance(tmp_path: Path) -> None:
    bank = _v2(tmp_path)
    (tmp_path / "original/bank-00000.provenance.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="provenance hash"):
        data.derive_v3(bank, tmp_path / "v3")


def _candidates(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    bank, candidates = tmp_path / "bank", tmp_path / "candidates"
    bank.mkdir()
    candidates.mkdir()
    (bank / "manifest.json").write_text(json.dumps({"shards": {}}))
    cuts = np.arange(9, dtype=float)
    monkeypatch.setattr(
        data,
        "target_definition",
        lambda bank: (cuts, {"700_999:D1": 2, "ge1000:D1": 2}, {"bank-observation"}),
    )
    for oi, owner in enumerate(("CTRL", "B8")):
        for seed in range(10, 15):
            path = candidates / f"{owner}_s{seed}.npz"
            arrays = {
                "action": np.array([0, 0]),
                "q": np.tile([1, 0, 0, 0, 0, 0], (2, 1)).astype(np.float32),
                "length": np.array([800, 1100]),
                "stratum": np.array(["700_999:D1", "ge1000:D1"]),
                "fingerprint": np.array([f"{owner}-{seed}-{j}" for j in range(2)]),
                "decision": np.array([300, 900]),
                "death_decision": np.array([600, -1]),
                "mask": np.ones((2, 6), bool),
                "scalars": np.zeros((2, 60), np.float32),
                "packed": np.zeros((2, 3064), np.uint8),
            }
            data._atomic_npz(path, **arrays)
            data._atomic_json(
                path.with_suffix(".receipt.json"),
                {
                    "schema": "t4c-outside-candidates/v1",
                    "owner": owner,
                    "seed": seed,
                    "bank_manifest_sha256": data.sha256(bank / "manifest.json"),
                    "cuts": cuts.tolist(),
                    "sha256": data.sha256(path),
                    "inputs_sha256": {},
                },
            )
    return bank, candidates


def test_exact_outside_matching_and_no_replacement(tmp_path: Path, monkeypatch) -> None:
    bank, candidates = _candidates(tmp_path, monkeypatch)
    result = data.build_outside(candidates, tmp_path / "outside", bank)
    assert result["rows"] == result["unique_fingerprints"] == 4
    assert result["matching"]["selected_counts"] == result["matching"]["target_counts"]
    arrays = data.load_arrays(tmp_path / "outside/bank-00000.npz")
    assert (arrays["row_weight"] == 0.5).all()
    assert (arrays["row_family"] == 2).all()
    assert arrays["surviving"][:, 0].all()


def test_outside_insufficient_stratum_fails_without_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    bank, candidates = _candidates(tmp_path, monkeypatch)
    monkeypatch.setattr(
        data,
        "target_definition",
        lambda bank: (np.arange(9, dtype=float), {"700_999:D1": 11}, set()),
    )
    with pytest.raises(ValueError, match="insufficient"):
        data.build_outside(candidates, tmp_path / "outside", bank)
    assert not (tmp_path / "outside/manifest.json").exists()


def test_capture_source_snapshot_preserves_original_receipt(tmp_path: Path) -> None:
    from research.t4c_20261016.data_sources import verify_capture_inputs

    repo, snapshots = tmp_path / "repo", tmp_path / "snapshots"
    repo.mkdir()
    snapshots.mkdir()
    source = repo / "collector.py"
    source.write_text("original capture code\n")
    expected = data.sha256(source)
    inputs = {str(source): expected}
    assert verify_capture_inputs(inputs, snapshot_root=snapshots, repo=repo) == {}
    (snapshots / f"{expected}.py").write_bytes(source.read_bytes())
    source.write_text("formatted current code\n")
    recovered = verify_capture_inputs(inputs, snapshot_root=snapshots, repo=repo)
    entry = recovered[f"{source}@{expected}"]
    assert entry["original_sha256"] == entry["snapshot_sha256"] == expected
    assert inputs == {str(source): expected}
    assert entry["live_source_sha256"] == data.sha256(source)


def test_source_fallback_never_accepts_changed_checkpoint_or_invalid_snapshot(
    tmp_path: Path,
) -> None:
    from research.t4c_20261016.data_sources import verify_capture_inputs

    repo, snapshots = tmp_path / "repo", tmp_path / "snapshots"
    repo.mkdir()
    snapshots.mkdir()
    checkpoint = repo / "final.pth"
    checkpoint.write_bytes(b"old")
    expected = data.sha256(checkpoint)
    (snapshots / f"{expected}.py").write_bytes(b"old")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        verify_capture_inputs({str(checkpoint): expected}, snapshot_root=snapshots, repo=repo)
    source = repo / "collector.py"
    source.write_bytes(b"changed")
    (snapshots / f"{expected}.py").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="snapshot unavailable or changed"):
        verify_capture_inputs({str(source): expected}, snapshot_root=snapshots, repo=repo)
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"changed")
    (snapshots / f"{expected}.py").write_bytes(b"old")
    with pytest.raises(ValueError, match="input changed"):
        verify_capture_inputs({str(outside): expected}, snapshot_root=snapshots, repo=repo)


@pytest.mark.parametrize("existing", ["both", "payload", "receipt"])
def test_outside_finish_refuses_existing_parts_without_changing_bytes(
    tmp_path: Path, existing: str
) -> None:
    from collections import Counter

    collector = data.OutsideCollector.__new__(data.OutsideCollector)
    collector.owner, collector.seed = "CTRL", 10
    collector.bank = tmp_path / "bank"
    collector.bank.mkdir()
    (collector.bank / "manifest.json").write_text("{}\n")
    collector.rows = {"700_999:D1": [{"env": 0, "decision": 0, "stratum": "700_999:D1"}]}
    collector.seen, collector.cuts = Counter(), np.arange(9, dtype=float)
    out = tmp_path / "candidates"
    out.mkdir()
    payload = out / "CTRL_s10.npz"
    receipt = payload.with_suffix(".receipt.json")
    if existing == "both":
        collector.finish(out, np.array([-1]), [123], {})
    elif existing == "payload":
        payload.write_bytes(b"existing immutable payload")
    else:
        receipt.write_bytes(b'{"existing":"immutable receipt"}\n')
    before = {path: path.read_bytes() for path in (payload, receipt) if path.exists()}
    with pytest.raises(FileExistsError, match="refusing existing"):
        collector.finish(out, np.array([-1]), [123], {})
    assert {path: path.read_bytes() for path in (payload, receipt) if path.exists()} == before


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_bound_bank_requires_exact_npz_inventory(tmp_path: Path, change: str) -> None:
    bank = _v2(tmp_path)
    data.bound_files(bank)
    if change == "extra":
        np.savez_compressed(bank / "unregistered.npz", scalars=np.zeros((1, 60)))
    else:
        (bank / "bank-00000.npz").unlink()
    with pytest.raises(ValueError, match="npz set"):
        data.bound_files(bank)
