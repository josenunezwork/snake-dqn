"""Synthetic refusals for the F2 input boundary; no historical payloads are opened."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from research.a100_a2.protocol import NAMESPACE, allocation, unit_name
from research.bl_residual import f2_inputs as inputs


def _table() -> tuple[dict[str, np.ndarray], np.ndarray]:
    folds = np.repeat(np.arange(5), 2).astype(np.int16)
    table = {
        "world": np.asarray([f"w-{fold}" for fold in folds]),
        "fold": folds.copy(),
        "legal": np.ones((10, 6), bool),
        "died": np.zeros((10, 6), bool),
        "alive_steps": np.full((10, 6), 300, np.int16),
        "observed": np.ones((10, 6), bool),
        "decisive": np.zeros(10, bool),
        "a_star": np.full(10, -1, np.int16),
        "q": np.full((10, 6), np.nan, np.float32),
    }
    return table, folds


def _prediction_entries(
    tmp_path: Path, table: dict[str, np.ndarray], folds: np.ndarray
) -> list[dict]:
    entries = []
    for spec in allocation():
        if not (
            spec["table"] == "t5"
            and spec["encoder"] == "c0"
            and spec["target"] == "d300"
            and spec["dose"] in (25, 100)
        ):
            continue
        rows = np.flatnonzero(folds == spec["fold"])
        path = tmp_path / unit_name(spec["index"])
        data = {key: value[rows] for key, value in table.items()}
        data.update(
            namespace=np.asarray(NAMESPACE),
            allocation=np.asarray(inputs.ALLOCATION),
            index=np.asarray(spec["index"]),
            row_index=rows,
            p_d300=np.full((len(rows), 6), 0.25 + spec["dose"] / 1000, np.float32),
            **{
                key: np.asarray(spec[key]) for key in ("table", "encoder", "target", "dose", "seed")
            },
        )
        np.savez(path, **data)
        entries.append(
            {
                "index": spec["index"],
                "path": str(path),
                "complete": str(path) + ".complete",
                "sidecar": str(path) + ".sha256",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
        Path(str(path) + ".complete").write_text("complete\n")
    return entries


def test_safe_path_refuses_lexical_and_resolved_round0_ancestry(tmp_path: Path) -> None:
    with pytest.raises(inputs.LineageError):
        inputs.safe_path(tmp_path / "t34-round0-study-01" / "never-open.npz")
    target = tmp_path / "t34-round0-study-01"
    target.mkdir()
    link = tmp_path / "ordinary"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(inputs.LineageError):
        inputs.safe_path(link / "x")


def test_prediction_union_requires_full_verified_seed_fold_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, folds = _table()
    monkeypatch.setattr(inputs, "A2_EXPORTS", tmp_path)
    entries = _prediction_entries(tmp_path, table, folds)
    monkeypatch.setattr(
        inputs, "verify_export", lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    )
    loaded = inputs._load_predictions({"prediction_units": entries}, table, folds)
    assert loaded is not None and set(loaded) == {25, 100}
    assert loaded[25].shape == (10, 6)
    # A duplicate/missing fold cannot silently become a partial OOF result.
    entries[-1]["index"] = entries[-2]["index"]
    assert inputs._load_predictions({"prediction_units": entries}, table, folds) is None


def test_load_raw_accepts_identical_repeated_branch_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "part-1-decisive.npz"
    packed = np.asarray([[1, 2], [1, 2]], np.uint8)
    scalars = np.asarray([[3.0], [3.0]], np.float32)
    np.savez(shard, source_decision=np.asarray([2, 2]), state_packed=packed, state_scalars=scalars)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"shards":{"part-1-decisive.npz":"synthetic"}}')
    table = {
        "state_id": np.asarray(["world-1:d2"]),
        "scalars": scalars[:1],
        "horizon": np.full((1, 6), 300, np.int16),
        "legal": np.ones((1, 6), bool),
        "died": np.zeros((1, 6), bool),
        "world": np.asarray(["w-1"]),
        "boost_only_escape": np.asarray([True]),
    }
    monkeypatch.setattr(inputs, "validate_metadata", lambda _: None)
    monkeypatch.setattr(inputs, "_sha256", lambda _: "synthetic")
    monkeypatch.setattr(
        inputs, "unpack", lambda rows, _indices, _layout: (rows[:, :1], rows[:, :1])
    )
    result = inputs.load_raw(
        {"raw_manifests": [{"path": str(manifest)}]}, table, np.asarray([0], np.int16)
    )
    assert result["local"].shape == (1, 1)
    assert result["alive_steps"].tolist() == [[300] * 6]


def test_metadata_reference_keys_are_checked_for_forbidden_lineage(tmp_path: Path) -> None:
    with pytest.raises(inputs.LineageError):
        inputs._reject_forbidden_references({"t34-round0-study-01": "benign"}, tmp_path)


@pytest.mark.parametrize(
    ("marker", "sidecar"),
    [
        ("wrong\n", "a" * 64 + "  fit-0000.npz\n"),
        ("complete\n", "a" * 64 + "  wrong-name.npz\n"),
    ],
)
def test_prediction_sidecar_metadata_refuses_malformed_atomic_export(
    tmp_path: Path, marker: str, sidecar: str
) -> None:
    path = tmp_path / "fit-0000.npz"
    path.write_bytes(b"synthetic")
    Path(str(path) + ".complete").write_text(marker)
    Path(str(path) + ".sha256").write_text(sidecar)
    assert inputs._sidecar_entry(path, 0) is None
