"""Synthetic refusals for the F2 input boundary; no historical payloads are opened."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from research.a100_a2.protocol import NAMESPACE, allocation, unit_name
from research.bl_ppo.exports import digest, export_bytes
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
        stream = io.BytesIO()
        np.savez_compressed(stream, **data)
        export_bytes(path, stream.getvalue())
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


@pytest.mark.parametrize("component", [".sha256", ".complete", "metadata.json"])
def test_metadata_never_follows_symlink_to_payload(tmp_path, monkeypatch, component):
    path = tmp_path / "fit-0000.npz"
    path.write_bytes(b"synthetic NPZ placeholder")
    Path(str(path) + ".complete").write_text("complete\n")
    Path(str(path) + ".sha256").write_text(f"{digest(path)}  {path.name}\n")
    outcome = tmp_path / "closed-outcome.json"
    outcome.write_text('{"statistic": "must never be read"}')
    target = tmp_path / component if component == "metadata.json" else Path(str(path) + component)
    target.unlink(missing_ok=True)
    target.symlink_to(outcome)
    original = Path.read_text

    def no_outcome_read(p, *args, **kwargs):
        assert p != outcome, "outcome symlink followed"
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_outcome_read)
    if component == "metadata.json":
        with pytest.raises(ValueError):
            inputs._json(target)
    else:
        assert inputs._sidecar_entry(path, 0) is None


def test_metadata_root_symlink_refused_before_any_read(tmp_path, monkeypatch):
    target = tmp_path / "closed-root"
    target.mkdir()
    link = tmp_path / "t5-root"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(inputs, "_json", lambda *_: pytest.fail("redirected metadata was opened"))
    with pytest.raises(ValueError):
        inputs.metadata_inputs(t5_root=link, a2_exports=tmp_path, t4e_root=tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "payload",
        "missing-payload",
        "marker",
        "checksum-name",
        "factorial",
        "nan",
        "row-order",
        "fractional-row",
    ],
)
def test_prediction_refusals_use_real_atomic_export_checks(tmp_path, monkeypatch, fault):
    table, folds = _table()
    monkeypatch.setattr(inputs, "A2_EXPORTS", tmp_path)
    entries = _prediction_entries(tmp_path, table, folds)
    path = Path(entries[0]["path"])
    if fault == "payload":
        path.write_bytes(path.read_bytes() + b"tampered")
    elif fault == "missing-payload":
        path.unlink()
    elif fault == "marker":
        Path(str(path) + ".complete").write_text("partial\n")
    elif fault == "checksum-name":
        Path(str(path) + ".sha256").write_text(f"{digest(path)}  foreign.npz\n")
    else:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if fault == "factorial":
            data["encoder"] = np.asarray("s_random")
        elif fault == "nan":
            data["p_d300"][0, 0] = np.nan
        elif fault == "row-order":
            data["row_index"] = data["row_index"][::-1]
        else:
            data["row_index"] = data["row_index"].astype(float) + 0.25
        for suffix in ("", ".sha256", ".complete"):
            Path(str(path) + suffix).unlink()
        stream = io.BytesIO()
        np.savez_compressed(stream, **data)
        export_bytes(path, stream.getvalue())
        entries[0].update(sha256=digest(path), size=path.stat().st_size)
    assert inputs._load_predictions({"prediction_units": entries}, table, folds) is None


def _full_fixture(tmp_path, monkeypatch):
    """The original T5 schema and production dimensions, with invented outcomes."""
    t5 = tmp_path / "t5"
    a2 = tmp_path / "a2"
    t4e = tmp_path / "t4e"
    (t5 / "fits").mkdir(parents=True)
    a2.mkdir()
    n = 19976
    world_index = np.arange(n) % 342
    world_index[:1481] = np.arange(1481) % 269
    world = np.asarray([f"world-{i:05d}" for i in world_index])
    fold = (world_index % 5).astype(np.int16)
    legal = np.ones((n, 6), dtype=bool)
    legal[:16592, 5] = False
    died = np.ones((n, 6), dtype=bool)
    died[:, 1] = False
    table = {
        "world": world,
        "state_id": np.asarray([f"{w}:d{i:05d}" for i, w in enumerate(world)]),
        "scalars": np.zeros((n, 60), np.float32),
        "legal": legal,
        "died": died,
        "horizon": np.where(died, 30, 300).astype(np.int16),
        "truncated_at_300": ~died,
        "boost_only_escape": np.arange(n) < 1481,
    }
    np.savez_compressed(t5 / "features.npz", **table)
    np.savez_compressed(t5 / "fits/folds.npz", world=world, state_id=table["state_id"], fold=fold)
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"synthetic checkpoint; never deserialized")
    for key, value in {
        "T5_ROOT": t5,
        "A2_EXPORTS": a2,
        "T4E_ROOT": t4e,
        "B8_CHECKPOINT": checkpoint,
        "EXPECTED_T5_SHA": digest(t5 / "features.npz"),
        "FOLDS_SHA": digest(t5 / "fits/folds.npz"),
        "B8_SHA": digest(checkpoint),
    }.items():
        monkeypatch.setattr(inputs, key, value)
    manifest_shas = {}
    for split in ("train", "holdout"):
        path = t4e / "banks" / split / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"shards": {}, "split": split}))
        manifest_shas[split] = digest(path)
    monkeypatch.setattr(inputs, "EXPECTED_T4E_MANIFESTS", manifest_shas)
    part_receipts = {}
    for i in range(4):
        path = t5 / f"features-part-{i}.json"
        path.write_text(json.dumps({"schema": "synthetic-provenance", "part": i}))
        part_receipts[str(path)] = digest(path)
    (t5 / "features.json").write_text(
        json.dumps(
            {
                "schema": "t5-separability-features/v1",
                "output": str(t5 / "features.npz"),
                "output_sha256": inputs.EXPECTED_T5_SHA,
                "input_sha256": {
                    str(t4e / "banks" / split / "manifest.json"): sha
                    for split, sha in manifest_shas.items()
                },
                "part_receipts": part_receipts,
            }
        )
    )
    (a2 / "allocation.json").write_text(
        json.dumps({"namespace": NAMESPACE, "allocation": inputs.ALLOCATION})
    )
    (a2 / "materialization.json").write_text(
        json.dumps(
            {
                "schema": "a2-materialization/v1",
                "t5_states": 19976,
                "t5_legal_branches": 103264,
                "inputs": [
                    {"path": "t5/features.npz", "sha256": inputs.EXPECTED_T5_SHA},
                    {"path": "t5/folds.npz", "sha256": inputs.FOLDS_SHA},
                ],
            }
        )
    )
    normalized = table | {
        "fold": fold,
        "alive_steps": np.where(died, table["horizon"] - 1, table["horizon"]).astype(np.int16),
        "observed": legal,
        "decisive": table["boost_only_escape"],
        "a_star": np.full(n, -1, np.int16),
        "q": np.full((n, 6), np.nan, np.float32),
    }
    _prediction_entries(a2, normalized, fold)
    return t5, a2, t4e, checkpoint


def test_original_t5_schema_full_oof_pipeline_is_synthetic_and_fast(tmp_path, monkeypatch):
    from research.bl_residual import f2

    t5, a2, t4e, checkpoint = _full_fixture(tmp_path, monkeypatch)
    plan = inputs.metadata_inputs(t5_root=t5, a2_exports=a2, t4e_root=t4e, checkpoint=checkpoint)
    assert len(plan["prediction_units"]) == 20
    table, folds, predictions = inputs.load_inputs(plan)
    assert predictions is not None
    assert len(folds) == 19976
    assert (
        table["boost_only_escape"].sum(),
        len(np.unique(table["world"][table["boost_only_escape"]])),
    ) == (1481, 269)
    # All invented hazards tie, hence lowest legal index selects a death in both doses.
    result = f2.statistic(table, predictions)
    assert result["volume_struck"] is True
    assert result["paired_delta"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-folds",
        "synthetic-materialization",
        "lineage-key",
        "lineage-link",
        "omitted-metadata",
    ],
)
def test_metadata_refusals_happen_before_payload_hash_or_load(tmp_path, monkeypatch, fault):
    t5, a2, t4e, checkpoint = _full_fixture(tmp_path, monkeypatch)
    plan = inputs.metadata_inputs(t5_root=t5, a2_exports=a2, t4e_root=t4e, checkpoint=checkpoint)
    if fault == "foreign-folds":
        plan["folds"]["sha256"] = "f" * 64
    elif fault == "omitted-metadata":
        plan["metadata"] = []
    else:
        path = a2 / "materialization.json"
        data = json.loads(path.read_text())
        if fault == "synthetic-materialization":
            data["synthetic"] = True
        elif fault == "lineage-key":
            data["t34-round0-study-01/forbidden.npz"] = "never-open"
        else:
            target = tmp_path / "t34-round0-study-01"
            target.mkdir()
            (a2 / "ordinary-link").symlink_to(target, target_is_directory=True)
            data["source"] = "ordinary-link/payload.npz"
        path.write_text(json.dumps(data))
        for row in plan["metadata"]:
            if row["path"] == str(path):
                row.update(sha256=digest(path), size=path.stat().st_size)
    original_hash = inputs._sha256

    def no_payload_hash(path):
        assert Path(path).suffix not in (".npz", ".pth"), "refusal reached payload hash"
        return original_hash(path)

    monkeypatch.setattr(inputs, "_sha256", no_payload_hash)
    monkeypatch.setattr(inputs.np, "load", lambda *_a, **_k: pytest.fail("payload deserialized"))
    with pytest.raises(ValueError):
        inputs.validate_metadata(plan)
