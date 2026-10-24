"""Sealed F2 input boundary.

``metadata_inputs`` is deliberately sidecar-only: it does not open an NPZ,
hash an outcome payload, or deserialize an A2 prediction.  Payload reads are
kept in the two explicit run-time loaders below, after the input ledger has
been published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.a100_a2.data import EXPECTED_T4E_MANIFESTS, EXPECTED_T5_SHA
from research.a100_a2.protocol import B8_SHA, NAMESPACE, allocation, unit_name
from research.bl_ppo.exports import verify_export
from research.redesign_m3c_20261008.obs_pack import train_layout, unpack

T5_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016")
A2_EXPORTS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/"
    "pack-20261022-07/main-output/exports"
)
T4E_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016")
FORBIDDEN = "t34-round0-study-01"
ALLOCATION = "8045413220ac81e2ef7fc06435dabe6d83af4f8b09287c95c61b99943d88cc3b"
B8_CHECKPOINT = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
FOLDS_SHA = "8940a6217c271987da98db7b1a80471067d0807c9884932c0e5e08aeab1f9fb2"


class LineageError(ValueError):
    """A referenced path could reach the sealed Round-0 study."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(path: Path | str) -> Path:
    """Resolve a path only after rejecting lexical and symlinked forbidden ancestry."""
    raw = Path(path)
    if FORBIDDEN in raw.as_posix().lower():
        raise LineageError("forbidden Round-0 ancestry")
    resolved = raw.resolve(strict=False)
    if FORBIDDEN in resolved.as_posix().lower():
        raise LineageError("forbidden Round-0 ancestry")
    return resolved


def _file(path: Path, *, sha256: str | None = None) -> dict[str, Any]:
    path = safe_path(path)
    if not path.is_file():
        raise ValueError(f"missing F2 input: {path}")
    return {"path": str(path), "sha256": sha256, "size": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    path = safe_path(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not an object: {path}")
    return value


def _reject_forbidden_references(value: Any, base: Path) -> None:
    """Reject every string reference without following or opening its target."""
    if isinstance(value, str):
        if FORBIDDEN in value.lower():
            raise LineageError("forbidden Round-0 metadata reference")
        if value.startswith("/") or "/" in value or value.endswith((".json", ".npz")):
            safe_path(Path(value) if value.startswith("/") else base / value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _reject_forbidden_references(key, base)
            _reject_forbidden_references(child, base)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_references(child, base)


def _sidecar_entry(path: Path, index: int) -> dict[str, Any] | None:
    """Read only a checksum sidecar plus completion marker; never the NPZ itself."""
    path = safe_path(path)
    marker = safe_path(Path(str(path) + ".complete"))
    sidecar = safe_path(Path(str(path) + ".sha256"))
    if any(item.is_symlink() for item in (path, marker, sidecar)) or not (
        path.is_file() and marker.is_file() and sidecar.is_file()
    ):
        return None
    tokens = sidecar.read_text().strip().split()
    if (
        marker.read_text() != "complete\n"
        or len(tokens) != 2
        or tokens[1] != path.name
        or len(tokens[0]) != 64
        or any(c not in "0123456789abcdef" for c in tokens[0])
    ):
        return None
    return {
        "index": index,
        "path": str(path),
        "name": path.name,
        "sha256": tokens[0],
        "size": path.stat().st_size,
        "complete": str(marker),
        "sidecar": str(sidecar),
    }


def _selected_indices() -> dict[int, tuple[int, int]]:
    wanted: dict[int, tuple[int, int]] = {}
    for row in allocation():
        if (
            row["table"] == "t5"
            and row["encoder"] == "c0"
            and row["target"] == "d300"
            and row["dose"] in (25, 100)
        ):
            wanted[row["index"]] = (int(row["dose"]), int(row["fold"]))
    if len(wanted) != 20 or sorted(wanted) != list(range(10)) + list(range(20, 30)):
        raise RuntimeError("A2 c0 selection no longer has its frozen allocation indices")
    return wanted


def metadata_inputs(
    *,
    t5_root: Path = T5_ROOT,
    a2_exports: Path = A2_EXPORTS,
    t4e_root: Path = T4E_ROOT,
    checkpoint: Path = B8_CHECKPOINT,
) -> dict[str, Any]:
    """Prepare the hash/lineage plan without opening a table or prediction payload."""
    t5_root, a2_exports, t4e_root = map(safe_path, (t5_root, a2_exports, t4e_root))
    features = _file(t5_root / "features.npz", sha256=EXPECTED_T5_SHA)
    folds = _file(
        t5_root / "fits/folds.npz",
        sha256=FOLDS_SHA,
    )
    # These are the only T5 ancestry receipts; never discover arbitrary result
    # JSON files in this directory (some are later rescue/outcome payloads).
    metadata: list[dict[str, Any]] = []
    required_features_receipt = t5_root / "features.json"
    if not required_features_receipt.is_file():
        raise ValueError("missing required T5 features lineage receipt")
    for name in (
        "features.json",
        "features-part-0.json",
        "features-part-1.json",
        "features-part-2.json",
        "features-part-3.json",
    ):
        path = t5_root / name
        if not path.is_file():
            continue
        value = _json(path)
        _reject_forbidden_references(value, path.parent)
        metadata.append(_file(path, sha256=_sha256(path)))
    allocation_path, materialization_path = (
        a2_exports / "allocation.json",
        a2_exports / "materialization.json",
    )
    allocation_value, materialization = _json(allocation_path), _json(materialization_path)
    _reject_forbidden_references(allocation_value, allocation_path.parent)
    _reject_forbidden_references(materialization, materialization_path.parent)
    if allocation_value != {"namespace": NAMESPACE, "allocation": ALLOCATION}:
        raise ValueError("A2 export namespace/allocation differs from the frozen factorial")
    metadata.extend(
        (
            _file(allocation_path, sha256=_sha256(allocation_path)),
            _file(materialization_path, sha256=_sha256(materialization_path)),
        )
    )
    manifests: list[dict[str, Any]] = []
    for split, expected in EXPECTED_T4E_MANIFESTS.items():
        path = t4e_root / "banks" / split / "manifest.json"
        value = _json(path)
        _reject_forbidden_references(value, path.parent)
        manifests.append(_file(path, sha256=expected))
    units = []
    for index, (dose, fold) in _selected_indices().items():
        entry = _sidecar_entry(a2_exports / unit_name(index), index)
        if entry is not None:
            entry.update(dose=dose, fold=fold, seed=allocation()[index]["seed"], spec="t5/c0/d300")
            units.append(entry)
    checkpoint = _file(
        checkpoint,
        sha256=B8_SHA,
    )
    return {
        "features": features,
        "folds": folds,
        "metadata": metadata,
        "prediction_units": units,
        "allocation": ALLOCATION,
        "namespace": NAMESPACE,
        "checkpoint": checkpoint,
        "raw_manifests": manifests,
        "t4e_root": str(t4e_root),
    }


def validate_metadata(inputs: dict[str, Any]) -> None:
    """Verify fixed identity and recursive lineage before deserializing NPZ bytes."""
    if inputs.get("allocation") != ALLOCATION or inputs.get("namespace") != NAMESPACE:
        raise ValueError("foreign A2 allocation")
    if not (T5_ROOT / "features.json").is_file():
        raise ValueError("missing required T5 features lineage receipt")
    expected_metadata = {
        str(safe_path(T5_ROOT / name))
        for name in (
            "features.json",
            "features-part-0.json",
            "features-part-1.json",
            "features-part-2.json",
            "features-part-3.json",
        )
        if (T5_ROOT / name).is_file()
    } | {
        str(safe_path(A2_EXPORTS / "allocation.json")),
        str(safe_path(A2_EXPORTS / "materialization.json")),
    }
    metadata = inputs.get("metadata")
    if (
        not isinstance(metadata, list)
        or {item.get("path") for item in metadata} != expected_metadata
    ):
        raise ValueError("F2 metadata ancestry set differs")
    expected_manifests = {
        str(safe_path(T4E_ROOT / "banks" / split / "manifest.json"))
        for split in EXPECTED_T4E_MANIFESTS
    }
    manifests = inputs.get("raw_manifests")
    if (
        not isinstance(manifests, list)
        or {item.get("path") for item in manifests} != expected_manifests
    ):
        raise ValueError("F2 raw manifest ancestry set differs")
    # Only ancestry metadata is opened in this phase.  This must finish before
    # any feature, fold, prediction, shard, or checkpoint payload is hashed.
    for item in metadata + manifests:
        path = safe_path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError("metadata hash differs")
        _reject_forbidden_references(_json(path), path.parent)
    for split, expected in EXPECTED_T4E_MANIFESTS.items():
        path = str(safe_path(T4E_ROOT / "banks" / split / "manifest.json"))
        bound = [item for item in manifests if item["path"] == path]
        if len(bound) != 1 or bound[0].get("sha256") != expected:
            raise ValueError("T4e manifest pin differs from A2")
    materialization = _json(A2_EXPORTS / "materialization.json")
    if (
        materialization.get("schema") != "a2-materialization/v1"
        or materialization.get("synthetic") is True
        or materialization.get("t5_states") != 19976
        or materialization.get("t5_legal_branches") != 103264
    ):
        raise ValueError("A2 materialization identity differs")
    materialized = {
        item.get("path"): item.get("sha256") for item in materialization.get("inputs", [])
    }
    if (
        materialized.get("t5/features.npz") != EXPECTED_T5_SHA
        or materialized.get("t5/folds.npz") != FOLDS_SHA
    ):
        raise ValueError("A2 materialization does not bind original T5 table/folds")
    for name, expected in (("features", EXPECTED_T5_SHA),):
        item = inputs.get(name, {})
        if item.get("sha256") != expected or _sha256(safe_path(item["path"])) != expected:
            raise ValueError(f"pinned {name} hash differs")
    folds = inputs.get("folds", {})
    if folds.get("sha256") != FOLDS_SHA or _sha256(safe_path(folds["path"])) != FOLDS_SHA:
        raise ValueError("pinned folds hash differs")
    checkpoint = inputs.get("checkpoint", {})
    if checkpoint.get("sha256") != B8_SHA or _sha256(safe_path(checkpoint["path"])) != B8_SHA:
        raise ValueError("B8 checkpoint identity differs")


def _load_predictions(
    inputs: dict[str, Any], table: dict[str, np.ndarray], folds: np.ndarray
) -> dict[int, np.ndarray] | None:
    wanted = _selected_indices()
    entries = inputs.get("prediction_units", [])
    if len(entries) != 20 or {entry.get("index") for entry in entries} != set(wanted):
        return None
    output: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {25: [], 100: []}
    seen: set[int] = set()
    for entry in entries:
        index = entry["index"]
        path = safe_path(entry["path"])
        complete = safe_path(entry["complete"])
        sidecar = safe_path(entry["sidecar"])
        canonical = safe_path(A2_EXPORTS / unit_name(index))
        if (
            index in seen
            or path != canonical
            or complete != safe_path(Path(str(canonical) + ".complete"))
            or sidecar != safe_path(Path(str(canonical) + ".sha256"))
            or path.is_symlink()
            or complete.is_symlink()
            or sidecar.is_symlink()
            or not complete.is_file()
            or complete.read_text() != "complete\n"
        ):
            return None
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            return None
        try:
            if verify_export(path) != entry["sha256"]:
                return None
        except (OSError, ValueError):
            return None
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "namespace",
                "allocation",
                "index",
                "row_index",
                "world",
                "fold",
                "p_d300",
                "legal",
                "died",
                "alive_steps",
                "observed",
                "decisive",
                "a_star",
                "q",
                "table",
                "encoder",
                "target",
                "dose",
                "seed",
            }
            if not required.issubset(archive.files):
                return None
            if (str(archive["namespace"]), str(archive["allocation"]), int(archive["index"])) != (
                NAMESPACE,
                ALLOCATION,
                index,
            ):
                return None
            spec = allocation()[index]
            if any(
                str(archive[key]) != str(spec[key])
                for key in ("table", "encoder", "target", "dose", "seed")
            ):
                return None
            rows = archive["row_index"].astype(np.int64, copy=False)
            if (
                rows.ndim != 1
                or len(rows) != len(np.unique(rows))
                or np.any(rows < 0)
                or np.any(rows >= len(folds))
            ):
                return None
            fold = spec["fold"]
            if not np.array_equal(rows, np.flatnonzero(folds == fold)):
                return None
            for key in ("world", "fold", "legal", "died", "alive_steps", "observed", "decisive"):
                if not np.array_equal(archive[key], table[key][rows]):
                    return None
            values = archive["p_d300"].astype(np.float32, copy=False)
            if (
                values.shape != (len(rows), 6)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
            ):
                return None
            output[spec["dose"]].append((int(spec["seed"]), rows, values.copy()))
        seen.add(index)
    result: dict[int, np.ndarray] = {}
    for dose, chunks in output.items():
        if len(chunks) != 10:
            return None
        per_seed: list[np.ndarray] = []
        for seed in (20261022, 20261023):
            pieces = [(rows, values) for unit_seed, rows, values in chunks if unit_seed == seed]
            if len(pieces) != 5:
                return None
            merged = np.empty((len(folds), 6), np.float32)
            written = np.zeros(len(folds), bool)
            for rows, values in pieces:
                if written[rows].any():
                    return None
                merged[rows] = values
                written[rows] = True
            if not written.all():
                return None
            per_seed.append(merged)
        result[dose] = ((per_seed[0] + per_seed[1]) / np.float32(2.0)).astype(np.float32)
    return result


def load_inputs(
    inputs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, np.ndarray] | None]:
    """Open the T5 table and only the selected, hash-verified A2 c0 exports."""
    validate_metadata(inputs)
    with np.load(safe_path(inputs["features"]["path"]), allow_pickle=False) as archive:
        table = {name: archive[name].copy() for name in archive.files}
    with np.load(safe_path(inputs["folds"]["path"]), allow_pickle=False) as archive:
        if not {"world", "state_id", "fold"}.issubset(archive.files):
            raise ValueError("malformed published fold table")
        folds = archive["fold"].astype(np.int16, copy=True)
        if not np.array_equal(archive["world"], table["world"]) or not np.array_equal(
            archive["state_id"], table["state_id"]
        ):
            raise ValueError("published folds do not exactly align with state IDs")
    if np.any((folds < 0) | (folds > 4)) or len(np.unique(table["state_id"].astype(str))) != len(
        folds
    ):
        raise ValueError("published folds contain an invalid identity")
    for world in np.unique(table["world"].astype(str)):
        if np.unique(folds[table["world"].astype(str) == world]).size != 1:
            raise ValueError("parent world spans frozen folds")
    if (
        len(folds) != 19976
        or len(table["world"]) != 19976
        or len(np.unique(table["world"].astype(str))) != 342
        or int(np.asarray(table["legal"], bool).sum()) != 103264
    ):
        raise ValueError("T5 dimensions differ from the registered table")
    # A2 exported normalized endpoint fields, while the original T5 table
    # deliberately retains its raw horizon/truncation contract for F2's exact
    # H300 endpoint.  Present both representations to the verifier.
    legal = np.asarray(table["legal"], bool)
    died = np.asarray(table["died"], bool)
    horizon = np.asarray(table["horizon"])
    if died.shape != legal.shape or horizon.shape != legal.shape:
        raise ValueError("T5 endpoint arrays do not align")
    table["alive_steps"] = np.where(died, horizon - 1, horizon).astype(np.int16)
    table["observed"] = legal.copy()
    table["decisive"] = np.asarray(table["boost_only_escape"], bool)
    table["fold"] = folds
    return table, folds, _load_predictions(inputs, table, folds)


def load_raw(
    inputs: dict[str, Any], table: dict[str, np.ndarray], folds: np.ndarray
) -> dict[str, np.ndarray]:
    """Stream hash-checked T4e shards and join observations by canonical state ID only."""
    validate_metadata(inputs)
    wanted: dict[tuple[int, int], int] = {}
    for row, state_id in enumerate(table["state_id"]):
        left, decision = str(state_id).split(":d")
        key = (int(left.split("-")[-1]), int(decision))
        if key in wanted:
            raise ValueError("duplicate canonical T5 state identity")
        wanted[key] = row
    packed: list[np.ndarray | None] = [None] * len(wanted)
    scalar: list[np.ndarray | None] = [None] * len(wanted)
    for manifest_item in inputs["raw_manifests"]:
        manifest_path = safe_path(manifest_item["path"])
        manifest = _json(manifest_path)
        for name, expected in manifest.get("shards", {}).items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError("malformed T4e shard manifest")
            path = safe_path(manifest_path.parent / name)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError("T4e shard hash differs")
            with np.load(path, allow_pickle=False) as archive:
                decisions = archive["source_decision"]
                shard_packed, shard_scalars = archive["state_packed"], archive["state_scalars"]
                world = int(path.stem.split("-")[1])
                for index, decision in enumerate(decisions):
                    row = wanted.get((world, int(decision)))
                    if row is not None:
                        candidate_packed, candidate_scalar = (
                            shard_packed[index],
                            shard_scalars[index],
                        )
                        if packed[row] is not None and (
                            packed[row].dtype != candidate_packed.dtype
                            or packed[row].tobytes() != candidate_packed.tobytes()
                            or scalar[row].dtype != candidate_scalar.dtype
                            or scalar[row].tobytes() != candidate_scalar.tobytes()
                        ):
                            raise ValueError("conflicting canonical T4e observation")
                        packed[row] = candidate_packed.copy()
                        scalar[row] = candidate_scalar.copy()
    if any(value is None for value in packed) or any(value is None for value in scalar):
        raise ValueError("canonical T4e join is incomplete")
    scalar_array = np.stack(scalar)  # type: ignore[arg-type]
    if (
        scalar_array.dtype != np.asarray(table["scalars"]).dtype
        or scalar_array.tobytes() != np.asarray(table["scalars"]).tobytes()
    ):
        raise ValueError("canonical T4e scalar identity differs")
    local, global_ = unpack(np.stack(packed), np.arange(len(packed)), train_layout())  # type: ignore[arg-type]
    horizon = np.asarray(table["horizon"])
    legal = np.asarray(table["legal"], bool)
    died = np.asarray(table["died"], bool)
    if horizon.shape != legal.shape or died.shape != legal.shape:
        raise ValueError("T5 endpoint shape differs")
    return {
        "local": local,
        "global": global_,
        "scalars": scalar_array.astype(np.float32),
        "world": np.asarray(table["world"]).astype("U64"),
        "fold": folds.astype(np.int16),
        "legal": legal,
        "died": died,
        "alive_steps": np.where(died, horizon - 1, horizon).astype(np.int16),
        "observed": legal.copy(),
        "decisive": np.asarray(table["boost_only_escape"], bool),
        "a_star": np.full(len(folds), -1, np.int16),
        "q": np.full((len(folds), 6), np.nan, np.float32),
    }
