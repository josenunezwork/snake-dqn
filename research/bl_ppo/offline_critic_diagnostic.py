"""Offline-only round-14 late-body critic representation diagnostic.

This module consumes the immutable exported natural complete-life rows after a
P1 refusal.  It is deliberately separate from the P1 gate: its fits never
alter a checkpoint, a PopArt head, or an admission decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-ppo-offline-critic-diagnostic/v1"
MIXES = ("scripted", "frozen", "mixed")
PROVENANCE = "natural"
LENGTH_BUCKETS = (
    (0.0, 800.0, "lt800"),
    (800.0, 1000.0, "800_999"),
    (1000.0, 1200.0, "1000_1199"),
    (1200.0, np.inf, "ge1200"),
)
_FIELDS = ("features", "returns", "lengths", "mixes", "provenance", "world_ids", "life_ids")


def _array(data: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(data[name])
    except KeyError as exc:
        raise ValueError(f"offline sidecar missing {name}") from exc


def _load_split(data: Any, split: str, *, allow_empty: bool = False) -> dict[str, np.ndarray]:
    """Load and validate one explicitly named exported split."""
    values = {field: _array(data, f"{split}_{field}") for field in _FIELDS}
    features, returns, lengths = values["features"], values["returns"], values["lengths"]
    size = len(returns)
    if features.ndim != 2 or features.shape != (size, 257):
        raise ValueError(f"{split}_features must have shape [N, 257]")
    if any(values[field].ndim != 1 or len(values[field]) != size for field in _FIELDS[1:]):
        raise ValueError(f"{split} sidecar columns must be matched vectors")
    if not size and not allow_empty:
        raise ValueError(f"{split} sidecar has no rows")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(returns).all()
        or not np.isfinite(lengths).all()
    ):
        raise ValueError(f"{split} sidecar contains non-finite values")
    if np.any(lengths < 0):
        raise ValueError(f"{split}_lengths must be non-negative")
    for name in ("world_ids", "life_ids"):
        if not np.issubdtype(values[name].dtype, np.integer):
            raise ValueError(f"{split}_{name} must be integral")
    for name in ("mixes", "provenance"):
        if values[name].dtype.kind not in {"U", "S", "O"}:
            raise ValueError(f"{split}_{name} must be strings")
    return values


def _load_npz(
    path: Path, *, allow_empty: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        return _load_split(data, "fit", allow_empty=allow_empty), _load_split(
            data, "heldout", allow_empty=allow_empty
        )


def _load_index(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    index_digest = verify_export(path)
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
        chunks = index["chunks"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("offline sidecar index requires a chunks list") from exc
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("offline sidecar index chunks must be a non-empty list")
    collected: dict[str, list[dict[str, np.ndarray]]] = {"fit": [], "heldout": []}
    for row in chunks:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("offline sidecar index chunk requires a relative path")
        relative = Path(row["path"])
        candidate = path.parent / relative
        if relative.is_absolute() or ".." in relative.parts or not candidate.is_file():
            raise ValueError("offline sidecar index chunk path is unsafe or missing")
        chunk_digest = verify_export(candidate)
        if not isinstance(row.get("sha256"), str) or chunk_digest != row["sha256"]:
            raise ValueError(f"offline sidecar chunk digest mismatch: {relative}")
        fit, heldout = _load_npz(candidate, allow_empty=True)
        count = len(fit["returns"]) + len(heldout["returns"])
        if row.get("rows") != count:
            raise ValueError(f"offline sidecar chunk row count mismatch: {relative}")
        collected["fit"].append(fit)
        collected["heldout"].append(heldout)
    values = tuple(
        {field: np.concatenate([part[field] for part in collected[split]]) for field in _FIELDS}
        for split in ("fit", "heldout")
    )
    return (
        values[0],
        values[1],
        {
            "index_path": str(path),
            "index_sha256": index_digest,
            "chunks": [{"path": row["path"], "sha256": row["sha256"]} for row in chunks],
        },
    )


def load_sidecar(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read a hash-checked NPZ or chunk index and select natural rows only."""
    if path.suffix == ".json":
        fit, heldout, binding = _load_index(path)
    else:
        sidecar_digest = verify_export(path)
        fit, heldout = _load_npz(path)
        binding = {"sidecar_path": str(path), "sidecar_sha256": sidecar_digest}
    if set(fit["world_ids"].tolist()) & set(heldout["world_ids"].tolist()):
        raise ValueError("offline sidecar fit and heldout worlds overlap")
    return _natural(fit, "fit"), _natural(heldout, "heldout"), binding


def _natural(rows: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    natural = np.asarray(rows["provenance"], dtype=str) == PROVENANCE
    chosen = {name: value[natural] for name, value in rows.items()}
    if not len(chosen["returns"]):
        raise ValueError(f"offline sidecar has no natural {split} rows")
    h = chosen["features"][:, -1]
    if np.any((h < 0.0) | (h > 1.0)):
        raise ValueError(f"natural {split} normalized h must lie in [0, 1]")
    invalid_mixes = set(np.asarray(chosen["mixes"], dtype=str)) - set(MIXES)
    if invalid_mixes:
        raise ValueError(f"natural {split} rows contain invalid mixes: {sorted(invalid_mixes)}")
    return chosen


def _design(features: np.ndarray, lengths: np.ndarray, head: str) -> np.ndarray:
    numeric_features = np.asarray(features, dtype=np.float64)
    z, h = numeric_features[:, :-1], numeric_features[:, -1:]
    ones = np.ones((len(features), 1), dtype=np.float64)
    if head == "registered_z_h":
        return np.concatenate((ones, z, h), axis=1)
    if head == "interaction_z_h_hz":
        return np.concatenate((ones, z, h, h * z), axis=1)
    if head == "baseline_1_h_L":
        return np.concatenate(
            (ones, h, np.asarray(lengths, dtype=np.float64).reshape(-1, 1)), axis=1
        )
    raise ValueError(f"unknown offline diagnostic head {head}")


def _fit_predict(
    fit: dict[str, np.ndarray], heldout: dict[str, np.ndarray], head: str
) -> np.ndarray:
    # np.linalg.lstsq is deterministic on the one-threaded CPU invocation and
    # is the exact convergence solution for these linear affine heads.
    coefficient, *_ = np.linalg.lstsq(
        _design(fit["features"], fit["lengths"], head),
        np.asarray(fit["returns"], dtype=np.float64),
        rcond=None,
    )
    return _design(heldout["features"], heldout["lengths"], head) @ coefficient


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    variance = float(np.var(target))
    mse = float(np.mean(np.square(error)))
    return {
        "count": int(len(target)),
        "explained_variance": float(1.0 - float(np.var(error)) / max(variance, 1e-8)),
        "mse": mse,
        "bias": float(np.mean(error)),
        "target_variance": variance,
        "ev_defined": bool(variance > 0.0),
    }


def _metric_set(
    target: np.ndarray, predictions: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    baseline_mse = _metrics(target[mask], predictions["baseline_1_h_L"][mask])["mse"]
    result: dict[str, Any] = {}
    for name, prediction in predictions.items():
        metrics = _metrics(target[mask], prediction[mask])
        metrics["baseline_mse"] = baseline_mse
        result[name] = metrics
    return result


def _strata(rows: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mix in MIXES:
        mix_mask = np.asarray(rows["mixes"], dtype=str) == mix
        by_length: dict[str, Any] = {}
        for lower, upper, label in LENGTH_BUCKETS:
            mask = mix_mask & (rows["lengths"] >= lower) & (rows["lengths"] < upper)
            by_length[label] = (
                _metric_set(rows["returns"], predictions, mask) if mask.any() else None
            )
        result[mix] = by_length
    return result


def diagnose(fit: dict[str, np.ndarray], heldout: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit predeclared heads only on natural FIT rows and evaluate HELDOUT."""
    heads = ("registered_z_h", "interaction_z_h_hz", "baseline_1_h_L")
    predictions = {head: _fit_predict(fit, heldout, head) for head in heads}
    late = heldout["lengths"] >= 800.0
    if not late.any():
        raise ValueError("offline sidecar has no natural heldout L >= 800 rows")
    overall = _metric_set(heldout["returns"], predictions, late)
    return {
        "schema": SCHEMA,
        "purpose": "offline diagnostic only; no substitution into P1",
        "fit": {
            "natural_rows": int(len(fit["returns"])),
            "natural_lives": int(len(set(zip(fit["world_ids"], fit["life_ids"])))),
        },
        "heldout": {
            "natural_rows": int(len(heldout["returns"])),
            "natural_lives": int(len(set(zip(heldout["world_ids"], heldout["life_ids"])))),
        },
        "fit_objective": (
            "ordinary least squares on raw return targets; exact deterministic minimizer of the "
            "fixed-affine-normalized quadratic PopArt loss"
        ),
        "heads": {
            "registered_z_h": "linear [1, z, h]",
            "interaction_z_h_hz": "linear [1, z, h, h*z]",
            "baseline_1_h_L": "linear [1, h, L]",
        },
        "heldout_natural_late": overall,
        "heldout_natural_by_mix_and_length": _strata(heldout, predictions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite offline diagnostic: {args.out}")
    fit, heldout, binding = load_sidecar(args.sidecar)
    result = diagnose(fit, heldout)
    result["input_binding"] = binding
    export_json(args.out, result)
    print(
        json.dumps(
            {"out": str(args.out), "heldout_natural_rows": result["heldout"]["natural_rows"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
