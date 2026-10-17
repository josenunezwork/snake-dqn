"""Derive the immutable T11 LOCAL30 two-sided label bank from T4-prep v2.

This producer deliberately has no trainer or device imports.  It validates the
v2 payload, aligned provenance, and every referenced world receipt before it
selects the short-horizon decisive population.  Output is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
V2 = ARTIFACTS / "t4-prep-20261015/bank-v2"
OUT = ARTIFACTS / "redesign-t11-mac-20261017/bank-local30"
LENGTH_BUCKETS = ("lt400", "400_699", "700_999", "ge1000")


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON receipt atomically after all validation succeeds."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write one compressed numerical payload atomically."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def length_bucket(length: np.ndarray) -> np.ndarray:
    """Return the registered LOCAL30 length bucket for each row."""
    length = np.asarray(length)
    result = np.full(length.shape, "ge1000", dtype="U8")
    result[length < 1000] = "700_999"
    result[length < 700] = "400_699"
    result[length < 400] = "lt400"
    return result


def deciles(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Assign ties to the upper registered score decile, numbered one through ten."""
    cuts = np.asarray(cuts, dtype=np.float64)
    if cuts.shape != (9,) or not np.isfinite(cuts).all() or np.any(np.diff(cuts) < 0):
        raise ValueError("decile cutpoints must be nine finite nondecreasing values")
    return np.searchsorted(cuts, np.asarray(score, dtype=np.float64), side="right") + 1


def _strata(length: np.ndarray, q: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Build joint length-bucket and negative-raw-Q-decile strata."""
    return np.char.add(
        np.char.add(length_bucket(length), ":D"),
        deciles(-np.asarray(q).max(axis=1), cuts).astype("U2"),
    )


def _bound_source(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate exact v2 shard inventory and return its declared digest binding."""
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get("shards")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("source manifest must declare nonempty shards")
    present = {path.name for path in source.glob("*.npz")}
    if set(declared) != present:
        raise ValueError("source NPZ inventory differs from manifest")
    binding = {"manifest.json": sha256(manifest_path)}
    for name, expected in declared.items():
        path = source / name
        if Path(name).name != name or not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"source shard digest mismatch: {name}")
        binding[name] = expected
    return manifest, binding


def _source_rows(
    source: Path, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate aligned v2 provenance and every original world receipt hash."""
    source_info = manifest.get("source")
    if not isinstance(source_info, dict) or not isinstance(source_info.get("files"), dict):
        raise ValueError("source manifest lacks original provenance file binding")
    original = Path(source_info.get("path", ""))
    source_files: Mapping[str, str] = source_info["files"]
    provenance = original / "bank-00000.provenance.jsonl"
    expected = source_files.get(provenance.name)
    if not isinstance(expected, str) or sha256(provenance) != expected:
        raise ValueError("original provenance hash mismatch")
    rows = [json.loads(line) for line in provenance.read_text().splitlines() if line]
    n = len(arrays["action"])
    if len(rows) != n or sorted(row.get("row") for row in rows) != list(range(n)):
        raise ValueError("original provenance row indices must be contiguous and aligned")
    rows.sort(key=lambda row: int(row["row"]))
    receipts: dict[str, str] = {str(provenance): expected}
    deaths: dict[str, int] = {}
    for key in sorted({str(row["source_key"]) for row in rows}):
        relative = f"worlds/{key}.json"
        receipt_path = original / relative
        expected = source_files.get(relative)
        if not isinstance(expected, str) or sha256(receipt_path) != expected:
            raise ValueError(f"original world receipt hash mismatch: {key}")
        receipt = json.loads(receipt_path.read_text())
        death = receipt.get("death_decision")
        if death is not None and (not isinstance(death, int) or death < 0):
            raise ValueError(f"invalid original death decision: {key}")
        deaths[key] = -1 if death is None else death
        receipts[str(receipt_path)] = expected
    action = np.asarray(arrays["action"])
    original_surviving = np.asarray(arrays["surviving_original"])
    decisive = ~original_surviving[np.arange(n), action]
    if not np.array_equal(decisive, np.asarray([row.get("decisive") for row in rows], dtype=bool)):
        raise ValueError("original survival side disagrees with aligned provenance")
    for row in rows:
        row["death_decision"] = deaths[str(row["source_key"])]
    return rows, receipts


def derive(source: Path = V2, out: Path = OUT) -> dict[str, Any]:
    """Create the hash-bound LOCAL30 bank, failing closed on counter shortages."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("derived output must be empty and outside its immutable source")
    manifest, source_binding = _bound_source(source)
    shard_names = sorted(manifest["shards"])
    if shard_names != ["bank-00000.npz"]:
        raise ValueError("LOCAL30 derivation currently requires the registered single v2 shard")
    with np.load(source / shard_names[0], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {
        "scalars",
        "mask",
        "surviving",
        "surviving_original",
        "action",
        "weight",
        "q",
        "length",
        "packed",
    }
    if not required <= set(arrays):
        raise ValueError("v2 source payload lacks required label-bank arrays")
    if arrays["q"].shape != (len(arrays["action"]), 6) or not np.isfinite(arrays["q"]).all():
        raise ValueError("v2 source raw Q must be finite shape (n, 6)")
    rows, receipt_binding = _source_rows(source, manifest, arrays)
    source_index = np.arange(len(arrays["action"]), dtype=np.int64)
    action = arrays["action"].astype(np.int64, copy=False)
    counter = arrays["surviving_original"][source_index, action]
    decision = np.asarray([int(row["decision"]) for row in rows], dtype=np.int64)
    death = np.asarray([int(row["death_decision"]) for row in rows], dtype=np.int64)
    gap = death - decision
    decisive = ~counter
    selected_decisive = decisive & (gap >= 0) & (gap <= 30)
    if not selected_decisive.any():
        raise ValueError("LOCAL30 selection produced no decisive rows")
    cuts = np.quantile(-arrays["q"][selected_decisive].max(axis=1), np.arange(1, 10) / 10)
    all_strata = _strata(arrays["length"], arrays["q"], cuts)
    original_decisive_counts = Counter(all_strata[decisive].tolist())
    original_counter_counts = Counter(all_strata[counter].tolist())
    selected_indices = np.flatnonzero(selected_decisive)
    # Both lists are source-index ordered; pairing is stable and without replacement.
    pairs: list[dict[str, int | str]] = []
    selected_counter: list[int] = []
    source_counts = Counter(all_strata[selected_indices].tolist())
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    shortages: dict[str, dict[str, int]] = {}
    for stratum in sorted(source_counts):
        decisive_rows = selected_indices[all_strata[selected_indices] == stratum]
        counter_rows = np.flatnonzero(counter & (all_strata == stratum))
        available_counts[stratum] = int(len(counter_rows))
        selected_counts[stratum] = int(len(decisive_rows))
        if len(counter_rows) < len(decisive_rows):
            shortages[stratum] = {
                "decisive": int(len(decisive_rows)),
                "counter": int(len(counter_rows)),
            }
            continue
        for decisive_row, counter_row in zip(
            decisive_rows, counter_rows[: len(decisive_rows)], strict=True
        ):
            pairs.append(
                {
                    "stratum": str(stratum),
                    "decisive_source_row": int(decisive_row),
                    "counter_source_row": int(counter_row),
                }
            )
            selected_counter.append(int(counter_row))
    if shortages:
        raise ValueError(
            f"LOCAL30 counter support shortage: {json.dumps(shortages, sort_keys=True)}"
        )
    ordered_decisive = np.asarray(sorted(selected_indices.tolist()), dtype=np.int64)
    ordered_counter = np.asarray(sorted(selected_counter), dtype=np.int64)
    chosen = np.concatenate((ordered_decisive, ordered_counter))
    family = np.concatenate(
        (
            np.zeros(len(ordered_decisive), dtype=np.int8),
            np.ones(len(ordered_counter), dtype=np.int8),
        )
    )
    result = {key: value[chosen].copy() for key, value in arrays.items()}
    result["source_row"] = chosen.astype(np.int64, copy=False)
    result["decision"] = decision[chosen]
    result["death_decision"] = death[chosen]
    result["row_family"] = family
    result["row_weight"] = np.ones(len(chosen), dtype=np.float32)
    if not np.array_equal(
        ~result["surviving_original"][np.arange(len(chosen)), result["action"]], family == 0
    ):
        raise RuntimeError("derived family identity does not preserve original survival side")
    if not np.array_equal(result["source_row"], chosen):
        raise RuntimeError("derived source row ordering changed")
    if source_binding != _bound_source(source)[1]:
        raise RuntimeError("v2 source changed during derivation")
    out.mkdir(parents=True, exist_ok=False)
    payload = out / "bank-00000.npz"
    _atomic_npz(payload, **result)
    receipt = {
        "schema": "t11-local30-derived-label-bank/v1",
        "rows": int(len(chosen)),
        "family_counts": {
            "decisive": int(len(ordered_decisive)),
            "counter": int(len(ordered_counter)),
        },
        "original_family_counts": {"decisive": int(decisive.sum()), "counter": int(counter.sum())},
        "source": {"path": str(source), "files": source_binding},
        "original_provenance": {"receipt_hashes": receipt_binding},
        "selection": {
            "side_definition": "decisive iff not surviving_original[source_row, action_B8]",
            "decisive_gap": "0 <= death_decision - decision <= 30 inclusive",
            "decile_score": "negative maximum of all six stored raw B8 Q values",
            "decile_cutpoints": cuts.tolist(),
            "ties": "upper decile via searchsorted(side=right)",
            "length_buckets": list(LENGTH_BUCKETS),
            "matching": "stable source-row order within joint stratum; no replacement",
            "source_counts": dict(sorted(source_counts.items())),
            "original_decisive_counts": dict(sorted(original_decisive_counts.items())),
            "original_counter_counts": dict(sorted(original_counter_counts.items())),
            "counter_available_counts": dict(sorted(available_counts.items())),
            "selected_counts": dict(sorted(selected_counts.items())),
            "shortages": shortages,
            "selected_decisive_source_rows": ordered_decisive.tolist(),
            "selected_counter_source_rows": ordered_counter.tolist(),
            "matched_pairs": pairs,
        },
        "payload_contract": (
            "source arrays preserved exactly; source_row, decision, death_decision, "
            "row_family, and row_weight added"
        ),
        "shards": {payload.name: sha256(payload)},
    }
    _atomic_json(out / "manifest.json", receipt)
    return receipt


def main() -> None:
    """Derive the registered T11 bank from command line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=V2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
