"""Merge disjoint T8 restoration partitions with canonical T5 row identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.t5b_risk_and_fuse_20261016.fuse import T5_DATA, load_t5_data, sha256
from research.t8_probes_20261017.arrival_fit import align_arrival


def merge(parts: list[Path], out: Path) -> dict:
    """Validate partition provenance and coverage before publishing merged features."""
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    receipts, arrays = [], []
    for path in parts:
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt["output_sha256"] != sha256(path):
            raise ValueError(f"partition hash mismatch: {path}")
        receipts.append(receipt)
        with np.load(path, allow_pickle=False) as z:
            arrays.append({key: z[key].copy() for key in z.files})
    count = len(parts)
    if {r["shard"]["index"] for r in receipts} != set(range(count)):
        raise ValueError("missing or duplicate partition index")
    for receipt in receipts:
        if receipt["shard"]["count"] != count:
            raise ValueError("partition count mismatch")
        for key in ("source_and_input_sha256", "checkpoint_sha256", "canonical_t5_data_sha256"):
            if receipt[key] != receipts[0][key]:
                raise ValueError(f"partition provenance differs: {key}")
    combined = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    ids = combined["state_id"]
    data = load_t5_data(T5_DATA)
    if len(set(ids.tolist())) != len(ids) or set(ids) != set(data["state_id"]):
        raise ValueError("partitions do not exactly cover the canonical bank")
    lookup = {str(value): index for index, value in enumerate(ids)}
    order = np.array([lookup[str(value)] for value in data["state_id"]])
    combined = {key: value[order] for key, value in combined.items()}
    for key in ("world", "split"):
        np.testing.assert_array_equal(combined[key], data[key])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **combined)
    align_arrival(data, out)
    result = {
        "authority": "diagnostic only",
        "states": len(ids),
        "worlds": len(np.unique(combined["world"])),
        "byte_verified_states": len(ids),
        "parts": receipts,
        "output_sha256": sha256(out),
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts, args.out), indent=2))


if __name__ == "__main__":
    main()
