"""Independent raw-array audit of the published T4c banks; no model or training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")


def digest(path: Path) -> str:
    """Hash a file in bounded chunks."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def arrays(directory: Path) -> dict:
    """Verify the manifest's shard before reading its arrays."""
    manifest = json.loads((directory / "manifest.json").read_text())
    shard = directory / "bank-00000.npz"
    assert digest(shard) == manifest["shards"][shard.name]
    with np.load(shard, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def main() -> None:
    """Independently recompute the required counts and row-level invariants."""
    original_dir = ARTIFACTS / "t4-prep-20261015/bank-v2"
    derived_dir = ARTIFACTS / "t4c-data-20261016/bank-v3"
    outside_dir = ARTIFACTS / "t4c-data-20261016/outside-bank"
    original, derived, outside = map(arrays, (original_dir, derived_dir, outside_dir))
    for name in original:
        assert np.array_equal(original[name], derived[name]), name
    decisive = ~original["surviving_original"][
        np.arange(len(original["action"])), original["action"]
    ]
    gap = derived["death_decision"] - derived["decision"]
    assert (gap[decisive] >= 0).all() and (gap[decisive] < 300).all()
    expected_weight = np.where(decisive & (gap >= 100), 0.3, 1).astype(np.float32)
    assert np.array_equal(expected_weight, derived["weight"] * derived["row_weight"])
    assert np.array_equal(derived["row_family"], (~decisive).astype(np.int8))
    n = len(outside["action"])
    assert n == int(decisive.sum()) == 10885
    assert np.all(outside["row_family"] == 2)
    assert np.all(outside["weight"] * outside["row_weight"] == np.float32(0.5))
    assert np.all(outside["length"] >= 700)
    assert np.all(outside["decision"] <= 4699)
    died = outside["death_decision"] >= 0
    distances = (outside["death_decision"] - outside["decision"])[died]
    assert np.all(distances >= 300)
    labels = np.where(outside["mask"], outside["q"].astype(np.float64), -np.inf).argmax(1)
    assert np.array_equal(labels, outside["action"])
    assert np.all(outside["surviving"].sum(1) == 1)
    assert np.all(outside["surviving"][np.arange(n), labels])
    assert len(np.unique(outside["fingerprint"])) == n
    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref
    from research.t4a_posthoc_20261016.spillover import observation_fingerprints

    old_local, old_global = unpack_ref(original["packed"], train_layout())
    old_fp = observation_fingerprints(old_local, old_global, original["scalars"])
    del old_local, old_global
    new_local, new_global = unpack_ref(outside["packed"], train_layout())
    new_fp = observation_fingerprints(new_local, new_global, outside["scalars"])
    del new_local, new_global
    assert np.array_equal(new_fp, outside["fingerprint"])
    assert not np.intersect1d(old_fp, new_fp).size
    cuts = np.quantile(-original["q"][decisive].max(1), np.arange(1, 10) / 10)

    def counts(length: np.ndarray, q: np.ndarray) -> Counter:
        buckets = np.where(length < 1000, "700_999", "ge1000")
        bins = np.searchsorted(cuts, -q.max(1), side="right") + 1
        return Counter(f"{bucket}:D{decile}" for bucket, decile in zip(buckets, bins))

    target = counts(original["length"][decisive], original["q"][decisive])
    selected = counts(outside["length"], outside["q"])
    assert target == selected and len(target) == 20
    manifest = json.loads((outside_dir / "manifest.json").read_text())
    assert manifest["matching"]["target_counts"] == dict(target)
    assert manifest["matching"]["selected_counts"] == dict(selected)
    owners = dict(Counter(outside["source_owner"].tolist()))
    assert set(owners) == {"B8", "CTRL"}
    assert owners == manifest["source_counts"]
    result = {
        "schema": "t4c-independent-data-audit/v1",
        "pass": True,
        "decisive_rows": int(decisive.sum()),
        "in_window_counter_rows": int((~decisive).sum()),
        "decisive_full_weight": int(np.sum(decisive & (gap < 100))),
        "decisive_reduced_weight": int(np.sum(decisive & (gap >= 100))),
        "outside_rows": n,
        "source_counts": owners,
        "matched_strata": dict(selected),
        "minimum_death_distance": int(distances.min()),
        "bindings": {
            str(directory / "manifest.json"): digest(directory / "manifest.json")
            for directory in (original_dir, derived_dir, outside_dir)
        },
        "checks": [
            "all original v2 arrays unchanged",
            "distance weights and families",
            "exact 1:1 stratum counts without duplicate fingerprints",
            "outside length and death/censor distance",
            "recomputed fingerprints and zero v2 bank overlap",
            "B8 advisory singleton labels",
            "outside weight .5",
            "both trajectory owners represented",
            "shard hashes",
        ],
    }
    destination = (
        Path(__file__).resolve().parents[2] / "docs/research/t4c_data_audit_2026-10-16.json"
    )
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
