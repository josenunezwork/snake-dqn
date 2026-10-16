"""Independent real-array consistency checks for the immutable T4c bank parts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from research.t4c_20261016 import data

import numpy as np


def audit(root: Path = data.ROOT, source: Path = data.V2) -> dict:
    """Check original-array preservation, exact row quotas and canonical identities."""
    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

    v3, outside = root / "bank-v3", root / "outside-bank"
    original = data.load_arrays(source / "bank-00000.npz")
    derived = data.load_arrays(v3 / "bank-00000.npz")
    for key, value in original.items():
        if key not in derived or not np.array_equal(value, derived[key]):
            raise AssertionError(f"original v2 array changed: {key}")
    n = len(derived["action"])
    decisive = ~derived["surviving_original"][np.arange(n), derived["action"]]
    gap = derived["death_decision"] - derived["decision"]
    expected = np.ones(n, np.float32)
    expected[decisive & (gap >= 100)] = np.float32(0.3)
    assert ((gap[decisive] >= 0) & (gap[decisive] < 300)).all()
    np.testing.assert_array_equal(derived["row_weight"], expected)
    np.testing.assert_array_equal(derived["row_family"], (~decisive).astype(np.int8))
    assert (derived["weight"] == 1).all()
    cuts, quotas, bank_fingerprints = data.target_definition(source)
    arrays = data.load_arrays(outside / "bank-00000.npz")
    count = len(arrays["action"])
    assert count == int(decisive.sum()) == 10885
    assert (arrays["length"] >= 700).all() and (arrays["decision"] <= 4699).all()
    death_gap = arrays["death_decision"] - arrays["decision"]
    assert ((arrays["death_decision"] < 0) | (death_gap >= 300)).all()
    assert np.isfinite(arrays["q"]).all() and np.isfinite(arrays["scalars"]).all()
    assert (arrays["weight"] == 1).all() and (arrays["row_weight"] == 0.5).all()
    assert (arrays["row_family"] == 2).all()
    labels = np.zeros((count, 6), bool)
    labels[np.arange(count), arrays["action"]] = True
    np.testing.assert_array_equal(arrays["surviving"], labels)
    np.testing.assert_array_equal(arrays["surviving_original"], labels)
    action = np.where(arrays["mask"], arrays["q"].astype(np.float64), -np.inf).argmax(1)
    np.testing.assert_array_equal(arrays["action"], action)
    assigned = data.strata(arrays["length"], arrays["q"], cuts)
    assert dict(Counter(assigned.tolist())) == quotas
    layout = train_layout()
    local, glob = unpack_ref(arrays["packed"], layout)
    fingerprints = data.observation_fingerprints(local, glob, arrays["scalars"])
    np.testing.assert_array_equal(fingerprints, arrays["fingerprint"])
    assert len(set(fingerprints.tolist())) == count
    assert not (set(fingerprints.tolist()) & bank_fingerprints)
    assert set(arrays["source_owner"].tolist()) == {"B8", "CTRL"}
    files = [
        directory / name
        for directory in (source, v3, outside)
        for name in ("manifest.json", "bank-00000.npz")
    ]
    result = {
        "schema": "t4c-data-array-audit/v1",
        "pass": True,
        "inputs_sha256": {str(path): data.sha256(path) for path in files},
        "source_sha256": data.sha256(Path(__file__)),
        "preserved_v2_members": sorted(original),
        "primary_rows": n,
        "decisive_rows": int(decisive.sum()),
        "in_window_counter_rows": int((~decisive).sum()),
        "decisive_weight_one_rows": int(np.sum(decisive & (expected == 1))),
        "decisive_weight_point_three_rows": int(np.sum(decisive & (expected == np.float32(0.3)))),
        "outside_rows": count,
        "outside_unique_canonical_observations": count,
        "outside_bank_observation_overlap": 0,
        "outside_stratum_counts": quotas,
        "outside_source_counts": dict(Counter(arrays["source_owner"].tolist())),
        "outside_minimum_length": int(arrays["length"].min()),
        "outside_minimum_observed_death_gap": int(death_gap[arrays["death_decision"] >= 0].min()),
        "outside_action_contract": (
            "B8 advisory argmax preservation label; no counterfactual survival claim"
        ),
    }
    data._atomic_json(root / "data-quality-audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
