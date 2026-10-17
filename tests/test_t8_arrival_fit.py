"""The T8 classifier join must preserve canonical state/world identity."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from research.t8_probes_20261017.arrival_fit import align_arrival  # noqa: E402


def test_arrival_join_reorders_by_identity_and_rejects_missing_or_changed_world(tmp_path) -> None:
    data = {
        "state_id": np.array(["s0", "s1"]),
        "world": np.array(["w0", "w1"]),
        "split": np.array(["train", "holdout"]),
    }
    features = np.arange(36, dtype=np.float32).reshape(2, 18)
    path = tmp_path / "features.npz"
    np.savez(
        path,
        state_id=data["state_id"][::-1],
        world=data["world"][::-1],
        split=data["split"][::-1],
        features=features[::-1],
    )
    np.testing.assert_array_equal(align_arrival(data, path), features)
    np.savez(
        path,
        state_id=data["state_id"],
        world=np.array(["w1", "w0"]),
        split=data["split"],
        features=features,
    )
    with pytest.raises(ValueError, match="world"):
        align_arrival(data, path)
    np.savez(
        path,
        state_id=np.array(["s0", "s0"]),
        world=data["world"],
        split=data["split"],
        features=features,
    )
    with pytest.raises(ValueError, match="identities"):
        align_arrival(data, path)
