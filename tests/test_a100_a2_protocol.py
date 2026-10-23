"""A2 outcome-blind resume must not silently recompute or mix allocations."""

import json

import numpy as np
import pytest

from research.a100_a2.protocol import (
    NAMESPACE,
    allocation,
    build_ledger,
    empty_ledger,
    publish_fit,
    read_ledger,
)
from research.bl_ppo.exports import export_json


def test_full_factorial_and_atomic_skip(tmp_path):
    assert len(allocation()) == 480
    assert len({tuple(u.values()) for u in allocation()}) == 480
    out = tmp_path / "attempt"
    export_json(out / "allocation.json", {"namespace": NAMESPACE, "allocation": "abc"})
    publish_fit(out, 2, {"p_d300": np.zeros((1, 6))}, NAMESPACE, "abc")
    ledger = tmp_path / "ledger.json"
    build_ledger([out], ledger, NAMESPACE, "abc")
    assert set(read_ledger(ledger, NAMESPACE, "abc")) == {2}
    with pytest.raises(ValueError, match="foreign"):
        read_ledger(ledger, "other", "abc")
    with pytest.raises(FileExistsError):
        publish_fit(out, 2, {}, NAMESPACE, "abc")
    with pytest.raises(ValueError, match="duplicate"):
        build_ledger([out, out], tmp_path / "dup", NAMESPACE, "abc")


def test_foreign_and_partial_refused(tmp_path):
    export_json(tmp_path / "allocation.json", {"namespace": NAMESPACE, "allocation": "wrong"})
    with pytest.raises(ValueError, match="foreign"):
        build_ledger([tmp_path], tmp_path / "ledger", NAMESPACE, "abc")
    ledger = tmp_path / "empty.json"
    ledger.write_text(json.dumps(empty_ledger(NAMESPACE, "abc")))
    assert read_ledger(ledger, NAMESPACE, "abc") == {}
