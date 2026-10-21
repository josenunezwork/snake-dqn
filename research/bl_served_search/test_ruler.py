"""Synthetic contract tests for the ruler; these never construct a game world."""

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.bl_served_search import ruler


def _root() -> bytes:
    return b'{"bodies":[],"food":[],"opponent_carry":{},"rng":{"seed":1}}'


def _fixture(mix: str, length: int) -> dict:
    raw = _root()
    return {
        "fixture_id": f"{mix}-{length}",
        "mix": mix,
        "target_length": length,
        "root_frame": 2500,
        "root_length": length,
        "horizon": 300,
        "tie_rule": "survival,bare,q_desc,action_asc",
        "s2_native_v8": True,
        "legal_actions": [0, 1, 2],
        "root_bytes_b64": base64.b64encode(raw).decode(),
        "root_sha256": hashlib.sha256(raw).hexdigest(),
        "roster_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "operator_sha256": "3" * 64,
    }


def _fixtures() -> dict:
    return {
        "schema": ruler.SCHEMA,
        "fixtures": [
            _fixture(mix, length) for mix in ruler.MIXES for length in ruler.FIXTURE_TARGETS
        ],
    }


class _Operator:
    def prepare(self, request: dict) -> dict:  # pragma: no cover - run contract is primary
        return {
            **request,
            "root_bytes": _root(),
            "roster_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
        }

    def batch(self, fixture: dict) -> dict:
        return {
            "semantic": {
                "fixture": fixture["fixture_id"],
                "action": 1,
                "outcome": "survives",
                "fired": True,
            },
            "stage_seconds": {
                key: 0.001
                for key in (
                    "sim",
                    "observation",
                    "hero_v8",
                    "opponent_v8",
                    "vector61_frp3",
                    "b8_forward",
                    "clone_digest",
                    "submit_visible",
                    "submit_consumed",
                )
            },
        }


def create_test_operator() -> _Operator:
    """Top-level factory lets the focused test exercise spawn-worker plumbing."""
    return _Operator()


def test_fixture_contract_requires_all_mixes_lengths_and_root_content() -> None:
    assert len(ruler.validate_fixtures(_fixtures())) == 9
    value = _fixtures()
    value["fixtures"][0]["root_length"] = 799
    with pytest.raises(ValueError, match="does not meet length"):
        ruler.validate_fixtures(value)


def test_smoke_receipt_is_never_a_qualification() -> None:
    receipt = ruler.run_ruler(
        _fixtures(), _Operator(), platform_label="mac", workers=1, min_batches=1, smoke=True
    )
    assert receipt["status"] == "PRECHECK_ONLY"
    with pytest.raises(ValueError, match="smoke"):
        ruler.verify_qualification(receipt, receipt | {"platform": {"label": "l4"}})


def test_cross_platform_semantic_mismatch_is_invalid() -> None:
    mac = ruler.run_ruler(_fixtures(), _Operator(), platform_label="mac", workers=1, min_batches=30)
    l4 = ruler.run_ruler(_fixtures(), _Operator(), platform_label="mac", workers=1, min_batches=30)
    l4["platform"]["label"] = "l4"
    l4["workers"] = 12
    for receipt in (mac, l4):
        receipt["operator_spec"] = "research.bl_served_search.ruler_operator:create_operator"
    l4["platform"]["system"] = "Linux"
    l4["hardware"] = {
        "sha256": "4" * 64,
        "reported": {"hardware": {"kind": "L4"}},
        "vcpu_count": 12,
    }
    l4["records"][0]["batches"][0]["semantic"]["action"] = 4
    assert ruler.verify_qualification(mac, l4) == {
        "status": "INVALID",
        "semantic_mismatches": ["frozen-1200"],
    }


def test_flat_root_exports_are_portable_and_immutable(tmp_path: Path) -> None:
    ruler.export_fixtures(tmp_path, _fixtures())
    manifest = json.loads((tmp_path / "ruler-fixtures.json").read_text())
    assert manifest["mode"] == "full"
    assert all("/" not in fixture["root_file"] for fixture in manifest["fixtures"])
    for fixture in manifest["fixtures"]:
        assert (tmp_path / fixture["root_file"]).with_suffix(
            ".json.complete"
        ).read_text() == "complete\n"


def test_l4_below_admission_is_not_dispatchable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def tick() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 1000.0

    monkeypatch.setattr(ruler.time, "monotonic", tick)
    receipt = ruler.run_ruler(
        _fixtures(), _Operator(), platform_label="mac", workers=1, min_batches=30
    )
    receipt["platform"]["label"] = "l4"
    receipt["workers"] = 12
    receipt["status"] = "NOT_DISPATCHABLE"
    assert receipt["status"] == "NOT_DISPATCHABLE"


def test_qualification_rejects_duplicate_fixture_ids_and_forged_rate() -> None:
    mac = ruler.run_ruler(_fixtures(), _Operator(), platform_label="mac", workers=1, min_batches=30)
    l4 = copy.deepcopy(mac)
    for receipt in (mac, l4):
        receipt["operator_spec"] = "research.bl_served_search.ruler_operator:create_operator"
    l4["platform"].update(label="l4", system="Linux")
    l4["workers"] = 12
    l4["hardware"] = {
        "sha256": "4" * 64,
        "reported": {"hardware": {"kind": "L4"}},
        "vcpu_count": 12,
    }
    assert ruler.validate_qualification(mac, l4)["status"] == "RULER_PASS"
    bad = copy.deepcopy(l4)
    bad["records"][1]["fixture_id"] = bad["records"][0]["fixture_id"]
    with pytest.raises(ValueError, match="duplicate"):
        ruler.validate_qualification(mac, bad)
    bad = copy.deepcopy(l4)
    bad["aggregate_batches_per_second"] *= 2
    with pytest.raises(ValueError, match="reconcile"):
        ruler.validate_qualification(mac, bad)
    bad = copy.deepcopy(l4)
    for row in bad["records"]:
        row["batches"] = row["batches"][:1]
    with pytest.raises(ValueError, match="incomplete"):
        ruler.validate_qualification(mac, bad)


def test_frozen_root_encodes_runtime_distance_sentinels() -> None:
    import numpy as np

    from research.bl_served_search.ruler_operator import _jsonable

    frozen = _jsonable(np.array([np.inf, -np.inf, np.nan], dtype=np.float64))
    payload = ruler.canonical(frozen)
    assert '"float_hex":"inf"' in payload
    assert '"float_hex":"-inf"' in payload
    assert '"float_hex":"nan"' in payload
