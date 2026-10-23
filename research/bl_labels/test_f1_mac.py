"""Mac equivalence must reject partial, reordered or semantically different cells."""

from __future__ import annotations

import copy
import hashlib

import pytest

from research.bl_labels.exports import publish_json
from research.bl_labels.f1_ruler import compare_mac_configs
from research.bl_labels.f1_units import empty_ledger, load_ledger


def cells() -> tuple[list[dict], list[str]]:
    ids = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(64)]
    configs = []
    for label, workers in zip(("one", "half", "all"), (1, 3, 6)):
        configs.append(
            {
                "status": "COMPLETE",
                "platform": "mac",
                "transport_probe": False,
                "schedule_label": label,
                "workers": workers,
                "root_count": 64,
                "roots_per_hour": 1000 * workers,
                "records": [
                    {
                        "root_id": root,
                        "digest": "a" * 64,
                        "label_sha256": "b" * 64,
                        "state_digest": "c" * 64,
                        "bare_audit_mismatches": 0,
                    }
                    for root in ids
                ],
            }
        )
    return configs, ids


def test_mac_invariance_requires_all_three_native_cells() -> None:
    configs, ids = cells()
    result = compare_mac_configs(configs, ids)
    assert result["status"] == "PASS"
    assert result["workers"] == [1, 3, 6]
    assert result["mac_w6_to_l4_ratio"] == 6000 / 6194
    with pytest.raises(ValueError, match="three complete"):
        compare_mac_configs(configs[:2], ids)


@pytest.mark.parametrize("field", ["digest", "label_sha256", "state_digest"])
def test_mac_invariance_refuses_any_root_semantic_difference(field: str) -> None:
    configs, ids = cells()
    configs[2]["records"][63][field] = "d" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_mac_configs(configs, ids)


@pytest.mark.parametrize("change", ["order", "missing", "probe", "workers", "audit", "hash"])
def test_mac_invariance_cannot_pass_malformed_cell(change: str) -> None:
    configs, ids = cells()
    if change == "order":
        configs[1]["records"].reverse()
    elif change == "missing":
        configs[1]["records"].pop()
    elif change == "probe":
        configs[1]["transport_probe"] = True
    elif change == "workers":
        configs[1]["workers"] = 2
    elif change == "audit":
        configs[1]["records"][0]["bare_audit_mismatches"] = 1
    else:
        del configs[1]["records"][0]["label_sha256"]
    with pytest.raises(ValueError, match="INVALID"):
        compare_mac_configs(configs, ids)


def test_mac_ledger_cannot_consume_x86_units(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    value = empty_ledger("a" * 64, "mac", source_commit="frozen-pin")
    publish_json(path, value)
    assert load_ledger(path, "a" * 64, "mac", "frozen-pin") == {}
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(path, "a" * 64, "cpu", "frozen-pin")
    mutated = copy.deepcopy(value)
    mutated["source_commit"] = "other-pin"
    other = tmp_path / "foreign.json"
    publish_json(other, mutated)
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(other, "a" * 64, "mac", "frozen-pin")


@pytest.mark.parametrize("rate", [None, True, 0, -1, float("nan"), float("inf")])
def test_mac_invariance_requires_measured_finite_rate(rate) -> None:
    configs, ids = cells()
    configs[0]["roots_per_hour"] = rate
    with pytest.raises(ValueError, match="unmeasured rate"):
        compare_mac_configs(configs, ids)
