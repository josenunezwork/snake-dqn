from __future__ import annotations

import json
import sys
import types

import pytest

from research.bl_labels import exports
from research.bl_labels.manifests import canary_allocation, make_manifest, namespace


def test_completion_last_ledger_refuses_foreign_and_duplicate(tmp_path):
    allocation = [{"allocation_id": "watch6-000", "seed": 0}]
    ns = "f0-test"
    unit = {
        "schema": exports.SCHEMA,
        "namespace": ns,
        "allocation": allocation[0],
        "status": "COMPLETE",
        "rows": [],
        "snapshots": [],
        "counters": {},
    }
    exports.publish_unit(tmp_path, ns, allocation, unit)
    ledger = tmp_path / "portable" / "verified-ledger.json"
    built = exports.build_ledger(tmp_path, ledger, ns, allocation)
    assert built["units"][0]["unit_id"] == "watch6-000"
    assert exports.load_ledger(ledger, ns, allocation) == {"watch6-000"}
    bad = json.loads(ledger.read_text())
    bad["units"].append(bad["units"][0])
    with pytest.raises(FileExistsError, match="immutable F0 export differs"):
        exports.publish_json(ledger, bad)


def test_canary_manifest_freezes_counts_and_effective_cap(tmp_path):
    b8, frp3, control, ledger = (tmp_path / name for name in ("b8", "frp3", "control", "ledger"))
    for path in (b8, frp3, control):
        path.write_text("x")
    commit = "a" * 40
    exports.publish_json(
        ledger, exports.empty_ledger(namespace(commit), canary_allocation(namespace(commit)))
    )
    (tmp_path / "SEED_AUDIT.json").write_text("{}")
    manifest = make_manifest(commit, tmp_path / "out", control, ledger, b8, frp3)
    assert len(canary_allocation()) == 36
    assert manifest["requirements"]["gpu"] == "cpu"
    assert manifest["budget"]["unit_cap"] == 0.2
    assert manifest["budget"]["rate_units_per_hour"] == 0.08
    assert manifest["f0"]["allocation"]["watch6_decisions"] == 14400
    assert manifest["resume"]["max_outcome_blind_retries"] == 2


def test_attempt_aggregator_unions_lives_and_refuses_duplicate(tmp_path):
    allocation = [{"allocation_id": "watch6-000", "seed": 1}]
    ns = "f0-union"
    first, second = tmp_path / "first", tmp_path / "second"
    for root, life in ((first, 0), (second, 1)):
        exports.publish_life(
            root,
            ns,
            allocation,
            {
                "schema": "bl-labels-f0-life/v1",
                "namespace": ns,
                "allocation_id": "watch6-000",
                "allocation": allocation[0],
                "life": life,
                "death_decision": 10,
                "rows": [],
            },
        )
    ledger = tmp_path / "union.json"
    result = exports.aggregate_attempts([first, second], ledger, ns, allocation)
    assert [row["unit_id"] for row in result["units"]] == [
        "watch6-000--life-000000",
        "watch6-000--life-000001",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        exports.aggregate_attempts([first, first], tmp_path / "duplicate.json", ns, allocation)


def test_invalid_attempt_receipt_invalidates_its_leaves(tmp_path):
    allocation, ns, attempt = (
        [{"allocation_id": "watch6-000", "seed": 1}],
        "f0-invalid",
        tmp_path / "attempt",
    )
    exports.publish_life(
        attempt,
        ns,
        allocation,
        {
            "schema": "bl-labels-f0-life/v1",
            "namespace": ns,
            "allocation_id": "watch6-000",
            "allocation": allocation[0],
            "life": 0,
            "death_decision": 1,
            "rows": [],
        },
    )
    exports.publish_json(attempt / "runtime-result.json", {"status": "INVALID"})
    with pytest.raises(ValueError, match="INVALID F0 attempt"):
        exports.aggregate_attempts([attempt], tmp_path / "union.json", ns, allocation)


def test_mac_analysis_prioritizes_incomplete_and_insufficient_before_rate():
    incomplete = {
        "mac_aggregate": {"status": "INCOMPLETE", "summary": {"rescue_bearing_lives": 99}}
    }
    assert exports.analyze_aggregate(incomplete, 1.0)["verdict"] == "INCOMPLETE"
    insufficient = {
        "mac_aggregate": {"status": "INSUFFICIENT", "summary": {"rescue_bearing_lives": 0}}
    }
    assert exports.analyze_aggregate(insufficient, 1.0)["verdict"] == "INSUFFICIENT"
    collected = {"mac_aggregate": {"status": "COLLECTED", "summary": {"rescue_bearing_lives": 7}}}
    assert exports.analyze_aggregate(collected, 1.0)["verdict"] == "KILL"


def test_runtime_value_error_publishes_terminal_invalid_marker(monkeypatch, tmp_path):
    from research.bl_labels import runtime

    class FakeInvalid(RuntimeError):
        pass

    fake = types.SimpleNamespace(
        FactoryInvalid=FakeInvalid,
        collect_unit=lambda *_args: (_ for _ in ()).throw(ValueError("bad restore")),
    )
    monkeypatch.setitem(sys.modules, "research.bl_labels.factory", fake)
    allocation = [{"allocation_id": "watch6-000", "seed": 1, "stream": "watch6"}]
    value, error = runtime._collect(
        (
            allocation[0],
            tmp_path / "b8",
            tmp_path / "frp",
            "f0-invalid",
            tmp_path,
            allocation,
            set(),
            None,
        )
    )
    assert error == "bad restore" and value["status"] == "INVALID"
    marker = tmp_path / "INVALID-watch6-000.json"
    assert marker.is_file() and json.loads(marker.read_text())["namespace"] == "f0-invalid"
