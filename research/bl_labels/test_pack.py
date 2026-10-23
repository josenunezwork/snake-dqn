from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from research.bl_labels import exports, manifests
from research.bl_labels.manifests import (
    canary_allocation,
    fresh_seed_audit,
    make_manifest,
    namespace,
)


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
    assert manifest["budget"]["unit_cap"] == 0.18
    assert manifest["budget"]["rate_units_per_hour"] == 0.08
    assert manifest["f0"]["allocation"]["watch6_decisions"] == 14400
    assert manifest["resume"]["max_outcome_blind_retries"] == 2
    probe = manifests.make_family_probe(manifest, tmp_path / "probe-out")
    assert probe["budget"]["unit_cap"] == 0.02
    assert probe["timeout_minutes"] == 5
    assert probe["inputs"] == manifest["inputs"]
    assert "resume" not in probe and probe["resume_exemption"]["kind"] == "short-canary"


def test_fresh_seed_audit_uses_literal_whole_word_rg_and_refuses_hits(monkeypatch):
    calls = []

    def clean(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr("research.bl_labels.manifests.subprocess.run", clean)
    commit = "c" * 40
    audit = fresh_seed_audit(commit, [123, 456])
    assert audit["hits_by_root"] and not any(audit["hits_by_root"].values())
    assert len(calls) == 2
    assert calls[0][0][:6] == ["rg", "-I", "-l", "-F", "-w", "--no-config"]
    assert calls[0][1]["timeout"] == 600

    monkeypatch.setattr(
        "research.bl_labels.manifests.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "/tmp/hit\n", ""),
    )
    with pytest.raises(ValueError, match="fresh seed collision"):
        fresh_seed_audit(commit, [123])


def test_fresh_seed_audit_real_whole_word_collision_and_scan_error(monkeypatch, tmp_path):
    repo, artifacts = tmp_path / "repo", tmp_path / "snake-dqn-artifacts"
    repo.mkdir()
    artifacts.mkdir()
    monkeypatch.setattr(manifests, "ROOT", repo)
    (repo / "adjacent.txt").write_text("x123y\n")
    clean = manifests.fresh_seed_audit("d" * 40, [123])
    assert not any(clean["hits_by_root"].values())
    (artifacts / "standalone.txt").write_text("value 123, here\n")
    with pytest.raises(ValueError, match="fresh seed collision"):
        manifests.fresh_seed_audit("d" * 40, [123])

    monkeypatch.setattr(
        manifests.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, "", "permission denied"),
    )
    with pytest.raises(RuntimeError, match="fresh seed scan failed"):
        manifests.fresh_seed_audit("d" * 40, [456])


def test_family_receipt_binding_requires_qualified_check(monkeypatch, tmp_path):
    template = tmp_path / "template.json"
    receipt = tmp_path / "receipt.json"
    exports.publish_json(template, {"canary": {"template": True}, "commit": "a" * 40})
    receipt.write_text("receipt")
    seen = []
    monkeypatch.setattr(
        "research.colab_cli.canary.check", lambda value, repo: seen.append((value, repo))
    )
    bound = manifests.bind_family_receipt(template, receipt, tmp_path / "bound.json")
    assert seen and bound["canary"]["receipt"]["sha256"] == manifests._sha(receipt)


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
    # A resume ledger may preserve partial lives, but cannot take the one
    # scientific look until every allocation has published its session.
    assert result["mac_aggregate"]["status"] == "INCOMPLETE"
    assert result["mac_aggregate"]["summary"] is None
    assert result["mac_aggregate"]["derived_effective_label"] is None
    with pytest.raises(ValueError, match="duplicate"):
        exports.aggregate_attempts([first, first], tmp_path / "duplicate.json", ns, allocation)


def test_session_requires_exact_death_life_dependencies_for_resume_and_aggregate(tmp_path):
    allocation, ns, attempt = ([{"allocation_id": "watch6-000", "seed": 1}], "f0-deps", tmp_path)
    exports.publish_session(
        attempt,
        ns,
        allocation,
        {
            "schema": "bl-labels-f0-session/v1",
            "namespace": ns,
            "allocation_id": "watch6-000",
            "allocation": allocation[0],
            "counters": {"deaths": 1},
        },
    )
    ledger = tmp_path / "portable" / "verified-ledger.json"
    built = exports.build_ledger(attempt, ledger, ns, allocation)
    assert [row["unit_id"] for row in built["units"]] == ["watch6-000--session"]
    names = exports.load_ledger(ledger, ns, allocation)
    assert exports.completed_session_parents(ledger, names) == set()
    aggregate = exports.aggregate_attempts([attempt], tmp_path / "union.json", ns, allocation)
    assert aggregate["mac_aggregate"]["status"] == "INCOMPLETE"
    assert aggregate["mac_aggregate"]["summary"] is None

    session = attempt / "lives" / "watch6-000--session.json"
    malformed = json.loads(session.read_text())
    malformed["counters"] = {"deaths": "one"}
    malformed_root = tmp_path / "malformed"
    exports.publish_session(malformed_root, ns, allocation, malformed)
    exports.build_ledger(malformed_root, tmp_path / "bad" / "verified-ledger.json", ns, allocation)
    with pytest.raises(ValueError, match="invalid deaths metadata"):
        exports.load_ledger(tmp_path / "bad" / "verified-ledger.json", ns, allocation)


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
    incomplete = {"mac_aggregate": {"status": "INCOMPLETE", "summary": None}}
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
            False,
            None,
        )
    )
    assert error == "bad restore" and value["status"] == "INVALID"
    marker = tmp_path / "INVALID-watch6-000.json"
    assert marker.is_file() and json.loads(marker.read_text())["namespace"] == "f0-invalid"
