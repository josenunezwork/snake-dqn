"""Composed offline CPU guards; original desktop identity tests remain binding."""

from copy import deepcopy

import pytest

from research.bl_served_search import combined_policy as combined
from research.bl_served_search import host_load
from tests.test_served_background_report import _fixture_desktop_registry_replay  # noqa: F401
from tests.test_served_background_report import (
    _desktop_host,
    _desktop_sample,
)
from tests.test_served_remote_supervision import fixture as supervised_fixture

SECOND = 1_000_000_000


def fixture():
    registry, journal, cpu, observations, points, intervals = supervised_fixture()
    for sample in intervals:
        for row in sample["processes"]:
            row["kind"] = host_load.process_kind(row["command"])
            row["classification"] = {"kind": row["kind"]}
        desktop = _desktop_sample(sample["start_ns"], sample["end_ns"], app=25)
        sample["processes"].extend(desktop["processes"])
        combined._totals(sample)
    for point in points:
        for row in point["processes"]:
            row["kind"] = host_load.process_kind(row["command"])
            row["classification"] = {"kind": row["kind"]}
        point["processes"].extend(
            _desktop_sample(point["time_ns"], point["time_ns"] + SECOND, app=25)["processes"]
        )
    host = _desktop_host(intervals)
    host.update(
        schema=combined.SCHEMA,
        load_policy=combined.POLICY,
        supervisor_registry=registry,
        supervision_journal=journal,
        supervision_cpu_evidence=cpu,
        supervision_identity_observations=observations,
        supervision_point_screens=points,
        observations=[
            {
                "process_screen": {
                    "observation_ns": point["time_ns"],
                    "classified_processes": deepcopy(point["processes"]),
                    "probe_exempt_pids": [],
                }
            }
            for point in points
        ],
    )
    return host


def test_both_verified_classes_compose_without_mutating_raw_evidence():
    host = fixture()
    before = deepcopy(host)
    result = combined.replay(host)
    assert result["valid"], result["errors"]
    assert result["status"] == "OFFLINE_VALID"
    assert result["policy"] == combined.POLICY
    assert result["live_capture_capability"] == "NOT_ADMISSIBLE"
    assert result["observed_cpu_s"] == 0.1
    assert all(row["background_cpu_percent"] == 25 for row in result["ordinary_cpu_samples"])
    assert host == before


@pytest.mark.parametrize("field", ["kind", "classification", "pid", "command"])
def test_desktop_and_point_inventory_tampering_refused(field):
    host = fixture()
    host["supervision_point_screens"][0]["processes"][-1][field] = "foreign"
    result = combined.replay(host)
    assert not result["valid"]


def test_ordinary_twenty_percent_rule_survives_both_exemptions():
    host = fixture()
    row = {
        "pid": 991,
        "ppid": 1,
        "command": "/tmp/trainer",
        "cpu_percent": 20,
        "kind": "unknown",
        "classification": {"kind": "unknown"},
    }
    host["cpu_samples"][8]["processes"].append(row)
    combined._totals(host["cpu_samples"][8])
    result = combined.replay(host)
    assert not result["valid"]
    assert ">=20%" in str(result["errors"])


def test_shared_background_ceiling_includes_desktop_cpu():
    host = fixture()
    for sample in host["cpu_samples"]:
        for row in sample["processes"]:
            if row["kind"] == "desktop_app":
                row["cpu_percent"] = 151
        combined._totals(sample)
    result = combined.replay(host)
    assert not result["valid"]
    assert ">150%" in str(result["errors"])


def test_original_and_combined_policies_never_reinterpret_each_other():
    host = fixture()
    host["load_policy"] = "classified-desktop-v1"
    assert not combined.replay(host)["valid"]


@pytest.mark.parametrize("field", ["supervision_cpu_evidence", "supervision_journal"])
def test_combined_cannot_supply_missing_accounting_with_desktop_identity(field):
    host = fixture()
    del host[field]
    assert not combined.replay(host)["valid"]


def test_reports_keep_all_ticks_and_shared_desktop_background():
    from tests.test_served_supervision_report import _row

    host = fixture()
    proof = combined.replay(host)
    result = {
        "trace": [_row(0, 2 * SECOND, 2 * SECOND + 10), _row(1, 61 * SECOND, 61 * SECOND + 10)],
        "warmup_ticks": 0,
    }
    report = combined.summarize(result, host, proof)
    assert report["load_policy"] == combined.POLICY
    assert report["counts"]["ticks"] == 2
    assert report["counts"]["poll_overlap"] == 1
    # Authenticated desktop CPU is 25%, exceeding the <=20% sensitivity bar.
    assert report["counts"]["low_background"] == 0
    assert report["timing"]["overall"]["count"] == 2
    assert report["timing"]["low_background"]["count"] == 0
    assert len(report["desktop"]["ticks"]) == 2


def test_combined_report_rejects_changed_summary():
    host = fixture()
    proof = combined.replay(host)
    proof["observed_cpu_s"] += 1
    with pytest.raises(ValueError, match="exact independent replay"):
        combined.summarize({}, host, proof)


def bound_raw(monkeypatch):
    from research.bl_served_search import desktop_policy
    from research.bl_served_search import supervision_io as io

    host = fixture()
    registry = host["supervisor_registry"]
    registry.update(
        ruling_sha256=io.RULING_SHA256,
        ruling_hash=io.RULING_SHA256,
        policy_source_hash=io.digest(io.policy_sources()),
    )
    for event in [*host["supervision_journal"], *host["supervision_identity_observations"]]:
        if "policy_source_hash" in event:
            event["policy_source_hash"] = registry["policy_source_hash"]
    host["supervision_replay"] = combined.replay(host)
    assert host["supervision_replay"]["valid"]
    source, manifest = {"commit": "a" * 40}, {"fixture": True}
    supervision = {
        "load_policy": io.POLICY,
        "registry_sha256": io.digest(registry),
        "policy_source_sha256": io.policy_sources(),
        "ruling_sha256": io.RULING_SHA256,
    }
    raw = {
        "load_policy": combined.POLICY,
        "host": host,
        "source_before": source,
        "source_after": deepcopy(source),
        "manifest": manifest,
        "desktop_intent": {"policy": desktop_policy.POLICY},
        "supervision_intent": supervision,
    }
    raw["combined_intent"] = {
        "schema": "served-combined-intent/v1",
        "load_policy": combined.POLICY,
        "components": combined.COMPONENTS,
        "source": source,
        "manifest_sha256": io.digest(manifest),
        "desktop_intent_sha256": io.digest(raw["desktop_intent"]),
        "supervision_intent_sha256": io.digest(supervision),
    }
    # Accepted desktop intent semantics have their own exhaustive tests. This
    # fixture isolates composition bindings and cannot bypass the final gate.
    monkeypatch.setattr(desktop_policy, "validate_intent", lambda _: None)
    return raw


def test_even_bound_synthetic_combined_evidence_cannot_admit_live_panel(monkeypatch):
    with pytest.raises(ValueError, match="admission unavailable"):
        combined.replay_host(bound_raw(monkeypatch))


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "components",
        "manifest_sha256",
        "desktop_intent_sha256",
        "supervision_intent_sha256",
    ],
)
def test_combined_intent_binding_mismatches_refused(monkeypatch, field):
    raw = bound_raw(monkeypatch)
    raw["combined_intent"][field] = "foreign"
    with pytest.raises(ValueError, match="combined intent"):
        combined.replay_host(raw)


def test_combined_reported_cpu_must_equal_independent_replay(monkeypatch):
    raw = bound_raw(monkeypatch)
    raw["host"]["supervision_replay"]["observed_cpu_s"] += 1
    with pytest.raises(ValueError, match="reported accounting"):
        combined.replay_host(raw)
