"""Composed offline CPU guards; original desktop identity tests remain binding."""

from copy import deepcopy

import pytest

from research.bl_served_search import background_report
from research.bl_served_search import combined_policy as combined
from research.bl_served_search import host_load
from tests.test_served_desktop_apps import enrolled as _desktop_enrolled
from tests.test_served_remote_supervision import fixture as supervised_fixture

SECOND = 1_000_000_000


@pytest.fixture
def enrolled(tmp_path):
    """Use the real registry/export fixture without patching replay validators."""
    return _desktop_enrolled.__wrapped__(tmp_path)


def fixture(enrolled):
    desktop_registry, app_row, *_ = enrolled
    decision = desktop_registry.admit_process(app_row)
    assert decision["kind"] == "desktop_app"
    desktop_evidence = desktop_registry.read_export()
    # Synthetic capture clock starts at zero; the admission happened before it.
    for entry in desktop_evidence["history_log"]:
        entry["observation_ns"] = 0

    def desktop_rows(percent):
        return [
            {
                **app_row,
                "ppid": 1,
                "cpu_percent": percent,
                "kind": "desktop_app",
                "classification": decision,
            }
        ]

    registry, journal, cpu, observations, points, intervals = supervised_fixture()
    for sample in intervals:
        for row in sample["processes"]:
            row["kind"] = host_load.process_kind(row["command"])
            row["classification"] = background_report._replay_kind(row, desktop_evidence)
        sample["processes"].extend(desktop_rows(25))
        combined._totals(sample)
    for point in points:
        for row in point["processes"]:
            row["kind"] = host_load.process_kind(row["command"])
            row["classification"] = background_report._replay_kind(row, desktop_evidence)
        point["processes"].extend(desktop_rows(25))
    host = {
        "cpu_samples": intervals,
        "desktop_registry": desktop_evidence["registry"],
        "desktop_registry_sha256": desktop_evidence["registry_sha256"],
        "desktop_registry_evidence": desktop_evidence,
    }
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


def test_both_verified_classes_compose_without_mutating_raw_evidence(enrolled):
    host = fixture(enrolled)
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
def test_desktop_and_point_inventory_tampering_refused(field, enrolled):
    host = fixture(enrolled)
    host["supervision_point_screens"][0]["processes"][-1][field] = "foreign"
    result = combined.replay(host)
    assert not result["valid"]


def test_ordinary_twenty_percent_rule_survives_both_exemptions(enrolled):
    host = fixture(enrolled)
    row = {
        "pid": 991,
        "ppid": 1,
        "command": "/tmp/trainer",
        "cpu_percent": 20,
        "kind": "unknown",
        "classification": {"kind": "unknown"},
    }
    del row["classification"]
    row["classification"] = background_report._replay_kind(row, host["desktop_registry_evidence"])
    host["cpu_samples"][8]["processes"].append(row)
    combined._totals(host["cpu_samples"][8])
    result = combined.replay(host)
    assert not result["valid"]
    assert ">=20%" in str(result["errors"])


def test_shared_background_ceiling_includes_desktop_cpu(enrolled):
    host = fixture(enrolled)
    for sample in host["cpu_samples"]:
        for row in sample["processes"]:
            if row["kind"] == "desktop_app":
                row["cpu_percent"] = 151
        combined._totals(sample)
    result = combined.replay(host)
    assert not result["valid"]
    assert ">150%" in str(result["errors"])


def test_original_and_combined_policies_never_reinterpret_each_other(enrolled):
    host = fixture(enrolled)
    host["load_policy"] = "classified-desktop-v1"
    assert not combined.replay(host)["valid"]


@pytest.mark.parametrize("field", ["supervision_cpu_evidence", "supervision_journal"])
def test_combined_cannot_supply_missing_accounting_with_desktop_identity(field, enrolled):
    host = fixture(enrolled)
    del host[field]
    assert not combined.replay(host)["valid"]


def test_reports_keep_all_ticks_and_shared_desktop_background(enrolled):
    from tests.test_served_supervision_report import _row

    host = fixture(enrolled)
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
    assert report["desktop"]["inventory"] == proof["desktop_summary"]
    assert "timing" not in report["desktop"]


def test_combined_report_rejects_changed_summary(enrolled):
    host = fixture(enrolled)
    proof = combined.replay(host)
    proof["observed_cpu_s"] += 1
    with pytest.raises(ValueError, match="exact independent replay"):
        combined.summarize({}, host, proof)


def bound_raw(enrolled):
    from research.bl_served_search import desktop_policy
    from research.bl_served_search import supervision_io as io

    host = fixture(enrolled)
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
    from pathlib import Path

    from research.bl_paced_identity.record import export_json
    from research.bl_served_search.desktop_apps import DesktopRegistry
    from tests.test_served_tick_probe import _manifest

    # Seal the real fixture registry as the panel's immutable input. Keep its
    # admitted desktop rows and full verified history in the v5 host payload.
    registry_path = enrolled[0].path
    payload = enrolled[0].registry
    registry_path.unlink()
    export_json(registry_path, payload)
    enrollment = DesktopRegistry(registry_path).read_export()
    host["desktop_registry_evidence"]["registry_inputs"] = enrollment["registry_inputs"]

    # Offset the complete synthetic clock to permit strict pre-observation publication.
    def offset(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key
                    in {
                        "start_ns",
                        "end_ns",
                        "time_ns",
                        "mono_ns",
                        "observation_ns",
                        "capture_start_ns",
                        "capture_end_ns",
                        "exposure_start_ns",
                        "launch_ns",
                        "exit_ns",
                        "launch_start_ns",
                        "launch_end_ns",
                    }
                    and type(item) is int
                ):
                    value[key] += 10 * SECOND
                else:
                    offset(item)
        elif isinstance(value, list):
            for item in value:
                offset(item)

    del host["supervision_replay"]
    offset(host)
    host["status"] = "COMPLETE"
    host["supervision_replay"] = combined.replay(host)
    assert host["supervision_replay"]["valid"], host["supervision_replay"]["errors"]
    root = str(Path(combined.__file__).resolve().parents[2])
    manifest = _manifest()
    source = {"commit": "a" * 40, "tracked_dirty": "", "worktree": root, "source_sha256": {}}
    for path in [desktop_policy.RULING, desktop_policy.AMENDMENT]:
        source["source_sha256"][str(path.relative_to(root))] = io.file_digest(path)
    desktop_intent = desktop_policy.freeze_intent(registry_path, source, manifest)
    desktop_intent["published_ns"] = 1

    supervision = {
        "load_policy": io.POLICY,
        "registry_sha256": io.digest(registry),
        "policy_source_sha256": io.policy_sources(),
        "ruling_sha256": io.RULING_SHA256,
        "published_ns": 1,
    }
    raw = {
        "load_policy": combined.POLICY,
        "host": host,
        "source_before": source,
        "source_after": deepcopy(source),
        "manifest": manifest,
        "desktop_intent": desktop_intent,
        "qualification_receipt": {"published_ns": 2, "fixture": "offline-only-600s"},
        "supervision_intent": supervision,
    }
    raw["combined_intent"] = {
        "schema": "served-combined-intent/v1",
        "published_ns": 3,
        "qualification_receipt_sha256": io.digest(raw["qualification_receipt"]),
        "load_policy": combined.POLICY,
        "components": combined.COMPONENTS,
        "source": source,
        "manifest_sha256": io.digest(manifest),
        "desktop_intent_sha256": io.digest(raw["desktop_intent"]),
        "supervision_intent_sha256": io.digest(supervision),
    }
    return raw


def test_even_bound_synthetic_combined_evidence_cannot_admit_live_panel(enrolled):
    with pytest.raises(ValueError, match="admission unavailable"):
        combined.replay_host(bound_raw(enrolled))


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "components",
        "manifest_sha256",
        "desktop_intent_sha256",
        "supervision_intent_sha256",
        "qualification_receipt_sha256",
        "published_ns",
    ],
)
def test_combined_intent_binding_mismatches_refused(field, enrolled):
    raw = bound_raw(enrolled)
    raw["combined_intent"][field] = "foreign"
    with pytest.raises(ValueError, match="combined intent"):
        combined.replay_host(raw)


def test_combined_reported_cpu_must_equal_independent_replay(enrolled):
    raw = bound_raw(enrolled)
    raw["host"]["supervision_replay"]["observed_cpu_s"] += 1
    with pytest.raises(ValueError, match="reported accounting"):
        combined.replay_host(raw)


def test_reduce_trace_combined_branch_reaches_closed_gate_unmocked(enrolled):
    from research.bl_served_search.served_tick_probe import reduce_trace

    raw = bound_raw(enrolled)
    raw["cells"] = [{"cell": cell} for cell in raw["manifest"]["order"]]
    report = reduce_trace(raw)
    assert report["verdict"] == "INVALID"
    assert report["reason"].startswith(
        "supervision evidence: remote_supervision admission unavailable"
    )


@pytest.mark.parametrize(
    "owner", ["desktop_intent", "supervision_intent", "combined_intent", "qualification_receipt"]
)
@pytest.mark.parametrize("published", [None, True, -1, 10 * SECOND])
def test_combined_publication_must_precede_first_host_read(enrolled, owner, published):
    from research.bl_served_search import supervision_io as io

    raw = bound_raw(enrolled)
    raw[owner]["published_ns"] = published
    # Keep content hashes synchronized to isolate the publication ordering guard.
    for field, target in [
        ("desktop_intent_sha256", "desktop_intent"),
        ("supervision_intent_sha256", "supervision_intent"),
        ("qualification_receipt_sha256", "qualification_receipt"),
    ]:
        raw["combined_intent"][field] = io.digest(raw[target])
    with pytest.raises(ValueError, match="publication"):
        combined.replay_host(raw)


def test_combined_replay_validates_registry_schema(enrolled):
    host = fixture(enrolled)
    host["supervisor_registry"]["roots"][0]["repo"] = "relative"
    proof = combined.replay(host)
    assert not proof["valid"]
    assert "invalid supervision registry" in str(proof["errors"])


def test_synchronized_unregistered_desktop_forgery_is_refused(enrolled):
    host = fixture(enrolled)
    for rows in [
        *[s["processes"] for s in host["cpu_samples"]],
        *[p["processes"] for p in host["supervision_point_screens"]],
        *[o["process_screen"]["classified_processes"] for o in host["observations"]],
    ]:
        for row in rows:
            if row["kind"] == "desktop_app":
                row["command"] = "/Applications/Unregistered.app/Contents/MacOS/Unregistered"
    assert host["supervision_point_screens"] == combined._points(host)
    proof = combined.replay(host)
    assert not proof["valid"]
    assert "inventories differ" not in str(proof["errors"])
    assert "desktop" in str(proof["errors"])


def test_verified_app_kind_and_category_survive_projection(enrolled):
    host = fixture(enrolled)
    proof = combined.replay(host)
    assert proof["valid"], proof["errors"]
    for sample in proof["ordinary_cpu_samples"]:
        assert sample["background_cpu_percent"] == 25
        assert sample["desktop_app_cpu_percent"] == 25
        assert sample["desktop_renderer_cpu_percent"] == 0
        assert [row["kind"] for row in sample["processes"]] == ["desktop_app"]


@pytest.mark.parametrize("at", [5, 61])
def test_desktop_report_cannot_expose_a_conflicting_clean_timing_view(enrolled, at):
    from tests.test_served_supervision_report import _row

    host = fixture(enrolled)
    for sample in host["cpu_samples"]:
        sample["processes"][-1]["cpu_percent"] = 12
        combined._totals(sample)
    # A root's measured 12% CPU remains below the supervision guard. Together
    # with the desktop's 12%, it must exclude even an operation-free tick.
    root_id = host["supervision_cpu_evidence"]["roots"][0]
    for index, sample in enumerate(host["supervision_cpu_evidence"]["samples"]):
        sample["cpu_ns"][root_id] = 120_000_000 if index > at else 0
    for sample in host["cpu_samples"]:
        row = sample["processes"][0]
        amount = 120_000_000 if sample["start_ns"] == at * SECOND else 0
        row.update(cpu_percent=100 * amount / SECOND, cpu_delta_numerator=amount)
        combined._totals(sample)
    proof = combined.replay(host)
    assert proof["valid"], proof["errors"]
    report = combined.summarize(
        {"trace": [_row(0, at * SECOND, at * SECOND + 10)], "warmup_ticks": 0}, host, proof
    )
    assert report["counts"]["low_background"] == 0
    assert report["timing"]["low_background"]["count"] == 0
    assert "timing" not in report["desktop"]
    assert "counts" not in report["desktop"]


def test_operation_overlap_excludes_otherwise_low_desktop_tick(enrolled):
    from tests.test_served_supervision_report import _row

    host = fixture(enrolled)
    for sample in host["cpu_samples"]:
        sample["processes"][-1]["cpu_percent"] = 1
        combined._totals(sample)
    proof = combined.replay(host)
    assert proof["valid"], proof["errors"]
    report = combined.summarize(
        {"trace": [_row(0, 2 * SECOND, 2 * SECOND + 10)], "warmup_ticks": 0}, host, proof
    )
    assert report["counts"]["operation_overlap"] == 1
    assert report["counts"]["low_background"] == 0
    assert "timing" not in report["desktop"]


@pytest.mark.parametrize(
    "field", ["registry_sha256", "manifest_sha256", "ruling_sha256", "registry_input", "source"]
)
def test_combined_desktop_component_bindings_are_replayed_unmocked(enrolled, field):
    from research.bl_served_search import supervision_io as io

    raw = bound_raw(enrolled)
    raw["desktop_intent"][field] = {} if field in {"registry_input", "source"} else "f" * 64
    raw["combined_intent"]["desktop_intent_sha256"] = io.digest(raw["desktop_intent"])
    with pytest.raises(ValueError, match="desktop"):
        combined.replay_host(raw)


def test_qualification_receipt_must_be_published_before_combined_intent(enrolled):
    from research.bl_served_search import supervision_io as io

    raw = bound_raw(enrolled)
    raw["qualification_receipt"]["published_ns"] = 4
    raw["combined_intent"]["qualification_receipt_sha256"] = io.digest(raw["qualification_receipt"])
    with pytest.raises(ValueError, match="publication"):
        combined.replay_host(raw)


def test_combined_publication_precedes_capture_start_not_only_midpoint(enrolled):
    raw = bound_raw(enrolled)
    raw["host"]["observations"][0]["process_screen"]["capture_start_ns"] = 2
    with pytest.raises(ValueError, match="publication"):
        combined.replay_host(raw)
