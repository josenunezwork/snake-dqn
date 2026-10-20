"""Focused tests for optional T34 unpaced Colab exports.

These fixtures intentionally exercise only the descriptive loader: the export
transport must not become another route into the registered Mac decision.
"""

import hashlib
import json

from research.t34_pathb import analyze_v2 as analysis


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_sidecars(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n")
    path.with_name(f"{path.name}.complete").write_text("complete\n")
    return digest


def _registration():
    terminal = {"seed": 17, "mix": "scripted", "roster": ["scripted"]}
    return {
        "source_commit": "a" * 40,
        "bank_sha256": _digest("bank"),
        "terminal_rows": [terminal],
        "product_rows": [],
    }


def _mac_record(world, steps=None):
    return {
        "kind": "terminal",
        "seed": 17,
        "arms": {
            "bare": {
                "bare_prefix_trace": {
                    "initial_sha256": _digest("initial"),
                    "steps": steps if steps is not None else [_digest("one"), _digest("two")],
                    "ended_by_intervention": False,
                }
            }
        },
        "world": world,
    }


def _metrics(trace):
    return {
        "mass_integral": 150.0,
        "survival_fraction": 1.0,
        "no_death": 1,
        "deaths": 0,
        "decisions": 5000,
        "deaths_per_1k": 0.0,
        "boost_decisions": 50,
        "boost_share": 0.01,
        "action_counts": [1650, 1650, 1650, 25, 15, 10],
        "one_action_share": 0.33,
        "unconditional_faults": [],
        "overruns": 0,
        "median_length": 100.0,
        "p95_length": 200.0,
        "deaths_length_ge_800": 0,
        "late_deaths": 0,
        "late_large_overlap": 0,
        "identity_status": "PASS",
        "parity_status": "PASS",
        "bare_prefix_trace": trace,
    }


def _write_shard(tmp_path, *, trace=None, payload_digest=None):
    registration = _registration()
    world = registration["terminal_rows"][0]
    payload = {
        "schema": analysis.UNPACED_WORLD_SCHEMA,
        "registration_sha256": _digest("registration"),
        "source_commit": registration["source_commit"],
        "bank_sha256": registration["bank_sha256"],
        "world": world,
        "kind": "terminal",
        "mix": "scripted",
        "seed": 17,
        "horizon": 5000,
        "arm": "unpaced",
        "metrics": _metrics(
            trace
            or {
                "initial_sha256": _digest("initial"),
                "steps": [_digest("one"), _digest("two")],
                "ended_by_intervention": False,
            }
        ),
    }
    world_path = tmp_path / "world.json"
    world_path.write_text(json.dumps(payload, sort_keys=True))
    world_digest = _write_sidecars(world_path)
    shard = {
        "schema": analysis.UNPACED_SHARD_SCHEMA,
        "registration_sha256": _digest("registration"),
        "source_commit": registration["source_commit"],
        "bank_sha256": registration["bank_sha256"],
        "status": "COMPLETE",
        "worlds": [
            {
                "path": "world.json",
                "sha256": payload_digest or world_digest,
            }
        ],
    }
    shard_path = tmp_path / "shard.json"
    shard_path.write_text(json.dumps(shard, sort_keys=True))
    _write_sidecars(shard_path)
    return registration, world, shard_path


def test_unpaced_export_hash_identity_and_matching_prefix_are_descriptive(tmp_path):
    registration, world, shard = _write_shard(tmp_path)
    report = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    assert report["role"] == "descriptive_only"
    assert report["status"] == "COMPLETE"
    assert report["summary"] == {"expected": 1, "valid": 1, "missing_or_invalid": 0}
    assert report["rows"][0]["bare_prefix_divergence"] == {
        "status": "MATCHED_SHARED_PREFIX",
        "first_divergence_tick": None,
        "shared_steps": 2,
        "mac_steps": 2,
        "colab_steps": 2,
        "mac_ended_by_intervention": False,
        "colab_ended_by_intervention": False,
    }


def test_unpaced_export_reports_first_shared_bare_prefix_divergence(tmp_path):
    trace = {
        "initial_sha256": _digest("initial"),
        "steps": [_digest("one"), _digest("colab-two")],
        "ended_by_intervention": True,
    }
    registration, world, shard = _write_shard(tmp_path, trace=trace)
    report = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    divergence = report["rows"][0]["bare_prefix_divergence"]
    assert divergence["status"] == "DIVERGED"
    assert divergence["first_divergence_tick"] == 2
    assert divergence["shared_steps"] == 1


def test_complete_export_calculates_descriptive_skill_and_product_summaries(tmp_path, monkeypatch):
    registration, world, shard = _write_shard(tmp_path)
    monkeypatch.setattr(
        analysis, "skill_summary", lambda rows, arm: {"arm": arm, "rows": len(rows)}
    )
    monkeypatch.setattr(
        analysis, "product_summary", lambda rows, arm: {"arm": arm, "rows": len(rows)}
    )
    report = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    assert report["statistics"] == {
        "skill": {"arm": "unpaced", "rows": 1},
        "product": {"arm": "unpaced", "rows": 1},
    }


def test_malformed_unpaced_metrics_are_missing_descriptive_evidence(tmp_path):
    registration, world, shard = _write_shard(tmp_path)
    payload_path = tmp_path / "world.json"
    payload = json.loads(payload_path.read_text())
    payload["metrics"]["action_counts"][0] += 1
    payload_path.write_text(json.dumps(payload, sort_keys=True))
    shard_payload = json.loads(shard.read_text())
    shard_payload["worlds"][0]["sha256"] = _write_sidecars(payload_path)
    shard.write_text(json.dumps(shard_payload, sort_keys=True))
    _write_sidecars(shard)

    report = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    assert report["status"] == "MISSING"
    assert report["errors"]
    assert report["missing_rows"] == [{"kind": "terminal", "seed": 17, "reason": "not_exported"}]


def test_duplicate_unpaced_rows_invalidate_that_descriptive_world(tmp_path):
    registration, world, shard = _write_shard(tmp_path)
    shard_payload = json.loads(shard.read_text())
    shard_payload["worlds"].append(dict(shard_payload["worlds"][0]))
    shard.write_text(json.dumps(shard_payload, sort_keys=True))
    _write_sidecars(shard)
    report = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    assert report["status"] == "MISSING"
    assert report["missing_rows"] == [
        {"kind": "terminal", "seed": 17, "reason": "duplicate_export"}
    ]


def test_missing_or_hash_failed_unpaced_exports_are_reported_not_raised(tmp_path):
    registration = _registration()
    world = registration["terminal_rows"][0]
    missing = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], []
    )
    assert missing["status"] == "MISSING"
    assert missing["missing_rows"] == [{"kind": "terminal", "seed": 17, "reason": "not_exported"}]

    _, _, shard = _write_shard(tmp_path, payload_digest=_digest("wrong digest"))
    invalid = analysis.load_unpaced_descriptive(
        registration, _digest("registration"), [_mac_record(world)], [shard]
    )
    assert invalid["status"] == "MISSING"
    assert invalid["errors"]
    assert invalid["missing_rows"] == [{"kind": "terminal", "seed": 17, "reason": "not_exported"}]
