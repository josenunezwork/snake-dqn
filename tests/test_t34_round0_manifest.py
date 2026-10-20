"""Focused sizing rules for the bounded T34 Round 0 manifest builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_round0 import make_manifest as manifests
from research.t34_round0.make_manifest import (
    EXPORT_BYTES_PER_WORLD,
    EXPORT_SIZING_FACTOR,
    MAX_ESTIMATE_MINUTES,
    MAX_EXPORT_BYTES,
    ManifestRefused,
    maximum_chunk_pairs,
    size_chunk,
    size_round,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _manifest_inputs(tmp_path: Path, commit: str) -> dict[str, Path]:
    registration = tmp_path / "PREREGISTRATION_ROUND0.md"
    registration.write_text("draft registration bytes\n", encoding="utf-8")
    rows = []
    for mix, count in (("scripted", 96), ("frozen", 72), ("mixed", 72)):
        rows.extend(
            {
                "world_id": f"{mix}-{number}",
                "world_seed": len(rows) + number,
                "mix": mix,
                "opponents": ["b8"],
            }
            for number in range(count)
        )
    bank = _write(tmp_path / "BANK_ROUND0.json", {"schema": "t34-round0-bank/v1", "rows": rows})
    audit = _write(tmp_path / "SEED_AUDIT_ROUND0.json", {"schema": "test"})
    a5 = _write(
        tmp_path / "a5.json",
        {
            "schema": "t34-round0-a5/v1",
            "teacher_worlds_per_hour": 54.4,
            "bare_seconds_per_world": 9.0,
            "metric_seconds_per_pair": 4.0,
            "cpu_slowdown_factor": 3.0,
            "a6_measurement_sha256": "b" * 64,
            "a6_source_sha256": "c" * 64,
            "steady_window_seconds": 1200,
            "cold_fill_excluded": True,
            "active_row_steps_per_second": 3000,
            "two_lane_rehearsal_complete": True,
            "measured_export_bytes_per_pair": 1000,
            "evidence_sha256": "a" * 64,
        },
    )
    study_root = tmp_path / "study"
    ratification = _write(
        tmp_path / "RATIFICATION.json",
        {
            "schema": "t34-round0-ratification/v1",
            "status": "RATIFIED",
            "commit": commit,
            "operator": "operator",
            "approved_at": "2026-10-20T00:00:00Z",
            "registration_sha256": _sha(registration),
            "bank_sha256": _sha(bank),
            "audit_sha256": _sha(audit),
            "a5": {"path": "/operator/a5.json", "sha256": _sha(a5)},
            "source_hashes": {"research/t34_round0/runner.py": "d" * 64},
            "study_root": str(study_root),
            "sizing": size_round(
                240,
                teacher_worlds_per_hour=54.4,
                bare_seconds_per_world=9.0,
                metric_seconds_per_pair=4.0,
                cpu_slowdown_factor=3.0,
            ),
        },
    )
    return {
        "registration": registration,
        "bank": bank,
        "audit": audit,
        "a5": a5,
        "ratification": ratification,
        "public": _write(tmp_path / "PUBLIC_INPUTS.json", {"schema": "test"}),
        "study_root": study_root,
    }


def test_manifest_uses_fixed_attempt_path_and_recomputed_chunk_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    files = _manifest_inputs(tmp_path, commit)
    ratification_value = json.loads(files["ratification"].read_text())
    ratification_value["source_hashes"]["research/t34_exit_b8/PUBLIC_INPUTS.json"] = _sha(
        files["public"]
    )
    _write(files["ratification"], ratification_value)
    monkeypatch.setattr("research.t34_round0.sizing.verify_adapter", lambda *_args: None)
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})
    monkeypatch.setattr(manifests, "require_registration", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(manifests, "validate_supervisor", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(manifests, "qualify_source", lambda *_args, **_kwargs: None)

    result = manifests.make_manifests(
        a5_receipt=files["a5"],
        ratification=files["ratification"],
        output_root=files["study_root"],
        out_dir=files["study_root"] / "manifests",
        commit=commit,
        lanes=2,
        other_assignments={"existing": 2.0},
        setup_minutes=0,
        registration=files["registration"],
        bank=files["bank"],
        audit=files["audit"],
        public_inputs=files["public"],
    )

    first, second = result["manifests"]
    assert [(item["round0"]["start"], item["round0"]["count"]) for item in (first, second)] == [
        (0, 155),
        (155, 85),
    ]
    assert first["output_dir"] == str(files["study_root"] / "chunks/chunk-000/attempt-01/job")
    assert first["round0"]["layout"] == {
        "cpu_inference_processes": 12,
        "threads_per_process": 1,
        "worlds_per_process": 1,
    }
    assert first["env"]["T34_ROUND0_THREADS"] == "1"
    assert first["requirements"]["est_minutes"] <= 180
    assert first["timeout_minutes"] >= 2 * first["requirements"]["est_minutes"]
    assert first["budget"]["reserve_units"] == pytest.approx(first["budget"]["unit_cap"] * 0.20)
    assert first["max_export_bytes"] < MAX_EXPORT_BYTES
    assert (files["study_root"] / "manifests" / f"{first['job_id']}.json").is_file()
    import copy

    from research.t34_round0.dispatch import validate_manifest_closure
    from research.t34_round0.history import HistoryError

    validate_manifest_closure(first, ratification_value, files["ratification"])
    for field in ("script", "requirements", "env", "inputs", "source_paths"):
        altered = copy.deepcopy(first)
        altered[field] = {
            "script": "research/other.sh",
            "requirements": {},
            "env": {},
            "inputs": [],
            "source_paths": [],
        }[field]
        with pytest.raises(HistoryError, match="closure"):
            validate_manifest_closure(altered, ratification_value, files["ratification"])


def test_size_chunk_adds_all_same_runtime_component_costs() -> None:
    sizing = size_chunk(
        144,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )

    expected_workload = (144 * 3600 / 54.4 + 144 * 9.0 * 3 / 12 + 144 * 4.0 * 3 / 12) / 60.0
    assert sizing["teacher_seconds"] == pytest.approx(144 * 3600 / 54.4)
    assert sizing["measured_bare_seconds"] == pytest.approx(1296.0)
    assert sizing["bare_seconds"] == pytest.approx(324.0)
    assert sizing["metric_seconds"] == pytest.approx(144.0)
    assert sizing["workload_minutes"] == pytest.approx(expected_workload)
    assert sizing["estimate_minutes"] == 167
    assert sizing["timeout_minutes"] == 334
    assert sizing["unit_cap"] == pytest.approx(334 * 2 / 60)
    assert sizing["reserve_units"] == pytest.approx(334 * 2 / 60 * 0.20)


def test_maximum_chunk_pairs_enforces_time_and_strict_export_limit() -> None:
    maximum = maximum_chunk_pairs(
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )
    allowed = size_chunk(
        maximum,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )
    next_chunk = size_chunk(
        maximum + 1,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )

    assert allowed["estimate_minutes"] <= MAX_ESTIMATE_MINUTES
    assert allowed["export_bytes"] < MAX_EXPORT_BYTES
    assert (
        next_chunk["estimate_minutes"] > MAX_ESTIMATE_MINUTES
        or next_chunk["export_bytes"] >= MAX_EXPORT_BYTES
    )
    assert allowed["export_bytes"] == pytest.approx(
        maximum * EXPORT_BYTES_PER_WORLD * EXPORT_SIZING_FACTOR
    )


def test_size_round_partitions_the_240_world_bank_and_recomputes_each_cap() -> None:
    sizing = size_round(
        240,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
    )

    chunks = sizing["chunks"]
    assert [(chunk["start"], chunk["count"]) for chunk in chunks] == [(0, 155), (155, 85)]
    assert all(chunk["estimate_minutes"] <= MAX_ESTIMATE_MINUTES for chunk in chunks)
    assert all(chunk["timeout_minutes"] >= 2 * chunk["estimate_minutes"] for chunk in chunks)
    assert sizing["unit_cap"] == pytest.approx(sum(chunk["unit_cap"] for chunk in chunks))
    assert sizing["reserve_units"] == pytest.approx(sum(chunk["reserve_units"] for chunk in chunks))


@pytest.mark.parametrize(
    "pairs,teacher,bare,metric,slowdown",
    [
        (0, 55.0, 1.0, 1.0, 3.0),
        (1, 0.0, 1.0, 1.0, 3.0),
        (1, 55.0, 0.0, 1.0, 3.0),
        (1, 55.0, 1.0, float("inf"), 3.0),
    ],
)
def test_size_chunk_refuses_invalid_measured_inputs(
    pairs: int, teacher: float, bare: float, metric: float, slowdown: float
) -> None:
    with pytest.raises(ManifestRefused):
        size_chunk(
            pairs,
            teacher_worlds_per_hour=teacher,
            bare_seconds_per_world=bare,
            metric_seconds_per_pair=metric,
            cpu_slowdown_factor=slowdown,
        )


def test_nonzero_setup_is_charged_once_per_chunk() -> None:
    sizing = size_round(
        240,
        teacher_worlds_per_hour=54.4,
        bare_seconds_per_world=9.0,
        metric_seconds_per_pair=4.0,
        cpu_slowdown_factor=3.0,
        setup_minutes=15,
    )
    assert len(sizing["chunks"]) == 2
    assert sizing["chunk_pairs"] > 100
    assert all(c["workload_minutes"] <= 180 for c in sizing["chunks"])
    assert all(c["setup_minutes"] == 15 for c in sizing["chunks"])


def test_sizing_adapter_refuses_changed_derived_measurements(tmp_path: Path) -> None:
    from research.t34_round0.sizing import verify_adapter

    lanes = []
    for number in range(2):
        job = tmp_path / f"a5-{number}"
        (job / "exports").mkdir(parents=True)
        raw = {
            "worlds_per_hour": 54.4 + number,
            "active_row_steps_per_second": 377 + number,
            "export_bytes_per_world": 8882584.1,
        }
        benchmark = _write(job / "exports/benchmark.json", raw)
        receipt = _write(job / "receipt.json", {"status": "PASS"})
        lanes.append(
            {
                "path": str(benchmark),
                "sha256": _sha(benchmark),
                "receipt_sha256": _sha(receipt),
                **raw,
            }
        )
    costs = {
        "bare": {"rollout_elapsed_seconds": 1.0, "metric_elapsed_seconds": 0.2},
        "teacher": {"rollout_elapsed_seconds": 5.0, "metric_elapsed_seconds": 0.3},
    }
    pair_entries = []
    for number in range(2):
        pair = _write(tmp_path / f"pair-{number}.json", {"arms": costs})
        pair_entries.append({"path": pair.name, "sha256": _sha(pair)})
    chunk = _write(
        tmp_path / "chunk.json", {"pairs": pair_entries, "run_identity": {"source_hashes": {}}}
    )
    evidence = {
        "a5_lanes": lanes,
        "a6_verification": {"preflight": {"path": str(chunk)}},
        "a6_pair_costs": [costs, costs],
        "setup_minutes": 15,
    }
    adapter = {
        "evidence": evidence,
        "evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode()
        ).hexdigest(),
        "a6_measurement_sha256": _sha(chunk),
        "a6_source_sha256": hashlib.sha256(b"{}").hexdigest(),
        "teacher_worlds_per_hour": 54.4,
        "bare_seconds_per_world": 10.0,
        "metric_seconds_per_pair": 5.0,
        "cpu_slowdown_factor": 3.0,
        "setup_minutes": 15,
        "active_row_steps_per_second": 377,
        "measured_export_bytes_per_pair": 8882584.1,
    }
    verify_adapter(adapter)
    for field in (
        "teacher_worlds_per_hour",
        "bare_seconds_per_world",
        "metric_seconds_per_pair",
        "setup_minutes",
    ):
        with pytest.raises(ValueError, match="derived fields"):
            verify_adapter(dict(adapter, **{field: adapter[field] + 1}))
    with pytest.raises(ValueError, match="digest"):
        verify_adapter(dict(adapter, evidence_sha256="0" * 64))
