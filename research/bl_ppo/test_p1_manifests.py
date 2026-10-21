"""P1 admission and upload constraints fail closed before a paid assignment."""

from pathlib import Path

import pytest

from research.bl_ppo.p1_inputs import validate_catalog
from research.bl_ppo.p1_manifests import QUAL_PAUSE, TARGET, measured_plan, provisional_plan


def qualified() -> dict:
    return {
        "schema": "bl-p1-complete-job-qualification/v1",
        "status": "PASS",
        "critic_qualified": True,
        "seconds_per_actor_transition": 1 / 1440,
        "per_job_overhead_seconds": 600,
        "prefix_seconds": 5000,
        "transfer_and_release_verified": True,
    }


def test_chunks_cover_exact_fixed_endpoint_with_bounded_time_and_budget() -> None:
    plan = provisional_plan()
    assert plan["status"] == "FORECAST_ONLY_NOT_ADMITTED"
    assert sum(row["actor_transitions"] for row in plan["chunks"]) + QUAL_PAUSE == TARGET
    assert plan["invoice_units"] <= 120
    previous = QUAL_PAUSE
    for row in plan["chunks"]:
        assert row["start"] == previous and row["target"] == TARGET
        assert row["timeout_minutes"] >= 2 * row["est_minutes"]
        assert row["timeout_minutes"] <= 720
        assert row["timeout_minutes"] * 2 / 60 + row["reserve_units"] <= row["unit_cap"]
        previous = row["pause_at"]


@pytest.mark.parametrize(
    "patch",
    [
        {"critic_qualified": False},
        {"transfer_and_release_verified": False},
        {"seconds_per_actor_transition": float("nan")},
        {"seconds_per_actor_transition": 1 / 800},
        {"seconds_per_actor_transition": 0},
        {"prefix_seconds": -1},
        {"prefix_seconds": 48 * 3600},
        {"schema": "bl-ppo-e4-ruler/v1"},
    ],
)
def test_qualification_cannot_admit_collection_only_or_failed_evidence(patch: dict) -> None:
    evidence = qualified() | patch
    with pytest.raises(ValueError):
        measured_plan(evidence)


def test_complete_job_evidence_replaces_provisional_collection_forecast() -> None:
    plan = measured_plan(qualified())
    assert plan["status"] == "MEASURED_CONDITIONALLY_ADMITTED"
    assert plan["invoice_units"] <= 120 and plan["complete_trajectory_hours"] <= 46


def catalog() -> dict:
    return {
        "schema": "bl-p1-runtime-downloads/v1",
        "files": [
            {
                "name": "snapshot.pt",
                "url": "https://example.org/snapshot.pt",
                "sha256": "a" * 64,
                "size_bytes": 42,
            }
        ],
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"name": "../snapshot.pt"},
        {"name": "/snapshot.pt"},
        {"name": "snapshot.pt.complete"},
        {"url": "file:///tmp/secret"},
        {"url": "https://user:password@example.org/x"},
        {"size_bytes": True},
        {"size_bytes": 3 * 1024**3},
        {"sha256": "a" * 63},
    ],
)
def test_runtime_catalog_refuses_unbounded_or_unsafe_inputs(patch: dict) -> None:
    value = catalog()
    value["files"][0].update(patch)
    with pytest.raises(ValueError):
        validate_catalog(value)


def test_runtime_catalog_deduplicates_transitive_names() -> None:
    value = catalog()
    value["files"].append(dict(value["files"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_catalog(value)


def test_runtime_download_verifies_bytes_before_creating_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import json

    from research.bl_ppo.p1_inputs import materialize

    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "https://example.org/snapshot.pt"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"wrong"))
    path = tmp_path / "downloads.json"
    path.write_text(json.dumps(catalog()))
    with pytest.raises(ValueError, match="size/hash"):
        materialize(path, tmp_path / "runtime")
    assert not list((tmp_path / "runtime").iterdir())
