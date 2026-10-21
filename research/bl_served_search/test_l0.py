"""Focused tests for L0's saved-timeline integrity boundary."""

from pathlib import Path

from research.bl_served_search.exports import export_json, verify_export
from research.bl_served_search.l0 import SMOKE, digest, inspect, timeline_entries


def test_timeline_entries_requires_every_completion_and_candidate_frame() -> None:
    assert timeline_entries({"kind": "rescue"}) is None
    assert timeline_entries({"evaluation_timeline": [{"completion_seconds": 1.0}]}) is None
    assert timeline_entries(
        {"evaluation_timeline": [{"completion_seconds": 1.0, "candidate_frame": 90}]}
    ) == [{"completion_seconds": 1.0, "candidate_frame": 90}]


def test_retained_smoke_is_incomplete_for_the_required_replay() -> None:
    audit, hashes = inspect(SMOKE)
    assert audit["lad_records"] == 80
    assert audit["rescue_capable"] == 34
    assert audit["delay_only"] == 41
    assert audit["missing_timeline_records"] == 80
    assert audit["n1_diagnostic"]["n1_is_not_substituted"]
    assert len(hashes) == 160
    example = next(SMOKE.glob("lad-*.json"))
    assert hashes[str(example)] == digest(example)


def test_l0_exports_use_completion_marker_and_hash_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "l0.json"
    export_json(path, {"status": "INCOMPLETE"})
    assert path.with_name("l0.json.complete").read_text() == "complete\n"
    assert verify_export(path)
