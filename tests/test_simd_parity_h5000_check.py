"""Comparison logic of research/simd_parity_h5000_20261001/aa_check.py.

Uses two saved LIVE records of the apex-safety screen (run-v1, frozen world
2920083565, arms A and B) as fixtures. No episode is ever played here: every
episode runner is monkeypatched to raise.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "simd_parity_h5000"


@pytest.fixture(autouse=True)
def no_episodes(monkeypatch):
    """Make every episode runner fatal for the whole module."""
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    def boom(*_args, **_kwargs):
        raise AssertionError("an episode runner was called in a comparison-only test")

    monkeypatch.setattr(ee, "run_simd_eval", boom)
    monkeypatch.setattr(te, "rollout", boom)


@pytest.fixture
def aa():
    from research.simd_parity_h5000_20261001 import aa_check

    return aa_check


def _live(arm: str) -> dict:
    return json.loads((FIXTURES / f"{arm}-frozen-2920083565.json").read_text())


def _as_simd(aa, live: dict) -> tuple[dict, dict]:
    """A SIMD-shaped copy: same record plus the SIMD-only keys, new wrapper labels."""
    from src.simd_env.vector61_policy import vector61_provenance

    entry = copy.deepcopy(live)
    entry.update(schema_version=aa.SCHEMA, authority=aa.AUTHORITY, wall_seconds=0.1)
    provenance = vector61_provenance("rowwise", bool(live["safety_veto"]))
    entry["record"]["vector61_policy"] = copy.deepcopy(provenance)
    entry["record"]["world_runtime_spec"] = {"engine": "simd", "body_storage_capacity": 123}
    entry["record"]["world_runtime_spec_digest"] = "x" * 64
    return entry, provenance


def test_runners_are_fatal():
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    with pytest.raises(AssertionError):
        ee.run_simd_eval()
    with pytest.raises(AssertionError):
        te.rollout()


def test_excluded_keys_are_exactly_the_simd_only_ones(aa):
    assert set(aa.SIMD_ONLY_RECORD_KEYS) == {
        "world_runtime_spec",
        "world_runtime_spec_digest",
        "vector61_policy",
    }
    assert set(aa.EXCLUDED_ENTRY_KEYS) == {"wall_seconds", "schema_version", "authority"}


@pytest.mark.parametrize("arm", ["A", "B"])
def test_identical_record_passes(aa, arm):
    live = _live(arm)
    simd, provenance = _as_simd(aa, live)
    result = aa.compare_entry(live, simd, provenance)
    assert result["identical"], result["first_difference"]
    assert result["record_identical_minus_simd_only_keys"]
    assert all(cell["equal"] for cell in result["headline"].values())
    if arm == "B":
        assert result["headline"]["probes.safety_veto.counters"]["live"]["vetoes_applied"] == 9


@pytest.mark.parametrize(
    "mutate, path",
    [
        (
            lambda r: r.__setitem__("mass_integral", r["mass_integral"] + 1e-12),
            "record.mass_integral",
        ),
        (lambda r: r.__setitem__("deaths", 0), "record.deaths"),
        (lambda r: r["probes"].__setitem__("death_cause", "enemy"), "record.probes.death_cause"),
        (lambda r: r.__setitem__("kills", 1.0), "record.kills"),  # int vs float is a difference
        (
            lambda r: r["probes"]["safety_veto"]["counters"].__setitem__("vetoes_applied", 8),
            "record.probes.safety_veto.counters.vetoes_applied",
        ),
        (lambda r: r["denominators"].pop("alive_frames"), "record.denominators.alive_frames"),
        (lambda r: r.__setitem__("new_key", 1), "record.new_key"),
        (
            lambda r: r["world_identity"]["ordered_slot_content_hashes"].pop(),
            "record.world_identity.ordered_slot_content_hashes[len]",
        ),
    ],
)
def test_mutated_copy_is_caught_with_its_path(aa, mutate, path):
    live = _live("B")
    simd, provenance = _as_simd(aa, live)
    mutate(simd["record"])
    result = aa.compare_entry(live, simd, provenance)
    assert not result["identical"]
    assert not result["record_identical_minus_simd_only_keys"]
    assert result["first_difference"]["path"] == path
    report = aa.tally([result, aa.compare_entry(live, _as_simd(aa, live)[0], provenance)])
    assert (report["identical"], report["differing"]) == (1, 1)
    assert report["first_differing"]["first_difference"]["path"] == path


def test_wrapper_and_provenance_mismatches_are_caught(aa):
    live = _live("A")
    simd, provenance = _as_simd(aa, live)
    simd["roster_member_sha256s"] = list(reversed(simd["roster_member_sha256s"]))
    result = aa.compare_entry(live, simd, provenance)
    assert result["first_difference"]["path"] == "entry.roster_member_sha256s"

    simd, provenance = _as_simd(aa, live)
    simd["record"]["vector61_policy"]["bit_exact_forward"] = False  # e.g. batched forward
    assert aa.compare_entry(live, simd, provenance)["first_difference"]["path"] == (
        "record.vector61_policy"
    )

    simd, provenance = _as_simd(aa, live)
    leaky = copy.deepcopy(live)
    leaky["record"]["world_runtime_spec"] = {}
    assert not aa.compare_entry(leaky, simd, provenance)["identical"]


def test_runtime_wrapper_keys_do_not_count(aa):
    live = _live("B")
    simd, provenance = _as_simd(aa, live)
    simd["wall_seconds"] = 1e9
    simd["schema_version"] = "other"
    assert aa.compare_entry(live, simd, provenance)["identical"]
