"""Comparison logic of research/simd_parity_v7v8_h5000_20261003/h5000_check.py.

No episode is ever played here: every episode runner is monkeypatched to raise. The
dry-run test reads the saved live v7/v8 screen entries when they exist (JSON only).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.simd_env.vector61_policy import vector61_provenance


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
def hc():
    from research.simd_parity_v7v8_h5000_20261003 import h5000_check

    return h5000_check


def _live_entry(hc, name: str = "v8-screen-B", seed: int = 11) -> dict:
    target = hc.TARGETS[name]
    method = hc.target_spec(target).method
    return {
        "arm": target.arm,
        "mix": "mixed",
        "world_index": 0,
        "world_seed": seed,
        "safety_veto_method": method,
        "wall_seconds": 30.0,
        "roster_member_sha256s": ["a" * 64],
        "record": {
            "seed": seed,
            "deaths": 1,
            "mass_integral": 12.5,
            "probes": {"safety_veto": {"method": method, "counters": {"decisions": 9}}},
        },
        "veto_diagnostics": {
            "decisions": 9,
            "head_risky_vetoes": 1,
            "apply_seconds_total": 0.25,
            "reference_lambda": target.reference_lambda,
            "v7": {"rerank_changes": 2, "area_eval_seconds_max": 0.1, "v5": {"kept": 7}},
        },
    }


def _simd_record(hc, live: dict, name: str = "v8-screen-B") -> dict:
    target = hc.TARGETS[name]
    record = copy.deepcopy(live["record"])
    record["vector61_policy"] = vector61_provenance(
        "rowwise", target.variant, target.lam, target.reference_lambda
    )
    record["world_runtime_spec"] = {"engine": "simd", "body_storage_capacity": 123}
    record["world_runtime_spec_digest"] = "x" * 64
    diagnostics = copy.deepcopy(live["veto_diagnostics"])
    diagnostics["apply_seconds_total"] = 9.0  # timing never compares
    diagnostics["v7"]["area_eval_seconds_max"] = 3.0
    record["veto_diagnostics"] = diagnostics
    return record


def test_runners_are_fatal():
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    with pytest.raises(AssertionError):
        ee.run_simd_eval()
    with pytest.raises(AssertionError):
        te.rollout()


def test_targets_resolve_to_the_live_methods(hc):
    from src.evaluation import safety_veto_v7, safety_veto_v8

    assert set(hc.DEFAULT_TARGETS) == {"v7-screen-B", "v8-screen-B"}
    assert hc.target_spec(hc.TARGETS["v7-screen-B"]).method == safety_veto_v7.method_for(4.0)
    v8 = hc.TARGETS["v8-screen-B"]
    assert hc.target_spec(v8).method == safety_veto_v8.method_for(8.0)
    assert v8.reference_lambda == 4.0
    for target in hc.TARGETS.values():
        assert hc.target_spec(target).variant == target.variant in ("v7", "v8")
    assert set(hc.SIMD_ONLY_RECORD_KEYS) == {
        "world_runtime_spec",
        "world_runtime_spec_digest",
        "vector61_policy",
        "veto_diagnostics",
    }


def test_strip_timing_removes_every_seconds_key_at_any_depth(hc):
    value = {
        "a": 1,
        "x_seconds_total": 2.0,
        "n": {"mean_apply_seconds": None, "b": [{"c_seconds": 1}]},
    }
    assert hc.strip_timing(value) == {"a": 1, "n": {"b": [{}]}}


@pytest.mark.parametrize("name", ["v7-screen-B", "v8-screen-B"])
def test_identical_world_passes(hc, name):
    live = _live_entry(hc, name)
    result = hc.compare_world(live, _simd_record(hc, live, name), hc.TARGETS[name])
    assert result["identical"], result


@pytest.mark.parametrize(
    "mutate, flag",
    [
        (lambda r: r.update(deaths=0), "record_identical"),
        (lambda r: r.update(mass_integral=12), "record_identical"),  # int vs float
        (lambda r: r["probes"]["safety_veto"]["counters"].update(decisions=8), "record_identical"),
        (lambda r: r.update(extra_key=1), "record_identical"),
        (lambda r: r["veto_diagnostics"].update(head_risky_vetoes=0), "diagnostics_identical"),
        (lambda r: r["veto_diagnostics"]["v7"]["v5"].update(kept=6), "diagnostics_identical"),
        (lambda r: r.pop("veto_diagnostics"), "diagnostics_identical"),
        (lambda r: r["vector61_policy"].update(forward="batched"), "provenance_ok"),
        (lambda r: r["vector61_policy"].pop("safety_veto_reference_lambda"), "provenance_ok"),
    ],
)
def test_any_difference_fails_and_is_located(hc, mutate, flag):
    live = _live_entry(hc)
    simd = _simd_record(hc, live)
    mutate(simd)
    result = hc.compare_world(live, simd, hc.TARGETS["v8-screen-B"])
    assert not result["identical"] and not result[flag], result


def test_a_live_record_holding_simd_only_keys_fails(hc):
    live = _live_entry(hc)
    simd = _simd_record(hc, live)
    live["record"]["vector61_policy"] = simd["vector61_policy"]
    result = hc.compare_world(live, simd, hc.TARGETS["v8-screen-B"])
    assert not result["identical"] and result["live_has_simd_only_keys"] == ["vector61_policy"]


def test_tally_passes_only_when_every_planned_world_is_identical(hc):
    ok = {"identical": True}
    bad = {"identical": False}
    assert hc.tally([ok, ok], 2)["pass"]
    assert not hc.tally([ok], 2)["pass"]  # incomplete (e.g. a thermal stop)
    assert not hc.tally([ok, bad], 2)["pass"]
    assert hc.tally([ok, bad], 2)["different"] == 1


def test_select_entries_orders_by_world_index_and_refuses_gaps(hc):
    target = hc.TARGETS["v7-screen-B"]
    entries = [
        {"arm": "B", "mix": "frozen", "world_index": i, "world_seed": 100 + i} for i in (3, 0, 2, 1)
    ]
    entries += [{"arm": "A", "mix": "frozen", "world_index": 9, "world_seed": 9}]
    chosen = hc.select_entries(entries, target, "frozen", 3)
    assert [e["world_index"] for e in chosen] == [0, 1, 2]
    with pytest.raises(ValueError, match="only 4 live entries"):
        hc.select_entries(entries, target, "frozen", 5)
    with pytest.raises(ValueError, match="duplicate world_index"):
        hc.select_entries(entries + [dict(entries[0])], target, "frozen", 2)


def test_check_entry_flags_the_wrong_arm_method_or_reference(hc):
    target = hc.TARGETS["v8-screen-B"]
    live = _live_entry(hc)
    assert hc.check_entry(live, target) == []
    wrong = copy.deepcopy(live)
    wrong["veto_diagnostics"]["reference_lambda"] = None
    assert any("reference_lambda" in p for p in hc.check_entry(wrong, target))
    wrong = copy.deepcopy(live)
    wrong["safety_veto_method"] = "free-space-veto/v8-space-and-head(lambda=16.0)"
    assert hc.check_entry(wrong, target)
    assert hc.check_entry(live, hc.TARGETS["v7-screen-B"])  # arm/method mismatch


def test_check_intent_accepts_dict_and_string_arm_descriptions(hc):
    from research.apex_safety_20260926 import dev_screen as ds

    common = {
        "config": {"sha256": ds.CONFIG_SHA256},
        "profile": {"digest": ds.PROFILE_DIGEST, "horizon": ds.HORIZON},
        "hero": {"sha256": ds.CHAMPION[1]},
    }
    v8 = {
        **common,
        "arms": {"B": {"lambda": 8.0, "method": "free-space-veto/v8-space-and-head(lambda=8.0)"}},
    }
    assert hc.check_intent(v8, hc.TARGETS["v8-screen-B"]) == []
    v7 = {
        **common,
        "arms": {"B": "champion + free-space-veto/v7-space-preference(lambda=4.0) (...)"},
    }
    assert hc.check_intent(v7, hc.TARGETS["v7-screen-B"]) == []
    assert hc.check_intent({**v8, "hero": {"sha256": "0" * 64}}, hc.TARGETS["v8-screen-B"])
    assert hc.check_intent(v8, hc.TARGETS["v8-sweep-H1600"])


def _have_live_inputs(hc) -> bool:
    return all(
        (Path(run) / "intent.json").is_file()
        for name in hc.DEFAULT_TARGETS
        for run in hc.TARGETS[name].runs
    )


def test_dry_run_validates_the_saved_live_inputs(hc, capsys):
    """Reads the saved live v7/v8 screen entries (JSON only); no torch, lock or episode."""
    if not _have_live_inputs(hc):
        pytest.skip("saved live v7/v8 screen artifacts not available")
    assert hc.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "planned SIMD episodes: 24" in out
    for name in hc.DEFAULT_TARGETS:
        loaded = hc.load_target(name, 4)
        for mix, chosen in loaded["plan"].items():
            assert [int(e["world_index"]) for e in chosen] == [0, 1, 2, 3], (name, mix)
            assert all(hc.check_entry(e, hc.TARGETS[name]) == [] for e in chosen)
