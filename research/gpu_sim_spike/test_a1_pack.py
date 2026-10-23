"""Offline contracts for A1's full-tick ruler transport pack."""

from __future__ import annotations

import json

from research.gpu_sim_spike import a100_ruler, make_a1_pack


def _row(mix: str, worlds: int, rate: float, tick: float) -> dict:
    return {
        "mix": mix,
        "requested_worlds": worlds,
        "median_complete_steps_per_second": rate,
        "median_tick_seconds": tick,
    }


def test_a1_kill_tiers_are_fixed_by_every_1024_mix() -> None:
    rows = [
        _row(mix, worlds, 21_000.0, 1.0) for mix in a100_ruler.MIXES for worlds in a100_ruler.SHAPES
    ]
    control = _row("long-scripted", 6, 1.0, 10.0)
    assert (
        a100_ruler.classify(rows, control, {"classification": "float tie"})["tier"]
        == "FUND_WP1C_WP2"
    )
    rows[-1]["median_complete_steps_per_second"] = 1_999.0
    assert (
        a100_ruler.classify(rows, control, {"classification": "float tie"})["tier"] == "STOP_ENGINE"
    )
    rows[-1]["median_complete_steps_per_second"] = 2_000.0
    assert (
        a100_ruler.classify(rows, control, {"classification": "float tie"})["tier"] == "FUND_WP1B"
    )
    rows[0]["median_tick_seconds"] = 3.0
    assert (
        a100_ruler.classify(rows, control, {"classification": "float tie"})["tier"]
        == "EVENT_PATH_REJECTED"
    )
    assert a100_ruler.classify(rows, control, {"classification": "bug"})["tier"] == "INVALID"


def test_a1_main_preserves_plan_acceptance_envelope_and_canary_is_short(
    tmp_path, monkeypatch
) -> None:
    commit = "a" * 40
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(make_a1_pack.empty_ledger(commit)))
    monkeypatch.setattr(make_a1_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(make_a1_pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(make_a1_pack, "assert_upload_budget", lambda value, repo: None)
    main = make_a1_pack.manifest(
        commit=commit, output_dir=tmp_path / "main-output", ledger=ledger, canary=False
    )
    canary = make_a1_pack.manifest(
        commit=commit, output_dir=tmp_path / "canary-output", ledger=ledger, canary=True
    )
    assert main["requirements"]["allowed_gpus"] == ["A100"]
    assert main["env"]["GPU_SIM_SPIKE_COMMIT"] == commit
    assert main["timeout_minutes"] == 120
    assert main["budget"] == {
        "unit_cap": 20.0,
        "rate_units_per_hour": 10.0,
        "reserve_units": 4.0,
        "other_assignments": {},
    }
    assert main["resume_exemption"]["kind"] == "acceptance"
    assert "resume" not in main
    assert main["required_exports"] == ["a1-ruler.json", "a0-native-oracle.json"]
    assert canary["timeout_minutes"] == 18
    assert canary["budget"]["unit_cap"] == 3.0
    assert canary["resume_exemption"]["kind"] == "short-canary"
    assert canary["env"]["A1_RULER_SMOKE"] == "1"
