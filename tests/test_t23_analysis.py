"""Focused decision-boundary fixtures for the preregistered T23 analyzer."""

from __future__ import annotations

import copy
import hashlib

from research.t23_r0_screen_20261017 import analyze as t23


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(delta: float = 25.0, extension: int = 0) -> dict:
    rows = []
    for seed in t23.SEEDS:
        for step in t23.STEPS:
            for mix, count in t23.MIX_COUNTS.items():
                for index in range(count * (extension + 1)):
                    world = f"{seed}-{mix}-{index}"
                    common = {
                        "schema": "t23-eval-row/v1",
                        "training_seed": seed,
                        "checkpoint_step": step,
                        "mix": mix,
                        "world_id": world,
                        "world_seed": seed * 10000 + index,
                        "horizon": 5000,
                        "frames_stepped": 5000,
                        "clean_exit": True,
                        "b8_checkpoint_sha256": t23.B8_SHA256,
                        "world_bank_sha256": _digest(f"bank-{seed}"),
                        "world_profile_sha256": _digest(f"profile-{mix}"),
                        "initial_world_sha256": _digest(world),
                        "alive_frame_fraction": 1.0,
                        "no_death_episode": True,
                    }
                    rows.append(
                        {
                            **common,
                            "arm": "b8",
                            "candidate_checkpoint_sha256": t23.B8_SHA256,
                            "mass_integral": 100.0,
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "arm": "candidate",
                            "candidate_checkpoint_sha256": _digest(f"candidate-{seed}-{step}"),
                            "mass_integral": 100.0 + delta,
                        }
                    )
    manifest = {
        "schema": "t23-eval-manifest/v1",
        "extension_round": extension,
        "tripwires": {str(s): False for s in t23.SEEDS},
        "rows": rows,
    }
    if extension:
        manifest["extension_authorization"] = {
            "prior_verdict": "EXTEND",
            "frozen_checkpoint_sha256s": {
                f"{seed}:{step}": _digest(f"candidate-{seed}-{step}")
                for seed in t23.SEEDS
                for step in t23.STEPS
            },
        }
    return manifest


def test_go_and_descriptive_steps_are_reported() -> None:
    result = t23.analyze(_manifest(), bootstrap_draws=10)
    assert result["verdict"] == "GO", result["validation_errors"]
    assert result["steps"]["2000000"]["descriptive_only"] is True
    assert result["steps"]["4000000"]["descriptive_only"] is True
    assert result["steps"]["8000000"]["descriptive_only"] is False
    assert result["steps"]["8000000"]["G"]["df"] == 4


def test_kill_precedes_go_for_scripted_alive_frame_guard() -> None:
    manifest = _manifest()
    # A mass loss alone is permitted by the KILL guard; the scripted mass point
    # will prevent GO, but it does not turn into a false survival KILL.
    for row in manifest["rows"]:
        if (
            row["training_seed"] == 7101
            and row["checkpoint_step"] == 8_000_000
            and row["mix"] == "scripted"
            and row["arm"] == "candidate"
        ):
            row["mass_integral"] = 99.0
    result = t23.analyze(manifest, bootstrap_draws=10)
    # It remains a GO: the scripted point estimate remains nonnegative and the
    # equal-mix five-seed effect satisfies its independent decision rule.
    assert result["verdict"] == "GO", result["validation_errors"]
    for row in manifest["rows"]:
        if (
            row["training_seed"] == 7101
            and row["checkpoint_step"] == 8_000_000
            and row["mix"] == "scripted"
            and row["arm"] == "candidate"
        ):
            row["alive_frame_fraction"] = 0.8
    result = t23.analyze(manifest, bootstrap_draws=10)
    assert result["verdict"] == "KILL"
    assert any("scripted alive-frame guard" in reason for reason in result["kill_reasons"])


def test_extension_then_inconclusive_and_invalid_identity() -> None:
    # Vary one seed so the upper limit is positive without establishing a GO.
    manifest = _manifest(delta=0.0, extension=1)
    for row in manifest["rows"]:
        if (
            row["arm"] == "candidate"
            and row["checkpoint_step"] == 8_000_000
            and row["training_seed"] == 7101
        ):
            row["mass_integral"] += 1.0
    result = t23.analyze(manifest, bootstrap_draws=10)
    assert result["verdict"] == "INCONCLUSIVE"
    bad = copy.deepcopy(manifest)
    bad["rows"][0]["b8_checkpoint_sha256"] = "0" * 64
    assert t23.analyze(bad, bootstrap_draws=10)["verdict"] == "INVALID"


def test_early_terminal_is_valid_h5000_scoring() -> None:
    manifest = _manifest()
    manifest["rows"][0]["frames_stepped"] = 19
    result = t23.analyze(manifest, bootstrap_draws=10)
    assert result["verdict"] == "GO"
