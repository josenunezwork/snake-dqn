"""Synthetic-only E2 statistical, pairing and no-peek bank contract checks."""

from __future__ import annotations

import copy

import pytest

from research.bl_ppo import bank, screen
from research.bl_ppo.rule import RULE


def _refresh_ledgers(results: dict) -> None:
    """Populate the compact synthetic fixture with E8-reconciling ledgers."""
    aggregate = screen._empty_e8_ledgers()
    for row in results["rows"]:
        ledger = {}
        for arm in ("R", "F"):
            if arm not in row["controllers"]:
                continue
            delta = (
                row["controllers"][arm]["mass_integral"] - row["controllers"]["B8"]["mass_integral"]
            )
            ledger[arm] = {
                "mass": {
                    "both_alive_growth": delta * row["horizon"],
                    "candidate_only": 0.0,
                    "b8_only": 0.0,
                    "total": delta * row["horizon"],
                    "per_frame": delta,
                },
                "occupancy_by_frame_length": {},
                "endpoint": None,
                "candidate_growth_slope_l_ge_800": None,
            }
        row["e8_ledger"] = ledger
        screen._add_e8_ledger(aggregate, row["mix"], ledger)
    results["e8_ledgers"] = aggregate


@pytest.fixture(scope="module")
def fixture() -> tuple[dict, dict, dict]:
    value = bank.materialize()
    audit = {
        "schema": "bl-ppo-screen-seed-audit/v1",
        "method": "safe-root-pruned-literal-history-v1",
        "explicit_safe_file_count": 1,
        "pass": True,
        "bank_semantic_sha256": value["allocation_sha256"],
        "collisions": [],
        "seeds": sorted(bank.all_seeds(value)),
        "concrete_hits": {},
        "historical_frp_exclusions": [{}] * 7,
        "published_inventories": [{}],
        "historical_generated_seed_count": 1,
    }
    hashes = {
        "R": "a" * 64,
        "F": "b" * 64,
        "B8": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    }
    rows = []
    for world in value["rows"]:
        rows.append(
            {
                **{k: world[k] for k in ("world_id", "world_seed", "mix", "horizon")},
                "roster_sha256": bank.canonical_sha(world["opponents"]),
                "cpu_identity": "synthetic",
                "controllers": {
                    arm: {
                        "mass_integral": mass,
                        "alive_frame_fraction": 1.0,
                        "no_death_episode": True,
                        "frames_stepped": 5000,
                        "checkpoint_sha256": hashes[arm],
                    }
                    for arm, mass in (("R", 120), ("F", 105), ("B8", 100))
                },
            }
        )
    results = {
        "schema": "bl-ppo-screen-results/v2",
        "complete": True,
        "cap_expired": False,
        "bank_semantic_sha256": value["allocation_sha256"],
        "rows": rows,
        "parity_pass": True,
        "checkpoint_sha256": hashes,
        "training_status": {
            arm: {
                "complete": True,
                "tripwires": [],
                "receipt_sha256": "c" * 64,
                "target_transitions": 8_000_000,
                "counters": {
                    "hero_transitions": 10_000_000,
                    "actor_transitions": 8_000_000,
                    "critic_refit_transitions": 2_000_000,
                    "critic_refit_fit_transitions": 2_000_000,
                },
                "critic_refit": {
                    "required_transitions": 2_000_000,
                    "observed_transitions": 2_000_000,
                    "observed_fit_transitions": 2_000_000,
                    "popart": True,
                    "qualified": True,
                    "heldout_late_body": {
                        "count": 1,
                        "explained_variance": 0.2,
                        "mse": 0.5,
                        "baseline_mse": 1.0,
                        "beats_baseline": True,
                    },
                },
                "critic_qualified": True,
                "boost_contract_floor": {
                    "candidate_rate": 0.004,
                    "b8_rate": 0.004,
                    "floor": 0.005,
                    "below_floor": True,
                },
                "e9_rule_sha256": bank.canonical_sha(RULE),
                "e9_receipt_sha256": "d" * 64,
                "checkpoint_sha256": hashes[arm],
            }
            for arm in ("R", "F")
        },
    }
    _refresh_ledgers(results)
    return value, audit, results


def test_allocation_and_fixture_disjointness(fixture: tuple) -> None:
    value, _, _ = fixture
    bank.validate_bank(value)
    assert len(bank.all_seeds(value)) == 230
    assert {
        m: sum(r["mix"] == m for r in value["rows"]) for m in bank.MIX_COUNTS
    } == bank.MIX_COUNTS
    edited = copy.deepcopy(value)
    edited["rows"][0]["opponents"].reverse()
    # Scripted roster may contain clones; edit a real semantic field instead.
    edited["rows"][0]["horizon"] = 4999
    with pytest.raises(ValueError):
        bank.validate_bank(edited)


def test_equal_mix_pooling_not_row_weighted() -> None:
    stats = screen.paired_bounds(
        {"scripted": [30.0] * 96, "frozen": [0.0] * 32, "mixed": [0.0] * 32}
    )
    assert stats["estimate"] == 10
    assert stats["lower"] == stats["upper"] == 10
    assert stats["df"] is None


def test_one_sided_welch_interval() -> None:
    from scipy.stats import t

    groups = {"scripted": list(range(96)), "frozen": list(range(32)), "mixed": list(range(32))}
    stats = screen.paired_bounds(groups)
    assert stats["lower"] == pytest.approx(
        stats["estimate"] - t.isf(0.1, stats["df"]) * stats["se"]
    )


def test_go_inconclusive_and_recurrent_contrast(fixture: tuple) -> None:
    decision = screen.analyze(*fixture)
    assert decision["arms"]["R"]["status"] == "GO"
    assert decision["arms"]["F"]["status"] == "INCONCLUSIVE"
    assert decision["R_minus_F"]["estimate"] == 15
    assert not decision["registered_verdict_eligible"]


def test_kill_precedes_go_and_zero_bound_kills(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"]["tripwires"] = [
        {
            "kind": "boost_share",
            "window_count": 100_000,
            "action_counts": [20_000, 20_000, 20_000, 40_000, 0, 0],
            "b8_boost_count": 10_000,
        }
    ]
    for row in results["rows"]:
        row["controllers"]["F"]["mass_integral"] = 100
    _refresh_ledgers(results)
    decision = screen.analyze(value, audit, results)
    assert decision["arms"]["R"]["status"] == "KILL"
    assert decision["arms"]["F"]["status"] == "KILL"


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "roster", "nonfinite", "endpoint", "parity"]
)
def test_incomplete_evidence_has_precedence(fixture: tuple, corruption: str) -> None:
    value, audit, results = copy.deepcopy(fixture)
    if corruption == "missing":
        results["rows"].pop()
    elif corruption == "duplicate":
        results["rows"][-1] = results["rows"][0]
    elif corruption == "roster":
        results["rows"][0]["roster_sha256"] = "wrong"
    elif corruption == "nonfinite":
        results["rows"][0]["controllers"]["R"]["mass_integral"] = float("nan")
    elif corruption == "endpoint":
        results["rows"][0]["controllers"]["R"]["checkpoint_sha256"] = "e" * 64
    else:
        results["parity_pass"] = False
    assert screen.analyze(value, audit, results)["status"] == "INCOMPLETE"


def test_nonnegative_no_death_and_survival_guard_required(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["rows"][0]["controllers"]["R"]["no_death_episode"] = False
    assert screen.analyze(value, audit, results)["arms"]["R"]["status"] == "INCONCLUSIVE"
    for row in results["rows"]:
        row["controllers"]["R"]["alive_frame_fraction"] = 0.9
        row["controllers"]["R"]["no_death_episode"] = True
    decision = screen.analyze(value, audit, results)
    assert not decision["arms"]["R"]["survival_band_v2"]["pass"]
    assert decision["arms"]["R"]["status"] == "INCONCLUSIVE"


def test_sealed_directory_is_excluded_before_traversal(tmp_path, monkeypatch) -> None:
    import os
    from pathlib import Path

    from research.t34_round0d import bank as prior

    assert "t34-round0-study-01" in prior.LIVE_EXCLUSIONS
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    for directory in (repo / "docs", repo / "research", artifacts / "t34-round0-study-01"):
        directory.mkdir(parents=True)
    (artifacts / "t34-round0-study-01" / "hidden.json").write_text('{"seed": 987654321}')
    (repo / "docs" / "visible.json").write_text('{"seed": 123456789}')
    for index in range(7):
        directory = artifacts / f"frp-{index}" / "namespaces"
        directory.mkdir(parents=True)
        (directory / "training_exclusions.json").write_text("[]")
    import subprocess

    original_run = subprocess.run

    def guarded_run(command, **kwargs):
        assert all("t34-round0-study-01" not in str(arg) for arg in command)
        for argument in command:
            if str(argument).startswith(str(tmp_path)):
                assert not __import__("pathlib").Path(argument).is_dir()
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    original_scandir = os.scandir

    def guarded_scandir(path):
        assert "t34-round0-study-01" not in Path(path).parts
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)
    sealed = artifacts / "t34-round0-study-01"
    sealed.chmod(0)
    try:
        receipt = bank._scan_history([987654321, 123456789], repo=repo, artifacts=artifacts)
    finally:
        sealed.chmod(0o700)
    found = {number for values in receipt["concrete_hits"].values() for number in values}
    assert found == {123456789}


def test_intentional_tripwire_can_kill_before_endpoint(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"].update(
        {
            "complete": False,
            "stop_reason": "tripwire",
            "counters": {
                "hero_transitions": 0,
                "actor_transitions": 0,
                "critic_refit_transitions": 0,
                "critic_refit_fit_transitions": 0,
            },
            "tripwires": [{"kind": "nonfinite", "stage": "gradient"}],
        }
    )
    results["checkpoint_sha256"].pop("R")
    for row in results["rows"]:
        row["controllers"].pop("R")
    _refresh_ledgers(results)
    decision = screen.analyze(value, audit, results)
    assert decision["status"] == "COMPLETE"
    assert decision["arms"]["R"]["status"] == "KILL"
    assert decision["R_minus_F"] is None


def test_both_scientific_stops_need_no_outcome_episodes(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    for arm in ("R", "F"):
        results["training_status"][arm].update(
            {
                "complete": False,
                "stop_reason": "tripwire",
                "counters": {
                    "hero_transitions": 100_000,
                    "actor_transitions": 100_000,
                    "critic_refit_transitions": 0,
                    "critic_refit_fit_transitions": 0,
                },
                "tripwires": [
                    {
                        "kind": "action_collapse",
                        "window_count": 100_000,
                        "action_counts": [91_000, 9000, 0, 0, 0, 0],
                    }
                ],
            }
        )
        results["checkpoint_sha256"].pop(arm)
    results["rows"] = []
    _refresh_ledgers(results)
    decision = screen.analyze(value, audit, results)
    assert decision["status"] == "COMPLETE"
    assert all(x["status"] == "KILL" for x in decision["arms"].values())


@pytest.mark.parametrize(
    "evidence",
    [
        "NaN",
        {"kind": "made_up"},
        {"kind": "nonfinite", "stage": "unknown"},
        {
            "kind": "action_collapse",
            "window_count": 100_000,
            "action_counts": [-1, 100_001, 0, 0, 0, 0],
        },
    ],
)
def test_malformed_tripwire_is_incomplete(fixture: tuple, evidence) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"]["stop_reason"] = "tripwire"
    results["training_status"]["R"]["tripwires"] = [evidence]
    assert screen.analyze(value, audit, results)["status"] == "INCOMPLETE"


def test_terminal_world_metrics_and_bare_advisory_actions(tmp_path, monkeypatch) -> None:
    import numpy as np
    import torch

    class Model:
        def eval(self):
            return self

        def __call__(self, *args):
            return torch.tensor([[0.0, 1.0, 2.0, 100.0, 0.0, 0.0]])

    class Driver:
        frame = 0

        def observe_hero(self):
            return {
                "local": np.zeros((1, 7, 31, 31), dtype=np.uint8),
                "global": np.zeros((1, 3, 25, 25), dtype=np.uint8),
                "scalars": np.zeros((1, 60), dtype=np.float32),
                "mask": np.array([[True, True, True, False, False, False]]),
            }

        def step(self, action):
            assert action == 2
            self.frame += 1
            return {"alive": self.frame < 3, "done": self.frame == 3, "mass": 10 * self.frame}

    monkeypatch.setattr("src.model.ego2s_network.load_ego2s_checkpoint", lambda *args: Model())
    path = tmp_path / "b8.pt"
    path.write_bytes(b"synthetic")
    metrics = screen.evaluate_world(
        {"world_seed": 1, "opponents": [], "horizon": 5},
        path,
        "B8",
        driver_factory=lambda *args: Driver(),
    )
    assert metrics["mass_integral"] == 6
    assert metrics["alive_frame_fraction"] == 0.4
    assert metrics["no_death_episode"] is False
    assert metrics["frames_stepped"] == 5
    assert metrics["active_frames"] == 3


def test_tiny_parity_cannot_qualify_screen() -> None:
    value = {
        "schema": "bl-ppo-runtime-parity/v1",
        "preflight_only": True,
        "runtime": {"system": "Darwin"},
        "rows": [],
    }
    result = screen.compare_parity(value, value)
    assert not result["pass"]
    assert "tiny preflight cannot qualify full screen" in result["errors"]


def test_complete_endpoint_requires_exact_selected_transition_target(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"]["counters"]["hero_transitions"] = 0
    assert screen.analyze(value, audit, results)["status"] == "INCOMPLETE"


def test_endpoint_screen_refuses_training_not_bound_to_frozen_e9_receipt(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"]["e9_receipt_sha256"] = "not-a-sha"
    decision = screen.analyze(value, audit, results)
    assert decision["status"] == "INCOMPLETE"
    assert "frozen admitted E9 rule" in decision["errors"][0]


def test_endpoint_screen_requires_two_million_actual_critic_fit_transitions(fixture: tuple) -> None:
    value, audit, results = copy.deepcopy(fixture)
    results["training_status"]["R"]["counters"]["critic_refit_fit_transitions"] = 1_999_999
    results["training_status"]["R"]["critic_refit"]["observed_fit_transitions"] = 1_999_999
    decision = screen.analyze(value, audit, results)
    assert decision["status"] == "INCOMPLETE"
    assert "critic must pass PopArt" in decision["errors"][0]


def test_parity_compares_actions_observations_not_cpu_logit_bits() -> None:
    trace = [
        {
            "frame": 0,
            "action": 1,
            "observation_sha256": "x",
            "alive": False,
            "mass": 0.0,
            "logits_hex": [float(x).hex() for x in (0, 1, 0, 0, 0, 0)],
            "top2_margin": 1.0,
        }
    ]

    def metrics(steps):
        return {"frames_stepped": 5000, "trace": steps, "trace_sha256": bank.canonical_sha(steps)}

    left = {
        "schema": "bl-ppo-runtime-parity/v1",
        "preflight_only": False,
        "runtime": {"system": "Darwin"},
        "rows": [{"world_id": str(i), "controllers": {"B8": metrics(trace)}} for i in range(3)],
    }
    right = copy.deepcopy(left)
    changed = copy.deepcopy(trace)
    changed[0]["logits_hex"][1] = float(1.0000001).hex()
    right["rows"][0]["controllers"]["B8"] = metrics(changed)
    compared = screen.compare_parity(left, right)
    assert compared["pass"]
    assert compared["logit_numerical_differences"][0]["different_logit_values"] == 1
    changed[0]["action"] = 2
    right["rows"][0]["controllers"]["B8"] = metrics(changed)
    assert not screen.compare_parity(left, right)["pass"]
