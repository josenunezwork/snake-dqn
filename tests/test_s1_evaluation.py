"""Contract tests for S1's local evaluator and final-look analyzer."""

from __future__ import annotations

import pytest

from research.s1_mature_20261018 import analyze, bank, evaluate


def test_t90_is_one_sided_df4() -> None:
    result = analyze.t90([10.0, 10.0, 10.0, 10.0, 10.0])
    assert result["n"] == 5 and result["df"] == 4
    assert result["lower90"] == result["upper90"] == 10.0
    with pytest.raises(ValueError, match="five"):
        analyze.t90([1.0] * 4)


def test_evaluator_rejects_more_than_two_mac_workers(tmp_path) -> None:
    with pytest.raises(ValueError, match="1..2"):
        evaluate.execute_ordered(
            [],
            checkpoint=tmp_path / "missing",
            carrier=("x", "y"),
            workers=3,
            partials_dir=tmp_path / "partials",
            binding={},
        )


def _input(monkeypatch: pytest.MonkeyPatch, *, n1_mass: float = 25.0) -> dict:
    seeds = (11, 12, 13, 14, 15)
    mixes = {"scripted": 64, "frozen": 32, "mixed": 32}
    worlds: dict[str, list[dict]] = {}
    for seed in seeds:
        rows = []
        for mix, count in mixes.items():
            for offset in range(count):
                number = seed * 1000 + len(rows)
                rows.append(
                    {
                        "world_seed": number,
                        "mix": mix,
                        "world_semantic_sha256": f"{number:064x}",
                        "initial_world_sha256": f"{number + 1:064x}",
                        "roster_semantic_sha256": f"{number + 2:064x}",
                    }
                )
        worlds[str(seed)] = rows
    raw = {"schema": bank.SCHEMA, "worlds": worlds}
    raw["allocation_semantic_sha256"] = bank.canonical_sha({"worlds": worlds})
    monkeypatch.setattr(bank, "TRAINING_SEEDS", seeds)
    monkeypatch.setattr(bank, "MIX_COUNTS", mixes)
    monkeypatch.setattr(bank, "worlds_for_seed", lambda value, seed: value["worlds"][str(seed)])
    source = {"research/s1_mature_20261018/evaluate.py": "e" * 64}
    monkeypatch.setattr(evaluate, "source_manifest", lambda: source)
    shards = []
    for seed in seeds:
        items = worlds[str(seed)]
        for arm, mass in (("N1", n1_mass), ("N3", 21.0), ("B8", 0.0)):
            checkpoint = evaluate.B8_SHA256 if arm == "B8" else ("1" if arm == "N1" else "2") * 64
            identity = {
                "arm": arm,
                "training_seed": seed,
                "endpoint": 8_000_000,
                "checkpoint_sha256": checkpoint,
                "b8_sha256": evaluate.B8_SHA256,
                "bank_sha256": analyze._digest(raw),
                "world_hashes": [w["world_semantic_sha256"] for w in items],
                "world_count": 128,
                "source_manifest": source,
            }
            rows = [
                {
                    "schema": evaluate.ROW_SCHEMA,
                    "arm": arm,
                    "training_seed": seed,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_sha256": checkpoint,
                    "b8_checkpoint_sha256": evaluate.B8_SHA256,
                    "world_bank_sha256": identity["bank_sha256"],
                    "evaluator_source_sha256": "e" * 64,
                    "clean_exit": True,
                    "mass_integral": mass,
                    "alive_frame_fraction": 1.0,
                    "no_death_episode": True,
                    "horizon": 5000,
                    "frames_stepped": 5000,
                    **world,
                }
                for world in items
            ]
            provenance = (
                None
                if arm == "B8"
                else {
                    "schema": "s1-training-provenance/v1",
                    "arm": arm,
                    "training_seed": seed,
                    "endpoint": 8_000_000,
                    "clean_exit": True,
                    "tripwire_triggered": False,
                    "checkpoint_sha256": checkpoint,
                    "trainer_source_sha256": "a" * 64,
                    "runtime_parity_receipt_sha256": "b" * 64,
                    "recipe": {
                        "worlds": 128,
                        "actors": 3 if arm == "N1" else 6,
                        "learner_seats": 1 if arm == "N1" else 3,
                        "schedule_package": "mature",
                        "collection_profile": "task-aligned-n1" if arm == "N1" else "legacy",
                        "transitions": 8_000_000,
                        "namespace": f"test/{arm}/{seed}",
                    },
                    "counters": {
                        "actual_learner_decisions": 8_000_000,
                        "transition_budget_overshoot": 0,
                        "registered_overshoot_bound": (3 if arm == "N1" else 6)
                        * 128
                        * (1 if arm == "N1" else 3)
                        - 1,
                    },
                }
            )
            # The bank method is the authoritative namespace interface in production.
            if provenance:
                provenance["recipe"]["namespace"] = (
                    bank.training_namespace(arm, seed)
                    if False
                    else provenance["recipe"]["namespace"]
                )
            shards.append(
                {
                    "schema": evaluate.SCHEMA,
                    "status": "completed",
                    "clean_exit": True,
                    "identity": identity,
                    "training_provenance": provenance,
                    "rows": rows,
                    "rows_sha256": analyze._digest(rows),
                }
            )
    # The namespace check is deliberate; patch it only for this isolated shape fixture.
    monkeypatch.setattr(bank, "training_namespace", lambda arm, seed: f"test/{arm}/{seed}")
    return {
        "schema": "s1-analysis-input/v1",
        "bank": raw,
        "bank_audit": {
            "pass": True,
            "world_bank_semantic_sha256": raw["allocation_semantic_sha256"],
        },
        "shards": shards,
    }


def test_analysis_orders_go_and_ranks_lb90_mean_then_n1(monkeypatch: pytest.MonkeyPatch) -> None:
    result = analyze.analyze(_input(monkeypatch), bootstrap_draws=20)
    assert result["verdict"] == "GO"
    assert result["selected_recipe"] == "N1"
    assert result["paired_n1_minus_n3"]["df"] == 4
    assert result["hierarchical_world_within_seed_bootstrap"]["draws"] == 20


def test_analysis_withholds_outcomes_on_a_tampered_row(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _input(monkeypatch)
    value["shards"][0]["rows"][0]["world_seed"] += 1
    result = analyze.analyze(value, bootstrap_draws=10)
    assert result["verdict"] == "INVALID/INCOMPLETE"
    assert result["outcomes_withheld"] is True


def test_a_go_survives_the_other_recipe_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    """KILL is arm-local once the full evidence package is integrity-valid."""
    value = _input(monkeypatch, n1_mass=25.0)
    # N3's upper interval becomes nonpositive while N1 remains a qualifying GO.
    for shard in value["shards"]:
        if shard["identity"]["arm"] == "N3":
            for row in shard["rows"]:
                row["mass_integral"] = 0.0
            shard["rows_sha256"] = analyze._digest(shard["rows"])
    result = analyze.analyze(value, bootstrap_draws=10)
    assert result["arms"]["N3"]["verdict"] == "KILL"
    assert result["verdict"] == "GO"
    assert result["selected_recipe"] == "N1"


def test_survival_guard_has_registered_numeric_tolerance() -> None:
    stats = {
        "alive_fraction_differences": {"scripted": -0.03 - 5e-10, "frozen": 0.0, "mixed": 0.0},
        "min_scripted_seed_alive": -0.15 - 5e-10,
        "no_death_episode_fraction": {"scripted": 0.5 - 5e-10, "frozen": 0.5, "mixed": 0.5},
        "G": {"upper90": 1.0, "lower90": -1.0, "mean": 0.0, "seed_effects": [0.0] * 5},
        "mix_mass_differences": {"scripted": 0.0},
    }
    assert analyze._verdict(stats)[0] == "INCONCLUSIVE"
    stats["alive_fraction_differences"]["scripted"] = -0.03 - 2e-9
    assert analyze._verdict(stats)[0] == "KILL"


def test_authenticated_tripwire_kills_one_arm_without_invalidating_healthy_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _input(monkeypatch)
    value["shards"] = [s for s in value["shards"] if s["identity"]["arm"] != "N3"]
    value["training_outcomes"] = [
        {
            "schema": "s1-training-outcome/v1",
            "status": "TRIPWIRE_KILL",
            "arm": "N3",
            "training_seed": 11,
            "b8_sha256": evaluate.B8_SHA256,
            "reason": "nonfinite loss",
            "raw_summary_halt": "nonfinite",
            "trainer_source_sha256": "a" * 64,
            "summary_sha256": "b" * 64,
            "log_sha256": "c" * 64,
            "recipe": {},
            "counters": {},
        }
    ]
    result = analyze.analyze(value, bootstrap_draws=10)
    assert result["verdict"] == "GO"
    assert result["arms"]["N3"]["verdict"] == "KILL"
    assert result["paired_n1_minus_n3"]["status"] == "UNAVAILABLE"
