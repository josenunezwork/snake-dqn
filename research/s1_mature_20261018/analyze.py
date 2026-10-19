#!/usr/bin/env python3
"""Fail-closed S1 final-look analysis.

The independent unit is the training seed.  World resampling is reported only
as a hierarchical sensitivity analysis and cannot convert 640 worlds into more
than five independent training replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research.s1_mature_20261018 import bank, evaluate
from src.scripts.eval_stats import student_t_isf

ARMS = ("N1", "N3")
B8 = "B8"
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
GUARD_TOLERANCE = 1e-9


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def t90(values: list[float]) -> dict[str, float | int]:
    if len(values) != 5:
        raise ValueError("S1 requires exactly five independent seed blocks")
    sd = statistics.stdev(values)
    se = sd / math.sqrt(5)
    critical = student_t_isf(0.10, 4)
    m = mean(values)
    return {
        "n": 5,
        "df": 4,
        "mean": m,
        "sample_sd": sd,
        "standard_error": se,
        "critical_one_sided_90": critical,
        "lower90": m - critical * se,
        "upper90": m + critical * se,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_input(
    value: Mapping[str, Any], tripwire_arms: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate all 1,920 expected rows before exposing any study outcome."""
    errors: list[str] = []
    if value.get("schema") != "s1-analysis-input/v1":
        return [], ["wrong analysis input schema"]
    raw_bank, audit, shards = value.get("bank"), value.get("bank_audit"), value.get("shards")
    if not isinstance(raw_bank, dict) or raw_bank.get("schema") != bank.SCHEMA:
        return [], ["missing S1 bank"]
    allocation = {"worlds": raw_bank.get("worlds")}
    # Timing worlds were added after the initial bank adapter.  Include them
    # only once the allocator binds them in its allocation digest.
    if "timing_worlds" in raw_bank:
        allocation["timing_worlds"] = raw_bank.get("timing_worlds")
    if raw_bank.get("allocation_semantic_sha256") != bank.canonical_sha(allocation):
        errors.append("bank semantic digest mismatch")
    if (
        not isinstance(audit, dict)
        or audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != raw_bank.get("allocation_semantic_sha256")
    ):
        errors.append("bank collision audit failed or unbound")
    if not isinstance(shards, list):
        return [], errors + ["shards must be a list"]
    bank_sha = _digest(raw_bank)
    source = evaluate.source_manifest()
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for shard in shards:
        try:
            if (
                not isinstance(shard, dict)
                or shard.get("schema") != evaluate.SCHEMA
                or shard.get("status") != "completed"
                or shard.get("clean_exit") is not True
            ):
                raise ValueError("incomplete evaluation shard")
            identity = shard["identity"]
            seed, arm = identity["training_seed"], identity["arm"]
            if seed not in bank.TRAINING_SEEDS or arm not in (*ARMS, B8):
                raise ValueError("unregistered seed or arm")
            if (seed, arm) in indexed:
                raise ValueError("duplicate evaluation identity")
            worlds = bank.worlds_for_seed(raw_bank, seed)
            if (
                identity.get("endpoint") != 8_000_000
                or identity.get("bank_sha256") != bank_sha
                or identity.get("b8_sha256") != evaluate.B8_SHA256
                or identity.get("world_hashes") != [w["world_semantic_sha256"] for w in worlds]
                or identity.get("source_manifest") != source
            ):
                raise ValueError("identity binding mismatch")
            if arm == B8:
                if (
                    identity.get("checkpoint_sha256") != evaluate.B8_SHA256
                    or shard.get("training_provenance") is not None
                ):
                    raise ValueError("B8 comparator receipt mismatch")
            else:
                provenance = shard.get("training_provenance")
                needed = {
                    "schema": "s1-training-provenance/v1",
                    "arm": arm,
                    "training_seed": seed,
                    "endpoint": 8_000_000,
                    "clean_exit": True,
                    "tripwire_triggered": False,
                    "checkpoint_sha256": identity.get("checkpoint_sha256"),
                }
                if not isinstance(provenance, dict) or any(
                    provenance.get(k) != v for k, v in needed.items()
                ):
                    raise ValueError("candidate training provenance mismatch")
                if any(
                    not isinstance(provenance.get(k), str) or len(provenance[k]) != 64
                    for k in ("trainer_source_sha256", "runtime_parity_receipt_sha256")
                ):
                    raise ValueError("candidate provenance lacks trainer/runtime hash binding")
                actors, seats = (3, 1) if arm == "N1" else (6, 3)
                recipe = provenance.get("recipe")
                counters = provenance.get("counters")
                expected = {
                    "worlds": 128,
                    "actors": actors,
                    "learner_seats": seats,
                    "schedule_package": "mature",
                    "collection_profile": "task-aligned-n1" if arm == "N1" else "legacy",
                    "transitions": 8_000_000,
                    "namespace": bank.training_namespace(arm, seed),
                }
                if (
                    not isinstance(recipe, dict)
                    or any(recipe.get(k) != v for k, v in expected.items())
                    or not isinstance(counters, dict)
                ):
                    raise ValueError("candidate recipe/counters mismatch")
                decisions, overshoot, bound = (
                    counters.get(k)
                    for k in (
                        "actual_learner_decisions",
                        "transition_budget_overshoot",
                        "registered_overshoot_bound",
                    )
                )
                if (
                    type(decisions) is not int
                    or type(overshoot) is not int
                    or type(bound) is not int
                    or not 8_000_000 <= decisions <= 8_000_000 + bound
                    or overshoot != decisions - 8_000_000
                    or bound != actors * 128 * seats - 1
                ):
                    raise ValueError("candidate counter accounting mismatch")
            shard_rows = shard.get("rows")
            if (
                not isinstance(shard_rows, list)
                or len(shard_rows) != 128
                or shard.get("rows_sha256") != _digest(shard_rows)
            ):
                raise ValueError("row collection hash/count mismatch")
            for row, world in zip(shard_rows, worlds):
                required = {
                    "schema": evaluate.ROW_SCHEMA,
                    "arm": arm,
                    "training_seed": seed,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                    "b8_checkpoint_sha256": evaluate.B8_SHA256,
                    "world_bank_sha256": bank_sha,
                    "clean_exit": True,
                    "world_seed": world["world_seed"],
                    "mix": world["mix"],
                    "world_semantic_sha256": world["world_semantic_sha256"],
                    "horizon": 5000,
                    "frames_stepped": 5000,
                }
                if not isinstance(row, dict) or any(row.get(k) != v for k, v in required.items()):
                    raise ValueError("row/world identity mismatch")
                for metric in ("mass_integral", "alive_frame_fraction"):
                    x = row.get(metric)
                    if (
                        isinstance(x, bool)
                        or not isinstance(x, (int, float))
                        or not math.isfinite(x)
                        or x < 0
                    ):
                        raise ValueError("invalid metric")
                if row["alive_frame_fraction"] > 1 or type(row.get("no_death_episode")) is not bool:
                    raise ValueError("invalid survival metric")
            indexed[seed, arm] = shard
            rows.extend(shard_rows)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    halted = tripwire_arms or set()
    for seed in bank.TRAINING_SEEDS:
        for arm in (*ARMS, B8):
            if arm in halted:
                continue
            if (seed, arm) not in indexed:
                errors.append(f"missing final identity {arm}/{seed}")
    return rows, sorted(set(errors))


def _effects(
    rows: list[dict[str, Any]], arm: str
) -> tuple[dict[str, Any], dict[int, dict[str, list[float]]]]:
    grouped: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["training_seed"], row["mix"], row["world_seed"]][row["arm"]] = row
    effects = {seed: {mix: [] for mix in MIX_COUNTS} for seed in bank.TRAINING_SEEDS}
    alive = {seed: {mix: [] for mix in MIX_COUNTS} for seed in bank.TRAINING_SEEDS}
    nodeath = {mix: [] for mix in MIX_COUNTS}
    for (seed, mix, _), pair in grouped.items():
        if arm in pair and B8 in pair:
            effects[seed][mix].append(pair[arm]["mass_integral"] - pair[B8]["mass_integral"])
            alive[seed][mix].append(
                pair[arm]["alive_frame_fraction"] - pair[B8]["alive_frame_fraction"]
            )
            nodeath[mix].append(float(pair[arm]["no_death_episode"]))
    seed_effects = {
        seed: mean(mean(effects[seed][mix]) for mix in MIX_COUNTS) for seed in bank.TRAINING_SEEDS
    }
    g = {**t90(list(seed_effects.values())), "seed_effects": list(seed_effects.values())}
    mix_effects = {
        mix: mean(mean(effects[seed][mix]) for seed in bank.TRAINING_SEEDS) for mix in MIX_COUNTS
    }
    alive_effects = {
        mix: mean(mean(alive[seed][mix]) for seed in bank.TRAINING_SEEDS) for mix in MIX_COUNTS
    }
    nodeath_effects = {mix: mean(nodeath[mix]) for mix in MIX_COUNTS}
    return {
        "G": g,
        "seed_effects": seed_effects,
        "mix_mass_differences": mix_effects,
        "alive_fraction_differences": alive_effects,
        "no_death_episode_fraction": nodeath_effects,
        "min_scripted_seed_alive": min(mean(alive[s]["scripted"]) for s in bank.TRAINING_SEEDS),
    }, effects


def _verdict(stats: Mapping[str, Any], tripwire: bool = False) -> tuple[str, list[str]]:
    guards = [
        m for m, x in stats["alive_fraction_differences"].items() if x < -0.03 - GUARD_TOLERANCE
    ]
    if stats["min_scripted_seed_alive"] < -0.15 - GUARD_TOLERANCE:
        guards.append("scripted per-seed alive")
    guards += [
        m + " no-death"
        for m, x in stats["no_death_episode_fraction"].items()
        if x < 0.5 - GUARD_TOLERANCE
    ]
    if tripwire:
        return "KILL", ["training tripwire"]
    if guards or stats["G"]["upper90"] <= 0:
        return "KILL", guards + (["upper90(G) <= 0"] if stats["G"]["upper90"] <= 0 else [])
    if (
        stats["G"]["lower90"] > 0
        and stats["G"]["mean"] >= 20
        and stats["mix_mass_differences"]["scripted"] >= 0
        and sum(x > 0 for x in stats["G"]["seed_effects"]) >= 4
    ):
        return "GO", []
    return "INCONCLUSIVE", []


def _bootstrap(
    effects: Mapping[str, Mapping[int, Mapping[str, list[float]]]], draws: int
) -> dict[str, Any]:
    rng = np.random.default_rng(20261018)
    samples = {"N1": [], "N3": [], "N1-N3": []}
    for _ in range(draws):
        values: dict[str, list[float]] = {a: [] for a in ARMS}
        for seed in rng.choice(bank.TRAINING_SEEDS, 5, replace=True):
            indices = {
                m: rng.integers(
                    0, len(effects["N1"][int(seed)][m]), len(effects["N1"][int(seed)][m])
                )
                for m in MIX_COUNTS
            }
            for arm in ARMS:
                values[arm].append(
                    mean(
                        mean([effects[arm][int(seed)][m][i] for i in indices[m]])
                        for m in MIX_COUNTS
                    )
                )
        averages = {a: mean(values[a]) for a in ARMS}
        samples["N1"].append(averages["N1"])
        samples["N3"].append(averages["N3"])
        samples["N1-N3"].append(averages["N1"] - averages["N3"])
    return {
        "draws": draws,
        "rng_seed": 20261018,
        "method": "resample seed blocks then paired worlds within seed; sensitivity only",
        "central90": {
            k: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
            for k, v in samples.items()
        },
    }


def analyze(value: Mapping[str, Any], bootstrap_draws: int = 2000) -> dict[str, Any]:
    outcomes = value.get("training_outcomes", [])
    outcome_errors: list[str] = []
    tripwire_arms: set[str] = set()
    if not isinstance(outcomes, list):
        outcome_errors.append("training_outcomes must be a list")
    else:
        seen: set[str] = set()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                outcome_errors.append("training outcome is not an object")
                continue
            arm = outcome.get("arm")
            required = {
                "schema": "s1-training-outcome/v1",
                "status": "TRIPWIRE_KILL",
                "arm": arm,
                "b8_sha256": evaluate.B8_SHA256,
            }
            if (
                arm not in ARMS
                or arm in seen
                or any(outcome.get(k) != v for k, v in required.items())
            ):
                outcome_errors.append("invalid or duplicate training outcome")
                continue
            seed = outcome.get("training_seed")
            recipe, counters = outcome.get("recipe"), outcome.get("counters")
            hashes = ("trainer_source_sha256", "summary_sha256", "log_sha256")
            if (
                seed not in bank.TRAINING_SEEDS
                or not isinstance(outcome.get("reason"), str)
                or not isinstance(outcome.get("raw_summary_halt"), (str, dict))
                or not isinstance(recipe, dict)
                or not isinstance(counters, dict)
                or any(not isinstance(outcome.get(h), str) or len(outcome[h]) != 64 for h in hashes)
            ):
                outcome_errors.append("tripwire outcome lacks authenticated receipt bindings")
                continue
            seen.add(arm)
            tripwire_arms.add(arm)
    rows, errors = validate_input(value, tripwire_arms)
    errors = outcome_errors + errors
    if errors:
        return {
            "schema": "s1-analysis/v1",
            "verdict": "INVALID/INCOMPLETE",
            "validation_errors": errors,
            "outcomes_withheld": True,
        }
    arm_stats, effects = {}, {}
    for arm in ARMS:
        if arm in tripwire_arms:
            arm_stats[arm] = {
                "verdict": "KILL",
                "kill_reasons": [
                    outcome["reason"]
                    for outcome in outcomes
                    if isinstance(outcome, dict) and outcome.get("arm") == arm
                ],
                "tripwire_outcome": True,
            }
            continue
        stats, effect = _effects(rows, arm)
        verdict, reasons = _verdict(stats)
        arm_stats[arm] = {**stats, "verdict": verdict, "kill_reasons": reasons}
        effects[arm] = effect
    contrast = (
        t90(
            [
                arm_stats["N1"]["seed_effects"][s] - arm_stats["N3"]["seed_effects"][s]
                for s in bank.TRAINING_SEEDS
            ]
        )
        if not tripwire_arms
        else {"status": "UNAVAILABLE", "reason": "tripwire KILL prevents paired recipe contrast"}
    )
    # Integrity is study-wide.  Efficacy outcomes are arm-local: a KILL closes
    # only that recipe and cannot erase a separately qualified GO that buys its
    # own fresh Mac strict-preparation step.
    kills = [a for a in ARMS if arm_stats[a]["verdict"] == "KILL"]
    gos = [a for a in ARMS if arm_stats[a]["verdict"] == "GO"]
    ranking = sorted(
        gos,
        key=lambda a: (
            -arm_stats[a]["G"]["lower90"],
            -arm_stats[a]["G"]["mean"],
            0 if a == "N1" else 1,
        ),
    )
    verdict = "GO" if gos else ("KILL" if kills else "INCONCLUSIVE")
    return {
        "schema": "s1-analysis/v1",
        "verdict": verdict,
        "selected_recipe": ranking[0] if ranking else None,
        "go_ranking": ranking,
        "arms": arm_stats,
        "paired_n1_minus_n3": contrast,
        "hierarchical_world_within_seed_bootstrap": (
            _bootstrap(effects, bootstrap_draws)
            if not tripwire_arms
            else {"status": "UNAVAILABLE", "reason": "tripwire KILL prevents paired bootstrap"}
        ),
        "validation_errors": [],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument(
        "--independent-audit",
        type=Path,
        required=True,
        help="PASS s1-independent-audit/v1 bound to the exact input file bytes",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    a = p.parse_args(argv)
    input_bytes = a.input.read_bytes()
    audit = json.loads(a.independent_audit.read_text())
    if (
        audit.get("schema") != "s1-independent-audit/v1"
        or audit.get("status") != "PASS"
        or audit.get("analysis_input_sha256") != hashlib.sha256(input_bytes).hexdigest()
    ):
        p.error("--independent-audit must PASS and bind the exact analysis input bytes")
    result = analyze(json.loads(input_bytes), a.bootstrap_draws)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    a.out.write_text(text) if a.out else print(text)
    return 2 if result["verdict"] == "INVALID/INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
