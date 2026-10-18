"""Analyze the preregistered T23 R0 paired H5000 learning screen.

The input is a ``t23-eval-manifest/v1`` object with ``rows`` (or a JSON/JSONL
file of ``t23-eval-row/v1`` records).  A row is one terminal-hero result; the
candidate and released-B8 rows share the same training seed, checkpoint step,
mix, and world id.  This program deliberately treats incomplete operational
evidence as INVALID rather than estimating around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.scripts.eval_stats import student_t_isf

SEEDS = (7101, 7102, 7103, 7104, 7105)
STEPS = (2_000_000, 4_000_000, 8_000_000)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
SCHEMA = "t23-analysis/v1"


def _sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def _one_sided_90(values: list[float]) -> dict[str, float | int]:
    if len(values) != 5:
        raise ValueError("T23 requires exactly five independent seed effects")
    mean = _mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(5)
    critical = student_t_isf(0.10, 4)
    return {
        "n": 5,
        "df": 4,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "critical_one_sided_90": critical,
        "lower90": mean - critical * se,
        "upper90": mean + critical * se,
    }


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("training_seed")),
            str(row.get("checkpoint_step")),
            str(row.get("mix")),
            str(row.get("world_id")),
            str(row.get("arm")),
        ),
    )
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    """Read native atomic receipts; preserve invalid evidence in an INVALID report."""
    if not path.is_dir():
        return {
            "rows": [],
            "ingest_errors": ["CLI analysis requires the native evaluation receipt directory"],
        }
    from research.t23_r0_screen_20261017 import bank as bank_module
    from research.t23_r0_screen_20261017 import evaluate as evaluator

    registered = bank_module.load_world_bank()
    registered_sha = bank_module.canonical_sha(registered)
    dependency_sha = hashlib.sha256(
        evaluator.canonical(evaluator.source_manifest()).encode()
    ).hexdigest()
    evaluator_sha = hashlib.sha256(Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()
    rows, initial_rows, errors = [], [], []
    tripwires: dict[str, bool] = {}
    receipts: dict[str, str] = {}
    initial_receipts: dict[str, str] = {}
    phases, banks, profiles, sources = set(), set(), set(), set()
    for shard in sorted(path.rglob("*.json")):
        try:
            value = json.loads(shard.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema") != "t23-eval-shard/v1":
                continue
            relative = str(shard.relative_to(path))
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            receipts[relative] = digest
            identity = value.get("identity", {})
            provenance = value.get("training_provenance", {})
            evidence = value.get("tripwire_evidence", {})
            seed, phase = identity.get("training_seed"), identity.get("phase")
            if seed not in SEEDS or phase not in ("initial", "extension"):
                raise ValueError("invalid seed/phase")
            triggered = evidence.get("triggered")
            if not isinstance(triggered, bool):
                raise ValueError("missing typed training tripwire evidence")
            tripwires[str(seed)] = tripwires.get(str(seed), False) or triggered
            if value.get("status") != "completed" or value.get("clean_exit") is not True:
                raise ValueError("incomplete/unclean evaluator receipt")
            for field in ("summary", "manifest", "telemetry", "run_log"):
                if (
                    not _sha(provenance.get(f"{field}_sha256"))
                    or identity.get(f"training_{field}_sha256") != provenance[f"{field}_sha256"]
                ):
                    raise ValueError(f"missing or mismatched training {field} provenance")
            if (
                identity.get("training_tripwire_triggered") is not triggered
                or provenance.get("tripwire_triggered") is not triggered
            ):
                raise ValueError("tripwire disposition disagrees with bound provenance")
            if identity.get("evaluation_dependencies_sha256") != dependency_sha:
                raise ValueError("evaluation dependency source drift")
            if provenance.get("clean") is not True:
                raise ValueError("unclean training evidence")
            if evidence.get("passed") is not (not triggered):
                raise ValueError("inconsistent tripwire evidence")
            step = {"2000k": 2_000_000, "4000k": 4_000_000, "8000k": 8_000_000}.get(
                identity.get("checkpoint_label")
            )
            if step is None:
                raise ValueError("invalid checkpoint milestone")
            for key in (
                "checkpoint_sha256",
                "bank_sha256",
                "profile_digest",
                "evaluator_source_sha256",
            ):
                if not _sha(identity.get(key)):
                    raise ValueError(f"missing identity hash {key}")
            if identity.get("b8_sha256") != B8_SHA256:
                raise ValueError("wrong B8 checkpoint")
            if (
                value.get("bank_sha256") != identity["bank_sha256"]
                or value.get("profile_sha256") != identity["profile_digest"]
            ):
                raise ValueError("receipt bank/profile binding disagrees")
            shard_rows = value.get("rows")
            if not isinstance(shard_rows, list) or len(shard_rows) != 2 * identity.get(
                "world_count", -1
            ):
                raise ValueError("paired sample-count mismatch")
            canonical_rows = json.dumps(
                shard_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if value.get("rows_sha256") != hashlib.sha256(canonical_rows).hexdigest():
                raise ValueError("row digest mismatch")
            if (
                identity["bank_sha256"] != registered_sha
                or identity["evaluator_source_sha256"] != evaluator_sha
                or identity["profile_digest"] != registered["profile"]["base_profile_digest"]
            ):
                raise ValueError("receipt not bound to the frozen T23 bank/evaluator")
            shard_index, shard_count = identity.get("shard"), identity.get("shards")
            if (
                type(shard_index) is not int
                or type(shard_count) is not int
                or not 0 <= shard_index < shard_count
            ):
                raise ValueError("invalid shard allocation")
            expected_worlds = [
                w
                for i, w in enumerate(
                    sorted(
                        bank_module.worlds_for_seed(registered, seed, phase),
                        key=lambda w: (w["mix"], w["world_seed"]),
                    )
                )
                if i % shard_count == shard_index
            ]
            expected_map = {w["world_semantic_sha256"]: w for w in expected_worlds}
            declared = identity.get("world_hashes")
            if declared != list(expected_map):
                raise ValueError("declared worlds differ from the registered shard")
            if (
                not isinstance(declared, list)
                or len(declared) != identity["world_count"]
                or len(set(declared)) != len(declared)
            ):
                raise ValueError("invalid declared world allocation")
            seen = set()
            for row in shard_rows:
                if not isinstance(row, dict) or row.get("schema") != "t23-eval-row/v1":
                    raise ValueError("invalid row schema")
                arm, world = row.get("arm"), row.get("world_semantic_sha256")
                if arm not in ("candidate", "b8") or world not in declared or (arm, world) in seen:
                    raise ValueError("row not in exact paired world allocation")
                seen.add((arm, world))
                expected_world = expected_map[world]
                for row_key, bank_key in (
                    ("world_seed", "world_seed"),
                    ("mix", "mix"),
                    ("initial_world_sha256", "initial_world_sha256"),
                    ("opponent_specs_sha256", "roster_semantic_sha256"),
                ):
                    if row.get(row_key) != expected_world[bank_key]:
                        raise ValueError(f"row {row_key} differs from registered world")
                expected = {
                    "training_seed": seed,
                    "checkpoint_step": step,
                    "candidate_checkpoint_sha256": (
                        identity["checkpoint_sha256"] if arm == "candidate" else B8_SHA256
                    ),
                    "b8_checkpoint_sha256": B8_SHA256,
                    "world_bank_sha256": identity["bank_sha256"],
                    "world_profile_sha256": identity["profile_digest"],
                    "evaluator_source_sha256": identity["evaluator_source_sha256"],
                    "world_id": world,
                }
                if any(row.get(key) != expected_value for key, expected_value in expected.items()):
                    raise ValueError(
                        "row seed/checkpoint/world/source identity disagrees with receipt"
                    )
            if seen != {(arm, world) for arm in ("candidate", "b8") for world in declared}:
                raise ValueError("missing paired world")
            phases.add(phase)
            banks.add(identity["bank_sha256"])
            profiles.add(identity["profile_digest"])
            sources.add(identity["evaluator_source_sha256"])
            rows.extend(shard_rows)
            if phase == "initial":
                initial_rows.extend(shard_rows)
                initial_receipts[relative] = digest
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(f"{shard}: {error}")
    if len(banks) > 1 or len(profiles) > 1 or len(sources) > 1:
        errors.append("bank/profile/evaluator source drift across shards")
    extension = int("extension" in phases)
    manifest = {
        "schema": "t23-eval-manifest/v1",
        "rows": rows,
        "tripwires": tripwires,
        "extension_round": extension,
        "shard_receipt_sha256s": receipts,
        "initial_rows_sha256": _rows_digest(initial_rows),
        "initial_shard_receipt_sha256s": initial_receipts,
        "world_bank_sha256s": sorted(banks),
        "ingest_errors": errors,
    }
    if extension:
        try:
            authorization = json.loads((path / "extension_authorization.json").read_text())
            prior = authorization["prior_result"]
            prior_digest = hashlib.sha256(
                json.dumps(prior, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            if (
                authorization.get("prior_result_sha256") != prior_digest
                or prior.get("verdict") != "EXTEND"
                or prior.get("extension_round") != 0
            ):
                raise ValueError("extension is not bound to the first EXTEND verdict")
            for key in (
                "initial_rows_sha256",
                "initial_shard_receipt_sha256s",
                "world_bank_sha256s",
            ):
                if prior.get(key) != manifest[key]:
                    raise ValueError(f"extension changed original evidence: {key}")
            if authorization.get("frozen_checkpoint_sha256s") != prior.get(
                "frozen_checkpoint_sha256s"
            ):
                raise ValueError("extension frozen checkpoint map disagrees with prior result")
            manifest["extension_authorization"] = authorization
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"extension authorization: {error}")
    return manifest


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _validate(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], int, dict[int, bool]]:
    errors: list[str] = []
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return [], ["manifest rows must be a list"], 0, {}
    extension = manifest.get("extension_round", 0)
    if extension not in (0, 1):
        _error(errors, "extension_round must be 0 or 1 (extension is allowed exactly once)")
        extension = 0
    if extension == 1:
        authorization = manifest.get("extension_authorization")
        if not isinstance(authorization, dict) or authorization.get("prior_verdict") != "EXTEND":
            _error(errors, "extension round requires a bound prior EXTEND authorization")
        elif not isinstance(authorization.get("frozen_checkpoint_sha256s"), dict):
            _error(errors, "extension authorization lacks frozen checkpoint hashes")
    if manifest.get("schema", "t23-eval-manifest/v1") != "t23-eval-manifest/v1":
        _error(errors, "wrong manifest schema")
    for ingest_error in manifest.get("ingest_errors", []):
        _error(errors, f"receipt ingestion: {ingest_error}")
    if "rows_sha256" in manifest:
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        if manifest["rows_sha256"] != hashlib.sha256(canonical).hexdigest():
            _error(errors, "rows_sha256 mismatch")
    tripwires = manifest.get("tripwires")
    if not isinstance(tripwires, dict):
        _error(errors, "missing or malformed seed-keyed tripwires evidence")
        tripwires = {}
    normalized_tripwires: dict[int, bool] = {}
    for seed in SEEDS:
        if str(seed) not in tripwires and seed not in tripwires:
            _error(errors, f"missing tripwire evidence for seed {seed}")
        value = tripwires.get(str(seed), tripwires.get(seed, False))
        if isinstance(value, dict):
            value = value.get("triggered", False)
        if not isinstance(value, bool):
            _error(errors, f"tripwire for seed {seed} is not boolean")
        normalized_tripwires[seed] = bool(value)

    required = (
        "training_seed",
        "checkpoint_step",
        "arm",
        "mix",
        "world_id",
        "world_seed",
        "horizon",
        "frames_stepped",
        "clean_exit",
        "candidate_checkpoint_sha256",
        "b8_checkpoint_sha256",
        "world_bank_sha256",
        "world_profile_sha256",
        "initial_world_sha256",
        "mass_integral",
        "alive_frame_fraction",
        "no_death_episode",
    )
    seen: set[tuple[Any, ...]] = set()
    candidate_hashes: dict[tuple[int, int], str] = {}
    seed_worlds: dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _error(errors, f"row {index} is not an object")
            continue
        if row.get("schema", "t23-eval-row/v1") != "t23-eval-row/v1":
            _error(errors, f"row {index} wrong schema")
        for key in required:
            if key not in row:
                _error(errors, f"row {index} missing {key}")
        seed, step, arm, mix = (
            row.get("training_seed"),
            row.get("checkpoint_step"),
            row.get("arm"),
            row.get("mix"),
        )
        if seed not in SEEDS:
            _error(errors, f"row {index} invalid training_seed")
        if step not in STEPS:
            _error(errors, f"row {index} invalid checkpoint_step")
        if arm not in ("candidate", "b8"):
            _error(errors, f"row {index} invalid arm")
        if mix not in MIX_COUNTS:
            _error(errors, f"row {index} invalid mix")
        if (
            row.get("horizon") != 5000
            or not isinstance(row.get("frames_stepped"), int)
            or not 0 <= row["frames_stepped"] <= 5000
        ):
            _error(errors, f"row {index} invalid H5000 horizon/frame accounting")
        if row.get("clean_exit") is not True:
            _error(errors, f"row {index} unclean exit")
        if not all(
            _sha(row.get(key))
            for key in (
                "candidate_checkpoint_sha256",
                "b8_checkpoint_sha256",
                "world_bank_sha256",
                "world_profile_sha256",
                "initial_world_sha256",
            )
        ):
            _error(errors, f"row {index} has missing or malformed hash")
        if row.get("b8_checkpoint_sha256") != B8_SHA256:
            _error(errors, f"row {index} B8 identity mismatch")
        if not _finite(row.get("mass_integral")):
            _error(errors, f"row {index} nonfinite mass_integral")
        if (
            not _finite(row.get("alive_frame_fraction"))
            or not 0 <= float(row.get("alive_frame_fraction", -1)) <= 1
        ):
            _error(errors, f"row {index} invalid alive_frame_fraction")
        if not isinstance(row.get("no_death_episode"), bool):
            _error(errors, f"row {index} invalid no_death_episode")
        key = (seed, step, mix, row.get("world_id"), arm)
        if key in seen:
            _error(errors, f"duplicate row identity {key}")
        seen.add(key)
        if seed in SEEDS and isinstance(row.get("initial_world_sha256"), str):
            seed_worlds[seed].add(row["initial_world_sha256"])
        if (
            arm == "candidate"
            and seed in SEEDS
            and step in STEPS
            and _sha(row.get("candidate_checkpoint_sha256"))
        ):
            old = candidate_hashes.setdefault((seed, step), row["candidate_checkpoint_sha256"])
            if old != row["candidate_checkpoint_sha256"]:
                _error(errors, f"candidate hash drift seed {seed} step {step}")
    if extension == 1 and isinstance(manifest.get("extension_authorization"), dict):
        frozen = manifest["extension_authorization"].get("frozen_checkpoint_sha256s", {})
        for (seed, step), digest in candidate_hashes.items():
            expected = frozen.get(f"{seed}:{step}")
            if expected != digest:
                _error(errors, f"extension checkpoint not frozen seed {seed} step {step}")
    for left, right in combinations(SEEDS, 2):
        if seed_worlds[left] & seed_worlds[right]:
            _error(errors, f"world collision between training seeds {left}/{right}")
    return rows, errors, extension, normalized_tripwires


def _pair(
    rows: list[dict[str, Any]], errors: list[str], extension: int
) -> dict[int, dict[int, dict[str, list[dict[str, Any]]]]]:
    grouped: dict[tuple[int, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if (
            row.get("training_seed") in SEEDS
            and row.get("checkpoint_step") in STEPS
            and row.get("mix") in MIX_COUNTS
        ):
            grouped[
                (row["training_seed"], row["checkpoint_step"], row["mix"], str(row.get("world_id")))
            ][row.get("arm")] = row
    paired: dict[int, dict[int, dict[str, list[dict[str, Any]]]]] = {
        s: {p: {m: [] for m in MIX_COUNTS} for p in STEPS} for s in SEEDS
    }
    for identity, arms in grouped.items():
        seed, step, mix, world = identity
        if set(arms) != {"candidate", "b8"}:
            _error(errors, f"unpaired candidate/B8 world {seed}/{step}/{mix}/{world}")
            continue
        candidate, b8 = arms["candidate"], arms["b8"]
        for key in (
            "world_seed",
            "world_bank_sha256",
            "world_profile_sha256",
            "initial_world_sha256",
        ):
            if candidate.get(key) != b8.get(key):
                _error(errors, f"pair identity mismatch {key} at {seed}/{step}/{mix}/{world}")
        paired[seed][step][mix].append(
            {
                "mass": float(candidate["mass_integral"]) - float(b8["mass_integral"]),
                "alive": float(candidate["alive_frame_fraction"])
                - float(b8["alive_frame_fraction"]),
                "candidate_no_death": bool(candidate["no_death_episode"]),
                "world_id": world,
                "candidate_mass": float(candidate["mass_integral"]),
                "b8_mass": float(b8["mass_integral"]),
                "candidate_alive": float(candidate["alive_frame_fraction"]),
                "b8_alive": float(b8["alive_frame_fraction"]),
                "b8_no_death": bool(b8["no_death_episode"]),
            }
        )
    multiplier = extension + 1
    for seed in SEEDS:
        for step in STEPS:
            for mix, base in MIX_COUNTS.items():
                actual = len(paired[seed][step][mix])
                # 2M/4M are descriptive: an entirely omitted descriptive
                # checkpoint never invalidates the registered 8M decision,
                # while a partially supplied descriptive checkpoint is invalid.
                if step != 8_000_000 and not any(
                    paired[s][step][m] for s in SEEDS for m in MIX_COUNTS
                ):
                    continue
                step_multiplier = multiplier
                if extension and step != 8_000_000:
                    first_count = len(paired[SEEDS[0]][step]["scripted"])
                    step_multiplier = 1 if first_count == MIX_COUNTS["scripted"] else 2
                if actual != base * step_multiplier:
                    _error(
                        errors,
                        f"sample-count mismatch seed {seed} step {step} {mix}: "
                        f"{actual} != {base * step_multiplier}",
                    )
    return paired


def _bootstrap(
    seed_mix_mass: dict[int, dict[str, list[float]]], draws: int = 2000
) -> dict[str, Any]:
    """Descriptive hierarchical paired bootstrap: seed then paired worlds within mix."""
    rng = np.random.default_rng(20261017)
    samples = []
    for _ in range(draws):
        effects = []
        for seed in rng.choice(SEEDS, size=5, replace=True):
            means = [
                _mean(
                    rng.choice(
                        seed_mix_mass[int(seed)][mix],
                        size=len(seed_mix_mass[int(seed)][mix]),
                        replace=True,
                    )
                )
                for mix in MIX_COUNTS
            ]
            effects.append(_mean(means))
        samples.append(_mean(effects))
    return {
        "method": (
            "hierarchical paired bootstrap; resample training seeds "
            "then paired worlds within each mix"
        ),
        "seed": 20261017,
        "draws": draws,
        "ci90_percentile": [float(x) for x in np.quantile(samples, [0.05, 0.95])],
    }


def analyze(manifest: dict[str, Any], bootstrap_draws: int = 2000) -> dict[str, Any]:
    rows, errors, extension, tripwires = _validate(manifest)
    # Validation is total and fail-closed: malformed rows must yield an
    # INVALID receipt, never an exception or a partially computed decision.
    if errors:
        return {
            "schema": SCHEMA,
            "extension_round": extension,
            "validation_errors": errors,
            "verdict": "INVALID",
            "only_8m_decisive": True,
            "training_disposition": "KILL" if any(tripwires.values()) else None,
            "training_disposition_note": (
                "A registered training tripwire was observed, but missing or unclean "
                "evaluation evidence makes the overall screen INVALID rather than an efficacy KILL."
                if any(tripwires.values())
                else None
            ),
        }
    paired = _pair(rows, errors, extension)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "extension_round": extension,
        "validation_errors": errors,
        "verdict": "INVALID" if errors else None,
        "only_8m_decisive": True,
    }
    if errors:
        return result
    result.update(
        {
            "frozen_checkpoint_sha256s": {
                f"{row['training_seed']}:{row['checkpoint_step']}": row[
                    "candidate_checkpoint_sha256"
                ]
                for row in rows
                if row["arm"] == "candidate"
            },
            "initial_rows_sha256": manifest.get("initial_rows_sha256", _rows_digest(rows)),
            "initial_shard_receipt_sha256s": manifest.get("initial_shard_receipt_sha256s", {}),
            "world_bank_sha256s": manifest.get(
                "world_bank_sha256s", sorted({row["world_bank_sha256"] for row in rows})
            ),
        }
    )
    all_steps: dict[str, Any] = {}
    for step in STEPS:
        if not any(paired[seed][step][mix] for seed in SEEDS for mix in MIX_COUNTS):
            all_steps[str(step)] = {"descriptive_only": True, "available": False}
            continue
        seed_mix_mass = {
            seed: {mix: [x["mass"] for x in paired[seed][step][mix]] for mix in MIX_COUNTS}
            for seed in SEEDS
        }
        effects = [_mean([_mean(seed_mix_mass[seed][mix]) for mix in MIX_COUNTS]) for seed in SEEDS]
        mixes = {}
        for mix in MIX_COUNTS:
            episodes = [x for seed in SEEDS for x in paired[seed][step][mix]]
            mass = [x["mass"] for x in episodes]
            alive = [x["alive"] for x in episodes]
            mixes[mix] = {
                "count": len(episodes),
                "mass_point_delta": _mean(mass),
                "paired_mass_variance": statistics.variance(mass),
                "alive_frame_point_delta": _mean(alive),
                "candidate_no_death_survival": _mean(
                    [float(x["candidate_no_death"]) for x in episodes]
                ),
            }
        for mix in MIX_COUNTS:
            entries = [x for seed in SEEDS for x in paired[seed][step][mix]]
            mixes[mix].update(
                {
                    key: _mean(float(x[key]) for x in entries)
                    for key in (
                        "candidate_mass",
                        "b8_mass",
                        "candidate_alive",
                        "b8_alive",
                        "candidate_no_death",
                        "b8_no_death",
                    )
                }
            )
            mixes[mix]["paired_alive_variance"] = statistics.variance(x["alive"] for x in entries)
            mixes[mix]["paired_episode_variance"] = statistics.variance(
                float(x["candidate_no_death"]) - float(x["b8_no_death"]) for x in entries
            )
        seed_mix_summary = {
            str(seed): {
                mix: {
                    "count": len(paired[seed][step][mix]),
                    "mass_delta": _mean(x["mass"] for x in paired[seed][step][mix]),
                    "alive_delta": _mean(x["alive"] for x in paired[seed][step][mix]),
                    "candidate_no_death": _mean(
                        float(x["candidate_no_death"]) for x in paired[seed][step][mix]
                    ),
                    "paired_mass_variance": statistics.variance(
                        x["mass"] for x in paired[seed][step][mix]
                    ),
                }
                for mix in MIX_COUNTS
            }
            for seed in SEEDS
        }
        all_steps[str(step)] = {
            "descriptive_only": step != 8_000_000,
            "available": True,
            "seed_effects_equal_mix": dict(zip(map(str, SEEDS), effects)),
            "G": _one_sided_90(effects),
            "mixes": mixes,
            "per_seed_mix": seed_mix_summary,
            "bootstrap_sensitivity": _bootstrap(seed_mix_mass, bootstrap_draws),
        }
    decisive = all_steps[str(8_000_000)]
    if all_steps["4000000"].get("available"):
        result["descriptive_8m_minus_4m_seed_effects"] = {
            seed: value - all_steps["4000000"]["seed_effects_equal_mix"][seed]
            for seed, value in decisive["seed_effects_equal_mix"].items()
        }
    result["low_boost_floor"] = "unevaluable: boost-eligible support counts were not instrumented"

    kill_reasons = [f"training tripwire seed {seed}" for seed, hit in tripwires.items() if hit]
    for mix, details in decisive["mixes"].items():
        if details["alive_frame_point_delta"] < -0.03 - 1e-12:
            kill_reasons.append(f"alive-frame guard failed {mix}")
        if details["candidate_no_death_survival"] < 0.50:
            kill_reasons.append(f"no-death survival guard failed {mix}")
    # The -0.15 per-seed scripted guard is alive-frame fraction (section 6's
    # point survival guard), not the mass effect.  Scripted mass is a GO-only
    # point condition below.
    for seed in SEEDS:
        scripted_alive = _mean([x["alive"] for x in paired[seed][8_000_000]["scripted"]])
        if scripted_alive < -0.15 - 1e-12:
            kill_reasons.append(f"seed {seed} scripted alive-frame guard failed")
    g = decisive["G"]
    if g["upper90"] <= 0:
        kill_reasons.append("upper90(G) <= 0")
    go = (
        g["lower90"] > 0
        and g["mean"] >= 20
        and decisive["mixes"]["scripted"]["mass_point_delta"] >= 0
        and sum(value > 0 for value in decisive["seed_effects_equal_mix"].values()) >= 4
        and not kill_reasons
    )
    if kill_reasons:
        verdict = "KILL"
    elif go:
        verdict = "GO"
    elif extension == 0:
        verdict = "EXTEND"
    else:
        verdict = "INCONCLUSIVE"
    result.update(
        {
            "steps": all_steps,
            "tripwires": tripwires,
            "kill_reasons": kill_reasons,
            "go_conditions": {
                "lower90_positive": g["lower90"] > 0,
                "mean_G_at_least_20": g["mean"] >= 20,
                "scripted_mass_nonnegative": decisive["mixes"]["scripted"]["mass_point_delta"] >= 0,
                "at_least_four_positive_seed_effects": sum(
                    value > 0 for value in decisive["seed_effects_equal_mix"].values()
                )
                >= 4,
            },
            "verdict": verdict,
        }
    )
    return result


def markdown(result: dict[str, Any]) -> str:
    """Human-readable counterpart of the complete seed-level JSON verdict."""
    lines = [
        "# T23 R0 analysis",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        "Only 8M decides. GO permits long-run planning; it is not a strict pass or release.",
        "",
    ]
    if result["validation_errors"]:
        lines += ["## Invalid or incomplete evidence", ""] + [
            f"- {error}" for error in result["validation_errors"]
        ]
        if result.get("training_disposition"):
            lines += [
                "",
                f"Training disposition: {result['training_disposition']}. "
                "Efficacy evidence is incomplete.",
            ]
        return "\n".join(lines) + "\n"
    decisive = result["steps"]["8000000"]
    g = decisive["G"]
    lines += [
        f"G = {g['mean']:.6g}; one-sided 90% Student-t limits (df=4): "
        f"lower {g['lower90']:.6g}, upper {g['upper90']:.6g}.",
        "",
        "| Training seed | Equal-mix G | Scripted Δalive |",
        "|---|---:|---:|",
    ]
    for seed, effect in decisive["seed_effects_equal_mix"].items():
        lines.append(
            f"| {seed} | {effect:.6g} | "
            f"{decisive['per_seed_mix'][seed]['scripted']['alive_delta']:.6g} |"
        )
    lines += [
        "",
        "| Mix | Paired worlds | Candidate mass | B8 mass | Δmass | "
        "Candidate alive | B8 alive | Candidate no-death | B8 no-death |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mix, row in decisive["mixes"].items():
        lines.append(
            f"| {mix} | {row['count']} | {row['candidate_mass']:.6g} | {row['b8_mass']:.6g} | "
            f"{row['mass_point_delta']:.6g} | {row['candidate_alive']:.6g} | "
            f"{row['b8_alive']:.6g} | {row['candidate_no_death']:.6g} | "
            f"{row['b8_no_death']:.6g} |"
        )
    lines += ["", "## Decision evidence", ""]
    lines += [f"- {reason}" for reason in result["kill_reasons"]] or [
        "- All registered KILL guards passed."
    ]
    lines += [
        f"- {name}: {'pass' if passed else 'fail'}"
        for name, passed in result["go_conditions"].items()
    ]
    lines += [
        "",
        "## Descriptive checkpoints",
        "",
        "| Checkpoint | Mean equal-mix G | Role |",
        "|---|---:|---|",
    ]
    for step, data in result["steps"].items():
        mean = f"{data['G']['mean']:.6g}" if data.get("available") else "unavailable"
        lines.append(
            f"| {int(step) // 1000000}M | {mean} | "
            f"{'descriptive' if data['descriptive_only'] else 'decisive'} |"
        )
    lines += [
        "",
        f"Extension round: {result['extension_round']} (at most one).",
        "Hierarchical paired-bootstrap sensitivity: "
        f"{decisive['bootstrap_sensitivity']['ci90_percentile']} (descriptive).",
        "Per-seed/mix paired variances and frozen artifact identities are retained in the JSON.",
        f"Low-boost floor: {result['low_boost_floor']}.",
    ]
    if result.get("descriptive_8m_minus_4m_seed_effects"):
        lines += [
            "",
            "8M minus 4M seed effects: "
            + ", ".join(
                f"{seed}: {effect:.6g}"
                for seed, effect in result["descriptive_8m_minus_4m_seed_effects"].items()
            )
            + ".",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    result = analyze(_load(args.input), args.bootstrap_draws)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "t23_verdict.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "t23_verdict.md").write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {"verdict": result["verdict"], "validation_errors": result["validation_errors"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
