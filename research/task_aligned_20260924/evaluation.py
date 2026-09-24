"""Bounded, same-contract package evaluation; no simulator or checkpoint imports.

The adapter owns all gameplay and policy operations. This module validates its
physical event stream, saves per-world facts, and analyzes only named reports.
Running a CLI evaluation is numerical work and requires controller admission.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

STUDY_ID = "task-aligned-20260924"
LINEAGES = (2026096101, 2026096102, 2026096103)
COUNTERS = (
    "native_frames",
    "hero_transitions",
    "model_forwards",
    "checkpoint_loads",
    "optimizer_updates",
)
METRICS = (
    "mass_integral",
    "ambient_food",
    "survival_fraction",
    "endpoint",
    "boost_fraction",
    "encounter_frames",
)
T975_DF31 = 2.0395134463964077


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON, rejecting nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    if any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _counts(values: Mapping[str, Any]) -> dict[str, int]:
    if set(values) != set(COUNTERS):
        raise ValueError("exact physical-counter schema required")
    result = {key: _integer(values[key], key) for key in COUNTERS}
    if result["optimizer_updates"]:
        raise ValueError("evaluation cannot update a learner")
    return result


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Validate the fixed scientific scope without opening any artifacts."""
    if spec.get("study_id") != STUDY_ID or spec.get("schema_version") != 1:
        raise ValueError("wrong study/schema")
    if spec.get("horizon") != 3000 or spec.get("lineages") != list(LINEAGES):
        raise ValueError("evaluation fixes H3000 and all three CZ lineages")
    _digest(spec.get("contract_sha256"), "contract_sha256")
    profiles = spec.get("profiles", {})
    if set(profiles) != {"solo", "food_pressure", "mixed"} or profiles["solo"] != []:
        raise ValueError("exact three-profile scope required")
    for name in ("food_pressure", "mixed"):
        if len(profiles[name]) != 5 or set(profiles[name]) != {"greedy_food", "random_safe"}:
            raise ValueError("opponent profiles require five declared scripted opponents")
    if profiles["food_pressure"] == profiles["mixed"]:
        raise ValueError("opponent mixtures must differ")
    worlds = spec.get("world_seeds", {})
    if set(worlds) != set(profiles):
        raise ValueError("world seeds required for every profile")
    flat = []
    for name, seeds in worlds.items():
        if len(seeds) != 32 or len(set(seeds)) != 32:
            raise ValueError(f"{name} requires exactly 32 unique world families")
        flat.extend(_integer(seed, "world_seed") for seed in seeds)
    if len(set(flat)) != len(flat):
        raise ValueError("profile world namespaces must be disjoint")
    namespaces = spec.get("training_seed_namespaces")
    held_namespace = spec.get("held_seed_namespace")
    if (
        not isinstance(namespaces, list)
        or not namespaces
        or any(not isinstance(name, str) or not name for name in namespaces)
        or not isinstance(held_namespace, str)
        or not held_namespace
        or held_namespace in namespaces
    ):
        raise ValueError("explicit distinct training/held seed namespaces are required")
    training = spec.get("training_world_seeds")
    if not isinstance(training, list) or not training:
        raise ValueError("actual bounded training world seed IDs are required")
    training_ids = {_integer(seed, "training world seed") for seed in training}
    if set(flat) & training_ids:
        raise ValueError("actual training and held world IDs overlap")
    if spec.get("novelty_scope") != "declared_seed_namespaces_only":
        raise ValueError("no implicit geometry or all-history novelty claim")
    if spec.get("decision") != {
        "relative_mass_gain": 0.10,
        "absolute_mass_gain": 1.0,
        "solo_mass_food_loss": 0.05,
        "solo_survival_loss": 0.02,
        "minimum_exposed_worlds": 8,
    }:
        raise ValueError("decision tolerances differ from reviewed package scope")


@dataclasses.dataclass(frozen=True)
class EpisodeRequest:
    """One procedural initial world; no outcome-selected food placement."""

    world_seed: int
    profile: str
    opponents: tuple[str, ...]
    horizon: int


@dataclasses.dataclass(frozen=True)
class FrameMetrics:
    """Post-move physical hero facts; encounter is pre-action L1 <= 16."""

    alive: bool
    mass: float
    ambient_food: int
    boost: bool
    encounter: bool


@dataclasses.dataclass
class EpisodeTrace:
    """Adapter result, finite at hero death or at the requested horizon."""

    frames: Iterable[FrameMetrics]
    initial_descriptor: Mapping[str, Any]
    counters: Mapping[str, int]


@dataclasses.dataclass
class EvaluationRuntime:
    """Created only inside an admitted numerical process by the adapter."""

    run_episode: Callable[[EpisodeRequest], EpisodeTrace]
    contract_sha256: str
    initial_counters: Mapping[str, int]
    run_batch: (
        Callable[[Sequence[EpisodeRequest]], tuple[Sequence[EpisodeTrace], Mapping[str, int]]]
        | None
    ) = None


def _reduce_episode(request: EpisodeRequest, trace: EpisodeTrace) -> dict[str, Any]:
    required = {"world_seed", "hero", "opponents", "food", "rng_sha256"}
    if not required <= set(trace.initial_descriptor):
        raise ValueError("initial descriptor lacks geometry, food, or RNG identity")
    if trace.initial_descriptor["world_seed"] != request.world_seed:
        raise ValueError("adapter returned the wrong world")
    _digest(trace.initial_descriptor["rng_sha256"], "initial RNG digest")
    mass_sum = 0.0
    food = alive_frames = boost_frames = encounter_frames = steps = 0
    dead = False
    columns = {name: [] for name in ("alive", "mass", "ambient_food", "boost", "encounter")}
    for frame in trace.frames:
        if dead or steps >= request.horizon:
            raise ValueError("trace continued after terminal hero death or horizon")
        if not all(type(v) is bool for v in (frame.alive, frame.boost, frame.encounter)):
            raise ValueError("frame flags must be bool")
        mass = _finite(frame.mass, "physical mass")
        if not frame.alive and mass != 0:
            raise ValueError("terminal frame mass must contribute zero")
        steps += 1
        mass_sum += mass
        food += _integer(frame.ambient_food, "physical ambient food events")
        alive_frames += int(frame.alive)
        boost_frames += int(frame.boost)
        encounter_frames += int(frame.encounter and steps > 16)
        for name in columns:
            columns[name].append(getattr(frame, name))
        dead = not frame.alive
    if steps == 0 or (not dead and steps != request.horizon):
        raise ValueError("trace ended early without a terminal death")
    counters = _counts(trace.counters)
    if counters["native_frames"] != steps or counters["hero_transitions"] != steps:
        raise ValueError("trace physical frame/valid-hero-transition counters disagree")
    return {
        "world_seed": request.world_seed,
        "profile": request.profile,
        "horizon": request.horizon,
        "observed_frames": steps,
        "initial_descriptor": dict(trace.initial_descriptor),
        "initial_sha256": canonical_hash(trace.initial_descriptor),
        "living_mass_sum": mass_sum,
        "mass_integral": mass_sum / request.horizon,
        "ambient_food": food,
        "survival_fraction": alive_frames / request.horizon,
        "endpoint": int(not dead),
        "boost_fraction": boost_frames / request.horizon,
        "encounter_frames": encounter_frames,
        "counters": counters,
        "frame_columns": columns,
    }


def save_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically publish one complete JSON report; never replace prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("role") == "anchor":
        if policy.get("name") not in {"greedy_food", "random_safe"}:
            raise ValueError("unknown scripted anchor")
        if policy.get("lineage") is not None:
            raise ValueError("anchors are shared controls, not duplicated by lineage")
    elif policy.get("role") in {"parent", "candidate"}:
        if policy.get("lineage") not in LINEAGES:
            raise ValueError("unknown or omitted learned lineage")
        _digest(policy.get("checkpoint_sha256"), "checkpoint identity")
    else:
        raise ValueError("role must be parent, candidate, or anchor")


def evaluate_cell(
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    profile: str,
    runtime: EvaluationRuntime,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate one fixed policy/profile cell: exactly 32 procedural worlds."""
    validate_spec(spec)
    _validate_policy(policy)
    if output_path.exists():
        raise FileExistsError("completed output exists; never repeat a cell")
    if runtime.contract_sha256 != spec["contract_sha256"]:
        raise ValueError("runtime contract differs from frozen spec")
    counters = _counts(runtime.initial_counters)
    if any(counters[key] for key in COUNTERS if key != "checkpoint_loads"):
        raise ValueError("adapter factory may load weights but cannot perform hidden gameplay")
    requests = [
        EpisodeRequest(seed, profile, tuple(spec["profiles"][profile]), spec["horizon"])
        for seed in spec["world_seeds"][profile]
    ]
    shared = dict.fromkeys(COUNTERS, 0)
    if runtime.run_batch is None:
        traces = [runtime.run_episode(request) for request in requests]
    else:
        traces, shared_values = runtime.run_batch(requests)
        shared = _counts(shared_values)
        if len(traces) != len(requests):
            raise ValueError("batched adapter returned the wrong number of worlds")
        if any(shared[key] for key in COUNTERS if key != "model_forwards"):
            raise ValueError("only actual batched model calls may be shared counters")
    for key in COUNTERS:
        counters[key] += shared[key]
    cases = []
    for request, trace in zip(requests, traces):
        row = _reduce_episode(request, trace)
        cases.append(row)
        for key in COUNTERS:
            counters[key] += row["counters"][key]
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "stage_id": output_path.stem,
        "status": "PASS",
        "kind": "policy_evaluation",
        "spec_sha256": canonical_hash(spec),
        "contract_sha256": runtime.contract_sha256,
        "policy": dict(policy),
        "profile": profile,
        "cases": cases,
        "cases_sha256": canonical_hash(cases),
        "counters": counters,
        "initial_counters": dict(runtime.initial_counters),
        "shared_counters": shared,
        "summary": {key: statistics.mean(row[key] for row in cases) for key in METRICS},
        "artifacts": [],
        "promotion_eligible": False,
    }
    validate_report(spec, report)
    save_report(output_path, report)
    return report


def validate_report(spec: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Check saved per-world facts and accounting without simulator/model operations."""
    validate_spec(spec)
    if report.get("status") != "PASS" or report.get("kind") != "policy_evaluation":
        raise ValueError("incomplete or wrong report kind")
    if report.get("schema_version") != 1 or report.get("study_id") != STUDY_ID:
        raise ValueError("report study/schema differs")
    if report.get("spec_sha256") != canonical_hash(spec):
        raise ValueError("report spec identity differs")
    if report.get("contract_sha256") != spec["contract_sha256"]:
        raise ValueError("report runtime contract differs")
    _validate_policy(report["policy"])
    profile, cases = report["profile"], report["cases"]
    if [row["world_seed"] for row in cases] != spec["world_seeds"][profile]:
        raise ValueError("missing, duplicate, reordered, or wrong-world cases")
    if report["cases_sha256"] != canonical_hash(cases):
        raise ValueError("case identity mismatch")
    counts = _counts(report["initial_counters"])
    if any(counts[key] for key in COUNTERS if key != "checkpoint_loads"):
        raise ValueError("non-load operations outside the recorded episode stream")
    shared = _counts(report.get("shared_counters", dict.fromkeys(COUNTERS, 0)))
    if any(shared[key] for key in COUNTERS if key != "model_forwards"):
        raise ValueError("shared counters may attribute actual batched model calls only")
    for key in COUNTERS:
        counts[key] += shared[key]
    for row in cases:
        horizon = spec["horizon"]
        columns = row["frame_columns"]
        names = ("alive", "mass", "ambient_food", "boost", "encounter")
        if set(columns) != set(names) or len({len(columns[name]) for name in names}) != 1:
            raise ValueError("frame columns are incomplete or have different lengths")
        frames = (
            FrameMetrics(**dict(zip(names, values)))
            for values in zip(*(columns[name] for name in names))
        )
        request = EpisodeRequest(
            row["world_seed"], profile, tuple(spec["profiles"][profile]), horizon
        )
        reconstructed = _reduce_episode(
            request, EpisodeTrace(frames, row["initial_descriptor"], row["counters"])
        )
        if reconstructed != row:
            raise ValueError("saved case does not match its physical frame columns")
        if row["profile"] != profile or row["horizon"] != horizon:
            raise ValueError("case profile or horizon differs")
        steps = _integer(row["observed_frames"], "observed_frames", 1)
        if steps > horizon or row["endpoint"] not in (0, 1):
            raise ValueError("invalid episode endpoint or frame count")
        if row["endpoint"] and steps != horizon:
            raise ValueError("survivor did not reach horizon")
        if row["initial_sha256"] != canonical_hash(row["initial_descriptor"]):
            raise ValueError("initial descriptor hash differs")
        if not {"world_seed", "hero", "opponents", "food", "rng_sha256"} <= set(
            row["initial_descriptor"]
        ):
            raise ValueError("incomplete initial descriptor")
        _digest(row["initial_descriptor"]["rng_sha256"], "initial RNG digest")
        if row["initial_descriptor"]["world_seed"] != row["world_seed"]:
            raise ValueError("descriptor world differs")
        for key in METRICS:
            _finite(row[key], key)
        if row["mass_integral"] != _finite(row["living_mass_sum"], "mass sum") / horizon:
            raise ValueError("mass denominator is not the complete horizon")
        if row["survival_fraction"] != (steps - (not row["endpoint"])) / horizon:
            raise ValueError("survival does not follow terminal hero accounting")
        if row["boost_fraction"] > steps / horizon or row["encounter_frames"] > steps:
            raise ValueError("invalid event counts")
        current = _counts(row["counters"])
        if current["native_frames"] != steps or current["hero_transitions"] != steps:
            raise ValueError("physical counters disagree with observed frames")
        for key in COUNTERS:
            counts[key] += current[key]
    if counts != _counts(report["counters"]):
        raise ValueError("physical counters do not sum")
    for key in METRICS:
        if report["summary"][key] != statistics.mean(row[key] for row in cases):
            raise ValueError("saved summary differs from cases")


def _paired(
    candidate: Sequence[Mapping[str, Any]], parent: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    deltas = [c[metric] - p[metric] for c, p in zip(candidate, parent)]
    mean = statistics.mean(deltas)
    half = T975_DF31 * statistics.stdev(deltas) / math.sqrt(32)
    return {
        "parent_mean": statistics.mean(row[metric] for row in parent),
        "candidate_mean": statistics.mean(row[metric] for row in candidate),
        "mean_delta": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "world_deltas": deltas,
        "n_worlds": 32,
        "df": 31,
        "interval_scope": "fixed_lineage_paired_worlds_not_training_population",
    }


def analyze_reports(
    spec: Mapping[str, Any], reports: Sequence[Mapping[str, Any]], output_path: Path | None = None
) -> dict[str, Any]:
    """Reduce exactly 24 saved cells, preserving all lineages and both anchors."""
    validate_spec(spec)
    cells = {}
    for report in reports:
        validate_report(spec, report)
        policy = report["policy"]
        identity = policy.get("name") if policy["role"] == "anchor" else policy["lineage"]
        key = (policy["role"], identity, report["profile"])
        if key in cells:
            raise ValueError("duplicate evaluation cell")
        cells[key] = report
    expected = {
        (role, seed, profile)
        for role in ("parent", "candidate")
        for seed in LINEAGES
        for profile in spec["profiles"]
    }
    expected |= {
        ("anchor", name, profile)
        for name in ("greedy_food", "random_safe")
        for profile in spec["profiles"]
    }
    if set(cells) != expected:
        raise ValueError("decision requires all 18 learned and six anchor cells")
    for role in ("parent", "candidate"):
        for seed in LINEAGES:
            identities = {
                canonical_hash(cells[(role, seed, profile)]["policy"])
                for profile in spec["profiles"]
            }
            if len(identities) != 1:
                raise ValueError("a policy changed identity between evaluation profiles")
    outcomes, gains, solo_guards, harms, futile = {}, [], [], [], []
    exposure_ok = True
    exposure_counts = {}
    for profile in spec["profiles"]:
        # Every policy must see the identical initial native world for each seed.
        reference = cells[("anchor", "greedy_food", profile)]["cases"]
        for key, report in cells.items():
            if key[2] == profile and [r["initial_sha256"] for r in report["cases"]] != [
                r["initial_sha256"] for r in reference
            ]:
                raise ValueError("initial native state pairing differs between controllers")
        if profile != "solo":
            required_roles = [("parent", seed, profile) for seed in LINEAGES]
            required_roles.append(("anchor", "greedy_food", profile))
            counts = {
                f"{key[0]}:{key[1]}": sum(
                    row["encounter_frames"] > 0 for row in cells[key]["cases"]
                )
                for key in required_roles
            }
            exposure_counts[profile] = counts
            exposure_ok &= all(
                value >= spec["decision"]["minimum_exposed_worlds"] for value in counts.values()
            )
    for seed in LINEAGES:
        by_profile = {}
        for profile in spec["profiles"]:
            parent = cells[("parent", seed, profile)]["cases"]
            candidate = cells[("candidate", seed, profile)]["cases"]
            paired = {metric: _paired(candidate, parent, metric) for metric in METRICS}
            if profile == "solo":
                margins = {
                    "mass_integral": 0.05 * paired["mass_integral"]["parent_mean"],
                    "ambient_food": 0.05 * paired["ambient_food"]["parent_mean"],
                    "survival_fraction": 0.02,
                }
                guards = {
                    name: paired[name]["ci_low"] >= -margin for name, margin in margins.items()
                }
                solo_guards.extend(guards.values())
                harms.extend(paired[name]["ci_high"] < -margin for name, margin in margins.items())
                paired["retention_margins"] = margins
                paired["retention_pass"] = guards
            else:
                mass = paired["mass_integral"]
                margin = max(1.0, 0.10 * mass["parent_mean"])
                gain = mass["mean_delta"] >= margin and mass["ci_low"] > 0.0
                gains.append(gain)
                futile.append(mass["ci_high"] < margin)
                harms.append(mass["ci_high"] < 0.0)
                paired["practical_mass_margin"] = margin
                paired["gain_pass"] = gain
            by_profile[profile] = paired
        outcomes[str(seed)] = by_profile
    if any(harms):
        decision = "NEGATIVE_RELIABILITY_SCREEN"
    elif not exposure_ok:
        decision = "INCONCLUSIVE_EXPOSURE"
    elif all(gains) and all(solo_guards):
        decision = "ADVANCE_PACKAGE_SCREEN"
    elif all(futile):
        decision = "NEGATIVE_PRACTICAL_GAIN_SCREEN"
    else:
        decision = "INCONCLUSIVE_SCREEN"
    result = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "stage_id": "analysis",
        "status": "PASS",
        "kind": "paired_decision",
        "decision": decision,
        "spec_sha256": canonical_hash(spec),
        "per_lineage": outcomes,
        "baseline_exposure_pass": exposure_ok,
        "baseline_exposure_worlds": exposure_counts,
        "n_trained_lineages": 3,
        "source_report_sha256": [canonical_hash(report) for report in reports],
        "anchors": {
            f"{key[1]}:{key[2]}": value["summary"]
            for key, value in cells.items()
            if key[0] == "anchor"
        },
        "counters": dict.fromkeys(COUNTERS, 0),
        "artifacts": [],
        "promotion_eligible": False,
        "causal_factor_claim": False,
        "training_population_claim": False,
        "interpretation": (
            "Prospective package utility screen; old study outcomes remain unchanged."
        ),
    }
    if output_path is not None:
        save_report(output_path, result)
    return result


def main() -> None:
    """Dispatch only explicit named paths; never discover historical artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--spec", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--profile", required=True)
    evaluate.add_argument("--adapter", required=True, help="module:factory(spec, policy)")
    evaluate.add_argument("--out", type=Path, required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--spec", type=Path, required=True)
    analysis.add_argument("--reports", type=Path, nargs="+", required=True)
    analysis.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    if args.out.exists():
        raise FileExistsError("refusing to repeat a completed stage")
    if args.command == "analyze":
        analyze_reports(spec, [json.loads(path.read_text()) for path in args.reports], args.out)
    else:
        policy = json.loads(args.policy.read_text())
        _validate_policy(policy)
        if args.profile not in spec["profiles"]:
            raise ValueError("profile absent from frozen spec")
        module, function = args.adapter.split(":", 1)
        factory = getattr(importlib.import_module(module), function)
        runtime = factory(spec, policy)
        evaluate_cell(spec, policy, args.profile, runtime, args.out)


if __name__ == "__main__":
    main()
