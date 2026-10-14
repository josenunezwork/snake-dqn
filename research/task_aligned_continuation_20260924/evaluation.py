"""One fixed continuation: pure saved-data reduction and admitted runtime callbacks.

The completed three-lineage study remains unchanged. Only its stdlib trace types,
physical case reducer, canonical JSON publisher and paired-interval primitive are
reused. Importing this module loads no model, artifact, simulator or runtime.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from research.task_aligned_20260924 import evaluation as primitives

STUDY_ID = "task-aligned-continuation-20260924"
LINEAGE = 2026096102
LINEAGES = (LINEAGE,)
HORIZON = 3000
HELD_NAMESPACE = f"{STUDY_ID}/held-v1"
PROFILES = {
    "solo": [],
    "food_pressure": ["greedy_food"] * 4 + ["random_safe"],
    "mixed": ["greedy_food"] * 2 + ["random_safe"] * 3,
}
DECISION = {
    "relative_mass_gain": 0.10,
    "absolute_mass_gain": 1.0,
    "profile_mass_loss": 0.05,
    "solo_food_loss": 0.05,
    "solo_survival_loss": 0.02,
    "minimum_exposed_worlds": 8,
}
COUNTERS = primitives.COUNTERS
METRICS = primitives.METRICS
EpisodeRequest = primitives.EpisodeRequest
EpisodeTrace = primitives.EpisodeTrace
EvaluationRuntime = primitives.EvaluationRuntime
FrameMetrics = primitives.FrameMetrics
canonical_hash = primitives.canonical_hash
save_report = primitives.save_report


def make_world_seeds() -> dict[str, list[int]]:
    """Return prospective IDs only; never construct or screen native worlds."""
    return {
        profile: [
            int.from_bytes(
                hashlib.sha256(f"{HELD_NAMESPACE}/{profile}/{index}".encode()).digest()[:8],
                "big",
            )
            for index in range(32)
        ]
        for profile in PROFILES
    }


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Require the complete new evaluation contract, including sealed child identity."""
    if spec.get("study_id") != STUDY_ID or spec.get("schema_version") != 1:
        raise ValueError("Wrong continuation study/schema")
    if spec.get("lineages") != [LINEAGE] or spec.get("horizon") != HORIZON:
        raise ValueError("One fixed6102 lineage and H3000 are required")
    if spec.get("evaluation_device") != "cpu" or spec.get("batch_worlds") != 32:
        raise ValueError("Qualified CPU E32 serving is required")
    if (
        spec.get("source_update") != 35613
        or spec.get("candidate_update") != 66618
        or spec.get("continuation_rollouts") != 31005
    ):
        raise ValueError("The source, child endpoint or fixed continuation dose differs")
    primitives._digest(spec.get("contract_sha256"), "contract_sha256")
    if not isinstance(spec.get("source_config"), Mapping) or not spec["source_config"]:
        raise ValueError("The exact inherited source configuration must accompany the contract")
    bindings = spec.get("policy_sha256", {})
    if set(bindings) != {"parent", "candidate"}:
        raise ValueError("Both exact sealed checkpoint identities are required")
    for role, digest in bindings.items():
        primitives._digest(digest, f"{role} checkpoint")
    if bindings["parent"] == bindings["candidate"]:
        raise ValueError("Parent and trained child identities must differ")
    if spec.get("profiles") != PROFILES or spec.get("decision") != DECISION:
        raise ValueError("Prospective profiles or decision tolerances differ")
    if (
        spec.get("held_seed_namespace") != HELD_NAMESPACE
        or spec.get("world_seeds") != make_world_seeds()
    ):
        raise ValueError("New fixed held namespace/IDs differ")
    held = [seed for values in spec["world_seeds"].values() for seed in values]
    if len(set(held)) != 96:
        raise ValueError("Held profile world IDs collide")
    namespaces = spec.get("training_seed_namespaces")
    if (
        not isinstance(namespaces, list)
        or not namespaces
        or any(not isinstance(name, str) or not name for name in namespaces)
        or HELD_NAMESPACE in namespaces
    ):
        raise ValueError("Explicit distinct training namespaces are required")
    for field in ("training_world_seeds", "excluded_world_seeds"):
        values = spec.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Explicit nonempty {field} IDs are required")
        ids = {primitives._integer(seed, field) for seed in values}
        if set(held) & ids:
            raise ValueError(f"Held IDs overlap {field}")
    if spec.get("novelty_scope") != "declared_seed_namespaces_only":
        raise ValueError("No implicit all-history geometry novelty claim is permitted")


def _policy(spec: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    if policy.get("role") == "anchor":
        if policy.get("lineage") is not None or policy.get("name") not in (
            "greedy_food",
            "random_safe",
        ):
            raise ValueError("Exactly two shared scripted anchors are allowed")
    elif policy.get("role") in ("parent", "candidate"):
        if (
            policy.get("lineage") != LINEAGE
            or policy.get("checkpoint_sha256") != spec["policy_sha256"][policy["role"]]
        ):
            raise ValueError("Learned policy differs from the sealed continuation identity")
    else:
        raise ValueError("Unknown policy role")


def evaluate_cell(
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    profile: str,
    runtime: EvaluationRuntime,
    out: Path,
) -> dict[str, Any]:
    """Run one admitted E32 cell through the qualified external adapter."""
    validate_spec(spec)
    _policy(spec, policy)
    if profile not in PROFILES or out.exists():
        raise ValueError("Unknown profile or already existing cell output")
    if runtime.contract_sha256 != spec["contract_sha256"] or runtime.run_batch is None:
        raise ValueError("Exact qualified batched serving contract is required")
    requests = [
        EpisodeRequest(seed, profile, tuple(PROFILES[profile]), HORIZON)
        for seed in spec["world_seeds"][profile]
    ]
    initial = primitives._counts(runtime.initial_counters)
    if any(initial[key] for key in COUNTERS if key != "checkpoint_loads"):
        raise ValueError("Only a real policy load may precede the recorded cell")
    if initial["checkpoint_loads"] > int(policy["role"] != "anchor"):
        raise ValueError("Unexpected checkpoint loads before the cell")
    traces, shared_values = runtime.run_batch(requests)
    if len(traces) != 32:
        raise ValueError("Runtime omitted or duplicated a requested world")
    shared = primitives._counts(shared_values)
    cases = [primitives._reduce_episode(request, trace) for request, trace in zip(requests, traces)]
    report = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "stage_id": out.stem,
        "status": "PASS",
        "kind": "policy_evaluation",
        "spec_sha256": canonical_hash(spec),
        "contract_sha256": runtime.contract_sha256,
        "policy": dict(policy),
        "profile": profile,
        "cases": cases,
        "cases_sha256": canonical_hash(cases),
        "counters": {
            key: initial[key] + shared[key] + sum(row["counters"][key] for row in cases)
            for key in COUNTERS
        },
        "initial_counters": initial,
        "shared_counters": shared,
        "summary": {key: statistics.mean(row[key] for row in cases) for key in METRICS},
        "artifacts": [],
        "promotion_eligible": False,
    }
    validate_report(spec, report)
    save_report(out, report)
    return report


def validate_report(spec: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Reconstruct exact physical case facts and E32 accounting from saved columns."""
    validate_spec(spec)
    if (
        report.get("schema_version") != 1
        or report.get("study_id") != STUDY_ID
        or report.get("kind") != "policy_evaluation"
        or report.get("status") != "PASS"
        or report.get("promotion_eligible") is not False
        or report.get("spec_sha256") != canonical_hash(spec)
        or report.get("contract_sha256") != spec["contract_sha256"]
    ):
        raise ValueError("Cell schema, status, scope or contract differs")
    policy = report["policy"]
    _policy(spec, policy)
    profile, cases = report["profile"], report["cases"]
    if (
        profile not in PROFILES
        or [row["world_seed"] for row in cases] != spec["world_seeds"][profile]
    ):
        raise ValueError("Missing, duplicate, reordered or incorrect worlds")
    if report["cases_sha256"] != canonical_hash(cases):
        raise ValueError("Case hash differs")
    initial, shared = (
        primitives._counts(report[name]) for name in ("initial_counters", "shared_counters")
    )
    if any(initial[key] for key in COUNTERS if key != "checkpoint_loads"):
        raise ValueError("Non-load work attributed outside the episode stream")
    if initial["checkpoint_loads"] > int(policy["role"] != "anchor"):
        raise ValueError("Unexpected checkpoint loads")
    counts = {key: initial[key] + shared[key] for key in COUNTERS}
    names = ("alive", "mass", "ambient_food", "boost", "encounter")
    for row in cases:
        columns = row["frame_columns"]
        if set(columns) != set(names) or len({len(columns[key]) for key in names}) != 1:
            raise ValueError("Incomplete or unequal frame columns")
        frames = (
            FrameMetrics(**dict(zip(names, values)))
            for values in zip(*(columns[name] for name in names))
        )
        request = EpisodeRequest(row["world_seed"], profile, tuple(PROFILES[profile]), HORIZON)
        reconstructed = primitives._reduce_episode(
            request, EpisodeTrace(frames, row["initial_descriptor"], row["counters"])
        )
        if reconstructed != row:
            raise ValueError("Saved case differs from physical frame arithmetic")
        if len(row["initial_descriptor"]["opponents"]) != len(PROFILES[profile]):
            raise ValueError("Initial physical roster differs")
        if any(
            row["counters"][key]
            for key in ("model_forwards", "checkpoint_loads", "optimizer_updates")
        ):
            raise ValueError("Shared batched/load work falsely attributed to one world")
        for key in COUNTERS:
            counts[key] += row["counters"][key]
    expected_shared = dict.fromkeys(COUNTERS, 0)
    if policy["role"] != "anchor":
        expected_shared["model_forwards"] = max(row["observed_frames"] for row in cases)
    if shared != expected_shared or counts != primitives._counts(report["counters"]):
        raise ValueError("Physical cell/shared counter arithmetic differs")
    if report["summary"] != {key: statistics.mean(row[key] for row in cases) for key in METRICS}:
        raise ValueError("Saved summary differs from complete cases")


def analyze_reports(
    spec: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Analyze only the complete twelve-cell single-child matrix; never replay worlds."""
    validate_spec(spec)
    cells: dict[tuple[str, Any, str], Mapping[str, Any]] = {}
    for report in reports:
        validate_report(spec, report)
        policy = report["policy"]
        identity = policy["name"] if policy["role"] == "anchor" else policy["lineage"]
        key = (policy["role"], identity, report["profile"])
        if key in cells:
            raise ValueError("Duplicate evaluation cell")
        cells[key] = report
    identities = (
        ("parent", LINEAGE),
        ("candidate", LINEAGE),
        ("anchor", "greedy_food"),
        ("anchor", "random_safe"),
    )
    if set(cells) != {
        (role, identity, profile) for role, identity in identities for profile in PROFILES
    }:
        raise ValueError("Exactly six learned and six shared-anchor cells are required")
    for role, identity in identities:
        if len({canonical_hash(cells[(role, identity, p)]["policy"]) for p in PROFILES}) != 1:
            raise ValueError("Policy identity changed between profiles")
        loads = sum(cells[(role, identity, p)]["counters"]["checkpoint_loads"] for p in PROFILES)
        if loads != int(role != "anchor"):
            raise ValueError("Each learned policy must load exactly once; anchors never load")
    exposure, paired_profiles, harms, gains, retention = {}, {}, [], [], []
    for profile in PROFILES:
        reference = cells[("anchor", "greedy_food", profile)]["cases"]
        for role, identity in identities:
            if [row["initial_sha256"] for row in cells[(role, identity, profile)]["cases"]] != [
                row["initial_sha256"] for row in reference
            ]:
                raise ValueError("Controllers did not begin from paired native worlds")
        if profile != "solo":
            exposure[profile] = {
                f"{role}:{identity}": sum(
                    row["encounter_frames"] > 0 for row in cells[(role, identity, profile)]["cases"]
                )
                for role, identity in (("parent", LINEAGE), ("anchor", "greedy_food"))
            }
        parent = cells[("parent", LINEAGE, profile)]["cases"]
        child = cells[("candidate", LINEAGE, profile)]["cases"]
        paired = {metric: primitives._paired(child, parent, metric) for metric in METRICS}
        mass = paired["mass_integral"]
        mass_loss = DECISION["profile_mass_loss"] * mass["parent_mean"]
        profile_harm = {"mass_integral": mass["ci_high"] < -mass_loss}
        paired["mass_loss_margin"] = mass_loss
        if profile == "solo":
            margins = {
                "mass_integral": mass_loss,
                "ambient_food": DECISION["solo_food_loss"] * paired["ambient_food"]["parent_mean"],
                "survival_fraction": DECISION["solo_survival_loss"],
            }
            guards = {name: paired[name]["ci_low"] >= -margin for name, margin in margins.items()}
            profile_harm.update(
                {name: paired[name]["ci_high"] < -margin for name, margin in margins.items()}
            )
            paired.update(retention_margins=margins, retention_pass=guards)
            retention.extend(guards.values())
        else:
            margin = max(
                DECISION["absolute_mass_gain"], DECISION["relative_mass_gain"] * mass["parent_mean"]
            )
            gain = mass["mean_delta"] >= margin and mass["ci_low"] > 0.0
            paired.update(practical_mass_margin=margin, gain_pass=gain)
            gains.append(gain)
        paired["clear_harm"] = profile_harm
        harms.extend(profile_harm.values())
        paired_profiles[profile] = paired
    exposure_pass = all(count >= 8 for group in exposure.values() for count in group.values())
    if any(harms):
        decision = "NEGATIVE_CONTINUATION_SCREEN"
    elif not exposure_pass:
        decision = "INCONCLUSIVE_EXPOSURE"
    elif all(gains) and all(retention):
        decision = "ADVANCE_CONTINUATION_SCREEN"
    else:
        decision = "INCONCLUSIVE_SCREEN"
    totals = {key: sum(report["counters"][key] for report in reports) for key in COUNTERS}
    if (
        totals["native_frames"] > 1152000
        or totals["hero_transitions"] > 1152000
        or totals["model_forwards"] > 18000
    ):
        raise ValueError("Fixed evaluation physical ceiling exceeded")
    result = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "stage_id": "analysis",
        "kind": "paired_continuation_decision",
        "status": "PASS",
        "decision": decision,
        "spec_sha256": canonical_hash(spec),
        "per_lineage": {str(LINEAGE): paired_profiles},
        "baseline_exposure_pass": exposure_pass,
        "baseline_exposure_worlds": exposure,
        "n_trained_lineages": 1,
        "source_report_sha256": [canonical_hash(report) for report in reports],
        "anchors": {
            f"{identity}:{profile}": cell["summary"]
            for (role, identity, profile), cell in cells.items()
            if role == "anchor"
        },
        "audited_evaluation_counters": totals,
        "counters": dict.fromkeys(COUNTERS, 0),
        "artifacts": [],
        "promotion_eligible": False,
        "causal_factor_claim": False,
        "training_population_claim": False,
        "interpretation": (
            "One prospectively fixed continuation versus its own source; no incumbent "
            "comparison or champion promotion, and no change to any completed study decision."
        ),
    }
    if output_path is not None:
        save_report(output_path, result)
    return result
