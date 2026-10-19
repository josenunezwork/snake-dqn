"""Fail-closed one-look analysis for the Phase 0 search-teacher premise test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.phase0_search_20261019 import bank
from src.scripts.eval_stats import student_t_isf

SCHEMA = "phase0-search-teacher-analysis/v1"
INPUT_SCHEMA = "phase0-search-teacher-analysis-input/v1"
MIXES = ("scripted", "frozen", "mixed")
EPS = 1e-12
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _runner_digest(value: Any) -> str:
    """Match the runner's create-only canonical JSON digest (trailing newline)."""
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def _input_digest(value: Any) -> str:
    """Bind valid input exactly; retain malformed/non-finite evidence safely."""
    try:
        return digest(value)
    except (TypeError, ValueError):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _rows(value: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[str]]:
    if value.get("schema") not in {INPUT_SCHEMA, "phase0-search-teacher-merge/v1"}:
        return [], ["wrong input schema"]
    raw = value.get("rows", value.get("worlds"))
    if not isinstance(raw, list):
        return [], ["rows must be a list"]
    return raw, []


def _validate(value: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    rows, errors = _rows(value)
    raw_bank, audit = value.get("bank"), value.get("bank_audit")
    if value.get("complete") is not True:
        errors.append("merged result is incomplete")
    plan_sha = value.get("plan_sha256")
    if not _sha256(plan_sha):
        errors.append("merged result lacks a pinned plan digest")
    bank_file_sha = value.get("bank_file_sha256")
    if not _sha256(bank_file_sha):
        errors.append("merged result has invalid bank file digest")
    binding = value.get("plan_binding")
    if not isinstance(binding, Mapping) or binding.get("kind") != "selection":
        errors.append("merged result is not bound to a selection plan")
    else:
        inputs = binding.get("input_sha256")
        source = binding.get("source_closure_sha256")
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("bank") != bank_file_sha
            or not _sha256(inputs.get("audit"))
            or not _sha256(inputs.get("config"))
            or not _sha256(inputs.get("preregistration"))
            or inputs.get("checkpoint") != B8_SHA256
        ):
            errors.append("selection plan input bindings are incomplete")
        if (
            not isinstance(source, Mapping)
            or not source
            or not all(_sha256(item) for item in source.values())
        ):
            errors.append("selection plan source closure is incomplete")
        if not isinstance(binding.get("config"), Mapping):
            errors.append("selection plan lacks config binding")
    claimed_merged = value.get("merged_sha256")
    if not _sha256(claimed_merged):
        errors.append("merged result lacks a valid receipt digest")
    else:
        unsigned = dict(value)
        unsigned.pop("merged_sha256", None)
        if claimed_merged != _runner_digest(unsigned):
            errors.append("merged receipt digest mismatch")
    if not isinstance(raw_bank, Mapping) or raw_bank.get("schema") != bank.SCHEMA:
        errors.append("missing or malformed Phase 0 bank")
        return {}, errors
    expected_bank_digest = bank.digest(
        {"selection": raw_bank.get("selection"), "fixtures": raw_bank.get("fixtures")}
    )
    if raw_bank.get("allocation_semantic_sha256") != expected_bank_digest:
        errors.append("bank semantic digest mismatch")
    if (
        not isinstance(audit, Mapping)
        or audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != expected_bank_digest
    ):
        errors.append("missing, failed, or unbound collision audit")
    expected = {
        str(row["world_id"]): row
        for row in raw_bank.get("selection", [])
        if isinstance(row, Mapping)
    }
    if len(expected) != sum(bank.MIX_COUNTS.values()):
        errors.append("selection bank does not contain 160 registered worlds")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("non-object result row")
            continue
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or world_id in indexed:
            errors.append("missing or duplicate world_id")
            continue
        indexed[world_id] = row
        registered = expected.get(world_id)
        if registered is None:
            errors.append(f"unregistered world {world_id}")
            continue
        if row.get("complete") is not True:
            errors.append(f"incomplete world {world_id}")
        if row.get("plan_sha256") != plan_sha:
            errors.append(f"plan binding mismatch {world_id}")
        if bank_file_sha is not None and row.get("bank_sha256") != bank_file_sha:
            errors.append(f"bank file binding mismatch {world_id}")
        elif row.get("bank_sha256") not in {
            expected_bank_digest,
            digest(raw_bank),
            value.get("bank_file_sha256"),
        }:
            errors.append(f"bank binding mismatch {world_id}")
        if row.get("world_digest", row.get("world_semantic_sha256")) != registered.get(
            "world_semantic_sha256"
        ):
            errors.append(f"world digest mismatch {world_id}")
        if row.get("mix") != registered.get("mix") or row.get("world_seed") != registered.get(
            "world_seed"
        ):
            errors.append(f"world identity mismatch {world_id}")
        identity = row.get("world_identity")
        if not isinstance(identity, Mapping) or identity.get("horizon") != bank.HORIZON:
            errors.append(f"missing world identity {world_id}")
        elif identity.get("world_semantic_sha256") != registered.get(
            "world_semantic_sha256"
        ) or identity.get("opponent_semantic_sha256") != registered.get("opponent_semantic_sha256"):
            errors.append(f"world identity digest mismatch {world_id}")
        if row.get("opponent_semantics", row.get("opponent_semantic_sha256")) not in (
            None,
            registered.get("opponent_semantic_sha256"),
        ):
            errors.append(f"opponent semantic mismatch {world_id}")
        arms = row.get("arms")
        paired = row.get("paired")
        if (
            not isinstance(arms, Mapping)
            or set(arms) != {"b8", "search"}
            or not isinstance(paired, Mapping)
        ):
            errors.append(f"missing paired arms {world_id}")
            continue
        arm_values: dict[str, Mapping[str, Any]] = {}
        for arm in ("b8", "search"):
            data = arms[arm]
            if (
                not isinstance(data, Mapping)
                or not _finite(data.get("mass_integral"))
                or type(data.get("survived")) is not bool
            ):
                errors.append(f"invalid {arm} endpoint {world_id}")
            elif data.get("frames_scored") != bank.HORIZON:
                errors.append(f"non-H5000 endpoint {arm} {world_id}")
            else:
                arm_values[arm] = data
        if len(arm_values) == 2:
            delta = float(arm_values["search"]["mass_integral"]) - float(
                arm_values["b8"]["mass_integral"]
            )
            if not _finite(paired.get("mass_delta")) or not math.isclose(
                float(paired["mass_delta"]), delta, rel_tol=0.0, abs_tol=EPS
            ):
                errors.append(f"paired mass mismatch {world_id}")
    if set(indexed) != set(expected):
        errors.append("result rows do not exactly cover the registered selection bank")
    return indexed, errors


def _effect(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_mix: dict[str, list[float]] = defaultdict(list)
    survival: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"search": [], "b8": []})
    for row in rows:
        mix = str(row["mix"])
        by_mix[mix].append(float(row["paired"]["mass_delta"]))
        survival[mix]["search"].append(float(row["arms"]["search"]["survived"]))
        survival[mix]["b8"].append(float(row["arms"]["b8"]["survived"]))
    per_mix: dict[str, Any] = {}
    components = []
    for mix in MIXES:
        values = by_mix[mix]
        if len(values) != bank.MIX_COUNTS[mix]:
            raise ValueError(f"wrong {mix} count")
        variance = statistics.variance(values)
        contribution = variance / len(values) / (len(MIXES) ** 2)
        components.append((contribution, len(values) - 1))
        per_mix[mix] = {
            "n": len(values),
            "mean_mass_delta": statistics.mean(values),
            "paired_world_variance": variance,
            "search_no_death_fraction": statistics.mean(survival[mix]["search"]),
            "b8_no_death_fraction": statistics.mean(survival[mix]["b8"]),
        }
    point = statistics.mean([per_mix[mix]["mean_mass_delta"] for mix in MIXES])
    variance = sum(component for component, _ in components)
    se = math.sqrt(variance)
    welch_df = (
        variance * variance / sum(component * component / df for component, df in components)
        if variance
        else None
    )
    critical = (
        student_t_isf(0.10, max(1, int(math.floor(welch_df)))) if welch_df is not None else 0.0
    )
    return {
        "equal_mix_mass_effect": point,
        "stratified_standard_error": se,
        "welch_df": welch_df,
        "quantile_df": math.floor(welch_df) if welch_df is not None else None,
        "critical_one_sided_90": critical,
        "lower90": point - critical * se,
        "upper90": point + critical * se,
        "per_mix": per_mix,
        "method": (
            "equal-mix paired world means; stratified world-level SE; "
            "Welch-Satterthwaite df; one-sided 90% t bound"
        ),
    }


def analyze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply INVALID/INCOMPLETE, STOP_TRAINING, then GO precedence."""
    try:
        indexed, errors = _validate(value)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        indexed, errors = {}, [f"malformed evidence: {type(exc).__name__}: {exc}"]
    output: dict[str, Any] = {
        "schema": SCHEMA,
        "input_digest": _input_digest(value),
        "errors": errors,
    }
    if errors:
        return {
            **output,
            "decision": "INVALID_INCOMPLETE",
            "decision_basis": "all registered rows and bindings are required before estimation",
        }
    stats = _effect(indexed.values())
    scripted = stats["per_mix"]["scripted"]
    stop_reasons = []
    if stats["upper90"] < 40.0:
        stop_reasons.append("upper90 pooled paired mass effect is below +40")
    if scripted["search_no_death_fraction"] < scripted["b8_no_death_fraction"] + 0.05:
        stop_reasons.append("scripted search no-death fraction is below B8 + 0.05")
    go = (
        scripted["search_no_death_fraction"] >= 0.66
        and stats["equal_mix_mass_effect"] >= 80.0
        and stats["lower90"] > 40.0
    )
    decision = "STOP_TRAINING" if stop_reasons else "GO" if go else "INCONCLUSIVE"
    return {
        **output,
        "decision": decision,
        "decision_precedence": ["INVALID_INCOMPLETE", "STOP_TRAINING", "GO", "INCONCLUSIVE"],
        "statistics": stats,
        "stop_reasons": stop_reasons,
        "go_conditions": {
            "scripted_no_death_at_least_0_66": scripted["search_no_death_fraction"] >= 0.66,
            "pooled_effect_at_least_80": stats["equal_mix_mass_effect"] >= 80.0,
            "lower90_above_40": stats["lower90"] > 40.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        bank.write_create(args.out, result)
    else:
        print(text, end="")
    return 0 if result["decision"] != "INVALID_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
