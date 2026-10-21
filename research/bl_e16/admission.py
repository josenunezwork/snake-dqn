"""Fail-closed E16 input nomination, arm preconditions and total-cost admission."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import digest, verify_export

ARMS = ("A", "G", "R", "S", "P")
IMPLEMENTED = {"A", "G"}
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
G_HISTORY = "e3-geometry7-tower7x16x64-grid6x10-gru43x64-window128/v1"
G_NO_HISTORY = "e3-geometry7-tower7x16x64-grid6x10-no-history/v1"


class AdmissionRefused(ValueError):
    """A missing or inconsistent prerequisite prevents scientific execution."""


def forecast(actor_rate: float, episode_rate: float, treatments: int = 1) -> dict[str, Any]:
    """Price literal per-arm preparation, fixed whole-life N, setup and reserve."""
    if treatments not in (1, 2):
        raise AdmissionRefused("one or two treatments required")
    if any(not math.isfinite(x) or x <= 0 for x in (actor_rate, episode_rate)):
        raise AdmissionRefused("positive measured rates required")
    arms = 1 + treatments
    training = arms * 6_000_000
    evaluation = (arms + 2) * 288 * 5000
    hours = training / actor_rate / 3600 + evaluation / episode_rate / 3600 + 2.0
    estimate = math.ceil(hours * 60)
    timeout = max(3, 2 * estimate)
    reasons = []
    if treatments > (2 if actor_rate >= 2000 else 1):
        reasons.append("E4 allows only one treatment below 2000 hero transitions/s")
    if episode_rate < 800:
        reasons.append("disjoint full-policy H5000 episode rate below 800 world-steps/s")
    if hours > 10 or hours * 2 + 4 > 20:
        reasons.append("forecast exceeds total allocation/unit cap or reserve")
    if timeout > 600 or timeout / 60 * 2 + 4 > 20:
        reasons.append("required 2x-estimate timeout cannot fit 10h/20u with 4u reserve")
    return {
        "training_transitions": training,
        "evaluation_world_steps": evaluation,
        "controllers": arms + 2,
        "setup_export_hours": 2.0,
        "allocation_hours": hours,
        "units_at_2_per_hour": hours * 2,
        "reserve_units": 4.0,
        "estimate_minutes": estimate,
        "timeout_minutes": timeout,
        "timeout_liability_units": timeout / 60 * 2,
        "unit_cap": 20.0,
        "status": "NOT_DISPATCHABLE" if reasons else "FORECAST_FITS",
        "reasons": reasons,
    }


def arm_preconditions(arm: str, facts: Mapping[str, Any]) -> list[str]:
    """List scientific prerequisites, keeping unavailable mechanisms unranked."""
    if arm not in ARMS:
        raise AdmissionRefused("unknown E16 arm")
    missing = []
    if facts.get("E2_F_nominated_before_efficacy") is not True:
        missing.append("E2 F endpoint nominated before efficacy")
    if facts.get("E8_ready") is not True or facts.get("E12_parent") != "PASS":
        missing.append("E8-amended, E12-qualified common F parent")
    if arm == "G":
        if facts.get("E3") != "GO":
            missing.append("E3 GO")
        if facts.get("E14") != "PASS" or facts.get("E14_architecture") != facts.get(
            "G_architecture"
        ):
            missing.append("E14 PASS on the exact E3 geometry architecture")
    if arm == "R":
        if not isinstance(facts.get("U_upper90"), (int, float)) or facts["U_upper90"] < 40:
            missing.append("E10 UB90(U) >= 40")
        if facts.get("E7") != "TRANSFER_ADMISSIBLE":
            missing.append("E7 TRANSFER_ADMISSIBLE")
        if (
            not isinstance(facts.get("productive_coverage"), (int, float))
            or facts["productive_coverage"] < 0.163
        ):
            missing.append("productive confirmed coverage >= .163")
        tau = facts.get("tau_requirement")
        if not isinstance(tau, (int, float)) or not 0 <= tau <= 1:
            missing.append("(40+C)/confirmed gross supply <= 1")
    if arm == "S" and facts.get("E1") != "CONTINUE":
        missing.append("E1 CONTINUE with useful branch-rate evidence")
    if arm == "P" and facts.get("PLR_protocol_frozen") is not True:
        missing.append("whole-world PLR source cap/final natural phase protocol")
    if arm not in IMPLEMENTED:
        missing.append("implementation stub: cannot execute or rank")
    return missing


def bound_json(binding: Mapping[str, str], base: Path) -> dict[str, Any]:
    """Read a sidecar-complete receipt and compare its plan-pinned digest."""
    path = Path(binding["path"])
    if not path.is_absolute():
        path = base / path
    observed = verify_export(path)
    if observed != binding.get("sha256"):
        raise AdmissionRefused(f"input digest mismatch: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AdmissionRefused("bound receipt must be an object")
    return value


def require_execution_plan(path: str | Path, arm: str | None = None) -> dict[str, Any]:
    """Verify a frozen plan before loading worlds or touching an optimizer.

    Receipt adapters produce bl-e16-prerequisite/v1 objects referencing their
    original evidence exports; adapters cannot replace those original bytes.
    A NOT_DISPATCHABLE template is intentionally refused here.
    """
    path = Path(path)
    verify_export(path)
    plan = json.loads(path.read_text())
    if (
        plan.get("schema") != "bl-e16-plan/v1"
        or plan.get("status") != "ADMITTED"
        or plan.get("scientific") is not True
    ):
        raise AdmissionRefused("E16 plan is not admitted for scientific execution")
    arms = plan.get("arms")
    if not isinstance(arms, list) or not arms or arms[0] != "A" or len(set(arms)) != len(arms):
        raise AdmissionRefused("plan needs unique A control followed by treatments")
    if len(arms) not in (2, 3) or any(a not in IMPLEMENTED for a in arms):
        raise AdmissionRefused("only implemented A/G arms can execute")
    if arm is not None and arm not in arms:
        raise AdmissionRefused("requested arm is absent from frozen family")
    commit = plan.get("commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise AdmissionRefused("full source commit required")
    if plan.get("b8", {}).get("sha256") != B8_SHA256:
        raise AdmissionRefused("B8 comparator must be the immutable released s3 bytes")
    b8_path = path.parent / plan["b8"]["path"]
    if digest(b8_path) != B8_SHA256:
        raise AdmissionRefused("B8 comparator bytes mismatch")
    architecture = plan.get("G_architecture")
    if architecture not in {G_HISTORY, G_NO_HISTORY}:
        raise AdmissionRefused("exact E3 geometry variant required")
    if plan.get("use_history") is not (architecture == G_HISTORY):
        raise AdmissionRefused("geometry history flag differs from qualified architecture")
    package_root = Path(__file__).resolve().parents[2]
    required_sources = {
        f"research/bl_e16/{name}"
        for name in (
            "REGISTRATION.md",
            "REGISTRATION_ERRATUM_01.md",
            "SEED_RECIPE.md",
            "PARITY_RULE.md",
            "admission.py",
            "evaluate.py",
            "runner.py",
            "model.py",
            "training.py",
            "metrics.py",
            "bank.py",
            "spool.py",
            "checkpoint.py",
            "budget.py",
        )
    }
    # Include the actual borrowed execution closure; E2 is still being built.
    for directory in (
        "src",
        "research/bl_ppo",
        "research/bl_info_value",
        "research/apex_safety_20260926",
        "research/t34_speed_s2",
    ):
        required_sources.update(
            str(source.relative_to(package_root))
            for source in (package_root / directory).rglob("*.py")
            if not source.name.startswith("test_")
        )
    hashes = plan.get("source_hashes", {})
    if not required_sources.issubset(hashes):
        raise AdmissionRefused("running-source and registration closure must be pinned")
    for relative, expected_sha in hashes.items():
        source = Path(relative)
        if source.is_absolute() or ".." in source.parts:
            raise AdmissionRefused("source hash path must be repository-relative")
        if digest(package_root / source) != expected_sha:
            raise AdmissionRefused(f"running source identity differs: {relative}")
    parent = plan.get("parent", {})
    parent_path = path.parent / parent.get("path", "")
    if verify_export(parent_path) != parent.get("sha256"):
        raise AdmissionRefused("parent endpoint identity mismatch")
    receipts = {}
    for name in ("E2_F", "E12_parent", "E4", "episode_rate", "E3", "E14"):
        if name not in plan.get("receipts", {}):
            raise AdmissionRefused(f"missing hash-bound {name} evidence")
        receipt = bound_json(plan["receipts"][name], path.parent)
        if receipt.get("schema") != "bl-e16-prerequisite/v1" or receipt.get("kind") != name:
            raise AdmissionRefused(f"unsupported {name} prerequisite adapter")
        sources = receipt.get("sources")
        if not sources:
            raise AdmissionRefused(f"{name} adapter needs original evidence")
        for binding in sources:
            bound_json(binding, path.parent)
        receipts[name] = receipt
    if (
        receipts["E2_F"].get("parent_sha256") != parent["sha256"]
        or receipts["E2_F"].get("nominated_before_efficacy") is not True
    ):
        raise AdmissionRefused("parent nomination must precede efficacy readout")
    if (
        receipts["E12_parent"].get("parent_sha256") != parent["sha256"]
        or receipts["E12_parent"].get("status") != "PASS"
    ):
        raise AdmissionRefused("E12 qualification does not bind the common parent")
    if (
        receipts["E3"].get("architecture") != architecture
        or receipts["E14"].get("architecture") != architecture
    ):
        raise AdmissionRefused("E3, E14 and instantiated geometry architecture must match")
    if (
        receipts["E14"].get("full_path_p95_ms", float("inf")) > 11.688
        or receipts["E14"].get("overruns") != 0
    ):
        raise AdmissionRefused("E14 full path exceeds the fixed serving-cost limit")
    facts = {
        "E2_F_nominated_before_efficacy": True,
        "E8_ready": receipts["E2_F"].get("E8_ready"),
        "E12_parent": "PASS",
        "E3": receipts["E3"].get("status"),
        "E14": receipts["E14"].get("status"),
        "E14_architecture": receipts["E14"].get("architecture"),
        "G_architecture": plan.get("G_architecture"),
    }
    for selected in arms:
        reasons = arm_preconditions(selected, facts)
        if reasons:
            raise AdmissionRefused("; ".join(reasons))
    episode = receipts["episode_rate"]
    if (
        episode.get("horizon") != 5000
        or episode.get("disjoint") is not True
        or episode.get("full_policy") is not True
    ):
        raise AdmissionRefused("episode timing must be disjoint full-policy H5000")
    priced = forecast(
        float(receipts["E4"]["hero_transitions_per_second"]),
        float(episode["world_steps_per_second"]),
        len(arms) - 1,
    )
    if priced["status"] != "FORECAST_FITS" or plan.get("forecast") != priced:
        raise AdmissionRefused("forecast is missing, changed or NOT_DISPATCHABLE")
    if plan.get("critic_transitions") != 2_000_000 or plan.get("actor_transitions") != 4_000_000:
        raise AdmissionRefused("fixed critic/actor blocks cannot be resized")
    from research.bl_e16.bank import check_audit, validate_bank

    bank = bound_json(plan["bank"], path.parent)
    audit = bound_json(plan["audit"], path.parent)
    validate_bank(bank)
    check_audit(bank, audit)
    from research.bl_e16.budget import validate_reservation

    budget = bound_json(plan["study_budget"], path.parent)
    plan["_reservations"] = validate_reservation(budget, arms)
    plan["_base"] = str(path.parent)
    plan["_facts"] = facts
    plan["_bank"] = bank
    plan["_receipts"] = receipts
    return plan
