"""Measured searched-label divergence; unexplained mismatches fail closed."""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_decision(reference: dict, candidate: dict, *, evidence: dict | None = None) -> dict:
    """Classify every mismatch, requiring affirmative evidence for RNG/tie claims.

    Merely using a counter RNG does not explain a mismatch. An RNG attribution
    requires a replay with matched events which recovers the reference label.
    Until that experiment is recorded the disagreement remains an unexplained bug.
    """
    evidence = evidence or {}
    agreement = int(reference["action"]) == int(candidate["action"])
    classification = "agreement"
    if not agreement:
        classification = "bug"
        first, second = int(reference["action"]), int(candidate["action"])
        q = np.asarray(reference["q"], dtype=np.float32)
        margin = abs(float(q[first]) - float(q[second]))
        tolerance = 4 * max(abs(float(np.spacing(q[first]))), abs(float(np.spacing(q[second]))))
        outcomes = reference["outcomes"]
        counts_tied = outcomes[first]["alive_steps"] == outcomes.get(second, {}).get("alive_steps")
        if counts_tied and margin <= tolerance and evidence.get("identical_branch_outcomes"):
            classification = "float tie"
        elif evidence.get("matched_event_replay_recovers_label"):
            classification = "RNG-stream difference"
    errors = []
    for action, result in reference["outcomes"].items():
        actual = candidate["outcomes"].get(action)
        if actual is None:
            raise ValueError(f"candidate missing legal branch {action}")
        errors.append(
            {
                "action": int(action),
                "reference_mass_integral": result["mass_integral"],
                "candidate_mass_integral": actual["mass_integral"],
                "mass_integral_error": actual["mass_integral"] - result["mass_integral"],
                "alive_steps_error": actual["alive_steps"] - result["alive_steps"],
            }
        )
    return {
        "agreement": agreement,
        "classification": classification,
        "evidence": evidence,
        "reference_action": reference["action"],
        "candidate_action": candidate["action"],
        "branches": errors,
        "mean_absolute_mass_integral_error": float(
            np.mean([abs(row["mass_integral_error"]) for row in errors])
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict:
    """Return sample counts alongside agreement; a tiny preflight is not a gate."""
    count = len(rows)
    return {
        "decisions": count,
        "first_action_agreement": sum(row["agreement"] for row in rows) / max(1, count),
        "classification_counts": {
            label: sum(row["classification"] == label for row in rows)
            for label in ("agreement", "float tie", "RNG-stream difference", "bug")
        },
        "per_branch_mean_absolute_mass_integral_error": (
            float(np.mean([row["mean_absolute_mass_integral_error"] for row in rows]))
            if rows
            else None
        ),
        "qualification": "observed finite fixture sample only; no serving or registered verdict",
    }
