"""Price the fixed S1 plan locally; unknown final-export liability is a refusal.

This command never contacts Colab. A small CPU adapter smoke cannot qualify the
production counter boundary or final-export transfer from an L4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.s1_mature_20261018 import bank, final_artifacts, ledger

ROOT = Path(__file__).resolve().parents[2]


def binding(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def forecast(
    eval_path: Path,
    training_path: Path,
    transport_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate the conditional forecast and keep unresolved liability explicit.

    The conditional case reserves one from-scratch operational rerun for each
    recipe.  It is a budget liability, not permission to resume or to select an
    intermediate checkpoint.
    """
    timing, training = json.loads(eval_path.read_text()), json.loads(training_path.read_text())
    allocation = bank.load_world_bank()
    if timing.get("status") != "completed" or timing.get("episode_count", 0) < 10:
        raise ValueError("at least ten completed disjoint timing worlds are required")
    if timing.get("workers") != 2:
        raise ValueError("price the registered two-worker evaluator")
    if timing["identity"]["world_hashes"] != [
        world["world_semantic_sha256"] for world in allocation["timing_worlds"]
    ]:
        raise ValueError("timing did not use the registered disjoint fixture")
    for relative, digest in timing["identity"]["source_manifest"].items():
        if binding(ROOT / relative)["sha256"] != digest:
            raise ValueError(f"timing source changed: {relative}")
    samples = timing["parallel_elapsed_seconds"]
    if len(samples) != timing["episode_count"] or any(
        type(item) not in (int, float) or not math.isfinite(item) or item <= 0 for item in samples
    ):
        raise ValueError("episode timings must be finite and positive")
    # Twice the slowest measured world, plus ten minutes of warmup/serialization
    # per identity at two reserved workers; audit and all local preflights get an
    # additional two worker-hours. The watchdog enforces this allocation bound.
    episode_worker_hours = 1920 * 2 * max(samples) / 3600
    mac_hours = episode_worker_hours + 15 * 2 * 600 / 3600 + 2.0
    mac_elapsed = mac_hours / 2
    reasons = []
    counter_pass = training.get("status") == "PASS" and training.get("counter_admission") == "PASS"
    counter_sources = training.get("identity", {}).get("source_manifest", {})
    counter_pass = counter_pass and all(
        counter_sources.get(relative) == binding(ROOT / relative)["sha256"]
        for relative in (
            "research/redesign_m3c_20261008/train_m3c.py",
            "research/s1_mature_20261018/train.py",
        )
    )
    if not counter_pass:
        reasons.append("Real M3-C training preflight has not passed the unchanged counter bound.")
    transport: dict[str, Any] = {
        "status": "UNQUALIFIED",
        "measured_export_bytes_per_second_lower_bound": None,
        "required_cli_acceptance_field": "s1_final_artifact_transfer.bytes_per_second_lower_bound",
        "export_liability_formula": "final_bundle_bytes_upper_bound / "
        "measured_export_bytes_per_second_lower_bound",
        "excluded_by_registration": "full-state, replay, optimizer, actor, RNG, and mid-run "
        "checkpoint exports; they cannot support an exact resume or decision.",
        "per_attempt_export_cap_bytes": final_artifacts.MAX_FINAL_EXPORT_BYTES,
        "primary_attempts": 10,
        "conditional_attempts": 12,
        "primary_export_cap_bytes": 10 * final_artifacts.MAX_FINAL_EXPORT_BYTES,
        "conditional_export_cap_bytes": 12 * final_artifacts.MAX_FINAL_EXPORT_BYTES,
    }
    if transport_path is not None:
        try:
            transport = final_artifacts.transport_bound(json.loads(transport_path.read_text()))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reasons.append(f"Final-artifact transport refused: {exc}")
    if transport.get("status") != "PASS":
        reasons.append(
            "Final-only L4 export liability is unqualified; "
            "no finite complete upper bound is admitted."
        )
    else:
        for arm, row in transport["qualified_arms"].items():
            base = (30 if arm == "n3" else 68) / 60
            if (
                base + row["transport_seconds_upper_bound"] / 3600
                > (60 if arm == "n3" else 136) / 60
            ):
                reasons.append(f"{arm} training plus final-only export exceeds unchanged timeout")
    qualified_transport = transport.get("status") == "PASS"
    per_arm_transport = (
        {
            arm: transport["qualified_arms"][arm]["transport_seconds_upper_bound"] / 3600
            for arm in ("n3", "n1")
        }
        if qualified_transport
        else None
    )
    # Primary plan: five N3 and five N1 runs.  Conditional case: one additional
    # from-scratch run of each recipe, allowed only after an operational failure.
    primary_training = 5 * (30 + 68) / 60
    conditional_training = 6 * (30 + 68) / 60
    primary_transport = 5 * sum(per_arm_transport.values()) if per_arm_transport else None
    conditional_transport = 6 * sum(per_arm_transport.values()) if per_arm_transport else None
    planning = {
        "units": (
            2 * (conditional_training + conditional_transport) + 4
            if conditional_transport is not None
            else None
        ),
        "gpu_allocation_hours": (
            conditional_training + conditional_transport + 2
            if conditional_transport is not None
            else None
        ),
        "mac_worker_hours": mac_hours,
        "elapsed_hours": (
            conditional_training + conditional_transport + 2 + mac_elapsed
            if conditional_transport is not None
            else None
        ),
    }
    # The ledger reserves hard manifest timeout plus a separate five-minute
    # cleanup tail.  Export transport must fit the timeout above, so it is not
    # added a second time here.
    hard_training = 6 * ((60 + 5) + (136 + 5)) / 60
    hard_reservation = {
        "units": 2 * hard_training + 4,
        "gpu_allocation_hours": hard_training + 2,
        "mac_worker_hours": mac_hours,
        "elapsed_hours": hard_training + 2 + mac_elapsed,
    }
    exceeded = [key for key, value in hard_reservation.items() if value > ledger.USABLE_CAPS[key]]
    reasons.extend(f"Conditional forecast exceeds usable {key}" for key in exceeded)
    return {
        "schema": "s1-mature-pricing/v1",
        "status": "REFUSED" if reasons else "PASS",
        "dispatch_authorized": False,
        "evidence": {
            "evaluation": binding(eval_path),
            "training": binding(training_path),
            **({"transport": binding(transport_path)} if transport_path is not None else {}),
        },
        "qualified_q0b_planning": {
            "evidence": binding(ROOT / "research/q0q1_20261017/Q0B_A3_REQUALIFICATION_20261018.md"),
            "n3_estimate_minutes": 30,
            "n1_estimate_minutes": 68,
            "estimated_training_gpu_hours": 5 * (30 + 68) / 60,
            "estimated_training_units_at_2_per_hour": 2 * 5 * (30 + 68) / 60,
            "limitation": "Historical training estimates exclude final-only transfer; the current "
            "receipt must bind final size and measured CLI throughput.",
        },
        "conditional_planning_forecast": {
            **planning,
            "primary_training_gpu_hours": primary_training,
            "primary_transport_gpu_hours": primary_transport,
            "conditional_training_gpu_hours": conditional_training,
            "conditional_transport_gpu_hours": conditional_transport,
            "known_base_gpu_hours_before_setup_export_cleanup": conditional_training + 2,
            "known_base_units_before_setup_export_cleanup": 2 * (conditional_training + 2),
            "known_cleanup_gpu_hours": 12 * 5 / 60,
            "known_base_plus_cleanup_gpu_hours": conditional_training + 2 + 12 * 5 / 60,
            "known_base_plus_cleanup_units": 2 * (conditional_training + 2 + 12 * 5 / 60),
            "reruns": {"n3": 1, "n1": 1},
        },
        "conditional_hard_reservation": hard_reservation,
        "final_artifact_transport": transport,
        "production_counter_qualification": {
            "status": "PASS" if counter_pass else "REFUSED",
            "required_invariant": "overshoot <= actors * worlds * learner_seats - 1",
            "tiny_export_roundtrip_is_not_production_qualification": True,
        },
        "mac_pricing": {
            "max_measured_episode_seconds": max(samples),
            "episode_factor": 2,
            "startup_seconds_per_identity": 600,
            "audit_and_preflight_worker_hours": 2,
        },
        "complete_upper_bound": None if reasons else hard_reservation,
        "usable_caps": ledger.USABLE_CAPS,
        "nominal_caps": ledger.NOMINAL_CAPS,
        "refusal_reasons": reasons,
        "registration_status": "DRAFT_PENDING_OWNER_REVIEW",
        "training_preflight_status": training.get("status"),
        "next_step": "Obtain the current-source CLI acceptance receipt for final-only L4 exports, "
        "then "
        "resolve the conditional one-rerun-per-arm hard-reservation cap conflict before admission.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-preflight", type=Path, required=True)
    parser.add_argument("--training-preflight", type=Path, required=True)
    parser.add_argument("--transport-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = forecast(args.eval_preflight, args.training_preflight, args.transport_receipt)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.out)}))
    return 1 if result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
