"""Price the fixed S1 plan locally; unknown full-state liability is a refusal.

This command never contacts Colab. A small CPU adapter smoke cannot qualify the
production full-state counter boundary or a multi-GiB RPC transfer on an L4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.s1_mature_20261018 import bank, checkpoint_sizing, ledger

ROOT = Path(__file__).resolve().parents[2]


def binding(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def forecast(
    eval_path: Path,
    training_path: Path,
    transport_path: Path | None = None,
    sizing_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate the conditional forecast and keep unresolved liability explicit."""
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
    training_gpu = 5 * ((60 + 5) + (136 + 5)) / 60
    conservative = {
        "units": 2 * training_gpu + 4,
        "gpu_allocation_hours": training_gpu + 2,
        "mac_worker_hours": mac_hours,
        "elapsed_hours": training_gpu + 2 + mac_elapsed,
    }
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
        "complete_upper_bound": None,
        "measured_export_bytes_per_second_lower_bound": None,
        "required_cli_acceptance_field": "s1_full_replay_transfer.bytes_per_second_lower_bound",
        "capacity": {arm: checkpoint_sizing.capacity(arm) for arm in ("n3", "n1")},
        "export_liability_formula": "5 * (4 * archive_N3_bytes + final_N3_bytes + "
        "4 * archive_N1_bytes + final_N1_bytes) / measured_export_bytes_per_second_lower_bound",
        "archive_bound_formula": "ceil(snapshot_bytes_upper_bound * 1.01) + 1 MiB",
        "snapshot_count_per_run": 4,
        "snapshot_targets": list(checkpoint_sizing.SNAPSHOTS),
        "additional_liability": "full-size snapshot capture/serialization/hash/archive/chunk/copy "
        "preparation for every snapshot and final archive; deferred barrier packets included",
    }
    sizing_path = (
        sizing_path or ROOT / "research/s1_mature_20261018/CHECKPOINT_SIZE_MEASUREMENT.json"
    )
    if sizing_path.is_file():
        sizes = json.loads(sizing_path.read_text())
        for arm in ("n3", "n1"):
            if (
                sizes["arms"][arm]["fixed_capacity_array_bytes"]
                != checkpoint_sizing.capacity(arm)["fixed_capacity_array_bytes"]
            ):
                raise ValueError(f"{arm} sizing receipt no longer matches the production capacity")
        transport["checkpoint_size_receipt"] = binding(sizing_path)
        transport["checkpoint_planning_sizes"] = sizes["arms"]
        transport["ten_run_snapshot_export_bytes_planning_upper_bound"] = 5 * sum(
            item["four_snapshot_export_bytes_planning_upper_bound"]
            for item in sizes["arms"].values()
        )
        transport["snapshot_export_gpu_hours_formula"] = (
            f"{transport['ten_run_snapshot_export_bytes_planning_upper_bound']} / "
            "measured_export_bytes_per_second_lower_bound / 3600; "
            "plus ten final archives and all preparation"
        )
    if transport_path is not None:
        try:
            transport = checkpoint_sizing.transport_bound(
                json.loads(transport_path.read_text()), timing_sha256=binding(eval_path)["sha256"]
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reasons.append(f"Full-replay transport refused: {exc}")
    if transport.get("status") != "PASS":
        reasons.append(
            "Full-size L4 full-replay checkpoint/export liability is unqualified; "
            "no finite complete upper bound is admitted."
        )
    else:
        for arm, row in transport["qualified_arms"].items():
            if row["per_run_gpu_hours_upper_bound"] > (60 if arm == "n3" else 136) / 60:
                reasons.append(f"{arm} training plus preparation/export exceeds unchanged timeout")
    exceeded = [key for key, value in conservative.items() if value > ledger.USABLE_CAPS[key]]
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
            "limitation": "Historical training estimates exclude the newly required "
            "full-replay transport qualification.",
        },
        "conditional_reserved_forecast": conservative,
        "full_replay_transport": transport,
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
        "complete_upper_bound": None if reasons else conservative,
        "usable_caps": ledger.USABLE_CAPS,
        "nominal_caps": ledger.NOMINAL_CAPS,
        "refusal_reasons": reasons,
        "registration_status": "DRAFT_PENDING_OWNER_REVIEW",
        "training_preflight_status": training.get("status"),
        "next_step": "Qualify production-size snapshot preparation and CLI acceptance measured "
        "L4 export throughput inside unchanged timeouts and envelopes "
        "before approval and admission.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-preflight", type=Path, required=True)
    parser.add_argument("--training-preflight", type=Path, required=True)
    parser.add_argument("--transport-receipt", type=Path)
    parser.add_argument("--checkpoint-sizing", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = forecast(
        args.eval_preflight, args.training_preflight, args.transport_receipt, args.checkpoint_sizing
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.out)}))
    return 1 if result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
