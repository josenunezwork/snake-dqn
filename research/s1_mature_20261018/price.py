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

from research.s1_mature_20261018 import bank, ledger

ROOT = Path(__file__).resolve().parents[2]


def binding(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def forecast(eval_path: Path, training_path: Path) -> dict[str, Any]:
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
    reasons = [
        "Full-size L4 full-replay checkpoint/export liability is unqualified; "
        "no finite complete upper bound is admitted.",
        "The real 12000-decision smoke exceeded the inherited K*W*N-1 counter bound "
        "during full-state barrier drain; the inherited production invariant is unchanged.",
    ]
    exceeded = [key for key, value in conservative.items() if value > ledger.USABLE_CAPS[key]]
    reasons.extend(f"Conditional forecast exceeds usable {key}" for key in exceeded)
    return {
        "schema": "s1-mature-pricing/v1",
        "status": "REFUSED",
        "dispatch_authorized": False,
        "evidence": {"evaluation": binding(eval_path), "training": binding(training_path)},
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
        "full_replay_transport": {"status": "UNQUALIFIED", "complete_upper_bound": None},
        "production_counter_qualification": {
            "status": "REFUSED",
            "required_invariant": "overshoot <= actors * worlds * learner_seats - 1",
            "tiny_export_roundtrip_is_not_production_qualification": True,
        },
        "mac_pricing": {
            "max_measured_episode_seconds": max(samples),
            "episode_factor": 2,
            "startup_seconds_per_identity": 600,
            "audit_and_preflight_worker_hours": 2,
        },
        "complete_upper_bound": None,
        "usable_caps": ledger.USABLE_CAPS,
        "nominal_caps": ledger.NOMINAL_CAPS,
        "refusal_reasons": reasons,
        "registration_status": "DRAFT_PENDING_OWNER_REVIEW",
        "training_preflight_status": training.get("status"),
        "next_step": "Resolve the counter compatibility and qualify the complete full-size "
        "transport bound prospectively inside the unchanged envelopes "
        "before approval and admission.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-preflight", type=Path, required=True)
    parser.add_argument("--training-preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = forecast(args.eval_preflight, args.training_preflight)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.out)}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
