"""Conditional T22+T24 Amdahl sensitivity; no A100 execution or throughput claim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def project(before_path: Path, after_path: Path) -> dict:
    """Charge shared v8 once, retaining each saved N=3 row's distinct time shares."""
    here = Path(__file__).resolve().parent
    t22 = here.parent / "t22_frozen_pool_20261017"
    paths = [before_path, after_path, t22 / "a100_projection.json"]
    before, after, prior = [json.loads(path.read_text()) for path in paths]
    old_host_ms = prior["old_host_pool_ms"]
    t22_result = t22 / "results_final_current.json"
    paths.append(t22_result)
    t22_b8_v8_ms = json.loads(t22_result.read_text())["measurement"]["phase_ms_per_frame"][
        "b8-s3:v8_scalars"
    ]
    # Matched final telemetry counters from the same hash-bound saved Drive
    # inputs used by T22. v8_s also contains redecision / Q transfer, so credit
    # between zero and all of this timer rather than inventing its feature split.
    learner_v8_s = {4: 83.192, 6: 149.256}
    if len(before["cases"]) != len(after["cases"]):
        raise ValueError("before/after case counts must match")
    rows = []
    for old_case, new_case in zip(before["cases"], after["cases"]):
        assert old_case["body_length"] == new_case["body_length"]
        assert old_case["action_stream_sha256"] == new_case["action_stream_sha256"]
        key = "learner_plus_b8_v8_shared_candidate"
        ratio = median(new_case["timing_samples_ms"][key]) / median(
            old_case["timing_samples_ms"][key]
        )
        for row in prior["rows"]:
            actors = row["actors"]
            host_share = row["host_frozen_fraction"]
            old_denominator = 1.0 - host_share + host_share * prior["remaining_host_ratio"]
            b8_share = host_share * t22_b8_v8_ms / old_host_ms
            learner_share = learner_v8_s[actors] / row["env_s"]
            estimates = []
            for feature_fraction in (0.0, 1.0):
                denominator = old_denominator - (1.0 - ratio) * (
                    b8_share + feature_fraction * learner_share
                )
                estimates.append(row["baseline_tr_s"] / denominator)
            rows.append(
                {
                    "injected_body_length": old_case["body_length"],
                    "actors": actors,
                    "shared_v8_ratio": ratio,
                    "baseline_tr_s": row["baseline_tr_s"],
                    "matched_transitions": row["matched_transitions"],
                    "learner_v8_s": learner_v8_s[actors],
                    "rate_authority": row["rate_authority"],
                    "t22_projected_tr_s": row["projected_tr_s"],
                    "pool_only_credit_tr_s": estimates[0],
                    "full_learner_timer_credit_tr_s": estimates[1],
                    "upper_vs_mac_5214": estimates[1] / 5214.0,
                    "learner_v8_timer_share": learner_share,
                    "b8_v8_share_of_original_actor_time": b8_share,
                }
            )
    return {
        "schema": "t24-a100-projection/v1",
        "authority": "Conditional sensitivity, not an A100 measurement",
        "formula": (
            "baseline_rate / (1-h+h*r22 - (1-rv8)*(h*b8_v8_ms/old_host_ms + f*v8_s/env_s))"
        ),
        "f": "0 credits only B8 feature savings; 1 optimistically scales entire learner v8 timer",
        "assumptions": [
            "Same saved a4/a6 N=3 counters and input hashes as T22; no new Colab execution",
            "Mac relative shared-kernel speedup transfers to x86; unmeasured",
            "Shared-call cost allocated proportionally to old learner and B8 feature costs",
            "T22 host-pool savings retained; B8 shared v8 charged exactly once",
            "No credit for lazy vector-v8, GPU forward, raster, runtime or replay savings",
            "Learner v8 timer includes Q transfer/redecision; feature fraction is unknown",
            "L1200 ratios are sensitivity on old time shares, not a late-game throughput forecast",
            "Scheduling, device contention, startup and transfer overheads remain unmeasured",
        ],
        "mac_n3_baseline_tr_s": 5214,
        "target_tr_s": 20900,
        "isolated_learner_capacity_tr_s": prior["isolated_learner_capacity_tr_s"],
        "rows": rows,
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "t22_drive_input_sha256": prior["input_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.before, args.after)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["rows"], indent=2))
