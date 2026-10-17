"""Reproduce a bounded Amdahl estimate from Mac timings and saved Colab counters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def project(drive: Path, results: Path) -> dict:
    """Charge shared B8 work, keep GPU time, and use each actor count's own counters."""
    before = json.loads((results / "results_final_reference.json").read_text())["measurement"]
    after = json.loads((results / "results_final_current.json").read_text())["measurement"]
    assert before["action_stream_sha256"] == after["action_stream_sha256"]
    bp, ap = before["phase_ms_per_frame"], after["phase_ms_per_frame"]
    shared = max(0.0, ap["learner_b8:shared_b_raster"] - bp["learner_b8:separate_b_raster"])
    old_host = before["dispatch_ms_per_frame"] - bp["b8-s3:forward"]
    new_host = after["dispatch_ms_per_frame"] - ap["b8-s3:forward"] + shared
    ratio = new_host / old_host
    input_hashes = {}

    def read(path: Path) -> str:
        payload = path.read_bytes()
        input_hashes[str(path.relative_to(drive))] = hashlib.sha256(payload).hexdigest()
        return payload.decode()

    configs = json.loads(read(drive / "colab_throughput_results.json"))["configs"]
    a4 = next(row for row in configs if row["actors"] == 4)
    rows = []
    for actors, directory, rate, authority in (
        (4, drive / a4["name"] / "attempt-001", a4["transitions_per_second"], "saved scored rate"),
        (
            6,
            drive / "extra_a6",
            7720.0,
            "owner-supplied approximate rate; sensitivity scales linearly",
        ),
    ):
        telemetry = [json.loads(line) for line in read(directory / "telemetry.jsonl").splitlines()]
        logs = [json.loads(line) for line in read(directory / "log.jsonl").splitlines()]
        final = telemetry[-1]
        log = next(row for row in reversed(logs) if row["transitions"] == final["transitions"])
        env, frozen, gpu = log["env_s"], final["frozen_forward_s"], final["frozen_forward_gpu_s"]
        host_share = (frozen - gpu) / env
        multiplier = 1.0 / (1.0 - host_share + host_share * ratio)
        rows.append(
            {
                "actors": actors,
                "baseline_tr_s": rate,
                "rate_authority": authority,
                "matched_transitions": final["transitions"],
                "env_s": env,
                "frozen_s": frozen,
                "frozen_gpu_s": gpu,
                "host_frozen_fraction": host_share,
                "projected_tr_s": rate * multiplier,
                "multiplier": multiplier,
                "zero_host_pool_ceiling_tr_s": rate / (1.0 - host_share),
            }
        )
    return {
        "schema": "t22-a100-projection/v1",
        "shared_b8_charge_ms": shared,
        "old_host_pool_ms": old_host,
        "new_host_pool_ms": new_host,
        "remaining_host_ratio": ratio,
        "formula": (
            "baseline_rate / (1 - host_frozen_fraction + "
            "host_frozen_fraction * remaining_host_ratio)"
        ),
        "assumptions": [
            "Mac relative host speedup transfers to x86; unmeasured",
            "Cumulative actor-time fractions approximate the rate window",
            "GPU frozen time and all other actor time unchanged",
            "No extra credit for runtime-control pruning outside pool dispatch",
            "No startup, transfer, GPU contention or scheduling penalty modeled",
        ],
        "target_tr_s": 20900,
        "isolated_learner_capacity_tr_s": 82.7 * 256,
        "rows": rows,
        "input_sha256": input_hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = project(args.drive_root, args.results)
    (args.results / "a100_projection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
