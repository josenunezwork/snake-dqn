"""T34 label chunks with a student rollout policy under CPU process fan-out.

The ratified ``research.t34_exit_b8.label`` refuses ``nproc > 1`` with a
``--policy-factory``/``--student-checkpoint`` because a policy object cannot be
pickled into spawned workers.  This module keeps every other bound of that
entrypoint and sends only *references* (factory string + checkpoint path) to the
workers; each worker builds its own policy through the ratified
``_load_policy`` seam and runs the unchanged kernel ``run_worlds``.

Without a student (or with ``nproc == 1``) the call is delegated verbatim to the
ratified ``run_label_chunk``, so the B8 path is literally the same code.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping

from research.t34_exit_b8 import label as ratified

WORKER = "research.t34_claim_a.label:run_student_worlds"


def _has_student(config: Mapping[str, Any]) -> bool:
    return bool(config.get("policy_factory") or config.get("student_checkpoint"))


def student_provenance(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """SHA-bind the carrier and rollout checkpoints a chunk was labelled with."""
    if not _has_student(config):
        return None
    from research.t34_claim_a.student import sha256_file

    carrier = Path(str(config["checkpoint"]))
    rollout = Path(str(config.get("student_checkpoint") or carrier))
    return {
        "policy_factory": str(config.get("policy_factory") or ""),
        "carrier_checkpoint_sha256": sha256_file(carrier),
        "rollout_checkpoint_sha256": sha256_file(rollout),
        "rollout_is_carrier": sha256_file(carrier) == sha256_file(rollout),
    }


def run_student_worlds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fan-out worker: construct the student policy in-process, then run the kernel."""
    policy = ratified._load_policy(config)
    if policy is None:
        raise ValueError("student fan-out worker requires --policy-factory")
    runner = ratified._kernel_runner(str(config.get("kernel", "v1")))
    return runner({**config, "policy": policy})


def _validate_student_fanout(config: Mapping[str, Any]) -> tuple[int, int, str, int]:
    """Every bound of the ratified entrypoint, minus only its student refusal."""
    kernel = str(config.get("kernel", "v1"))
    if kernel not in {"v1", "v2"}:
        raise ValueError("kernel must be v1 or v2")
    if bool(config.get("preflight", False)) or bool(config.get("mac_full_round", False)):
        raise ValueError("student fan-out is a production CPU recipe, not preflight/full-round")
    threads = int(config.get("threads", 1))
    device = str(config.get("device", "cpu"))
    nproc = int(config.get("nproc", 1))
    max_worlds = int(config.get("max_worlds", config.get("worlds", 1)))
    if not 1 <= max_worlds <= 10000:
        raise ValueError("max_worlds must be positive")
    seconds = float(config.get("seconds", 10200))
    if not math.isfinite(seconds) or not 0 < seconds <= 10200:
        raise ValueError("label wall bound exceeds the registered limit")
    if not 1 <= int(config.get("frames", 5000)) <= 5000:
        raise ValueError("label frames must lie in [1, 5000]")
    if os.uname().sysname == "Darwin" and device == "cpu" and nproc * threads > 2:
        raise ValueError("Mac fanout is capped at two total CPU threads")
    if device != "cpu" or threads != 1:
        raise ValueError("label process fanout requires CPU and one thread per process")
    if int(config.get("horizon", 300)) != 300:
        raise ValueError("T34 labels require deterministic H=300")
    for key in ("round", "shard", "attempt"):
        if key not in config:
            raise ValueError(f"label chunk requires {key}")
    reference = config.get("policy_factory")
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("student fan-out requires --policy-factory module:callable")
    if config.get("student_checkpoint") and not Path(str(config["student_checkpoint"])).is_file():
        raise ValueError("student checkpoint does not exist")
    return threads, nproc, kernel, max_worlds


def run_label_chunk(config: Mapping[str, Any]) -> dict[str, Any]:
    """Ratified label chunk, extended with student CPU process fan-out."""
    if int(config.get("nproc", 1)) <= 1 or not _has_student(config):
        result = ratified.run_label_chunk(config)
        provenance = student_provenance(config)
        if provenance is not None:
            result["student"] = provenance
        return result
    threads, nproc, kernel, max_worlds = _validate_student_fanout(config)
    beta = config.get("beta")
    if beta is None:
        beta = {"r2": 0.5, "r3": 0.0}.get(str(config["round"]).lower(), 1.0)
    triggers = {
        "confirm_interval": 4 if str(config["round"]).lower() == "r0" else 8,
        **dict(config.get("triggers", {})),
    }
    for name in (
        "confirm_interval",
        "confirm_min_length",
        "low_length_probability",
        "death_window",
        "max_rewinds",
    ):
        if config.get(name) is not None:
            triggers[name] = config[name]
    from research.t34_exit_b8.process_fanout import run_fanout
    from research.t34_exit_b8.trajectory import make_world_rows

    effective_nproc = min(nproc, max_worlds)
    worker_config = {
        **config,
        "threads": 1,
        "device": "cpu",
        "beta": beta,
        "triggers": triggers,
        "worlds_per_process": int(config.get("worlds_per_process", 1)),
    }
    # Only picklable references cross the spawn boundary.
    worker_config.pop("policy", None)
    for key in ("checkpoint", "student_checkpoint", "output_dir"):
        if worker_config.get(key) is not None:
            worker_config[key] = str(worker_config[key])
    result = run_fanout(
        worker_config,
        runner=WORKER,
        rows=make_world_rows(config, max_worlds),
        nproc=effective_nproc,
    )
    result.update(
        {
            "schema": f"t34-exit-label-chunk/{kernel}",
            "kernel": kernel,
            "preflight": False,
            "mac_full_round": False,
            "round": str(config["round"]),
            "shard": int(config["shard"]),
            "attempt": int(config["attempt"]),
            "recipe": {
                "workers": 1,
                "nproc": effective_nproc,
                "threads_per_worker": threads,
                "device": "cpu",
                "horizon": int(config.get("horizon", 300)),
            },
            "student": student_provenance(config),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--policy-factory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--max-worlds", type=int, required=True)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--seconds", type=float, default=10200)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--nproc",
        type=int,
        default=(
            min(2, max(1, int(os.cpu_count() or 1)))
            if os.uname().sysname == "Darwin"
            else max(1, int(os.cpu_count() or 1))
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kernel", choices=("v1", "v2"), default="v2")
    parser.add_argument("--mac-full-round", action="store_true")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--mixes", help="comma-separated mix cycle (default: ratified bank)")
    parser.add_argument("--beta", type=float, choices=(0.0, 0.5, 1.0))
    parser.add_argument("--confirm-interval", type=int)
    parser.add_argument("--confirm-min-length", type=int)
    parser.add_argument("--low-length-probability", type=float)
    parser.add_argument("--death-window", type=int)
    parser.add_argument("--max-rewinds", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = {key: value for key, value in vars(args).items() if value is not None}
    result = run_label_chunk(config)
    ratified._write_once(args.out, result)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
