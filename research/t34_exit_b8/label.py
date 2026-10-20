"""Bounded T34 label-chunk entrypoint and cross-runtime replay report CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def _kernel_runner(kernel: str) -> Any:
    """Select the additive v2 teacher without changing historical v1 labels."""
    if kernel == "v1":
        from research.t34_exit_b8.trajectory import run_worlds

        return run_worlds
    if kernel == "v2":
        from research.t34_exit_b8.teacher_v2 import run_worlds

        return run_worlds
    raise ValueError("kernel must be v1 or v2")


def _cpu_ceiling(threads: int) -> None:
    """Install the cap only after legacy import-time env mutations finish."""
    if threads < 1:
        raise ValueError("threads must be positive")
    for key, value in {
        "NUMBA_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    import numba

    # Phase-0/T33 modify NUMBA_NUM_THREADS while importing.  Do that before
    # creating a pool so later compiler config reloads stay coherent.
    import research.phase0_search_20261019.teacher  # noqa: F401
    import research.t33_batched_teacher.teacher  # noqa: F401

    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    numba.config.reload_config()
    numba.set_num_threads(threads)


def _load_policy(config: Mapping[str, Any]) -> Any | None:
    reference = config.get("policy_factory")
    if not reference:
        if config.get("student_checkpoint"):
            raise ValueError("--student-checkpoint requires --policy-factory")
        return None
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("--policy-factory must be module:callable")
    module, name = reference.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module), name)
        policy = factory(
            device=str(config["device"]),
            checkpoint=Path(str(config.get("student_checkpoint") or config["checkpoint"])),
        )
        if not callable(getattr(policy, "evaluate", None)) or not callable(
            getattr(policy, "evaluate_rows", None)
        ):
            raise TypeError(
                "policy must implement evaluate(world) and evaluate_rows(sim, rows, net)"
            )
        return policy
    except Exception as exc:
        raise RuntimeError(f"unsupported rollout-policy adapter {reference!r}") from exc


def run_label_chunk(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect an explicitly bounded label chunk through the shared trajectory API."""
    kernel = str(config.get("kernel", "v1"))
    mac_full_round = bool(config.get("mac_full_round", False))
    preflight = bool(config.get("preflight", False))
    threads = int(config.get("threads", 5))
    device = str(config.get("device", "cuda"))
    nproc = int(config.get("nproc", 1))
    max_worlds = int(config.get("max_worlds", config.get("worlds", 1)))
    if not 1 <= max_worlds <= 10000:
        raise ValueError("max_worlds must be positive")
    seconds = float(config.get("seconds", 600 if preflight else 10200))
    if not math.isfinite(seconds) or not 0 < seconds <= (600 if preflight else 10200):
        raise ValueError("label wall bound exceeds the registered limit")
    if not 1 <= int(config.get("frames", 5000)) <= 5000:
        raise ValueError("label frames must lie in [1, 5000]")
    if preflight:
        if max_worlds > 2 or int(config.get("frames", 5000)) > 500:
            raise ValueError("A6 label preflight is capped at two worlds and 500 frames")
        if threads > 1 or device != "cpu":
            raise ValueError("A6 label preflight uses CPU and one thread per process")
    elif mac_full_round:
        if kernel != "v2" or device != "cpu" or not 1 <= threads <= 2:
            raise ValueError("Mac full rounds require v2, CPU, and one or two CPU threads")
    elif device == "cuda" and nproc == 1 and threads != 5:
        raise ValueError("production labels require CUDA and five Numba threads per worker")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if os.uname().sysname == "Darwin" and device == "cpu" and nproc * threads > 2:
        raise ValueError("Mac fanout is capped at two total CPU threads")
    if nproc > 1 and (device != "cpu" or threads != 1):
        raise ValueError("label process fanout requires CPU and one thread per process")
    if nproc > 1 and (config.get("policy_factory") or config.get("student_checkpoint")):
        raise ValueError("process fanout does not support a rollout-policy adapter")
    if int(config.get("horizon", 300)) != 300:
        raise ValueError("T34 labels require deterministic H=300")
    for key in ("round", "shard", "attempt"):
        if key not in config:
            raise ValueError(f"label chunk requires {key}")
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
    if nproc > 1 and not bool(config.get("_fanout_worker", False)):
        from research.t34_exit_b8.process_fanout import run_fanout
        from research.t34_exit_b8.trajectory import make_world_rows

        effective_nproc = min(nproc, max_worlds)
        result = run_fanout(
            {
                **config,
                "threads": 1,
                "device": "cpu",
                "beta": beta,
                "triggers": triggers,
                "worlds_per_process": int(config.get("worlds_per_process", 1)),
            },
            runner=(
                "research.t34_exit_b8.teacher_v2:run_worlds"
                if kernel == "v2"
                else "research.t34_exit_b8.trajectory:run_worlds"
            ),
            rows=make_world_rows(config, max_worlds),
            nproc=effective_nproc,
        )
        nproc = effective_nproc
    else:
        _cpu_ceiling(threads)
        import numba

        # Phase-0 compatibility imports can rewrite these defaults; the configured
        # limit remains the actual Numba ceiling and labels use the recipe device.
        os.environ["SNAKE_DQN_DEVICE"] = device
        numba.set_num_threads(threads)
        result = _kernel_runner(kernel)(
            {
                **config,
                "policy": _load_policy(config),
                "max_worlds": max_worlds,
                "beta": beta,
                "triggers": triggers,
                "worlds_per_process": (
                    min(int(config.get("worlds_per_process", 1)), max_worlds)
                    if device == "cpu"
                    else 256
                ),
            }
        )
    result.update(
        {
            "schema": f"t34-exit-label-chunk/{kernel}",
            "kernel": kernel,
            "preflight": preflight,
            "mac_full_round": mac_full_round,
            "round": str(config["round"]),
            "shard": int(config["shard"]),
            "attempt": int(config["attempt"]),
            "recipe": {
                "workers": 1,
                "nproc": nproc,
                "threads_per_worker": threads,
                "device": device,
                "horizon": int(config.get("horizon", 300)),
            },
            **({"fanout": result["fanout"]} if "fanout" in result else {}),
        }
    )
    return result


def _write_once(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with path.with_name(path.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(hashlib.sha256(payload.encode()).hexdigest() + "\n")
    with path.with_name(path.name + ".complete").open("x", encoding="utf-8") as stream:
        stream.write("complete\n")


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
    parser.add_argument("--kernel", choices=("v1", "v2"), default="v1")
    parser.add_argument("--mac-full-round", action="store_true")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--beta", type=float, choices=(0.0, 0.5, 1.0))
    parser.add_argument("--confirm-interval", type=int)
    parser.add_argument("--confirm-min-length", type=int)
    parser.add_argument("--low-length-probability", type=float)
    parser.add_argument("--death-window", type=int)
    parser.add_argument("--max-rewinds", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run_label_chunk(vars(args))
    _write_once(args.out, result)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
