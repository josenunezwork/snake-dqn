"""Mac-only serial-oracle and lockstep profiling with immutable receipts.

Run each arm in a fresh interpreter with one CPU thread. The serial modules are
verbatim fixtures, loaded with their own policy and engine dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/t34_shadow_batching"
SERIAL_HASHES = {
    "policy_serial.py": "ae0be54e70957fa3936b0b950cdc20fcf1b16fdfce3852c9618b110103c08635",
    "engine_serial.py": "ad1324eeffc0b82c6b8b8eca6fc1ffd328e8a0f8213d0d24952506cf161feb05",
    "teacher_v2_serial.py": "db5f58add590dcf56146c1057e154231b96eac377a40d45bc4b5d8d7de0a9d7d",
}


def serial_module() -> Any:
    """Load a hash-pinned oracle without importing candidate policy/engine."""
    aliases = ("policy", "engine", "teacher_v2")
    saved = {name: sys.modules.get(f"research.t34_exit_b8.{name}") for name in aliases}
    try:
        for name in aliases:
            filename = f"{name}_serial.py"
            path = FIXTURES / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != SERIAL_HASHES[filename]:
                raise RuntimeError(f"serial oracle changed: {filename}")
            private = f"_t34_frozen_{name}"
            spec = importlib.util.spec_from_file_location(private, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[private] = module
            sys.modules[f"research.t34_exit_b8.{name}"] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in saved.items():
            key = f"research.t34_exit_b8.{name}"
            if prior is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prior


def export_hashes(path: Path) -> dict[str, str]:
    """Bind all exported world payloads and sidecars by relative path."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.glob("world-*.json*"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("serial", "batched"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--namespace", default="t34-shadow-batch-profile/v1")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("this receipt is restricted to Mac CPU")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.out.exists():
        raise FileExistsError(args.out)
    source_names = ("teacher_v2.py", "policy.py", "engine.py", "lockstep.py")
    sources = [
        ROOT / "research/t34_exit_b8/profile_shadow_batching.py",
        ROOT / "research/t34_exit_b8/trajectory.py",
        ROOT / "research/t34_shadow/shadow.py",
    ]
    sources += (
        [FIXTURES / name for name in SERIAL_HASHES]
        if args.implementation == "serial"
        else [ROOT / "research/t34_exit_b8" / name for name in source_names]
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    start_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.implementation == "serial":
        module = serial_module()
    else:
        from research.t34_exit_b8 import teacher_v2 as module
    power = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    lid = subprocess.check_output(["ioreg", "-r", "-k", "AppleClamshellState"], text=True)
    if "AC Power" not in power or '"AppleClamshellState" = No' not in lid:
        raise RuntimeError("profiling requires AC power and an open lid")
    # Exclude all-mix, full-width import/Numba warm-up on a separate bank.
    warm = args.out.with_suffix(".warmup-worlds")
    warm_result = module.run_worlds(
        {
            "checkpoint": args.checkpoint,
            "output_dir": warm,
            "namespace": "t34-shadow-batch-warmup/v1",
            "mixes": ("scripted", "frozen", "mixed"),
            "max_worlds": args.worlds,
            "worlds_per_process": args.worlds,
            "frames": 2,
            "shadow_depth": 2,
            "horizon": 2,
            "seconds": 180,
            "device": "cpu",
        }
    )
    if not warm_result["complete"]:
        raise RuntimeError(f"warm-up did not complete: {warm_result.get('invalid_reason')}")
    worlds_dir = args.out.with_suffix(".worlds")
    config = {
        "checkpoint": args.checkpoint,
        "output_dir": worlds_dir,
        "namespace": args.namespace,
        "mixes": ("scripted", "frozen", "mixed"),
        "max_worlds": args.worlds,
        "worlds_per_process": args.worlds,
        "frames": args.frames,
        "horizon": 300,
        "shadow_depth": 300,
        "beta": 1.0,
        "device": "cpu",
        "seconds": args.seconds,
    }
    started = time.monotonic()
    result = module.run_worlds(config)
    elapsed = time.monotonic() - started
    replay_results = []
    if args.replay:
        from research.t34_exit_b8.exports import load_world
        from research.t34_exit_b8.replay import replay_world

        for path in sorted(worlds_dir.glob("world-*.json")):
            replay = replay_world(load_world(path), args.checkpoint)
            replay_results.append(
                {"world": path.name, "accepted": replay.accepted, "mismatch": replay.mismatch}
            )
    final_source_sha256 = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    source_unchanged = final_source_sha256 == source_sha256
    receipt = {
        "schema": "t34-shadow-batching-profile/v1",
        "implementation": args.implementation,
        "argv": sys.argv,
        "start_commit": start_commit,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "platform": platform.platform(),
        "threads": 1,
        "warmup_excluded": True,
        "warmup_worlds": args.worlds,
        "warmup_mixes": ["scripted", "frozen", "mixed"],
        "replay_requested": args.replay,
        "shape": {"worlds": args.worlds, "frames": args.frames, "horizon": 300},
        "elapsed_seconds": elapsed,
        "result": result,
        "exports": export_hashes(worlds_dir),
        "replay": replay_results,
        "source_sha256": source_sha256,
        "final_source_sha256": final_source_sha256,
        "source_unchanged": source_unchanged,
        "serial_sha256": SERIAL_HASHES,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if not source_unchanged:
        raise RuntimeError("profile source changed during collection; diagnostic receipt only")
    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "complete": result["complete"],
                "elapsed_seconds": elapsed,
                "row_steps_per_second": result["active_row_steps_per_second"],
                "worlds": result["worlds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
