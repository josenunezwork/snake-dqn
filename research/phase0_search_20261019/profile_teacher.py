"""Bounded wall-time and category profiling for the Phase-0 search teacher.

The fixture world is built outside the measured interval.  One small warm-up
decision is also run before any reported round so Numba and Torch initialization
does not contaminate the receipt.  The default CPU path uses one native thread,
and no process, Colab, pod, tmux, or serving operation is started here.

Examples::

    ./venv/bin/python -m research.phase0_search_20261019.profile_teacher \
        --implementation both --mixes scripted frozen mixed --length 800 \
        --rounds 3 --out /tmp/phase0-teacher-profile.json

    ./venv/bin/python -m research.phase0_search_20261019.profile_teacher \
        --implementation batched --device mps --mixes mixed --rounds 3
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

from research.phase0_search_20261019 import preflight, teacher
from research.phase0_search_20261019.teacher import SearchConfig, SearchTeacher, configure_cpu

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = preflight.B8
DEFAULT_MIXES = ("scripted", "frozen", "mixed")
CATEGORY_NAMES = (
    "sim_stepping",
    "observation_building",
    "b8_forward",
    "frozen_opponent_forward",
    "scripted_opponent_logic",
    "clone_snapshot",
)


class CategoryTimer:
    """Collect exclusive elapsed time around one existing function."""

    def __init__(self, name: str, elapsed: dict[str, float], calls: dict[str, int]) -> None:
        self.name = name
        self.elapsed = elapsed
        self.calls = calls

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.elapsed[self.name] += time.perf_counter() - started
                self.calls[self.name] += 1

        wrapped.__name__ = getattr(function, "__name__", self.name)
        wrapped.__doc__ = getattr(function, "__doc__", None)
        return wrapped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _config(path: Path | None) -> SearchConfig:
    if path is None:
        return SearchConfig()
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "search" in raw:
        raw = raw["search"]
    return SearchConfig(**raw)


def _rows(bank_path: Path, mixes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    bank = json.loads(bank_path.read_text())
    rows = {str(row["mix"]): row for row in bank["fixtures"]["timing"]}
    missing = [mix for mix in mixes if mix not in rows]
    if missing:
        raise ValueError(f"timing fixture lacks requested mix(es): {missing}")
    return {mix: rows[mix] for mix in mixes}


@contextlib.contextmanager
def _instrument(elapsed: dict[str, float], calls: dict[str, int]) -> Iterator[None]:
    """Patch only leaf category functions for the duration of one decision."""
    import src.simd_env.batch_sim as batch_sim
    import src.simd_env.ego2s_policy as ego2s_policy
    import src.simd_env.ego_raster as ego_raster
    import src.simd_env.fast_anchors as fast_anchors
    import src.simd_env.vector61_policy as vector61_policy

    wrappers = (
        (batch_sim.BatchSim, "step", "sim_stepping"),
        (ego2s_policy, "forward_q", "b8_forward"),
        (vector61_policy.Vector61SimdPolicy, "q_values", "frozen_opponent_forward"),
        (fast_anchors.FastProfileAnchorSimdPolicy, "actions", "scripted_opponent_logic"),
        (teacher, "_replicate", "clone_snapshot"),
    )
    patches = []
    for owner, attribute, name in wrappers:
        original = getattr(owner, attribute)
        patches.append(
            patch.object(owner, attribute, CategoryTimer(name, elapsed, calls).wrap(original))
        )
    # ego2s_policy imported this function into its module namespace, while the
    # teacher imports it from ego_raster at call time.  Patch both aliases with
    # one wrapper so all raster builds are counted once.
    raster_wrapper = CategoryTimer("observation_building", elapsed, calls).wrap(
        ego_raster.build_ego_raster
    )
    patches.extend(
        (
            patch.object(ego_raster, "build_ego_raster", raster_wrapper),
            patch.object(ego2s_policy, "build_ego_raster", raster_wrapper),
        )
    )
    with _patches(patches):
        yield


class _patches:
    """Apply a tuple of unittest mock patches without importing a test runner."""

    def __init__(self, patches: list[Any]) -> None:
        self.patches = patches

    def __enter__(self) -> None:
        for item in self.patches:
            item.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for item in reversed(self.patches):
            item.stop()


def _fixture(row: dict[str, Any], checkpoint: Path, length: int) -> Any:
    from src.simd_env.grid_scenarios import inject_serpentines

    world = teacher.build_world(row, checkpoint)
    inject_serpentines([world.sim], [length, 300, 400, 200, 100, 100], box_height=11)
    world.sim.check_grid_invariants()
    return world


def _warmup(row: dict[str, Any], checkpoint: Path, config: SearchConfig, device: str) -> None:
    world = _fixture(row, checkpoint, 200)
    warm_config = replace(config, rollouts=1, horizon=2, length_gate=0)
    teacher_instance = SearchTeacher(warm_config, implementation="batched", device=device)
    teacher.step_world(world, teacher_instance)


def _measure_one(
    row: dict[str, Any],
    checkpoint: Path,
    config: SearchConfig,
    length: int,
    implementation: str,
    device: str,
    round_index: int,
) -> dict[str, Any]:
    world = _fixture(row, checkpoint, length)
    instance = SearchTeacher(
        replace(config, length_gate=0), implementation=implementation, device=device
    )
    elapsed = {name: 0.0 for name in CATEGORY_NAMES}
    calls = {name: 0 for name in CATEGORY_NAMES}
    started = time.perf_counter()
    with _instrument(elapsed, calls):
        selected = teacher.step_world(world, instance)
    total = time.perf_counter() - started
    return {
        "round": round_index,
        "mix": row["mix"],
        "length": length,
        "implementation": implementation,
        "device": device,
        "total_seconds": total,
        "categories_seconds": elapsed,
        "calls": calls,
        "python_residual_seconds": total - sum(elapsed.values()),
        "action": int(selected["action"]),
        "b8_action": int(selected["b8_action"]),
        "searched": bool(selected["searched"]),
        "scores": {str(key): float(value) for key, value in selected["scores"].items()},
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    configure_cpu()
    from research.phase0_search_20261019.runner import operational_guard

    operational_guard()
    config = _config(args.config)
    bank_path = args.bank.resolve()
    checkpoint = args.checkpoint.resolve()
    mixes = tuple(args.mixes)
    rows = _rows(bank_path, mixes)
    implementations = (
        ("batched", "reference") if args.implementation == "both" else (args.implementation,)
    )
    devices = [
        args.device if implementation == "batched" else "cpu" for implementation in implementations
    ]

    # Warm each requested device once.  Reference is CPU-only by contract.
    for device in dict.fromkeys(devices):
        _warmup(rows[mixes[0]], checkpoint, config, device)

    measurements = []
    for implementation, device in zip(implementations, devices):
        for mix in mixes:
            for round_index in range(1, args.rounds + 1):
                measurements.append(
                    _measure_one(
                        rows[mix],
                        checkpoint,
                        config,
                        args.length,
                        implementation,
                        device,
                        round_index,
                    )
                )
    return {
        "schema": "phase0-search-teacher-profile/v1",
        "git_head": _git_head(),
        "command": [sys.executable, *sys.argv],
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted(
                [
                    Path(__file__),
                    Path(teacher.__file__),
                    Path(preflight.__file__),
                    *(ROOT / "src/simd_env").glob("*.py"),
                    ROOT / "src/model/ego2s_network.py",
                ]
            )
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_threads": 1,
            "warmup_excluded": True,
        },
        "inputs": {
            "bank": str(bank_path),
            "bank_sha256": _sha256(bank_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "teacher_sha256": _sha256(ROOT / "research/phase0_search_20261019/teacher.py"),
            "config": asdict(config),
            "length": args.length,
            "mixes": list(mixes),
            "rounds": args.rounds,
        },
        "measurements": measurements,
        "category_definition": {
            "sim_stepping": "BatchSim.step elapsed time",
            "observation_building": "ego_raster.build_ego_raster elapsed time",
            "b8_forward": "ego2s_policy.forward_q elapsed time, including device synchronization",
            "frozen_opponent_forward": "Vector61SimdPolicy.q_values elapsed time",
            "scripted_opponent_logic": "FastProfileAnchorSimdPolicy.actions elapsed time",
            "clone_snapshot": "teacher._replicate elapsed time",
            "python_residual": "total wall time minus wrapped category intervals",
        },
        "limitations": [
            "The L=800 serpentine is a timing fixture, not a gameplay outcome.",
            "Frozen-opponent runtime preparation and vector61 featurization remain in "
            "Python residual.",
            "Reference implementation is CPU-only; requested non-CPU device is used "
            "only for batched implementation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank", type=Path, default=Path("research/phase0_search_20261019/WORLD_BANK.json")
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--implementation", choices=("batched", "reference", "both"), default="batched"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--mixes", nargs="+", choices=DEFAULT_MIXES, default=list(DEFAULT_MIXES))
    parser.add_argument("--length", type=int, default=800)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.length < 1 or args.rounds < 1:
        parser.error("--length and --rounds must be positive")
    if args.out is not None and args.out.exists():
        parser.error("refusing to overwrite an existing profiling receipt")
    result = _run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(encoded, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(encoded)
        print(
            json.dumps(
                {"out": str(args.out), "measurements": len(result["measurements"])}, sort_keys=True
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
