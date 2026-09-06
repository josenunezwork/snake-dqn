#!/usr/bin/env python3
"""Run one reproducible CPU or MPS PQN sampler diagnostic experiment.

The resulting checkpoints are explicitly experimental and are not promotion
candidates.  This runner exists to compare the legacy fixed-minibatch sampler
with full shuffled rollout reuse while holding the rest of the recipe fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.checkpoint_io import atomic_torch_save  # noqa: E402  # isort: skip
from src.training.pqn_trainer import (  # noqa: E402  # isort: skip
    PQNConfig,
    PQNTrainer,
    TripwireError,
)

EXPERIMENTAL_STATUS = "EXPERIMENTAL_NOT_PROMOTED"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one sampler arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=("legacy", "epoch"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=125_000)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--rollout-len", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sgd-epochs",
        type=int,
        default=None,
        help="Epoch count for --sampler epoch (defaults to 1); invalid for legacy.",
    )
    parser.add_argument(
        "--eps-decay-steps",
        type=int,
        default=None,
        help="Linear epsilon-decay horizon (default: 60%% of --total-steps).",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--pad-sgd-batches",
        action="store_true",
        help="Pad network forwards to the batch cap; loss and telemetry use only real rows.",
    )
    parser.add_argument("--action-collapse-patience", type=int, default=1)
    parser.add_argument("--action-collapse-min-samples", type=int, default=0)
    parser.add_argument(
        "--action-collapse-raw-actions",
        action="store_true",
        help="Guard on unique rollout actions before sampling and augmentation.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid resource budgets and ambiguous sampler configurations."""
    for name in ("total_steps", "threads", "envs", "rollout_len"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("--max-seconds must be finite and positive")
    if args.eps_decay_steps is not None and args.eps_decay_steps <= 0:
        raise ValueError("--eps-decay-steps must be positive")
    if args.sgd_epochs is not None and args.sgd_epochs <= 0:
        raise ValueError("--sgd-epochs must be positive")
    if args.sampler == "legacy" and args.sgd_epochs is not None:
        raise ValueError("--sgd-epochs applies only to --sampler epoch")
    if args.action_collapse_patience <= 0 or args.action_collapse_min_samples < 0:
        raise ValueError("collapse patience must be positive and minimum samples nonnegative")


def build_config(args: argparse.Namespace) -> PQNConfig:
    """Build the fixed experimental recipe, varying only sampler controls."""
    validate_args(args)
    sgd_epochs = 1 if args.sampler == "epoch" and args.sgd_epochs is None else args.sgd_epochs
    return PQNConfig(
        num_envs=args.envs,
        num_snakes=6,
        rollout_len=args.rollout_len,
        gamma=0.997,
        minibatches=4,
        minibatch_size=256,
        sgd_epochs=sgd_epochs if args.sampler == "epoch" else None,
        pool_capacity=0,
        hero_frac=1.0,
        eps_decay_steps=args.eps_decay_steps or max(1, int(args.total_steps * 0.6)),
        flip_augment=True,
        max_frames=5000,
        seed=args.seed,
        mechanics_version=2,
        reward_version=2,
        profile=args.profile,
        pad_sgd_batches=args.pad_sgd_batches,
        sgd_seed=args.seed + 10_000_000,
        action_collapse_patience=args.action_collapse_patience,
        action_collapse_min_samples=args.action_collapse_min_samples,
        action_collapse_raw_actions=args.action_collapse_raw_actions,
    )


def resolve_device(name: str) -> torch.device:
    """Return a supported diagnostic device without silently falling back."""
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but MPS is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def prepare_output(path: Path) -> None:
    """Create an empty output directory without overwriting an old experiment."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty diagnostic output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _tensor_hash(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_hashes() -> Dict[str, str]:
    """Hash every source input to the diagnostic, including this runner."""
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve()]
    for package in ("core", "model", "simd_env", "training"):
        paths.extend(sorted((root / "src" / package).rglob("*.py")))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _native_thread_environment() -> Dict[str, str | None]:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def _synchronize(device: torch.device) -> None:
    """Flush asynchronous MPS work before timing an update."""
    if device.type == "mps":
        torch.mps.synchronize()


def _device_memory_bytes(device: torch.device) -> Dict[str, int | None]:
    """Return available device memory counters for the selected backend."""
    if device.type != "mps":
        return {"current_allocated": None, "driver_allocated": None}
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def _checkpoint(trainer: PQNTrainer, args: argparse.Namespace) -> Dict[str, object]:
    state = trainer.checkpoint_state()
    state.update(
        {
            "evaluation_status": EXPERIMENTAL_STATUS,
            "experiment": "pqn_sampler_diagnostic",
            "sampler": args.sampler,
            "seed": args.seed,
        }
    )
    return state


def _provenance(
    args: argparse.Namespace, config: PQNConfig, initial_hash: str, device: torch.device
) -> Dict[str, Any]:
    return {
        "experiment": "pqn_sampler_diagnostic",
        "evaluation_status": EXPERIMENTAL_STATUS,
        "command": sys.argv,
        "git_sha": _git(["git", "rev-parse", "HEAD"]),
        "git_diff_sha256": hashlib.sha256(_git(["git", "diff", "--binary"]).encode()).hexdigest(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "threads": args.threads,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "native_thread_environment": _native_thread_environment(),
        "config": asdict(config),
        "initial_model_tensor_sha256": initial_hash,
        "source_sha256": _source_hashes(),
    }


def _telemetry_record(
    tel: Any,
    wall_seconds: float,
    update_wall_seconds: float,
    step_delta: int,
    device_memory: Dict[str, int | None],
) -> Dict[str, Any]:
    """Serialize full trainer telemetry plus experiment accounting fields."""
    record = asdict(tel)
    record.update(
        {
            "wall_seconds": wall_seconds,
            "update_wall_seconds": update_wall_seconds,
            "agent_step_delta": step_delta,
            "device_memory_bytes": device_memory,
            "optimizer_metrics": {
                "loss": tel.loss,
                "grad_norm": tel.grad_norm,
                "optimizer_steps": tel.optimizer_steps,
            },
            "sampler_metrics": {
                "eligible_hero_transitions": tel.eligible_hero_transitions,
                "sampled_transition_draws": tel.sampled_transition_draws,
                "unique_sampled_transitions": tel.unique_sampled_transitions,
                "valid_slot_fraction": tel.valid_slot_fraction,
            },
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    """Run an isolated diagnostic arm and write reproducible artifacts."""
    args = parse_args(argv)
    try:
        validate_args(args)
        device = resolve_device(args.device)
        prepare_output(args.out_dir)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc

    # Seed before constructing the network so paired arms start from identical weights.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    config = build_config(args)
    trainer = PQNTrainer(config, device=device)
    initial_hash = _tensor_hash(trainer.network.state_dict())
    atomic_torch_save(_checkpoint(trainer, args), args.out_dir / "initial.pth")
    (args.out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (args.out_dir / "provenance.json").write_text(
        json.dumps(_provenance(args, config, initial_hash, device), indent=2) + "\n"
    )

    started = time.monotonic()
    status, detail = "finished", ""
    previous_steps = trainer.agent_steps
    with (args.out_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        while trainer.agent_steps < args.total_steps:
            if time.monotonic() - started >= args.max_seconds:
                status, detail = "partial_wall_budget", "wall budget reached between updates"
                break
            try:
                _synchronize(device)
                update_started = time.monotonic()
                telemetry = trainer.update()
                _synchronize(device)
            except TripwireError as exc:
                status, detail = "tripwire", str(exc)
                telemetry = exc.telemetry
                _synchronize(device)
                if telemetry is None:
                    break
            step_delta = trainer.agent_steps - previous_steps
            previous_steps = trainer.agent_steps
            history.write(
                json.dumps(
                    _telemetry_record(
                        telemetry,
                        time.monotonic() - started,
                        time.monotonic() - update_started,
                        step_delta,
                        _device_memory_bytes(device),
                    )
                )
                + "\n"
            )
            history.flush()
            if status == "tripwire":
                break

    elapsed = time.monotonic() - started
    final_hash = _tensor_hash(trainer.network.state_dict())
    atomic_torch_save(_checkpoint(trainer, args), args.out_dir / "latest_pqn.pth")
    summary = {
        "experiment": "pqn_sampler_diagnostic",
        "evaluation_status": EXPERIMENTAL_STATUS,
        "status": status,
        "detail": detail,
        "sampler": args.sampler,
        "device": str(device),
        "device_memory_bytes": _device_memory_bytes(device),
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "agent_steps": trainer.agent_steps,
        "target_agent_steps": args.total_steps,
        "ru_maxrss_bytes": _rss_bytes(),
        "initial_model_tensor_sha256": initial_hash,
        "final_model_tensor_sha256": final_hash,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if status == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
