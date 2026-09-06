#!/usr/bin/env python3
"""Run one small paired PQN reward diagnostic; checkpoints are experimental."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.simd_env.batch_sim import BatchSim  # noqa: E402  # isort: skip
from src.training.pqn_trainer import (  # noqa: E402  # isort: skip
    PQNConfig,
    PQNTrainer,
    TripwireError,
)


class DiagnosticBatchSim(BatchSim):
    """BatchSim with an opt-in post-step alive logical-length reward term."""

    def __init__(self, *args: Any, mass_reward_coefficient: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mass_reward_coefficient = mass_reward_coefficient

    def get_reward(self) -> np.ndarray:
        base = super().get_reward()
        return base + self.mass_reward_coefficient * self.get_lengths() * self.get_alive()


def _git(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _tensor_hash(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_hashes() -> Dict[str, str]:
    """Hash this runner and the source packages that determine its recipe."""
    root = Path(__file__).resolve().parents[2]
    source_root = root / "src"
    paths = [Path(__file__).resolve()]
    for package in ("core", "model", "simd_env", "training"):
        paths.extend(sorted((source_root / package).rglob("*.py")))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _thread_environment() -> Dict[str, str | None]:
    """Record native thread-pool caps supplied by the diagnostic launcher."""
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def _checkpoint(trainer: PQNTrainer, arm: str, coefficient: float) -> Dict[str, object]:
    state = trainer.checkpoint_state()
    state.update(
        {
            "diagnostic_arm": arm,
            "mass_reward_coefficient": coefficient,
            "evaluation_status": "EXPERIMENTAL_NOT_PROMOTED",
        }
    )
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "mass"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=125_000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--no-flip-augment", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total_steps <= 0 or args.threads <= 0 or args.max_seconds <= 0:
        raise SystemExit("--total-steps, --threads, and --max-seconds must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite existing diagnostic output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    coefficient = 0.003 if args.arm == "mass" else 0.0
    cfg = PQNConfig(
        num_envs=16,
        num_snakes=6,
        rollout_len=16,
        gamma=0.997,
        minibatches=4,
        minibatch_size=256,
        pool_capacity=0,
        hero_frac=1.0,
        eps_decay_steps=int(args.total_steps * 0.6),
        flip_augment=not args.no_flip_augment,
        max_frames=5000,
        seed=args.seed,
        mechanics_version=2,
        reward_version=2,
    )
    trainer = PQNTrainer(cfg, device=torch.device("cpu"))
    sim_type = DiagnosticBatchSim
    trainer.sim = sim_type(
        trainer.sim.cfg,
        seeds=[args.seed + env for env in range(cfg.num_envs)],
        train_mode=True,
        mass_reward_coefficient=coefficient,
    )
    initial_hash = _tensor_hash(trainer.network.state_dict())
    atomic_torch_save(_checkpoint(trainer, args.arm, coefficient), args.out_dir / "initial.pth")
    provenance = {
        "command": sys.argv,
        "git_sha": _git(["git", "rev-parse", "HEAD"]),
        "git_diff_sha256": hashlib.sha256(_git(["git", "diff", "--binary"]).encode()).hexdigest(),
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "source_sha256": _source_hashes(),
        "threads": args.threads,
        "native_thread_environment": _thread_environment(),
        "config": asdict(cfg),
        "initial_model_tensor_sha256": initial_hash,
        "experimental_reward": {
            "formula": "base + coefficient * poststep_alive_logical_length",
            "mass_reward_coefficient": coefficient,
            "evaluation_status": "EXPERIMENTAL_NOT_PROMOTED",
        },
    }
    (args.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    status, detail = "completed", ""
    previous_steps = trainer.agent_steps
    with (args.out_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        while trainer.agent_steps < args.total_steps:
            if time.monotonic() - started >= args.max_seconds:
                status, detail = "partial_wall_budget", "wall budget reached between updates"
                break
            try:
                tel = trainer.update()
            except TripwireError as exc:
                status, detail = "tripwire", str(exc)
                break
            step_delta = trainer.agent_steps - previous_steps
            previous_steps = trainer.agent_steps
            record = asdict(tel)
            record.update(
                {
                    "wall_seconds": time.monotonic() - started,
                    "agent_step_delta": step_delta,
                    "alive": int(trainer.sim.get_alive().sum()),
                    "train_sample_draws": cfg.minibatches * min(step_delta, cfg.minibatch_size),
                }
            )
            history.write(json.dumps(record) + "\n")
            history.flush()
    elapsed = time.monotonic() - started
    atomic_torch_save(_checkpoint(trainer, args.arm, coefficient), args.out_dir / "latest_pqn.pth")
    summary = {
        "status": status,
        "detail": detail,
        "arm": args.arm,
        "seed": args.seed,
        "mass_reward_coefficient": coefficient,
        "evaluation_status": "EXPERIMENTAL_NOT_PROMOTED",
        "elapsed_seconds": elapsed,
        "agent_steps": trainer.agent_steps,
        "target_agent_steps": args.total_steps,
        "ru_maxrss_bytes": _rss_bytes(),
        "initial_model_tensor_sha256": initial_hash,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
