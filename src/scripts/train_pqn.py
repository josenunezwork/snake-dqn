#!/usr/bin/env python3
"""CLI wrapper around :class:`~src.training.pqn_trainer.PQNTrainer` (blueprint P3).

PQN is synchronous Q(lambda) on vectorized envs — **no replay buffer, no target
network, no PER** (see :mod:`src.training.pqn_trainer`). This script is the
thin operational surface over that trainer: it maps CLI flags (and an optional
YAML ``--config``) onto a :class:`~src.training.pqn_trainer.PQNConfig`, runs the
update loop while printing a periodic learning curve, halts-and-flags on a
tripwire (NaN/inf, max|Q| blow-up, action collapse), and writes a raster
checkpoint (obs_spec ``raster31v2``, algo ``pqn``) that
:meth:`~src.model.inference_agent.InferenceAgent.from_checkpoint` reloads and
``tournament_eval --engine simd`` can gate.

Examples:
    # Local learning smoke (CPU, tiny): 32 envs x 6 snakes, 800 updates
    SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/train_pqn.py \\
        --total-steps 150000 --envs 32 --snakes 6 --rollout-len 24 \\
        --out-dir /tmp/pqn_smoke --eps-decay-steps 120000

    # Serious local run (a few hours on CPU / minutes on CUDA)
    ./venv/bin/python src/scripts/train_pqn.py \\
        --total-steps 5000000 --envs 64 --snakes 6 --out-dir runs/pqn_local

    # No self-play (hero-only rollouts, useful for isolating the learner)
    ./venv/bin/python src/scripts/train_pqn.py --no-self-play --total-steps 100000

``--total-steps`` counts HERO agent-steps (transitions the learner trains on),
matching the trainer's ``agent_steps`` odometer and the ε schedule — the loop
runs updates until that budget is met.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.device_manager import DeviceManager  # noqa: E402
from src.training.pqn_trainer import (  # noqa: E402
    PQNConfig,
    PQNTelemetry,
    PQNTrainer,
    TripwireError,
)

# YAML keys under which PQN knobs may be nested in a --config file. We read a
# flat ``pqn:`` block plus a few shared ``game:``/``rewards:`` switches so the
# same mechanics-v2 config the vector pipeline uses also drives PQN.
_PQN_FIELDS = {f.name for f in fields(PQNConfig)}


def _load_config_overrides(path: str) -> Dict[str, Any]:
    """Read PQN overrides from a YAML config file.

    Recognized sources (later ones win):
      * a top-level ``pqn:`` mapping whose keys match :class:`PQNConfig` fields;
      * ``game.mechanics_version`` / ``game.arena_type`` (shared sim switches);
      * ``rewards.version`` -> ``reward_version``.

    Unknown keys are ignored (with a warning) so a full training config can be
    passed without erroring on vector-only knobs.

    Args:
        path: Path to a YAML config file.

    Returns:
        A dict of ``PQNConfig`` field -> value overrides.
    """
    import yaml  # local import: only needed when --config is used

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    overrides: Dict[str, Any] = {}

    game = raw.get("game", {}) or {}
    if "mechanics_version" in game:
        overrides["mechanics_version"] = int(game["mechanics_version"])
    if "arena_type" in game:
        overrides["arena_type"] = str(game["arena_type"])
    if "num_snakes" in game:
        overrides["num_snakes"] = int(game["num_snakes"])
    if "max_frames" in game:
        overrides["max_frames"] = int(game["max_frames"])

    rewards = raw.get("rewards", {}) or {}
    if "version" in rewards:
        overrides["reward_version"] = int(rewards["version"])

    pqn = raw.get("pqn", {}) or {}
    for key, value in pqn.items():
        if key in _PQN_FIELDS:
            overrides[key] = value
        else:
            print(f"[train_pqn] warning: ignoring unknown pqn config key {key!r}", file=sys.stderr)

    return overrides


def build_config(args: argparse.Namespace) -> PQNConfig:
    """Merge defaults, an optional YAML ``--config``, and CLI flags into a config.

    Precedence (low -> high): :class:`PQNConfig` defaults, then the ``--config``
    file, then any explicitly-passed CLI flag. CLI flags that were left at their
    sentinel ``None`` do not override the config/defaults.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The resolved :class:`PQNConfig`.
    """
    cfg_kwargs: Dict[str, Any] = {}
    if args.config:
        cfg_kwargs.update(_load_config_overrides(args.config))

    # Map CLI flags (only those explicitly provided) onto config fields.
    cli_map = {
        "num_envs": args.envs,
        "num_snakes": args.snakes,
        "rollout_len": args.rollout_len,
        "gamma": args.gamma,
        "lambda_": args.lambda_,
        "lr": args.lr,
        "minibatches": args.minibatches,
        "minibatch_size": args.minibatch_size,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay_steps": args.eps_decay_steps,
        "hero_frac": args.hero_frac,
        "pool_capacity": args.pool_capacity,
        "pool_add_interval": args.pool_add_interval,
        "flip_augment": args.flip_augment,
        "max_abs_q_alarm": args.max_abs_q_alarm,
        "seed": args.seed,
        "arena_type": args.arena_type,
        "mechanics_version": args.mechanics_version,
        "profile": args.profile or None,
        "kill_scale": args.kill_scale,
        "death_value": args.death_value,
    }
    for key, value in cli_map.items():
        if value is not None:
            cfg_kwargs[key] = value

    # --no-self-play collapses the opponent pool and forces every slot to hero.
    if args.no_self_play:
        cfg_kwargs["pool_capacity"] = 0
        cfg_kwargs["hero_frac"] = 1.0

    return PQNConfig(**cfg_kwargs)


def _resolve_device(name: Optional[str]) -> torch.device:
    """Resolve a device string, honoring the :class:`DeviceManager` default.

    Args:
        name: ``"cpu"``/``"cuda"``/``"mps"`` or ``None`` for auto-select.

    Returns:
        The chosen :class:`torch.device`.
    """
    if name:
        device = torch.device(name)
        DeviceManager.override_device(device)
        return device
    return DeviceManager.get_device()


def _format_row(tel: PQNTelemetry) -> str:
    """One-line telemetry summary for the live learning curve."""
    return (
        f"upd {tel.update:>5} | steps {tel.agent_steps:>10,} | "
        f"eps {tel.epsilon:5.3f} | loss {tel.loss:9.4f} | "
        f"|Q|~ {tel.mean_abs_q:7.3f} max {tel.max_abs_q:8.3f} | "
        f"gnorm {tel.grad_norm:7.3f} | R/step {tel.mean_reward:+7.4f} | "
        f"H {tel.action_entropy:5.3f} | kills {tel.kills_per_ep:5.1f} | "
        f"boost {100 * tel.boost_fraction:4.1f}% | pool {tel.pool_size}"
    )


def _telemetry_record(tel: PQNTelemetry) -> Dict[str, Any]:
    """A JSON-serializable dict of a telemetry snapshot (for the history file)."""
    return {
        "update": tel.update,
        "agent_steps": tel.agent_steps,
        "epsilon": tel.epsilon,
        "loss": tel.loss,
        "grad_norm": tel.grad_norm,
        "mean_abs_q": tel.mean_abs_q,
        "max_abs_q": tel.max_abs_q,
        "mean_reward": tel.mean_reward,
        "action_entropy": tel.action_entropy,
        "kills_per_ep": tel.kills_per_ep,
        "boost_fraction": tel.boost_fraction,
        "pool_size": tel.pool_size,
    }


def train_loop(
    trainer: PQNTrainer,
    total_steps: int,
    log_every: int,
    ckpt_every: int,
    out_dir: Path,
) -> Tuple[List[PQNTelemetry], Optional[str]]:
    """Run updates until ``total_steps`` hero agent-steps are reached.

    Prints a telemetry row every ``log_every`` updates, snapshots a checkpoint
    every ``ckpt_every`` updates (and always at the end), and writes the full
    telemetry history to ``out_dir/history.jsonl``. A :class:`TripwireError`
    halts the loop, still saving a final checkpoint and flagging the run.

    Args:
        trainer: The configured :class:`PQNTrainer`.
        total_steps: Target hero agent-step budget.
        log_every: Print cadence (updates).
        ckpt_every: Checkpoint cadence (updates); ``0`` disables periodic saves.
        out_dir: Directory for checkpoints and the history file.

    Returns:
        ``(history, tripped)``: the telemetry list collected before the
        budget/tripwire ended the loop, and the tripwire message (``None`` on a
        clean finish).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    latest_path = out_dir / "latest_pqn.pth"

    history: List[PQNTelemetry] = []
    start = time.time()
    tripped: Optional[str] = None

    with history_path.open("w", encoding="utf-8") as hist_fh:
        while trainer.agent_steps < total_steps:
            try:
                tel = trainer.update()
            except TripwireError as exc:
                tripped = str(exc)
                print(f"\n[TRIPWIRE] halting: {exc}", file=sys.stderr)
                break

            history.append(tel)
            hist_fh.write(json.dumps(_telemetry_record(tel)) + "\n")
            hist_fh.flush()

            if tel.update % log_every == 0:
                elapsed = time.time() - start
                sps = tel.agent_steps / elapsed if elapsed > 0 else 0.0
                print(f"{_format_row(tel)} | {sps:,.0f} steps/s")

            if ckpt_every and tel.update > 0 and tel.update % ckpt_every == 0:
                trainer.save_checkpoint(str(latest_path))

    # Always leave a final checkpoint (even on a tripwire halt).
    trainer.save_checkpoint(str(latest_path))
    elapsed = time.time() - start
    print(
        f"\n[train_pqn] done: {len(history)} updates, "
        f"{trainer.agent_steps:,} agent-steps in {elapsed:.1f}s "
        f"({trainer.agent_steps / elapsed:,.0f} steps/s)"
    )
    print(f"[train_pqn] checkpoint: {latest_path}")
    if tripped:
        print(f"[train_pqn] FLAGGED (tripwire): {tripped}", file=sys.stderr)

    return history, tripped


def _print_curve_summary(history: Sequence[PQNTelemetry]) -> None:
    """Print a compact start-vs-end learning-curve summary."""
    if not history:
        print("[train_pqn] no updates completed.")
        return
    head = history[: max(1, len(history) // 10)]
    tail = history[-max(1, len(history) // 10) :]

    def _avg(rows: Sequence[PQNTelemetry], attr: str) -> float:
        return sum(getattr(r, attr) for r in rows) / len(rows)

    print("\n[train_pqn] learning-curve summary (first 10% -> last 10%):")
    for attr, label in (
        ("loss", "loss"),
        ("mean_reward", "reward/step"),
        ("mean_abs_q", "mean|Q|"),
        ("action_entropy", "entropy"),
        ("kills_per_ep", "kills/rollout"),
        ("boost_fraction", "boost frac"),
    ):
        print(f"  {label:>14}: {_avg(head, attr):+.4f}  ->  {_avg(tail, attr):+.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args, build the trainer, run the loop, and save a checkpoint.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code: ``0`` on a clean run, ``2`` if a tripwire fired.
    """
    p = argparse.ArgumentParser(
        description="Train the dual-scale raster network with PQN Q(lambda) (blueprint P3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
        help="Target HERO agent-steps (transitions trained on).",
    )
    p.add_argument("--out-dir", default="runs/pqn", help="Directory for checkpoints + history.")
    p.add_argument("--config", default=None, help="Optional YAML config with a pqn: block.")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto-select).")

    # Sim shape.
    p.add_argument("--envs", type=int, default=None, help="E parallel envs.")
    p.add_argument("--snakes", type=int, default=None, help="S snakes per env.")
    p.add_argument("--rollout-len", type=int, default=None, help="T steps per update.")

    # Optimization / algorithm.
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lambda", dest="lambda_", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--minibatches", type=int, default=None)
    p.add_argument("--minibatch-size", type=int, default=None)

    # Exploration.
    p.add_argument("--eps-start", type=float, default=None)
    p.add_argument("--eps-end", type=float, default=None)
    p.add_argument("--eps-decay-steps", type=int, default=None, help="Agent-steps to decay eps.")

    # Self-play.
    p.add_argument("--hero-frac", type=float, default=None, help="P(slot is hero); blueprint 0.8.")
    p.add_argument("--pool-capacity", type=int, default=None, help="Max frozen opponents (<=10).")
    p.add_argument("--pool-add-interval", type=int, default=None, help="Add snapshot every N upd.")
    p.add_argument(
        "--no-self-play",
        action="store_true",
        help="Hero-only rollouts (pool_capacity=0, hero_frac=1.0).",
    )

    # Augmentation / tripwire / sim.
    flip = p.add_mutually_exclusive_group()
    flip.add_argument(
        "--flip-augment", dest="flip_augment", action="store_const", const=True, default=None
    )
    flip.add_argument("--no-flip-augment", dest="flip_augment", action="store_const", const=False)
    p.add_argument("--max-abs-q-alarm", type=float, default=None, help="Tripwire max|Q| threshold.")
    p.add_argument(
        "--kill-scale",
        type=float,
        default=None,
        help="Reward per victim-length for a kill (default 0.3). Sweep to probe kills-0.",
    )
    p.add_argument(
        "--death-value",
        type=float,
        default=None,
        help="Death reward AND trapped-state bootstrap value (default -3.0). Sweepable.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print a CUDA-synced per-phase time breakdown each update (featurize/forward/sim/targets/sgd).",  # noqa: E501
    )
    p.add_argument("--arena-type", default=None, help="rectangular / circular.")
    p.add_argument(
        "--mechanics-version", type=int, default=None, help="Sim mechanics (blueprint 2)."
    )
    p.add_argument("--seed", type=int, default=None)

    # Logging cadence.
    p.add_argument("--log-every", type=int, default=10, help="Print a row every N updates.")
    p.add_argument(
        "--ckpt-every", type=int, default=200, help="Checkpoint every N updates (0=off)."
    )

    args = p.parse_args(argv)

    device = _resolve_device(args.device)
    config = build_config(args)
    out_dir = Path(args.out_dir)

    print("[train_pqn] device:", device)
    print("[train_pqn] config:", json.dumps(asdict(config), indent=2, sort_keys=True))

    trainer = PQNTrainer(config, device=device)
    print("[train_pqn] network:", repr(trainer.network))
    print("[train_pqn] params:", f"{trainer.network.get_num_parameters()['total']:,}")

    history, tripped = train_loop(
        trainer,
        total_steps=args.total_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        out_dir=out_dir,
    )
    _print_curve_summary(history)

    # Exit 2 if a tripwire halted the run (halt-and-flag contract).
    return 2 if tripped else 0


if __name__ == "__main__":
    raise SystemExit(main())
