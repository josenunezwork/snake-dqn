#!/usr/bin/env python
"""3-arm kill-diagnosis orchestrator (blueprint §0.9 / P1).

Runs one arm of the discriminating experiment that decides WHY the incumbent
learns ~0 kills, so the P4 campaign budget can be aimed at the binding cause.
All three remedies (economics, opponent pool, more data) ship regardless; this
experiment only sets the emphasis.

Arms (all warm-start ``saved_snakes/champion_a5_freespace_20260621.pth``, local
CPU Ape-X):

  A  economics only      mechanics v2 + reward v2, no opponent pool
  B  economics + pool    arm A + 80/20 frozen-champion opponent pool
  C  baseline (control)  mechanics v1 + reward v1 (current economics)

Each arm: (1) fine-tunes the champion for ``--total-steps`` learner steps under
the arm's config, then (2) evaluates the resulting checkpoint under a common
mechanics-v2 gate (so kills are physically possible for every arm) with the
repaired tournament_eval + behavioral probes, and writes a JSON verdict
``{arm, kills_per_episode, kill_opportunities, mass_integral_delta}``.

Decision rule (compare the three verdicts, per blueprint §0.9):

  * A >> C on kills/ep       -> economics is the primary cause (confirmed).
  * B >> A on kills/ep       -> opponent quality adds materially on top.
  * A ~= C ~= 0 kills but
    kill_opportunities rising -> data VOLUME binds; shift P4 budget from reward
                                 sweeps to longer self-play generations.

Usage:
  # one arm, ~overnight CPU run (tune --total-steps to your machine)
  ./venv/bin/python src/scripts/run_kill_diagnosis.py --arm A \
      --total-steps 300000 --num-actors 6 --out-dir logs/diag/A

  # fast wiring smoke (tiny, proves the pipeline end to end)
  ./venv/bin/python src/scripts/run_kill_diagnosis.py --arm C --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAMPION = "saved_snakes/champion_a5_freespace_20260621.pth"
# Frozen 61-D free-space champions used as arm-B opponent pool. (All are the
# free-space contract; the 58-D baselines are excluded — they are not
# obs-compatible opponents for a 61-D hero.)
POOL_CHECKPOINTS = [
    "saved_snakes/best_apex_fs.pth",
    "saved_snakes/best_apex_stage1_fs.pth",
    "saved_snakes/best_apex_pre_fs.pth",
]
# Common gate mechanics: kills must be physically possible for EVERY arm's
# checkpoint, so all arms are evaluated under mechanics v2.
EVAL_CONFIG = "configs/mechanics_v2.yaml"

ARM_CONFIG = {
    "A": "configs/diag_arm_a.yaml",
    "B": "configs/diag_arm_b.yaml",
    "C": "configs/diag_arm_c.yaml",
}
# Arms A/B change reward economics vs the champion contract, so their resume
# needs the reward-contract override; arm C matches the champion contract.
ARM_NEEDS_OVERRIDE = {"A": True, "B": True, "C": False}


def _build_pool_dir(out_dir: Path) -> Optional[str]:
    """Create a pool directory of symlinks to the frozen champions (arm B)."""
    pool_dir = out_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for ckpt in POOL_CHECKPOINTS:
        src = REPO_ROOT / ckpt
        if not src.exists():
            print(f"  [pool] skip missing {ckpt}")
            continue
        link = pool_dir / src.name
        if not link.exists():
            link.symlink_to(src)
        made += 1
    if made == 0:
        print("  [pool] WARNING: no pool checkpoints found; arm B falls back to mirror")
        return None
    return str(pool_dir)


def _train_command(
    arm: str, config: str, total_steps: int, num_actors: int, out_dir: Path
) -> List[str]:
    """Assemble the apex_train subprocess command for an arm."""
    cmd = [
        sys.executable,
        "src/scripts/apex_train.py",
        "--config",
        config,
        "--resume",
        CHAMPION,
        "--total-steps",
        str(total_steps),
        "--num-actors",
        str(num_actors),
        "--checkpoint-dir",
        str(out_dir),
    ]
    if ARM_NEEDS_OVERRIDE[arm]:
        cmd.append("--override-reward-contract")
    if arm == "B":
        pool_dir = _build_pool_dir(out_dir)
        if pool_dir:
            cmd += ["--opponent-pool-dir", pool_dir, "--pool-latest-fraction", "0.8"]
    return cmd


def _find_trained_checkpoint(out_dir: Path) -> Optional[Path]:
    """Return the newest checkpoint apex_train wrote into out_dir."""
    candidates = sorted(out_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer a rolling 'latest' checkpoint if present (see checkpoint-naming memo).
    for p in candidates:
        if "latest" in p.name:
            return p
    return candidates[0] if candidates else None


def _eval_command(checkpoint: Path, frames: int, seeds: str, out_json: Path) -> List[str]:
    """Assemble the tournament_eval subprocess command (mechanics-v2 gate)."""
    return [
        sys.executable,
        "src/scripts/tournament_eval.py",
        str(checkpoint),
        "--baseline",
        CHAMPION,
        "--config",
        EVAL_CONFIG,
        "--mixes",
        "scripted,mixed",
        "--frames",
        str(frames),
        "--seeds",
        seeds,
        "--json-output",
        str(out_json),
    ]


def _summarize_eval(eval_json: Path) -> Dict[str, float]:
    """Pull kills/ep, kill-opportunities, and mass-integral delta from a gate JSON."""
    data = json.loads(eval_json.read_text())
    cand = data["candidates"][0]
    kills: List[float] = []
    opps: List[float] = []
    deltas: List[float] = []
    for mix in cand["per_mix"].values():
        summary = mix["summary"]
        probes = summary.get("probes", {})
        kills.append(float(summary.get("kills", 0.0)))
        opps.append(float(probes.get("kill_opportunity_count", 0.0)))
        deltas.append(float(mix["paired"]["mean_delta"]))
    n = max(len(kills), 1)
    return {
        "kills_per_episode": sum(kills) / n,
        "kill_opportunities": sum(opps) / n,
        "mass_integral_delta": sum(deltas) / n,
    }


def run_arm(
    arm: str,
    total_steps: int,
    num_actors: int,
    out_dir: Path,
    frames: int,
    seeds: str,
) -> Dict[str, object]:
    """Run one diagnosis arm end to end and return its verdict dict."""
    arm = arm.upper()
    if arm not in ARM_CONFIG:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_CONFIG)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ARM_CONFIG[arm]

    print(f"=== ARM {arm}: training ({config}, {total_steps} steps) ===", flush=True)
    train_cmd = _train_command(arm, config, total_steps, num_actors, out_dir)
    print("  " + " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)

    checkpoint = _find_trained_checkpoint(out_dir)
    if checkpoint is None:
        raise RuntimeError(f"arm {arm}: no checkpoint produced in {out_dir}")

    print(f"=== ARM {arm}: evaluating {checkpoint.name} (mechanics-v2 gate) ===", flush=True)
    eval_json = out_dir / "gate.json"
    eval_cmd = _eval_command(checkpoint, frames, seeds, eval_json)
    print("  " + " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)

    metrics = _summarize_eval(eval_json)
    verdict = {
        "arm": arm,
        "config": config,
        "checkpoint": str(checkpoint),
        "total_steps": total_steps,
        **metrics,
    }
    verdict_path = out_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))
    print(f"=== ARM {arm} verdict -> {verdict_path} ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--num-actors", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--frames", type=int, default=3000, help="Gate rollout length")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in range(20)),
        help="Comma-separated paired gate seeds",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end wiring check (overrides steps/actors/frames/seeds).",
    )
    args = parser.parse_args()

    arm = args.arm.upper()
    if args.smoke:
        total_steps, num_actors, frames, seeds = 200, 2, 200, "0,1"
    else:
        total_steps, num_actors, frames, seeds = (
            args.total_steps,
            args.num_actors,
            args.frames,
            args.seeds,
        )
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "logs" / "diag" / arm

    run_arm(arm, total_steps, num_actors, out_dir, frames, seeds)


if __name__ == "__main__":
    main()
