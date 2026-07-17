#!/usr/bin/env python
"""Config-sweep orchestrator for the PQN raster trainer.

Launches a grid of ``train_pqn.py`` runs (bounded concurrency, thread-capped so
they don't fight for CPU cores), then gates each finished run through the
repaired ``tournament_eval --engine simd`` vs the scripted anchors and prints a
leaderboard ranked by mass-integral. This is how you turn "N pods / N GPU slots"
into one experiment with one answer — e.g. sweep the reward knobs to probe the
kills-0 / boost-drift pathology.

Grid axes map to ``train_pqn`` flags: ``kill_scale`` -> --kill-scale,
``death_value`` -> --death-value, ``lr`` -> --lr, ``lambda`` -> --lambda,
``eps_decay_steps`` -> --eps-decay-steps, ``seed`` -> --seed.

Usage:
  # 3-point kill-reward sweep, 3 concurrent GPU slots, 30M steps each
  ./venv/bin/python src/scripts/sweep.py \
      --grid kill_scale=0.3,0.6,1.2 \
      --total-steps 30000000 --envs 256 --snakes 6 --parallel 3 \
      --out-dir runs/sweep_kill

  # 2-axis grid (kill_scale x lr), pick the best by the simd gate
  ./venv/bin/python src/scripts/sweep.py \
      --grid kill_scale=0.3,1.0 lr=2.5e-4,5e-4 \
      --total-steps 30000000 --parallel 4 --out-dir runs/sweep_grid

Each config's checkpoint lands at ``<out-dir>/<name>/latest_pqn.pth`` and its
gate JSON at ``<out-dir>/<name>/gate.json``; the ranked ``leaderboard.json`` +
a printed table land in ``<out-dir>/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# train_pqn's halt-and-flag contract: a tripwire (NaN/inf, max|Q| blow-up, action
# collapse) halts the run, still saves latest_pqn.pth, and exits 2. The checkpoint
# therefore exists and gates fine — it just must not be trusted or promoted.
TRIPWIRE_RC = 2

# Seconds between subprocess polls (tests drive this to 0).
POLL_SECONDS = 5

# Grid key -> train_pqn CLI flag.
_FLAG = {
    "kill_scale": "--kill-scale",
    "death_value": "--death-value",
    "lr": "--lr",
    "lambda": "--lambda",
    "eps_decay_steps": "--eps-decay-steps",
    "seed": "--seed",
    "rollout_len": "--rollout-len",
}


def parse_grid(pairs: List[str]) -> Dict[str, List[str]]:
    """Parse ``key=v1,v2`` grid pairs into an ordered dict of value lists."""
    grid: Dict[str, List[str]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"grid entry {pair!r} must be key=v1,v2,...")
        key, vals = pair.split("=", 1)
        key = key.strip()
        if key not in _FLAG:
            raise ValueError(f"unknown grid key {key!r}; known: {sorted(_FLAG)}")
        grid[key] = [v.strip() for v in vals.split(",") if v.strip()]
    return grid


def config_name(combo: Dict[str, str]) -> str:
    """Short filesystem-safe name for a config combo."""
    return "_".join(f"{k}{v}" for k, v in combo.items()) or "base"


def build_train_cmd(combo: Dict[str, str], args: argparse.Namespace, out_dir: Path) -> List[str]:
    """Assemble the train_pqn command for one config."""
    cmd = [
        sys.executable,
        "src/scripts/train_pqn.py",
        "--device",
        args.device,
        "--total-steps",
        str(args.total_steps),
        "--envs",
        str(args.envs),
        "--snakes",
        str(args.snakes),
        "--rollout-len",
        str(args.rollout_len),
        "--ckpt-every",
        str(args.ckpt_every),
        "--out-dir",
        str(out_dir),
    ]
    for key, val in combo.items():
        cmd += [_FLAG[key], val]
    return cmd


def build_gate_cmd(ckpt: Path, args: argparse.Namespace, out_json: Path) -> List[str]:
    """Assemble the tournament_eval --engine simd gate command (raster ckpt)."""
    seeds = ",".join(str(s) for s in range(args.gate_seeds))
    return [
        sys.executable,
        "src/scripts/tournament_eval.py",
        str(ckpt),
        "--engine",
        "simd",
        "--baseline",
        "scripted:random_safe",
        "--opponents",
        "scripted:greedy_food",
        "--mixes",
        "scripted,mixed",
        "--config",
        "configs/mechanics_v2.yaml",
        "--frames",
        str(args.gate_frames),
        "--seeds",
        seeds,
        "--json-output",
        str(out_json),
    ]


def summarize_gate(gate_json: Path) -> Dict[str, object]:
    """Mean mass-integral / kills / boost / survival across the gate's mixes.

    tournament_eval catches a per-candidate failure, emits ``{"candidate", "error",
    "decision"}`` with no ``per_mix``, still writes the JSON and still exits 0 — so a
    single bad checkpoint must not be able to raise out of the sweep loop.

    Args:
        gate_json: Path to the gate JSON tournament_eval wrote.

    Returns:
        The mean metrics, or ``{"error": <reason>}`` if the gate recorded a failure
        for this candidate or the JSON is unusable. Never raises.
    """
    try:
        data = json.loads(gate_json.read_text())
        candidates = data.get("candidates") or []
        if not candidates:
            return {"error": "gate json has no candidates"}
        cand = candidates[0]
        per_mix = cand.get("per_mix")
        if not per_mix:
            return {"error": f"eval failed: {cand.get('error', 'gate json has no per_mix')}"}
        mi, kills, boost, surv = [], [], [], []
        for mix in per_mix.values():
            s = mix["summary"]
            p = s.get("probes", {})
            mi.append(float(s.get("mass_integral", 0.0)))
            kills.append(float(s.get("kills", 0.0)))
            surv.append(float(s.get("survival_fraction", 0.0)))
            boost.append(float(p.get("boost_frame_fraction", 0.0)))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"error": f"unreadable gate json: {exc}"}
    n = max(len(mi), 1)
    return {
        "mass_integral": sum(mi) / n,
        "kills": sum(kills) / n,
        "boost": sum(boost) / n,
        "survival": sum(surv) / n,
    }


def _status_marker(result: Dict[str, object]) -> str:
    """TRIPWIRE / crash marker for a result row, or "" if the run exited cleanly."""
    rc = int(result.get("train_rc", 0) or 0)
    if rc == TRIPWIRE_RC:
        return "TRIPWIRE(rc=2)"
    if rc != 0:
        return f"TRAIN_RC={rc}"
    return ""


def is_clean(result: Dict[str, object]) -> bool:
    """True when the run gated successfully AND its training exited 0.

    A tripwired (rc=2) run still writes a checkpoint that gates fine, so exit code
    is the only thing separating a trustworthy row from a corrupted one.
    """
    return "error" not in result and int(result.get("train_rc", 0) or 0) == 0


def rank_key(result: Dict[str, object]) -> Tuple[int, float]:
    """Sort key (descending): clean runs first, then tripped/crashed, errors last."""
    if "error" in result:
        status = 0
    elif not is_clean(result):
        status = 1
    else:
        status = 2
    try:
        mass = float(result.get("mass_integral", float("-inf")))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        mass = float("-inf")
    return (status, mass)


def _thread_env(cap: int) -> Dict[str, str]:
    """Env with BLAS/OMP thread caps so parallel runs don't oversubscribe cores."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(cap)
    return env


def run_sweep(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Launch the grid with bounded concurrency, gate each, return results."""
    grid = parse_grid(args.grid)
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    env = _thread_env(args.threads)

    print(
        f"[sweep] {len(combos)} configs, {args.parallel} concurrent, "
        f"{args.total_steps:,} steps each -> {out_root}"
    )
    for c in combos:
        print(f"[sweep]   {config_name(c)}: {c}")

    # (combo, out_dir, Popen or None, gated result or None)
    pending = list(combos)
    running: List[Tuple[Dict[str, str], Path, subprocess.Popen]] = []
    results: List[Dict[str, object]] = []
    log = open(out_root / "sweep.log", "a", encoding="utf-8")

    def launch(combo: Dict[str, str]) -> None:
        name = config_name(combo)
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        cmd = build_train_cmd(combo, args, d)
        fh = open(d / "train.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        running.append((combo, d, p))
        print(f"[sweep] launched {name} (pid {p.pid})", flush=True)

    def gate(combo: Dict[str, str], d: Path, train_rc: int) -> None:
        name = config_name(combo)
        ckpt = d / "latest_pqn.pth"
        entry: Dict[str, object] = {
            "name": name,
            "config": combo,
            "out_dir": str(d),
            "train_rc": train_rc,
            "tripped": train_rc == TRIPWIRE_RC,
        }
        marker = _status_marker(entry)
        suffix = f"  [{marker}]" if marker else ""
        if not ckpt.exists():
            entry["error"] = "no checkpoint produced"
            results.append(entry)
            print(f"[sweep] {name}: NO CHECKPOINT{suffix}", flush=True)
            return
        gate_json = d / "gate.json"
        gc = build_gate_cmd(ckpt, args, gate_json)
        rc = subprocess.run(gc, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc.returncode != 0 or not gate_json.exists():
            entry["error"] = f"gate failed (rc={rc.returncode})"
            results.append(entry)
            print(f"[sweep] {name}: GATE FAILED{suffix}", flush=True)
            return
        entry.update(summarize_gate(gate_json))
        results.append(entry)
        if "error" in entry:
            print(f"[sweep] {name}: GATE ERROR: {entry['error']}{suffix}", flush=True)
            return
        print(
            f"[sweep] {name}: mass_int={entry['mass_integral']:.1f} "
            f"kills={entry['kills']:.1f} boost={entry['boost']*100:.0f}% "
            f"surv={entry['survival']:.2f}{suffix}",
            flush=True,
        )

    while pending or running:
        while pending and len(running) < args.parallel:
            launch(pending.pop(0))
        time.sleep(POLL_SECONDS)
        for combo, d, p in list(running):
            train_rc = p.poll()
            if train_rc is None:
                continue
            running.remove((combo, d, p))
            try:
                gate(combo, d, train_rc)
            except Exception as exc:  # noqa: BLE001 - one bad config must not orphan the rest
                # Every other config in this sweep is a multi-hour job; an exception
                # escaping here kills the orchestrator and orphans them.
                results.append(
                    {
                        "name": config_name(combo),
                        "config": combo,
                        "out_dir": str(d),
                        "train_rc": train_rc,
                        "tripped": train_rc == TRIPWIRE_RC,
                        "error": f"gate crashed: {exc}",
                    }
                )
                print(f"[sweep] {config_name(combo)}: GATE CRASHED: {exc}", flush=True)

    log.close()
    return results


def write_leaderboard(results: List[Dict[str, object]], out_dir: Path) -> None:
    """Rank by mass-integral (tripped runs below clean ones, errors last).

    Writes ``leaderboard.json`` and prints the table. Only a clean run — gated AND
    exited 0 — is ever announced as the Winner.
    """
    ranked = sorted(results, key=rank_key, reverse=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(ranked, indent=2))
    print("\n=== SWEEP LEADERBOARD (by mass-integral vs anchors) ===")
    print(
        f"{'rank':>4}  {'config':32s} {'mass_int':>9} {'kills':>6} "
        f"{'boost':>6} {'surv':>6}  {'flags'}"
    )
    for i, r in enumerate(ranked, 1):
        marker = _status_marker(r)
        if "error" in r:
            flags = f"ERR: {r['error']}" + (f"  {marker}" if marker else "")
            print(f"{i:>4}  {r['name']:32s} {flags}")
            continue
        print(
            f"{i:>4}  {r['name']:32s} {r['mass_integral']:9.1f} {r['kills']:6.1f} "
            f"{r['boost']*100:5.0f}% {r['survival']:6.2f}  {marker}"
        )
    if not ranked:
        return
    top = ranked[0]
    if is_clean(top):
        print(f"\nWinner: {top['name']}  ->  {top['out_dir']}/latest_pqn.pth")
        return
    reason = str(top.get("error")) if "error" in top else _status_marker(top)
    print("\nNo clean winner: every config errored or exited non-zero.")
    print(f"Best-ranked was {top['name']} ({reason}); inspect {top['out_dir']}/train.log.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--grid", nargs="+", required=True, help="key=v1,v2 pairs (see --help).")
    p.add_argument("--out-dir", default="runs/sweep", help="Sweep output root.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--snakes", type=int, default=6)
    p.add_argument("--rollout-len", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--parallel", type=int, default=3, help="Concurrent runs (VRAM-bounded).")
    p.add_argument("--threads", type=int, default=6, help="Per-run OMP/BLAS thread cap.")
    p.add_argument("--gate-frames", type=int, default=1500)
    p.add_argument("--gate-seeds", type=int, default=12)
    args = p.parse_args()

    results = run_sweep(args)
    write_leaderboard(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
