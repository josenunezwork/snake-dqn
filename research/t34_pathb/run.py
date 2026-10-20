"""Mac-only, bounded T34 Path B development harness.

The command never starts a service or tmux session.  Each invocation performs
one bounded operation and records enough identity to reject partial results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

HERE, REPO = Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.t34_pathb import bank


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    checkpoint = REPO / "saved_snakes" / "b8_s3_ego2s_20261019.pth"
    if not checkpoint.exists():
        # Registry remains source of truth across filename changes.
        from web.backend.served_checkpoint import resolve_served_checkpoint
        checkpoint = Path(str(resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"}).path))
    expected = sum(len(b["terminal"]) * 2 + len(b["product"]) * 2 for b in plan["blocks"])
    return {"schema": "t34-pathb-manifest/v1", "created_utc": _utc(), "protocol_sha256": sha(HERE / "PROTOCOL.md"),
            "source_commit": _git("rev-parse", "HEAD"), "namespace": plan["namespace"],
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "bank": plan,
            "settings": {"horizon": args.horizon, "fps": [12, 24], "workers": args.workers,
                         "max_hours": args.max_hours, "smoke": args.smoke}, "expected_records": expected,
            "no_tmux": True, "mac_only": True, "development_only": True}


def create_bank(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if not out.is_absolute() or out.exists():
        raise RuntimeError("--out must be a fresh absolute directory")
    if args.workers != 1:
        raise RuntimeError("T34 permits exactly one assistance worker (parent + worker <=2 CPU threads)")
    plan = bank.allocation(blocks=args.blocks, scripted=args.scripted_worlds, frozen=args.frozen_worlds,
                           mixed=args.mixed_worlds, product=args.product_worlds, smoke=args.smoke, repo=REPO)
    _write_new(out / "manifest.json", _manifest(args, plan))
    return out


def _arm_session(seed: int, fps: int, mode: str, smoke: bool):
    from research.t15_mixed_roster_20261017.episodes import new_session
    from web.backend import pathb
    session = new_session(seed, 6)
    session.set_speed(fps)
    assistance = pathb.install(session, mode=mode, smoke=smoke) if mode != "bare" else None
    return session, assistance


def _product_metrics(session: Any, horizon: int, fps: int, assistance: Any | None) -> dict[str, Any]:
    hero = next(s for s in session.game.snakes if int(s.id) == int(session.hero_id))
    deaths = 0; masses = []; lengths = []; long_deaths = late_deaths = overlap = 0
    started = time.monotonic()
    for index in range(horizon):
        before_alive, before_length = bool(hero.is_alive), int(hero.length)
        session.step(); session.snapshot()
        masses.append(int(hero.length) if hero.is_alive else 0); lengths.append(int(hero.length))
        died = int(hero.id) in session.game.frame_death_causes
        deaths += int(died)
        long_deaths += int(died and before_length >= 800)
        late_deaths += int(died and index + 1 >= 2500)
        overlap += int(died and before_length >= 800 and index + 1 >= 2500)
        # Pacing is part of the deadline contract; do not compensate work away.
        elapsed = time.monotonic() - started - index / fps
        if elapsed < 1.0 / fps: time.sleep(1.0 / fps - elapsed)
    decisions = max(1, horizon)
    state = assistance.state() if assistance is not None else {}
    return {"mass_integral": sum(masses), "relative_mass": 0.0, "deaths_per_1k": deaths * 1000.0 / decisions,
            "median_length": sorted(lengths)[len(lengths)//2], "p95_length": sorted(lengths)[int(.95*(len(lengths)-1))],
            "deaths_length_ge_800": long_deaths, "late_deaths": late_deaths, "late_large_overlap": overlap,
            "controller": state}


def _record_product(root: Path, block: int, row: dict[str, Any], fps: int, args: argparse.Namespace) -> None:
    target = root / "records" / f"product-b{block:02d}-s{row['seed']}-{fps}.json"
    if target.exists(): raise RuntimeError(f"existing record: {target}")
    arms = {}
    for mode in ("bare", "async", "sync"):
        session = assistance = None
        try:
            session, assistance = _arm_session(int(row["seed"]), fps, mode, args.smoke)
            arms[mode] = _product_metrics(session, args.horizon, fps, assistance)
        finally:
            if assistance is not None: assistance.close(wait=True)
    _write_new(target, {"schema": "t34-pathb-product/v1", "status": "complete", "kind": "product", "block": block,
                        "seed": row["seed"], "fps": fps, "arms": arms})


def efficacy(args: argparse.Namespace) -> None:
    root = args.out.resolve(); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol_sha256"] != sha(HERE / "PROTOCOL.md"): raise RuntimeError("protocol changed")
    deadline = time.monotonic() + args.max_hours * 3600
    # Product uses the actual session constructor. Terminal uses historical roster plans;
    # execution is intentionally refused until the live terminal adapter is supplied.
    for block in manifest["bank"]["blocks"]:
        for row in block["product"]:
            for fps in (12, 24):
                if time.monotonic() >= deadline: raise RuntimeError("max-hours cap reached; bank INCOMPLETE")
                _record_product(root, int(block["block"]), row, fps, args)
    raise RuntimeError("terminal historical-roster live adapter is required; no terminal efficacy claim created")


def timing(args: argparse.Namespace) -> None:
    from research.t15_mixed_roster_20261017.measurements import _timed_tick
    root = args.out.resolve(); seed = json.loads((root / "manifest.json").read_text())["bank"]["blocks"][0]["product"][0]["seed"]
    for mode in ("bare", "async"):
        for fps in (12, 24):
            session, obj = _arm_session(seed, fps, mode, args.smoke); samples = []
            try:
                for _ in range(args.horizon): samples.append(_timed_tick(session))
            finally:
                if obj is not None: obj.close(wait=True)
            _write_new(root / "timing" / f"{mode}-{fps}.json", {"status":"complete", "mode":mode,"fps":fps,
                       "samples": samples, "client_latency":"NOT_MEASURED"})


def estimate(args: argparse.Namespace) -> None:
    worlds = args.blocks * (args.scripted_worlds + args.frozen_worlds + args.mixed_worlds + args.product_worlds)
    # Conservative no-speedup lower bound: every paired arm is paced at 24fps.
    hours = worlds * args.horizon * 3 * 2 / 24 / 3600
    print(json.dumps({"mac_core_hours_lower_bound": hours, "limitations": "pacing-only analytic bound; smoke timing required for estimate"}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("bank", "efficacy", "timing", "smoke", "estimate")); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--scripted-worlds", type=int, default=96); p.add_argument("--frozen-worlds", type=int, default=48); p.add_argument("--mixed-worlds", type=int, default=48); p.add_argument("--product-worlds", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5000); p.add_argument("--max-hours", type=float, default=24.0); p.add_argument("--smoke", action="store_true")
    a=p.parse_args()
    if a.action == "estimate": estimate(a)
    elif a.action in ("bank", "smoke"):
        if a.action == "smoke": a.smoke=True; a.blocks=1; a.scripted_worlds=a.frozen_worlds=a.mixed_worlds=0; a.product_worlds=1; a.horizon=min(a.horizon, 8)
        print(json.dumps({"bank": str(create_bank(a))}))
    elif a.action == "efficacy": efficacy(a)
    else: timing(a)
    return 0

if __name__ == "__main__": raise SystemExit(main())
