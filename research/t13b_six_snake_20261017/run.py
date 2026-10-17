"""Run fixed T13b episodes with four single-thread CPU workers plus coordinator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
SIX_ROSTERS = ("all_b8", "all_frp3", "all_champion", "mixed")
RUNS = tuple((roster, 6) for roster in SIX_ROSTERS) + (("all_b8", 8),)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stem(roster: str, snakes: int, seed: int) -> str:
    return f"{roster}-n{snakes}-{seed}"


def _sources() -> list[Path]:
    """Hash all runtime and study source, including immutable T13 reuse source."""
    groups = (
        ROOT / "src",
        ROOT / "web/backend",
        ROOT / "research/t13_product_roster_20261017",
        ROOT / "research/t12_b8_serving_20261017",
        ROOT / "research/apex_veto_serving_20261001",
        STUDY,
    )
    paths = [p for group in groups for p in group.rglob("*.py")]
    paths += list((ROOT / "configs").glob("*.yaml"))
    paths += [
        ROOT / "web/backend/served_checkpoint_pins.json",
        ROOT / "docs/research/specs/t13b_six_snake_spec_2026-10-17.md",
        ROOT / "docs/research/t13b_six_snake_roster_2026-10-17.md",
        STUDY / "SEED_AUDIT.json",
        STUDY / "protocol.md",
    ]
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--episode", choices=SIX_ROSTERS)
    parser.add_argument("--snakes", type=int, choices=(6, 8))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int, default=5000)
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    if args.episode:
        if args.seed is None or args.snakes is None:
            raise ValueError("single episode requires --seed and --snakes")
        if args.snakes == 8 and args.episode != "all_b8":
            raise ValueError("eight-snake supplemental run is all_b8 only")
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from research.t13b_six_snake_20261017.episodes import run_watch

        result = run_watch(args.episode, args.seed, horizon=args.horizon, snakes=args.snakes)
        with args.out.open("x") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.write("\n")
        return
    if args.horizon != 5000:
        raise ValueError("full T13b measurement requires H5000")
    audit = json.loads((STUDY / "SEED_AUDIT.json").read_text())
    if not audit["pass"] or len(audit["world_seeds"]) != 25:
        raise ValueError("passing 25-world freshness receipt required")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    (args.out / "logs").mkdir()
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in _sources()}
    intent = {
        "schema": "t13b-run/v1",
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "numba")
        },
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": source_hashes,
        "world_seeds": audit["world_seeds"],
        "runs": [{"roster": roster, "snake_count": snakes} for roster, snakes in RUNS],
        "horizon": 5000,
        "device": "cpu",
        "workers": args.workers,
        "max_processes": args.workers + 1,
        "command": sys.argv,
        "started_unix": time.time(),
    }
    (args.out / "intent.json").write_text(json.dumps(intent, indent=2, sort_keys=True) + "\n")

    def launch(roster: str, snakes: int, seed: int) -> dict:
        name = stem(roster, snakes, seed)
        result = args.out / "episodes" / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "research.t13b_six_snake_20261017.run",
            "--episode",
            roster,
            "--snakes",
            str(snakes),
            "--seed",
            str(seed),
            "--out",
            str(result),
        ]
        with (args.out / "logs" / f"{name}.log").open("x") as log:
            started = time.time()
            process = subprocess.Popen(
                command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
            )
            code = process.wait()
        return {
            "roster": roster,
            "snake_count": snakes,
            "seed": seed,
            "pid": process.pid,
            "returncode": code,
            "wall_seconds": time.time() - started,
            "sha256": digest(result) if result.exists() else None,
        }

    receipts = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(launch, roster, snakes, seed)
            for seed in audit["world_seeds"]
            for roster, snakes in RUNS
        ]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            with (args.out / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            print(json.dumps({"completed": len(receipts), **receipt}), flush=True)
            if receipt["returncode"]:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"episode failed; evidence retained: {receipt}")
    if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("measurement source changed during run")
    (args.out / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "episodes": len(receipts),
                "finished_unix": time.time(),
                "receipts": receipts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
