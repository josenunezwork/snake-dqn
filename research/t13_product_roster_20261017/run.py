"""Run fixed T13 episodes: one coordinator plus at most four CPU subprocesses.

No retries, tmux interaction, or external-process management. Each episode has an
exclusive JSON result and stderr log; a failed batch is preserved for diagnosis.
"""

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
ROSTERS = ("all_b8", "all_frp3", "all_champion", "mixed")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--episode", choices=ROSTERS)
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
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from research.t13_product_roster_20261017.episodes import run_watch

        result = run_watch(args.episode, args.seed, horizon=args.horizon)
        with args.out.open("x") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.write("\n")
        return
    if args.horizon != 5000:
        raise ValueError("full T13 measurement requires H5000")
    audit = json.loads((STUDY / "SEED_AUDIT.json").read_text())
    if not audit["pass"] or len(audit["world_seeds"]) != 25:
        raise ValueError("passing 25-world freshness receipt required")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "episodes").mkdir()
    (args.out / "logs").mkdir()
    sources = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "web/backend").rglob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "web/backend/served_checkpoint_pins.json",
        *STUDY.glob("*.py"),
        STUDY / "SEED_AUDIT.json",
        STUDY / "protocol.md",
        ROOT / "docs/research/specs/t13_roster_spec_2026-10-17.md",
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    intent = {
        "schema": "t13-run/v1",
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
        "rosters": list(ROSTERS),
        "horizon": 5000,
        "device": "cpu",
        "workers": args.workers,
        "max_processes": args.workers + 1,
        "command": sys.argv,
        "started_unix": time.time(),
    }
    (args.out / "intent.json").write_text(json.dumps(intent, indent=2, sort_keys=True) + "\n")

    def launch(roster: str, seed: int) -> dict:
        stem = f"{roster}-{seed}"
        result = args.out / "episodes" / f"{stem}.json"
        command = [
            sys.executable,
            "-m",
            "research.t13_product_roster_20261017.run",
            "--episode",
            roster,
            "--seed",
            str(seed),
            "--out",
            str(result),
        ]
        with (args.out / "logs" / f"{stem}.log").open("x") as log:
            started = time.time()
            process = subprocess.Popen(
                command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
            )
            code = process.wait()
        return {
            "roster": roster,
            "seed": seed,
            "pid": process.pid,
            "returncode": code,
            "wall_seconds": time.time() - started,
            "sha256": digest(result) if result.exists() else None,
        }

    receipts = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(launch, roster, seed) for seed in audit["world_seeds"] for roster in ROSTERS
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
