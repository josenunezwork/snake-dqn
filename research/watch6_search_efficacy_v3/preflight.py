"""Excluded Mac actual-entrypoint fixture plus long-body H300 branch timing for Watch6 V3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.run import ledger_identity
from tools.ops.package_spec import empty_ledger, publish_json, verify_immutable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ledger = args.out / "verified-ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", ledger_identity()))
    ckpt_dir = str(p.ROOT.parent / "snake-dqn/saved_snakes")
    env = {
        **os.environ,
        "SNAKE_DQN_DEVICE": "cpu",
        "SNAKE_WATCH_SOURCE": "live",
        "SNAKE_CKPT_DIR": ckpt_dir,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch6_search_efficacy_v3.run",
            "--ledger",
            str(ledger),
            "--out",
            str(args.out / "tiny"),
            "--workers",
            "1",
            "--fixture",
            "--fixture-horizon",
            "12",
            "--canary-performance-out",
            str(args.out / "canary-performance.json"),
        ],
        cwd=p.ROOT,
        env=env,
        check=True,
    )
    tiny_seconds = time.monotonic() - started
    for idx in range(len(p.FIXTURE_SEEDS)):
        verify_immutable(args.out / "tiny" / f"fixture-pair-{idx:02d}.json")
        verify_immutable(args.out / "tiny" / f"fixture-pair-{idx:02d}.meta.json")
    if list((args.out / "tiny").glob("pair-*.json")):
        raise ValueError("fixture mode must never emit scientific pair-XX.json filenames")
    verify_immutable(args.out / "canary-performance.json")

    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    os.environ["SNAKE_CKPT_DIR"] = ckpt_dir
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.performance import branch_fixture

    configure_cpu()
    timings = [branch_fixture(seed) for seed in p.FIXTURE_SEEDS]
    payload = {
        "schema": "watch6-v3-preflight/v1",
        "science": p.identity(),
        "tiny_pair_count": len(p.FIXTURE_SEEDS),
        "tiny_frames_per_arm": 12,
        "tiny_Plus_canary_check_seconds": tiny_seconds,
        "branch_timings": timings,
        "cpu_count": os.cpu_count(),
        "scope": "Mac real-entrypoint preflight + H300 long-body branch timings; no outcome claim",
    }
    publish_json(args.out / "benchmark.json", payload)
    print(json.dumps({"tiny_seconds": tiny_seconds, "branch_count": len(timings)}), flush=True)


if __name__ == "__main__":
    main()
