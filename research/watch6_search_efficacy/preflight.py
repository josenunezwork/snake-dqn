"""Excluded Mac actual-entrypoint fixture plus product-config long-body branch timing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import ledger_identity
from tools.ops.package_spec import empty_ledger, publish_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ledger = args.out / "verified-ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", ledger_identity()))
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch6_search_efficacy.run",
            "--ledger",
            str(ledger),
            "--out",
            str(args.out / "tiny"),
            "--fixture",
            "--fixture-horizon",
            "12",
        ],
        check=True,
    )
    tiny_seconds = time.monotonic() - started
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.performance import branch_fixture

    configure_cpu()
    timings = [branch_fixture(seed) for seed in p.FIXTURE_SEEDS]
    publish_json(
        args.out / "benchmark.json",
        {
            "science": p.identity(),
            "tiny_pair_count": 3,
            "tiny_frames_per_arm": 12,
            "tiny_seconds": tiny_seconds,
            "branch_timings": timings,
            "cpu_count": os.cpu_count(),
            "scope": "Mac loaded-host throughput fixture; no product outcome or latency claim",
        },
    )
    print(json.dumps({"tiny_seconds": tiny_seconds, "branch_timings": timings}))


if __name__ == "__main__":
    main()
