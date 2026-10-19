"""Matched three-round old repeated-search versus single-prefix ablation probe."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import types
from dataclasses import asdict
from pathlib import Path

from research.phase0_search_20261019 import teacher
from research.t33_batched_teacher import ablation
from research.t33_batched_teacher.benchmark import prepared
from research.t33_batched_teacher.evidence import identity

BASE = "f76820776e39a6a53c13197f6f0948aa7c6a95f4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    teacher.configure_cpu()
    old = types.ModuleType("t33_repeated_reference")
    exec(
        compile(
            subprocess.check_output(
                ["git", "show", f"{BASE}:research/t33_batched_teacher/ablation.py"], text=True
            ),
            "<frozen ablation>",
            "exec",
        ),
        {**old.__dict__, "__file__": ablation.__file__},
    )
    # Explicit namespace needed because the frozen function resolves its own globals.
    namespace = {"__file__": ablation.__file__, "__name__": "t33_repeated_reference"}
    exec(
        subprocess.check_output(
            ["git", "show", f"{BASE}:research/t33_batched_teacher/ablation.py"], text=True
        ),
        namespace,
    )
    world = prepared(ablation.fixture(ablation.calibration_rows(smoke=True)[2], ablation.B8, 800))
    settings = asdict(teacher.SearchConfig(horizon=16, rollouts=4, length_gate=0))
    choices = ablation.variants([4, 8], [2, 4], 1.0)
    funcs = {"before": namespace["compare"], "after": ablation.compare}
    for fn in funcs.values():
        fn(world, settings, choices, None)
    results = {key: [] for key in funcs}
    for _ in range(3):
        outputs = {}
        for key, fn in funcs.items():
            start = time.perf_counter()
            outputs[key] = fn(world, settings, choices, None)
            results[key].append(time.perf_counter() - start)
        for old_variant, new_variant in zip(
            outputs["before"]["variants"], outputs["after"]["variants"]
        ):
            assert old_variant["scores"] == new_variant["scores"]
            assert old_variant["action"] == new_variant["action"]
    rows = 6 * settings["rollouts"]
    receipt = {
        "schema": "t33-prefix-microbench/v1",
        "base": BASE,
        "source": identity(),
        "settings": settings,
        "choices": choices,
        "threads": 1,
        "rounds": results,
        "resident_rows": rows,
        "equivalent_full_rows_per_s": {k: rows / statistics.median(v) for k, v in results.items()},
        "speedup": statistics.median(results["before"]) / statistics.median(results["after"]),
        "parity": "all variant scores and choices equal in all three rounds",
        "note": "frozen old compare uses current independent reference.scores; same host kernels",
    }
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "source"}))


if __name__ == "__main__":
    main()
