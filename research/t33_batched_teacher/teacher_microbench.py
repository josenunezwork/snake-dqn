"""Reproducible, source-bound microbenchmark for the T33 teacher hot path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

from research.phase0_search_20261019 import teacher as phase0
from research.t33_batched_teacher import teacher as candidate
from tests.test_t33_teacher_integration import _prepared_world

BASELINE_OBJECT = "f76820776e39a6a53c13197f6f0948aa7c6a95f4"
ROOT = Path(__file__).resolve().parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _baseline() -> tuple[types.ModuleType, bytes]:
    """Load the frozen baseline from its git object without touching a worktree."""
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_OBJECT}:research/t33_batched_teacher/teacher.py"], text=False
    )
    module = types.ModuleType("t33_teacher_frozen_microbench")
    module.__file__ = "t33_teacher_frozen_microbench.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, source


def _worlds(count: int):
    rows = json.loads((ROOT / "research/phase0_search_20261019/WORLD_BANK.json").read_text())
    row = next(item for item in rows["fixtures"]["timing"] if item["mix"] == "scripted")
    return [_prepared_world(row, 800) for _ in range(count)]


def _rounds(module, *, current: bool, rounds: int) -> dict:
    """Measure isolated assembly and short non-calibration continuation rounds."""
    assembly, search = [], []
    for _ in range(rounds):
        worlds = _worlds(2)
        started = time.perf_counter()
        if current:
            module.merge_worlds(worlds, 4, horizon=32)
        else:
            module.merge_worlds(worlds, 4)
        assembly.append(time.perf_counter() - started)
    for _ in range(rounds):
        worlds = _worlds(2)
        config = phase0.SearchConfig(rollouts=4, horizon=8, epsilon=0.5, length_gate=0)
        started = time.perf_counter()
        module.BatchedTeacher(config, common_random=False).choose(worlds)
        search.append(time.perf_counter() - started)
    return {
        "assembly_seconds": assembly,
        "assembly_rows_per_second": [48 / value for value in assembly],
        "search_seconds": search,
        "search_source_decisions_per_second": [2 / value for value in search],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    baseline, baseline_bytes = _baseline()
    candidate_bytes = (ROOT / "research/t33_batched_teacher/teacher.py").read_bytes()
    result = {
        "kind": "T33 round-2 teacher source-bound microbenchmark",
        "argv": sys.argv,
        "baseline_git_object": BASELINE_OBJECT,
        "baseline_teacher_sha256": _sha(baseline_bytes),
        "candidate_teacher_sha256": _sha(candidate_bytes),
        "rounds": args.rounds,
        "fixture": {"mix": "scripted", "source_worlds": 2, "rollouts": 4},
        "before": _rounds(baseline, current=False, rounds=args.rounds),
        "after": _rounds(candidate, current=True, rounds=args.rounds),
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
