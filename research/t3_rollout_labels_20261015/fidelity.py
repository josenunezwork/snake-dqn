"""Reproduce a deterministic random sample of T1 continuation outcomes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from research.t1_commitment_20261015.replay import (
    ARTIFACTS,
    advance_recorded,
    build_replay,
    configure_cpu,
    load_inputs,
)
from research.t1_commitment_20261015.study import rollout
from research.t3_rollout_labels_20261015.engine import batched_rollout


def _recorded_measurements() -> list[tuple[str, dict, dict]]:
    rows: list[tuple[str, dict, dict]] = []
    for path in sorted((ARTIFACTS / "rows").glob("b8_s*.jsonl")):
        arm = path.stem
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for scan in row["scans"]:
                if scan["advisory_legal_actions"]:
                    rows.append((arm, row, scan))
    if not rows:
        raise FileNotFoundError(f"no T1 measured rows under {ARTIFACTS / 'rows'}")
    return rows


def fidelity_check(samples: int = 20, seed: int = 20261015) -> dict[str, Any]:
    """Compare all recorded advisory continuations for a reproducible T1 sample."""
    configure_cpu()
    pool = _recorded_measurements()
    picked = random.Random(seed).sample(pool, min(samples, len(pool)))
    total_frames, total_seconds, comparisons = 0, 0.0, []
    cache: dict[str, tuple[Any, Any, Any]] = {}
    for arm, row, scan in picked:
        if arm not in cache:
            data, series = load_inputs(arm)
            from research.b8_death_census_20261014 import census

            cache[arm] = (series, census.load_context(), data)
        series, context, _data = cache[arm]
        replay = build_replay(arm, [int(row["env"])], context)
        decision = int(scan["decision"])
        advance_recorded(replay, series, [int(row["env"])], decision)
        prepared = None

        def capture(current: Any) -> None:
            nonlocal prepared
            prepared = current.clone(prepared=True)

        # The observer is the true T1 pre-action state: its frame, food and
        # respawn phase has already run.  Do not manufacture it from a raw frame.
        replay.step([int(series["action"][int(row["env"]), decision])], capture)
        assert prepared is not None
        actions = [int(a) for a in scan["advisory_legal_actions"]]
        result = batched_rollout(prepared, actions, decision)
        receipt = result.pop("_batch_receipt")
        total_frames += int(receipt["env_frames"])
        total_seconds += float(receipt["wall_seconds"])
        for action in actions:
            expected = rollout(prepared, action, decision)
            actual = result[action]
            for key in ("survives", "frames_survived", "frames_executed", "death_decision"):
                assert actual[key] == expected[key], (
                    arm,
                    row["env"],
                    decision,
                    action,
                    key,
                    actual[key],
                    expected[key],
                )
        comparisons.append(
            {"arm": arm, "env": row["env"], "decision": decision, "actions": actions}
        )
    return {
        "status": "PASS",
        "samples": len(picked),
        "comparisons": comparisons,
        "env_frames": total_frames,
        "wall_seconds": total_seconds,
        "env_frames_per_second": total_frames / total_seconds if total_seconds else 0.0,
        "batches_per_second": len(picked) / total_seconds if total_seconds else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    receipt = fidelity_check(args.samples, args.seed)
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
