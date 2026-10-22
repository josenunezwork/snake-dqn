"""Check real tiny episode semantics across serial and bounded spawned execution."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from research.bl_density.runner import run_episode
from research.bl_ppo.exports import export_json, verify_export


def semantic(value: Any) -> Any:
    """Exclude elapsed CPU/wall measurements; preserve all gameplay/features."""
    if isinstance(value, dict):
        return {
            key: semantic(item)
            for key, item in value.items()
            if not key.endswith("seconds")
            and not key.startswith("metrics_")
            and key not in {"worker_pid", "cpu_seconds_by_stage"}
        }
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(semantic(value), sort_keys=True, allow_nan=False).encode()


def verify(reference: Path, b8: Path, workers: int) -> dict[str, Any]:
    """Replay a small completed tiny reference bank with at most two workers."""
    if workers not in (1, 2):
        raise ValueError("Mac semantic probe permits one or two workers")
    verify_export(reference / "summary.json")
    summary = json.loads((reference / "summary.json").read_text())
    if not summary.get("tiny_preflight") or summary["status"] != "COMPLETE":
        raise ValueError("complete tiny reference only")
    records = []
    for path in sorted((reference / "episodes").glob("*.json")):
        verify_export(path)
        records.append(json.loads(path.read_text()))
    if not 1 <= len(records) <= 12 or any(r["horizon"] > 8 for r in records):
        raise ValueError("probe exceeds tiny episode allowance")
    jobs = []
    for record in records:
        plan = record.get("plan") or {
            "seed": record["seed"],
            "mix": record["mix"],
            "specs": record["construction"]["roster_specs"][:5],
        }
        jobs.append(
            {
                "b8": str(b8),
                "arm": record["arm"],
                "mix": record["mix"],
                "index": int(record["episode_id"].rsplit("/", 1)[1]),
                "horizon": record["horizon"],
                "tiny": True,
                "deadline": None,
                "plan": plan,
            }
        )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        replayed = list(pool.map(run_episode, jobs))
    rows = []
    for expected, actual in zip(records, replayed):
        expected_bytes, actual_bytes = _canonical(expected), _canonical(actual)
        if expected_bytes != actual_bytes:
            raise ValueError("serial/parallel semantic mismatch: " + expected["episode_id"])
        rows.append(
            {
                "episode_id": expected["episode_id"],
                "semantic_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "h300_calls": actual["h300_calls"],
                "h300_branch_decisions": actual["h300_branch_decisions"],
            }
        )
    return {
        "schema": "bl-density-parallel-semantic-probe/v1",
        "status": "IDENTICAL",
        "workers": workers,
        "episodes": rows,
        "scope": "tiny CPU gameplay/features only; timing excluded; no cross-platform claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.b8, args.workers)
    export_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
