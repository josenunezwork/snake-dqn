"""Materialize the actual reused T3 dense inventory without running an episode.

The inventory is 64 source trajectories: two B8 checkpoints on the same 32
scripted worlds. This development replication preserves the source checkpoint
of each trajectory; it does not claim 64 independent physical worlds. Fresh
registered retention still uses only released B8 s3 and independent namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/dense")
ARMS = ("b8_s0", "b8_s3")
NAMESPACE = "t3-dense-ceiling/v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _original_worlds() -> tuple[list[int], dict[int, list]]:
    from research.b8_death_census_20261014.census import load_context

    _context, seeds, rosters = load_context(32)
    return seeds, {seed: [list(spec) for spec in rosters[seed]] for seed in seeds}


def validate_t3_row(row: Mapping[str, Any]) -> None:
    """Refuse any episode that is not its declared original T3 arm/env case."""
    arm, env = row.get("t3_source_arm"), row.get("t3_env")
    if arm not in ARMS or type(env) is not int or env not in range(32):
        raise ValueError("invalid original T3 source-arm/env coordinate")
    path = ROOT / "ceiling" / f"{arm}_{env}.json"
    expected_receipt = {"path": str(path), "sha256": _sha(path)}
    receipt = json.loads(path.read_text())
    seeds, rosters = _original_worlds()
    if (
        row.get("t3_namespace") != NAMESPACE
        or row.get("t3_world") != f"{arm}_{env}"
        or row.get("t3_receipt") != expected_receipt
        or row.get("t3_materialized_sha256") != expected_receipt["sha256"]
        or row.get("t3_checkpoint") != receipt.get("checkpoint")
        or row.get("horizon") != 5000
        or row.get("seed") != seeds[env]
        or row.get("seed") != receipt.get("world_seed")
        or row.get("mix") != "scripted"
        or row.get("specs") != rosters[seeds[env]]
        or receipt.get("status") != "PASS"
        or receipt.get("prefix_identity") is not True
        or receipt.get("arm") != arm
        or receipt.get("env") != env
    ):
        raise ValueError("T3 row differs from its immutable original receipt/roster")
    checkpoint = receipt["checkpoint"]
    if _sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("original T3 checkpoint bytes changed")


def materialize_t3() -> list[dict[str, Any]]:
    """Read the complete original 2x32 bank and its pinned source identities."""
    summary = json.loads((ROOT / "ceiling-go-for-bank.json").read_text())
    if summary.get("status") != "PASS" or summary.get("gate", {}).get("episodes") != 64:
        raise ValueError("T3 dense inventory lacks its original complete receipt")
    seeds, rosters = _original_worlds()
    rows = []
    for arm in ARMS:
        for env in range(32):
            path = ROOT / "ceiling" / f"{arm}_{env}.json"
            receipt = json.loads(path.read_text())
            row = {
                "t3_world": f"{arm}_{env}",
                "t3_source_arm": arm,
                "t3_env": env,
                "t3_namespace": NAMESPACE,
                "t3_receipt": {"path": str(path), "sha256": _sha(path)},
                "t3_materialized_sha256": _sha(path),
                "t3_checkpoint": receipt["checkpoint"],
                "seed": seeds[env],
                "mix": "scripted",
                "specs": [list(spec) for spec in rosters[seeds[env]]],
                "horizon": 5000,
            }
            validate_t3_row(row)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = materialize_t3()
    with args.out.open("x") as stream:
        json.dump(rows, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps({"trajectories": len(rows), "unique_physical_seeds": 32, "out": str(args.out)})
    )


if __name__ == "__main__":
    main()
