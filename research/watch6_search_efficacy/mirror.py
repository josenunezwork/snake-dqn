"""Excluded, full H14400 Mac mirror qualification; never a scientific pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import digest, publish_json


def execution_identity() -> str:
    """Bind gameplay/mirror code independently of later operational preparation."""
    paths = set()
    for name in p.source_paths():
        path = p.ROOT / name
        paths.update(path.rglob("*.py") if path.is_dir() else [path])
    values = {
        str(path.relative_to(p.ROOT)): digest(path)
        for path in sorted(paths)
        if path.is_file()
        and (
            not str(path.relative_to(p.ROOT)).startswith("research/watch6_search_efficacy/")
            or path.name in ("engine.py", "product.py", "mirror.py")
        )
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate(receipt: dict) -> None:
    """Refuse incomplete, stale, non-Mac or respawn-free mirror evidence."""
    if (
        receipt.get("status") != "PASS"
        or receipt.get("platform") != "Darwin"
        or receipt.get("execution_identity") != execution_identity()
        or receipt.get("seed") != p.FIXTURE_SEEDS[0]
        or receipt.get("horizon") != p.HORIZON
        or receipt.get("mirror_identity_frames") != p.HORIZON
        or receipt.get("mirror_mismatches") != 0
        or len(receipt.get("respawns_per_slot", [])) != 6
        or receipt["respawns_per_slot"][0] < 2
        or sum(receipt["respawns_per_slot"]) < 3
    ):
        raise ValueError("requires current full Mac bare mirror through several respawns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac qualification only")
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.engine import run_arm

    configure_cpu()
    identity = execution_identity()
    started = time.monotonic()
    raw = run_arm(p.FIXTURE_SEEDS[0], False, lambda: None, p.HORIZON)
    receipt = {
        "schema": "watch6-full-mirror/v1",
        "status": "PASS",
        "platform": platform.system(),
        "execution_identity": identity,
        "seed": p.FIXTURE_SEEDS[0],
        "fixture": True,
        "horizon": p.HORIZON,
        "mirror_identity_frames": raw["mirror_identity_frames"],
        "mirror_mismatches": 0,
        "respawns_per_slot": raw["respawns_per_slot"],
        "seconds": time.monotonic() - started,
        "main_seconds_per_frame": raw["timing"]["main_seconds"] / p.HORIZON,
    }
    validate(receipt)
    publish_json(args.out, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
