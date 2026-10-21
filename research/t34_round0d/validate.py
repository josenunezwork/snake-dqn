"""Read-only validation of the excluded Round 0-D A6 Mac preflight receipt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.t34_round0d.bank import check_audit
from research.t34_round0d.common import (
    B8_SHA,
    PACKAGE,
    head,
    read_json,
    sha256,
    source_hashes,
    write_once,
)


def validate(preflight_dir: Path) -> dict[str, Any]:
    """Check an excluded preflight without producing worlds, approving, or allocating."""
    bank = check_audit(PACKAGE / "BANK_ROUND0D.json", PACKAGE / "SEED_AUDIT_ROUND0D.json")
    excluded = {row["world_id"] for row in bank["excluded_preflight_rows"]}
    prereg = (PACKAGE / "PREREGISTRATION_ROUND0D.md").read_text()
    if "DRAFT" not in prereg.splitlines()[2]:
        raise ValueError("draft preregistration status is absent")
    chunk = read_json(preflight_dir / "chunk.json")
    operational = chunk.get("operational", {})
    if (
        chunk.get("schema") != "t34-round0d-chunk/v1"
        or chunk.get("status") != "complete"
        or chunk.get("preflight") is not True
        or operational.get("threads") != 1
        or operational.get("device") != "cpu"
        or chunk.get("platform", {}).get("system") != "Darwin"
    ):
        raise ValueError("A6 is not a complete excluded Darwin CPU/one-thread preflight")
    identity = chunk.get("run_identity", {})
    if identity.get("source_hashes") != source_hashes():
        raise ValueError("A6 executable source differs from current source")
    if identity.get("checkpoint_sha256") != B8_SHA:
        raise ValueError("A6 did not run released B8 s3")
    if set(chunk["slice"]["world_ids"]) - excluded:
        raise ValueError("A6 touched a selection world")
    sealed = {item["world_id"]: item for item in chunk.get("sealed_labels", [])}
    for item in chunk["pairs"]:
        pair_path = preflight_dir / item["path"]
        if sha256(pair_path) != item["sha256"]:
            raise ValueError("A6 pair hash differs")
        pair = read_json(pair_path)
        if pair["world"]["world_id"] not in excluded or pair["world"]["horizon"] != 500:
            raise ValueError("A6 contains selection data")
        for arm in ("bare", "teacher"):
            meta = pair["arms"][arm]["trace"]
            trace = read_json(preflight_dir / meta["path"])
            if sha256(preflight_dir / meta["path"]) != meta["sha256"] or trace.get("labels"):
                raise ValueError("A6 public trace differs or carries labels")
        seal = sealed.get(pair["world"]["world_id"])
        if seal is None or sha256(preflight_dir / seal["path"]) != seal["sha256"]:
            raise ValueError("A6 sealed label export is missing or differs")
    return {
        "schema": "t34-round0d-preflight-validation/v1",
        "status": "PASS",
        "commit": head(),
        "preflight": {
            "path": str(preflight_dir / "chunk.json"),
            "sha256": sha256(preflight_dir / "chunk.json"),
            "pairs": len(chunk["pairs"]),
            "elapsed_seconds": chunk["elapsed_seconds"],
            "horizon": 500,
        },
        "source_hashes": source_hashes(),
        "ratified": False,
        "dispatch_authorized": False,
        "scientific_arms_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, validate(args.preflight_dir))


if __name__ == "__main__":
    main()
