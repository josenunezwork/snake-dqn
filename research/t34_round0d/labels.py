"""Sealed teacher-arm labels: admissible to dense R1 fitting only after a published GO.

Rule (PREREGISTRATION_ROUND0D.md, "Sealed labels"): the 240 teacher-arm label
exports under ``chunks/*/attempt-*/job/exports/round0d/sealed/`` are never opened
by replay, analysis, or any operator before the sole decision. They are
admissible to R1 *fitting* only if the published ``analysis/decision.json`` is
GO, and never to any screen, probe, or gate bank. This module is the only
sanctioned reader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from research.t34_round0d.common import read_json, sha256


class SealedLabelsRefused(PermissionError):
    """The sealed Round 0-D labels are not admissible for this request."""


def admit(registration: Mapping[str, Any], purpose: str) -> list[dict[str, Any]]:
    """Return the hash-bound sealed inventory after verifying the published GO."""
    from research.t34_round0d.history import policy, require_published_history

    if purpose != "dense_r1_fit":
        raise SealedLabelsRefused("sealed labels are never admitted to screens, probes or gates")
    root, _, _ = policy(registration)
    decision_path = root / "analysis" / "decision.json"
    if not decision_path.is_file():
        raise SealedLabelsRefused("no sole Round 0-D decision exists; labels stay sealed")
    terminals = require_published_history(registration)
    analysis = [t for t in terminals if t["phase_path"] == str(root / "analysis")]
    decision = read_json(decision_path)
    if (
        len(analysis) != 1
        or analysis[0]["status"] != "COMPLETE"
        or analysis[0]["files"].get("decision.json") != sha256(decision_path)
        or decision.get("schema") != "t34-round0d-decision/v1"
        or decision.get("decision") != "GO"
        or decision.get("evidence_authority") != "TIER_1_DEVELOPMENT"
    ):
        raise SealedLabelsRefused("the published sole Round 0-D decision is not GO")
    inventory = []
    for chunk_sha in decision["chunk_sha256"]:
        chunk_path = Path(chunk_sha)
        if sha256(chunk_path) != decision["chunk_sha256"][chunk_sha]:
            raise SealedLabelsRefused("chunk receipt changed after the decision")
        receipt = read_json(chunk_path)
        for item in receipt["sealed_labels"]:
            path = chunk_path.parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise SealedLabelsRefused("sealed label export changed after sealing")
            inventory.append({**item, "absolute_path": str(path)})
    if len(inventory) != 240:
        raise SealedLabelsRefused("sealed inventory must contain exactly the 240 teacher worlds")
    return inventory


def iter_labels(inventory: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield normalized label rows (decisive/delay/confirm roles retained)."""
    from research.t34_exit_b8.exports import load_world

    for item in inventory:
        world = load_world(item["absolute_path"])
        for label in world["labels"]:
            yield {"world_id": world["world_id"], **label}


def measure_yield(inventory: list[dict[str, Any]]) -> dict[str, float]:
    """Dense R1 yield inputs (after GO only): decisive rows/world, decisive-bearing fraction."""
    from research.t34_exit_b8.exports import load_world

    rows, bearing = 0, 0
    for item in inventory:
        world = load_world(item["absolute_path"])
        decisive = sum(label.get("label_kind") == "decisive" for label in world["labels"])
        rows += decisive
        bearing += int(decisive > 0)
    return {
        "decisive_rows_per_world": rows / len(inventory),
        "decisive_world_fraction": bearing / len(inventory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratification", type=Path, required=True)
    args = parser.parse_args()
    inventory = admit(read_json(args.ratification), "dense_r1_fit")
    print(json.dumps({"admitted_worlds": len(inventory)}))


if __name__ == "__main__":
    main()
