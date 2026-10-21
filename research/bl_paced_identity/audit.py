"""Verify a bounded preflight export tree, including its fresh replay, off tick."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .events import sha256
from .preflight import TinyGame
from .record import read_export, read_journal
from .replay import replay_events, replay_journal


def audit_preflight(directory: Path) -> dict[str, Any]:
    """Verify only explicitly referenced X7 fixtures; never scan other artifacts."""
    receipt = read_export(directory / "receipt.json")
    manifest = read_export(directory / "manifest.json")
    semantic = read_export(directory / "semantic.json")
    replay = read_export(directory / "replay.json")
    fixtures = read_export(directory / "interleavings/fixtures.json")
    if manifest["receipt_sha256"] != sha256(receipt) or manifest["dispatchable"] is not False:
        raise ValueError("manifest/receipt binding or dispatchability mismatch")
    if semantic != receipt["oracles"]["semantic"] or replay != receipt["oracles"]["replay"]:
        raise ValueError("oracle export/join mismatch")
    for label in ("source_sha256", "reference_sha256"):
        for raw_path, expected in receipt[label].items():
            if hashlib.sha256(Path(raw_path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"source changed: {raw_path}")
    if replay_journal(directory / "real-game-trace", TinyGame) != replay:
        raise ValueError("fresh independent real-game replay changed")
    for row in fixtures["fixtures"]:
        fixture_id = row["fixture_id"]
        if Path(fixture_id).name != fixture_id:
            raise ValueError("unsafe fixture name")
        header, events, seal = read_journal(directory / "interleavings" / fixture_id)
        observed = replay_events(header, events)
        if seal != row["seal"] or observed["scheduler_verdict"] != row["expected"]:
            raise ValueError(f"fixture identity changed: {fixture_id}")
    if receipt["verdict"] != "INCOMPLETE" or receipt["l3_identity_prerequisites"] != "BLOCKED":
        raise ValueError("tiny preflight incorrectly qualified a candidate")
    if receipt["oracles"]["paced"]["verdict"] != "NOT_RUN":
        raise ValueError("preflight contains unexpected paced evidence")
    return {
        "audit": "PASS",
        "receipt_sha256": sha256(receipt),
        "manifest_sha256": sha256(manifest),
        "fixture_count": fixtures["fixture_count"],
        "replay": replay["verdict"],
        "qualification": receipt["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_preflight(args.preflight.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
