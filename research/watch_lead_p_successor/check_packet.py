"""Offline dry-run inspection; never allocates and never reads sealed outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest, export_json
from research.watch_lead_p_successor.delivery import TRANSFER_CAP_SECONDS
from research.watch_lead_p_successor.runtime import verify_sources
from research.watch_lead_p_successor.units import CHECKPOINTS, read_ledger


def check(packet: Path, repo: Path) -> dict:
    seal = json.loads((packet / "seal.json").read_text())
    actual_files = {str(p.relative_to(packet)) for p in packet.rglob("*") if p.is_file()}
    if actual_files != set(seal) | {"seal.json"}:
        raise ValueError("packet has missing/unlisted sealed files")
    for name, checksum in seal.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe sealed packet path")
        if digest(packet / name) != checksum:
            raise ValueError("packet seal drift")
    bound = json.loads((packet / "inputs/binding.json").read_text())
    sources = json.loads((packet / "inputs/sources.json").read_text())
    verify_sources(repo, sources, bound)
    ledger = json.loads((packet / "inputs/verified-ledger.json").read_text())
    attempt = json.loads((packet / "envelope.json").read_text())["attempt"]
    verified = read_ledger(ledger, bound)
    if not attempt and verified:
        raise ValueError("initial engineering packet requires empty first ledger")
    for name, checksum in CHECKPOINTS.items():
        if digest(packet / "inputs" / name) != checksum:
            raise ValueError("staged checkpoint drift")
    from research.colab_cli.supervisor import Refused, validate

    probe = json.loads((packet / "transport-manifest.TEMPLATE.json").read_text())
    full = json.loads((packet / "full-session-manifest.TEMPLATE.json").read_text())
    validate(probe, repo)
    # The deliberately absent family receipt must still block the long packet.
    try:
        validate(full, repo)
    except Refused as exc:
        if "canary receipt" not in str(exc):
            raise
        full_refusal = str(exc)
    else:
        raise ValueError("long template unexpectedly admitted without target proof")
    target = json.loads((packet / "transport-target.json").read_text())
    observed = subprocess.check_output(
        ["git", "rev-parse", f"{probe['commit']}:research/colab_cli"], cwd=repo, text=True
    ).strip()
    if observed != target["source_transport_tree"]:
        raise ValueError("source transport pin differs")
    return {
        "packet_sha256": digest(packet / "seal.json"),
        "source_commit": probe["commit"],
        "sealed_files": len(seal),
        "staged_checkpoint_sha256": CHECKPOINTS,
        "native_template_validation": "PASS_OFFLINE_ONLY",
        "long_template_validation": "EXPECTED_REFUSAL",
        "long_refusal": full_refusal,
        "transport": target,
        "transfer_cap_seconds": TRANSFER_CAP_SECONDS,
        "dispatch_status": "DISPATCH_NOT_RUN",
        "closed_payloads_read": False,
        "remaining": [
            "final published astra-impl-colab-transport tree and independent approval; re-pin",
            "Argon implementation cross-review",
            "same-pin native family receipt and >=45s measured transfer margin",
            "target full H6000 timings, actual tails, sustained contention and shard scheduling",
            "fresh scientific allocation audit, frozen envelopes and independent ratification",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.packet, Path(__file__).resolve().parents[2])
    export_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
