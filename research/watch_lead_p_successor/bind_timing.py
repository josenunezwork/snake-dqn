"""Future operator preparation only: bind measured transport proof, never dispatch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p_successor.delivery import require_transfer_margin
from research.watch_lead_p_successor.prepare import write


def bind(packet: Path, receipt: Path, qualified_record: Path, output: Path) -> dict:
    """Refuse pending transport, weak margin, foreign family/source or unapproved tree."""
    repo = Path(__file__).resolve().parents[2]
    target = json.loads((packet / "transport-target.json").read_text())
    if not target.get("target_transport_tree") or not target.get("target_transport_commit"):
        raise ValueError("re-pin to the final transport producer first")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD:research/colab_cli"], cwd=repo, text=True
    ).strip()
    if actual != target["target_transport_tree"]:
        raise ValueError("current transport differs from packet pin")
    from tools.ops.qualification import check

    status = json.loads(qualified_record.read_text()).get("independent_review_status", "")
    if not isinstance(status, str) or status.upper().startswith(("PENDING", "HOLD", "REJECT")):
        raise ValueError("independent transport approval is still pending/refused")
    check(repo, qualified_record)  # Final transport checker is the approval authority.
    seal = json.loads((packet / "seal.json").read_text())
    for name, checksum in seal.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest(packet / name) != checksum:
            raise ValueError("packet seal drift")
    if output.exists() or not output.is_absolute():
        raise ValueError("fresh absolute long-packet directory required")
    manifest = json.loads((packet / "full-session-manifest.TEMPLATE.json").read_text())
    receipt = receipt.resolve(strict=True)
    manifest["canary"] = {"receipt": {"path": str(receipt), "sha256": digest(receipt)}}
    manifest["output_dir"] = str(output / "output")
    from research.colab_cli.supervisor import validate

    # This re-derives source, export custody, hardware, exact inputs and stop proof.
    validate(manifest, repo)
    proof = json.loads(receipt.read_text())["canary_evidence"]
    margin = require_transfer_margin(
        {
            "transfer_seconds": proof["transfer_seconds"],
            "elapsed_seconds": proof["upload_elapsed_seconds"],
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", manifest)
    write(
        output / "proof-binding.json",
        {
            "packet_seal_sha256": digest(packet / "seal.json"),
            "canary_receipt_sha256": digest(receipt),
            "qualification_sha256": digest(qualified_record),
            "transfer_margin": margin,
            "transport": target,
        },
    )
    (output / "DISPATCH-NOT-RUN.txt").write_text(
        "PREPARED ONLY. Operator must use qualified ops dry-run and separately dispatch.\n"
        "Fresh source/input/receipt proof bound; no allocation, retry or data look was performed.\n"
    )
    write(
        output / "seal.json", {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--qualified-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bind(args.packet, args.receipt, args.qualified_record, args.output)


if __name__ == "__main__":
    main()
