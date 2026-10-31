"""Seal main.json from main-unsealed.json using the final passing canary receipt in ready-queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research.colab_cli import canary
from tools.ops import auto_retry, ready_queue
from tools.ops.package_spec import digest


def resolve_canary_receipt(ops: Path, canary_id: str) -> Path:
    """Follow ready_queue auto-retry and internal -retry-1 chains to the final PASS receipt."""
    queue = auto_retry.read(ops / "ready-queue.json")
    row = ready_queue.latest(queue, canary_id)
    if row.get("status") != "PASS":
        raise ValueError(f"canary {canary_id!r} has not completed with status PASS")
    name = row["argv"][1]
    internal = ops / "manifests" / f"{name}-retry-1.json"
    manifest_path = internal if internal.exists() else Path(row["manifest"])
    manifest = auto_retry.read(manifest_path)
    receipt_path = Path(manifest["output_dir"]) / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"canary receipt missing at {receipt_path}")
    return receipt_path


def seal(
    packet: Path,
    broker: Path,
    *,
    ops: Path | None = None,
    canary_id: str | None = None,
    canary_receipt: Path | None = None,
) -> Path:
    """Bind canary receipt into main-unsealed.json, validate canary.check, write main.json once."""
    packet = packet.resolve()
    broker = broker.resolve()
    if (canary_receipt is None) == (ops is None or canary_id is None):
        raise ValueError("supply either --canary-receipt OR both --ops and --canary-id")
    receipt_path = (
        canary_receipt.resolve()
        if canary_receipt is not None
        else resolve_canary_receipt(ops.resolve(), str(canary_id))
    )
    unsealed_path = packet / "main-unsealed.json"
    if not unsealed_path.is_file():
        raise ValueError(f"missing unsealed template: {unsealed_path}")
    out_path = packet / "main.json"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sealed manifest: {out_path}")

    manifest = json.loads(unsealed_path.read_text(encoding="utf-8"))
    manifest["canary"] = {
        "schema": canary.SCHEMA,
        "receipt": {
            "path": str(receipt_path),
            "sha256": digest(receipt_path),
        },
    }
    canary.check(manifest, broker)
    with out_path.open("xt", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--canary-id")
    parser.add_argument("--canary-receipt", type=Path)
    args = parser.parse_args()
    try:
        out = seal(
            args.packet,
            args.broker,
            ops=args.ops,
            canary_id=args.canary_id,
            canary_receipt=args.canary_receipt,
        )
    except (OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print(f"SEALED_NOT_DISPATCHED {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
