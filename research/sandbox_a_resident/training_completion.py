"""Create-only manual TRAIN-01 completion packet after verified mechanical stop."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.sandbox_a_resident.training_prepare import write_once
from research.sandbox_a_resident.training_resume import completion_packet
from tools.sandbox.registry import digest, safe_path


def prepare_completion(
    original: Path,
    receipt: Path,
    exports: Path,
    packet: Path,
    *,
    index: int,
    job_id: str,
    output: Path,
) -> None:
    """Seal a successor without allocating, selecting weights or reading outcomes."""
    for path in (original, receipt, exports, packet, output):
        safe_path(path)
    manifest_path = original / "dispatch-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    executed_path = receipt.parent / "manifest.json"
    executed = json.loads(executed_path.read_text())
    for key in (
        "job_id",
        "commit",
        "script",
        "env",
        "source_paths",
        "timeout_minutes",
        "requirements",
        "inputs",
        "resume",
    ):
        if executed.get(key) != manifest.get(key):
            raise ValueError(f"executed manifest differs from frozen workload: {key}")
    control = json.loads((original / "control.json").read_text())
    previous = manifest.get("completion_attempt", {}).get("index", 0)
    if index != previous + 1:
        raise ValueError("completion attempts must advance exactly once, at most two")
    result = completion_packet(
        executed,
        json.loads(receipt.read_text()),
        exports,
        control,
        completion_index=index,
        job_id=job_id,
        output_dir=output,
        original_manifest_sha256=digest(executed_path),
    )
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", control)
    write_once(
        packet / "registry-check.json", json.loads((original / "registry-check.json").read_text())
    )
    broker = result.manifest
    # Only the newest eligible full trajectory is required for continuation.
    # Older completed units are covered by that trajectory's iteration counter.
    latest_unit = result.ledger["units"][-1]
    result.ledger["units"] = [latest_unit]
    write_once(packet / "verified-ledger.json", result.ledger)
    destinations = {row["path"] for row in latest_unit["exports"]}
    destinations |= {
        name + suffix for name in list(destinations) for suffix in (".sha256", ".complete")
    }
    broker["inputs"] = [
        row
        for row in manifest["inputs"]
        if row["dest"] != "verified-ledger.json" and not row["dest"].startswith("resume/")
    ] + [row for row in result.input_records if row["dest"] in destinations]
    broker["inputs"].append(
        {
            "path": str(packet / "verified-ledger.json"),
            "dest": "verified-ledger.json",
            "sha256": digest(packet / "verified-ledger.json"),
        }
    )
    broker["sandbox_admission"]["manifest_path"] = str(packet / "sandbox.json")
    write_once(packet / "dispatch-manifest.json", broker)
    sandbox = copy.deepcopy(json.loads((original / "sandbox.json").read_text()))
    old_kinds = {row["sha256"]: row["kind"] for row in sandbox["inputs"] if "sha256" in row}
    worlds = [row for row in sandbox["inputs"] if row["kind"] == "worlds"]
    sandbox["inputs"] = worlds + [
        {
            "kind": (
                "training-export"
                if row["dest"].startswith("resume/")
                else (
                    "verified-ledger"
                    if row["dest"] == "verified-ledger.json"
                    else old_kinds[row["sha256"]]
                )
            ),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in broker["inputs"]
    ]
    sandbox["outputs"] = {
        "label": "simulator-D",
        "paths": [str(output)],
        "broker_manifest": {
            "path": str(packet / "dispatch-manifest.json"),
            "sha256": digest(packet / "dispatch-manifest.json"),
        },
    }
    with (packet / "sandbox.json").open("x") as stream:
        stream.write(
            "\n".join(
                ("{" if i == 0 else " ")
                + json.dumps(key)
                + ": "
                + json.dumps(value)
                + ("," if i < 9 else "}")
                for i, (key, value) in enumerate(sandbox.items())
            )
            + "\n"
        )
    print(
        f"PREPARED_MANUAL_COMPLETION index={index} "
        f"iteration={result.latest_iteration} allocations=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "receipt", "exports", "packet", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    prepare_completion(
        args.original.resolve(),
        args.receipt.resolve(),
        args.exports.resolve(),
        args.packet.resolve(),
        index=args.index,
        job_id=args.job_id,
        output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
