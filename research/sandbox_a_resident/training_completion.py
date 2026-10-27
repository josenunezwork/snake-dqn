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
    successor_template: Path | None = None,
) -> None:
    """Seal a successor without allocating, selecting weights or reading outcomes."""
    paths = (original, receipt, exports, packet, output)
    if successor_template is not None:
        paths += (successor_template,)
    for path in paths:
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
    template = original if successor_template is None else successor_template
    successor_control = json.loads((template / "control.json").read_text())
    successor_manifest = json.loads((template / "dispatch-manifest.json").read_text())
    if (
        successor_manifest.get("commit") != successor_control.get("source_commit")
        or json.loads(successor_manifest.get("env", {}).get("SANDBOX_A_CONTROL_JSON", "null"))
        != successor_control
    ):
        raise ValueError("successor template control/source pin differs from its manifest")
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
        successor_manifest=successor_manifest,
        successor_control=successor_control,
    )
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", successor_control)
    write_once(
        packet / "registry-check.json", json.loads((template / "registry-check.json").read_text())
    )
    broker = result.manifest
    write_once(packet / "verified-ledger.json", result.ledger)
    broker["inputs"].append(
        {
            "path": str(packet / "verified-ledger.json"),
            "dest": "verified-ledger.json",
            "sha256": digest(packet / "verified-ledger.json"),
        }
    )
    broker["sandbox_admission"]["manifest_path"] = str(packet / "sandbox.json")
    write_once(packet / "dispatch-manifest.json", broker)
    sandbox = copy.deepcopy(json.loads((template / "sandbox.json").read_text()))
    old_kinds = {row["sha256"]: row["kind"] for row in sandbox["inputs"] if "sha256" in row}
    worlds = [row for row in sandbox["inputs"] if row["kind"] == "worlds"]
    sandbox["inputs"] = worlds + [
        {
            "kind": (
                "training-export-sidecar"
                if row["dest"].startswith("resume/")
                and row["dest"].endswith((".sha256", ".complete"))
                else (
                    "training-export"
                    if row["dest"].startswith("resume/")
                    else (
                        "verified-ledger"
                        if row["dest"] == "verified-ledger.json"
                        else old_kinds[row["sha256"]]
                    )
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
    parser.add_argument(
        "--successor-template",
        type=Path,
        help="sealed target packet whose control/source pin is the new successor",
    )
    args = parser.parse_args()
    prepare_completion(
        args.original.resolve(),
        args.receipt.resolve(),
        args.exports.resolve(),
        args.packet.resolve(),
        index=args.index,
        job_id=args.job_id,
        output=args.output.resolve(),
        successor_template=(
            None if args.successor_template is None else args.successor_template.resolve()
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
