"""Stage explicitly authorized training inputs after metadata-only Codex-V admission."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_round30.admission import freeze_inventory, require_codex_v_approval
from research.bl_round30.protocol import AMENDMENT_SHA, B8_SHA, JUDGMENT_SHA, NAMESPACE, SALT


def prepare(
    output: Path,
    inventory_path: Path,
    approval_path: Path,
    partition_manifest: Path,
    coverage_approval: Path,
    life_id_map: Path,
    train: Path,
    b8: Path,
    commit: str,
    *,
    stage: str = "r30-1",
    gate: Path | None = None,
) -> dict:
    """No data checksums/copy until provenance and independent approval pass."""
    inventory = json.loads(inventory_path.read_text())
    admitted = freeze_inventory(inventory)
    require_codex_v_approval(admitted, json.loads(approval_path.read_text()))
    if inventory["split_salt"] != SALT:
        raise ValueError("no split redraw")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("full published source pin required")
    manifest = json.loads(partition_manifest.read_text())
    coverage = json.loads(coverage_approval.read_text())
    manifest_sha = digest(partition_manifest)
    if (
        manifest.get("schema") != "r30-partition-manifest/v1"
        or manifest.get("inventory_sha256") != admitted.inventory_sha256
        or manifest.get("allocation_sha256") != admitted.allocation_sha256
        or manifest.get("life_id_map_sha256") != digest(life_id_map)
        or manifest.get("partitions", {}).get("train", {}).get("sha256") != digest(train)
    ):
        raise ValueError("training bytes/identity map are outside the frozen materialization")
    if (
        coverage.get("schema") != "r30-partition-coverage-approval/v1"
        or coverage.get("approved") is not True
        or coverage.get("reviewer_role") != "Codex-V"
        or coverage.get("partition_manifest_sha256") != manifest_sha
        or coverage.get("inventory_sha256") != admitted.inventory_sha256
        or coverage.get("allocation_sha256") != admitted.allocation_sha256
        or coverage.get("life_id_map_sha256") != digest(life_id_map)
    ):
        raise ValueError("independent partition coverage approval is absent or stale")
    if digest(b8) != B8_SHA:
        raise ValueError("B8 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "inventory.json", inventory)
    export_json(output / "approval.json", json.loads(approval_path.read_text()))
    export_json(output / "partition-manifest.json", manifest)
    export_json(output / "coverage-approval.json", coverage)
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(Path(str(life_id_map) + suffix), output / ("life-id-map.json" + suffix))
    shutil.copyfile(train, output / "train.npz")
    shutil.copyfile(b8, output / "b8.pth")
    hashes = {
        name: digest(output / name)
        for name in (
            "train.npz",
            "b8.pth",
            "partition-manifest.json",
            "coverage-approval.json",
            "life-id-map.json",
        )
    }
    control = {
        "schema": "r30-control/v2",
        "synthetic": False,
        "stage": stage,
        "namespace": NAMESPACE,
        "commit": commit,
        "ruling_sha256": JUDGMENT_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "allocation": admitted.allocation_sha256,
        "salt": SALT,
        "inventory_sha256": digest(output / "inventory.json"),
        "approval_sha256": digest(output / "approval.json"),
        "partition_manifest_sha256": hashes["partition-manifest.json"],
        "coverage_approval_sha256": hashes["coverage-approval.json"],
        "life_id_map_sha256": hashes["life-id-map.json"],
        "horizon": 300,
        "mass_convention": "full-horizon-dead-zero",
        "continuation_sha256": inventory["continuation_sha256"],
        "input_hashes": hashes,
    }
    if stage == "r30-2":
        if gate is None:
            raise ValueError("HOLD: F4 closure/optimizer/conversion/archive gate required")
        from research.bl_round30.protocol import require_successor_gate

        value = json.loads(gate.read_text())
        diagnosis_path = gate.parent / "diagnosis.json"
        from research.bl_round30.protocol import bound_json

        diagnosis = bound_json(diagnosis_path, value["diagnosis_sha256"])
        optimizer = require_successor_gate(value, admitted.allocation_sha256, diagnosis)
        for suffix in ("", ".sha256", ".complete"):
            shutil.copyfile(
                Path(str(diagnosis_path) + suffix), output / ("diagnosis.json" + suffix)
            )
        hashes["diagnosis.json"] = digest(output / "diagnosis.json")
        export_json(output / "successor-gate.json", value)
        control["successor_gate_sha256"] = digest(output / "successor-gate.json")
        control["diagnosis_sha256"] = value["diagnosis_sha256"]
        control["conversion_control"] = {
            "name": value["control_name"],
            "procedure": optimizer,
            "endpoint_sha256": value["control_endpoint_sha256"],
        }
    export_json(output / "control.json", control)
    export_json(
        output / "verified-ledger.json",
        {
            "schema": "r30-job-ledger/v1",
            "namespace": NAMESPACE,
            "allocation": digest(output / "control.json"),
            "units": [],
        },
    )
    from research.bl_round30.job import preflight_inputs

    return preflight_inputs(output, mode=stage, commit=commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "output",
        "inventory",
        "approval",
        "partition-manifest",
        "coverage-approval",
        "life-id-map",
        "train",
        "b8",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", choices=("r30-1", "r30-2"), default="r30-1")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    value = prepare(
        args.output,
        args.inventory,
        args.approval,
        args.partition_manifest,
        args.coverage_approval,
        args.life_id_map,
        args.train,
        args.b8,
        args.commit,
        stage=args.stage,
        gate=args.gate,
    )
    print(json.dumps({"allocation": value["allocation"], "stage": args.stage}))


if __name__ == "__main__":
    main()
