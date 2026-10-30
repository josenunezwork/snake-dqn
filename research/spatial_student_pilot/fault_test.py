"""Offline outage/export/verified-ledger-resume qualification, no gameplay."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

from research.spatial_student_pilot import resume
from research.spatial_student_pilot.config import PilotConfig


def run_fault_test(root: Path) -> dict:
    """Interrupt completion-last publication and transport only verified exports."""
    started = time.monotonic()
    cfg = PilotConfig().development("preflight")
    first = resume.UnitStore(root / "first", cfg, resume.empty_ledger(cfg), root)
    payload = {"fixture_only": True, "state": torch.arange(8)}
    path = first.publish("collect-0000", payload)
    original = path.read_bytes()
    publish = resume._publish_bytes

    def outage(destination: Path, content: bytes) -> None:
        if str(destination).endswith(".complete"):
            raise InterruptedError("offline injected outage before completion marker")
        publish(destination, content)

    with patch.object(resume, "_publish_bytes", side_effect=outage):
        try:
            first.publish("collect-0001", payload)
        except InterruptedError:
            pass
        else:
            raise AssertionError("outage was not injected")
    # The ledger builder must remain outcome-blind even for this fault fixture.
    with patch("torch.load", side_effect=AssertionError("ledger decoded an outcome")):
        ledger = resume.build_ledger(cfg, [first.output])
    assert [e["unit_id"] for e in ledger["units"]] == ["collect-0000"]
    export = root / "export"
    for binding in resume.upload_inputs(ledger, [first.output], cfg):
        dest = export / binding["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binding["path"], dest)
    resumed = resume.UnitStore(root / "resumed", cfg, ledger, export)
    assert resumed.has("collect-0000") and not resumed.has("collect-0001")
    assert torch.equal(resumed.get("collect-0000")["state"], payload["state"])
    assert (export / ledger["units"][0]["path"]).read_bytes() == original
    try:
        resumed.publish("collect-0000", payload)
    except ValueError:
        pass
    else:
        raise AssertionError("verified unit recomputed")
    resumed.publish("collect-0001", payload)
    for key, value in (
        ("namespace", "foreign"),
        ("allocation_index", 1),
        ("config_sha256", "0" * 64),
    ):
        foreign = copy.deepcopy(ledger)
        foreign[key] = value
        try:
            resume.validate_ledger(foreign, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("foreign resume identity admitted")
    exported = export / ledger["units"][0]["path"]
    damaged = bytearray(exported.read_bytes())
    damaged[-1] ^= 1
    exported.write_bytes(damaged)
    try:
        resume.build_ledger(cfg, [export])
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt exported payload admitted")
    seconds = time.monotonic() - started
    assert seconds < 60, "offline fault-test exceeded its registered bound"
    return dict(
        schema="spatial-offline-fault-test/v1",
        status="PASS",
        scientific_data=False,
        interrupted_publication_excluded=True,
        ledger_outcome_blind=True,
        verified_export_bytes_preserved=True,
        verified_unit_recompute_refused=True,
        missing_unit_completed_in_fresh_attempt=True,
        foreign_identity_refused=True,
        corrupt_export_refused=True,
        elapsed_seconds=seconds,
    )


def main() -> None:
    """Retain a create-only qualification receipt outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="spatial-fault-") as scratch:
        report = run_fault_test(Path(scratch))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
