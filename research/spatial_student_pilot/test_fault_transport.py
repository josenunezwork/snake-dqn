"""One virtual outage + failed terminal export + real pilot full-state resume."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.colab_cli import supervisor
from research.spatial_student_pilot.collect import collect_batch, select_dataset
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.fit import fit_student
from research.spatial_student_pilot.resume import (
    UnitStore,
    build_ledger,
    empty_ledger,
    upload_inputs,
)
from research.spatial_student_pilot.test_batching import FixedPolicy, exact, networks
from tests import test_colab_cli_fault_properties as harness
from tests.test_colab_cli_supervisor import _manifest

_isolate_account = harness._isolate_account


def test_outage_terminal_export_failure_and_exact_pilot_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolate_account: object
) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    cfg = replace(PilotConfig().development("preflight"), bc_updates=2, improvement_updates=2)
    batch = collect_batch(
        cfg,
        range(6),
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
    )
    data = select_dataset(cfg, [batch])
    labels = [
        dict(r, target_action=r["b8_action"]) for r in data["roots"] if r["partition"] == "train"
    ]
    heldout = [r for r in data["roots"] if r["partition"] == "assessment"]
    kwargs = dict(device="cpu", progress=lambda _: None, heldout=heldout)
    reference = UnitStore(tmp_path / "reference", cfg, empty_ledger(cfg), tmp_path)
    fit_student(cfg, data["demos"], labels, store=reference, stop=lambda: False, **kwargs)
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    with pytest.raises(InterruptedError):
        fit_student(
            cfg, data["demos"], labels, store=first, stop=lambda: first.has("fit-0000002"), **kwargs
        )
    # A final export fails after both complete full-state snapshots were salvaged.
    first.publish("search-0000", {"fault_fixture_only": True})
    local_ledger = build_ledger(cfg, [first.output])
    binary = harness._fake_env(monkeypatch, tmp_path, {"schedule": []})
    monkeypatch.setenv("FAKE_COLAB_MODE", "success")
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 60

    def observer(event: str, context: dict) -> None:
        if event != "ready":
            return
        cli = context["cli"]
        remote = tmp_path / "remote" / cli.session / context["root"].lstrip("/") / "output"
        shutil.copytree(first.output, remote / "spatial")
        inventory = [
            dict(path="spatial/" + e["path"], size=e["size"], sha256=e["sha256"])
            for e in local_ledger["units"]
        ]
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        state["runtime"][cli.session]["terminal"]["exports"] = inventory
        state_path.write_text(json.dumps(state))
        damaged = remote / inventory[-1]["path"]
        content = bytearray(damaged.read_bytes())
        content[-1] ^= 1
        damaged.write_bytes(content)
        monkeypatch.setenv(
            "FAKE_COLAB_FAULT_PLAN",
            json.dumps(
                {
                    "seed": 29,
                    "max_latency_seconds": 120,
                    "schedule": [
                        dict(call="exec", contains="read_state", fault="timeout", phase="pre")
                    ]
                    * 4,
                }
            ),
        )

    # Retain real export verification; force separate finite files to expose salvage.
    def collect(cli: object, root: str, inventory: list, output: Path, limit: int) -> None:
        for entry in inventory:
            supervisor.export_file(cli, root, entry, output, limit)
            cli.verified_exports[entry["path"]] = dict(entry)

    monkeypatch.setattr(supervisor, "collect_exports", collect)
    with pytest.raises(supervisor.Refused, match="download hash mismatch"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    receipt_path = tmp_path / "job-output/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert receipt["stop_verified"] and receipt["cleanup_error"] is None
    assert receipt["status"] == "INCOMPLETE" and receipt["elapsed_seconds"] > 300
    assert state["allocation_count"] == 1 and list(state["job_launches"].values()) == [1]
    assert state["sessions"] == {}
    exports = tmp_path / "job-output/exports/spatial"
    ledger = build_ledger(cfg, [exports], [receipt_path])
    assert {e["unit_id"] for e in ledger["units"]} == {"fit-0000001", "fit-0000002"}
    inputs = tmp_path / "resume-inputs"
    for binding in upload_inputs(ledger, [exports], cfg):
        target = inputs / binding["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binding["path"], target)
    resumed = UnitStore(tmp_path / "resumed", cfg, ledger, inputs)
    with pytest.raises(ValueError, match="recomputed"):
        resumed.publish("fit-0000002", {})
    fit_student(cfg, data["demos"], labels, store=resumed, stop=lambda: False, **kwargs)
    exact(resumed.get("fit-0000004"), reference.get("fit-0000004"))
    assert time.monotonic() - started < 60
