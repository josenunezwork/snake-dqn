"""Review-17: real dispatch_preferred files -> C/T writer -> c1/c2 admission.

Only provider/control boundaries are stubbed. Variant reservation, preference
publication, accounting refresh and GPU receipt publication use real producers.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from research.compute.gpu_preference import prepare_preference
from research.sandbox_a_resident.ct_dispatch_binding import resolve_dispatch_binding
from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger
from research.sandbox_a_resident.ct_resume import CTResumeRefused, write_continuation_packet
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_units import allocation, publish_stage_snapshot
from research.sandbox_a_resident.test_ct_runner_and_resume import (
    _clean_supervisor_log,
    _rebind_forged_packet,
    _save_tiny_checkpoints,
    _write_tiny_initial_packet,
)
from tools.ops import dispatch
from tools.ops.gpu_receipt import publish
from tools.sandbox.registry import digest


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stopped(manifest_path: Path, index: int, *, h100: bool) -> dict[str, Any]:
    value = json.loads(manifest_path.read_text())
    value.setdefault("lease_seconds", 180)
    output = Path(value["output_dir"])
    write(output / "manifest.json", value)
    hardware = {"ready": True, "hardware": {"kind": value["requirements"]["gpu"]}}
    write(output / "hardware.json", hardware)
    receipt = {
        "job_id": value["job_id"],
        "commit": value["commit"],
        "manifest_sha256": digest(output / "manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "cleanup_error": None,
        "generation": f"generation-{index}",
        "endpoint": "test-endpoint",
        "terminal": {
            "job_id": value["job_id"],
            "generation": f"generation-{index}",
            "status": "failed",
            "reason": "timeout",
            "returncode": 138,
        },
    }
    write(output / "receipt.json", receipt)
    write(
        output / "poll-000008.json",
        {
            "snapshot": {
                "job_log_tail": "2.9.1+cu126 12.6 (9, 0)\n" if h100 else "2.9.1+cu126 12.6 (8, 0)\n"
            }
        },
    )
    return receipt


def dispatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool) -> dict:
    checkpoints = _save_tiny_checkpoints(tmp_path)
    ctrl = CTControl(source_commit="a" * 40, worlds=6, collection_ticks=12, tiny=True)
    initial = tmp_path / "initial"
    _write_tiny_initial_packet(initial, ctrl, checkpoints, tmp_path / "attempt-0")
    canonical = json.loads((initial / "dispatch-manifest.json").read_text())
    prepare_preference(canonical)
    _rebind_forged_packet(initial, canonical)
    seal = json.loads((initial / "SEAL.json").read_text())
    seal["sandbox_sha256"] = digest(initial / "sandbox.json")
    write(initial / "SEAL.json", seal)
    ops = tmp_path / "ops"
    monkeypatch.setattr(dispatch, "validate", lambda *a, **kw: None)
    monkeypatch.setattr(dispatch, "qualified", lambda *a, **kw: None)
    monkeypatch.setattr(
        dispatch,
        "command",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, stdout="[?] other-endpoint | Hardware: L4 | Shape: Standard | Variant: GPU\n"
        ),
    )
    # Isolate accounting's local broker inventory from the real account as well.
    home = tmp_path / "home"
    home.mkdir()
    write(
        home / ".config/colab-cli/snake/active-other.json",
        {
            "session": "other",
            "endpoint": "other-endpoint",
            "rate_units_per_hour": 1.54,
        },
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("OPS_ALLOC_POLL_INTERVAL_SECONDS", ".01")

    def start(broker, cwd, state, manifest, log, name, deadline, *, gpu_receipt):
        selected = json.loads(manifest.read_text())
        output = Path(selected["output_dir"])
        if fallback and selected["requirements"]["gpu"] == "H100":
            write(
                output / "receipt.json",
                {"job_id": selected["job_id"], "commit": selected["commit"], "endpoint": None},
            )
            log.write_text("EXIT=1\n")
        else:
            stopped(manifest, 1, h100=not fallback)
            publish(manifest, gpu_receipt)
            log.write_text("qualified transport stop observed\nEXIT=1\n")

    def reconcile(broker, manifest, state, log, proof, deadline):
        output = Path(json.loads(manifest.read_text())["output_dir"])
        write(
            proof,
            {
                "schema": "colab-allocation-unavailable/v1",
                "classification": "ALLOCATION_UNAVAILABLE",
                "mode": "LIVE",
                "endpoint_ever_assigned": False,
                "vm_compute_started": False,
                "recovery": {"endpoint": None},
                "receipt": {
                    "path": str(output / "receipt.json"),
                    "sha256": digest(output / "receipt.json"),
                },
            },
        )

    monkeypatch.setattr(dispatch, "start", start)
    monkeypatch.setattr(dispatch, "wait_allocation", lambda *a: "failed")
    monkeypatch.setattr(dispatch, "allocation_unavailable", lambda *a: fallback)
    monkeypatch.setattr(dispatch, "reconcile_unavailable", reconcile)
    args = argparse.Namespace(name="lane", state_dir=tmp_path / "state", dry_run=False)
    dispatch.dispatch_preferred(args, canonical, ops, tmp_path, tmp_path, args.state_dir)
    name = "lane-a100-fallback" if fallback else "lane"
    output = Path(canonical["output_dir"] + ("-a100-fallback" if fallback else ""))
    receipt = json.loads((output / "receipt.json").read_text())
    evidence = {
        "preference": ops / "dispatch/lane/gpu-preference.json",
        "accounting": ops / f"dispatch/{name}/{name}.accounting.json",
        "gpu_receipt": ops / f"dispatch/{name}/gpu-receipt.json",
    }
    # This is the real mismatch: sealed packet, accounting/live, and normalized
    # output manifests all have distinct hashes; no packet fixture substitutes it.
    assert digest(initial / "dispatch-manifest.json") != receipt["manifest_sha256"]
    assert digest(ops / f"manifests/{name}.live.json") != receipt["manifest_sha256"]
    return dict(
        initial=initial,
        canonical=canonical,
        control=ctrl.to_dict(),
        receipt=receipt,
        evidence=evidence,
        output=output,
        log=ops / f"logs/{name}.log",
    )


def stage(exports: Path, index: int, control: dict) -> None:
    publish_stage_snapshot(
        exports,
        index,
        lambda dst: torch.save({"kind": "collection", "stage": index}, dst),
        metadata={"allocation": allocation(control), "source_commit": control["source_commit"]},
    )


@pytest.mark.parametrize("fallback", [False, True])
def test_actual_dispatch_files_admit_two_manual_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    f = dispatched(tmp_path, monkeypatch, fallback)
    prior, roots = f["initial"], []
    for index in (1, 2):
        if index == 1:
            receipt, output, log = f["receipt"], f["output"], f["log"]
            evidence = f["evidence"]
        else:
            manifest_path = prior / "dispatch-manifest.json"
            output = Path(json.loads(manifest_path.read_text())["output_dir"])
            # Exercise an accounting/live binding on c1 as well for H100. For
            # fixed A100, the original exact-hash path remains supported.
            if fallback:
                # Real ordinary-dispatch accounting producer; no preference/GPU
                # sidecar exists for this explicit A100 continuation.
                source = tmp_path / "ops-c1/manifests/lane-c1.json"
                write(source, json.loads(manifest_path.read_text()))
                accounting = tmp_path / "ops-c1/dispatch/lane-c1/lane-c1.accounting.json"
                accounting.parent.mkdir(parents=True)
                live = dispatch.refresh_assignments(source, accounting, float("inf"))
                receipt = stopped(live, index, h100=False)
                evidence = {"accounting": accounting}
            else:
                ops = tmp_path / "ops-c1"
                value = json.loads(manifest_path.read_text())
                args = argparse.Namespace(
                    name="lane-c1", state_dir=tmp_path / "state-c1", dry_run=False
                )
                dispatch.dispatch_preferred(args, value, ops, tmp_path, tmp_path, args.state_dir)
                receipt = json.loads((output / "receipt.json").read_text())
                evidence = {
                    "preference": ops / "dispatch/lane-c1/gpu-preference.json",
                    "accounting": ops / "dispatch/lane-c1/lane-c1.accounting.json",
                    "gpu_receipt": ops / "dispatch/lane-c1/gpu-receipt.json",
                }
            log = _clean_supervisor_log(tmp_path, "second.log")
        exports = output / "exports/ct"
        stage(exports, index, f["control"])
        roots.append(exports)
        successor = tmp_path / f"c{index}"
        result = write_continuation_packet(
            successor,
            prior_packet_dir=prior,
            terminal_receipt=receipt,
            export_root=roots,
            completion_index=index,
            job_id=f"tiny-c{index}",
            output_dir=tmp_path / f"attempt-{index}",
            supervisor_log_path=log,
            use_compact_xz_for_stages=None,
            validate_registry_inputs=False,
            dispatch_evidence=evidence,
        )
        assert result["verified_stages"] == list(range(1, index + 1))
        value = json.loads((successor / "dispatch-manifest.json").read_text())
        assert value["commit"] == f["control"]["source_commit"]
        assert value["requirements"]["gpu"] == ("A100" if fallback else "H100")
        if fallback:
            assert "prefer" not in value["requirements"]
        assert value["completion_attempt"]["prior_manifest_sha256"] == digest(
            prior / "dispatch-manifest.json"
        )
        validate_uploaded_ledger(
            successor, copy.deepcopy(value), f["control"], require_committed=False
        )
        if evidence:
            # Revalidation of evidence outside output roots is mandatory too.
            accounting = evidence["accounting"]
            raw = accounting.read_bytes()
            accounting.write_bytes(raw + b" ")
            with pytest.raises(ValueError, match="sealed dispatch evidence changed"):
                validate_uploaded_ledger(
                    successor, copy.deepcopy(value), f["control"], require_committed=False
                )
            accounting.write_bytes(raw)
        prior = successor


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-capability",
        "foreign-job",
        "live-env",
        "live-rate",
        "live-input",
        "live-lease",
        "executed-env",
        "executed-lease",
        "accounting-sha",
        "variant-sha",
        "hardware",
        "gpu-receipt-sha",
        "seal",
        "missing-hardware-or-reconciliation",
        "scientific-fault",
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_dispatch_binding_fault_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, fault: str
) -> None:
    f = dispatched(tmp_path, monkeypatch, fallback)
    exports = f["output"] / "exports/ct"
    stage(exports, 1, f["control"])
    if fault == "foreign-capability":
        write(
            f["output"] / "poll-000008.json",
            {"snapshot": {"job_log_tail": "2.9.1+cu126 12.6 (8, 9)\n"}},
        )
    elif fault == "foreign-job":
        f["receipt"]["job_id"] = "foreign-job"
    elif fault == "scientific-fault":
        write(exports / "fault.json", {"scientific_fault": True})
    elif fault == "missing-hardware-or-reconciliation":
        if fallback:
            (f["evidence"]["preference"].parent / "unavailable.json").unlink()
        else:
            (f["output"] / "hardware.json").unlink()
    else:
        accounting = json.loads(f["evidence"]["accounting"].read_text())
        if fault.startswith("live-"):
            path = Path(accounting["manifest"])
            value = json.loads(path.read_text())
            if fault == "live-env":
                value["env"]["SNAKE_DQN_DEVICE"] = "cpu"
            if fault == "live-rate":
                value["budget"]["rate_units_per_hour"] = 1
            if fault == "live-input":
                value["inputs"][0]["sha256"] = "b" * 64
            if fault == "live-lease":
                value["lease_seconds"] = 600
        elif fault.startswith("executed-"):
            path = f["output"] / "manifest.json"
            value = json.loads(path.read_text())
            if fault == "executed-env":
                value["env"]["SNAKE_DQN_DEVICE"] = "cpu"
            else:
                value["lease_seconds"] = 600
        elif fault == "accounting-sha":
            path, value = f["evidence"]["accounting"], accounting
            value["source_manifest_sha256"] = "b" * 64
        elif fault == "variant-sha":
            path = f["evidence"]["preference"]
            value = json.loads(path.read_text())
            value["variants"][0]["sha256"] = "b" * 64
        elif fault == "hardware":
            path = f["output"] / "hardware.json"
            value = {"hardware": {"kind": "L4"}}
        elif fault == "gpu-receipt-sha":
            path = f["evidence"]["gpu_receipt"]
            value = json.loads(path.read_text())
            value["supervisor_receipt"]["sha256"] = "b" * 64
        else:
            path = f["initial"] / "SEAL.json"
            value = json.loads(path.read_text())
            value["manifest_sha256"] = "b" * 64
        write(path, value)
    reasons = {
        "foreign-capability": "unrecognized job_log_tail line",
        "foreign-job": "foreign job or ambiguous variant identity",
        "accounting-sha": "accounting source/live manifest hash mismatch",
        "variant-sha": "variant differs from pure sealed derivation",
        "hardware": "execution or observed hardware identity differs",
        "gpu-receipt-sha": "GPU receipt supervisor_receipt hash/path differs",
        "seal": "SEAL.json does not bind canonical dispatch-manifest.json",
        "missing-hardware-or-reconciliation": "unreadable",
        "scientific-fault": "scientific",
    }
    reason = reasons.get(fault)
    if fault.startswith("live-"):
        reason = "accounting source/live manifest hash mismatch"
    elif fault.startswith("executed-"):
        reason = "retained manifest changes fields beyond supervisor lease default"
    with pytest.raises((ValueError, FileNotFoundError), match=reason):
        write_continuation_packet(
            tmp_path / "refused",
            prior_packet_dir=f["initial"],
            terminal_receipt=f["receipt"],
            export_root=exports,
            completion_index=1,
            job_id="tiny-c1",
            output_dir=tmp_path / "c1-out",
            supervisor_log_path=f["log"],
            use_compact_xz_for_stages=None,
            validate_registry_inputs=False,
            dispatch_evidence=f["evidence"],
        )
    assert not (tmp_path / "refused").exists()


def test_changed_live_manifest_without_explicit_binding_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = dispatched(tmp_path, monkeypatch, False)
    with pytest.raises(CTResumeRefused, match="verified released incomplete"):
        from research.sandbox_a_resident.ct_resume import verify_mechanical_stop

        verify_mechanical_stop(
            f["receipt"],
            f["canonical"],
            digest(f["initial"] / "dispatch-manifest.json"),
            export_root=f["output"],
        )
    # A bound refreshed accounting map may differ, but every other field is frozen.
    execution, _, proof = resolve_dispatch_binding(
        f["initial"], f["canonical"], f["receipt"], paths=f["evidence"]
    )
    assert execution["lease_seconds"] == 180
    assert execution["budget"]["other_assignments"] == {"other-endpoint": 1.54}
    assert proof["files"]["packet_seal"]["sha256"] == digest(f["initial"] / "SEAL.json")


@pytest.mark.parametrize("surface", ["live", "executed"])
@pytest.mark.parametrize("fallback", [False, True])
def test_coherently_rehashed_foreign_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, fallback: bool
) -> None:
    f = dispatched(tmp_path, monkeypatch, fallback)
    accounting_path = f["evidence"]["accounting"]
    accounting = json.loads(accounting_path.read_text())
    live_path = Path(accounting["manifest"])
    output = f["output"]
    if surface == "live":
        live = json.loads(live_path.read_text())
        live["env"]["SNAKE_DQN_DEVICE"] = "cpu"
        write(live_path, live)
        accounting["manifest_sha256"] = digest(live_path)
        write(accounting_path, accounting)
        live.setdefault("lease_seconds", 180)
        write(output / "manifest.json", live)
    else:
        executed = json.loads((output / "manifest.json").read_text())
        executed["lease_seconds"] = 600
        write(output / "manifest.json", executed)
    f["receipt"]["manifest_sha256"] = digest(output / "manifest.json")
    write(output / "receipt.json", f["receipt"])
    f["evidence"]["gpu_receipt"].unlink()
    publish(live_path, f["evidence"]["gpu_receipt"])
    with pytest.raises(CTResumeRefused, match="changes fields beyond"):
        resolve_dispatch_binding(f["initial"], f["canonical"], f["receipt"], paths=f["evidence"])


@pytest.mark.parametrize("fault", ["endpoint", "compute", "receipt-sha", "dry-run"])
def test_fallback_reconciliation_fault_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    f = dispatched(tmp_path, monkeypatch, True)
    proof_path = f["evidence"]["preference"].parent / "unavailable.json"
    proof = json.loads(proof_path.read_text())
    if fault == "endpoint":
        path = Path(proof["receipt"]["path"])
        receipt = json.loads(path.read_text())
        receipt["endpoint"] = "assigned-endpoint"
        write(path, receipt)
        proof["receipt"]["sha256"] = digest(path)
    elif fault == "compute":
        proof["vm_compute_started"] = True
    elif fault == "receipt-sha":
        proof["receipt"]["sha256"] = "b" * 64
    else:
        proof["mode"] = "DRY_RUN"
    write(proof_path, proof)
    with pytest.raises(CTResumeRefused, match="fallback lacks bound pre-endpoint reconciliation"):
        resolve_dispatch_binding(f["initial"], f["canonical"], f["receipt"], paths=f["evidence"])
