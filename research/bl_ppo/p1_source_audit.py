"""Audit saved P1 source finalisation evidence without running any science.

This command publishes a create-only mechanical finding. It cannot relabel a
pause snapshot, train, replay, score, allocate, or issue a source PASS receipt.
"""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_runtime import _require_pause, _validate_operational_proof
from research.bl_ppo.recovery import read_snapshot, snapshot_files


def reconcile(pack: Path) -> dict[str, Any]:
    """Preserve conservative frozen debits and add this verified failed attempt."""
    from tools.ops.spend import verify_receipts

    path = pack / "source-dispatch-debit.json"
    verify_export(path)
    prior = json.loads(path.read_text())
    source_receipt = pack / "source-output/receipt.json"
    paths = [Path(row["path"]) for row in prior["attempts"]] + [source_receipt]
    spends = verify_receipts(paths)
    for row in prior["attempts"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("frozen debit receipt hash differs")
    for item in paths:
        receipt = json.loads(item.read_text())
        if receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None:
            raise ValueError("debit receipt lacks verified release")
    source = json.loads(source_receipt.read_text())
    gross = Decimal(str(source["usage_before"]["balance"])) - Decimal(
        str(source["usage_after"]["balance"])
    )
    own = Decimal(str(spends[-1].units))
    charge = max(own, gross).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    previous = sum(Decimal(str(row["charged_units"])) for row in prior["attempts"])
    charged = previous + charge
    liability = charged + Decimal("2.00") + Decimal("9.62") + Decimal("3.86")
    return {
        "status": "HOLD_JUDGE_REQUIRED",
        "prior_debit_sha256": verify_export(path),
        "verified_receipts": len(spends),
        "prior_charged_units": float(previous),
        "source_own_units": float(own),
        "source_gross_units": float(gross),
        "source_charged_units": float(charge),
        "charged_units": float(charged),
        "remaining_units": float(Decimal("25.50") - charged),
        "new_canary_cap_units": 2.0,
        "source_retry_cap_units": 9.62,
        "replacement_reserved_units": 3.86,
        "all_in_liability_units": float(liability),
        "envelope_excess_units": float(max(Decimal(0), liability - Decimal("25.50"))),
        "warning": prior.get("warning"),
        "automatic_retry_authorized": False,
        "reason": "CHILD_NONZERO_EXIT after science is outside round-18 SUPERVISOR/transport retry",
        "receipt_sha256": {str(row.path): row.sha256 for row in spends},
    }


def audit(pack: Path, output: Path) -> dict[str, Any]:
    """Verify immutable exported state and report missing periodic proof."""
    pack = pack.resolve()
    source = pack / "source-output"
    receipt_path = source / "receipt.json"
    supervisor = json.loads(receipt_path.read_text())
    if (
        supervisor.get("status") != "INCOMPLETE"
        or supervisor.get("failure_class") != "CHILD_NONZERO_EXIT"
        or supervisor.get("stop_verified") is not True
        or supervisor.get("cleanup_error") is not None
    ):
        raise ValueError("audit requires the stopped finalisation-failure receipt")
    root = source / "exports"
    statuses = {}
    snapshots = {}
    evidence = {str(receipt_path): digest(receipt_path)}
    for phase, durability in (("origin", False), ("source-continuation", True)):
        path = root / phase / "training-status.json"
        evidence[str(path)] = verify_export(path)
        status = json.loads(path.read_text())
        status["_output"] = str(path.parent)
        snapshot = _require_pause(status, pause=0, durability=durability)
        for item in snapshot_files(snapshot):
            evidence[str(item)] = verify_export(item)
        statuses[phase], snapshots[phase] = status, snapshot
    continuation = statuses["source-continuation"]
    final_state = read_snapshot(snapshots["source-continuation"])
    control = {
        "resume": {
            "namespace": None,
            "allocation": None,
        }
    }
    operational = []
    for path in sorted(root.rglob("operational-*.pt")):
        for item in snapshot_files(path):
            evidence[str(item)] = verify_export(item)
        state = read_snapshot(path)
        metadata = state.get("operational", {})
        operational.append(
            {
                "path": str(path),
                "sha256": verify_export(path),
                "index": metadata.get("index"),
                "label": metadata.get("label"),
                "phase": metadata.get("execution", {}).get("phase"),
                "actor_transitions": state["counters"].get("actor_transitions"),
            }
        )
        control["resume"] = {key: metadata.get(key) for key in ("namespace", "allocation")}
    trace_meta = continuation.get("operational_trace")
    trace_anchor = None
    if trace_meta:
        path = root / "source-continuation" / trace_meta["path"]
        evidence[str(path)] = verify_export(path)
        if evidence[str(path)] != trace_meta["sha256"]:
            raise ValueError("operational trace differs from immutable training status")
        trace_anchor = json.loads(path.read_text()).get("snapshot")
    try:
        _validate_operational_proof(
            control,
            root / "source-continuation",
            continuation.get("operational_snapshots", []),
            trace_meta,
            final_state["counters"],
        )
        finding = None
    except ValueError as exc:
        finding = str(exc)
    value = {
        "schema": "bl-p1-source-finalisation-audit/v1",
        "status": "HOLD_MISSING_OPERATIONAL_PROOF" if finding else "PROOF_PRESENT_REVIEW_REQUIRED",
        "offline_complete_source_continuation": False,
        "finding": finding,
        "critic_qualified": continuation["critic_qualified"],
        "actor_transitions": final_state["counters"]["actor_transitions"],
        "origin_has_operational_label": read_snapshot(snapshots["origin"])
        .get("operational", {})
        .get("label")
        == "OPERATIONAL",
        "operational_exports": operational,
        "trace_anchor": trace_anchor,
        "evidence_sha256": evidence,
        "provider_calls": 0,
        "new_gameplay": False,
        "refit_or_gate_rerun": False,
        "source_receipt_written": False,
    }
    if (pack / "source-dispatch-debit.json").exists():
        value["debit"] = reconcile(pack)
    export_json(output, value)
    return value


def main() -> None:
    """Publish a new audit without modifying any original artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.pack, args.output)
    print(json.dumps({"status": value["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
