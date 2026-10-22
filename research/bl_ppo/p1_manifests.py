"""Prepare WP-3 catalogs, forecasts and supervisor drafts without provider calls."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_ppo.manifests import ROOT, _base, _bounded_source_paths, input_row
from research.bl_ppo.p1_contract import SHAPE
from research.bl_ppo.p1_inputs import validate_catalog

ARTIFACTS = ROOT.parent / "snake-dqn-artifacts/big-lever/exp/bl-p1"
TARGET = SHAPE.target
QUAL_PAUSE = SHAPE.qualification_pause
E4_RATE = 1440.0
RATE = 2.0
RESERVE = 24.0
PHASES = ("y3-source", "y3-replacement", "qual-1m", "tranche", "eval")


def complete_json(path: Path) -> dict[str, Any]:
    """Read a finite, create-only result with both supervisor sidecars."""
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("P1 evidence must be an object")
    return value


def provisional_plan() -> dict[str, Any]:
    """Price fixed boundaries from E4; this is expressly not admission evidence."""
    boundaries = [20_000_000, 40_000_000, 60_000_000, 80_000_000, TARGET]
    start = QUAL_PAUSE
    chunks = []
    for index, pause in enumerate(boundaries, 1):
        estimate = math.ceil((pause - start) / E4_RATE / 60 * 1.10 + 12)
        chunks.append(
            {
                "chunk": index,
                "start": start,
                "target": TARGET,
                "pause_at": pause,
                "actor_transitions": pause - start,
                "est_minutes": estimate,
                "timeout_minutes": 2 * estimate,
                "unit_cap": 24.0,
                "reserve_units": 4.8,
            }
        )
        start = pause
    nominal_hours = sum(c["est_minutes"] for c in chunks) / 60
    return {
        "schema": "bl-p1-tranche-plan/v1",
        "status": "FORECAST_ONLY_NOT_ADMITTED",
        "source": "round-11 E4 active collection ~1440 hero transitions/s",
        "collection_only_hours_100m": TARGET / E4_RATE / 3600,
        "provisional_overhead_multiplier": 1.10,
        "provisional_per_job_overhead_minutes": 12,
        "nominal_hours_remaining": nominal_hours,
        "two_x_liability_units": 2 * nominal_hours * RATE,
        "reserve_units": RESERVE,
        "invoice_units": 2 * nominal_hours * RATE + RESERVE,
        "unit_cap": 120,
        "one_l4_at_a_time": True,
        "complete_job_rate_required": True,
        "chunks": chunks,
    }


def measured_plan(qualification: dict[str, Any]) -> dict[str, Any]:
    """Replace provisional rates with the complete-job qualification invoice."""
    if qualification.get("schema") != "bl-p1-complete-job-qualification/v1":
        raise ValueError("missing complete-job qualification schema")
    if qualification.get("status") != "PASS" or qualification.get("critic_qualified") is not True:
        raise ValueError("P1 requires PASS qualification and critic gate")
    for key in ("seconds_per_actor_transition", "per_job_overhead_seconds", "prefix_seconds"):
        value = qualification.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"missing complete-job component: {key}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"nonfinite/negative complete-job component: {key}")
    if qualification["seconds_per_actor_transition"] <= 0:
        raise ValueError("complete-job transition rate must be positive")
    if qualification.get("transfer_and_release_verified") is not True:
        raise ValueError("complete-job qualification lacks final transfer/release evidence")
    plan = provisional_plan()
    for chunk in plan["chunks"]:
        seconds = (
            chunk["actor_transitions"] * qualification["seconds_per_actor_transition"]
            + qualification["per_job_overhead_seconds"]
        )
        chunk["est_minutes"] = math.ceil(seconds / 60)
        chunk["timeout_minutes"] = 2 * chunk["est_minutes"]
        if chunk["timeout_minutes"] > 720:
            raise ValueError("measured P1 chunk cannot fit <=12h timeout")
        if chunk["timeout_minutes"] * RATE / 60 > 24 - 4.8 + 1e-9:
            raise ValueError("measured chunk exceeds liability plus reserved envelope")
    nominal = sum(chunk["est_minutes"] for chunk in plan["chunks"]) / 60
    total = nominal + qualification["prefix_seconds"] / 3600
    invoice = 2 * nominal * RATE + RESERVE
    if total > 46 or invoice > 120:
        raise ValueError("P1 NOT_READY: measured forecast exceeds 46h or 120u invoice")
    plan.update(
        status="MEASURED_CONDITIONALLY_ADMITTED",
        source="immutable 1M complete-job qualification including updates and probes",
        nominal_hours_remaining=nominal,
        complete_trajectory_hours=total,
        two_x_liability_units=2 * nominal * RATE,
        invoice_units=invoice,
        qualification=qualification,
    )
    return plan


def _source_identity(commit: str) -> dict[str, str]:
    paths = [
        "REGISTRATION_P1.md",
        "P1_EXECUTION_NOTE.md",
        "P1_BANK_PIN.json",
        "E8_AMENDMENT.md",
        "E8_LATE_BODY_AMENDMENT.md",
        "E8_LATE_BODY_AMENDMENT_02.md",
        "collector.py",
        "late_body.py",
        "model.py",
        "learner.py",
        "run.py",
        "recovery.py",
        "refit.py",
        "diagnostics.py",
        "offline_critic_diagnostic.py",
        "exports.py",
        "p1_runtime.py",
        "p1_contract.py",
        "p1_eval.py",
        "p1_manifests.py",
        "p1_qualification.py",
        "p1_inputs.py",
        "p1_colab.sh",
    ]
    import hashlib

    result = {}
    for name in paths:
        relative = f"research/bl_ppo/{name}"
        body = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if (ROOT / relative).read_bytes() != body:
            raise ValueError(f"working source differs from executable pin: {relative}")
        result[relative] = hashlib.sha256(body).hexdigest()
    return result


def bundle_inputs(root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep Contents staging at two inputs, including actual receipt sidecars."""
    public = [row for row in rows if row["dest"] == "public-inputs.json"]
    packed = [row for row in rows if row["dest"] != "public-inputs.json"]
    payloads = {}
    for row in packed:
        if Path(row["dest"]).name != row["dest"]:
            raise ValueError("P1 input bundle permits only flat regular files")
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("P1 input changed while preparing bundle")
        payloads[row["dest"]] = path.read_bytes()
        if row["dest"].endswith(".complete"):
            verify_export(Path(str(path).removesuffix(".complete")))
    import hashlib

    for name in list(payloads):
        if name.endswith(".sha256"):
            target = name.removesuffix(".sha256")
            if target not in payloads:
                raise ValueError("P1 bundle sidecar lacks its payload")
            # Renamed admission/audit inputs need transport-name sidecars.
            checksum = hashlib.sha256(payloads[target]).hexdigest()
            payloads[name] = f"{checksum}  {target}\n".encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, body in sorted(payloads.items()):
                header = tarfile.TarInfo(name)
                header.size = len(body)
                archive.addfile(header, io.BytesIO(body))
    payload = buffer.getvalue()
    if len(payload) > 6 * 1024**2:
        raise ValueError("P1 input bundle exceeds 6 MiB; use runtime download catalogs")
    export_bytes(root / "p1-inputs.tar.gz", payload)
    return public + [input_row(root / "p1-inputs.tar.gz", "p1-inputs.tar.gz")]


def prepare(
    *,
    root: Path,
    commit: str,
    phase: str,
    chunk: int = 1,
    downloads: Path | None = None,
    evidence: Path | None = None,
    resume_name: str | None = None,
    expected_trace_name: str | None = None,
    inventory: dict[str, Any] | None = None,
    bank: Path | None = None,
    audit: Path | None = None,
    parity: Path | None = None,
    endpoint_receipt: Path | None = None,
    previous_receipt: Path | None = None,
    hold_reason: str | None = None,
) -> dict[str, Any]:
    """Write a fresh attempt, immutable manifest and real source/upload receipt."""
    if phase not in PHASES or not 1 <= chunk <= 5:
        raise ValueError("unknown P1 phase/chunk")
    identity = _source_identity(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = provisional_plan()
    blockers = ["Claude: fresh usage/inventory, preceding terminal receipt and verified release"]
    if hold_reason:
        blockers.append(f"HOLD: {hold_reason}")
    prerequisites: dict[str, Any] = {}
    if phase == "y3-source":
        # Round-14: source-02 consumed 1.6u of the shared Y3 8u envelope.
        # Prior L4 source took 49min; reserve ~25min for the single support
        # continuation. This is a forecast, never a measured complete-job rate.
        estimate, timeout, cap, pause = 75, 150, 6.4, 0
        blockers.append("independent Codex-V review before dispatch")
        blockers.append(
            "Claude: reconcile source-02 debit (1.6u) and all later Y3 attempts; "
            "source plus replacement must stay within original 8u Y3 envelope"
        )
    elif phase == "y3-replacement":
        estimate, timeout, cap, pause = 20, 40, 2.0, 0
        blockers.append("Claude: source terminal receipt and distinct replacement identity")
        blockers.append(
            "Claude: reprice replacement from actual residual Y3 envelope "
            "after source-02 and source-latebody-02, never a new 2u allowance"
        )
    elif phase == "qual-1m":
        estimate, timeout, cap, pause = 40, 80, 4.0, QUAL_PAUSE
        if evidence:
            row = complete_json(evidence)
            if row.get("status") != "PASS" or row.get("schema") != "bl-p1-y3/v1":
                raise ValueError("qualification requires Y3 PASS")
            prerequisites["y3"] = input_row(evidence, "admission.json")
        else:
            blockers.append("immutable Y3 PASS on different L4 assignments")
    elif phase == "tranche":
        if evidence:
            plan = measured_plan(complete_json(evidence))
            prerequisites["qualification"] = input_row(evidence, "admission.json")
        else:
            blockers.append("Y3 PASS, critic PASS, complete-job 1M qualification PASS")
        item = plan["chunks"][chunk - 1]
        estimate, timeout, cap, pause = (
            item["est_minutes"],
            item["timeout_minutes"],
            item["unit_cap"],
            item["pause_at"],
        )
        if chunk > 1 and previous_receipt is None:
            blockers.append("immutable preceding tranche receipt and exact boundary snapshot")
    else:
        estimate, timeout, cap, pause = 90, 180, 10.0, TARGET
        if bank is None:
            blockers.append("fresh 128/mix collision-audited paired bank")
        if audit is None:
            blockers.append("fresh bank historical collision audit")
        if parity is None:
            blockers.append("E13 per-runtime parity receipt")
        if endpoint_receipt is None:
            blockers.append("100M endpoint receipt + checkpoint digest")
    control: dict[str, Any] = {
        "schema": "bl-p1-control/v1",
        "phase": phase,
        "target": TARGET,
        "pause_at": pause,
        "start_at": plan["chunks"][chunk - 1]["start"] if phase == "tranche" else 0,
        "requested_pause_at": 1_000_000 if phase == "qual-1m" else pause,
        "seed": 20261021,
        "arm": "F",
        "worlds": 128,
        "processes": 12,
        "rollout_steps": 128,
        "p1_recipe": "F",
        "chunk_index": chunk,
        "durability_min_actions": 1000,
        "durability_min_actor_updates": 3,
        "durability_require_episode_boundary": True,
        "checkpoint_every_seconds": 1800,
        "max_snapshot_bytes": 167772160,
        "source_commit": commit,
        "registration_sha256": identity["research/bl_ppo/REGISTRATION_P1.md"],
        "late_body_amendment_sha256": identity["research/bl_ppo/E8_LATE_BODY_AMENDMENT_02.md"],
        "resume_name": resume_name,
        "expected_trace_name": expected_trace_name,
        "prerequisites": prerequisites,
    }
    if bank is not None and audit is not None:
        from research.bl_ppo.p1_eval import check_audit

        allocation, seed_audit = complete_json(bank), complete_json(audit)
        check_audit(allocation, seed_audit)
        control.update(
            evaluation_round=allocation["round_index"],
            evaluation_bank_sha256=allocation["allocation_sha256"],
            evaluation_audit_sha256=digest(audit),
            evaluation_excluded_seeds=sorted(int(row["world_seed"]) for row in allocation["rows"]),
        )
    elif phase != "eval":
        blockers.append("pre-data frozen fresh evaluation bank and collision audit")
    inputs = [input_row(ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json")]
    if downloads is not None:
        download_rows = validate_catalog(json.loads(downloads.read_text()))
        names = {row["name"] for row in download_rows}
        if resume_name and resume_name not in names:
            raise ValueError("resume snapshot missing from transitive download catalog")
        if expected_trace_name and expected_trace_name not in names:
            raise ValueError("Y3 expected trace missing from download catalog")
        if phase == "eval" and resume_name and endpoint_receipt:
            endpoint = complete_json(endpoint_receipt)
            downloaded = next(row for row in download_rows if row["name"] == resume_name)
            if downloaded["sha256"] != endpoint.get("checkpoint_sha256"):
                raise ValueError("evaluation runtime download differs from the fixed 100M endpoint")
        inputs.append(input_row(downloads, "runtime-downloads.json"))
        control["runtime_downloads_sha256"] = digest(downloads)
    elif phase != "y3-source":
        blockers.append("runtime-fetch catalog with complete transitive snapshot set")
    if phase in {"y3-replacement", "qual-1m", "tranche", "eval"} and resume_name is None:
        blockers.append("exact origin/preceding boundary snapshot name")
    if phase == "y3-replacement" and expected_trace_name is None:
        blockers.append("source continuation tape and full-state identity")
    if previous_receipt is not None:
        previous = complete_json(previous_receipt)
        if (
            previous.get("schema") != "bl-p1-tranche/v1"
            or previous.get("status") != "PAUSED"
            or previous.get("source_commit") != commit
            or previous.get("target") != TARGET
            or previous.get("actor_transitions") != control["start_at"]
        ):
            raise ValueError("preceding tranche receipt differs from the frozen boundary")
        control["previous_receipt_sha256"] = digest(previous_receipt)
    if evidence is not None:
        for suffix in ("", ".sha256", ".complete"):
            inputs.append(input_row(Path(str(evidence) + suffix), "admission.json" + suffix))
    if bank is not None and phase == "eval":
        for suffix in ("", ".sha256", ".complete"):
            inputs.append(input_row(Path(str(bank) + suffix), "bank.json" + suffix))
    for path, name in (
        (audit, "bank-audit.json"),
        (parity, "parity.json"),
        (endpoint_receipt, "endpoint-receipt.json"),
        (previous_receipt, "previous.json"),
    ):
        if path is not None:
            complete_json(path)
            for suffix in ("", ".sha256", ".complete"):
                inputs.append(input_row(Path(str(path) + suffix), name + suffix))
    control["dispatch_blockers"] = blockers
    control["scientific_admission"] = not any(
        not b.startswith(("Claude:", "independent")) for b in blockers
    )
    export_json(root / "p1-control.json", control)
    for suffix in ("", ".sha256", ".complete"):
        inputs.append(input_row(root / ("p1-control.json" + suffix), "p1-control.json" + suffix))
    manifest = _base(
        job_id=f"bl-p1-{phase}-{chunk:02d}-{commit[:8]}-{root.name}",
        commit=commit,
        output=root / "colab-output",
        estimate=estimate,
        timeout=timeout,
        cap=cap,
    )
    manifest.update(
        script="research/bl_ppo/p1_colab.sh",
        claim="P1_FIXED_ENDPOINT_DEVELOPMENT_ONLY",
        inputs=bundle_inputs(root, inputs),
        env={
            "BL_P1_MODE": phase,
            "BL_PPO_COMMIT": commit,
            "BL_PUBLIC_INPUTS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
        max_export_bytes=2 * 1024**3,
    )
    manifest["budget"]["other_assignments"] = inventory or {}
    manifest["source_paths"] = _bounded_source_paths(manifest)
    manifest["source_paths"] += [
        "research/bl_upload",
        "research/t34_exit_b8/public_inputs.py",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
    ]
    manifest["source_paths"] = sorted(set(manifest["source_paths"]))
    manifest["bl_p1"] = {"control": control, "forecast": plan, "blockers": blockers}
    export_json(root / "manifest.json", manifest)
    from research.bl_upload.qualify import qualify

    upload = qualify(manifest, repo=ROOT)
    export_json(root / "upload-preflight.json", upload)
    receipt = {
        "schema": "bl-p1-preparation/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "phase": phase,
        "commit": commit,
        "source_sha256": identity,
        "manifest_sha256": digest(root / "manifest.json"),
        "upload_receipt_sha256": digest(root / "upload-preflight.json"),
        "forecast": plan,
        "estimate_minutes": estimate,
        "timeout_minutes": timeout,
        "unit_cap": cap,
        "reserve_units": cap * 0.2,
        "blockers": blockers,
        "dispatch_ready": False,
        "colab_calls": 0,
    }
    export_json(root / "receipt.json", receipt)
    python = ROOT.parent / "snake-dqn/venv/bin/python"
    dispatch = (
        f"P1 {phase}: PREPARED_NOT_DISPATCHED. Claude is the operator.\n"
        + "\n".join(f"BLOCK: {b}" for b in blockers)
        + "\nNo automatic retry or unregistered recovery. One P1 L4 at a time.\n"
        + f"Executable pin: {commit}\n"
        + f"cd {ROOT}\n"
        + f"Estimate {estimate} min; timeout {timeout} min; cap {cap}u; reserve {cap*.2}u.\n"
        + f"{python} -m research.bl_ppo.upload_readiness --manifest {root}/manifest.json "
        + f"--receipt {root}/upload-preflight.json --verify\n"
        + f"{ROOT}/research/colab_cli/launch.sh --check --state-dir {root}/state "
        + "--colab /Users/josenunez/.local/bin/colab validate "
        + f"{root}/manifest.json\n"
        + "Before assignment: usage + fresh inventory + preceding terminal receipt + "
        + "verified disappearance. Inventory changes require a NEW prepared attempt/receipt.\n"
        + "Run only after every blocker is discharged and this exact manifest is reviewed.\n"
        + f"{ROOT}/research/colab_cli/launch.sh --state-dir {root}/state "
        + "--colab /Users/josenunez/.local/bin/colab run "
        + f"{root}/manifest.json\n"
    )
    export_bytes(root / "DISPATCH.txt", dispatch.encode())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--downloads", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--resume-name")
    parser.add_argument("--expected-trace-name")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--endpoint-receipt", type=Path)
    parser.add_argument("--previous-receipt", type=Path)
    parser.add_argument("--hold-reason", help="Explicit unresolved blocker; never admits dispatch")
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text()) if args.inventory else None
    print(
        json.dumps(
            prepare(
                root=args.root,
                commit=args.commit,
                phase=args.phase,
                chunk=args.chunk,
                downloads=args.downloads,
                evidence=args.evidence,
                resume_name=args.resume_name,
                expected_trace_name=args.expected_trace_name,
                bank=args.bank,
                inventory=inventory,
                audit=args.audit,
                parity=args.parity,
                endpoint_receipt=args.endpoint_receipt,
                previous_receipt=args.previous_receipt,
                hold_reason=args.hold_reason,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
