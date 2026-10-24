"""Host-only V8 budget reconstruction package; the qualified runtime stays frozen.

The previous amendment's exact source and package remain evidence. This companion
calls its unchanged native/canary/archive acceptance path and permits only the
new judgment's local metadata and fresh job/output identity changes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path("/Users/josenunez/Projects/ml")
ROOT = BASE / "snake-dqn-gpu-v8"
BROKER = BASE / "snake-dqn-v8-metadata-qualified-02.noindex"
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
PREVIOUS = EXP / "gpu-v8-main-amended-20261024-01"
ART = EXP / "gpu-v8-main-budget-20261024-02"
RULING = EXP / "JUDGMENT_V8_ADMISSION_BUDGET_20261024.md"
RULING_SHA = "c6e6683fb4df8e5831c1d55c640050fe43c265253d58fde586f0041de17a3e59"
PIN = "669405e2bef629f7f69bc951cf49486d8b0a87bf"
JOB = "gpu-v8-split-main-669405e2bef6-budget-02"
PRIOR_SOURCE = ROOT / "tools/v8_split_main_amend.py"
PRIOR_SHA = "45c3e34a98fbc11969a97c1a2ae8151c4087accffa518415ae2138a70be23a2d"
HOST = BASE / "snake-dqn-gpufix"
PY = BASE / "snake-dqn/venv/bin/python"
TESTS = ROOT / "tests/test_v8_budget_main_amend.py"
AMENDMENT = ROOT / "tools/V8_ADMISSION_BUDGET_AMENDMENT_20261024.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load companion: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def previous() -> Any:
    if digest(PRIOR_SOURCE) != PRIOR_SHA:
        raise ValueError("previous amendment implementation changed")
    return load(PRIOR_SOURCE, "v8_preserved_main_amendment")


def authority() -> dict[str, str]:
    if digest(RULING) != RULING_SHA:
        raise ValueError("missing or wrong budget authority hash")
    return {"path": str(RULING), "sha256": RULING_SHA}


def implementation() -> dict[str, dict[str, str]]:
    """Bind the complete local dispatch and admission implementation closure."""
    paths = {
        "source": Path(__file__).resolve(),
        "tests": TESTS,
        "amendment": AMENDMENT,
        "previous_source": PRIOR_SOURCE,
        "previous_tests": ROOT / "tests/test_v8_split_main_amend.py",
        "previous_amendment": ROOT / "tools/V8_SPLIT_MAIN_CAP_AMENDMENT_20261024.md",
        "accounting": ROOT / "tools/v8_split_admission.py",
        "accounting_tests": ROOT / "tests/test_v8_split_admission.py",
        "evidence": ROOT / "tools/v8_admission_evidence.py",
        "evidence_tests": ROOT / "tests/test_v8_admission_evidence.py",
        "dispatch": HOST / "tools/ops/dispatch.py",
        "repack": HOST / "tools/ops/repack_gpu_upload.py",
        "receipt_helper": HOST / "tools/ops/receipt_dispatch.py",
        "native_binding": HOST / "research/gpu_sim_spike/native_compile.py",
    }
    budget_helper = ROOT / "tools/v8_admission_budget.py"
    if budget_helper.exists():
        paths["budget_helper"] = budget_helper
        paths["budget_tests"] = ROOT / "tests/test_v8_admission_budget.py"
    return {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()}


def no_main() -> None:
    previous().no_main_evidence()
    for path in (ART / "main-output", ART / "state-main"):
        if path.exists():
            raise ValueError(f"initial main identity already used: {path}")
    ops = BASE / "snake-dqn-artifacts/ops"
    for path in (ops / "dispatch" / JOB, ops / "manifests" / f"{JOB}.json"):
        if path.exists():
            raise ValueError(f"initial main already reserved or run: {path}")


def build(evidence: dict[str, Any]) -> dict[str, Any]:
    authority()
    old = previous()
    value = old.build(evidence, old.authority(old.RULING, old.RULING_SHA))
    value = copy.deepcopy(value)
    value["job_id"] = JOB
    value["output_dir"] = str(ART / "main-output")
    value["v8_split"]["effective_envelope"] = {"wall_minutes": 150, "units": 25}
    value["v8_split"]["canary"].update(wall_minutes=30, units=5)
    value["cap_amendment"].update(
        preparation_receipt=str(ART / "PREPARATION.json"),
        base_envelopes={"canary": [15, 2.5], "main": [60, 10], "packet": [75, 12.5]},
        effective_envelopes={"canary": [30, 5], "main": [120, 20], "packet": [150, 25]},
    )
    value["admission_budget_amendment"] = {
        "ruling": authority(),
        "sibling_amendment": {"path": str(AMENDMENT), "sha256": digest(AMENDMENT)},
        "previous_split_main_ruling": {"path": str(old.RULING), "sha256": old.RULING_SHA},
        "policy_version": "v8-admission-budget-20261024-v1",
        "authority_kind": "ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL",
        "packet_cap": {"wall_minutes": 150, "units": 25},
        "historical_canary_charge_ceiling": {"wall_minutes": 30, "units": 5},
        "reserved_main": {"wall_minutes": 120, "units": 20},
        "base_topup": {"wall_minutes": 3, "units": 0.5},
        "effective_topup": {"wall_minutes": 6, "units": 1},
        "reconstruction_record": str(ART / "RECONSTRUCTION.json"),
        "no_new_canary_or_completion_funded": True,
    }
    return value


def validate_final(value: dict[str, Any]) -> dict[str, Any]:
    authority()
    old = previous()
    evidence = old.accept_package()
    no_main()
    if value != build(evidence):
        raise ValueError("manifest differs from exact budget amendment/scientific contract")
    old.require_resume_policy(value)
    old.supervisor_validate(value, repo=BROKER)
    return evidence


def dispatch_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {BROKER}
export PYTHONPATH={BROKER}:{ROOT}
export PYTHONDONTWRITEBYTECODE=1
exec {PY} {Path(__file__).resolve()} dispatch "$@"
"""


def prepare() -> dict[str, Any]:
    authority()
    old = previous()
    evidence = old.accept_package()
    no_main()
    value = build(evidence)
    validate_final(value)
    if ART.exists():
        raise ValueError("fresh sibling package already exists")
    ART.mkdir()
    old.v8_split_units.publish_json(ART / "v8-split-main.json", value)
    old.pack._write(ART / "dispatch-main.sh", dispatch_script())
    (ART / "dispatch-main.sh").chmod(0o755)
    old.pack._write(ART / "DISPATCH.txt", f"nice -n 15 bash {ART}/dispatch-main.sh\n")
    receipt = {
        "status": "PREPARED_OFFLINE_VALIDATED_NOT_LIVE_ADMITTED",
        "allocations": 0,
        "runtime_commit": PIN,
        "ruling": authority(),
        "previous_ruling": {"path": str(old.RULING), "sha256": old.RULING_SHA},
        "previous_preparation": {
            "path": str(PREVIOUS / "PREPARATION.json"),
            "sha256": old._verify_file(PREVIOUS / "PREPARATION.json"),
        },
        "evidence": evidence,
        "implementation": implementation(),
        "manifest_sha256": digest(ART / "v8-split-main.json"),
        "dispatch_sha256": digest(ART / "dispatch-main.sh"),
        "operator_accounting": "Fresh reviewed reconstruction/ledger and live boundary required",
    }
    old.v8_split_units.publish_json(ART / "PREPARATION.json", receipt)
    return receipt


def verify_preparation(require_review: bool = True) -> dict[str, Any]:
    old = previous()
    old._verify_file(ART / "PREPARATION.json")
    prep = json.loads((ART / "PREPARATION.json").read_text())
    if (
        prep.get("ruling") != authority()
        or prep.get("implementation") != implementation()
        or prep.get("manifest_sha256") != old._verify_file(ART / "v8-split-main.json")
        or prep.get("dispatch_sha256") != old._verify_file(ART / "dispatch-main.sh")
    ):
        raise ValueError("prepared source/tests/manifest/dispatch binding differs")
    value = json.loads((ART / "v8-split-main.json").read_text())
    validate_final(value)
    if require_review:
        old._verify_file(ART / "CODEX_V_LIGHT.json")
        review = json.loads((ART / "CODEX_V_LIGHT.json").read_text())
        if (
            review.get("status") != "ACCEPT_OFFLINE"
            or review.get("preparation_sha256") != digest(ART / "PREPARATION.json")
            or review.get("manifest_sha256") != digest(ART / "v8-split-main.json")
            or review.get("implementation") != implementation()
        ):
            raise ValueError("exact budget package lacks independent Codex-V acceptance")
    return value


def admission() -> None:
    """Revalidate all accounting at dispatch, including its exact live review."""
    verify_preparation()
    from tools import v8_split_admission as accounting

    path = ART / "OPERATOR_ADMISSION.json"
    accounting.verified(path)
    body = json.loads(path.read_text())
    body_sha = accounting.validate_live(body)
    review = body.get("codex_v_review", {})
    review_path = Path(review.get("path", ""))
    if accounting.verified(review_path) != review.get("sha256"):
        raise ValueError("accounting review binding differs")
    accepted = json.loads(review_path.read_text())
    if accepted.get("status") != "ACCEPT" or accepted.get("accounting_sha256") != body_sha:
        raise ValueError("no independent Codex-V ACCEPT for exact live accounting")


def campaign_entry() -> str:
    """Describe the actual accepted boundary only after operator finalization."""
    from tools.v8_admission_evidence import sourced

    body = json.loads((ART / "OPERATOR_ADMISSION.json").read_text())
    reconstruction = sourced(body["accounting_evidence"]["reconstruction"])
    ledger = body["accounting_evidence"]["ledger"]
    return (
        f"\n### V8 budget initial-main live admission — {JOB}\n\n"
        f"Astra authority `{RULING}` SHA256 `{RULING_SHA}`; "
        "authority_kind=ASTRA_RECONSTRUCTION_NOT_RECOVERED_SEAL. "
        f"Operative reconstructed T0 `{body['original_t0']}`; "
        "frozen reconstructed spendable effective limit C="
        f"`{reconstruction['reconstructed_spendable_effective_limit_units']}` units. "
        f"Current effective limit `{body['effective_spendable_limit_units']}` units. "
        f"Reconstruction `{body['accounting_evidence']['reconstruction']['path']}` "
        f"SHA256 `{body['accounting_evidence']['reconstruction']['sha256']}`. "
        f"Conservatively inventoried ledger `{ledger['path']}` SHA256 `{ledger['sha256']}`. "
        f"Exact operator admission `{ART / 'OPERATOR_ADMISSION.json'}` "
        f"SHA256 `{digest(ART / 'OPERATOR_ADMISSION.json')}`. "
        "Packet 150 min/25 u; historical canary portion 30/5; "
        "initial main reserved once at 120/20 including reserve. "
        "This records admission before dispatch; it is not evidence of execution.\n"
    )


def record_campaign() -> None:
    admission()
    path = ROOT / "docs/research/final_run_20261019/CAMPAIGN_LOG.md"
    entry = campaign_entry()
    existing = path.read_text()
    if f"### V8 budget initial-main live admission — {JOB}" in existing:
        if entry not in existing:
            raise ValueError("different campaign admission already recorded")
        return
    with path.open("a") as stream:
        stream.write(entry)
        stream.flush()
        os.fsync(stream.fileno())


def inventory(output: Path) -> dict[str, Any]:
    """Hash local operational records without inspecting scientific outcomes.

    This is an inventory seed for the operator, never a completeness attestation.
    Broker records are hashed as full bytes; no account credentials are printed.
    """
    roots = [
        BASE / "snake-dqn-artifacts/ops/manifests",
        BASE / "snake-dqn-artifacts/ops/state",
        BASE / "snake-dqn-artifacts/ops/dispatch",
        Path("/Users/josenunez/.config/colab-cli/snake"),
        EXP,
        BASE / "snake-dqn-artifacts/a100-retention-20261023",
    ]
    paths = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.is_symlink() or any(
                part in {"exports", "units", "operator-fill", "__pycache__"}
                for part in path.relative_to(root).parts[:-1]
            ):
                continue
            name = path.name.lower()
            if (
                name in {"receipt.json", "manifest.json", "canary.json", "main.json"}
                or any(word in name for word in ("admission", "ledger", "commitment"))
                or name.startswith(("active-", "pending-"))
                or root == roots[0]
            ):
                paths.add(path.resolve())
    value = {
        "status": "INVENTORY_SEED_NOT_COMPLETE_ACCOUNTING",
        "complete_history": False,
        "complete_liabilities": False,
        "roots": [{"path": str(root), "present": root.is_dir()} for root in roots],
        "records": [
            {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in sorted(paths)
        ],
        "operator_requirements": [
            "Reconcile broker pending/active/history, notebooks, dispatches and receipts.",
            "Inventory admitted commitments and required-work holds beyond this seed too.",
            "Do not treat packages as actual assignments or scientific outcomes as accounting.",
            "Bind full raw provider evidence separately; unknown exposure is never zero.",
        ],
    }
    previous().v8_split_units.publish_json(output, value)
    return {"status": value["status"], "records": len(paths), "path": str(output)}


def dispatch_window(body: dict[str, Any], now: datetime) -> None:
    """Keep the bounded wrapper wait inside freshness and the calendar ceiling."""
    from tools.v8_admission_evidence import utc

    if utc(body["valid_until_utc"]) < now + timedelta(seconds=600):
        raise ValueError("live accounting cannot cover the 600-second wrapper allocation window")
    if utc(body["calendar_deadline_utc"]) < now + timedelta(minutes=130):
        raise ValueError("calendar cannot cover full main plus 600-second allocation window")


def dispatch(dry: bool) -> dict[str, Any]:
    value = verify_preparation()
    ready = (ART / "OPERATOR_ADMISSION.json").exists()
    if ready or not dry:
        admission()
        log = ROOT / "docs/research/final_run_20261019/CAMPAIGN_LOG.md"
        if campaign_entry() not in log.read_text():
            raise ValueError(
                "accepted C/T0/ledger must be recorded in CAMPAIGN_LOG before dispatch"
            )
    helper = load(HOST / "tools/ops/repack_gpu_upload.py", "budget_v8_upload_checker")
    with tempfile.TemporaryDirectory(prefix="v8-budget-upload-") as temporary:
        upload = helper.check_upload(value, BROKER, Path(temporary) / "upload")
    if ready or not dry:
        # Upload verification may take time. Never carry its earlier live check
        # through to the governed launch without rechecking freshness and clock.
        admission()
        body = json.loads((ART / "OPERATOR_ADMISSION.json").read_text())
        dispatch_window(body, datetime.now(timezone.utc))
    command = [
        str(PY),
        str(HOST / "tools/ops/dispatch.py"),
        JOB,
        str(ART / "v8-split-main.json"),
        "--broker",
        str(BROKER),
        "--cwd",
        str(BROKER),
        "--state-dir",
        str(ART / "state-main"),
        "--dispatch-txt",
        str(ART / "DISPATCH.txt"),
    ]
    if dry:
        command.append("--dry-run")
    subprocess.run(
        command,
        cwd=HOST,
        env=dict(os.environ, PYTHONPATH=str(HOST), PYTHONDONTWRITEBYTECODE="1"),
        check=True,
    )
    return {
        "status": "PASS_OFFLINE_DRY_RUN" if dry else "DISPATCHED",
        "live_admission": (
            "PASS" if ready else "NOT_READY: missing reviewed OPERATOR_ADMISSION.json"
        ),
        "dispatch_ready": ready,
        "upload": {
            key: upload[key]
            for key in ("source_files", "source_archive_bytes", "largest_upload_bytes")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("prepare", "verify", "admission", "record-campaign", "inventory", "dispatch"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare()
    elif args.operation == "dispatch":
        result = dispatch(args.dry_run)
    elif args.operation == "admission":
        admission()
        result = {"status": "PASS_ADMISSION", "allocations": 0}
    elif args.operation == "record-campaign":
        record_campaign()
        result = {"status": "RECORDED_ADMISSION_BEFORE_DISPATCH", "allocations": 0}
    elif args.operation == "inventory":
        if args.output is None or not args.output.is_absolute():
            parser.error("inventory requires absolute --output")
        result = inventory(args.output)
    else:
        verify_preparation()
        result = {"status": "PASS_PREPARATION", "allocations": 0}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
