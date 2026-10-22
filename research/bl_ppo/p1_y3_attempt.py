"""Prepare the source-latebody-05 Y3 package without allocating a runtime.

The only paid actions are deliberately left as explicit operator commands in
``DISPATCH.txt``.  This module freezes the ledger, reviewed source, debit and
canary/source budgets before either command can be used.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_ppo.manifests import input_row
from research.bl_ppo.p1_admission import reconcile_y3_05
from research.bl_ppo.p1_inputs import unpack_bundle
from research.bl_ppo.p1_manifests import bundle_inputs


def _complete(path: Path) -> dict[str, Any]:
    from research.bl_ppo.exports import verify_export

    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("attempt evidence must be a JSON object")
    return value


def _ledger(root: Path, *, namespace: str, allocation: str) -> Path:
    path = root / "verified-ledger.json"
    export_json(
        path,
        {
            "schema": "bl-p1-verified-ledger/v1",
            "namespace": namespace,
            "allocation": allocation,
            "units": [],
        },
    )
    return path


def _check_code_review(review: Path, commit: str) -> None:
    """Require an independent exact-source review before its bytes are bundled."""
    value = _complete(review)
    from research.bl_ppo.p1_manifests import _source_identity

    if (
        value.get("schema") != "bl-p1-codex-v-review/v1"
        or value.get("status") != "PASS"
        or value.get("unresolved_findings") != []
        or value.get("no_new_cuda_path") is not True
        or value.get("source_sha256") != _source_identity(commit)
    ):
        raise ValueError("code review does not bind the exact Y3-05 executable source")


def _source_template(
    source: Path, root: Path, commit: str, code_review: Path, debit: Path, ledger: Path
) -> Path:
    """Rebind a prior source shell while replacing every mutable Y3-05 input."""
    original = json.loads(source.read_text())
    if original.get("script") != "research/bl_ppo/p1_colab.sh":
        raise ValueError("requires a P1 source manifest/template")
    bundled = next(
        (row for row in original.get("inputs", []) if row.get("dest") == "p1-inputs.tar.gz"), None
    )
    if not bundled or digest(Path(bundled["path"])) != bundled["sha256"]:
        raise ValueError("base source input bundle is absent or mutated")
    unpacked = root / "prior-inputs"
    unpack_bundle(Path(bundled["path"]), unpacked)
    control = json.loads((unpacked / "p1-control.json").read_text())
    namespace = "bl_p1_y3_20261022"
    allocation = "source_latebody_05"
    control.update(
        schema="bl-p1-control/v1",
        phase="y3-source",
        source_commit=commit,
        checkpoint_every_seconds=900,
        resume={
            "unit_kind": "operational-snapshot",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
            "namespace": namespace,
            "allocation": allocation,
        },
        operational_snapshot={
            "cadence_seconds": 900,
            "label": "OPERATIONAL",
            "replay_min_actions": 1000,
            "replay_min_updates": 3,
        },
        operational_replay_required=True,
        resume_amendment_sha256=digest(Path(__file__).with_name("P1_Y3_RESUME_AMENDMENT_05.md")),
        y3_05_envelope={
            "base_total_units": 12.75,
            "effective_total_units": 25.5,
            "canary_base": {
                "forecast_minutes": 15,
                "timeout_minutes": 15,
                "cap_units": 1.0,
                "reserve_units": 0.2,
            },
            "canary_effective": {"timeout_minutes": 30, "cap_units": 2.0, "reserve_units": 0.4},
            "rate_units_per_hour": 1.54,
            "rate_tolerance_units_per_hour": 0.0,
            "source_base": {
                "forecast_minutes": 75,
                "timeout_minutes": 150,
                "cap_units": 4.81,
                "reserve_units": 0.96,
            },
            "source_effective": {"timeout_minutes": 300, "cap_units": 9.62, "reserve_units": 1.92},
            "replacement_base": {
                "forecast_minutes": 30,
                "timeout_minutes": 60,
                "cap_units": 1.93,
                "reserve_units": 0.39,
            },
            "replacement_effective": {
                "timeout_minutes": 120,
                "cap_units": 3.86,
                "reserve_units": 0.78,
            },
        },
    )
    control["prerequisites"] = {
        "codex_v_code_review": input_row(code_review, "codex-v-review.json"),
        "y3_05_debit": input_row(debit, "y3-debit.json"),
        "verified_ledger": input_row(ledger, "verified-ledger.json"),
    }
    from research.bl_ppo.export_budget import freeze_control_budget

    byte_budget = freeze_control_budget(
        control, timeout_minutes=300, max_export_bytes=original["max_export_bytes"]
    )
    new_control = root / "p1-control.json"
    export_json(new_control, control)
    rows = [row for row in original["inputs"] if row["dest"] != "p1-inputs.tar.gz"]
    for path in unpacked.iterdir():
        if path.name.startswith(("p1-control.json", "codex-v-review.json", "y3-debit.json")):
            continue
        rows.append(input_row(path, path.name))
    for payload, name in (
        (new_control, "p1-control.json"),
        (ledger, "verified-ledger.json"),
        (code_review, "codex-v-review.json"),
        (debit, "y3-debit.json"),
    ):
        for suffix in ("", ".sha256", ".complete"):
            rows.append(input_row(Path(str(payload) + suffix), name + suffix))
    template = copy.deepcopy(original)
    template.update(
        commit=commit,
        job_id=f"bl-p1-y3-source-latebody-05-{commit[:8]}",
        output_dir=str((root / "source-output").resolve()),
        inputs=bundle_inputs(root, rows),
        timeout_minutes=300,
        device="cuda:0",
        resume=control["resume"],
        max_total_export_bytes=byte_budget["max_total_export_bytes"],
    )
    template.pop("canary", None)
    template["requirements"]["est_minutes"] = 75
    template["budget"].update(
        unit_cap=9.62,
        reserve_units=1.92,
        rate_units_per_hour=1.54,
        rate_tolerance_units_per_hour=0.0,
        other_assignments={},
    )
    template["env"]["BL_PPO_COMMIT"] = commit
    template["env"]["BL_PPO_DEVICE"] = "cuda:0"
    template["bl_p1"]["control"] = control
    template["bl_p1"]["forecast"] = {
        "status": "Y3_ONLY_NO_TRANCHE_ADMISSION",
        "envelope": control["y3_05_envelope"],
    }
    template["bl_p1"]["blockers"] = [
        "genuine current-main family canary receipt required before source admit"
    ]
    target = root / "source-template-precanary.json"
    export_json(target, template)
    return target


def prepare(
    *,
    source: Path,
    root: Path,
    commit: str,
    acceptance: Path,
    code_review: Path,
    receipts: list[Path],
) -> dict[str, Any]:
    """Create a fresh, evidence-bound canary/source attempt; never dispatch."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    _check_code_review(code_review, commit)
    debit_value = reconcile_y3_05(receipts)
    debit = root / "y3-debit.json"
    export_json(debit, debit_value)
    ledger = _ledger(root, namespace="bl_p1_y3_20261022", allocation="source_latebody_05")
    template = _source_template(source, root, commit, code_review, debit, ledger)
    from research.bl_ppo import p1_family_canary

    family_root = root / "family-canary"
    p1_family_canary.prepare(template, family_root, commit, acceptance)
    status = {
        "schema": "bl-p1-y3-attempt/v1",
        "status": "PREPARED_NOT_DISPATCHED",
        "attempt": "source-latebody-05",
        "commit": commit,
        "source_template_precanary_sha256": digest(template),
        "family_preparation_sha256": digest(family_root / "preparation.json"),
        "code_review_sha256": digest(code_review),
        "debit_sha256": digest(debit),
        "authority": [
            {"path": str(path), "sha256": digest(path)}
            for path in (
                Path(__file__).resolve().parents[3]
                / "snake-dqn-artifacts/big-lever/exp/bl-p1/JUDGMENT_Y3_ENVELOPE.md",
                Path(__file__).resolve().parents[3]
                / "snake-dqn-artifacts/big-lever/JUDGMENT_RESUME_POLICY.md",
            )
        ],
        "owner_rule": "Owner request: 25.5u total; 2x caps/reserves/timeouts frozen before data",
        "ledger_sha256": digest(ledger),
        "envelope": debit_value,
        "canary_manifest": str(family_root / "canary-manifest.json"),
        "source_template": str(family_root / "source-template.json"),
        "colab_calls": 0,
    }
    export_json(root / "preparation.json", status)
    repo = Path(__file__).resolve().parents[2]
    python = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
    wrapper = f"{repo}/tools/ops/dispatch.sh"
    options = f"--broker {repo} --cwd {repo} --dispatch-txt {root}/DISPATCH.txt"
    dispatch = (
        "P1 Y3 source-latebody-05. Claude/operator only.\n"
        "Check fresh usage and sessions; require zero other assignments at <=1.54u/h.\n"
        f"cd {repo}\n"
        f"{wrapper} p1-y3-05-canary-20261022 {family_root}/canary-manifest.json {options}\n"
        "Wait for the passing canary terminal receipt, verified exports and release.\n"
        f"{python} -m research.bl_ppo.p1_family_canary admit --root {family_root} "
        f"--receipt {family_root}/canary-output/receipt.json --acceptance {acceptance} "
        f"--review {root}/independent-artifact-review.json\n"
        f"{wrapper} p1-y3-05-source-20261022 {family_root}/source-admitted.json {options}\n"
    )
    export_bytes(root / "DISPATCH.txt", dispatch.encode())
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--code-review", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                source=args.source,
                root=args.root,
                commit=args.commit,
                acceptance=args.acceptance,
                code_review=args.code_review,
                receipts=args.receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
