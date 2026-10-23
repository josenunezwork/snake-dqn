"""Prepare the source-bound F1 ruler; all live operations belong to the operator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import manifests
from .exports import _verify, publish_json
from .f1_ruler import load_bank
from .f1_units import empty_ledger, load_ledger

SCRIPT = "research/bl_labels/f1_colab.sh"
QUALIFIED_TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"
QUALIFIED_RECORD = manifests.QUALIFIED_RECORD
# Base forecasts/caps; owner factor applies once, including reserves/timeouts.
STAGES = {
    "cpu-probe": {"minutes": 10, "units": 0.02, "reserve": 0.004, "finalize": 60},
    "cpu-science": {"minutes": 45, "units": 0.08, "reserve": 0.016, "finalize": 150},
    "l4-probe": {"minutes": 5, "units": 0.18, "reserve": 0.036, "finalize": 60},
    "l4-science": {"minutes": 23.5, "units": 0.76, "reserve": 0.152, "finalize": 150},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: Path, dest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "dest": dest}


def triplet(path: Path, dest: str) -> list[dict[str, str]]:
    _verify(path)
    return [
        _input(Path(str(path) + suffix), dest + suffix) for suffix in ("", ".sha256", ".complete")
    ]


def envelope(stage: str) -> dict[str, Any]:
    base = STAGES[stage]
    return {
        "base": dict(base),
        "effective": {k: 2 * v for k, v in base.items()},
        "owner_multiplier": 2,
    }


def qualification() -> dict[str, Any]:
    record = json.loads(QUALIFIED_RECORD.read_text())
    if record["tree_hash"] != QUALIFIED_TREE:
        raise ValueError("F1 qualified transport tree differs")
    path = Path(record["acceptance_result"])
    ready, reason = False, "acceptance missing or hash changed"
    if path.is_file() and _sha(path) == record["acceptance_result_sha256"]:
        result = json.loads(path.read_text())
        transfer = result.get("final_artifact_transfer", {})
        ready = (
            result.get("status") == "PASS"
            and result.get("qualified") is True
            and transfer.get("status") == "PASS"
            and transfer.get("hardware") == "L4"
            and transfer.get("verified_on_mac") is True
        )
        reason = "qualified L4 mitigation acceptance" if ready else "L4 acceptance not passed"
    return {
        "tree_hash": QUALIFIED_TREE,
        "record_sha256": _sha(QUALIFIED_RECORD),
        "acceptance_result": str(path),
        "acceptance_result_sha256": record["acceptance_result_sha256"],
        "l4_ready": ready,
        "reason": reason,
    }


def make_template(
    commit: str,
    output: Path,
    bank: Path,
    ledger: Path,
    b8: Path,
    frp3: Path,
    *,
    platform: str,
    qualified: dict[str, Any],
    is_probe: bool = False,
) -> dict[str, Any]:
    from research.bl_upload import assert_upload_budget, convert_public_checkpoints

    if platform == "l4" and not qualified["l4_ready"]:
        raise ValueError("F1 L4 NOT_RUN: stall mitigation is not qualified")
    bank_sha = _verify(bank)
    saved = load_ledger(ledger, bank_sha, platform, commit)
    stage = f"{platform}-" + ("probe" if is_probe else "science")
    limits = envelope(stage)["effective"]
    inputs = triplet(bank, "inputs/f1-root-bank.json") + triplet(
        ledger, "inputs/f1-verified-ledger.json"
    )
    ledger_value = json.loads(ledger.read_text())
    for row in ledger_value["units"]:
        inputs += triplet(ledger.parent / row["path"], "inputs/" + row["path"])
    checkpoints = manifests._pool_inputs(b8, frp3)
    if len(checkpoints) != 6:
        raise ValueError("F1 requires all six pinned public checkpoints")
    inputs += checkpoints
    timeout = limits["minutes"]
    rate = 1.54 if platform == "l4" else 0.08
    if rate * timeout / 60 + limits["reserve"] > limits["units"]:
        raise ValueError("F1 whole-session forecast plus reserve exceeds cap")
    value = {
        "job_id": f"bl-f1-{platform}-" + ("probe-" if is_probe else "ruler-") + commit[:8],
        "commit": commit,
        "script": SCRIPT,
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_labels/requirements.txt",
            "F1_DEVICE": "cpu",
            "F1_COMMIT": commit,
            "F1_PLATFORM": platform,
            "PYTHONUNBUFFERED": "1",
            "F1_SESSION_SECONDS": str(int(timeout * 60)),
            "F1_FINALIZE_SECONDS": str(int(limits["finalize"])),
            **{
                key: "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "requirements": {
            "gpu": "L4" if platform == "l4" else "cpu",
            "allowed_gpus": ["L4"] if platform == "l4" else ["cpu"],
            "min_vcpu": 12 if platform == "l4" else 2,
            "est_minutes": STAGES[stage]["minutes"],
        },
        "timeout_minutes": timeout,
        "output_dir": str(output.resolve()),
        "inputs": inputs,
        "budget": {
            "unit_cap": limits["units"],
            "rate_units_per_hour": rate,
            "reserve_units": limits["reserve"],
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 64 * 1024**2,
        "source_paths": manifests.source_paths(),
        "required_exports": ["runtime-result.json", "session-budget.json"]
        + [
            f"units/config-{label}.json"
            for label in ("one", "half", "all")
            if is_probe or label not in saved
        ],
        "qualified_transport": qualified,
        "canary": {"template": True, "family": SCRIPT},
        "resume": {
            "unit_kind": "64-root-worker-schedule",
            "ledger_path": "inputs/f1-verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f1": {
            "bank_sha256": bank_sha,
            "stage": stage,
            "envelope": envelope(stage),
            "schedules": ["one", "half", "all"],
            "workers": "1,max(1,nproc//2),nproc",
            "series_base_units": 1.04,
            "series_effective_units": 2.08,
            "effective_ruling_ceiling_units": 2.5,
            "retry_policy": (
                "original plus at most two manual mechanical completions; "
                "all attempts share stage and series caps; "
                "verified stop and missing units required; "
                "no automatic job retry"
            ),
            "already_verified_schedules": sorted(saved),
        },
    }
    if is_probe:
        value["env"].update(
            F1_TRANSPORT_PROBE="1",
            SNAKE_COLAB_CANARY="1",
            SNAKE_COLAB_CANARY_PACKAGE="research/bl_labels",
        )
        value["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
        value.pop("resume")
        value["resume_exemption"] = {
            "kind": "short-canary",
            "reason": (
                "One frozen root through the actual F1/F0 entrypoint; "
                "excluded from timing comparison"
            ),
        }
    value = convert_public_checkpoints(value, repo=manifests.ROOT)
    assert_upload_budget(value, repo=manifests.ROOT)
    return value


def bind(template: Path, receipt: Path, out: Path) -> None:
    """Require the actual same-pin, same-hardware PASS before materializing main."""
    value = json.loads(template.read_text())
    from research.colab_cli.canary import check

    value["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": _sha(receipt)}}
    # The qualified verifier checks pin/script/hardware/input inventory identity,
    # completed workload evidence, verified exports, and backend release.
    check(value, manifests.ROOT)
    publish_json(out, value)


def build(commit: str, bank: Path, b8: Path, frp3: Path, out: Path, output_root: Path) -> None:
    manifests.require_source_pin(commit)
    load_bank(bank)
    qualified = qualification()
    from tools.ops.import_preflight import check

    preflight = check(
        manifests.ROOT,
        commit,
        manifests.source_paths(),
        "research/bl_labels/requirements.txt",
        [*manifests.RUNTIME_ROOTS, "research.bl_labels.f1_ruler"],
    )
    publish_json(out / "clean-import-preflight.json", preflight)
    publish_json(
        out / "budget-plan.json",
        {
            "schema": "bl-labels-f1-budget/v1",
            "stages": {k: envelope(k) for k in STAGES},
            "base_series_units": 1.04,
            "effective_series_units": 2.08,
            "base_series_reserve": 0.208,
            "effective_series_reserve": 0.416,
            "reserves_are_inside_caps": True,
            "effective_ruling_ceiling": 2.5,
            "max_completion_retries": 2,
            "retry_liability": "shared stage caps, never additive or reset",
            "basis": (
                "F0 observed setup116.8s and original qualified-session guards; "
                "source natural replay and full matrix charged to whole session; "
                "timing-only rate excludes separately reported replay setup"
            ),
            "qualified": qualified,
        },
    )
    for platform in ("cpu", "l4"):
        if platform == "l4" and not qualified["l4_ready"]:
            publish_json(
                out / "f1-l4-NOT_RUN.json",
                {"status": "NOT_RUN", "reason": "stall mitigation qualification absent"},
            )
            continue
        ledger = out / f"f1-{platform}-verified-ledger.json"
        publish_json(ledger, empty_ledger(_verify(bank), platform, source_commit=commit))
        for is_probe in (True, False):
            name = f"f1-{platform}-" + ("probe.json" if is_probe else "template.json")
            output = output_root / (platform + ("-probe" if is_probe else "-ruler"))
            publish_json(
                out / name,
                make_template(
                    commit,
                    output,
                    bank,
                    ledger,
                    b8,
                    frp3,
                    platform=platform,
                    qualified=qualified,
                    is_probe=is_probe,
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build")
    prepare.add_argument("--commit", required=True)
    for name in ("bank", "b8", "frp3", "out", "output-root"):
        prepare.add_argument("--" + name, type=Path, required=True)
    binder = sub.add_parser("bind")
    for name in ("template", "receipt", "out"):
        binder.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bind":
        bind(args.template, args.receipt, args.out)
    else:
        build(args.commit, args.bank, args.b8, args.frp3, args.out, args.output_root)


if __name__ == "__main__":
    main()
