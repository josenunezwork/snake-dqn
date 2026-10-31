"""Prepare commit-pinned Colab canary + unsealed/sealed main packets; never dispatches."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from research.colab_cli import canary
from research.watch6_alllen_screen import protocol as p
from research.watch6_alllen_screen import seed_audit
from research.watch6_alllen_screen.run import completed
from research.watch_replay.prepare import release_bundle
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs


def prepare(
    out: Path,
    commit: str,
    benchmark: Path,
    *,
    namespace: str = p.NAMESPACE,
    count: int = p.COUNT,
    horizon: int = p.HORIZON,
    workers: int = 4,
    canary_receipt: Path | None = None,
    ledger: Path | None = None,
    previous_receipts: list[Path] | None = None,
    skip_seed_audit: bool = False,
) -> dict[str, Any]:
    """Stage identical inputs for short canary and 48-pair main exploration run."""
    if not 1 <= count <= 64:
        raise ValueError("packet count must be 1..64")
    if workers not in (1, 2, 4, 6, 8):
        raise ValueError("workers must be in {1, 2, 4, 6, 8}")
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=p.ROOT, text=True
    )
    if len(commit) != 40 or not any(
        line.strip().startswith("origin/") for line in refs.splitlines()
    ):
        raise ValueError("requires full 40-char published origin/* commit pin")
    if subprocess.check_output(["git", "diff", commit, "--", *p.source_paths()], cwd=p.ROOT):
        raise ValueError("working tree executable bytes differ from commit pin")

    measured = json.loads(benchmark.read_text(encoding="utf-8"))
    if measured.get("commit") != commit:
        raise ValueError("Mac preflight benchmark must match current commit pin")

    if not skip_seed_audit and namespace == p.NAMESPACE:
        audit_report = seed_audit.audit(repo=p.ROOT)
        if audit_report.get("status") != "CLEAR":
            raise ValueError("seed disjointness audit did not return CLEAR")

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    contracts = {
        "main": {
            **p.allocation(namespace, count, horizon, audit_live_mirror=False),
            "commit": commit,
        },
        "canary": {
            **p.allocation(
                p.CANARY_NAMESPACE, p.CANARY_COUNT, p.CANARY_HORIZON, audit_live_mirror=True
            ),
            "commit": commit,
        },
    }

    previous_receipts = previous_receipts or []
    if len(previous_receipts) > 2 or bool(previous_receipts) != bool(ledger):
        raise ValueError(
            "one original plus at most two completion attempts; ledger requires prior receipts"
        )
    verified: list[dict[str, Any]] = []
    if ledger:
        verified = completed(ledger, contracts["main"])
        if len(verified) == count:
            raise ValueError("completion retry requires missing units")
        seen_receipts: set[str] = set()
        prior_exports: dict[int, dict[str, Any]] = {}
        for attempt, receipt_path in enumerate(previous_receipts, 1):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            former_path = receipt_path.parent / "manifest.json"
            former = json.loads(former_path.read_text(encoding="utf-8"))
            error = str(receipt.get("error", "")).upper()
            if (
                digest(receipt_path) in seen_receipts
                or digest(former_path) != receipt.get("manifest_sha256")
                or former.get("commit") != commit
                or former.get("watch6_alllen_screen", {}).get("allocation")
                != p.contract_identity(contracts["main"])
                or former.get("watch6_alllen_screen", {}).get("attempt") != attempt
                or receipt.get("stop_verified") is not True
                or receipt.get("status") == "PASS"
                or not any(
                    word in error
                    for word in (
                        "TIMEOUT",
                        "LEASE",
                        "TRANSFER",
                        "UPLOAD",
                        "STAGING",
                        "INTERRUPT",
                        "RUNTIME_LOST",
                    )
                )
            ):
                raise ValueError("completion requires exact stopped mechanically failed lineage")
            seen_receipts.add(digest(receipt_path))
            for row in receipt.get("exports", []):
                rel_name = row.get("path", "")
                if rel_name.startswith("units/unit-") and rel_name.endswith(".json"):
                    idx = int(Path(rel_name).stem.split("-", 1)[1])
                    if idx in prior_exports:
                        raise ValueError("duplicate completed unit across prior attempts")
                    prior_exports[idx] = {"sha256": row["sha256"], "size": row["size"]}
        if {
            r["index"]: {"sha256": r["sha256"], "size": r["size"]} for r in verified
        } != prior_exports:
            raise ValueError("ledger must contain exactly all verified prior receipt units")

    inputs: list[Input] = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            (
                json.loads(ledger.read_text(encoding="utf-8"))
                if label == "main" and ledger is not None
                else empty_ledger(contract["namespace"], p.UNIT_KIND, p.contract_identity(contract))
            ),
        )
        inputs.extend(
            [
                Input(out / f"{label}-contract.json", f"{label}-contract.json"),
                Input(out / f"{label}-ledger.json", f"{label}-ledger.json"),
            ]
        )

    saved_dir = p.ROOT.parent / "snake-dqn" / "saved_snakes"
    if not saved_dir.is_dir():
        saved_dir = p.ROOT / "saved_snakes"
    for ckpt_name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth"):
        inputs.append(Input(saved_dir / ckpt_name, f"checkpoints/{ckpt_name}"))

    pins = json.loads(
        (p.ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    staged = stage_inputs(out / "staged", inputs)

    singleton = float(
        statistics.median(
            float(item["seconds"]) / max(1, int(item["branches"]))
            for item in measured["branch_timings"]
        )
    )
    # 3-arm H6000 unit forecast: bare (0 roots) + L2' (~420 roots at L>=600) +
    # L>=1 (~720 roots at L<600 weighted 0.65x L=900 cost + ~420 roots at L>=600),
    # ~1.2 branches/root via exact_stop, plus main-loop GridBatchSim step time.
    horizon_scale = horizon / 6000.0
    expected_roots_3arm = (420.0 + 1140.0) * horizon_scale
    weighted_l900_equiv_roots = (840.0 + 720.0 * 0.65) * horizon_scale
    expected_branches_3arm = expected_roots_3arm * 1.2
    mac_unit_seconds = weighted_l900_equiv_roots * 1.2 * singleton + 90.0 * horizon_scale
    colab_unit_seconds = mac_unit_seconds * 1.35 * 1.15
    remaining_units = count - len(verified)
    waves = math.ceil(max(1, remaining_units) / workers)
    base_minutes = max(360, math.ceil(colab_unit_seconds * waves / 60.0 + 25.0))
    if base_minutes > 660:
        raise ValueError("base forecast exceeds 11h (22h effective 2x cap); split packet")

    forecast = {
        "mac_h300_l900_singleton_median_seconds": singleton,
        "expected_roots_per_3arm_unit": expected_roots_3arm,
        "expected_branches_per_3arm_unit": expected_branches_3arm,
        "cloud_slowdown_projection": 1.35,
        "contention_projection": 1.15,
        "workers": workers,
        "remaining_units": remaining_units,
        "base_minutes": base_minutes,
        "effective_timeout_minutes": 2 * base_minutes,
        "expected_l4_hours_central": round(colab_unit_seconds * waves / 3600.0, 2),
        "owner_2x_rule": "effective timeout, unit_cap, and reserve_units are 2x base forecast",
    }
    publish_json(out / "forecast.json", forecast)

    rate = 1.54
    for label in ("canary", "main"):
        is_probe = label == "canary"
        base = 15 if is_probe else base_minutes
        base_units = round(base / 60.0 * rate, 4)
        base_reserve = round(base_units * 0.2, 4)
        manifest: dict[str, Any] = {
            "job_id": f"watch6-alllen-{label}-{out.name}",
            "commit": commit,
            "script": p.SCRIPT,
            "source_paths": p.source_paths(),
            "device": "cpu",
            "lease_seconds": 300,
            "timeout_minutes": 2 * base,
            "output_dir": str(out / f"output-{label}"),
            "max_export_bytes": 32 * 1024**2,
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": base,
            },
            "budget": {
                "unit_cap": round(2.0 * base_units, 4),
                "reserve_units": round(2.0 * base_reserve, 4),
                "rate_units_per_hour": rate,
                "other_assignments": {},
            },
            "watch6_alllen_screen": {
                "lane": p.LANE,
                "decision_rule": p.DECISION_RULE,
                "allocation": p.contract_identity(contracts[label]),
                "attempt": len(previous_receipts) + 1,
                "base_minutes": base,
                "effective_timeout_minutes": 2 * base,
                "base_unit_cap": base_units,
                "effective_unit_cap": round(2.0 * base_units, 4),
                "base_reserve_units": base_reserve,
                "effective_reserve_units": round(2.0 * base_reserve, 4),
                "forecast": forecast,
            },
            "resume": {
                "unit_kind": p.UNIT_KIND,
                "ledger_path": f"inputs/{label}-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "required_exports": ["units/operational.json", "units/contract.json"],
            "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in staged],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH6_ALLLEN_CONTRACT": f"{label}-contract.json",
                "WATCH6_ALLLEN_LEDGER": f"{label}-ledger.json",
                "WATCH6_ALLLEN_WORKERS": "1" if is_probe else str(workers),
            },
        }
        if is_probe:
            manifest["required_exports"].extend(["units/unit-000000.json", "branches.json"])
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1",
                SNAKE_COLAB_CANARY_PACKAGE=p.PACKAGE,
            )
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(manifest),
            }
            (out / "canary.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            # Always write main-unsealed.json so ready_queue --prepare-command can seal
            # main.json create-only once the canary receipt exists.
            (out / "main-unsealed.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if canary_receipt is not None:
                sealed = dict(manifest)
                sealed["canary"] = {
                    "schema": canary.SCHEMA,
                    "receipt": {
                        "path": str(canary_receipt.resolve()),
                        "sha256": digest(canary_receipt),
                    },
                }
                (out / "main.json").write_text(
                    json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--namespace", default=p.NAMESPACE)
    parser.add_argument("--count", type=int, default=p.COUNT)
    parser.add_argument("--horizon", type=int, default=p.HORIZON)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.out,
                args.commit,
                args.benchmark,
                namespace=args.namespace,
                count=args.count,
                horizon=args.horizon,
                workers=args.workers,
                canary_receipt=args.canary_receipt,
                ledger=args.ledger,
                previous_receipts=args.previous_receipts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
