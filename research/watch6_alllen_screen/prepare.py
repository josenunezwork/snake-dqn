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
from research.watch6_alllen_screen.run import completed, shard_unverified_rows
from research.watch6_alllen_screen.uploads import (
    MAX_FILE_BYTES,
    METADATA,
    check_uploads,
    pack_inputs,
)
from research.watch_replay.prepare import release_bundle
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

# Measured packet-02 L4 W4 wall rate: 9 completed 3-arm H=6000 units in 5.7 h across 4 workers
# (~2.0 h = 7,200 s per 3-arm H=6000 unit per worker).
MEASURED_COLAB_W4_UNIT_SECONDS_H6000 = 7200.0

# Empirical 12-vCPU Colab Xeon scaling (watch6-canary-transport01, 6 physical cores / 12 threads):
# W=4 -> 4.0180 s/root; W=6 -> 4.1444 s/root (+3.15% contention, 1.45x machine throughput);
# W=8 -> 5.3798 s/root (+33.89% SMT oversubscription slowdown).
CONTENTION_FACTORS_12VCPU: dict[int, float] = {
    1: 1.0,
    2: 1.0,
    4: 1.0,
    6: round(4.1444 / 4.0180, 6),
    8: round(5.3798 / 4.0180, 6),
}

GPU_RATES: dict[str, float] = {
    "L4": 1.54,
    "A100": 8.0,
}


def compute_forecast(
    measured: dict[str, Any],
    *,
    assigned_units: int,
    total_remaining_units: int,
    horizon: int,
    workers: int,
    gpu: str,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Compute measured ~2 h/unit/worker Colab forecast with 12-vCPU contention scaling."""
    if workers not in CONTENTION_FACTORS_12VCPU:
        raise ValueError("workers must be in {1, 2, 4, 6, 8}")
    if gpu not in GPU_RATES:
        raise ValueError(f"unsupported gpu: {gpu}")
    singleton = float(
        statistics.median(
            float(item["seconds"]) / max(1, int(item["branches"]))
            for item in measured["branch_timings"]
        )
    )
    horizon_scale = horizon / 6000.0
    contention = CONTENTION_FACTORS_12VCPU[workers]
    colab_unit_seconds = MEASURED_COLAB_W4_UNIT_SECONDS_H6000 * horizon_scale * contention
    waves = math.ceil(assigned_units / workers) if assigned_units > 0 else 0
    base_minutes = (
        30 if waves == 0 else max(30, math.ceil(colab_unit_seconds * waves / 60.0 + 25.0))
    )
    if base_minutes > 660:
        raise ValueError("base forecast exceeds 11h (22h effective 2x cap); split across shards")
    central_hours = round(colab_unit_seconds * waves / 3600.0, 2)
    return {
        "basis": (
            "measured Colab packet-02 L4 W4 rate (9/48 3-arm H=6000 units in 5.7 h = ~7200 s "
            "per unit per worker) scaled by 12-vCPU Colab contention factor"
        ),
        "mac_h300_l900_singleton_median_seconds": singleton,
        "measured_colab_w4_h6000_unit_seconds": MEASURED_COLAB_W4_UNIT_SECONDS_H6000,
        "contention_factor_vs_w4": contention,
        "projected_unit_seconds_per_worker": round(colab_unit_seconds, 2),
        "gpu": gpu,
        "rate_units_per_hour": GPU_RATES[gpu],
        "workers": workers,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "total_remaining_units": total_remaining_units,
        "assigned_units": assigned_units,
        "remaining_units": assigned_units,
        "waves": waves,
        "base_minutes": base_minutes,
        "effective_timeout_minutes": 2 * base_minutes,
        "expected_runtime_hours_central": central_hours,
        "owner_2x_rule": "effective timeout, unit_cap, and reserve_units are 2x base forecast",
    }


def build_manifest(
    *,
    job_id: str,
    commit: str,
    output_dir: Path,
    gpu: str,
    base_minutes: int,
    contract: dict[str, Any],
    attempt: int,
    forecast: dict[str, Any],
    staged_inputs: list[dict[str, Any]],
    metadata_sha256: str,
    label: str,
    workers: int,
    is_probe: bool,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Construct one validated canary or main shard manifest."""
    rate = GPU_RATES[gpu]
    base = 15 if is_probe else base_minutes
    base_units = round(base / 60.0 * rate, 4)
    base_reserve = round(base_units * 0.2, 4)
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": p.source_paths(),
        "device": "cpu",
        "lease_seconds": 300,
        "timeout_minutes": 2 * base,
        "output_dir": str(output_dir),
        "max_export_bytes": MAX_FILE_BYTES,
        "requirements": {
            "gpu": gpu,
            "allowed_gpus": [gpu],
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
            "allocation": p.contract_identity(contract),
            "attempt": attempt,
            "shard_index": 0 if is_probe else shard_index,
            "shard_count": 1 if is_probe else shard_count,
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
        "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in staged_inputs],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "SNAKE_DQN_DEVICE": "cpu",
            "WATCH6_ALLLEN_CONTRACT": f"{label}-contract.json",
            "WATCH6_ALLLEN_LEDGER": f"{label}-ledger.json",
            "WATCH6_ALLLEN_METADATA_SHA256": metadata_sha256,
            "WATCH6_ALLLEN_SHARD_INDEX": "0" if is_probe else str(shard_index),
            "WATCH6_ALLLEN_SHARD_COUNT": "1" if is_probe else str(shard_count),
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
    return manifest


def prepare(
    out: Path,
    commit: str,
    benchmark: Path,
    *,
    namespace: str = p.NAMESPACE,
    count: int = p.COUNT,
    horizon: int = p.HORIZON,
    gpu: str = "L4",
    workers: int = 4,
    shard_index: int = 0,
    shard_count: int = 1,
    canary_receipt: Path | None = None,
    ledger: Path | None = None,
    previous_receipts: list[Path] | None = None,
    skip_seed_audit: bool = False,
) -> dict[str, Any]:
    """Stage packed inputs (<=8 transfers, <=60 s) for canary and sharded/single main run."""
    if not 1 <= count <= 64:
        raise ValueError("packet count must be 1..64")
    if workers not in CONTENTION_FACTORS_12VCPU:
        raise ValueError("workers must be in {1, 2, 4, 6, 8}")
    if gpu not in GPU_RATES:
        raise ValueError(f"gpu must be one of {sorted(GPU_RATES)}")
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

    pins = json.loads(
        (p.ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    inputs.append(Input(p.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json"))
    staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)
    metadata_sha256 = next(row["sha256"] for row in staged if row["dest"] == METADATA)

    skip_set = {int(row["index"]) for row in verified}
    assigned_rows = shard_unverified_rows(
        contracts["main"], skip_set, shard_index=shard_index, shard_count=shard_count
    )
    forecast = compute_forecast(
        measured,
        assigned_units=len(assigned_rows),
        total_remaining_units=count - len(skip_set),
        horizon=horizon,
        workers=workers,
        gpu=gpu,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    publish_json(out / "forecast.json", forecast)

    for label in ("canary", "main"):
        is_probe = label == "canary"
        manifest = build_manifest(
            job_id=f"watch6-alllen-{label}-{out.name}",
            commit=commit,
            output_dir=out / f"output-{label}",
            gpu=gpu,
            base_minutes=int(forecast["base_minutes"]),
            contract=contracts[label],
            attempt=len(previous_receipts) + 1,
            forecast=forecast,
            staged_inputs=staged,
            metadata_sha256=metadata_sha256,
            label=label,
            workers=workers,
            is_probe=is_probe,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        upload_report = check_uploads(manifest)
        publish_json(out / f"{label}-upload-preflight.json", upload_report)
        if is_probe:
            (out / "canary.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
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
                canary.check(sealed, p.ROOT)
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
    parser.add_argument("--gpu", default="L4", choices=sorted(GPU_RATES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
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
                gpu=args.gpu,
                workers=args.workers,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                canary_receipt=args.canary_receipt,
                ledger=args.ledger,
                previous_receipts=args.previous_receipts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
