"""Prepare exact hashed inputs/manifests; no allocation, account access or dispatch."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import completed, ledger_identity
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs


def prepare(
    out: Path,
    benchmark: Path,
    commit: str,
    approval: Path | None = None,
    ledger: Path | None = None,
    attempt: int = 1,
    previous_receipts: list[Path] | None = None,
) -> dict:
    """Bind published branch pin; conservative all-trigger compute envelope."""
    subprocess.run(["git", "cat-file", "-e", commit], cwd=p.ROOT, check=True)
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=p.ROOT, text=True
    )
    if "origin/astra-watch6-search-efficacy" not in refs:
        raise ValueError("pin must be pushed on requested branch")
    if subprocess.check_output(["git", "diff", commit, "--", *p.source_paths()], cwd=p.ROOT):
        raise ValueError("local executable source differs from pin")
    measured = json.loads(benchmark.read_text())
    if measured["science"] != p.identity():
        raise ValueError("stale Mac preflight")
    if type(attempt) is not int or attempt not in (1, 2, 3):
        raise ValueError("one original plus at most two completion attempts")
    previous_receipts = previous_receipts or []
    if attempt == 1 and (ledger is not None or previous_receipts):
        raise ValueError("first attempt must have empty ledger and no prior receipts")
    if attempt > 1:
        if ledger is None or len(previous_receipts) != attempt - 1:
            raise ValueError("completion needs verified ledger and all prior stopped attempts")
        verified = completed(ledger)
        if len(verified) == p.COUNT:
            raise ValueError("all units verified; never recompute")
        if len({digest(path) for path in previous_receipts}) != len(previous_receipts):
            raise ValueError("duplicate previous-attempt receipt")
        for previous_attempt, receipt_path in enumerate(previous_receipts, 1):
            receipt = json.loads(receipt_path.read_text())
            former_path = receipt_path.parent / "manifest.json"
            former = json.loads(former_path.read_text())
            if (
                digest(former_path) != receipt.get("manifest_sha256")
                or former.get("commit") != commit
                or former.get("watch6_efficacy", {}).get("science") != p.identity()
                or former.get("watch6_efficacy", {}).get("attempt") != previous_attempt
            ):
                raise ValueError("foreign source or completion-attempt lineage")
            error = str(receipt.get("error", "")).upper()
            mechanical = any(
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
            if (
                receipt.get("stop_verified") is not True
                or receipt.get("status") == "PASS"
                or not mechanical
            ):
                raise ValueError("completion requires stopped, verified mechanical failure")
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "verified-ledger.json"
    publish_json(
        ledger_path,
        (
            json.loads(ledger.read_text())
            if ledger
            else empty_ledger(p.NAMESPACE, "watch6", ledger_identity())
        ),
    )
    pending = out / "fable-approval.json"
    publish_json(
        pending,
        {
            "status": "PENDING",
            "judge": "Fable",
            "science": p.identity(),
            "registration_sha256": digest(Path(__file__).with_name("REGISTRATION.md")),
        },
    )
    saved = p.ROOT.parent / "snake-dqn/saved_snakes"
    contract = out / "runtime-contract.json"
    publish_json(contract, {"science": p.identity(), "allocation": p.allocation()})
    inputs = [
        Input(contract, "runtime-contract.json"),
        Input(ledger_path, "verified-ledger.json"),
        Input(approval or pending, "fable-approval.json"),
        Input(saved / "b8_s3_t7_20261016.pth", "checkpoints/b8_s3_t7_20261016.pth"),
        Input(
            saved / "frp3_m3_s12_u60000_20261005.pth",
            "checkpoints/frp3_m3_s12_u60000_20261005.pth",
        ),
    ]
    staged = stage_inputs(out / "staged", inputs)
    max_singleton_seconds = max(r["seconds"] / r["branches"] for r in measured["branch_timings"])
    # Fixed estimate: 2880 eligible roots/session, 1.2 complete branches/root forecast, not a bound.
    # Mac-to-Colab per-core slowdown3 and W4 host-contention multiplier1.5.
    # Setup/export allowance20min; no doubling measured rates or estimates.
    seconds = 2880 * 1.2 * max_singleton_seconds * 3 * 1.5
    base_minutes = math.ceil(seconds / 60 + 20)
    if base_minutes > 720:
        raise ValueError(f"NOT_READY forecast base={base_minutes} min >12h; no science shrunk")
    rate = 2.5  # conservative L4 planning rate; operator calibrates before admission
    forecast = {
        "base_minutes_per_shard": base_minutes,
        "effective_minutes": 2 * base_minutes,
        "max_mac_singleton_seconds": max_singleton_seconds,
        "cloud_slowdown_projection": 3,
        "four_worker_contention_projection": 1.5,
        "roots_per_session_upper_bound": 2880,
        "branches_per_root_forecast": 1.2,
        "branches_per_root_physical_maximum": 6,
        "workers": 4,
        "shards": 6,
        "pairs_per_shard": 4,
        "aggregate_expected_runtime_hours": base_minutes * 6 / 60,
        "status": "CONSERVATIVE_PROJECTION_NOT_MEASURED_COLAB_SPEED",
    }
    publish_json(out / "forecast.json", forecast)
    paths = p.source_paths()
    common = {
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": paths,
        "device": "cpu",
        "lease_seconds": 300,
        "max_export_bytes": 4 * 1024**2,
        "required_exports": ["efficacy/operational.json"],
        "resume": {
            "unit_kind": "paired-session",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "inputs": [{k: r[k] for k in ("path", "dest", "sha256")} for r in staged],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            "SNAKE_DQN_DEVICE": "cpu",
            "WATCH6_SCIENCE": p.identity(),
            "PYTHONUNBUFFERED": "1",
            **{
                k: "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                )
            },
        },
        "watch6_efficacy": {
            "science": p.identity(),
            "allocation": p.allocation(),
            "attempt": attempt,
            "forecast_sha256": digest(out / "forecast.json"),
        },
    }
    for label, shard, probe in [("canary", 0, True), *[(f"shard-{i}", i, False) for i in range(6)]]:
        value = json.loads(json.dumps(common))
        base = 15 if probe else base_minutes
        unit = base / 60 * rate
        value.update(
            job_id=f"watch6-search-{label}-{out.name}",
            output_dir=str(out / f"output-{label}"),
            timeout_minutes=2 * base,
            requirements={"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": base},
            budget={
                "unit_cap": 2 * unit,
                "rate_units_per_hour": rate,
                "reserve_units": 2 * unit * 0.2,
                "other_assignments": {},
            },
        )
        value["watch6_efficacy"].update(
            base_minutes=base, base_unit_cap=unit, base_reserve_units=unit * 0.2
        )
        value["env"].update(WATCH6_SHARD=str(shard), WATCH6_FIXTURE="1" if probe else "0")
        if probe:
            value["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch6_search_efficacy"
            )
            value["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(value),
            }
        # The uploaded contract and complete input inventory are identical for
        # canary and all shards; per-shard controls are pinned manifest env.
        (out / f"{label}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.out.resolve(),
                args.benchmark,
                args.commit,
                args.approval,
                args.ledger,
                args.attempt,
                args.previous_receipts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
