"""Prepare hashed inputs and manifests for Watch6-L1-V1 (A100+L4 canaries + 18 shards)."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

from research.colab_cli import canary
from research.t34_exit_b8 import public_inputs
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.admission import admit, completed_ledger
from research.watch6_l1_confirm.costs import (
    BASE_MINUTES_BY_GPU,
    CANARY_BASE_MINUTES,
    PRICED_CEILING_RATES_U_PER_H,
    empirical_forecast,
    verify_report_evidence,
)
from research.watch6_l1_confirm.mirror import validate as validate_mirror
from research.watch6_l1_confirm.run import ledger_identity
from research.watch6_l1_confirm.uploads import CHECKPOINTS, METADATA, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

ALLOWED_REMOTE_BRANCHES = (
    "origin/main",
    "origin/argon-l1v1-pkg-1101",
)

# Representative shard indices used for the two hardware-tier canaries (0 = A100, 12 = L4).
CANARY_SHARD_BY_GPU: dict[str, int] = {"A100": 0, "L4": p.A100_SHARDS}


def _require_published_pin(commit: str) -> None:
    subprocess.run(["git", "cat-file", "-e", commit], cwd=p.ROOT, check=True)
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=p.ROOT, text=True
    )
    if not any(branch in refs for branch in ALLOWED_REMOTE_BRANCHES):
        raise ValueError(
            f"commit {commit} must be pushed to origin/main or origin/argon-l1v1-pkg-1101"
        )
    if subprocess.check_output(["git", "diff", commit, "--", *p.source_paths()], cwd=p.ROOT):
        raise ValueError("local Watch6-L1-V1 executable source closure differs from pinned commit")


def prepare(
    out: Path,
    benchmark: Path,
    commit: str,
    mirror_receipt: Path,
    empirical_costs: Path,
    *,
    canary_receipts_by_gpu: dict[str, Path] | None = None,
    main_shards: bool = True,
    pending_canary: bool = False,
    ledger: Path | None = None,
    target_shards: list[int] | None = None,
    shard_meta: dict[int, dict] | None = None,
    reuse_staged_inputs: list[dict] | None = None,
    judge: str = "Astra",
) -> dict:
    """Prepare Watch6-L1-V1 A100/L4 canaries and/or 18 shard manifests without touching Colab."""
    canary_receipts_by_gpu = dict(canary_receipts_by_gpu or {})
    if judge not in ("Astra", "Fable"):
        raise ValueError("judge must be Astra or Fable")

    selected_shards = sorted(target_shards) if target_shards is not None else list(range(p.SHARDS))
    if any(s not in range(p.SHARDS) for s in selected_shards):
        raise ValueError(f"target_shards out of range 0..{p.SHARDS - 1}")
    required_gpus = (
        sorted({p.shard_spec(s)["gpu"] for s in selected_shards})
        if main_shards
        else list(p.TARGET_GPUS)
    )

    if main_shards and not pending_canary:
        missing_gpus = [g for g in required_gpus if g not in canary_receipts_by_gpu]
        if missing_gpus:
            raise ValueError(
                f"main preparation requires passing same-pin canary receipt(s) for {missing_gpus} "
                "or --pending-canary"
            )
    if pending_canary and (not main_shards or canary_receipts_by_gpu):
        raise ValueError("pending_canary requires main_shards and no pre-existing canary_receipts")

    validate_mirror(json.loads(mirror_receipt.read_text(encoding="utf-8")))
    _require_published_pin(commit)

    measured = json.loads(benchmark.read_text(encoding="utf-8"))
    if measured.get("science") != p.identity():
        raise ValueError("stale Mac preflight benchmark.json")

    cost_report = json.loads(empirical_costs.read_text(encoding="utf-8"))
    verify_report_evidence(cost_report)
    forecast = empirical_forecast(cost_report)

    shard_meta = shard_meta or {}
    if ledger is not None:
        verified = completed_ledger(ledger)
        if len(verified) == p.COUNT:
            raise ValueError("all 96 Watch6-L1-V1 worlds already verified; never recompute")

    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "forecast.json", forecast)

    if reuse_staged_inputs is not None:
        staged = reuse_staged_inputs
    else:
        ledger_path = out / "verified-ledger.json"
        publish_json(
            ledger_path,
            (
                json.loads(ledger.read_text(encoding="utf-8"))
                if ledger is not None
                else empty_ledger(p.NAMESPACE, "watch6", ledger_identity())
            ),
        )
        protocol_binding = out / "judge-approval.json"
        publish_json(
            protocol_binding,
            {
                "schema": "watch6-l1-protocol-binding/v1",
                "implementer": "Argon",
                "judge": judge,
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "registration_sha256": digest(p.registration()),
            },
        )
        contract = out / "runtime-contract.json"
        publish_json(
            contract,
            {
                "schema": "watch6-l1-runtime-contract/v1",
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "primary_allocation": p.allocation(),
            },
        )
        saved = p.ROOT.parent / "snake-dqn/saved_snakes"
        if not saved.is_dir():
            saved = p.ROOT / "saved_snakes"
        pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text("utf-8"))
        for ckpt_name, (pin_key, expected_size) in CHECKPOINTS.items():
            local_ckpt = saved / ckpt_name
            if (
                not local_ckpt.is_file()
                or local_ckpt.stat().st_size != expected_size
                or digest(local_ckpt) != pins["checkpoints"][pin_key]["checkpoint_sha256"]
            ):
                raise ValueError(f"local checkpoint differs from released pin: {local_ckpt}")

        public_catalog = p.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
        public_inputs._load(public_catalog)

        release_bundle = out / "release-evidence.tar.gz"
        with (
            release_bundle.open("xb") as raw_bundle,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw_bundle, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for name, pin in sorted(pins["checkpoints"].items()):
                strict_receipt = pin["strict_receipt"]
                for ref in strict_receipt.values():
                    if isinstance(ref, dict):
                        src_path = Path(strict_receipt["root"]) / ref["path"]
                        if digest(src_path) != ref["sha256"]:
                            raise ValueError("legacy release evidence digest mismatch")
                        data = src_path.read_bytes()
                        member = tarfile.TarInfo(name + "/" + ref["path"])
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))

        inputs = [
            Input(contract, "runtime-contract.json"),
            Input(ledger_path, "verified-ledger.json"),
            Input(protocol_binding, "judge-approval.json"),
            Input(mirror_receipt, "mac-mirror.json"),
            Input(empirical_costs, "cost-evidence.json"),
            Input(release_bundle, "release-evidence.tar.gz"),
            Input(public_catalog, "public-inputs.json"),
        ]
        staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)

    metadata_sha256 = next(row["sha256"] for row in staged if row["dest"] == METADATA)
    common = {
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": p.source_paths(),
        "device": "cpu",
        "lease_seconds": 600,
        "max_export_bytes": 4 * 1024**2,
        "required_exports": ["efficacy/operational.json", "performance.json"],
        "resume": {
            "unit_kind": "paired-world",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "inputs": [{k: r[k] for k in ("path", "dest", "sha256")} for r in staged],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            "SNAKE_DQN_DEVICE": "cpu",
            "SNAKE_WATCH_SOURCE": "live",
            "WATCH6_SCIENCE": p.identity(),
            "WATCH6_METADATA_SHA256": metadata_sha256,
            "PYTHONDONTWRITEBYTECODE": "1",
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
        "watch6_l1_v1": {
            "namespace": p.NAMESPACE,
            "judge": judge,
            "implementer": "Argon",
            "independent_implementation_review": True,
            "science": p.identity(),
            "registration_sha256": digest(p.registration()),
            "primary_allocation": p.allocation(),
            "forecast_sha256": digest(out / "forecast.json"),
        },
    }

    jobs: list[tuple[str, int, bool]] = []
    if not main_shards or pending_canary:
        for gpu in required_gpus:
            jobs.append((f"canary-{gpu.lower()}", CANARY_SHARD_BY_GPU[gpu], True))
    if main_shards:
        jobs.extend((f"shard-{s:02d}", s, False) for s in selected_shards)

    for label, shard, probe in jobs:
        spec = p.shard_spec(shard)
        gpu = spec["gpu"]
        workers = spec["workers"]
        rate = PRICED_CEILING_RATES_U_PER_H[gpu]
        base = CANARY_BASE_MINUTES if probe else BASE_MINUTES_BY_GPU[gpu]
        unit = base / 60.0 * rate
        meta = shard_meta.get(shard, {})
        phase = "canary" if probe else "primary"
        prestart_used = 0 if probe else int(meta.get("prestart_retries_used", 0))
        compute_attempt = 0 if probe else int(meta.get("compute_completion_attempt", 0))
        total_dispatches = (
            1
            if probe
            else int(meta.get("total_dispatches_for_shard", 1 + prestart_used + compute_attempt))
        )
        prior_receipts = [] if probe else list(meta.get("prior_receipts", []))

        value = json.loads(json.dumps(common))
        value.update(
            job_id=f"watch6-l1-{label}-{out.name}",
            output_dir=str(out / f"output-{label}"),
            timeout_minutes=2 * base,
            requirements={
                "gpu": gpu,
                "allowed_gpus": [gpu],
                "min_vcpu": 12,
                "est_minutes": base,
            },
            budget={
                "unit_cap": 2.0 * unit,
                "rate_units_per_hour": rate,
                "reserve_units": 2.0 * unit * 0.2,
                "other_assignments": {},
            },
        )
        value["watch6_l1_v1"].update(
            shard=shard,
            gpu=gpu,
            workers=workers,
            indices=spec["indices"],
            phase=phase,
            prestart_retries_used=prestart_used,
            compute_completion_attempt=compute_attempt,
            total_dispatches_for_shard=total_dispatches,
            prior_receipts=prior_receipts,
            base_minutes=base,
            base_unit_cap=unit,
            base_reserve_units=unit * 0.2,
        )
        value["env"].update(
            WATCH6_SHARD=str(shard),
            WATCH6_FIXTURE="1" if probe else "0",
            WATCH6_WORKERS=str(workers),
            WATCH6_SHARDS=str(p.SHARDS),
            WATCH6_BASE_MINUTES=str(base),
        )
        if probe:
            value["env"].update(
                SNAKE_COLAB_CANARY="1",
                SNAKE_COLAB_CANARY_PACKAGE="research/watch6_l1_confirm",
            )
            value["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(value),
            }
        else:
            qualification = (
                out / f"output-canary-{gpu.lower()}/receipt.json"
                if pending_canary
                else canary_receipts_by_gpu[gpu]
            )
            value["canary"] = {
                "schema": canary.SCHEMA,
                "receipt": {
                    "path": str(qualification.resolve()),
                    "sha256": "0" * 64 if pending_canary else digest(qualification),
                },
            }
            value["env"]["WATCH6_MEASUREMENTS"] = json.dumps(cost_report, sort_keys=True)
            if pending_canary:
                value["pending_qualification"] = True
            else:
                canary.check(value, p.ROOT)

        upload_report = admit(value, pending_qualification=pending_canary and not probe)
        publish_json(out / f"{label}-upload-preflight.json", upload_report)
        suffix = ".pending.json" if pending_canary and not probe else ".json"
        (out / f"{label}{suffix}").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mirror-receipt", type=Path, required=True)
    parser.add_argument("--empirical-costs", type=Path, required=True)
    parser.add_argument("--canary-a100-receipt", type=Path, default=None)
    parser.add_argument("--canary-l4-receipt", type=Path, default=None)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--pending-canary", action="store_true")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()

    by_gpu: dict[str, Path] = {}
    if args.canary_a100_receipt is not None:
        by_gpu["A100"] = args.canary_a100_receipt.resolve()
    if args.canary_l4_receipt is not None:
        by_gpu["L4"] = args.canary_l4_receipt.resolve()

    result = prepare(
        args.out.resolve(),
        args.benchmark.resolve(),
        args.commit,
        args.mirror_receipt.resolve(),
        args.empirical_costs.resolve(),
        canary_receipts_by_gpu=by_gpu,
        main_shards=not args.canary_only,
        pending_canary=args.pending_canary,
        ledger=args.ledger.resolve() if args.ledger else None,
        judge=args.judge,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
