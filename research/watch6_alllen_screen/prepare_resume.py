"""Dispatch-time ledger rebuild + 3-shard A100 resume packet preparation for Watch6 L>=1 screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.watch6_alllen_screen import protocol as p
from research.watch6_alllen_screen import resume, seed_audit
from research.watch6_alllen_screen.prepare import GPU_RATES, build_manifest, compute_forecast
from research.watch6_alllen_screen.run import shard_unverified_rows
from research.watch6_alllen_screen.uploads import METADATA, check_uploads, pack_inputs
from research.watch_replay.prepare import release_bundle
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

SIMULATION_EQUIVALENCE_PATHS: tuple[str, ...] = (
    "research/watch6_alllen_screen/protocol.py",
    "research/watch6_alllen_screen/engine.py",
    "research/watch_replay/engine.py",
    "research/watch_replay/product.py",
    "research/bl_served_search/exact_stop.py",
    "web/backend/served_checkpoint_pins.json",
)


def init_resume_dir(out_root: Path, *, shards: int = 3, gpu: str = "A100") -> dict[str, Any]:
    """Pre-create packet directories and GPU template hints so ready_queue sees A100 at enqueue."""
    if not 1 <= shards <= 3:
        raise ValueError("shards must be in 1..3")
    if gpu not in GPU_RATES:
        raise ValueError(f"unsupported gpu: {gpu}")
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    template = {
        "requirements": {
            "gpu": gpu,
            "allowed_gpus": [gpu],
            "min_vcpu": 12,
        }
    }
    canary_dir = out_root / "packet-canary"
    canary_dir.mkdir()
    (canary_dir / "canary-template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shard_dirs: list[str] = []
    for idx in range(shards):
        sdir = out_root / f"packet-shard{idx}"
        sdir.mkdir()
        (sdir / "main-template.json").write_text(
            json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shard_dirs.append(str(sdir))
    return {
        "out_root": str(out_root),
        "gpu": gpu,
        "shards": shards,
        "canary_dir": str(canary_dir),
        "shard_dirs": shard_dirs,
    }


def _validate_stopped_prior_outputs(
    prior_outputs: list[Path],
    *,
    commit: str,
    repo: Path,
    check_git_equivalence: bool = True,
) -> tuple[set[str], list[Path], list[dict[str, Any]]]:
    """Verify every prior attempt stopped cleanly and used identical simulation/protocol code."""
    if not prior_outputs:
        raise ValueError("at least one --prior-output directory is required")
    allowed_commits: set[str] = {commit}
    export_roots: list[Path] = []
    lineage: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    for prior_out in prior_outputs:
        prior_out = prior_out.resolve()
        receipt_path = prior_out / "receipt.json"
        manifest_path = prior_out / "manifest.json"
        if not receipt_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"prior attempt lacks receipt.json or manifest.json: {prior_out}")
        receipt_sha = digest(receipt_path)
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate prior receipt supplied: {receipt_path}")
        seen_receipts.add(receipt_sha)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if digest(manifest_path) != receipt.get("manifest_sha256"):
            raise ValueError(f"prior receipt manifest_sha256 mismatch: {prior_out}")
        if receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None:
            raise ValueError(f"prior attempt has not completed verified stop: {receipt_path}")
        if receipt.get("status") == "PASS":
            raise ValueError(f"prior attempt already completed PASS: {receipt_path}")
        prior_commit = str(manifest.get("commit", ""))
        if len(prior_commit) != 40:
            raise ValueError(f"invalid prior commit in {manifest_path}")
        if check_git_equivalence and prior_commit != commit:
            diff = subprocess.check_output(
                ["git", "diff", prior_commit, commit, "--", *SIMULATION_EQUIVALENCE_PATHS],
                cwd=repo,
            )
            if diff:
                raise ValueError(
                    f"simulation/protocol bytes changed between {prior_commit[:8]} and {commit[:8]}"
                )
        allowed_commits.add(prior_commit)
        export_roots.append(prior_out / "exports" / "units")
        lineage.append(
            {
                "output_dir": str(prior_out),
                "job_id": manifest.get("job_id"),
                "commit": prior_commit,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "stop_verified": True,
            }
        )
    return allowed_commits, export_roots, lineage


def build_resume_packets(
    out_root: Path,
    *,
    commit: str,
    benchmark: Path,
    prior_outputs: list[Path],
    namespace: str = p.NAMESPACE,
    count: int = p.COUNT,
    horizon: int = p.HORIZON,
    shards: int = 3,
    gpu: str = "A100",
    workers: int = 6,
    job_prefix: str = "watch6-alllen-a100",
    skip_seed_audit: bool = False,
    check_git_equivalence: bool = True,
) -> dict[str, Any]:
    """Rebuild verified ledger at dispatch time and emit 1 A100 canary + 3 A100 shard packets."""
    if not 1 <= shards <= 3:
        raise ValueError("shards must be in 1..3")
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

    out_root = out_root.resolve()
    if not out_root.exists():
        init_resume_dir(out_root, shards=shards, gpu=gpu)

    canary_dir = out_root / "packet-canary"
    canary_dir.mkdir(exist_ok=True)
    if (canary_dir / "canary.json").exists():
        raise FileExistsError(f"refusing to overwrite existing {canary_dir / 'canary.json'}")
    for idx in range(shards):
        sdir = out_root / f"packet-shard{idx}"
        sdir.mkdir(exist_ok=True)
        for fname in ("main-unsealed.json", "main.json"):
            if (sdir / fname).exists():
                raise FileExistsError(f"refusing to overwrite existing {sdir / fname}")

    allowed_commits, export_roots, lineage = _validate_stopped_prior_outputs(
        prior_outputs,
        commit=commit,
        repo=p.ROOT,
        check_git_equivalence=check_git_equivalence,
    )

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

    ledger_data = resume.build(contracts["main"], export_roots, allowed_commits=allowed_commits)
    verified_indices = {int(row["index"]) for row in ledger_data["units"]}
    remaining_total = count - len(verified_indices)
    if remaining_total <= 0:
        raise ValueError("all units are already verified; nothing left to resume")

    shared = out_root / "shared_inputs"
    shared.mkdir(parents=True, exist_ok=False)

    publish_json(shared / "main-contract.json", contracts["main"])
    publish_json(shared / "main-ledger.json", ledger_data)
    publish_json(shared / "canary-contract.json", contracts["canary"])
    publish_json(
        shared / "canary-ledger.json",
        empty_ledger(
            contracts["canary"]["namespace"],
            p.UNIT_KIND,
            p.contract_identity(contracts["canary"]),
        ),
    )

    pins = json.loads(
        (p.ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    bundle = shared / "release-evidence.tar.gz"
    release_bundle(bundle, pins)

    inputs = [
        Input(shared / "main-contract.json", "main-contract.json"),
        Input(shared / "main-ledger.json", "main-ledger.json"),
        Input(shared / "canary-contract.json", "canary-contract.json"),
        Input(shared / "canary-ledger.json", "canary-ledger.json"),
        Input(bundle, "release-evidence.tar.gz"),
        Input(p.ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json"),
    ]
    staged = pack_inputs(stage_inputs(shared / "staged", inputs), shared / METADATA)
    metadata_sha256 = next(row["sha256"] for row in staged if row["dest"] == METADATA)

    canary_forecast = compute_forecast(
        measured,
        assigned_units=1,
        total_remaining_units=remaining_total,
        horizon=p.CANARY_HORIZON,
        workers=1,
        gpu=gpu,
    )
    canary_manifest = build_manifest(
        job_id=f"{job_prefix}-canary",
        commit=commit,
        output_dir=canary_dir / "output-canary",
        gpu=gpu,
        base_minutes=15,
        contract=contracts["canary"],
        attempt=len(lineage) + 1,
        forecast=canary_forecast,
        staged_inputs=staged,
        metadata_sha256=metadata_sha256,
        label="canary",
        workers=1,
        is_probe=True,
    )
    canary_manifest["watch6_alllen_screen"]["prior_lineage"] = lineage
    canary_upload = check_uploads(canary_manifest)
    publish_json(canary_dir / "canary-upload-preflight.json", canary_upload)
    with (canary_dir / "canary.json").open("xt", encoding="utf-8") as stream:
        stream.write(json.dumps(canary_manifest, indent=2, sort_keys=True) + "\n")

    shard_summaries: list[dict[str, Any]] = []
    for shard_idx in range(shards):
        sdir = out_root / f"packet-shard{shard_idx}"
        assigned = shard_unverified_rows(
            contracts["main"],
            verified_indices,
            shard_index=shard_idx,
            shard_count=shards,
        )
        forecast = compute_forecast(
            measured,
            assigned_units=len(assigned),
            total_remaining_units=remaining_total,
            horizon=horizon,
            workers=workers,
            gpu=gpu,
            shard_index=shard_idx,
            shard_count=shards,
        )
        publish_json(sdir / "forecast.json", forecast)
        main_manifest = build_manifest(
            job_id=f"{job_prefix}-s{shard_idx}",
            commit=commit,
            output_dir=sdir / "output-main",
            gpu=gpu,
            base_minutes=int(forecast["base_minutes"]),
            contract=contracts["main"],
            attempt=len(lineage) + 1,
            forecast=forecast,
            staged_inputs=staged,
            metadata_sha256=metadata_sha256,
            label="main",
            workers=workers,
            is_probe=False,
            shard_index=shard_idx,
            shard_count=shards,
        )
        main_manifest["watch6_alllen_screen"]["prior_lineage"] = lineage
        main_manifest["watch6_alllen_screen"]["verified_skip_count"] = len(verified_indices)
        main_manifest["watch6_alllen_screen"]["assigned_indices"] = [
            int(row["index"]) for row in assigned
        ]
        upload_report = check_uploads(main_manifest)
        publish_json(sdir / "main-upload-preflight.json", upload_report)
        with (sdir / "main-unsealed.json").open("xt", encoding="utf-8") as stream:
            stream.write(json.dumps(main_manifest, indent=2, sort_keys=True) + "\n")
        shard_summaries.append(
            {
                "shard_index": shard_idx,
                "job_id": main_manifest["job_id"],
                "assigned_count": len(assigned),
                "assigned_indices": [int(row["index"]) for row in assigned],
                "base_minutes": forecast["base_minutes"],
                "timeout_minutes": main_manifest["timeout_minutes"],
                "unit_cap": main_manifest["budget"]["unit_cap"],
            }
        )

    summary = {
        "schema": "watch6-alllen-a100-resume-plan/v1",
        "namespace": namespace,
        "commit": commit,
        "gpu": gpu,
        "workers": workers,
        "shards": shards,
        "verified_count": len(verified_indices),
        "verified_indices": sorted(verified_indices),
        "remaining_count": remaining_total,
        "metadata_sha256": metadata_sha256,
        "upload_transfers_normal": canary_upload["normal_transfer_count"],
        "upload_transfers_fallback": canary_upload["transfer_count"],
        "upload_estimated_seconds": canary_upload["estimated_transfer_seconds"],
        "prior_lineage": lineage,
        "shard_plans": shard_summaries,
    }
    publish_json(out_root / "resume-plan.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_init = sub.add_parser(
        "init", help="pre-create A100 template hints before ready_queue enqueue"
    )
    p_init.add_argument("--out-root", type=Path, required=True)
    p_init.add_argument("--shards", type=int, default=3)
    p_init.add_argument("--gpu", default="A100", choices=sorted(GPU_RATES))

    p_build = sub.add_parser("build", help="rebuild dispatch-time ledger and emit canary + shards")
    p_build.add_argument("--out-root", type=Path, required=True)
    p_build.add_argument("--commit", required=True)
    p_build.add_argument("--benchmark", type=Path, required=True)
    p_build.add_argument("--prior-output", type=Path, nargs="+", required=True)
    p_build.add_argument("--namespace", default=p.NAMESPACE)
    p_build.add_argument("--count", type=int, default=p.COUNT)
    p_build.add_argument("--horizon", type=int, default=p.HORIZON)
    p_build.add_argument("--shards", type=int, default=3)
    p_build.add_argument("--gpu", default="A100", choices=sorted(GPU_RATES))
    p_build.add_argument("--workers", type=int, default=6)
    p_build.add_argument("--job-prefix", default="watch6-alllen-a100")

    args = parser.parse_args()
    try:
        if args.subcommand == "init":
            result = init_resume_dir(args.out_root, shards=args.shards, gpu=args.gpu)
        else:
            result = build_resume_packets(
                args.out_root,
                commit=args.commit,
                benchmark=args.benchmark,
                prior_outputs=args.prior_output,
                namespace=args.namespace,
                count=args.count,
                horizon=args.horizon,
                shards=args.shards,
                gpu=args.gpu,
                workers=args.workers,
                job_prefix=args.job_prefix,
            )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
