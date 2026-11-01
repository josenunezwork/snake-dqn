"""Prepare queue-ready Watch6 V3 manifests via the host accounting adapter (Ruling Points 2, 3, 6).

Enforces actual-byte broker closure equality on ``V3_BROKER_ROOT`` and runtime commit
``357e3504ab7f4b1e5c78209eee14391d16234c7a`` with unchanged ``protocol.source_paths()`` and
``protocol.identity()``. Never injects the accounting adapter into the runtime source closure.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.costs import empirical_forecast, verify_report_evidence
from research.watch6_search_efficacy_v3.mirror import validate as validate_mirror
from research.watch6_search_efficacy_v3.prepare import ALLOWED_REMOTE_BRANCHES
from research.watch6_search_efficacy_v3.run import ledger_identity
from research.watch6_search_efficacy_v3.uploads import METADATA, checkpoint_inputs, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs
from tools.watch6_v3_accounting import (
    BASELINE_COMMIT,
    BASELINE_SCIENCE,
    V3_BROKER_ROOT,
    bind_broker_protocol,
    compute_broker_protocol,
)
from tools.watch6_v3_accounting.admission import admit, completed_ledger


def _require_published_pin(
    commit: str,
    broker_root: Path,
    source_paths: list[str] | None = None,
) -> None:
    if commit != BASELINE_COMMIT:
        raise ValueError(f"Watch6 V3 runtime commit must remain {BASELINE_COMMIT}, got {commit!r}")
    resolved_broker = broker_root.resolve()
    if source_paths is not None:
        paths = source_paths
    else:
        raw_paths = compute_broker_protocol(resolved_broker)["source_paths"]
        paths = [str(item) for item in raw_paths]  # type: ignore[union-attr]
    subprocess.run(
        ["git", "--no-optional-locks", "cat-file", "-e", commit],
        cwd=resolved_broker,
        check=True,
        timeout=10,
    )
    refs = subprocess.check_output(
        ["git", "--no-optional-locks", "branch", "-r", "--contains", commit],
        cwd=resolved_broker,
        text=True,
        timeout=10,
    )
    if not any(branch in refs for branch in ALLOWED_REMOTE_BRANCHES):
        raise ValueError(
            f"commit {commit} must be pushed to origin/main or origin/argon-v3-confirm-1031"
        )
    if subprocess.check_output(
        ["git", "--no-optional-locks", "diff", commit, "--", *paths],
        cwd=resolved_broker,
        timeout=15,
    ):
        raise ValueError("local V3 executable source closure differs from pinned commit")


def prepare(
    out: Path,
    benchmark: Path,
    commit: str,
    mirror_receipt: Path,
    empirical_costs: Path,
    *,
    canary_receipt: Path | None = None,
    main_shards: bool = True,
    pending_canary: bool = False,
    ledger: Path | None = None,
    target_shards: list[int] | None = None,
    shard_meta: dict[int, dict] | None = None,
    reuse_staged_inputs: list[dict] | None = None,
    ops_dir: Path | None = None,
    broker_root: Path | None = None,
) -> dict:
    """Prepare V3 L4 W4 shard manifests pinned to baseline commit 357e3504 offline."""
    target_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with bind_broker_protocol(target_broker, require_head_at_baseline=True) as closure_info:
        resolved_broker = Path(closure_info["broker_root"])
        broker_science = closure_info["science"]

        if main_shards and canary_receipt is None and not pending_canary:
            raise ValueError(
                "main preparation requires passing same-pin canary receipt or --pending-canary"
            )
        if pending_canary and (not main_shards or canary_receipt is not None):
            raise ValueError(
                "pending_canary requires main_shards and no pre-existing canary_receipt"
            )

        source_paths = p.source_paths()
        if any("watch6_v3_accounting" in entry for entry in source_paths):
            raise ValueError("REFUSE: accounting adapter leaked into runtime source_paths")

        validate_mirror(json.loads(mirror_receipt.read_text(encoding="utf-8")))
        _require_published_pin(commit, resolved_broker, source_paths)

        measured = json.loads(benchmark.read_text(encoding="utf-8"))
        if measured.get("science") != broker_science or measured.get("science") != BASELINE_SCIENCE:
            raise ValueError("stale Mac preflight benchmark.json")

        cost_report = json.loads(empirical_costs.read_text(encoding="utf-8"))
        verify_report_evidence(cost_report)
        forecast = empirical_forecast(cost_report)

        shard_meta = shard_meta or {}
        if ledger is not None:
            verified = completed_ledger(ledger)
            if len(verified) == p.COUNT:
                raise ValueError("all 96 V3 units already verified; never recompute")

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
                    "schema": "watch6-v3-protocol-binding/v2",
                    "implementer": "Argon",
                    "judge": "Astra",
                    "namespace": p.NAMESPACE,
                    "science": broker_science,
                    "registration_sha256": digest(p.registration()),
                },
            )
            contract = out / "runtime-contract.json"
            publish_json(
                contract,
                {
                    "schema": "watch6-v3-runtime-contract/v2",
                    "namespace": p.NAMESPACE,
                    "science": broker_science,
                    "primary_allocation": p.allocation(),
                },
            )
            saved = resolved_broker.parent / "snake-dqn/saved_snakes"
            pins = json.loads(
                (resolved_broker / "web/backend/served_checkpoint_pins.json").read_text("utf-8")
            )
            release_bundle = out / "release-evidence.tar.gz"
            with (
                release_bundle.open("xb") as raw_bundle,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw_bundle, mtime=0) as compressed,
                tarfile.open(fileobj=compressed, mode="w") as archive,
            ):
                for name, pin in pins["checkpoints"].items():
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
                *checkpoint_inputs(saved, out / "checkpoint-parts"),
            ]
            staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)

        base_minutes = forecast["base_minutes_per_shard"]
        rate = 2.5
        metadata_sha256 = next(row["sha256"] for row in staged if row["dest"] == METADATA)
        common = {
            "commit": commit,
            "script": p.SCRIPT,
            "source_paths": source_paths,
            "device": "cpu",
            "lease_seconds": 300,
            "max_export_bytes": 4 * 1024**2,
            "required_exports": ["efficacy/operational.json", "performance.json"],
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
                "SNAKE_WATCH_SOURCE": "live",
                "WATCH6_SCIENCE": broker_science,
                "WATCH6_METADATA_SHA256": metadata_sha256,
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
            "watch6_v3": {
                "namespace": p.NAMESPACE,
                "judge": "Astra",
                "implementer": "Argon",
                "independent_implementation_review": True,
                "science": broker_science,
                "registration_sha256": digest(p.registration()),
                "primary_allocation": p.allocation(),
                "forecast_sha256": digest(out / "forecast.json"),
            },
        }

        selected_shards = (
            sorted(target_shards) if target_shards is not None else list(range(forecast["shards"]))
        )
        if any(s not in range(p.SHARDS) for s in selected_shards):
            raise ValueError("target_shards out of range 0..23")

        jobs: list[tuple[str, int, bool]] = []
        if not main_shards or pending_canary:
            jobs.append(("canary", 0, True))
        if main_shards:
            jobs.extend((f"shard-{s:02d}", s, False) for s in selected_shards)

        for label, shard, probe in jobs:
            value = json.loads(json.dumps(common))
            base = 15 if probe else base_minutes
            unit = base / 60 * rate
            meta = shard_meta.get(shard, {})
            phase = "canary" if probe else "primary"
            prestart_used = 0 if probe else int(meta.get("prestart_retries_used", 0))
            compute_attempt = 0 if probe else int(meta.get("compute_completion_attempt", 0))
            total_dispatches = (
                1
                if probe
                else int(
                    meta.get("total_dispatches_for_shard", 1 + prestart_used + compute_attempt)
                )
            )
            prior_receipts = [] if probe else list(meta.get("prior_receipts", []))

            value.update(
                job_id=f"watch6-v3-{label}-{out.name}",
                output_dir=str(out / f"output-{label}"),
                timeout_minutes=2 * base,
                requirements={
                    "gpu": "L4",
                    "allowed_gpus": ["L4"],
                    "min_vcpu": 12,
                    "est_minutes": base,
                },
                budget={
                    "unit_cap": 2 * unit,
                    "rate_units_per_hour": rate,
                    "reserve_units": 2 * unit * 0.2,
                    "other_assignments": {},
                },
            )
            value["watch6_v3"].update(
                shard=shard,
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
                WATCH6_WORKERS=str(p.WORKERS),
                WATCH6_SHARDS=str(p.SHARDS),
                WATCH6_BASE_MINUTES=str(base),
            )
            if probe:
                value["env"].update(
                    SNAKE_COLAB_CANARY="1",
                    SNAKE_COLAB_CANARY_PACKAGE="research/watch6_search_efficacy_v3",
                )
                value["canary"] = {
                    "schema": canary.SCHEMA,
                    "mode": "probe",
                    "family": canary.family(value),
                }
            else:
                qualification = (
                    out / "output-canary/receipt.json" if pending_canary else canary_receipt
                )
                assert qualification is not None
                value["canary"] = {
                    "schema": canary.SCHEMA,
                    "receipt": {
                        "path": str(qualification.resolve()),
                        "sha256": "0" * 64 if pending_canary else digest(qualification),
                    },
                }
                value["env"]["WATCH6_MEASUREMENTS"] = json.dumps(cost_report, sort_keys=True)
                value["env"]["WATCH6_FROZEN_LAYOUT"] = json.dumps(
                    {k: forecast[k] for k in ("workers", "shards", "base_minutes_per_shard")},
                    sort_keys=True,
                )
                if pending_canary:
                    value["pending_qualification"] = True
                else:
                    canary.check(value, resolved_broker)

            upload_report = admit(
                value,
                pending_qualification=pending_canary and not probe,
                ops_dir=ops_dir,
                broker_root=resolved_broker,
            )
            publish_json(out / f"{label}-upload-preflight.json", upload_report)
            suffix = ".pending.json" if pending_canary and not probe else ".json"
            (out / f"{label}{suffix}").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )

        return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mirror-receipt", type=Path, required=True)
    parser.add_argument("--empirical-costs", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--pending-canary", action="store_true")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--ops-dir", type=Path, default=None)
    parser.add_argument(
        "--broker-root",
        "--broker",
        dest="broker_root",
        type=Path,
        default=V3_BROKER_ROOT,
    )
    args = parser.parse_args()
    result = prepare(
        args.out.resolve(),
        args.benchmark.resolve(),
        args.commit,
        args.mirror_receipt.resolve(),
        args.empirical_costs.resolve(),
        canary_receipt=args.canary_receipt.resolve() if args.canary_receipt else None,
        main_shards=not args.canary_only,
        pending_canary=args.pending_canary,
        ledger=args.ledger.resolve() if args.ledger else None,
        ops_dir=args.ops_dir.resolve() if args.ops_dir else None,
        broker_root=args.broker_root,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
