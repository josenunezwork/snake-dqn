"""Outcome-blind Mac recovery generator for Watch6 V3 (pre-start + <=2 compute resumes)."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import (
    is_external_transport_compute_failure,
    is_prestart_transport_failure,
)
from research.watch6_search_efficacy_v3.prepare import prepare
from research.watch6_search_efficacy_v3.resume import PAIR_NAME_RE
from research.watch6_search_efficacy_v3.resume import build as build_ledger
from research.watch6_search_efficacy_v3.resume import verify_bound_receipt
from tools.ops.package_spec import digest, publish_json


def recover(
    *,
    base_package: Path,
    receipts: list[Path],
    target_shards: list[int],
    out: Path,
    benchmark: Path,
    mirror_receipt: Path,
    empirical_costs: Path,
    commit: str,
) -> dict:
    """Inspect stopped receipts outcome-blind and prepare a lawful V3 recovery package."""
    base_package = base_package.resolve(strict=True)
    if not target_shards or sorted(set(target_shards)) != sorted(target_shards):
        raise ValueError("specify one or more distinct target shards")
    if any(s not in range(p.SHARDS) for s in target_shards):
        raise ValueError("target shard out of range 0..23")
    if not receipts:
        raise ValueError("recovery requires at least one stopped prior receipt")

    bound = []
    unit_paths: list[Path] = []
    seen_slots: set[int] = set()
    seen_receipt_dirs: set[Path] = set()

    for r_path in receipts:
        resolved = r_path.resolve(strict=True)
        if resolved.parent in seen_receipt_dirs:
            raise ValueError(f"duplicate receipt output directory in recovery: {resolved}")
        seen_receipt_dirs.add(resolved.parent)
        manifest, receipt = verify_bound_receipt(resolved)
        bound.append((resolved, manifest, receipt))
        efficacy_dir = resolved.parent / "exports/efficacy"
        if efficacy_dir.is_dir():
            for candidate in sorted(efficacy_dir.iterdir()):
                match = PAIR_NAME_RE.fullmatch(candidate.name)
                if match is not None:
                    slot = int(match.group(1))
                    if slot in seen_slots:
                        raise ValueError(f"duplicate exported slot {slot} across receipts")
                    seen_slots.add(slot)
                    unit_paths.append(candidate)

    ledger_data = build_ledger(unit_paths, [r for r, _, _ in bound])
    verified_indices = {row["index"] for row in ledger_data["units"]}

    shard_meta: dict[int, dict] = {}
    for shard in sorted(target_shards):
        shard_slots = p.shard_indices(shard)
        missing_slots = [idx for idx in shard_slots if idx not in verified_indices]
        if not missing_slots:
            raise ValueError(f"shard {shard} already has all 4 pairs verified; never recompute")

        shard_attempts = [
            (r_path, manifest, receipt)
            for r_path, manifest, receipt in bound
            if int(manifest["env"]["WATCH6_SHARD"]) == shard
        ]
        if not shard_attempts:
            raise ValueError(f"no stopped prior receipts found for target shard {shard}")

        prior_records: list[dict] = []
        prestart_count = 0
        compute_failures = 0

        for r_path, _, receipt in shard_attempts:
            if receipt.get("status") == "PASS":
                raise ValueError(
                    f"shard {shard} has PASS receipt {r_path} yet missing slots {missing_slots}"
                )
            if is_prestart_transport_failure(receipt, r_path.parent):
                prestart_count += 1
                prior_records.append(
                    {
                        "receipt_path": str(r_path),
                        "receipt_sha256": digest(r_path),
                        "classification": "prestart",
                    }
                )
            elif is_external_transport_compute_failure(receipt, r_path.parent):
                compute_failures += 1
                prior_records.append(
                    {
                        "receipt_path": str(r_path),
                        "receipt_sha256": digest(r_path),
                        "classification": "external_transport_compute",
                    }
                )
            else:
                f_cls = receipt.get("failure_class")
                f_err = receipt.get("error")
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: shard {shard} receipt {r_path} failed with "
                    f"forbidden/non-transport/unverified error ({f_cls}: {f_err})"
                )

        if compute_failures > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} exhausted compute completion cap "
                f"({compute_failures} > {p.MAX_COMPUTE_COMPLETIONS_PER_SHARD}); "
                "reserve substitution is disabled to preserve unbiased primary-bank inference"
            )

        shard_meta[shard] = {
            "prestart_retries_used": prestart_count,
            "compute_completion_attempt": compute_failures,
            "total_dispatches_for_shard": 1 + prestart_count + compute_failures,
            "prior_receipts": prior_records,
        }

    base_canary_receipt = base_package / "output-canary/receipt.json"
    base_canary_manifest_path = base_package / "canary.json"
    can_reuse_base_canary = (
        not verified_indices
        and base_canary_receipt.is_file()
        and base_canary_manifest_path.is_file()
    )

    ledger_out = out.parent / f"{out.name}-verified-ledger.json"
    publish_json(ledger_out, ledger_data)

    reuse_inputs = None
    if can_reuse_base_canary:
        base_canary_manifest = json.loads(base_canary_manifest_path.read_text(encoding="utf-8"))
        reuse_inputs = base_canary_manifest["inputs"]

    forecast = prepare(
        out.resolve(),
        benchmark.resolve(),
        commit,
        mirror_receipt.resolve(),
        empirical_costs.resolve(),
        canary_receipt=base_canary_receipt.resolve() if can_reuse_base_canary else None,
        main_shards=True,
        pending_canary=not can_reuse_base_canary,
        ledger=None if can_reuse_base_canary else ledger_out.resolve(),
        target_shards=sorted(target_shards),
        shard_meta=shard_meta,
        reuse_staged_inputs=reuse_inputs,
    )
    summary = {
        "schema": "watch6-v3-recovery-plan/v2",
        "target_shards": sorted(target_shards),
        "verified_units_in_ledger": len(verified_indices),
        "reused_initial_canary": can_reuse_base_canary,
        "requires_short_recovery_canary": not can_reuse_base_canary,
        "shard_meta": {str(k): v for k, v in sorted(shard_meta.items())},
        "ledger_path": str(ledger_out),
        "package_dir": str(out.resolve()),
        "forecast": forecast,
    }
    publish_json(out / "recovery-plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument(
        "--receipts", "--receipt", dest="receipts", type=Path, nargs="+", required=True
    )
    parser.add_argument("--shards", "--shard", dest="shards", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--mirror-receipt", type=Path, required=True)
    parser.add_argument("--empirical-costs", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("recovery preparation must run on Mac")
    plan = recover(
        base_package=args.base_package,
        receipts=args.receipts,
        target_shards=args.shards,
        out=args.out,
        benchmark=args.benchmark,
        mirror_receipt=args.mirror_receipt,
        empirical_costs=args.empirical_costs,
        commit=args.commit,
    )
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == "__main__":
    main()
