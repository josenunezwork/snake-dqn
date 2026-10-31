"""Outcome-blind Mac recovery generator for Watch6 V3 (Tier 1/2/3 recovery)."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import (
    FORBIDDEN_TIMEOUT_SUBSTRINGS,
    is_external_transport_compute_failure,
)
from research.watch6_search_efficacy_v3.prepare import prepare
from research.watch6_search_efficacy_v3.resume import PAIR_NAME_RE
from research.watch6_search_efficacy_v3.resume import build as build_ledger
from research.watch6_search_efficacy_v3.resume import verify_bound_receipt
from tools.ops.package_spec import digest, publish_json


def _has_compute_progress(output_dir: Path) -> bool:
    for telemetry_path in sorted(output_dir.glob("telemetry-*.json")):
        try:
            if "WATCH6_V3_PROGRESS" in telemetry_path.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def is_prestart_transport_failure(receipt: dict, output_dir: Path) -> bool:
    """True iff attempt stopped cleanly with zero exports before scientific execution began."""
    if (
        receipt.get("status") == "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        return False
    if receipt.get("exports"):
        return False
    exports_dir = output_dir / "exports"
    if exports_dir.exists() and any(exports_dir.rglob("pair-*.json")):
        return False
    if receipt.get("terminal") is not None or (output_dir / "terminal.json").exists():
        return False
    failure_class = str(receipt.get("failure_class") or "")
    error_upper = str(receipt.get("error") or "").upper()
    if failure_class == "CHILD_NONZERO_EXIT":
        return False
    if any(
        token in failure_class.upper() or token in error_upper
        for token in FORBIDDEN_TIMEOUT_SUBSTRINGS
    ):
        return False
    if not (output_dir / "hardware.json").is_file():
        return True
    if failure_class in {"UPLOAD_CAP", "ACCOUNTING"}:
        return True
    if is_external_transport_compute_failure(receipt, output_dir) and not _has_compute_progress(
        output_dir
    ):
        return True
    return False


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
    existing_reserve_slots: set[int] = set()
    study_prestart_retries = 0

    for r_path in receipts:
        resolved = r_path.resolve(strict=True)
        manifest, receipt = verify_bound_receipt(resolved)
        bound.append((resolved, manifest, receipt))
        existing_reserve_slots.update(manifest["watch6_v3"].get("reserve_slots", []))
        if is_prestart_transport_failure(receipt, resolved.parent):
            study_prestart_retries += 1
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

    new_reserve_slots = set(existing_reserve_slots) - verified_indices
    reserve_lineage: dict[str, dict[str, list[dict]]] = {"shards": {}}
    shard_meta: dict[int, dict] = {}

    # Preserve existing reserve-lineage receipts for already-authorized unverified reserve slots.
    for _, manifest, _ in bound:
        prior_lineage = manifest["watch6_v3"].get("reserve_lineage", {}).get("shards", {})
        for s_key, entries in prior_lineage.items():
            if any(slot % p.SHARDS == int(s_key) for slot in new_reserve_slots):
                reserve_lineage["shards"][str(s_key)] = entries

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

        prestart_attempts = []
        primary_compute_failures = []
        reserve_compute_failures = []

        for r_path, manifest, receipt in shard_attempts:
            if receipt.get("status") == "PASS":
                raise ValueError(
                    f"shard {shard} has PASS receipt {r_path} yet missing slots {missing_slots}"
                )
            phase = manifest["watch6_v3"].get("phase", "primary")
            if is_prestart_transport_failure(receipt, r_path.parent):
                prestart_attempts.append((r_path, manifest, receipt))
            elif is_external_transport_compute_failure(receipt, r_path.parent):
                if phase == "reserve":
                    reserve_compute_failures.append((r_path, manifest, receipt))
                else:
                    primary_compute_failures.append((r_path, manifest, receipt))
            else:
                f_cls = receipt.get("failure_class")
                f_err = receipt.get("error")
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: shard {shard} receipt {r_path} failed with non-MCAR / "
                    f"child / wall-timeout error ({f_cls}: {f_err})"
                )

        if reserve_compute_failures:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} already exhausted its single reserve attempt"
            )

        is_new_prestart_retry = is_prestart_transport_failure(
            shard_attempts[-1][2], shard_attempts[-1][0].parent
        )
        prestart_count = len(prestart_attempts)
        if prestart_count > p.MAX_PRESTART_RETRIES_PER_SHARD:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} pre-start retries {prestart_count} "
                f"exceed cap {p.MAX_PRESTART_RETRIES_PER_SHARD}"
            )
        if is_new_prestart_retry and study_prestart_retries > p.MAX_PRESTART_RETRIES_STUDY:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: study-wide pre-start retries {study_prestart_retries} "
                f"exceed cap {p.MAX_PRESTART_RETRIES_STUDY}"
            )

        total_dispatches = len(shard_attempts) + 1
        if total_dispatches > p.MAX_TOTAL_DISPATCHES_PER_SHARD:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} total dispatches {total_dispatches} "
                f"exceed cap {p.MAX_TOTAL_DISPATCHES_PER_SHARD}"
            )

        n_compute = len(primary_compute_failures)
        if n_compute <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            shard_meta[shard] = {
                "prestart_retries_used": prestart_count,
                "compute_completion_attempt": n_compute,
                "total_dispatches_for_shard": total_dispatches,
            }
        elif n_compute == 1 + p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            new_reserve_slots.update(missing_slots)
            if len(new_reserve_slots) > p.MAX_RESERVE_SUBSTITUTIONS:
                raise ValueError(
                    f"INVALID_OR_INCOMPLETE: reserve substitutions {len(new_reserve_slots)} "
                    f"exceed study-wide cap {p.MAX_RESERVE_SUBSTITUTIONS}"
                )
            reserve_lineage["shards"][str(shard)] = [
                {"receipt_path": str(r_path), "receipt_sha256": digest(r_path)}
                for r_path, _, _ in primary_compute_failures
            ]
            shard_meta[shard] = {
                "prestart_retries_used": prestart_count,
                "compute_completion_attempt": 0,
                "total_dispatches_for_shard": total_dispatches,
            }
        else:
            raise ValueError(
                f"INVALID_OR_INCOMPLETE: shard {shard} exceeded primary compute attempts"
            )

    ordered_reserves = sorted(new_reserve_slots)
    base_canary_receipt = base_package / "output-canary/receipt.json"
    base_canary_manifest_path = base_package / "canary.json"
    can_reuse_base_canary = (
        not verified_indices
        and not ordered_reserves
        and base_canary_receipt.is_file()
        and base_canary_manifest_path.is_file()
    )

    ledger_out = out.parent / f"{out.name}-verified-ledger.json"
    publish_json(ledger_out, ledger_data)

    reuse_inputs = None
    if can_reuse_base_canary:
        base_canary_manifest = json.loads(base_canary_manifest_path.read_text())
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
        reserve_slots=ordered_reserves,
        reserve_lineage=reserve_lineage if ordered_reserves else {},
        shard_meta=shard_meta,
        reuse_staged_inputs=reuse_inputs,
    )
    summary = {
        "schema": "watch6-v3-recovery-plan/v1",
        "target_shards": sorted(target_shards),
        "verified_units_in_ledger": len(verified_indices),
        "reserve_slots": ordered_reserves,
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
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
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
