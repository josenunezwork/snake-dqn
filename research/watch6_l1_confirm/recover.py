"""Outcome-blind linear-lineage recovery packet builder for incomplete Watch6-L1-V1 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.admission import (
    is_external_transport_compute_failure,
    is_prestart_transport_failure,
)
from research.watch6_l1_confirm.prepare import prepare
from research.watch6_l1_confirm.resume import WORLD_NAME_RE
from research.watch6_l1_confirm.resume import build as build_verified_ledger
from tools.ops.package_spec import digest, publish_json


def build_recovery_packet(
    *,
    prior_receipts_by_shard: dict[int, list[Path]],
    all_stopped_main_receipts: list[Path],
    out_dir: Path,
    benchmark: Path,
    commit: str,
    mirror_receipt: Path,
    empirical_costs: Path,
    canary_receipts_by_gpu: dict[str, Path] | None = None,
    pending_canary: bool = True,
    judge: str = "Astra",
) -> dict:
    """Build study-wide union verified-ledger.json and fresh linear recovery manifests."""
    out_dir = out_dir.resolve()
    unit_paths: list[Path] = []
    seen_slots: set[int] = set()
    for r_path in all_stopped_main_receipts:
        resolved = r_path.resolve(strict=True)
        rec = json.loads(resolved.read_text(encoding="utf-8"))
        for row in rec.get("exports", []):
            rel = str(row.get("path", ""))
            if not rel.startswith("efficacy/"):
                continue
            match = WORLD_NAME_RE.fullmatch(Path(rel).name)
            if match is None:
                continue
            slot = int(match.group(1))
            if slot in seen_slots:
                raise ValueError(f"duplicate verified slot {slot} across prior receipts")
            seen_slots.add(slot)
            unit_paths.append((resolved.parent / "exports" / rel).resolve(strict=True))

    ledger = build_verified_ledger(unit_paths, all_stopped_main_receipts)
    verified_indices = {row["index"] for row in ledger["units"]}

    target_shards: list[int] = []
    shard_meta: dict[int, dict] = {}
    for shard, history_paths in sorted(prior_receipts_by_shard.items()):
        owned = set(p.shard_indices(shard))
        if owned.issubset(verified_indices):
            continue
        if not history_paths:
            raise ValueError(f"recovery shard {shard} requires prior stopped receipt history")
        prestart_used = 0
        compute_used = 0
        priors: list[dict] = []
        for hp in history_paths:
            rp = hp.resolve(strict=True)
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("status") == "PASS":
                raise ValueError(f"cannot recover shard {shard} after PASS receipt {rp}")
            if is_prestart_transport_failure(rec, rp.parent):
                cls = "prestart"
                prestart_used += 1
            elif is_external_transport_compute_failure(rec, rp.parent):
                cls = "external_transport_compute"
                compute_used += 1
            else:
                raise ValueError(f"forbidden non-retryable failure class on shard {shard}: {rp}")
            priors.append(
                {
                    "receipt_path": str(rp),
                    "receipt_sha256": digest(rp),
                    "classification": cls,
                }
            )
        if compute_used > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            raise ValueError(
                f"shard {shard} exhausted compute completion retry cap ({compute_used})"
            )
        target_shards.append(shard)
        shard_meta[shard] = {
            "prestart_retries_used": prestart_used,
            "compute_completion_attempt": compute_used,
            "total_dispatches_for_shard": 1 + prestart_used + compute_used,
            "prior_receipts": priors,
        }

    if not target_shards:
        raise ValueError("no incomplete shards require recovery")

    ledger_staging = out_dir.parent / f"{out_dir.name}-union-ledger.json"
    publish_json(ledger_staging, ledger)
    return prepare(
        out_dir,
        benchmark,
        commit,
        mirror_receipt,
        empirical_costs,
        canary_receipts_by_gpu=canary_receipts_by_gpu,
        main_shards=True,
        pending_canary=pending_canary,
        ledger=ledger_staging,
        target_shards=target_shards,
        shard_meta=shard_meta,
        judge=judge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mirror-receipt", type=Path, required=True)
    parser.add_argument("--empirical-costs", type=Path, required=True)
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()
    spec = json.loads(args.spec_json.resolve().read_text(encoding="utf-8"))
    by_shard = {int(k): [Path(x) for x in v] for k, v in spec["prior_receipts_by_shard"].items()}
    all_receipts = [Path(x) for x in spec["all_stopped_main_receipts"]]
    result = build_recovery_packet(
        prior_receipts_by_shard=by_shard,
        all_stopped_main_receipts=all_receipts,
        out_dir=args.out.resolve(),
        benchmark=args.benchmark.resolve(),
        commit=args.commit,
        mirror_receipt=args.mirror_receipt.resolve(),
        empirical_costs=args.empirical_costs.resolve(),
        judge=args.judge,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
