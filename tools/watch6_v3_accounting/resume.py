"""Mac outcome-blind verified unit ledger adapter for Watch6 V3 (Astra Ruling Points 2, 3, 4, 7).

Preserves every sidecar, .meta.json, export-inventory, primary-allocation, and clean-stop
requirement on allocated runtimes while allowing a PROVEN never-allocated pre-start receipt
to remain in the bound receipt set without rewriting its receipt or setting stop_verified=True.
Any receipt owning an exported scientific unit continues to require verified stop.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from tools.watch6_v3_accounting import (
    BASELINE_COMMIT,
    BASELINE_SCIENCE,
    V3_BROKER_ROOT,
    bind_broker_protocol,
    compute_broker_protocol,
    ensure_cli_dash_b,
    isolated_broker_execution,
    verify_baseline_closure,
)

# isort: split

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import admission as v3_admission
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3 import resume as v3_resume
from research.watch6_search_efficacy_v3.run import ledger_identity
from tools.ops.package_spec import digest, empty_ledger, publish_json, unit_id, verify_immutable
from tools.watch6_v3_accounting.proof import (
    has_forbidden_or_contradictory_output_artifacts,
    is_proven_never_allocated_attempt,
)

PAIR_NAME_RE = v3_resume.PAIR_NAME_RE


def verify_bound_receipt(
    receipt_path: Path,
    *,
    ops_dir: Path | None = None,
    broker_root: Path | None = None,
    _verified_broker: tuple[dict[str, str], dict[str, object]] | None = None,
) -> tuple[dict, dict]:
    """Verify a stopped or proven never-allocated V3 main-shard receipt and executed manifest."""
    eff_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with isolated_broker_execution("resume-receipt", broker_root=eff_broker):
        if _verified_broker is not None:
            closure_info, broker_proto = _verified_broker
        else:
            closure_info = verify_baseline_closure(eff_broker)
            broker_proto = compute_broker_protocol(Path(closure_info["broker_root"]))
        broker_science = closure_info["science"]
        broker_namespace = closure_info["namespace"]
        broker_alloc = broker_proto["allocation"]

        if receipt_path.is_symlink():
            raise ValueError(f"symlinked receipt forbidden: {receipt_path}")
        resolved_receipt = receipt_path.resolve(strict=True)
        if resolved_receipt.is_symlink() or not resolved_receipt.is_file():
            raise ValueError(f"missing regular receipt.json: {resolved_receipt}")
        receipt = json.loads(resolved_receipt.read_text(encoding="utf-8"))
        manifest_path = resolved_receipt.parent / "manifest.json"
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or digest(manifest_path) != receipt.get("manifest_sha256")
        ):
            raise ValueError(f"receipt manifest_sha256 binding mismatch: {resolved_receipt}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            receipt.get("job_id") != manifest.get("job_id")
            or receipt.get("commit") != manifest.get("commit")
            or manifest.get("commit") != BASELINE_COMMIT
        ):
            raise ValueError(
                f"receipt job_id/commit mismatch against baseline manifest.json: {resolved_receipt}"
            )

        prealloc_clean_unassigned = (
            receipt.get("endpoint") is None
            and receipt.get("stop_verified") is False
            and receipt.get("cleanup_error") is None
            and not has_forbidden_or_contradictory_output_artifacts(
                resolved_receipt.parent,
                job_id=manifest.get("job_id") if isinstance(manifest.get("job_id"), str) else None,
                session=(
                    receipt.get("session") if isinstance(receipt.get("session"), str) else None
                ),
            )
            and v3_admission.is_prestart_transport_failure(receipt, resolved_receipt.parent)
        )
        proven_never_allocated = is_proven_never_allocated_attempt(
            receipt,
            resolved_receipt.parent,
            ops_dir=ops_dir,
        )
        if not (prealloc_clean_unassigned or proven_never_allocated) and (
            receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
        ):
            raise ValueError(
                f"runtime stop not verified or cleanup_error present: {resolved_receipt}"
            )

        binding = manifest.get("watch6_v3", {})
        env = manifest.get("env", {})
        req = manifest.get("requirements", {})
        if (
            canary.is_canary(manifest)
            or env.get("WATCH6_FIXTURE") != "0"
            or binding.get("namespace") != broker_namespace
            or binding.get("science") != broker_science
            or binding.get("science") != BASELINE_SCIENCE
            or binding.get("primary_allocation") != broker_alloc
            or any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
            or req.get("gpu") != "L4"
            or req.get("allowed_gpus") != ["L4"]
            or "prefer" in req
            or int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS
            or int(env.get("WATCH6_SHARD", -1)) not in range(p.SHARDS)
        ):
            raise ValueError(
                "foreign, fixture, reserve, non-L4, or canary receipt rejected "
                f"from scientific ledger: {resolved_receipt}"
            )
        return manifest, receipt


def build(
    paths: list[Path],
    receipts: list[Path],
    *,
    ops_dir: Path | None = None,
    broker_root: Path | None = None,
) -> dict:
    """Verify immutable pair exports, sidecars, .meta.json, and receipts outcome-blind."""
    target_broker = broker_root if broker_root is not None else V3_BROKER_ROOT
    with (
        isolated_broker_execution("resume", broker_root=target_broker),
        bind_broker_protocol(target_broker, require_head_at_baseline=True) as closure_info,
    ):
        resolved_broker = Path(closure_info["broker_root"])
        broker_proto = compute_broker_protocol(resolved_broker)
        broker_science = closure_info["science"]
        if paths and not receipts:
            raise ValueError("completed units require stopped supervisor receipts")

        bound_receipts: list[tuple[Path, dict, dict]] = []
        seen_receipt_dirs: set[Path] = set()
        for r_path in receipts:
            if r_path.is_symlink():
                raise ValueError(f"symlinked receipt forbidden: {r_path}")
            resolved = r_path.resolve(strict=True)
            if resolved.parent in seen_receipt_dirs:
                raise ValueError(f"duplicate receipt output directory: {resolved}")
            seen_receipt_dirs.add(resolved.parent)
            manifest, receipt = verify_bound_receipt(
                resolved,
                ops_dir=ops_dir,
                broker_root=resolved_broker,
                _verified_broker=(closure_info, broker_proto),
            )
            bound_receipts.append((resolved, manifest, receipt))

        result = empty_ledger(p.NAMESPACE, "watch6", ledger_identity())
        seen_indices: set[int] = set()
        primary_alloc = p.allocation()

        for raw_path in sorted(paths, key=lambda item: item.name):
            match = PAIR_NAME_RE.fullmatch(raw_path.name)
            if match is None:
                raise ValueError(f"refusing non-scientific or fixture filename: {raw_path.name}")
            index = int(match.group(1))
            if index not in range(p.COUNT) or index in seen_indices:
                raise ValueError(f"foreign or duplicate allocation index: {index}")
            seen_indices.add(index)

            if raw_path.is_symlink():
                raise ValueError(f"symlinked unit export forbidden: {raw_path}")
            path = raw_path.resolve(strict=True)
            descriptor = verify_immutable(path)
            meta_path = path.with_suffix(".meta.json")
            verify_immutable(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            owners = [
                (r_path, manifest, receipt)
                for r_path, manifest, receipt in bound_receipts
                if path == (r_path.parent / "exports/efficacy" / path.name).resolve()
            ]
            if len(owners) != 1:
                raise ValueError(f"unit {path} lacks unique verified supervisor receipt root")
            owner_rpath, owner_manifest, owner_receipt = owners[0]
            if (
                owner_receipt.get("stop_verified") is not True
                or owner_receipt.get("cleanup_error") is not None
                or not isinstance(owner_receipt.get("endpoint"), str)
                or not owner_receipt.get("endpoint")
            ):
                raise ValueError(
                    f"exported unit {path} owned by unstopped or unallocated receipt {owner_rpath}"
                )

            owner_shard = int(owner_manifest["env"]["WATCH6_SHARD"])
            if index % p.SHARDS != owner_shard:
                raise ValueError(f"slot {index} does not belong to shard {owner_shard}")

            rel_export = f"efficacy/{path.name}"
            if not any(
                row.get("path") == rel_export
                and row.get("sha256") == descriptor["sha256"]
                and row.get("size") == descriptor["size"]
                for row in owner_receipt.get("exports", [])
            ):
                raise ValueError(
                    f"unit {path.name} missing or mismatched in receipt export inventory"
                )

            if meta.get("seed_kind") != "primary":
                raise ValueError(f"only primary seeds are permitted in V3 ({meta_path})")

            expected_meta = {
                "namespace": p.NAMESPACE,
                "science": broker_science,
                "seed_kind": "primary",
                "allocation": primary_alloc[index],
                "fixture": False,
                "sha256": descriptor["sha256"],
                "size": descriptor["size"],
            }
            if meta != expected_meta:
                raise ValueError(f"unit completion metadata mismatch: {meta_path}")

            result["units"].append(
                {
                    "index": index,
                    "unit_id": unit_id(p.NAMESPACE, "watch6", index),
                    "sha256": descriptor["sha256"],
                    "size": descriptor["size"],
                }
            )

        result["units"].sort(key=lambda row: row["index"])
        return result


def main() -> None:
    ensure_cli_dash_b("tools.watch6_v3_accounting.resume")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, nargs="*", default=[])
    parser.add_argument("--receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--ops-dir", type=Path, default=None)
    parser.add_argument(
        "--broker-root",
        "--broker",
        dest="broker_root",
        type=Path,
        default=V3_BROKER_ROOT,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("ledger verification must run on Mac")
    publish_json(
        args.out,
        build(args.units, args.receipts, ops_dir=args.ops_dir, broker_root=args.broker_root),
    )


if __name__ == "__main__":
    main()
