"""Mac outcome-blind verified unit ledger construction for Watch6 V3."""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import is_prestart_transport_failure
from research.watch6_search_efficacy_v3.run import ledger_identity
from tools.ops.package_spec import digest, empty_ledger, publish_json, unit_id, verify_immutable

PAIR_NAME_RE = re.compile(r"^pair-([0-9]{2})\.json$")


def verify_bound_receipt(receipt_path: Path) -> tuple[dict, dict]:
    """Verify a stopped V3 main-shard supervisor receipt and its executed manifest."""
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = receipt_path.parent / "manifest.json"
    if not manifest_path.is_file() or digest(manifest_path) != receipt.get("manifest_sha256"):
        raise ValueError(f"receipt manifest_sha256 binding mismatch: {receipt_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("job_id") != manifest.get("job_id") or receipt.get("commit") != manifest.get(
        "commit"
    ):
        raise ValueError(f"receipt job_id/commit mismatch against manifest.json: {receipt_path}")

    prealloc_unassigned = (
        receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("cleanup_error") is None
        and is_prestart_transport_failure(receipt, receipt_path.parent)
    )
    if not prealloc_unassigned and (
        receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
    ):
        raise ValueError(f"runtime stop not verified or cleanup_error present: {receipt_path}")

    binding = manifest.get("watch6_v3", {})
    env = manifest.get("env", {})
    if (
        canary.is_canary(manifest)
        or env.get("WATCH6_FIXTURE") != "0"
        or binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
        or any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
        or int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS
        or int(env.get("WATCH6_SHARD", -1)) not in range(p.SHARDS)
    ):
        raise ValueError(
            "foreign, fixture, reserve, or canary receipt rejected "
            f"from scientific ledger: {receipt_path}"
        )
    return manifest, receipt


def build(paths: list[Path], receipts: list[Path]) -> dict:
    """Verify immutable pair exports, sidecars, .meta.json, and stopped receipts outcome-blind."""
    if paths and not receipts:
        raise ValueError("completed units require stopped supervisor receipts")

    bound_receipts: list[tuple[Path, dict, dict]] = []
    seen_receipt_dirs: set[Path] = set()
    for r_path in receipts:
        resolved = r_path.resolve(strict=True)
        if resolved.parent in seen_receipt_dirs:
            raise ValueError(f"duplicate receipt output directory: {resolved}")
        seen_receipt_dirs.add(resolved.parent)
        manifest, receipt = verify_bound_receipt(resolved)
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
        _, owner_manifest, owner_receipt = owners[0]
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
            raise ValueError(f"unit {path.name} missing or mismatched in receipt export inventory")

        if meta.get("seed_kind") != "primary":
            raise ValueError(f"only primary seeds are permitted in V3 ({meta_path})")

        expected_meta = {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, nargs="*", default=[])
    parser.add_argument("--receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("ledger verification must run on Mac")
    publish_json(args.out, build(args.units, args.receipts))


if __name__ == "__main__":
    main()
