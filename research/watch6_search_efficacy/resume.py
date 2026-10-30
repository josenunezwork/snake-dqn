"""Mac outcome-blind verified unit ledger construction, never outcome reduction."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import ledger_identity
from tools.ops.package_spec import empty_ledger, publish_json, unit_id, verify_immutable


def build(paths: list[Path], receipts: list[Path]) -> dict:
    """Use filename/index and verified size/hash only; never parse pair outcomes."""
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt.get("stop_verified") is not True or receipt.get("cleanup_error"):
            raise ValueError("previous runtime stop not verified")
    if paths and not receipts:
        raise ValueError("completed units need stopped supervisor receipts")
    result = empty_ledger(p.NAMESPACE, "watch6", ledger_identity())
    seen = set()
    for path in paths:
        descriptor = verify_immutable(path)
        index = int(path.stem.removeprefix("pair-"))
        if path.name != f"pair-{index:02d}.json" or index not in range(p.COUNT) or index in seen:
            raise ValueError("foreign/duplicate allocation")
        seen.add(index)
        # Completion metadata published by runner independently binds namespace,
        # allocation and science, without opening scientific result payloads.
        meta_path = path.with_suffix(".meta.json")
        verify_immutable(meta_path)
        meta = json.loads(meta_path.read_text())
        expected = {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "allocation": p.allocation()[index],
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
        }
        if meta != expected:
            raise ValueError("foreign unit completion metadata")
        allowed = any(
            path.resolve().is_relative_to(Path(json.loads(r.read_text())["output_dir"]).resolve())
            for r in receipts
        )
        if not allowed:
            raise ValueError("unit outside verified supervisor output roots")
        result["units"].append(
            {
                "index": index,
                "unit_id": unit_id(p.NAMESPACE, "watch6", index),
                "sha256": descriptor["sha256"],
                "size": descriptor["size"],
            }
        )
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
