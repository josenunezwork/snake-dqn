"""Fail-closed replay comparison for Phase-0 Mac/x86 CPU parity receipts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.phase0_search_20261019.runner import canonical_sha256, read_json, write_create_only


def ulp_distance(a: float, b: float) -> int:
    """Distance in ordered float32 representations; rejects non-float32 inputs."""
    left, right = np.float32(a), np.float32(b)
    if not (np.isfinite(left) and np.isfinite(right)):
        return 2**31
    if left == 0 and right == 0:
        return 0
    li, ri = int(left.view(np.int32)), int(right.view(np.int32))
    li = li if li >= 0 else 0x80000000 - li
    ri = ri if ri >= 0 else 0x80000000 - ri
    return abs(li - ri)


def tie_eligible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Only a first action mismatch at an <=4-ULP top tie with same pre-state is eligible."""
    if left.get("searched") or right.get("searched"):
        return False
    if left.get("pre_world_digest") != right.get("pre_world_digest"):
        return False
    if list(left.get("selection_mask", [])) != list(right.get("selection_mask", [])):
        return False
    mask = np.asarray(left.get("selection_mask", []), dtype=bool)
    ql, qr = np.asarray(left.get("q", []), dtype=np.float32), np.asarray(
        right.get("q", []), dtype=np.float32
    )
    if len(mask) != len(ql) or len(ql) != len(qr) or not mask.any():
        return False
    indices = np.flatnonzero(mask)
    best = int(indices[np.argmax(ql[indices])])
    other = (
        int(left.get("action", -1))
        if int(left.get("action", -1)) != best
        else int(right.get("action", -1))
    )
    if other not in indices:
        return False
    return (
        ulp_distance(float(ql[best]), float(ql[other])) <= 4
        and ulp_distance(float(qr[best]), float(qr[other])) <= 4
    )


def compare_records(mac: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one world receipt. A tie does not pass unless forced replay later agrees."""
    if not isinstance(mac.get("arms"), Mapping) or not isinstance(other.get("arms"), Mapping):
        return {"pass": False, "reason": "missing_arms"}
    if mac.get("world_id") != other.get("world_id") or mac.get("world_digest") != other.get(
        "world_digest"
    ):
        return {"pass": False, "reason": "world_identity_mismatch"}
    for arm in ("b8", "search"):
        if arm not in mac["arms"] or arm not in other["arms"]:
            return {"pass": False, "reason": f"missing_{arm}_arm"}
        left, right = mac["arms"][arm].get("trace", []), other["arms"][arm].get("trace", [])
        if not left or not right:
            return {"pass": False, "reason": f"{arm}:empty_trace"}
        for index, (a, b) in enumerate(zip(left, right)):
            required = {
                "pre_world_digest",
                "post_world_digest",
                "q",
                "mask",
                "selection_mask",
                "action",
            }
            if not required <= set(a) or not required <= set(b):
                return {"pass": False, "reason": f"{arm}:incomplete_trace", "frame": index}
            same = a.get("action") == b.get("action") and a.get("post_world_digest") == b.get(
                "post_world_digest"
            )
            if same:
                continue
            if a.get("action") != b.get("action") and tie_eligible(a, b):
                return {
                    "pass": False,
                    "reason": f"{arm}:tie_requires_forced_mac_replay",
                    "frame": index,
                    "tie_eligible": True,
                    "mac_action": a.get("action"),
                    "other_action": b.get("action"),
                }
            return {
                "pass": False,
                "reason": f"{arm}:divergence",
                "frame": index,
                "tie_eligible": False,
            }
        if len(left) != len(right):
            return {"pass": False, "reason": f"{arm}:trace_length"}
        if mac["arms"][arm].get("mass_integral") != other["arms"][arm].get("mass_integral") or mac[
            "arms"
        ][arm].get("survived") != other["arms"][arm].get("survived"):
            return {"pass": False, "reason": f"{arm}:endpoint_mismatch"}
    return {"pass": True, "reason": "exact_action_and_world_identity"}


def compare_merged(mac_path: Path, other_path: Path, out: Path) -> dict[str, Any]:
    mac, other = read_json(mac_path), read_json(other_path)
    mac_rows, other_rows = mac.get("worlds", []), other.get("worlds", [])
    if not mac.get("complete") or not other.get("complete") or not mac_rows or not other_rows:
        report = {"pass": False, "reason": "incomplete_or_empty_merged_receipt"}
        write_create_only(out, report)
        return report
    for receipt in (mac, other):
        copy = dict(receipt)
        claimed = copy.pop("merged_sha256", None)
        if claimed != canonical_sha256(copy):
            report = {"pass": False, "reason": "merged_receipt_hash_mismatch"}
            write_create_only(out, report)
            return report
        binding = receipt.get("plan_binding")
        if not isinstance(binding, Mapping) or binding.get("kind") != "parity":
            report = {"pass": False, "reason": "not_a_parity_plan_receipt"}
            write_create_only(out, report)
            return report
        bank = receipt.get("bank")
        expected = (bank.get("fixtures") or {}).get("parity") if isinstance(bank, Mapping) else None
        if not isinstance(expected, list) or len(expected) != 4:
            report = {"pass": False, "reason": "missing_reserved_parity_fixture_bank"}
            write_create_only(out, report)
            return report
        expected_ids = [row.get("world_id") for row in expected if isinstance(row, Mapping)]
        if len(expected_ids) != 4 or len(expected_ids) != len(set(expected_ids)):
            report = {"pass": False, "reason": "invalid_reserved_parity_fixture_ids"}
            write_create_only(out, report)
            return report
        if {row.get("world_id") for row in receipt.get("worlds", [])} != set(expected_ids):
            report = {"pass": False, "reason": "receipt_does_not_cover_reserved_parity_worlds"}
            write_create_only(out, report)
            return report
    if mac.get("plan_binding") != other.get("plan_binding"):
        report = {"pass": False, "reason": "source_config_or_input_binding_mismatch"}
        write_create_only(out, report)
        return report
    mac_ids = [row.get("world_id") for row in mac_rows if isinstance(row, Mapping)]
    if len(mac_ids) != len(mac_rows) or len(mac_ids) != len(set(mac_ids)):
        report = {"pass": False, "reason": "duplicate_or_invalid_mac_world_ids"}
        write_create_only(out, report)
        return report
    ids = [row.get("world_id") for row in other_rows if isinstance(row, Mapping)]
    if len(ids) != len(other_rows) or len(ids) != len(set(ids)):
        report = {"pass": False, "reason": "duplicate_or_invalid_other_world_ids"}
        write_create_only(out, report)
        return report
    by_id = {row["world_id"]: row for row in other_rows}
    reports = [compare_records(row, by_id.get(row.get("world_id"), {})) for row in mac_rows]
    if set(row.get("world_id") for row in mac_rows) != set(by_id):
        reports.append({"pass": False, "reason": "parity_world_set_mismatch"})
    report = {
        "mac_merged_sha256": mac.get("merged_sha256"),
        "other_merged_sha256": other.get("merged_sha256"),
        "worlds": reports,
        "pass": all(item["pass"] for item in reports),
    }
    write_create_only(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_merged(args.mac, args.other, args.out)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
