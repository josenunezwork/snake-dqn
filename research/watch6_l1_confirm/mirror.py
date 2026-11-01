"""Mac 3-arm live/GridBatchSim mirror qualification for Watch6-L1-V1 on excluded fixture seed."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.engine import run_arm
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR
from tools.ops.package_spec import publish_json

SCHEMA = "watch6-l1-mac-mirror/v1"


def qualify(*, horizon: int = p.HORIZON) -> dict:
    """Verify full-horizon live GameSession / GridBatchSim parity across {bare, l2_prime, l1}."""
    if platform.system() != "Darwin":
        raise ValueError("Mac mirror qualification must run on Darwin")
    if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
        os.environ["SNAKE_CKPT_DIR"] = str(DEFAULT_LOCAL_SAVED_DIR)
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")

    seed = p.FIXTURE_SEEDS[0]
    arms_out: dict[str, dict] = {}
    for arm in p.ARMS:
        raw = run_arm(seed, arm, lambda: None, horizon=horizon, record=None)
        arms_out[arm] = {
            "frames": raw["frames"],
            "mirror_identity_frames": raw["mirror_identity_frames"],
            "respawns_per_slot": raw["respawns_per_slot"],
            "batches": raw["batches"],
            "branches_evaluated": raw["branches_evaluated"],
            "native_v8_kernel_rows": raw["native_v8_kernel_rows"],
        }
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "platform": platform.platform(),
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "fixture": True,
        "seed": seed,
        "horizon": horizon,
        "arms": arms_out,
    }
    if horizon == p.HORIZON:
        validate(report)
    return report


def validate(report: dict) -> None:
    """Validate a full 14,400-frame 3-arm Mac mirror qualification receipt."""
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("status") != "PASS"
        or report.get("namespace") != p.NAMESPACE
        or report.get("science") != p.identity()
        or report.get("fixture") is not True
        or report.get("seed") != p.FIXTURE_SEEDS[0]
        or report.get("horizon") != p.HORIZON
        or not str(report.get("platform", "")).startswith("macOS")
    ):
        raise ValueError("requires current full-14,400-frame Mac 3-arm mirror receipt")
    arms = report.get("arms", {})
    if not isinstance(arms, dict) or set(arms) != set(p.ARMS):
        raise ValueError("Mac mirror receipt missing required 3-arm coverage")
    for arm in p.ARMS:
        row = arms[arm]
        if (
            row.get("frames") != p.HORIZON
            or row.get("mirror_identity_frames") != p.HORIZON
            or len(row.get("respawns_per_slot", [])) != 6
        ):
            raise ValueError(f"incomplete 14,400-frame live/Grid mirror coverage on arm {arm}")
        if arm != "bare" and (
            int(row.get("batches", 0)) <= 0 or int(row.get("native_v8_kernel_rows", 0)) <= 0
        ):
            raise ValueError(
                f"search arm {arm} lacked search roots or native v8 kernel attestation"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=p.HORIZON)
    args = parser.parse_args()
    report = qualify(horizon=args.horizon)
    publish_json(args.out.resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
