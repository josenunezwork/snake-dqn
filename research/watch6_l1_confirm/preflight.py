"""Mac real-entrypoint preflight benchmark for Watch6-L1-V1 on excluded fixture seed."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR, run
from tools.ops.package_spec import publish_json, verify_immutable


def preflight(out: Path, *, horizon: int = 30) -> dict:
    """Run real `watch6_l1_confirm.run` entrypoint on 1 fixture world + H=300 branch check."""
    if platform.system() != "Darwin":
        raise ValueError("Mac preflight must run on Darwin")
    if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
        os.environ["SNAKE_CKPT_DIR"] = str(DEFAULT_LOCAL_SAVED_DIR)
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")

    out = out.resolve()
    efficacy_dir = out / "exports/efficacy"
    perf_path = out / "exports/performance.json"
    started = time.monotonic()
    summary = run(
        out=efficacy_dir,
        ledger_path=None,
        workers=1,
        shards=p.SHARDS,
        shard=0,
        fixture=True,
        fixture_horizon=horizon,
        fixture_count=1,
        canary_performance_out=perf_path,
        local_smoke=True,
    )
    elapsed = time.monotonic() - started

    unit_path = efficacy_dir / "fixture-world-00.json"
    unit_desc = verify_immutable(unit_path)
    verify_immutable(unit_path.with_suffix(".meta.json"))
    verify_immutable(efficacy_dir / "operational.json")
    verify_immutable(perf_path)

    unit_payload = json.loads(unit_path.read_text(encoding="utf-8"))
    perf_payload = json.loads(perf_path.read_text(encoding="utf-8"))
    if (
        perf_payload.get("native_v8", {}).get("kernel_rows", 0) <= 0
        or perf_payload.get("native_v8", {}).get("fallback_rows", -1) != 0
    ):
        raise ValueError("H300 branch fixture failed native v8 kernel attestation")

    benchmark = {
        "schema": "watch6-l1-mac-preflight/v1",
        "status": "PASS",
        "platform": platform.platform(),
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "fixture": True,
        "fixture_seed": p.FIXTURE_SEEDS[0],
        "fixture_horizon": horizon,
        "elapsed_seconds": elapsed,
        "unit_file": {
            "path": str(unit_path),
            "sha256": unit_desc["sha256"],
            "size": unit_desc["size"],
        },
        "arm_order": unit_payload["arm_order"],
        "arms_summary": {
            arm: {
                "frames": unit_payload["arms"][arm]["frames"],
                "mirror_identity_frames": unit_payload["arms"][arm]["mirror_identity_frames"],
                "batches": unit_payload["arms"][arm]["batches"],
                "branches_evaluated": unit_payload["arms"][arm]["branches_evaluated"],
                "native_v8_kernel_rows": unit_payload["arms"][arm]["native_v8_kernel_rows"],
            }
            for arm in p.ARMS
        },
        "h300_branch_fixture": {
            "seconds": perf_payload["seconds"],
            "branches": perf_payload["branches"],
            "frames_executed": perf_payload["frames_executed"],
            "native_v8": perf_payload["native_v8"],
        },
        "operational": summary,
    }
    publish_json(out / "benchmark.json", benchmark)
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    report = preflight(args.out, horizon=args.horizon)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
