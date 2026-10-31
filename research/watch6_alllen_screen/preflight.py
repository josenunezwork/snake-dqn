"""Measured local preflight of the real job.sh entrypoint + long-body H300 branch probe."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research.watch6_alllen_screen import protocol as p
from tools.ops.package_spec import empty_ledger, publish_json, verify_immutable


def canary_proof(operational_path: Path) -> dict[str, Any]:
    """Verify completed canary/preflight 3-arm units, live-mirror parity, and L>=1 search roots."""
    verify_immutable(operational_path)
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    contract_path = operational_path.parent / "contract.json"
    verify_immutable(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    if operational.get("exclusions") or operational.get("completed") != contract["count"]:
        raise ValueError("canary/preflight segment must complete all units without exclusions")

    expected_min_roots = max(1, int(contract["horizon"]) // 10)
    checked_units = []
    for row in operational["units"]:
        unit_path = operational_path.parent / f"unit-{int(row['index']):06d}.json"
        verify_immutable(unit_path)
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        for arm in p.ARMS:
            stats = unit["arms"][arm]
            if stats["frames"] != contract["horizon"]:
                raise ValueError(f"canary arm {arm} did not complete full horizon")
            if (
                contract.get("audit_live_mirror")
                and stats["mirror_identity_frames"] != contract["horizon"]
            ):
                raise ValueError(f"canary arm {arm} missing live-mirror parity proof")
        if unit["arms"]["bare"]["search_roots"] != 0:
            raise ValueError("bare arm unexpectedly executed search roots")
        if unit["arms"]["exact_stop_alllen"]["search_roots"] < expected_min_roots:
            raise ValueError("exact_stop_alllen arm did not exercise expected L>=1 search roots")
        checked_units.append(
            {
                "index": unit["allocation"]["index"],
                "seed": unit["allocation"]["seed"],
                "alllen_roots": unit["arms"]["exact_stop_alllen"]["search_roots"],
                "alllen_branches": unit["arms"]["exact_stop_alllen"]["branches_evaluated"],
                "mirror_identity_frames": unit["arms"]["exact_stop_alllen"][
                    "mirror_identity_frames"
                ],
            }
        )
    return {
        "namespace": contract["namespace"],
        "count": contract["count"],
        "horizon": contract["horizon"],
        "units": checked_units,
    }


def branches(out: Path, canary_operational: Path | None = None) -> list[dict[str, Any]]:
    """Measure three long-body (L=900 >= 600) H300 exact-stop roots with native kernel v8."""
    import numpy as np

    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch_replay.engine import dense_batch
    from research.watch_replay.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    configure_cpu()
    timings: list[dict[str, Any]] = []
    for seed in p.BRANCH_BENCHMARK_SEEDS:
        session = open_watch({"seed": seed}, 24, "bare")
        try:
            world = session.world
            inject_serpentines([world.sim], (900, 600, 400, 300, 150, 150), box_height=8)
            _prepare_sim(world.sim)
            world.runtime.prepare(world.sim)
            obs = build_ego_raster(
                world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            q = prepared_q(world, obs)
            legal, _ = _legal(q, world.sim.get_advisory_action_mask()[0, 0])
            started = time.monotonic()
            result = dense_batch(world, 0, q, legal)
            timings.append(
                {
                    "seed": seed,
                    "seconds": time.monotonic() - started,
                    "branches": int(result["branches_evaluated"]),
                    "chosen": int(result["chosen"]),
                    "native_v8": result["native_v8"],
                }
            )
        finally:
            session.close()
    report: dict[str, Any] = {"branch_timings": timings}
    if canary_operational is not None:
        report["canary_segment"] = canary_proof(canary_operational)
    publish_json(out, report)
    return timings


def run_local_preflight(
    out: Path,
    *,
    units: int = p.PREFLIGHT_COUNT,
    horizon: int = p.PREFLIGHT_HORIZON,
) -> dict[str, Any]:
    """Execute the real job.sh bash entrypoint on tiny excluded inputs and measure timings."""
    from research.watch_replay.prepare import release_bundle

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p.ROOT, text=True).strip()

    cli_root = out / "colab_cli_root"
    venv_bin = cli_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (cli_root / "repo").symlink_to(p.ROOT)
    (venv_bin / "python").symlink_to(Path(sys.executable).resolve())

    input_dir = out / "colab_input"
    ckpt_dir = input_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    saved_dir = p.ROOT.parent / "snake-dqn" / "saved_snakes"
    if not saved_dir.is_dir():
        saved_dir = p.ROOT / "saved_snakes"
    for ckpt_name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth"):
        (ckpt_dir / ckpt_name).symlink_to((saved_dir / ckpt_name).resolve())

    pins = json.loads(
        (p.ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    release_bundle(input_dir / "release-evidence.tar.gz", pins)

    contract = {
        **p.allocation(p.PREFLIGHT_NAMESPACE, units, horizon, audit_live_mirror=True),
        "commit": commit,
    }
    publish_json(input_dir / "preflight-contract.json", contract)
    publish_json(
        input_dir / "preflight-ledger.json",
        empty_ledger(contract["namespace"], p.UNIT_KIND, p.contract_identity(contract)),
    )

    output_dir = out / "colab_output"
    output_dir.mkdir()

    env = {
        **os.environ,
        "COLAB_CLI_ROOT": str(cli_root),
        "COLAB_INPUT_DIR": str(input_dir),
        "COLAB_OUTPUT_DIR": str(output_dir),
        "WATCH6_ALLLEN_CONTRACT": "preflight-contract.json",
        "WATCH6_ALLLEN_LEDGER": "preflight-ledger.json",
        "WATCH6_ALLLEN_WORKERS": "1",
        "SNAKE_COLAB_CANARY": "1",
        "SNAKE_DQN_DEVICE": "cpu",
    }
    cmd = ["bash", str(p.ROOT / p.SCRIPT)]
    if platform.system() == "Darwin":
        cmd = ["nice", "-n", "15", *cmd]

    started = time.monotonic()
    subprocess.run(cmd, cwd=p.ROOT, env=env, check=True)
    entrypoint_seconds = time.monotonic() - started

    branches_report = json.loads((output_dir / "branches.json").read_text(encoding="utf-8"))
    operational_report = json.loads(
        (output_dir / "units" / "operational.json").read_text(encoding="utf-8")
    )
    benchmark = {
        "schema": "watch6-alllen-screen-preflight/v1",
        "commit": commit,
        "entrypoint": p.SCRIPT,
        "entrypoint_total_seconds": entrypoint_seconds,
        "tiny_units": units,
        "tiny_horizon": horizon,
        "unit_timings": [
            {
                "index": row["index"],
                "seconds": row["seconds"],
                "arm_seconds": row["arm_seconds"],
                "summary": row["summary"],
            }
            for row in operational_report["units"]
        ],
        "branch_timings": branches_report["branch_timings"],
        "canary_segment": branches_report["canary_segment"],
        "cpu_count": os.cpu_count(),
        "scope": "excluded Mac real-entrypoint preflight (1-thread CPU, live-mirror audited)",
    }
    publish_json(out / "benchmark.json", benchmark)
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--branches-only", action="store_true")
    parser.add_argument("--canary-operational", type=Path)
    parser.add_argument("--units", type=int, default=p.PREFLIGHT_COUNT)
    parser.add_argument("--horizon", type=int, default=p.PREFLIGHT_HORIZON)
    args = parser.parse_args()
    if args.branches_only:
        branches(
            args.out.resolve(),
            args.canary_operational.resolve() if args.canary_operational else None,
        )
        return
    report = run_local_preflight(args.out, units=args.units, horizon=args.horizon)
    print(
        json.dumps(
            {
                "commit": report["commit"],
                "entrypoint_total_seconds": report["entrypoint_total_seconds"],
                "branch_timings": report["branch_timings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
