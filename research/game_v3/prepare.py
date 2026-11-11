"""Prepare Game V3.1 CUDA check Colab CLI packets (`canary` and `check` modes)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo.prepare import (
    CANARY_BASE_MINUTES,
    CANARY_TIMEOUT_MINUTES,
    CANARY_WALL_GUARD_SECONDS,
    GPU_FORECAST_RATES_UPH,
    SUPPORTED_GPUS,
    _build_canary_receipt_binding,
    _write_json_create_only,
    re_safe_attempt,
)
from research.arena_ppo.runtime import (
    sha256_file,
    write_create_only_artifact,
)
from research.colab_cli import canary as colab_canary
from research.game_v3.cuda_check import (
    CUDA_CHECK_REQUIRED_EXPORTS,
    _parse_int_list,
    build_cuda_check_plan,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILY_SCRIPT = "research/game_v3/job.sh"
FAMILY_PACKAGE = "research/game_v3"
REQUIREMENTS_RELPATH = "research/game_v3/requirements.txt"
GAME_V3_NAMESPACE = "game-v3-cuda-check-v1"

PREPARE_MODES: tuple[str, ...] = ("canary", "check")

# Science CUDA check envelope: workload sized <= 15 min on L4 (~3.8 min estimated compute +
# bootstrap); 15 min base / 30 min 2x whole-session timeout with 22 min Python wall guard (T - 8m).
CHECK_BASE_MINUTES = 15.0
CHECK_TIMEOUT_MINUTES = 30.0
CHECK_WALL_GUARD_SECONDS = (CHECK_TIMEOUT_MINUTES - 8.0) * 60.0  # 1320s (22 min)


def runtime_source_paths(commit: str) -> list[str]:
    """Return minimal hash-bound source inventory shipped to the Colab CLI runtime."""
    arena_runtime_names = {
        "__init__.py",
        "runtime.py",
    }
    game_v3_names = {
        "__init__.py",
        "config.py",
        "rng.py",
        "sim.py",
        "reference.py",
        "obs.py",
        "scripted.py",
        "bench.py",
        "cuda_check.py",
        "job.sh",
        "requirements.txt",
    }
    tracked = sorted(
        set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", commit, "research"],
                cwd=ROOT,
                text=True,
            ).splitlines()
        )
    )
    selected = [
        name
        for name in tracked
        if (
            name == "research/__init__.py"
            or (
                Path(name).parent.as_posix() == "research/arena_ppo"
                and Path(name).name in arena_runtime_names
            )
            or (
                Path(name).parent.as_posix() == "research/game_v3"
                and Path(name).name in game_v3_names
            )
        )
    ]
    required_relpaths = (
        "research/arena_ppo/__init__.py",
        "research/arena_ppo/runtime.py",
        *(
            f"research/game_v3/{req}"
            for req in (
                "__init__.py",
                "config.py",
                "rng.py",
                "sim.py",
                "obs.py",
                "scripted.py",
                "bench.py",
                "cuda_check.py",
                "job.sh",
                "requirements.txt",
            )
        ),
    )
    missing_core = sorted(req for req in required_relpaths if req not in selected)
    if missing_core:
        raise ValueError(
            f"Pinned commit {commit} is missing required game_v3 runtime files: {missing_core}"
        )
    return selected


def prepare_game_v3_packet(
    packet_dir: Path | str,
    *,
    commit: str = "HEAD",
    mode: str = "canary",
    gpu: str = "L4",
    output_dir: Path | str | None = None,
    canary_receipt: Path | str | None = None,
    allow_missing_canary_receipt: bool = False,
    parity_envs: int | None = None,
    parity_ticks: int | None = None,
    det_parity_ticks: int | None = None,
    bench_envs: Sequence[int] | None = None,
    bench_snakes: Sequence[int] | None = None,
    rebase_reset_envs: int | None = None,
    attempt: str = "attempt-01",
    allow_unmerged_commit: bool = False,
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Game V3.1 CUDA parity + benchmark (`canary` or `check`).

    Args:
        packet_dir: Fresh target packet directory on Mac.
        commit: Git revision to pin (must be an ancestor of ``origin/main`` unless
            ``allow_unmerged_commit=True`` is explicitly requested for local pre-merge checks).
        mode: ``"canary"`` (tiny CUDA smoke under the 30 min canary cap producing the receipt)
            or ``"check"`` (full ``cuda_check`` run bound to the canary receipt, sized <= 15 min
            on L4 with a 22 min Python wall guard and 30 min whole-session timeout).
        gpu: Target GPU in ``{"L4", "A100", "H100"}`` (default ``"L4"``).
        output_dir: Optional fresh absolute Mac output directory.
        canary_receipt: Family canary ``receipt.json`` path to bind for ``mode="check"``.
        parity_envs: Optional world count override for ``check`` parity.
        parity_ticks: Optional tick count override for ``check`` normal parity.
        det_parity_ticks: Optional tick count override for ``check`` deterministic parity.
        bench_envs: Optional world counts override for ``check`` benchmark grid.
        bench_snakes: Optional snake counts override for ``check`` benchmark grid.
        rebase_reset_envs: Optional world count override for ``check`` rebase/reset stage.
        attempt: Attempt slug suffix (e.g. ``"attempt-01"``).
        allow_unmerged_commit: Allow commits on local branch ahead of ``origin/main`` for
            offline validation or pre-merge branch dispatch.

    Returns:
        Summary dictionary including ``manifest_path``, ``manifest_sha256``, and ``check_plan``.
    """
    if mode not in PREPARE_MODES:
        raise ValueError(f"mode must be one of {PREPARE_MODES}, got {mode!r}")
    gpu_norm = str(gpu).strip().upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"gpu must be one of {SUPPORTED_GPUS} (T4 unsupported), got {gpu!r}")
    if not re_safe_attempt(attempt):
        raise ValueError(f"Invalid attempt slug {attempt!r}")
    if mode == "canary" and canary_receipt is not None:
        raise ValueError("canary mode produces a canary receipt and cannot bind --canary-receipt")
    if mode == "check" and canary_receipt is None and not allow_missing_canary_receipt:
        raise ValueError(
            "--mode check requires canary_receipt <path/to/receipt.json> "
            "(pass allow_missing_canary_receipt=True only for offline schema inspection)"
        )

    canary_block: dict[str, Any] | None = None
    if mode == "check" and canary_receipt is not None:
        canary_block = _build_canary_receipt_binding(canary_receipt, gpu_norm)

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    allowed_refs = ("HEAD", "origin/main") if allow_unmerged_commit else ("origin/main",)
    if not any(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolved_commit, ref],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).returncode
        == 0
        for ref in allowed_refs
    ):
        raise ValueError(
            f"Refusing to prepare packet from unmerged commit {resolved_commit}: "
            f"commit is not an ancestor of {' or '.join(allowed_refs)}"
        )

    source_paths = runtime_source_paths(resolved_commit)

    gpu_slug = gpu_norm.lower()
    if output_dir is None:
        output_dir = (
            ROOT.parent
            / "snake-dqn-artifacts"
            / "big-lever"
            / "exp"
            / "argon-20261026"
            / f"game-v3-{mode}-{gpu_slug}-{attempt}"
            / "output"
        )
    out_path = Path(output_dir).resolve()
    if out_path.exists():
        raise FileExistsError(f"output_dir must be a fresh non-existent directory: {out_path}")

    pkt = Path(packet_dir).resolve()
    pkt.mkdir(parents=True, exist_ok=False)

    # Identical initial verified-ledger.json across `canary` and `check` so `colab_canary.check`
    # validates matching input count, destinations, and SHA-256 digests.
    pkt_inputs_dir = pkt / "inputs"
    pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, Any] = {
        "schema": "game-v3-verified-ledger/v1",
        "namespace": GAME_V3_NAMESPACE,
        "unit_kind": "stage",
        "skip_verified": True,
        "verified_units": [],
    }
    ledger_path = pkt_inputs_dir / "verified-ledger.json"
    ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger_sha = write_create_only_artifact(ledger_path, ledger_bytes)
    _write_json_create_only(pkt / "verified-ledger.json", ledger)
    inputs_list: list[dict[str, str]] = [
        {
            "path": str(ledger_path),
            "dest": "verified-ledger.json",
            "sha256": ledger_sha,
        }
    ]

    rate_uph = GPU_FORECAST_RATES_UPH[gpu_norm]
    extra_env: dict[str, str] = {}

    if mode == "canary":
        check_plan = build_cuda_check_plan(
            device=torch.device("cuda"),
            parity_envs=4,
            parity_ticks=50,
            det_parity_ticks=25,
            parity_snakes=12,
            parity_arena_radius=120.0,
            rebase_threshold=24,
            bench_envs=(8,),
            bench_snakes_list=(12,),
            learner_slots=4,
            bench_arena_radius=180.0,
            warmup_steps=2,
            bench_steps=4,
            stage_steps=2,
            rebase_reset_envs=8,
            rebase_reset_snakes=12,
        )
        est_minutes = CANARY_BASE_MINUTES
        timeout_minutes = CANARY_TIMEOUT_MINUTES
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["GAME_V3_MAX_WALL_SECONDS"] = str(int(CANARY_WALL_GUARD_SECONDS))
        extra_env["SNAKE_COLAB_CANARY"] = "1"
        extra_env["SNAKE_COLAB_CANARY_PACKAGE"] = FAMILY_PACKAGE
        job_id = f"game-v3-canary-{gpu_slug}-{attempt}"
        max_export_bytes = 64 * 1024 * 1024
        resume_fields: dict[str, Any] = {
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    f"Game V3.1 {gpu_norm} family canary smoke (14 min base / 28 min timeout): "
                    "4-world x 50-tick CUDA vs CPU bitwise parity (incl. forced rebase_subticks, "
                    "reset_worlds, and 6-cardinal action selection) + tiny benchmark & export"
                ),
            }
        }
        canary_block = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": FAMILY_SCRIPT,
        }
    else:
        eff_p_envs = 32 if parity_envs is None else int(parity_envs)
        eff_p_ticks = 1000 if parity_ticks is None else int(parity_ticks)
        eff_det_ticks = min(eff_p_ticks, 250) if det_parity_ticks is None else int(det_parity_ticks)
        eff_b_envs = (64, 256) if bench_envs is None else tuple(int(v) for v in bench_envs)
        eff_b_snakes = (24, 64) if bench_snakes is None else tuple(int(v) for v in bench_snakes)
        eff_rr_envs = 256 if rebase_reset_envs is None else int(rebase_reset_envs)

        check_plan = build_cuda_check_plan(
            device=torch.device("cuda"),
            parity_envs=eff_p_envs,
            parity_ticks=eff_p_ticks,
            det_parity_ticks=eff_det_ticks,
            parity_snakes=24,
            parity_arena_radius=180.0,
            rebase_threshold=400,
            bench_envs=eff_b_envs,
            bench_snakes_list=eff_b_snakes,
            learner_slots=8,
            bench_arena_radius=600.0,
            warmup_steps=10,
            bench_steps=50,
            stage_steps=20,
            rebase_reset_envs=eff_rr_envs,
            rebase_reset_snakes=24,
        )
        if float(check_plan["estimated_l4_wall_seconds"]) > CHECK_BASE_MINUTES * 60.0:
            raise ValueError(
                f"check workload estimated L4 wall time "
                f"({check_plan['estimated_l4_wall_seconds']} s) exceeds "
                f"{CHECK_BASE_MINUTES * 60.0:.0f} s (~15 min) budget"
            )

        if parity_envs is not None:
            extra_env["GAME_V3_PARITY_ENVS"] = str(eff_p_envs)
        if parity_ticks is not None:
            extra_env["GAME_V3_PARITY_TICKS"] = str(eff_p_ticks)
        if det_parity_ticks is not None:
            extra_env["GAME_V3_DET_PARITY_TICKS"] = str(eff_det_ticks)
        if bench_envs is not None:
            extra_env["GAME_V3_BENCH_ENVS"] = ",".join(str(v) for v in eff_b_envs)
        if bench_snakes is not None:
            extra_env["GAME_V3_BENCH_SNAKES"] = ",".join(str(v) for v in eff_b_snakes)
        if rebase_reset_envs is not None:
            extra_env["GAME_V3_REBASE_RESET_ENVS"] = str(eff_rr_envs)

        est_minutes = CHECK_BASE_MINUTES
        timeout_minutes = CHECK_TIMEOUT_MINUTES
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["GAME_V3_MAX_WALL_SECONDS"] = str(int(CHECK_WALL_GUARD_SECONDS))
        job_id = f"game-v3-check-{gpu_slug}-{attempt}"
        max_export_bytes = 64 * 1024 * 1024
        resume_fields = {
            "resume_exemption": {
                "kind": "acceptance",
                "reason": (
                    f"Game V3.1 {gpu_norm} CUDA bitwise parity + throughput/VRAM check "
                    "(<15m L4 workload, 22m Python wall guard, 30m session timeout; per-stage "
                    "create-only JSONs + signal/wall-safe partial cuda_check_result.json)"
                ),
            }
        }

    required_exports = list(CUDA_CHECK_REQUIRED_EXPORTS)
    unit_cap = math.ceil(((timeout_minutes * rate_uph) / 60.0 - 1e-12) * 10000.0) / 10000.0
    manifest: dict[str, Any] = {
        "job_id": job_id[:64],
        "commit": resolved_commit,
        "script": FAMILY_SCRIPT,
        "device": "cuda",
        "source_paths": source_paths,
        "requirements": {
            "gpu": gpu_norm,
            "allowed_gpus": [gpu_norm],
            "min_vcpu": 2,
            "est_minutes": est_minutes,
        },
        "timeout_minutes": timeout_minutes,
        "output_dir": str(out_path),
        "max_export_bytes": int(max_export_bytes),
        "inputs": inputs_list,
        "env": {
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS_RELPATH,
            "GAME_V3_SOURCE_COMMIT": resolved_commit,
            "GAME_V3_MODE": mode,
            "SNAKE_DQN_DEVICE": "cuda",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **extra_env,
        },
        "required_exports": required_exports,
        "budget": {
            "rate_units_per_hour": rate_uph,
            "unit_cap": unit_cap,
            "reserve_units": 0,
            "other_assignments": {},
        },
        **resume_fields,
        "requirements_file": REQUIREMENTS_RELPATH,
        "technical_envelope": {
            "base_seconds": base_sec,
            "effective_seconds": eff_sec,
            "forecast_unit_cap": unit_cap,
            "mode": mode,
            "gpu": gpu_norm,
        },
    }
    if allow_unmerged_commit:
        manifest["unmerged_dev_packet"] = True
    if canary_block is not None:
        manifest["canary"] = canary_block

    try:
        if mode == "canary" or canary_receipt is not None:
            colab_canary.check(manifest, ROOT)
    except BaseException:
        shutil.rmtree(pkt, ignore_errors=True)
        raise

    manifest_path = pkt / "dispatch-manifest.json"
    _write_json_create_only(manifest_path, manifest)
    return {
        "packet_dir": str(pkt),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mode": mode,
        "job_id": manifest["job_id"],
        "commit": resolved_commit,
        "gpu": gpu_norm,
        "timeout_minutes": timeout_minutes,
        "est_minutes": est_minutes,
        "max_wall_seconds": int(extra_env["GAME_V3_MAX_WALL_SECONDS"]),
        "max_export_bytes": int(max_export_bytes),
        "unit_cap": unit_cap,
        "check_plan": check_plan,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m research.game_v3.prepare``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Target packet directory.")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=PREPARE_MODES, default="canary")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="L4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--allow-missing-canary-receipt", action="store_true")
    parser.add_argument("--parity-envs", type=int, default=None)
    parser.add_argument("--parity-ticks", type=int, default=None)
    parser.add_argument("--det-parity-ticks", type=int, default=None)
    parser.add_argument("--bench-envs", type=_parse_int_list, default=None)
    parser.add_argument("--bench-snakes", type=_parse_int_list, default=None)
    parser.add_argument("--rebase-reset-envs", type=int, default=None)
    parser.add_argument("--attempt", type=str, default="attempt-01")
    parser.add_argument("--allow-unmerged-commit", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "check" and args.canary_receipt is None:
        if not args.allow_missing_canary_receipt:
            parser.error(
                "--mode check requires --canary-receipt <path/to/receipt.json> "
                "(pass --allow-missing-canary-receipt only for offline schema inspection)"
            )

    info = prepare_game_v3_packet(
        args.packet,
        commit=args.commit,
        mode=args.mode,
        gpu=args.gpu,
        output_dir=args.output_dir,
        canary_receipt=args.canary_receipt,
        allow_missing_canary_receipt=bool(args.allow_missing_canary_receipt),
        parity_envs=args.parity_envs,
        parity_ticks=args.parity_ticks,
        det_parity_ticks=args.det_parity_ticks,
        bench_envs=args.bench_envs,
        bench_snakes=args.bench_snakes,
        rebase_reset_envs=args.rebase_reset_envs,
        attempt=args.attempt,
        allow_unmerged_commit=bool(args.allow_unmerged_commit),
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
