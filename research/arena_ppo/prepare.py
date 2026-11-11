"""Prepare Stage 1 Arena PPO Colab CLI packets (`canary`, `ruler`, and `train` modes)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from research.arena_ppo.ppo import (
    ARENA_PPO_NAMESPACE,
    ARENA_PPO_SCHEMA,
    ArenaPPOConfig,
    estimate_max_export_bytes,
    find_newest_verified_checkpoint,
)
from research.arena_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
)
from research.arena_ppo.runtime import (
    sha256_file,
    verify_completed_artifact,
    write_create_only_artifact,
)
from research.colab_cli import canary as colab_canary

ROOT = Path(__file__).resolve().parents[2]
FAMILY_SCRIPT = "research/arena_ppo/job.sh"
FAMILY_PACKAGE = "research/arena_ppo"
REQUIREMENTS_RELPATH = "research/arena_ppo/requirements.txt"

PREPARE_MODES: tuple[str, ...] = ("canary", "ruler", "train")
SUPPORTED_GPUS: tuple[str, ...] = ("L4", "A100", "H100")
GPU_FORECAST_RATES_UPH: dict[str, float] = {
    "L4": 1.54,
    "A100": 8.0,
    "H100": 12.0,
}

# Instrumented family canary envelope: 6 min base / 12 min 2x timeout (<= 12 min cap)
CANARY_BASE_MINUTES = 6.0
CANARY_TIMEOUT_MINUTES = 12.0
CANARY_WALL_GUARD_SECONDS = 540.0

# Uninstrumented science ruler envelope: 30 min base / 60 min runaway guard; Python guard = T - 8m
RULER_BASE_MINUTES = 30.0
RULER_TIMEOUT_MINUTES = 60.0
RULER_WALL_GUARD_SECONDS = (RULER_TIMEOUT_MINUTES - 8.0) * 60.0  # 3120s (52 min)


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing packet file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_source_paths(commit: str) -> list[str]:
    """Return minimal hash-bound source inventory shipped to the Colab CLI runtime."""
    gpu_spike_names = {
        "__init__.py",
        "device_sim.py",
        "stage_timing.py",
        "r2_cache.py",
        "anchors_device.py",
        "masks_device.py",
        *(
            f"{stem}.{suffix}"
            for stem in ("move_cuda", "event_cuda")
            for suffix in ("py", "cpp", "cu")
        ),
    }
    arena_ppo_names = {
        "__init__.py",
        "runtime.py",
        "sim.py",
        "obs.py",
        "net.py",
        "ppo.py",
        "ruler.py",
        "job.sh",
        "requirements.txt",
    }
    tracked = sorted(
        set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", commit, "src", "configs", "research"],
                cwd=ROOT,
                text=True,
            ).splitlines()
        )
    )
    selected = [
        name
        for name in tracked
        if (
            name in {"src/__init__.py", "research/__init__.py", "configs/mechanics_v2.yaml"}
            or (
                name.startswith(("src/core/", "src/model/", "src/simd_env/", "src/evaluation/"))
                and name.endswith(".py")
            )
            or (
                Path(name).parent.as_posix() == "research/gpu_sim_spike"
                and Path(name).name in gpu_spike_names
            )
            or (
                Path(name).parent.as_posix() == "research/arena_ppo"
                and Path(name).name in arena_ppo_names
            )
        )
    ]
    missing_core = sorted(
        f"research/arena_ppo/{req}"
        for req in ("__init__.py", "job.sh", "requirements.txt", "ruler.py", "ppo.py")
        if f"research/arena_ppo/{req}" not in selected
    )
    if missing_core:
        raise ValueError(
            f"Pinned commit {commit} is missing required research/arena_ppo files: {missing_core}"
        )
    return selected


def _stage_prior_checkpoint(
    prior_output_dir: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Stage newest verified non-probe checkpoint from a prior attempt directory."""
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")

    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    chosen_ckpt: Path | None = None
    for root_cand in search_roots:
        ckpt_sub = root_cand / "checkpoints"
        if not ckpt_sub.is_dir():
            continue
        final_cand = ckpt_sub / "final.pt"
        if (
            final_cand.is_file()
            and Path(f"{final_cand}.complete").is_file()
            and Path(f"{final_cand}.sha256").is_file()
        ):
            verify_completed_artifact(final_cand)
            import torch as _torch

            obj = _torch.load(final_cand, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and not bool(obj.get("canary_export_probe")):
                if (
                    obj.get("schema") != ARENA_PPO_SCHEMA
                    or obj.get("namespace") != ARENA_PPO_NAMESPACE
                ):
                    raise ValueError(f"Foreign checkpoint schema/namespace at {final_cand}")
                chosen_ckpt = final_cand
                break

        newest = find_newest_verified_checkpoint(ckpt_sub)
        if newest is not None:
            chosen_ckpt = newest
            break

    if chosen_ckpt is None:
        raise FileNotFoundError(
            f"No verified non-probe Arena PPO checkpoint found under {prior_dir}"
        )

    dst_ckpt = pkt_inputs_dir / chosen_ckpt.name
    ckpt_sha = write_create_only_artifact(
        dst_ckpt, lambda tmp, src=chosen_ckpt: shutil.copyfile(src, tmp)
    )
    inputs_list.append(
        {
            "path": str(dst_ckpt),
            "dest": chosen_ckpt.name,
            "sha256": ckpt_sha,
        }
    )
    entry = {
        "relpath": chosen_ckpt.name,
        "sha256": ckpt_sha,
        "size_bytes": int(dst_ckpt.stat().st_size),
    }
    ledger["resume_checkpoint"] = entry
    ledger["verified_units"].append({"unit_id": f"staged:{chosen_ckpt.stem}", **entry})
    return entry


def _stage_prior_ruler_cells(
    prior_output_dir: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
) -> list[str]:
    """Stage verified `cells/<cell_id>.json` files from a partial ruler directory."""
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")

    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    staged_ids: list[str] = []
    for root_cand in search_roots:
        cells_sub = root_cand / "cells"
        if not cells_sub.is_dir():
            continue
        for cell_json in sorted(cells_sub.glob("*.json")):
            if not (
                Path(f"{cell_json}.complete").is_file() and Path(f"{cell_json}.sha256").is_file()
            ):
                continue
            verify_completed_artifact(cell_json)
            rel_dest = f"cells/{cell_json.name}"
            dst_file = pkt_inputs_dir / "cells" / cell_json.name
            c_sha = write_create_only_artifact(
                dst_file, lambda tmp, src=cell_json: shutil.copyfile(src, tmp)
            )
            inputs_list.append(
                {
                    "path": str(dst_file),
                    "dest": rel_dest,
                    "sha256": c_sha,
                }
            )
            ledger["verified_units"].append(
                {
                    "unit_id": f"cell:{cell_json.stem}",
                    "relpath": rel_dest,
                    "sha256": c_sha,
                    "size_bytes": int(dst_file.stat().st_size),
                }
            )
            staged_ids.append(cell_json.stem)
        if staged_ids:
            break

    if not staged_ids:
        raise FileNotFoundError(f"No verified ruler cells/*.json found under {prior_dir}")
    return staged_ids


def _build_canary_receipt_binding(
    canary_receipt: Path | str,
    gpu_norm: str,
) -> dict[str, Any]:
    rcpt_path = Path(canary_receipt).resolve()
    if not rcpt_path.is_file():
        raise FileNotFoundError(f"Canary receipt not found: {rcpt_path}")
    rcpt_data = json.loads(rcpt_path.read_text(encoding="utf-8"))
    rcpt_hw = str((rcpt_data.get("canary_evidence") or {}).get("hardware") or "").strip()
    if rcpt_hw and rcpt_hw.upper() != gpu_norm:
        raise ValueError(f"--gpu {gpu_norm!r} does not match canary receipt hardware {rcpt_hw!r}")
    return {
        "receipt": {
            "path": str(rcpt_path),
            "sha256": sha256_file(rcpt_path),
        }
    }


def prepare_arena_ppo_packet(
    packet_dir: Path | str,
    *,
    commit: str = "HEAD",
    mode: str = "canary",
    gpu: str = "L4",
    hours: float = 2.0,
    output_dir: Path | str | None = None,
    prior_output_dir: Path | str | None = None,
    prior_kind: str = "auto",
    canary_receipt: Path | str | None = None,
    ppo_config: ArenaPPOConfig | None = None,
    measured_checkpoint_bytes: int | None = None,
    compile_model: bool = False,
    attempt: str = "attempt-01",
    allow_unmerged_commit: bool = False,
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Stage 1 Arena PPO (`canary`, `ruler`, or `train`).

    Args:
        packet_dir: Fresh target packet directory on Mac.
        commit: Git revision to pin (must be an ancestor of ``origin/main`` unless
            ``allow_unmerged_commit=True`` is explicitly requested for local pre-merge checks).
        mode: ``"canary"`` (<= 12 min instrumented family probe), ``"ruler"`` (<= 60 min
            uninstrumented science ruler bound to canary receipt), or ``"train"`` (resumable
            training session bound to canary receipt).
        gpu: Target GPU in ``{"L4", "A100", "H100"}`` (default ``"L4"``).
        hours: Session wall timeout in hours for ``mode="train"`` (`timeout = hours * 60` min).
        output_dir: Optional fresh absolute Mac output directory.
        prior_output_dir: Optional prior attempt output directory for resume staging.
        prior_kind: ``"auto"``, ``"checkpoint"``, or ``"ruler_cells"`` when staging prior output.
        canary_receipt: Family canary `receipt.json` path to bind for ``ruler`` and ``train``.
        ppo_config: Optional ``ArenaPPOConfig`` serialized into ``env["ARENA_PPO_CONFIG_JSON"]``.
        measured_checkpoint_bytes: Optional empirically measured single-checkpoint size in bytes
            for ``train`` ``max_export_bytes`` sizing.
        compile_model: Enable ``torch.compile`` flag in runtime environment.
        attempt: Attempt slug suffix (e.g. ``"attempt-01"``).
        allow_unmerged_commit: Allow commits on local branch ahead of ``origin/main`` for
            offline validation only.

    Returns:
        Summary dictionary including ``manifest_path`` and ``manifest_sha256``.
    """
    if mode not in PREPARE_MODES:
        raise ValueError(f"mode must be one of {PREPARE_MODES}, got {mode!r}")
    gpu_norm = str(gpu).strip().upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"gpu must be one of {SUPPORTED_GPUS} (T4 unsupported), got {gpu!r}")
    if not re_safe_attempt(attempt):
        raise ValueError(f"Invalid attempt slug {attempt!r}")

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
    pkt = Path(packet_dir).resolve()
    pkt.mkdir(parents=True, exist_ok=False)

    gpu_slug = gpu_norm.lower()
    if output_dir is None:
        output_dir = (
            ROOT.parent
            / "snake-dqn-artifacts"
            / "big-lever"
            / "exp"
            / "argon-20261026"
            / f"arena-ppo-{mode}-{gpu_slug}-{attempt}"
            / "output"
        )
    out_path = Path(output_dir).resolve()
    if out_path.exists():
        raise FileExistsError(f"output_dir must be a fresh non-existent directory: {out_path}")

    # Identical initial verified-ledger.json across `canary`, `ruler`, and `train` attempt-01
    # so `colab_canary.check` validates matching input count, destinations, and SHA-256 digests.
    pkt_inputs_dir = pkt / "inputs"
    pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, Any] = {
        "schema": "arena-ppo-verified-ledger/v1",
        "namespace": ARENA_PPO_NAMESPACE,
        "unit_kind": "checkpoint",
        "skip_verified": True,
        "verified_units": [],
    }
    inputs_list: list[dict[str, str]] = []
    if prior_output_dir is not None:
        eff_prior_kind = prior_kind
        if eff_prior_kind == "auto":
            eff_prior_kind = "ruler_cells" if mode == "ruler" else "checkpoint"
        if eff_prior_kind == "ruler_cells":
            _stage_prior_ruler_cells(Path(prior_output_dir), pkt_inputs_dir, inputs_list, ledger)
        elif eff_prior_kind == "checkpoint":
            _stage_prior_checkpoint(Path(prior_output_dir), pkt_inputs_dir, inputs_list, ledger)
        else:
            raise ValueError(f"Unsupported prior_kind={prior_kind!r}")

    ledger_path = pkt_inputs_dir / "verified-ledger.json"
    ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger_sha = write_create_only_artifact(ledger_path, ledger_bytes)
    _write_json_create_only(pkt / "verified-ledger.json", ledger)
    inputs_list.append(
        {
            "path": str(ledger_path),
            "dest": "verified-ledger.json",
            "sha256": ledger_sha,
        }
    )

    rate_uph = GPU_FORECAST_RATES_UPH[gpu_norm]
    extra_env: dict[str, str] = {}
    canary_block: dict[str, Any] | None = None

    if mode == "canary":
        if canary_receipt is not None:
            raise ValueError(
                "canary mode produces a canary receipt and cannot bind --canary-receipt"
            )
        est_minutes = CANARY_BASE_MINUTES
        timeout_minutes = CANARY_TIMEOUT_MINUTES
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(int(CANARY_WALL_GUARD_SECONDS))
        extra_env["SNAKE_COLAB_CANARY"] = "1"
        extra_env["SNAKE_COLAB_CANARY_PACKAGE"] = FAMILY_PACKAGE
        job_id = f"arena-ppo-canary-{gpu_slug}-{attempt}"
        required_exports = list(CANARY_REQUIRED_EXPORTS)
        max_export_bytes = 256 * 1024 * 1024
        resume_fields: dict[str, Any] = {
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    f"Stage 1 Arena PPO {gpu_norm} family canary probe (<= 12 min): real "
                    "indexed parity + tiny ruler + tiny 2-update train/checkpoint/resume"
                ),
            }
        }
        canary_block = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": FAMILY_SCRIPT,
        }
    elif mode == "ruler":
        est_minutes = RULER_BASE_MINUTES
        timeout_minutes = RULER_TIMEOUT_MINUTES
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(int(RULER_WALL_GUARD_SECONDS))
        if compile_model:
            extra_env["ARENA_PPO_COMPILE"] = "1"
        job_id = f"arena-ppo-ruler-{gpu_slug}-{attempt}"
        required_exports = list(RULER_REQUIRED_EXPORTS)
        max_export_bytes = 128 * 1024 * 1024
        resume_fields = {
            "resume_exemption": {
                "kind": "acceptance",
                "reason": (
                    f"Stage 1 Arena PPO {gpu_norm} throughput & VRAM science ruler (60m runaway "
                    "guard; per-cell create-only JSONs + signal-safe partial ruler_result.json)"
                ),
            }
        }
        if canary_receipt is not None:
            canary_block = _build_canary_receipt_binding(canary_receipt, gpu_norm)
    else:
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError(f"--hours must be finite and positive, got {hours}")
        timeout_minutes = round(float(hours) * 60.0, 2)
        if timeout_minutes < 6.0:
            raise ValueError(f"--hours results in timeout_minutes={timeout_minutes} < 6.0 min")
        est_minutes = round(timeout_minutes / 2.0, 2)
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        headroom_sec = min(480.0, max(60.0, eff_sec * 0.08))
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(int(max(30.0, eff_sec - headroom_sec)))
        cfg_obj = (
            ppo_config if ppo_config is not None else ArenaPPOConfig(compile_model=compile_model)
        )
        extra_env["ARENA_PPO_CONFIG_JSON"] = json.dumps(cfg_obj.to_dict(), sort_keys=True)
        job_id = f"arena-ppo-train-{gpu_slug}-{attempt}"
        required_exports = list(TRAIN_REQUIRED_EXPORTS)
        max_export_bytes = estimate_max_export_bytes(
            cfg_obj, measured_checkpoint_bytes=measured_checkpoint_bytes
        )
        resume_fields = {
            "resume": {
                "unit_kind": "checkpoint",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        }
        if canary_receipt is not None:
            canary_block = _build_canary_receipt_binding(canary_receipt, gpu_norm)

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
            "ARENA_PPO_SOURCE_COMMIT": resolved_commit,
            "ARENA_PPO_MODE": mode,
            "SNAKE_DQN_DEVICE": "cuda",
            "MAX_JOBS": "4",
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

    if mode == "canary" or canary_receipt is not None:
        colab_canary.check(manifest, ROOT)

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
        "max_export_bytes": int(max_export_bytes),
        "unit_cap": unit_cap,
    }


def re_safe_attempt(value: str) -> bool:
    """Check lowercase alphanumeric/hyphen attempt identifier."""
    return bool(value) and all(ch.islower() or ch.isdigit() or ch in "-_" for ch in value)


def main() -> int:
    """CLI entrypoint for `python -m research.arena_ppo.prepare`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Target packet directory.")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=PREPARE_MODES, default="canary")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="L4")
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Whole-session timeout in hours for --mode train (default: 2.0).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior-output-dir", type=Path, default=None)
    parser.add_argument(
        "--prior-kind", choices=("auto", "checkpoint", "ruler_cells"), default="auto"
    )
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--allow-missing-canary-receipt", action="store_true")
    parser.add_argument("--ppo-config-json", type=str, default=None)
    parser.add_argument("--measured-checkpoint-bytes", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--attempt", type=str, default="attempt-01")
    parser.add_argument("--allow-unmerged-commit", action="store_true")
    args = parser.parse_args()

    if args.mode in ("ruler", "train") and args.canary_receipt is None:
        if not args.allow_missing_canary_receipt:
            parser.error(
                f"--mode {args.mode} requires --canary-receipt <path/to/receipt.json> "
                "(pass --allow-missing-canary-receipt only for offline schema inspection)"
            )

    ppo_cfg = (
        ArenaPPOConfig.from_json(args.ppo_config_json) if args.ppo_config_json is not None else None
    )
    info = prepare_arena_ppo_packet(
        args.packet,
        commit=args.commit,
        mode=args.mode,
        gpu=args.gpu,
        hours=args.hours,
        output_dir=args.output_dir,
        prior_output_dir=args.prior_output_dir,
        prior_kind=args.prior_kind,
        canary_receipt=args.canary_receipt,
        ppo_config=ppo_cfg,
        measured_checkpoint_bytes=args.measured_checkpoint_bytes,
        compile_model=bool(args.compile),
        attempt=args.attempt,
        allow_unmerged_commit=bool(args.allow_unmerged_commit),
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
