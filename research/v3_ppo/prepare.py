"""Prepare Game V3.1 PPO Colab CLI packets (`canary`, `ruler`, and `train` modes)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo.prepare import (
    CANARY_BASE_MINUTES,
    CANARY_TIMEOUT_MINUTES,
    CANARY_WALL_GUARD_SECONDS,
    GPU_FORECAST_RATES_UPH,
    RULER_BASE_MINUTES,
    RULER_TIMEOUT_MINUTES,
    RULER_WALL_GUARD_SECONDS,
    SUPPORTED_GPUS,
    _build_canary_receipt_binding,
    _write_json_create_only,
    re_safe_attempt,
)
from research.arena_ppo.runtime import (
    sha256_file,
    verify_completed_artifact,
)
from research.colab_cli import canary as colab_canary
from research.game_v3.config import GameV3Config
from research.v3_ppo.config import (
    GPU_VRAM_BUDGET_BYTES,
    MAX_SUPERVISOR_EXPORT_BYTES,
    V3_PPO_NAMESPACE,
    VALID_NET_SIZES,
    VALID_OPPONENT_POLICIES,
    NetSize,
    OpponentPolicy,
    V3PPOConfig,
    assert_resume_config_compatible,
    default_train_envs_for_gpu,
    estimate_training_memory,
    get_net_preset,
    validate_checkpoint_export_caps,
    validate_memory_budget,
)
from research.v3_ppo.ppo import (
    resolve_resume_checkpoint,
    write_create_only_artifact,
)
from research.v3_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    DEFAULT_RULER_ENVS,
    DEFAULT_RULER_LEARNERS,
    DEFAULT_RULER_NETS,
    DEFAULT_RULER_ROLLOUT_LEN,
    DEFAULT_RULER_TIMED_UPDATES,
    DEFAULT_RULER_WARMUP_UPDATES,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
    _parse_int_csv,
    _parse_nets_csv,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILY_SCRIPT = "research/v3_ppo/job.sh"
FAMILY_PACKAGE = "research/v3_ppo"
REQUIREMENTS_RELPATH = "research/v3_ppo/requirements.txt"

PREPARE_MODES: tuple[str, ...] = ("canary", "ruler", "train")
UNLIMITED_HOURS_TOTAL_UPDATES: int = 1_000_000_000


def runtime_source_paths(commit: str) -> list[str]:
    """Return minimal hash-bound source inventory shipped to the Colab CLI runtime.

    Ships ONLY ``research/arena_ppo/{__init__,runtime}.py``, ``research/game_v3``, and
    ``research/v3_ppo`` — zero modules from the legacy grid simulator or observation layout.
    """
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
    }
    v3_ppo_names = {
        "__init__.py",
        "config.py",
        "net.py",
        "buffer.py",
        "eval.py",
        "ppo.py",
        "ruler.py",
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
            or (
                Path(name).parent.as_posix() == "research/v3_ppo"
                and Path(name).name in v3_ppo_names
            )
        )
    ]
    required_relpaths = (
        "research/arena_ppo/__init__.py",
        "research/arena_ppo/runtime.py",
        *(f"research/game_v3/{req}" for req in sorted(game_v3_names)),
        *(f"research/v3_ppo/{req}" for req in sorted(v3_ppo_names)),
    )
    missing_core = sorted(req for req in required_relpaths if req not in selected)
    if missing_core:
        raise ValueError(
            f"Pinned commit {commit} is missing required v3_ppo runtime files: {missing_core}"
        )
    return selected


def build_training_plan(
    cfg: V3PPOConfig,
    *,
    hours: float,
    expected_steps_per_sec: float,
    max_wall_seconds: float,
) -> dict[str, Any]:
    """Compute forecasted update rate and horizon for a ``train`` session."""
    agent_steps_per_update = int(cfg.rollout_len) * int(cfg.num_envs) * int(cfg.num_snakes)
    learner_steps_per_update = int(cfg.rollout_len) * int(cfg.total_learners)
    sec_per_update = float(agent_steps_per_update) / float(expected_steps_per_sec)
    expected_updates = int(max(1.0, math.floor(float(max_wall_seconds) / sec_per_update)))
    return {
        "hours": float(hours),
        "max_wall_seconds": float(max_wall_seconds),
        "expected_agent_steps_per_sec": float(expected_steps_per_sec),
        "agent_steps_per_update": agent_steps_per_update,
        "learner_steps_per_update": learner_steps_per_update,
        "expected_seconds_per_update": round(sec_per_update, 3),
        "expected_updates_in_session": expected_updates,
        "expected_agent_steps_in_session": expected_updates * agent_steps_per_update,
    }


def _check_foreign_checkpoints(ckpt_sub: Path) -> None:
    """Raise ``ValueError`` if completed checkpoints in ``ckpt_sub`` have a foreign namespace."""
    for idx_cand in sorted(ckpt_sub.glob("*.index.json")):
        if Path(f"{idx_cand}.complete").is_file() and Path(f"{idx_cand}.sha256").is_file():
            verify_completed_artifact(idx_cand)
            obj = json.loads(idx_cand.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("namespace") != V3_PPO_NAMESPACE:
                raise ValueError(f"Foreign checkpoint namespace at {idx_cand}")
    for pt_cand in sorted(ckpt_sub.glob("*.pt")):
        if Path(f"{pt_cand}.complete").is_file() and Path(f"{pt_cand}.sha256").is_file():
            verify_completed_artifact(pt_cand)
            obj = torch.load(pt_cand, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and obj.get("namespace") != V3_PPO_NAMESPACE:
                raise ValueError(f"Foreign checkpoint schema/namespace at {pt_cand}")


def _resolve_verified_prior_checkpoint(
    prior_dir: Path,
    *,
    light_resume_margin_minutes: float = 45.0,
) -> dict[str, Any] | None:
    """Locate newest verified V3 PPO checkpoint set (preferring LIGHT only past margin)."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        ckpt_sub = root_cand / "checkpoints"
        if not ckpt_sub.is_dir():
            continue
        _check_foreign_checkpoints(ckpt_sub)
        resolved = resolve_resume_checkpoint(
            ckpt_sub,
            require_markers=True,
            light_resume_margin_minutes=light_resume_margin_minutes,
        )
        if resolved is not None:
            return resolved
    return None


def _find_verified_prior_cells(prior_dir: Path) -> list[Path]:
    """Locate verified ``cells/*.json`` ruler artifacts under ``prior_dir``."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        cells_sub = root_cand / "cells"
        if not cells_sub.is_dir():
            continue
        found: list[Path] = []
        for cell_json in sorted(cells_sub.glob("*.json")):
            if not (
                Path(f"{cell_json}.complete").is_file() and Path(f"{cell_json}.sha256").is_file()
            ):
                continue
            verify_completed_artifact(cell_json)
            parsed = json.loads(cell_json.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and parsed.get("namespace") not in (None, V3_PPO_NAMESPACE):
                raise ValueError(f"Foreign ruler cell namespace at {cell_json}")
            if isinstance(parsed, dict) and "error" in parsed:
                continue
            found.append(cell_json)
        if found:
            return found
    return []


def _detect_prior_kind(prior_output_dir: Path) -> str:
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    has_ckpt = _resolve_verified_prior_checkpoint(prior_dir) is not None
    has_cells = bool(_find_verified_prior_cells(prior_dir))
    if has_ckpt and has_cells:
        raise ValueError(
            f"Ambiguous prior_output_dir {prior_dir} contains both verified checkpoints and "
            "ruler cells; pass explicit --prior-kind checkpoint or --prior-kind ruler_cells."
        )
    if has_ckpt:
        return "checkpoint"
    if has_cells:
        return "ruler_cells"
    raise FileNotFoundError(
        f"No verified V3 PPO checkpoints or ruler cells/*.json found under {prior_dir}"
    )


def _stage_prior_checkpoint(
    prior_output_dir: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
    *,
    light_resume_margin_minutes: float = 45.0,
) -> dict[str, Any]:
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    chosen = _resolve_verified_prior_checkpoint(
        prior_dir,
        light_resume_margin_minutes=light_resume_margin_minutes,
    )
    if chosen is None:
        raise FileNotFoundError(f"No verified V3 PPO checkpoint found under {prior_dir}")

    primary_src = Path(chosen["path"])
    dst_primary = pkt_inputs_dir / primary_src.name
    primary_sha = write_create_only_artifact(
        dst_primary, lambda tmp, src=primary_src: shutil.copyfile(src, tmp)
    )
    inputs_list.append(
        {
            "path": str(dst_primary),
            "dest": primary_src.name,
            "sha256": primary_sha,
        }
    )

    companion_entries: list[dict[str, Any]] = []
    for comp_src in chosen.get("files", [])[1:]:
        c_src = Path(comp_src)
        dst_comp = pkt_inputs_dir / c_src.name
        c_sha = write_create_only_artifact(
            dst_comp, lambda tmp, src=c_src: shutil.copyfile(src, tmp)
        )
        c_size = int(dst_comp.stat().st_size)
        inputs_list.append(
            {
                "path": str(dst_comp),
                "dest": c_src.name,
                "sha256": c_sha,
            }
        )
        comp_rec = {
            "relpath": c_src.name,
            "sha256": c_sha,
            "size_bytes": c_size,
        }
        companion_entries.append(comp_rec)
        ledger["verified_units"].append({"unit_id": f"staged:{c_src.stem}", **comp_rec})

    entry: dict[str, Any] = {
        "relpath": primary_src.name,
        "checkpoint_kind": str(chosen["kind"]),
        "sha256": primary_sha,
        "size_bytes": int(dst_primary.stat().st_size),
    }
    if companion_entries:
        entry["companion_files"] = companion_entries

    ledger["resume_checkpoint"] = entry
    ledger["verified_units"].append(
        {
            "unit_id": f"staged:{primary_src.stem}",
            "relpath": primary_src.name,
            "sha256": primary_sha,
            "size_bytes": int(dst_primary.stat().st_size),
        }
    )
    return entry


def _stage_prior_ruler_cells(
    prior_output_dir: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
) -> list[str]:
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    found_cells = _find_verified_prior_cells(prior_dir)
    if not found_cells:
        raise FileNotFoundError(f"No verified V3 PPO ruler cells/*.json found under {prior_dir}")
    staged_ids: list[str] = []
    for cell_json in found_cells:
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
    return staged_ids


def _extract_staged_checkpoint_config(
    pkt_inputs_dir: Path,
    resume_entry: dict[str, Any],
) -> V3PPOConfig | None:
    staged_primary = pkt_inputs_dir / str(resume_entry["relpath"])
    if staged_primary.name.endswith(".index.json"):
        idx_obj = json.loads(staged_primary.read_text(encoding="utf-8"))
        light_name = str(idx_obj["light_checkpoint"]["filename"])
        light_payload = torch.load(
            pkt_inputs_dir / light_name, map_location="cpu", weights_only=False
        )
        if isinstance(light_payload, dict) and isinstance(light_payload.get("config"), dict):
            return V3PPOConfig.from_dict(light_payload["config"])
        return None
    ckpt_payload = torch.load(staged_primary, map_location="cpu", weights_only=False)
    if isinstance(ckpt_payload, dict) and isinstance(ckpt_payload.get("config"), dict):
        return V3PPOConfig.from_dict(ckpt_payload["config"])
    return None


def _extract_canary_eval_metrics(canary_receipt: Path | str) -> dict[str, Any] | None:
    """Read measured default-eval wall seconds and combined peak bytes from a canary receipt."""
    rcpt_path = Path(canary_receipt).expanduser().resolve()
    if not rcpt_path.is_file():
        return None
    rcpt_obj = json.loads(rcpt_path.read_text(encoding="utf-8"))
    if not isinstance(rcpt_obj, dict):
        return None

    expected_export_shas: dict[str, str] = {}
    exports_list = rcpt_obj.get("exports")
    if isinstance(exports_list, list):
        for row in exports_list:
            if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("sha256"):
                expected_export_shas[row["path"]] = str(row["sha256"])

    checks_obj: dict[str, Any] | None = None
    for cand in (
        rcpt_path.parent / "exports" / "canary_checks.json",
        rcpt_path.parent / "canary_checks.json",
    ):
        if cand.is_file():
            exp_sha = expected_export_shas.get("canary_checks.json")
            if exp_sha is not None and sha256_file(cand) != exp_sha:
                raise ValueError(f"Canary export hash mismatch for {cand}: expected {exp_sha}")
            loaded = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                checks_obj = loaded
                break

    bench = (
        checks_obj.get("default_eval_benchmark")
        if isinstance(checks_obj, dict)
        and isinstance(checks_obj.get("default_eval_benchmark"), dict)
        else {}
    )
    raw_wall = (
        checks_obj.get("default_eval_wall_seconds")
        if isinstance(checks_obj, dict) and checks_obj.get("default_eval_wall_seconds") is not None
        else (
            bench.get("wall_seconds")
            if bench.get("wall_seconds") is not None
            else rcpt_obj.get("default_eval_wall_seconds")
        )
    )
    if (
        isinstance(raw_wall, bool)
        or not isinstance(raw_wall, (int, float))
        or not math.isfinite(float(raw_wall))
        or float(raw_wall) <= 0.0
    ):
        return None

    raw_peak = (
        checks_obj.get("default_eval_peak_memory_bytes")
        if isinstance(checks_obj, dict)
        and checks_obj.get("default_eval_peak_memory_bytes") is not None
        else (
            bench.get("peak_memory_bytes")
            if bench.get("peak_memory_bytes") is not None
            else rcpt_obj.get("default_eval_peak_memory_bytes")
        )
    )
    raw_res_envs = (
        checks_obj.get("default_eval_resident_train_envs")
        if isinstance(checks_obj, dict)
        and checks_obj.get("default_eval_resident_train_envs") is not None
        else bench.get("resident_train_envs")
    )
    out_metrics: dict[str, Any] = {"wall_seconds": float(raw_wall)}
    if not isinstance(raw_peak, bool) and isinstance(raw_peak, (int, float)) and int(raw_peak) > 0:
        out_metrics["peak_memory_bytes"] = int(raw_peak)
    if (
        not isinstance(raw_res_envs, bool)
        and isinstance(raw_res_envs, (int, float))
        and int(raw_res_envs) > 0
    ):
        out_metrics["resident_train_envs"] = int(raw_res_envs)
    return out_metrics


def prepare_v3_ppo_packet(
    packet_dir: Path | str,
    *,
    commit: str = "HEAD",
    mode: str = "canary",
    gpu: str = "L4",
    hours: float = 2.0,
    expected_steps_per_sec: float | None = None,
    net_size: NetSize | None = None,
    num_envs: int | None = None,
    num_snakes: int | None = None,
    arena_radius: float | None = None,
    learner_slots: int | None = None,
    opponent_policy: OpponentPolicy | None = None,
    rollout_len: int | None = None,
    ruler_envs: str | None = None,
    ruler_nets: str | None = None,
    ruler_learners: str | None = None,
    ruler_rollout_len: int = DEFAULT_RULER_ROLLOUT_LEN,
    ruler_warmup_updates: int = DEFAULT_RULER_WARMUP_UPDATES,
    ruler_timed_updates: int = DEFAULT_RULER_TIMED_UPDATES,
    ruler_base_minutes: float | None = None,
    amp: bool | None = None,
    use_channels_last: bool | None = None,
    fast_learner_ops: bool | None = None,
    include_fp32: bool = False,
    compile_model: bool = False,
    max_periodic_full_exports: int | None = None,
    light_resume_margin_minutes: float | None = None,
    eval_estimated_seconds: float | None = None,
    output_dir: Path | str | None = None,
    prior_output_dir: Path | str | None = None,
    prior_kind: str = "auto",
    canary_receipt: Path | str | None = None,
    allow_missing_canary_receipt: bool = False,
    ppo_config: V3PPOConfig | None = None,
    explicit_total_updates: bool = False,
    explicit_eval_estimated_seconds: bool = False,
    measured_packed_bytes_per_world: int | None = None,
    measured_checkpoint_bytes: int | None = None,
    attempt: str = "attempt-01",
    allow_unmerged_commit: bool = False,
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Game V3.1 PPO (`canary`, `ruler`, or `train`).

    Refuses configurations whose estimated VRAM exceeds the target GPU memory budget (`22 GB` L4,
    `40 GB` A100, `80 GB` H100) OR whose single-world/per-file/total-session checkpoint exports
    exceed calibrated shard (`256 MiB` default), supervisor (`2 GiB`/file), or session export caps
    (`12 GiB` L4, `24 GiB` A100, `40 GiB` H100).
    """
    if mode not in PREPARE_MODES:
        raise ValueError(f"mode must be one of {PREPARE_MODES}, got {mode!r}")
    gpu_norm = str(gpu).strip().upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"gpu must be one of {SUPPORTED_GPUS} (T4 unsupported), got {gpu!r}")
    if not re_safe_attempt(attempt):
        raise ValueError(f"Invalid attempt slug {attempt!r}")
    if net_size is not None:
        get_net_preset(net_size)
    if ruler_nets is not None:
        _parse_nets_csv(ruler_nets)
    if mode == "canary" and canary_receipt is not None:
        raise ValueError("canary mode produces a canary receipt and cannot bind --canary-receipt")
    if mode in ("ruler", "train") and canary_receipt is None and not allow_missing_canary_receipt:
        raise ValueError(
            f"--mode {mode} requires canary_receipt <path/to/receipt.json> "
            "(pass allow_missing_canary_receipt=True only for offline schema inspection)"
        )

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
            / f"v3-ppo-{mode}-{gpu_slug}-{attempt}"
            / "output"
        )
    out_path = Path(output_dir).resolve()
    if out_path.exists():
        raise FileExistsError(f"output_dir must be a fresh non-existent directory: {out_path}")

    pkt = Path(packet_dir).resolve()
    pkt.mkdir(parents=True, exist_ok=False)

    try:
        pkt_inputs_dir = pkt / "inputs"
        pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
        ledger: dict[str, Any] = {
            "schema": "v3-ppo-verified-ledger/v1",
            "namespace": V3_PPO_NAMESPACE,
            "unit_kind": "checkpoint",
            "skip_verified": True,
            "verified_units": [],
        }
        inputs_list: list[dict[str, str]] = []
        if prior_output_dir is not None:
            if prior_kind not in ("auto", "checkpoint", "ruler_cells"):
                raise ValueError(f"Unsupported prior_kind={prior_kind!r}")
            if mode == "train" and prior_kind == "ruler_cells":
                raise ValueError("mode='train' requires prior_kind in {'auto', 'checkpoint'}")
            if mode == "ruler" and prior_kind == "checkpoint":
                raise ValueError("mode='ruler' requires prior_kind in {'auto', 'ruler_cells'}")

            eff_prior_kind = prior_kind
            if eff_prior_kind == "auto":
                if mode == "train":
                    eff_prior_kind = "checkpoint"
                elif mode == "ruler":
                    eff_prior_kind = "ruler_cells"
                else:
                    eff_prior_kind = _detect_prior_kind(Path(prior_output_dir))
            if eff_prior_kind == "ruler_cells":
                _stage_prior_ruler_cells(
                    Path(prior_output_dir), pkt_inputs_dir, inputs_list, ledger
                )
            elif eff_prior_kind == "checkpoint":
                eff_stage_margin = (
                    float(light_resume_margin_minutes)
                    if light_resume_margin_minutes is not None
                    else (
                        float(ppo_config.light_resume_margin_minutes)
                        if ppo_config is not None
                        else 45.0
                    )
                )
                _stage_prior_checkpoint(
                    Path(prior_output_dir),
                    pkt_inputs_dir,
                    inputs_list,
                    ledger,
                    light_resume_margin_minutes=eff_stage_margin,
                )

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
        training_plan: dict[str, Any] | None = None
        memory_estimate: dict[str, Any] | None = None
        checkpoint_export_caps: dict[str, Any] | None = None

        canary_eval_metrics: dict[str, Any] | None = None

        if mode == "canary":
            est_minutes = CANARY_BASE_MINUTES
            timeout_minutes = CANARY_TIMEOUT_MINUTES
            base_sec = est_minutes * 60.0
            eff_sec = timeout_minutes * 60.0
            extra_env["V3_PPO_MAX_WALL_SECONDS"] = str(int(CANARY_WALL_GUARD_SECONDS))
            extra_env["V3_PPO_CANARY_TRAIN_ENVS"] = str(default_train_envs_for_gpu(gpu_norm))
            extra_env["SNAKE_COLAB_CANARY"] = "1"
            extra_env["SNAKE_COLAB_CANARY_PACKAGE"] = FAMILY_PACKAGE
            if compile_model:
                extra_env["V3_PPO_COMPILE"] = "1"
            if fast_learner_ops is False:
                extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"
            job_id = f"v3-ppo-canary-{gpu_slug}-{attempt}"
            required_exports = list(CANARY_REQUIRED_EXPORTS)
            max_export_bytes = 320 * 1024 * 1024
            resume_fields: dict[str, Any] = {
                "resume_exemption": {
                    "kind": "short-canary",
                    "reason": (
                        f"Game V3.1 PPO {gpu_norm} family canary probe (15 min base / 30 min "
                        "timeout): tiny S/M ruler + 2-update FULL-resume + 1 default eval"
                    ),
                }
            }
            canary_block = {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": FAMILY_SCRIPT,
            }
        elif mode == "ruler":
            if ruler_base_minutes is not None:
                if not math.isfinite(ruler_base_minutes) or not (
                    3.0 <= float(ruler_base_minutes) <= 30.0
                ):
                    raise ValueError(
                        f"--ruler-base-minutes must be in [3.0, 30.0], got {ruler_base_minutes}"
                    )
                est_minutes = round(float(ruler_base_minutes), 2)
                timeout_minutes = round(2.0 * est_minutes, 2)
                headroom_min = min(8.0, max(2.0, 0.2 * timeout_minutes))
                wall_guard_sec = int(max(60.0, (timeout_minutes - headroom_min) * 60.0))
            else:
                est_minutes = RULER_BASE_MINUTES
                timeout_minutes = RULER_TIMEOUT_MINUTES
                wall_guard_sec = int(RULER_WALL_GUARD_SECONDS)

            parsed_envs = (
                list(DEFAULT_RULER_ENVS) if ruler_envs is None else _parse_int_csv(ruler_envs)
            )
            parsed_nets = (
                list(DEFAULT_RULER_NETS) if ruler_nets is None else _parse_nets_csv(ruler_nets)
            )
            parsed_learners = (
                list(DEFAULT_RULER_LEARNERS)
                if ruler_learners is None
                else _parse_int_csv(ruler_learners)
            )
            eff_r_snakes = 24 if num_snakes is None else int(num_snakes)
            eff_r_radius = 600.0 if arena_radius is None else float(arena_radius)
            eff_r_rollout = int(ruler_rollout_len)
            eff_r_warmup = int(ruler_warmup_updates)
            eff_r_timed = int(ruler_timed_updates)
            if eff_r_rollout < 1:
                raise ValueError(f"--ruler-rollout-len must be >= 1, got {eff_r_rollout}")
            if eff_r_warmup < 0:
                raise ValueError(f"--ruler-warmup-updates must be >= 0, got {eff_r_warmup}")
            if eff_r_timed < 1:
                raise ValueError(f"--ruler-timed-updates must be >= 1, got {eff_r_timed}")

            eff_r_amp = True if amp is None else bool(amp)
            eff_r_cl = True if use_channels_last is None else bool(use_channels_last)
            eff_r_fast = True if fast_learner_ops is None else bool(fast_learner_ops)

            if any(k < 1 or k > eff_r_snakes for k in parsed_learners):
                raise ValueError(
                    f"ruler learners {parsed_learners} must be within [1, {eff_r_snakes}]"
                )

            ruler_game_cfg = GameV3Config(
                arena_radius=eff_r_radius,
                num_snakes=eff_r_snakes,
                max_life_ticks=0,
            )
            budget_bytes = GPU_VRAM_BUDGET_BYTES[gpu_norm]
            fitting_cells: list[str] = []
            smallest_est_bytes: int | None = None
            for e_val in parsed_envs:
                for n_sz in parsed_nets:
                    for k_val in parsed_learners:
                        cell_cfg = V3PPOConfig(
                            game_config=ruler_game_cfg,
                            num_envs=int(e_val),
                            learner_slots=int(k_val),
                            net_size=n_sz,
                            amp=eff_r_amp,
                            use_channels_last=eff_r_cl,
                            fast_learner_ops=eff_r_fast,
                            rollout_len=eff_r_rollout,
                            eval_every_minutes=0.0,
                            eval_every_updates=0,
                        )
                        est = estimate_training_memory(cell_cfg)
                        tot_b = int(est["estimated_peak_bytes"])
                        if smallest_est_bytes is None or tot_b < smallest_est_bytes:
                            smallest_est_bytes = tot_b
                        if tot_b <= budget_bytes:
                            fitting_cells.append(f"E{e_val}_{n_sz}_L{k_val}")

            if not fitting_cells:
                raise ValueError(
                    f"Refusing ruler packet on {gpu_norm}: zero requested ruler cells fit within "
                    f"{budget_bytes / 1e9:.2f} GB budget (smallest cell estimated at "
                    f"{(smallest_est_bytes or 0) / 1e9:.2f} GB)."
                )
            memory_estimate = {
                "gpu": gpu_norm,
                "budget_bytes": budget_bytes,
                "fitting_cells": fitting_cells,
                "total_requested_cells": len(parsed_envs) * len(parsed_nets) * len(parsed_learners),
            }

            base_sec = est_minutes * 60.0
            eff_sec = timeout_minutes * 60.0
            extra_env["V3_PPO_MAX_WALL_SECONDS"] = str(wall_guard_sec)
            if ruler_envs is not None:
                extra_env["V3_PPO_RULER_ENVS"] = ",".join(str(v) for v in parsed_envs)
            if ruler_nets is not None:
                extra_env["V3_PPO_RULER_NETS"] = ",".join(parsed_nets)
            if ruler_learners is not None:
                extra_env["V3_PPO_RULER_LEARNERS"] = ",".join(str(v) for v in parsed_learners)
            if arena_radius is not None:
                extra_env["V3_PPO_ARENA_RADIUS"] = str(eff_r_radius)
            if num_snakes is not None:
                extra_env["V3_PPO_NUM_SNAKES"] = str(eff_r_snakes)
            if eff_r_rollout != DEFAULT_RULER_ROLLOUT_LEN:
                extra_env["V3_PPO_RULER_ROLLOUT_LEN"] = str(eff_r_rollout)
            if eff_r_warmup != DEFAULT_RULER_WARMUP_UPDATES:
                extra_env["V3_PPO_RULER_WARMUP_UPDATES"] = str(eff_r_warmup)
            if eff_r_timed != DEFAULT_RULER_TIMED_UPDATES:
                extra_env["V3_PPO_RULER_TIMED_UPDATES"] = str(eff_r_timed)
            extra_env["V3_PPO_AMP"] = "1" if eff_r_amp else "0"
            extra_env["V3_PPO_CHANNELS_LAST"] = "1" if eff_r_cl else "0"
            if include_fp32:
                extra_env["V3_PPO_INCLUDE_FP32"] = "1"
            if compile_model:
                extra_env["V3_PPO_COMPILE"] = "1"
            if not eff_r_fast:
                extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"

            job_id = f"v3-ppo-ruler-{gpu_slug}-{attempt}"
            required_exports = list(RULER_REQUIRED_EXPORTS)
            max_export_bytes = 128 * 1024 * 1024
            resume_fields = {
                "resume_exemption": {
                    "kind": "acceptance",
                    "reason": (
                        f"Game V3.1 PPO {gpu_norm} throughput, stage-split & VRAM ruler "
                        "(<=60m runaway guard; per-cell create-only JSONs + signal-safe result)"
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
            max_wall_sec_int = int(max(30.0, eff_sec - headroom_sec))
            extra_env["V3_PPO_MAX_WALL_SECONDS"] = str(max_wall_sec_int)

            staged_base_cfg: V3PPOConfig | None = None
            if "resume_checkpoint" in ledger:
                staged_base_cfg = _extract_staged_checkpoint_config(
                    pkt_inputs_dir, ledger["resume_checkpoint"]
                )

            base_cfg = ppo_config or staged_base_cfg or V3PPOConfig()
            cfg_dict = base_cfg.to_dict()
            gc_dict = dict(cfg_dict["game_config"])
            if num_snakes is not None:
                gc_dict["num_snakes"] = int(num_snakes)
            if arena_radius is not None:
                gc_dict["arena_radius"] = float(arena_radius)
            cfg_dict["game_config"] = gc_dict

            if net_size is not None:
                if net_size != base_cfg.net_size:
                    cfg_dict["minibatch_size"] = None
                    cfg_dict["actor_chunk_learners"] = None
                cfg_dict["net_size"] = net_size
            if num_envs is not None:
                cfg_dict["num_envs"] = int(num_envs)
            if learner_slots is not None:
                cfg_dict["learner_slots"] = int(learner_slots)
            if opponent_policy is not None:
                cfg_dict["opponent_policy"] = opponent_policy
            if rollout_len is not None:
                cfg_dict["rollout_len"] = int(rollout_len)
            if amp is not None:
                cfg_dict["amp"] = bool(amp)
            if use_channels_last is not None:
                cfg_dict["use_channels_last"] = bool(use_channels_last)
            if fast_learner_ops is not None:
                cfg_dict["fast_learner_ops"] = bool(fast_learner_ops)
            if compile_model:
                cfg_dict["compile_model"] = True
            if max_periodic_full_exports is not None:
                cfg_dict["max_periodic_full_exports"] = int(max_periodic_full_exports)
            if light_resume_margin_minutes is not None:
                cfg_dict["light_resume_margin_minutes"] = float(light_resume_margin_minutes)

            if canary_receipt is not None:
                canary_block = _build_canary_receipt_binding(canary_receipt, gpu_norm)
                canary_eval_metrics = _extract_canary_eval_metrics(canary_receipt)

            default_eval_est = float(V3PPOConfig().eval_estimated_seconds)
            caller_overrode_eval_est = (
                eval_estimated_seconds is not None
                or bool(explicit_eval_estimated_seconds)
                or (
                    ppo_config is not None
                    and float(ppo_config.eval_estimated_seconds) != default_eval_est
                )
            )
            if eval_estimated_seconds is not None:
                cfg_dict["eval_estimated_seconds"] = float(eval_estimated_seconds)
            elif not caller_overrode_eval_est and canary_eval_metrics is not None:
                cfg_dict["eval_estimated_seconds"] = float(canary_eval_metrics["wall_seconds"])

            cfg_obj = V3PPOConfig.from_dict(cfg_dict)

            if staged_base_cfg is not None:
                assert_resume_config_compatible(
                    staged_base_cfg,
                    cfg_obj,
                    context="Continuation checkpoint config mismatch",
                    reference_label="ckpt",
                    candidate_label="requested",
                )

            if expected_steps_per_sec is not None:
                if (
                    not math.isfinite(expected_steps_per_sec)
                    or float(expected_steps_per_sec) <= 0.0
                ):
                    raise ValueError(
                        "expected_steps_per_sec must be finite and > 0, "
                        f"got {expected_steps_per_sec}"
                    )
                steps_per_update = (
                    int(cfg_obj.rollout_len) * int(cfg_obj.num_envs) * int(cfg_obj.num_snakes)
                )
                expected_sec_per_update = float(steps_per_update) / float(expected_steps_per_sec)
                if expected_sec_per_update > 0.5 * eff_sec:
                    raise ValueError(
                        f"Refusing train packet whose single update "
                        f"({expected_sec_per_update:.1f} s at {expected_steps_per_sec:.1f} "
                        f"steps/s) exceeds half the {hours:g} h session ({0.5 * eff_sec:.1f} s)."
                    )
                training_plan = build_training_plan(
                    cfg_obj,
                    hours=float(hours),
                    expected_steps_per_sec=float(expected_steps_per_sec),
                    max_wall_seconds=float(max_wall_sec_int),
                )
                print(
                    f"V3_PPO_PLAN {json.dumps(training_plan, sort_keys=True)}",
                    file=sys.stderr,
                    flush=True,
                )

            # Enforce GPU VRAM budget AND calibrated single-file + session checkpoint export caps!
            memory_estimate = validate_memory_budget(cfg_obj, gpu_norm)

            eff_world_bytes = measured_packed_bytes_per_world
            if eff_world_bytes is None and measured_checkpoint_bytes is not None:
                if int(measured_checkpoint_bytes) < 1:
                    raise ValueError(
                        f"measured_checkpoint_bytes must be >= 1, got {measured_checkpoint_bytes}"
                    )
                eff_world_bytes = max(
                    1, int(math.ceil(int(measured_checkpoint_bytes) / cfg_obj.num_envs))
                )

            if ppo_config is None or not explicit_total_updates:
                cfg_obj = replace(
                    cfg_obj,
                    total_updates=max(int(cfg_obj.total_updates), UNLIMITED_HOURS_TOTAL_UPDATES),
                )

            expected_updates_for_cap: int | None = None
            if training_plan is not None:
                expected_updates_for_cap = int(training_plan["expected_updates_in_session"])
                if explicit_total_updates and int(cfg_obj.total_updates) > 0:
                    expected_updates_for_cap = min(
                        expected_updates_for_cap, int(cfg_obj.total_updates)
                    )
            elif explicit_total_updates and int(cfg_obj.total_updates) > 0:
                expected_updates_for_cap = int(cfg_obj.total_updates)
            elif (
                int(cfg_obj.checkpoint_every_updates) > 0
                or int(cfg_obj.full_checkpoint_every_updates) > 0
            ):
                steps_per_upd = (
                    int(cfg_obj.rollout_len) * int(cfg_obj.num_envs) * int(cfg_obj.num_snakes)
                )
                fast_sec_per_upd = max(1.0, float(steps_per_upd) / 50_000.0)
                expected_updates_for_cap = max(
                    1, int(math.ceil(float(max_wall_sec_int) / fast_sec_per_upd))
                )

            checkpoint_export_caps = validate_checkpoint_export_caps(
                cfg_obj,
                gpu_norm,
                max_wall_seconds=eff_sec,
                expected_updates=expected_updates_for_cap,
                measured_packed_bytes_per_world=eff_world_bytes,
            )

            extra_env["V3_PPO_AMP"] = "1" if bool(cfg_obj.amp) else "0"
            extra_env["V3_PPO_CHANNELS_LAST"] = "1" if bool(cfg_obj.use_channels_last) else "0"
            if bool(cfg_obj.compile_model):
                extra_env["V3_PPO_COMPILE"] = "1"
            if not bool(cfg_obj.fast_learner_ops):
                extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"
            extra_env["V3_PPO_MAX_SESSION_EXPORT_BYTES"] = str(
                int(checkpoint_export_caps["session_total_export_cap_bytes"])
            )
            extra_env["V3_PPO_CONFIG_JSON"] = json.dumps(cfg_obj.to_dict(), sort_keys=True)
            job_id = f"v3-ppo-train-{gpu_slug}-{attempt}"
            required_exports = list(TRAIN_REQUIRED_EXPORTS)
            max_export_bytes = int(checkpoint_export_caps["per_file_max_export_bytes"])
            resume_fields = {
                "resume": {
                    "unit_kind": "checkpoint",
                    "ledger_path": "inputs/verified-ledger.json",
                    "skip_verified": True,
                    "max_outcome_blind_retries": 2,
                }
            }

        # Ensure any staged input file also satisfies supervisor per-file `max_export_bytes` check
        largest_input_bytes = max(
            (int(Path(inp["path"]).stat().st_size) for inp in inputs_list),
            default=0,
        )
        if largest_input_bytes > MAX_SUPERVISOR_EXPORT_BYTES:
            raise ValueError(
                f"Staged input file ({largest_input_bytes / 2**30:.2f} GiB) exceeds supervisor "
                f"2.00 GiB per-file limit."
            )
        if largest_input_bytes > max_export_bytes:
            max_export_bytes = min(
                MAX_SUPERVISOR_EXPORT_BYTES, int(math.ceil(largest_input_bytes * 1.10))
            )

        unit_cap = math.ceil(((timeout_minutes * rate_uph) / 60.0 - 1e-12) * 10000.0) / 10000.0
        tech_envelope: dict[str, Any] = {
            "base_seconds": base_sec,
            "effective_seconds": eff_sec,
            "forecast_unit_cap": unit_cap,
            "mode": mode,
            "gpu": gpu_norm,
        }
        if checkpoint_export_caps is not None:
            tech_envelope["session_total_export_cap_bytes"] = int(
                checkpoint_export_caps["session_total_export_cap_bytes"]
            )
            tech_envelope["worst_case_session_export_bytes"] = int(
                checkpoint_export_caps["worst_case_session_export_bytes"]
            )
        if canary_eval_metrics is not None:
            tech_envelope["canary_eval_wall_seconds"] = float(canary_eval_metrics["wall_seconds"])
            if "peak_memory_bytes" in canary_eval_metrics:
                tech_envelope["canary_eval_peak_memory_bytes"] = int(
                    canary_eval_metrics["peak_memory_bytes"]
                )

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
                "V3_PPO_SOURCE_COMMIT": resolved_commit,
                "V3_PPO_MODE": mode,
                "SNAKE_DQN_DEVICE": "cuda",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
            "technical_envelope": tech_envelope,
        }
        if allow_unmerged_commit:
            manifest["unmerged_dev_packet"] = True
        if canary_block is not None:
            manifest["canary"] = canary_block

        if mode == "canary" or canary_receipt is not None:
            colab_canary.check(manifest, ROOT)

        manifest_path = pkt / "dispatch-manifest.json"
        _write_json_create_only(manifest_path, manifest)
    except BaseException:
        shutil.rmtree(pkt, ignore_errors=True)
        raise

    summary: dict[str, Any] = {
        "packet_dir": str(pkt),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mode": mode,
        "job_id": manifest["job_id"],
        "commit": resolved_commit,
        "gpu": gpu_norm,
        "timeout_minutes": timeout_minutes,
        "est_minutes": est_minutes,
        "max_wall_seconds": int(extra_env["V3_PPO_MAX_WALL_SECONDS"]),
        "max_export_bytes": int(max_export_bytes),
        "unit_cap": unit_cap,
    }
    if memory_estimate is not None:
        summary["memory_estimate"] = memory_estimate
    if checkpoint_export_caps is not None:
        summary["checkpoint_export_caps"] = checkpoint_export_caps
    if training_plan is not None:
        summary["training_plan"] = training_plan
    if canary_eval_metrics is not None:
        summary["canary_eval_metrics"] = canary_eval_metrics
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.prepare``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        "--packet-dir",
        dest="packet",
        type=Path,
        required=True,
        help="Fresh target packet directory.",
    )
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=PREPARE_MODES, default="canary")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="L4")
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Whole-session timeout in hours for --mode train (default: 2.0).",
    )
    parser.add_argument("--expected-steps-per-sec", type=float, default=None)
    parser.add_argument("--net-size", choices=VALID_NET_SIZES, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-snakes", type=int, default=None)
    parser.add_argument("--arena-radius", type=float, default=None)
    parser.add_argument("--learner-slots", type=int, default=None)
    parser.add_argument("--opponent-policy", choices=VALID_OPPONENT_POLICIES, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)

    parser.add_argument("--ruler-envs", type=str, default=None)
    parser.add_argument("--ruler-nets", type=str, default=None)
    parser.add_argument("--ruler-learners", type=str, default=None)
    parser.add_argument("--ruler-rollout-len", type=int, default=DEFAULT_RULER_ROLLOUT_LEN)
    parser.add_argument("--ruler-warmup-updates", type=int, default=DEFAULT_RULER_WARMUP_UPDATES)
    parser.add_argument("--ruler-timed-updates", type=int, default=DEFAULT_RULER_TIMED_UPDATES)
    parser.add_argument("--ruler-base-minutes", type=float, default=None)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", "--fp32", dest="amp", action="store_false")

    cl_group = parser.add_mutually_exclusive_group()
    cl_group.add_argument(
        "--channels-last", dest="use_channels_last", action="store_true", default=None
    )
    cl_group.add_argument("--no-channels-last", dest="use_channels_last", action="store_false")

    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--fast-learner-ops", dest="fast_learner_ops", action="store_true", default=None
    )
    fast_group.add_argument("--no-fast-learner-ops", dest="fast_learner_ops", action="store_false")

    parser.add_argument("--include-fp32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-periodic-full-exports", type=int, default=None)
    parser.add_argument("--light-resume-margin-minutes", type=float, default=None)
    parser.add_argument("--eval-estimated-seconds", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior-output-dir", type=Path, default=None)
    parser.add_argument(
        "--prior-kind", choices=("auto", "checkpoint", "ruler_cells"), default="auto"
    )
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--allow-missing-canary-receipt", action="store_true")
    parser.add_argument("--ppo-config-json", type=str, default=None)
    parser.add_argument("--measured-packed-bytes-per-world", type=int, default=None)
    parser.add_argument("--measured-checkpoint-bytes", type=int, default=None)
    parser.add_argument("--attempt", type=str, default="attempt-01")
    parser.add_argument("--allow-unmerged-commit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m research.v3_ppo.prepare``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode in ("ruler", "train") and args.canary_receipt is None:
        if not args.allow_missing_canary_receipt:
            parser.error(
                f"--mode {args.mode} requires --canary-receipt <path/to/receipt.json> "
                "(pass --allow-missing-canary-receipt only for offline schema inspection)"
            )

    ppo_cfg: V3PPOConfig | None = None
    explicit_total_updates = False
    explicit_eval_estimated_seconds = False
    if args.ppo_config_json is not None:
        raw_source = args.ppo_config_json
        if raw_source.lstrip().startswith("{"):
            raw_dict = json.loads(raw_source)
        else:
            raw_dict = json.loads(Path(raw_source).read_text(encoding="utf-8"))
        if not isinstance(raw_dict, dict):
            raise ValueError("--ppo-config-json root must be a JSON object")
        explicit_total_updates = "total_updates" in raw_dict
        explicit_eval_estimated_seconds = "eval_estimated_seconds" in raw_dict
        ppo_cfg = V3PPOConfig.from_dict(raw_dict)

    info = prepare_v3_ppo_packet(
        args.packet,
        commit=args.commit,
        mode=args.mode,
        gpu=args.gpu,
        hours=args.hours,
        expected_steps_per_sec=args.expected_steps_per_sec,
        net_size=args.net_size,
        num_envs=args.num_envs,
        num_snakes=args.num_snakes,
        arena_radius=args.arena_radius,
        learner_slots=args.learner_slots,
        opponent_policy=args.opponent_policy,
        rollout_len=args.rollout_len,
        ruler_envs=args.ruler_envs,
        ruler_nets=args.ruler_nets,
        ruler_learners=args.ruler_learners,
        ruler_rollout_len=args.ruler_rollout_len,
        ruler_warmup_updates=args.ruler_warmup_updates,
        ruler_timed_updates=args.ruler_timed_updates,
        ruler_base_minutes=args.ruler_base_minutes,
        amp=args.amp,
        use_channels_last=args.use_channels_last,
        fast_learner_ops=args.fast_learner_ops,
        include_fp32=bool(args.include_fp32),
        compile_model=bool(args.compile),
        max_periodic_full_exports=args.max_periodic_full_exports,
        light_resume_margin_minutes=args.light_resume_margin_minutes,
        eval_estimated_seconds=args.eval_estimated_seconds,
        output_dir=args.output_dir,
        prior_output_dir=args.prior_output_dir,
        prior_kind=args.prior_kind,
        canary_receipt=args.canary_receipt,
        allow_missing_canary_receipt=bool(args.allow_missing_canary_receipt),
        ppo_config=ppo_cfg,
        explicit_total_updates=explicit_total_updates,
        explicit_eval_estimated_seconds=explicit_eval_estimated_seconds,
        measured_packed_bytes_per_world=args.measured_packed_bytes_per_world,
        measured_checkpoint_bytes=args.measured_checkpoint_bytes,
        attempt=args.attempt,
        allow_unmerged_commit=bool(args.allow_unmerged_commit),
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
