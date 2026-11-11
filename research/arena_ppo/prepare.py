"""Prepare Stage 1 Arena PPO Colab CLI packets (`canary`, `ruler`, `train`, and `eval` modes)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from research.arena_ppo.evaluate import (
    DEFAULT_EVAL_BURNIN,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TICKS,
    DEFAULT_EVAL_WORLDS,
    load_arena_ppo_policy,
    resolve_b8_checkpoint_path,
)
from research.arena_ppo.net import VALID_NET_ARCHS, NetArch, NetSize
from research.arena_ppo.ppo import (
    _RESUME_MUTABLE_CONFIG_KEYS,
    ARENA_PPO_NAMESPACE,
    ARENA_PPO_SCHEMA,
    UNLIMITED_HOURS_TOTAL_UPDATES,
    ArenaPPOConfig,
    build_training_plan,
    compute_max_update_seconds,
    estimate_max_export_bytes,
    find_newest_verified_checkpoint,
)
from research.arena_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    EVAL_REQUIRED_EXPORTS,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
    _parse_archs_csv,
    _parse_envs_csv,
    _parse_nets_csv,
)
from research.arena_ppo.runtime import (
    sha256_file,
    verify_completed_artifact,
    write_create_only_artifact,
)
from research.colab_cli import canary as colab_canary
from src.model.ego2s_network import load_ego2s_checkpoint

ROOT = Path(__file__).resolve().parents[2]
FAMILY_SCRIPT = "research/arena_ppo/job.sh"
FAMILY_PACKAGE = "research/arena_ppo"
REQUIREMENTS_RELPATH = "research/arena_ppo/requirements.txt"

PREPARE_MODES: tuple[str, ...] = ("canary", "ruler", "train", "eval")
SUPPORTED_GPUS: tuple[str, ...] = ("L4", "A100", "H100")
GPU_FORECAST_RATES_UPH: dict[str, float] = {
    "L4": 1.54,
    "A100": 8.0,
    "H100": 12.0,
}

# Instrumented family canary envelope: 15 min base / 30 min 2x timeout (30 min transport cap),
# 720s (12 min) Python wall guard (~2.26x over conservative ~318s 6-cell + 3-arch L4 canary).
CANARY_BASE_MINUTES = 15.0
CANARY_TIMEOUT_MINUTES = 30.0
CANARY_WALL_GUARD_SECONDS = 720.0

# Uninstrumented science ruler envelope: 30 min base / 60 min runaway guard; Python guard = T - 8m
RULER_BASE_MINUTES = 30.0
RULER_TIMEOUT_MINUTES = 60.0
RULER_WALL_GUARD_SECONDS = (RULER_TIMEOUT_MINUTES - 8.0) * 60.0  # 3120s (52 min)

# Fixed-opponent evaluation envelope: 15 min base / 30 min 2x timeout (<= 60m acceptance cap)
EVAL_BASE_MINUTES = 15.0
EVAL_TIMEOUT_MINUTES = 30.0
EVAL_WALL_GUARD_SECONDS = (EVAL_TIMEOUT_MINUTES - 6.0) * 60.0  # 1440s (24 min)


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
        "ego_device.py",
        "v8_device.py",
        *(
            f"{stem}.{suffix}"
            for stem in ("move_cuda", "event_cuda", "flood_cuda", "v8_indexed")
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
        "evaluate.py",
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
    req_files = ["__init__.py", "job.sh", "requirements.txt", "ruler.py", "ppo.py"]
    missing_core = sorted(
        f"research/arena_ppo/{req}"
        for req in req_files
        if f"research/arena_ppo/{req}" not in selected
    )
    if missing_core:
        raise ValueError(
            f"Pinned commit {commit} is missing required research/arena_ppo files: {missing_core}"
        )
    return selected


def _find_verified_prior_checkpoint(prior_dir: Path) -> Path | None:
    """Locate newest verified non-probe Arena PPO checkpoint under `prior_dir` (or `/exports`)."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

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
                return final_cand

        newest = find_newest_verified_checkpoint(ckpt_sub)
        if newest is not None:
            return newest
    return None


def _find_verified_prior_cells(prior_dir: Path) -> list[Path]:
    """Locate verified `cells/*.json` artifacts under `prior_dir` (or `/exports`)."""
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
            parsed_cell = json.loads(cell_json.read_text(encoding="utf-8"))
            if isinstance(parsed_cell, dict) and "error" in parsed_cell:
                continue
            found.append(cell_json)
        if found:
            return found
    return []


def _detect_prior_kind(prior_output_dir: Path) -> str:
    """Detect whether `prior_output_dir` holds checkpoints or ruler cells for `--mode canary`."""
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    has_ckpt = _find_verified_prior_checkpoint(prior_dir) is not None
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
        f"No verified Arena PPO checkpoints or ruler cells/*.json found under {prior_dir}"
    )


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

    chosen_ckpt = _find_verified_prior_checkpoint(prior_dir)
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

    found_cells = _find_verified_prior_cells(prior_dir)
    if not found_cells:
        raise FileNotFoundError(f"No verified ruler cells/*.json found under {prior_dir}")

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


def _stage_eval_checkpoint_file(
    ckpt_file: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Validate and stage an explicit Arena PPO learner checkpoint for `canary` / `eval`."""
    src_ckpt = Path(ckpt_file).resolve()
    if not src_ckpt.is_file():
        raise FileNotFoundError(f"eval_checkpoint does not exist: {src_ckpt}")
    loaded = load_arena_ppo_policy(src_ckpt, "cpu")
    dest_name = "eval_learner_checkpoint.pt"
    dst_ckpt = pkt_inputs_dir / dest_name
    ckpt_sha = write_create_only_artifact(
        dst_ckpt, lambda tmp, src=src_ckpt: shutil.copyfile(src, tmp)
    )
    inputs_list.append(
        {
            "path": str(dst_ckpt),
            "dest": dest_name,
            "sha256": ckpt_sha,
        }
    )
    entry = {
        "relpath": dest_name,
        "sha256": ckpt_sha,
        "size_bytes": int(dst_ckpt.stat().st_size),
    }
    if loaded.metadata.get("schema") == ARENA_PPO_SCHEMA:
        ledger["resume_checkpoint"] = entry
    else:
        ledger["eval_learner_checkpoint"] = entry
    ledger["verified_units"].append({"unit_id": "staged:eval_learner", **entry})
    return entry


def _stage_eval_opponent_file(
    opponent_spec: str,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
    *,
    explicit_opponent_checkpoint: Path | str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Validate and stage opponent weights (`b8` or `ckpt:<path>`) when needed."""
    raw = str(opponent_spec).strip()
    if raw in ("random_safe", "greedy_food", "untrained"):
        if explicit_opponent_checkpoint is not None:
            raise ValueError(f"eval_opponent_checkpoint is not used with opponent={raw!r}")
        return raw, None

    if raw in ("b8", "b8-s3") or raw.startswith("b8:"):
        b8_arg: str | Path | None = explicit_opponent_checkpoint
        if raw.startswith("b8:"):
            b8_arg = raw[len("b8:") :].strip()
        src_b8, _ = resolve_b8_checkpoint_path(b8_arg)
        _ = load_ego2s_checkpoint(str(src_b8), device="cpu")
        dest_name = "eval_opponent_checkpoint.pt"
        dst_b8 = pkt_inputs_dir / dest_name
        b8_sha = write_create_only_artifact(
            dst_b8, lambda tmp, src=src_b8: shutil.copyfile(src, tmp)
        )
        inputs_list.append({"path": str(dst_b8), "dest": dest_name, "sha256": b8_sha})
        entry = {
            "kind": "b8",
            "relpath": dest_name,
            "sha256": b8_sha,
            "size_bytes": int(dst_b8.stat().st_size),
        }
        ledger["eval_opponent_checkpoint"] = entry
        ledger["verified_units"].append({"unit_id": "staged:eval_opponent", **entry})
        return "b8", entry

    if raw == "ckpt" or raw.startswith("ckpt:"):
        ckpt_arg: str | Path | None = explicit_opponent_checkpoint
        if raw.startswith("ckpt:"):
            ckpt_arg = raw[len("ckpt:") :].strip()
        if ckpt_arg is None:
            raise ValueError("opponent='ckpt' requires ckpt:<PATH> or --eval-opponent-checkpoint")
        src_opp = Path(ckpt_arg).expanduser().resolve()
        _ = load_arena_ppo_policy(src_opp, "cpu")
        dest_name = "eval_opponent_checkpoint.pt"
        dst_opp = pkt_inputs_dir / dest_name
        opp_sha = write_create_only_artifact(
            dst_opp, lambda tmp, src=src_opp: shutil.copyfile(src, tmp)
        )
        inputs_list.append({"path": str(dst_opp), "dest": dest_name, "sha256": opp_sha})
        entry = {
            "kind": "ckpt",
            "relpath": dest_name,
            "sha256": opp_sha,
            "size_bytes": int(dst_opp.stat().st_size),
        }
        ledger["eval_opponent_checkpoint"] = entry
        ledger["verified_units"].append({"unit_id": "staged:eval_opponent", **entry})
        return "ckpt", entry

    raise ValueError(
        f"Unsupported eval_opponent {opponent_spec!r}; expected random_safe, greedy_food, "
        "untrained, b8, b8-s3, b8:<PATH>, or ckpt:<PATH>"
    )


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
    expected_steps_per_sec: float | None = None,
    net_arch: str | None = None,
    net_size: NetSize | None = None,
    num_envs: int | None = None,
    ruler_envs: str | None = None,
    ruler_nets: str | None = None,
    no_one_epoch_cell: bool = False,
    no_best_compile: bool = False,
    no_larger_minibatch: bool = False,
    ruler_base_minutes: float | None = None,
    amp: bool | None = None,
    include_fp32: bool = False,
    output_dir: Path | str | None = None,
    prior_output_dir: Path | str | None = None,
    prior_kind: str = "auto",
    eval_checkpoint: Path | str | None = None,
    eval_opponent: str | None = None,
    eval_opponent_checkpoint: Path | str | None = None,
    eval_learner_seats: int = 1,
    eval_worlds: int = DEFAULT_EVAL_WORLDS,
    eval_ticks: int = DEFAULT_EVAL_TICKS,
    eval_burnin: int = DEFAULT_EVAL_BURNIN,
    eval_seed: int = DEFAULT_EVAL_SEED,
    eval_greedy: bool = False,
    eval_base_minutes: float | None = None,
    canary_receipt: Path | str | None = None,
    ppo_config: ArenaPPOConfig | None = None,
    explicit_total_updates: bool = False,
    explicit_max_update_seconds: bool = False,
    measured_checkpoint_bytes: int | None = None,
    compile_model: bool = False,
    attempt: str = "attempt-01",
    allow_unmerged_commit: bool = False,
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Stage 1 Arena PPO (`canary`, `ruler`, `train`, or `eval`).

    Args:
        packet_dir: Fresh target packet directory on Mac.
        commit: Git revision to pin (must be an ancestor of ``origin/main`` unless
            ``allow_unmerged_commit=True`` is explicitly requested for local pre-merge checks).
        mode: ``"canary"`` (15 min base / 30 min timeout instrumented family probe), ``"ruler"``
            (<= 60 min uninstrumented science ruler bound to canary receipt), ``"train"``
            (resumable training session bound to canary receipt), or ``"eval"``
            (<= 60 min fixed-opponent evaluation bound to canary receipt).
        gpu: Target GPU in ``{"L4", "A100", "H100"}`` (default ``"L4"``).
        hours: Session wall timeout in hours for ``mode="train"`` (`timeout = hours * 60` min).
        expected_steps_per_sec: Expected agent-steps/s for ``train`` plan calculation.
        net_arch: Architecture override (`shared_trunk`, `ego_only`, or `per_seat` for ``train``;
            comma-separated list for ``ruler``).
        net_size: Optional network size override (`"S"` or `"M"`) for ``train``.
        num_envs: Optional world count override for ``train``.
        ruler_envs: Optional comma-separated world counts for ``ruler`` (e.g. ``"1024,2048,4096"``).
        ruler_nets: Optional comma-separated sizes for ``ruler`` (e.g. ``"S,M"``).
        no_one_epoch_cell: Disable the 1-PPO-epoch probe cell in ``ruler`` mode.
        output_dir: Optional fresh absolute Mac output directory.
        prior_output_dir: Optional prior attempt output directory for resume/eval staging.
        prior_kind: ``"auto"``, ``"checkpoint"``, or ``"ruler_cells"`` when staging prior output.
        eval_checkpoint: Optional explicit learner checkpoint path to stage for ``canary``/``eval``.
        eval_opponent: Opponent spec for ``eval`` (or ``canary`` when staging opponent weights).
        eval_opponent_checkpoint: Optional explicit opponent checkpoint path (`b8` or `ckpt`).
        eval_learner_seats: Learner seats `K` per world for ``eval``.
        eval_worlds: Parallel world count `W` for ``eval``.
        eval_ticks: Post-burn-in ticks `T` for ``eval``.
        eval_burnin: Burn-in ticks `B` for ``eval``.
        eval_seed: Deterministic seed `S` for ``eval``.
        eval_greedy: Greedy argmax learner actions for ``eval``.
        eval_base_minutes: Optional base duration in `[3.0, 30.0]` min for ``eval`` (`2x <= 60m`).
        canary_receipt: Family canary `receipt.json` path to bind for ``ruler``/``train``/``eval``.
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
    if eval_checkpoint is not None and prior_output_dir is not None:
        raise ValueError("Pass at most one of --eval-checkpoint or --prior-output-dir")

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
    staged_learner_relpath: str | None = None
    if eval_checkpoint is not None:
        if mode not in ("canary", "eval"):
            raise ValueError("--eval-checkpoint is supported only for --mode canary or --mode eval")
        staged_entry = _stage_eval_checkpoint_file(
            Path(eval_checkpoint), pkt_inputs_dir, inputs_list, ledger
        )
        staged_learner_relpath = str(staged_entry["relpath"])
    elif prior_output_dir is not None:
        if prior_kind not in ("auto", "checkpoint", "ruler_cells"):
            raise ValueError(f"Unsupported prior_kind={prior_kind!r}")
        if mode in ("train", "eval") and prior_kind == "ruler_cells":
            raise ValueError(f"mode={mode!r} requires prior_kind in {{'auto', 'checkpoint'}}")
        if mode == "ruler" and prior_kind == "checkpoint":
            raise ValueError("mode='ruler' requires prior_kind in {'auto', 'ruler_cells'}")

        eff_prior_kind = prior_kind
        if eff_prior_kind == "auto":
            if mode in ("train", "eval"):
                eff_prior_kind = "checkpoint"
            elif mode == "ruler":
                eff_prior_kind = "ruler_cells"
            else:
                eff_prior_kind = _detect_prior_kind(Path(prior_output_dir))
        if eff_prior_kind == "ruler_cells":
            _stage_prior_ruler_cells(Path(prior_output_dir), pkt_inputs_dir, inputs_list, ledger)
        elif eff_prior_kind == "checkpoint":
            staged_entry = _stage_prior_checkpoint(
                Path(prior_output_dir), pkt_inputs_dir, inputs_list, ledger
            )
            staged_learner_relpath = str(staged_entry["relpath"])
        else:
            raise ValueError(f"Unsupported prior_kind={prior_kind!r}")

    staged_opp_kind: str | None = None
    staged_opp_entry: dict[str, Any] | None = None
    if mode == "eval":
        eff_eval_opp = eval_opponent or "greedy_food"
        staged_opp_kind, staged_opp_entry = _stage_eval_opponent_file(
            eff_eval_opp,
            pkt_inputs_dir,
            inputs_list,
            ledger,
            explicit_opponent_checkpoint=eval_opponent_checkpoint,
        )
    elif mode == "canary" and (eval_opponent is not None or eval_opponent_checkpoint is not None):
        if eval_opponent is None:
            raise ValueError("--eval-opponent-checkpoint requires --eval-opponent")
        staged_opp_kind, staged_opp_entry = _stage_eval_opponent_file(
            eval_opponent,
            pkt_inputs_dir,
            inputs_list,
            ledger,
            explicit_opponent_checkpoint=eval_opponent_checkpoint,
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
                    f"Stage 1 Arena PPO {gpu_norm} family canary probe (15 min base / 30 min "
                    "timeout): real indexed parity + tiny 3-arch ruler + tiny 2-update "
                    "train/checkpoint/resume"
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
            if not math.isfinite(ruler_base_minutes) or float(ruler_base_minutes) < 3.0:
                raise ValueError(
                    f"--ruler-base-minutes must be >= 3.0 and <= 30.0, got {ruler_base_minutes}"
                )
            if float(ruler_base_minutes) > 30.0:
                raise ValueError(
                    f"--ruler-base-minutes must be <= 30.0 (2x <= 60m acceptance cap), "
                    f"got {ruler_base_minutes}"
                )
            est_minutes = round(float(ruler_base_minutes), 2)
            timeout_minutes = round(2.0 * est_minutes, 2)
            headroom_min = min(8.0, max(2.0, 0.2 * timeout_minutes))
            wall_guard_sec = int(max(60.0, (timeout_minutes - headroom_min) * 60.0))
        else:
            est_minutes = RULER_BASE_MINUTES
            timeout_minutes = RULER_TIMEOUT_MINUTES
            wall_guard_sec = int(RULER_WALL_GUARD_SECONDS)
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(wall_guard_sec)
        if net_arch is not None:
            parsed_archs = _parse_archs_csv(net_arch)
            extra_env["ARENA_PPO_RULER_ARCHS"] = ",".join(parsed_archs)
        if ruler_envs is not None:
            parsed_envs = _parse_envs_csv(ruler_envs)
            extra_env["ARENA_PPO_RULER_ENVS"] = ",".join(str(v) for v in parsed_envs)
        if ruler_nets is not None:
            parsed_nets = _parse_nets_csv(ruler_nets)
            extra_env["ARENA_PPO_RULER_NETS"] = ",".join(parsed_nets)
        if no_one_epoch_cell:
            extra_env["ARENA_PPO_NO_ONE_EPOCH_CELL"] = "1"
        if no_best_compile:
            extra_env["ARENA_PPO_NO_BEST_COMPILE"] = "1"
        if no_larger_minibatch:
            extra_env["ARENA_PPO_NO_LARGER_MINIBATCH"] = "1"
        eff_ruler_amp = True if amp is None else bool(amp)
        extra_env["ARENA_PPO_AMP"] = "1" if eff_ruler_amp else "0"
        if include_fp32:
            extra_env["ARENA_PPO_INCLUDE_FP32"] = "1"
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
    elif mode == "eval":
        if int(eval_learner_seats) < 0:
            raise ValueError(f"eval_learner_seats must be >= 0, got {eval_learner_seats}")
        if int(eval_worlds) < 1:
            raise ValueError(f"eval_worlds must be >= 1, got {eval_worlds}")
        if int(eval_ticks) < 1:
            raise ValueError(f"eval_ticks must be >= 1, got {eval_ticks}")
        if int(eval_burnin) < 0:
            raise ValueError(f"eval_burnin must be >= 0, got {eval_burnin}")
        if (int(eval_learner_seats) > 0 or staged_opp_kind == "untrained") and (
            staged_learner_relpath is None
        ):
            raise ValueError(
                "mode='eval' with learner_seats > 0 or opponent='untrained' requires "
                "--eval-checkpoint or --prior-output-dir"
            )
        if eval_base_minutes is not None:
            if not math.isfinite(eval_base_minutes) or float(eval_base_minutes) < 3.0:
                raise ValueError(
                    f"--eval-base-minutes must be >= 3.0 and <= 30.0, got {eval_base_minutes}"
                )
            if float(eval_base_minutes) > 30.0:
                raise ValueError(
                    f"--eval-base-minutes must be <= 30.0 (2x <= 60m acceptance cap), "
                    f"got {eval_base_minutes}"
                )
            est_minutes = round(float(eval_base_minutes), 2)
            timeout_minutes = round(2.0 * est_minutes, 2)
            headroom_min = min(6.0, max(2.0, 0.2 * timeout_minutes))
            wall_guard_sec = int(max(60.0, (timeout_minutes - headroom_min) * 60.0))
        else:
            est_minutes = EVAL_BASE_MINUTES
            timeout_minutes = EVAL_TIMEOUT_MINUTES
            wall_guard_sec = int(EVAL_WALL_GUARD_SECONDS)
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(wall_guard_sec)
        if staged_learner_relpath is not None:
            extra_env["ARENA_PPO_EVAL_CHECKPOINT_REL"] = staged_learner_relpath
        if staged_opp_kind in ("b8", "ckpt") and staged_opp_entry is not None:
            extra_env["ARENA_PPO_EVAL_OPPONENT_KIND"] = str(staged_opp_kind)
            extra_env["ARENA_PPO_EVAL_OPPONENT_REL"] = str(staged_opp_entry["relpath"])
        else:
            extra_env["ARENA_PPO_EVAL_OPPONENT"] = str(staged_opp_kind or "greedy_food")
        extra_env["ARENA_PPO_EVAL_LEARNER_SEATS"] = str(int(eval_learner_seats))
        extra_env["ARENA_PPO_EVAL_WORLDS"] = str(int(eval_worlds))
        extra_env["ARENA_PPO_EVAL_TICKS"] = str(int(eval_ticks))
        extra_env["ARENA_PPO_EVAL_BURNIN"] = str(int(eval_burnin))
        extra_env["ARENA_PPO_EVAL_SEED"] = str(int(eval_seed))
        if eval_greedy:
            extra_env["ARENA_PPO_EVAL_GREEDY"] = "1"
        job_id = f"arena-ppo-eval-{gpu_slug}-{attempt}"
        required_exports = list(EVAL_REQUIRED_EXPORTS)
        max_export_bytes = 64 * 1024 * 1024
        resume_fields = {
            "resume_exemption": {
                "kind": "acceptance",
                "reason": (
                    f"Stage 1 Arena PPO {gpu_norm} fixed-opponent evaluation (<= 60m runaway "
                    "guard; atomic create-only eval_result.json export)"
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
        extra_env["ARENA_PPO_MAX_WALL_SECONDS"] = str(max_wall_sec_int)
        extra_env["ARENA_PPO_HOURS"] = str(float(hours))
        resolved_train_arch: NetArch | None = None
        if net_arch is not None:
            arch_norm = net_arch.strip().lower()
            if arch_norm not in VALID_NET_ARCHS:
                raise ValueError(
                    f"train --net-arch must be one of {VALID_NET_ARCHS}, got {net_arch!r}"
                )
            resolved_train_arch = (
                "shared_trunk"
                if arch_norm == "shared_trunk"
                else ("ego_only" if arch_norm == "ego_only" else "per_seat")
            )
            extra_env["ARENA_PPO_NET_ARCH"] = resolved_train_arch

        staged_base_cfg: ArenaPPOConfig | None = None
        if "resume_checkpoint" in ledger:
            staged_ckpt_path = pkt_inputs_dir / str(ledger["resume_checkpoint"]["relpath"])
            import torch as _torch

            ckpt_payload = _torch.load(staged_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_payload, dict) and isinstance(ckpt_payload.get("config"), dict):
                raw_staged_cfg = dict(ckpt_payload["config"])
                raw_staged_cfg.setdefault("net_arch", "per_seat")
                staged_base_cfg = ArenaPPOConfig.from_dict(raw_staged_cfg)

        if ppo_config is not None:
            cfg_obj = ppo_config
            if (
                resolved_train_arch is not None
                or net_size is not None
                or num_envs is not None
                or amp is not None
            ):
                d = cfg_obj.to_dict()
                if resolved_train_arch is not None and resolved_train_arch != cfg_obj.net_arch:
                    d["net_arch"] = resolved_train_arch
                    d["minibatch_size"] = None
                    d["actor_chunk_envs"] = None
                if net_size is not None and net_size != cfg_obj.net_size:
                    d["net_size"] = net_size
                    d["minibatch_size"] = None
                    d["actor_chunk_envs"] = None
                if num_envs is not None:
                    d["num_envs"] = int(num_envs)
                if amp is not None:
                    d["amp"] = bool(amp)
                    d["use_bf16_autocast"] = bool(amp)
                cfg_obj = ArenaPPOConfig.from_dict(d)
        elif staged_base_cfg is not None:
            d = staged_base_cfg.to_dict()
            if resolved_train_arch is not None and resolved_train_arch != staged_base_cfg.net_arch:
                d["net_arch"] = resolved_train_arch
                d["minibatch_size"] = None
                d["actor_chunk_envs"] = None
            if net_size is not None and net_size != staged_base_cfg.net_size:
                d["net_size"] = net_size
                d["minibatch_size"] = None
                d["actor_chunk_envs"] = None
            if num_envs is not None:
                d["num_envs"] = int(num_envs)
            if compile_model:
                d["compile_model"] = True
            if amp is not None:
                d["amp"] = bool(amp)
                d["use_bf16_autocast"] = bool(amp)
            cfg_obj = ArenaPPOConfig.from_dict(d)
        else:
            cfg_obj = ArenaPPOConfig(
                net_arch=resolved_train_arch if resolved_train_arch is not None else "shared_trunk",
                net_size=net_size if net_size is not None else "S",
                num_envs=int(num_envs) if num_envs is not None else 1024,
                compile_model=compile_model,
                amp=(True if amp is None else bool(amp)),
            )
        extra_env["ARENA_PPO_AMP"] = "1" if bool(cfg_obj.amp) else "0"

        if staged_base_cfg is not None:
            staged_dict = staged_base_cfg.to_dict()
            curr_dict = cfg_obj.to_dict()
            priority_keys = ("net_arch", "net_size", "num_envs")
            ordered_keys = list(priority_keys) + sorted(
                (set(staged_dict) | set(curr_dict)) - set(priority_keys)
            )
            for key in ordered_keys:
                if key in _RESUME_MUTABLE_CONFIG_KEYS:
                    continue
                if staged_dict.get(key) != curr_dict.get(key):
                    raise ValueError(
                        f"Continuation checkpoint config mismatch on {key!r}: "
                        f"ckpt={staged_dict.get(key)!r} vs requested={curr_dict.get(key)!r}"
                    )

        if expected_steps_per_sec is not None:
            extra_env["ARENA_PPO_EXPECTED_STEPS_PER_SEC"] = str(float(expected_steps_per_sec))
            if not math.isfinite(expected_steps_per_sec) or float(expected_steps_per_sec) <= 0.0:
                raise ValueError(
                    f"expected_steps_per_sec must be finite and > 0, got {expected_steps_per_sec}"
                )
            steps_per_update = (
                int(cfg_obj.rollout_len) * int(cfg_obj.num_envs) * int(cfg_obj.num_snakes)
            )
            expected_sec_per_update = float(steps_per_update) / float(expected_steps_per_sec)
            half_session_sec = 0.5 * eff_sec
            if expected_sec_per_update > half_session_sec:
                raise ValueError(
                    f"Refusing train packet whose single update ({expected_sec_per_update:.1f} s "
                    f"at {expected_steps_per_sec:.1f} steps/s) exceeds half the {hours:g} h "
                    f"session ({half_session_sec:.1f} s); reduce --num-envs/rollout_len or "
                    "increase --hours."
                )
            if not explicit_max_update_seconds:
                cfg_obj = replace(
                    cfg_obj,
                    max_update_seconds=compute_max_update_seconds(
                        cfg_obj, float(expected_steps_per_sec)
                    ),
                )
            training_plan = build_training_plan(
                cfg_obj,
                hours=float(hours),
                expected_steps_per_sec=float(expected_steps_per_sec),
                max_wall_seconds=float(max_wall_sec_int),
            )
            print(
                f"ARENA_PPO_PLAN {json.dumps(training_plan, sort_keys=True)}",
                file=sys.stderr,
                flush=True,
            )
        if ppo_config is None or not explicit_total_updates:
            cfg_obj = replace(
                cfg_obj,
                total_updates=max(int(cfg_obj.total_updates), UNLIMITED_HOURS_TOTAL_UPDATES),
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
        "max_export_bytes": int(max_export_bytes),
        "unit_cap": unit_cap,
    }
    if training_plan is not None:
        summary["training_plan"] = training_plan
    return summary


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
    parser.add_argument(
        "--expected-steps-per-sec",
        type=float,
        default=None,
        help="Expected agent-steps/s for --mode train plan calculation.",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default=None,
        help="Architecture (`shared_trunk`, `ego_only`, `per_seat`; CSV allowed for --mode ruler).",
    )
    parser.add_argument("--net-size", choices=("S", "M"), default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--ruler-envs", type=str, default=None)
    parser.add_argument("--ruler-nets", type=str, default=None)
    parser.add_argument("--no-one-epoch-cell", action="store_true")
    parser.add_argument("--no-best-compile", action="store_true")
    parser.add_argument("--no-larger-minibatch", action="store_true")
    parser.add_argument("--ruler-base-minutes", type=float, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help=(
            "Enable CUDA mixed precision (DEFAULT ON: bf16 autocast on Ampere/Ada/Hopper, "
            "fp16+GradScaler else)."
        ),
    )
    amp_group.add_argument(
        "--no-amp",
        "--fp32",
        dest="amp",
        action="store_false",
        help="Disable CUDA mixed precision and run pure float32.",
    )
    parser.add_argument(
        "--include-fp32",
        action="store_true",
        help=(
            "In --mode ruler, measure both default mixed-precision (`E...`, bf16 ON on CUDA) "
            "and pure float32 (`E..._fp32`) cells in one run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior-output-dir", type=Path, default=None)
    parser.add_argument(
        "--prior-kind", choices=("auto", "checkpoint", "ruler_cells"), default="auto"
    )
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-opponent", type=str, default=None)
    parser.add_argument("--eval-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-learner-seats", type=int, default=1)
    parser.add_argument("--eval-worlds", type=int, default=DEFAULT_EVAL_WORLDS)
    parser.add_argument("--eval-ticks", type=int, default=DEFAULT_EVAL_TICKS)
    parser.add_argument("--eval-burnin", type=int, default=DEFAULT_EVAL_BURNIN)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--eval-greedy", action="store_true")
    parser.add_argument("--eval-base-minutes", type=float, default=None)
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--allow-missing-canary-receipt", action="store_true")
    parser.add_argument("--ppo-config-json", type=str, default=None)
    parser.add_argument("--measured-checkpoint-bytes", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--attempt", type=str, default="attempt-01")
    parser.add_argument("--allow-unmerged-commit", action="store_true")
    args = parser.parse_args()

    if args.mode in ("ruler", "train", "eval") and args.canary_receipt is None:
        if not args.allow_missing_canary_receipt:
            parser.error(
                f"--mode {args.mode} requires --canary-receipt <path/to/receipt.json> "
                "(pass --allow-missing-canary-receipt only for offline schema inspection)"
            )

    ppo_cfg: ArenaPPOConfig | None = None
    explicit_total_updates = False
    explicit_max_update_seconds = False
    if args.ppo_config_json is not None:
        raw_source = args.ppo_config_json
        if raw_source.lstrip().startswith("{"):
            raw_dict = json.loads(raw_source)
        else:
            raw_dict = json.loads(Path(raw_source).read_text(encoding="utf-8"))
        if not isinstance(raw_dict, dict):
            raise ValueError("--ppo-config-json root must be a JSON object")
        explicit_total_updates = "total_updates" in raw_dict
        explicit_max_update_seconds = "max_update_seconds" in raw_dict
        ppo_cfg = ArenaPPOConfig.from_dict(raw_dict)

    info = prepare_arena_ppo_packet(
        args.packet,
        commit=args.commit,
        mode=args.mode,
        gpu=args.gpu,
        hours=args.hours,
        expected_steps_per_sec=args.expected_steps_per_sec,
        net_arch=args.net_arch,
        net_size=args.net_size,
        num_envs=args.num_envs,
        ruler_envs=args.ruler_envs,
        ruler_nets=args.ruler_nets,
        no_one_epoch_cell=bool(args.no_one_epoch_cell),
        no_best_compile=bool(args.no_best_compile),
        no_larger_minibatch=bool(args.no_larger_minibatch),
        ruler_base_minutes=args.ruler_base_minutes,
        amp=args.amp,
        include_fp32=bool(args.include_fp32),
        output_dir=args.output_dir,
        prior_output_dir=args.prior_output_dir,
        prior_kind=args.prior_kind,
        eval_checkpoint=args.eval_checkpoint,
        eval_opponent=args.eval_opponent,
        eval_opponent_checkpoint=args.eval_opponent_checkpoint,
        eval_learner_seats=args.eval_learner_seats,
        eval_worlds=args.eval_worlds,
        eval_ticks=args.eval_ticks,
        eval_burnin=args.eval_burnin,
        eval_seed=args.eval_seed,
        eval_greedy=bool(args.eval_greedy),
        eval_base_minutes=args.eval_base_minutes,
        canary_receipt=args.canary_receipt,
        ppo_config=ppo_cfg,
        explicit_total_updates=explicit_total_updates,
        explicit_max_update_seconds=explicit_max_update_seconds,
        measured_checkpoint_bytes=args.measured_checkpoint_bytes,
        compile_model=bool(args.compile),
        attempt=args.attempt,
        allow_unmerged_commit=bool(args.allow_unmerged_commit),
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
