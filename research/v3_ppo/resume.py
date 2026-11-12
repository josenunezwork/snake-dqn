"""Resume checkpoint selection, staged continuation validation, and manifest writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.v3_ppo.checkpoint_io import (
    INIT_ANCHOR_LIGHT_NAME,
    V3_PPO_FULL_INDEX_SCHEMA,
    V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
    verify_full_checkpoint_set,
    verify_light_checkpoint,
)
from research.v3_ppo.config import V3_PPO_NAMESPACE, V3PPOConfig
from research.v3_ppo.exports import artifact_or_sidecar_exists, write_create_only_json
from research.v3_ppo.trainer import V3PPOTrainer


def collect_valid_checkpoints_in_dir(
    checkpoints_dir: Path,
    *,
    require_markers: bool = True,
    include_init_anchor: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(newest_full, newest_light, init_anchor_light)`` verified in ``checkpoints_dir``."""
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        return None, None, None

    valid_full: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx_cand in sorted(ckpt_dir.glob("full_*.index.json")):
        if require_markers and not (
            Path(f"{idx_cand}.complete").is_file() and Path(f"{idx_cand}.sha256").is_file()
        ):
            print(
                f"V3_PPO_SKIP_INCOMPLETE_FULL index={idx_cand.name} reason=missing_index_markers",
                flush=True,
            )
            continue
        try:
            meta = verify_full_checkpoint_set(idx_cand, require_markers=require_markers)
            valid_full.append(
                (
                    int(meta["updates_completed"]),
                    int(meta.get("restart_generation", 0)),
                    idx_cand.name,
                    meta,
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            print(
                f"V3_PPO_SKIP_INCOMPLETE_FULL index={idx_cand.name} "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    best_full: dict[str, Any] | None = None
    if valid_full:
        valid_full.sort(key=lambda t: (t[0], t[1], t[2]))
        best_full = valid_full[-1][3]

    valid_light: list[tuple[int, int, str, dict[str, Any]]] = []
    init_anchor_meta: dict[str, Any] | None = None
    for light_cand in sorted(ckpt_dir.glob("light_*.pt")):
        if not include_init_anchor and light_cand.name == INIT_ANCHOR_LIGHT_NAME:
            continue
        if require_markers and not (
            Path(f"{light_cand}.complete").is_file() and Path(f"{light_cand}.sha256").is_file()
        ):
            continue
        try:
            meta_l = verify_light_checkpoint(light_cand, require_markers=require_markers)
            if light_cand.name == INIT_ANCHOR_LIGHT_NAME:
                init_anchor_meta = meta_l
            else:
                valid_light.append(
                    (
                        int(meta_l["updates_completed"]),
                        int(meta_l.get("restart_generation", 0)),
                        light_cand.name,
                        meta_l,
                    )
                )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            print(
                f"V3_PPO_SKIP_INVALID_LIGHT file={light_cand.name} "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    best_light: dict[str, Any] | None = None
    if valid_light:
        valid_light.sort(key=lambda t: (t[0], t[1], t[2]))
        best_light = valid_light[-1][3]

    return best_full, best_light, init_anchor_meta


_collect_valid_checkpoints_in_dir = collect_valid_checkpoints_in_dir


def choose_between_full_and_light(
    best_full: dict[str, Any] | None,
    best_light: dict[str, Any] | None,
    *,
    light_resume_margin_minutes: float = 45.0,
    log_choice: bool = False,
    context: str = "resolve",
) -> dict[str, Any] | None:
    """Prefer newest FULL unless newest LIGHT has more updates AND is newer by ``> margin``."""
    margin_sec = max(0.0, float(light_resume_margin_minutes) * 60.0)
    chosen: dict[str, Any] | None = None
    reason = "none"
    delta_sec = 0.0

    if best_full is not None and best_light is not None:
        full_u = int(best_full["updates_completed"])
        light_u = int(best_light["updates_completed"])
        full_ts = float(best_full.get("created_unix_ts", 0.0))
        light_ts = float(best_light.get("created_unix_ts", 0.0))
        delta_sec = light_ts - full_ts
        if light_u > full_u and delta_sec > margin_sec:
            chosen = best_light
            reason = "light_newer_beyond_margin"
        elif full_u >= light_u:
            chosen = best_full
            reason = "full_at_or_ahead_of_light"
        else:
            chosen = best_full
            reason = "full_preferred_within_margin"
    elif best_full is not None:
        chosen = best_full
        reason = "only_full_available"
    elif best_light is not None:
        chosen = best_light
        reason = "only_light_available"

    if log_choice and chosen is not None:
        f_u_str = str(best_full["updates_completed"]) if best_full is not None else "none"
        l_u_str = str(best_light["updates_completed"]) if best_light is not None else "none"
        print(
            f"V3_PPO_RESUME_CHOICE context={context} chosen_kind={chosen['kind']} "
            f"chosen_file={Path(chosen['path']).name} "
            f"chosen_updates={chosen['updates_completed']} "
            f"full_updates={f_u_str} light_updates={l_u_str} "
            f"light_minus_full_sec={delta_sec:.1f} margin_sec={margin_sec:.1f} "
            f"reason={reason}",
            flush=True,
        )
    return chosen


_choose_between_full_and_light = choose_between_full_and_light


def resolve_resume_checkpoint(
    checkpoints_dir: Path,
    *,
    require_markers: bool = True,
    include_init_anchor: bool = False,
    light_resume_margin_minutes: float = 45.0,
    log_choice: bool = False,
) -> dict[str, Any] | None:
    """Resolve resume checkpoint: newest FULL unless newest LIGHT is newer by ``> margin``."""
    best_full, best_light, init_anchor = collect_valid_checkpoints_in_dir(
        checkpoints_dir,
        require_markers=require_markers,
        include_init_anchor=include_init_anchor,
    )
    chosen = choose_between_full_and_light(
        best_full,
        best_light,
        light_resume_margin_minutes=light_resume_margin_minutes,
        log_choice=log_choice,
        context="resolve_dir",
    )
    if chosen is not None:
        return chosen
    if include_init_anchor and init_anchor is not None:
        return init_anchor
    return None


def verify_staged_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    verify_markers: bool = True,
) -> dict[str, Any]:
    """Validate FULL `.index.json` or LIGHT `.pt` by restoring a trainer."""
    ckpt_path = Path(path)
    if ckpt_path.name.endswith(".index.json"):
        full_meta = verify_full_checkpoint_set(ckpt_path, require_markers=verify_markers)
        cfg = V3PPOConfig.from_dict(full_meta["config"])
        trainer = V3PPOTrainer(cfg, device=device)
        trainer.load_checkpoint(ckpt_path, verify_markers=verify_markers, restore_rng=False)
        return {
            "schema": V3_PPO_FULL_INDEX_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "checkpoint_kind": "full",
            "num_shards": int(full_meta["num_shards"]),
            "sha256": str(full_meta["sha256"]),
            "updates_completed": int(trainer.updates_completed),
            "agent_steps_collected": int(trainer.agent_steps_collected),
        }

    if verify_markers:
        verify_completed_artifact(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") != V3_PPO_LIGHT_CHECKPOINT_SCHEMA:
        raise ValueError(f"Invalid v3_ppo checkpoint at {ckpt_path}")
    cfg = V3PPOConfig.from_dict(raw["config"])
    trainer = V3PPOTrainer(cfg, device=device)
    load_info = trainer.load_checkpoint(ckpt_path, verify_markers=verify_markers, restore_rng=False)
    return {
        "schema": str(raw["schema"]),
        "namespace": str(raw["namespace"]),
        "checkpoint_kind": str(load_info["checkpoint_kind"]),
        "num_shards": int(load_info["num_shards"]),
        "sha256": sha256_file(ckpt_path),
        "updates_completed": int(trainer.updates_completed),
        "agent_steps_collected": int(trainer.agent_steps_collected),
    }


def write_latest_checkpoint_manifest(
    out_checkpoints_dir: Path,
    *,
    reason: str,
    counters: dict[str, Any],
    light_resume_margin_minutes: float = 45.0,
) -> Path:
    """Write create-only ``checkpoints/latest_manifest.json`` summarizing exported checkpoints."""
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_checkpoints_dir / "latest_manifest.json"
    if artifact_or_sidecar_exists(manifest_path):
        verify_completed_artifact(manifest_path)
        return manifest_path

    best = resolve_resume_checkpoint(
        out_checkpoints_dir,
        require_markers=True,
        light_resume_margin_minutes=light_resume_margin_minutes,
    )
    all_files = sorted(
        p.name
        for p in out_checkpoints_dir.iterdir()
        if p.is_file()
        and not p.name.endswith((".sha256", ".complete", ".meta.json"))
        and p.name != "latest_manifest.json"
    )
    payload: dict[str, Any] = {
        "schema": "v3-ppo-latest-checkpoint/v2",
        "namespace": V3_PPO_NAMESPACE,
        "reason": reason,
        "primary_checkpoint": best["path"].name if best is not None else "",
        "primary_kind": best["kind"] if best is not None else "none",
        "sha256": best["sha256"] if best is not None else "",
        "size_bytes": int(best["size_bytes"]) if best is not None else 0,
        "published_files": all_files,
        **counters,
    }
    write_create_only_json(manifest_path, payload)
    return manifest_path


def validate_staged_ledger_resume(
    *,
    input_dir: Path | None,
    ledger_path: Path | None,
) -> dict[str, Any] | None:
    """Validate optional staged continuation checkpoint (FULL set or LIGHT) in ``input_dir``."""
    if ledger_path is None or not Path(ledger_path).is_file():
        return None
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("namespace") not in (None, V3_PPO_NAMESPACE):
        raise ValueError(
            f"Foreign ledger namespace {ledger.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
        )
    resume_entry = ledger.get("resume_checkpoint")
    if not isinstance(resume_entry, dict):
        return None
    if input_dir is None:
        raise ValueError("ledger specifies resume_checkpoint, but input_dir is None")

    in_dir = Path(input_dir)
    primary_rel = str(resume_entry["relpath"])
    staged_primary = in_dir / primary_rel
    if not staged_primary.is_file():
        raise FileNotFoundError(f"Missing staged checkpoint at {staged_primary}")
    actual_sha = sha256_file(staged_primary)
    if actual_sha != resume_entry.get("sha256"):
        raise ValueError(
            f"Staged checkpoint SHA-256 mismatch: {actual_sha} != {resume_entry.get('sha256')}"
        )

    for comp in resume_entry.get("companion_files") or []:
        comp_path = in_dir / str(comp["relpath"])
        if not comp_path.is_file():
            raise FileNotFoundError(f"Missing staged FULL companion file: {comp_path}")
        c_sha = sha256_file(comp_path)
        if c_sha != comp.get("sha256"):
            raise ValueError(
                f"Staged companion SHA-256 mismatch on {comp_path.name}: "
                f"{c_sha} != {comp.get('sha256')}"
            )

    if staged_primary.name.endswith(".index.json"):
        meta = verify_full_checkpoint_set(staged_primary, require_markers=False)
    else:
        meta = verify_light_checkpoint(staged_primary, require_markers=False)
    meta["source"] = "staged_ledger"
    meta["require_markers"] = False
    meta["relpath"] = primary_rel
    return meta


_validate_staged_ledger_resume = validate_staged_ledger_resume


def select_best_resume_candidate(
    *,
    work_ckpts: Path,
    out_ckpts: Path,
    staged_meta: dict[str, Any] | None,
    include_init_anchor: bool = False,
    light_resume_margin_minutes: float = 45.0,
    log_choice: bool = True,
    context: str = "resume",
) -> dict[str, Any] | None:
    """Pick best verified checkpoint across ``work_ckpts``, ``out_ckpts``, and ``staged_meta``.

    Compares the globally newest valid FULL checkpoint against the globally newest valid LIGHT
    checkpoint: prefers FULL unless LIGHT has strictly more updates AND is newer by more than
    ``light_resume_margin_minutes`` (`45.0` min default), and logs the decision.
    """
    full_cands: list[dict[str, Any]] = []
    light_cands: list[dict[str, Any]] = []
    init_cands: list[dict[str, Any]] = []

    for src_name, d_path, inc_init in (
        ("work_checkpoints", work_ckpts, include_init_anchor),
        ("output_checkpoints", out_ckpts, False),
    ):
        bf, bl, bi = collect_valid_checkpoints_in_dir(
            d_path,
            require_markers=True,
            include_init_anchor=inc_init,
        )
        for item, target_list in ((bf, full_cands), (bl, light_cands), (bi, init_cands)):
            if item is not None:
                cand = dict(item)
                cand["source"] = src_name
                cand["require_markers"] = True
                cand["relpath"] = Path(item["path"]).name
                target_list.append(cand)

    if staged_meta is not None:
        if staged_meta["kind"] == "full":
            full_cands.append(staged_meta)
        elif str(staged_meta.get("relpath")) == INIT_ANCHOR_LIGHT_NAME:
            init_cands.append(staged_meta)
        else:
            light_cands.append(staged_meta)

    best_full: dict[str, Any] | None = None
    if full_cands:
        full_cands.sort(
            key=lambda c: (
                int(c["updates_completed"]),
                int(c.get("restart_generation", 0)),
                str(c["relpath"]),
            )
        )
        best_full = full_cands[-1]

    best_light: dict[str, Any] | None = None
    if light_cands:
        light_cands.sort(
            key=lambda c: (
                int(c["updates_completed"]),
                int(c.get("restart_generation", 0)),
                str(c["relpath"]),
            )
        )
        best_light = light_cands[-1]

    chosen = choose_between_full_and_light(
        best_full,
        best_light,
        light_resume_margin_minutes=light_resume_margin_minutes,
        log_choice=log_choice,
        context=context,
    )
    if chosen is not None:
        return chosen

    if include_init_anchor and init_cands:
        init_cands.sort(key=lambda c: int(c["updates_completed"]))
        return init_cands[-1]
    return None


_select_best_resume_candidate = select_best_resume_candidate
