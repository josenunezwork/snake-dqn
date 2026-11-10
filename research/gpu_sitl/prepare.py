"""Prepare Step B1b-Smoke or Step B1b Pilot Colab CLI packets (no dispatch)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from research.colab_cli import canary as colab_canary
from research.compute.gpu_preference import prepare_preference
from research.gpu_sitl.preflight import resolve_product_checkpoints
from research.gpu_sitl.probe_bank import verify_file_sha256, write_atomic_with_sha256
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    PROBE32_NAMESPACE,
    SHAKEOUT_PROBE_NAMESPACE,
    SHAKEOUT_SMOKE_NAMESPACE,
    SHAKEOUT_TRAIN_NAMESPACE,
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    audit_shakeout_seeds,
    shakeout_train_seeds,
    train_seeds,
)
from web.backend.served_checkpoint import B8_S3_SHA256 as B8_SHA
from web.backend.served_checkpoint import FRP3_S12_SHA256 as FRP3_SHA

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_DOC = "docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md"
FAMILY_SCRIPT = "research/gpu_sitl/job.sh"
FAMILY_PACKAGE = "research/gpu_sitl"
L4_RATE_UNITS_PER_HOUR = 1.54

SMOKE_REQUIRED_EXPORTS: tuple[str, ...] = (
    "smoke_result.json",
    "smoke_result.json.sha256",
    "smoke_result.json.complete",
)
SHAKEOUT_REQUIRED_EXPORTS: tuple[str, ...] = (
    "shakeout_result.json",
    "shakeout_result.json.sha256",
    "shakeout_result.json.complete",
)
PILOT_REQUIRED_EXPORTS: tuple[str, ...] = (
    "probe32_bank.pt",
    "probe32_bank.pt.sha256",
    "probe32_bank.pt.complete",
    "probe32_bank.pt.meta.json",
    "probe32_bank.pt.meta.json.sha256",
    "b8_sitl_student_final.pth",
    "b8_sitl_student_final.pth.sha256",
    "b8_sitl_student_final.pth.complete",
    "sitl_pilot_result.json",
    "sitl_pilot_result.json.sha256",
    "sitl_pilot_result.json.complete",
)


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_source_paths(commit: str) -> list[str]:
    """Return the audited source inventory shipped to the Colab CLI runtime."""
    gpu_names = {
        "__init__.py",
        "device_sim.py",
        "stage_timing.py",
        "r2_cache.py",
        "device_policy.py",
        "anchors_device.py",
        "ego_device.py",
        "masks_device.py",
        "v8_device.py",
        "vector_device.py",
        *(
            f"{stem}.{suffix}"
            for stem in ("flood_cuda", "move_cuda", "event_cuda", "vector_geometry", "v8_indexed")
            for suffix in ("py", "cpp", "cu")
        ),
    }
    resident_names = {
        "__init__.py",
        "device_sim.py",
        "ct_mechanics.py",
        "observation_tensor.py",
        "core.py",
    }
    watch_replay_names = {
        "__init__.py",
        "open_stop.py",
        "open_stop_ceiling.py",
        "exact_stop.py",
        "watch6_common.py",
        "watch6_selectivity.py",
    }
    gpu_sitl_names = {
        "__init__.py",
        "open_stop_gpu.py",
        "preflight.py",
        "seed_check.py",
        "probe_bank.py",
        "train_sitl.py",
        "smoke.py",
        "job.sh",
        "requirements.txt",
        "prepare.py",
    }
    tracked_set = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "src", "research"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    )
    for gs_name in gpu_sitl_names:
        rel_gs = f"research/gpu_sitl/{gs_name}"
        if (ROOT / rel_gs).is_file():
            tracked_set.add(rel_gs)
    tracked = sorted(tracked_set)
    return [
        name
        for name in tracked
        if (
            name in {"src/__init__.py", "research/__init__.py"}
            or (
                name.startswith(("src/core/", "src/model/", "src/simd_env/", "src/evaluation/"))
                and name.endswith(".py")
            )
            or (
                Path(name).parent.as_posix() == "research/gpu_sim_spike"
                and Path(name).name in gpu_names
            )
            or (
                Path(name).parent.as_posix() == "research/sandbox_a_resident"
                and Path(name).name in resident_names
            )
            or (
                Path(name).parent.as_posix() == "research/watch_replay"
                and Path(name).name in watch_replay_names
            )
            or (
                Path(name).parent.as_posix() == "research/gpu_sitl"
                and Path(name).name in gpu_sitl_names
            )
        )
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_prior_attempt_artifacts(
    prior_output_dir: Path,
    pkt_inputs_dir: Path,
    inputs_list: list[dict[str, str]],
    ledger: dict[str, Any],
) -> None:
    """Outcome-blind staging of verified probe bank and latest checkpoint from a prior attempt."""
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    pkt_inputs_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    # 1. Carry over verified_units from prior verified-ledger.json if present
    for sroot in search_roots:
        prior_ledger_p = sroot / "verified-ledger.json"
        if prior_ledger_p.is_file() and Path(f"{prior_ledger_p}.complete").is_file():
            verify_file_sha256(prior_ledger_p, require_complete=True)
            prior_ledger = json.loads(prior_ledger_p.read_text(encoding="utf-8"))
            if prior_ledger.get("namespace") != TRAIN_NAMESPACE:
                raise ValueError(
                    f"Foreign namespace in prior ledger {prior_ledger_p}: "
                    f"{prior_ledger.get('namespace')!r}"
                )
            if prior_ledger.get("probe_namespace", PROBE32_NAMESPACE) != PROBE32_NAMESPACE:
                raise ValueError(
                    f"Foreign probe_namespace in prior ledger {prior_ledger_p}: "
                    f"{prior_ledger.get('probe_namespace')!r}"
                )
            if isinstance(prior_ledger.get("world_seeds"), list) and prior_ledger["world_seeds"]:
                ledger["world_seeds"] = [int(s) for s in prior_ledger["world_seeds"]]
            if isinstance(prior_ledger.get("probe_seeds"), list) and prior_ledger["probe_seeds"]:
                ledger["probe_seeds"] = [int(s) for s in prior_ledger["probe_seeds"]]
            for u in prior_ledger.get("verified_units", []):
                if isinstance(u, dict) and not any(
                    existing.get("sha256") == u.get("sha256")
                    for existing in ledger["verified_units"]
                ):
                    ledger["verified_units"].append(dict(u))
            break

    # 2. Stage completed probe32_bank.pt (or partial probe32_bank.pt.ckpt.pt if harvest interrupted)
    staged_probe = False
    for sroot in search_roots:
        pb_cand = sroot / "probe32_bank.pt"
        if pb_cand.is_file() and Path(f"{pb_cand}.complete").is_file():
            pb_sha = verify_file_sha256(pb_cand, require_complete=True)
            dst_pb = pkt_inputs_dir / "probe32_bank.pt"
            write_atomic_with_sha256(
                dst_pb,
                lambda tmp, src=pb_cand: shutil.copyfile(src, tmp),
                write_complete_marker=True,
            )
            inputs_list.append(
                {
                    "path": str(dst_pb),
                    "dest": "probe32_bank.pt",
                    "sha256": pb_sha,
                }
            )
            ledger["probe32_bank"] = {
                "relpath": "probe32_bank.pt",
                "sha256": pb_sha,
                "size_bytes": int(dst_pb.stat().st_size),
            }
            staged_probe = True
            break

    if not staged_probe:
        for sroot in search_roots:
            hckpt_cand = sroot / "probe32_bank.pt.ckpt.pt"
            if hckpt_cand.is_file() and Path(f"{hckpt_cand}.complete").is_file():
                hckpt_sha = verify_file_sha256(hckpt_cand, require_complete=True)
                dst_hckpt = pkt_inputs_dir / "probe32_bank.pt.ckpt.pt"
                write_atomic_with_sha256(
                    dst_hckpt,
                    lambda tmp, src=hckpt_cand: shutil.copyfile(src, tmp),
                    write_complete_marker=True,
                )
                inputs_list.append(
                    {
                        "path": str(dst_hckpt),
                        "dest": "probe32_bank.pt.ckpt.pt",
                        "sha256": hckpt_sha,
                    }
                )
                ledger["probe32_harvest_ckpt"] = {
                    "relpath": "probe32_bank.pt.ckpt.pt",
                    "sha256": hckpt_sha,
                    "size_bytes": int(dst_hckpt.stat().st_size),
                }
                break

    # 3. Stage newest verified training checkpoint (checkpoints/sitl_state_*.pt)
    latest_ckpt_src: Path | None = None
    latest_ckpt_sha: str | None = None
    for sroot in search_roots:
        meta_cand = sroot / "checkpoints" / "latest_checkpoint.json"
        if meta_cand.is_file() and Path(f"{meta_cand}.complete").is_file():
            verify_file_sha256(meta_cand, require_complete=True)
            meta_obj = json.loads(meta_cand.read_text(encoding="utf-8"))
            rel_or_path = str(meta_obj.get("relpath") or meta_obj.get("checkpoint_path", ""))
            cand_pt = sroot / rel_or_path
            if not cand_pt.is_file():
                cand_pt = sroot / "checkpoints" / Path(rel_or_path).name
            if cand_pt.is_file() and Path(f"{cand_pt}.complete").is_file():
                c_sha = verify_file_sha256(cand_pt, require_complete=True)
                if meta_obj.get("sha256") and c_sha != meta_obj["sha256"]:
                    raise ValueError(
                        f"Checkpoint SHA256 mismatch vs latest_checkpoint.json: {cand_pt}"
                    )
                latest_ckpt_src = cand_pt
                latest_ckpt_sha = c_sha
                break

    if latest_ckpt_src is None:
        for sroot in search_roots:
            ckpt_dir = sroot / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            candidates = sorted(
                ckpt_dir.glob("sitl_state_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for cand_pt in candidates:
                if Path(f"{cand_pt}.complete").is_file() and Path(f"{cand_pt}.sha256").is_file():
                    latest_ckpt_src = cand_pt
                    latest_ckpt_sha = verify_file_sha256(cand_pt, require_complete=True)
                    break
            if latest_ckpt_src is not None:
                break

    if latest_ckpt_src is not None and latest_ckpt_sha is not None:
        dst_ckpt = pkt_inputs_dir / latest_ckpt_src.name
        write_atomic_with_sha256(
            dst_ckpt,
            lambda tmp, src=latest_ckpt_src: shutil.copyfile(src, tmp),
            write_complete_marker=True,
        )
        inputs_list.append(
            {
                "path": str(dst_ckpt),
                "dest": latest_ckpt_src.name,
                "sha256": latest_ckpt_sha,
            }
        )
        ckpt_entry = {
            "relpath": latest_ckpt_src.name,
            "sha256": latest_ckpt_sha,
            "size_bytes": int(dst_ckpt.stat().st_size),
        }
        ledger["resume_checkpoint"] = ckpt_entry
        if not any(u.get("sha256") == latest_ckpt_sha for u in ledger["verified_units"]):
            ledger["verified_units"].append(
                {
                    "unit_id": f"staged:{latest_ckpt_src.stem}",
                    **ckpt_entry,
                }
            )


def prepare_b1b_packet(
    packet_dir: Path | str,
    *,
    commit: str = "HEAD",
    mode: str = "smoke",
    output_dir: Path | str | None = None,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    smoke_result_json: Path | str | None = None,
    canary_receipt: Path | str | None = None,
    gpu: str | None = None,
    base_seconds: float | None = None,
    prior_output_dir: Path | str | None = None,
    attempt: str = "attempt-01",
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Step B1b-Smoke, Shakeout, or Pilot (no dispatch)."""
    if mode not in ("smoke", "shakeout", "pilot"):
        raise ValueError(f"mode must be 'smoke', 'shakeout', or 'pilot', got {mode!r}")

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    ancestor_ref = "HEAD" if mode == "shakeout" else "origin/main"
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_commit, ancestor_ref],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if ancestor_check.returncode != 0:
        raise ValueError(
            f"Refusing to prepare packet from unmerged commit {resolved_commit}: "
            f"commit is not an ancestor of {ancestor_ref}"
        )

    if b8_checkpoint is None or opponent_checkpoint is None:
        default_b8, default_opp = resolve_product_checkpoints()
        b8_checkpoint = b8_checkpoint or default_b8
        opponent_checkpoint = opponent_checkpoint or default_opp
    b8_path = Path(b8_checkpoint).resolve()
    opp_path = Path(opponent_checkpoint).resolve()

    if digest(b8_path) != B8_SHA:
        raise ValueError(f"B8 s3 checkpoint SHA256 mismatch at {b8_path}")
    if digest(opp_path) != FRP3_SHA:
        raise ValueError(f"FRP3 s12 checkpoint SHA256 mismatch at {opp_path}")

    reg_path = ROOT / REGISTRATION_DOC
    reg_sha = digest(reg_path)

    if mode == "shakeout":
        seed_audit = audit_shakeout_seeds(smoke_envs=4, probe_worlds=2, train_envs=4)
        if not bool(seed_audit["ok"]):
            raise RuntimeError(f"Step B1b shake-out seed audit failed: {seed_audit}")
    else:
        seed_audit = audit_b1b_seeds(scan_code_roots=[])
        if not bool(seed_audit["ok"]):
            raise RuntimeError(f"Step B1b seed audit failed: {seed_audit}")

    pkt = Path(packet_dir).resolve()
    pkt.mkdir(parents=True, exist_ok=False)

    if output_dir is None:
        output_dir = (
            ROOT.parent
            / "snake-dqn-artifacts"
            / "big-lever"
            / "exp"
            / "argon-20261026"
            / f"gpu-sitl-b1b-{mode}-{attempt}"
            / "output"
        )
    out_path = Path(output_dir).resolve()

    write_json(pkt / "seed-audit.json", seed_audit)

    # Both smoke (family canary) and pilot carry the identical deterministic verified-ledger
    # input destination and count so `canary.check` input-inventory validation succeeds.
    inputs_list: list[dict[str, str]] = [
        {"path": str(b8_path), "dest": "b8-s3.pth", "sha256": B8_SHA},
        {"path": str(opp_path), "dest": "frp3-s12.pth", "sha256": FRP3_SHA},
    ]
    pkt_inputs_dir = pkt / "inputs"
    pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
    if mode == "shakeout":
        ledger: dict[str, Any] = {
            "schema": "t34-gpu-sitl-verified-ledger/v1",
            "namespace": SHAKEOUT_TRAIN_NAMESPACE,
            "probe_namespace": SHAKEOUT_PROBE_NAMESPACE,
            "world_seeds": [int(s) for s in shakeout_train_seeds(4)],
            "unit_kind": "update",
            "skip_verified": True,
            "total_updates": 2,
            "min_unique_rescue_roots": 2,
            "not_evidence": True,
            "evidence_class": "SHAKEOUT_NOT_EVIDENCE",
            "verified_units": [],
        }
    else:
        ledger = {
            "schema": "t34-gpu-sitl-verified-ledger/v1",
            "namespace": TRAIN_NAMESPACE,
            "probe_namespace": PROBE32_NAMESPACE,
            "world_seeds": [int(s) for s in train_seeds(256)],
            "unit_kind": "update",
            "skip_verified": True,
            "total_updates": 1000,
            "min_unique_rescue_roots": 5000,
            "verified_units": [],
        }
    if mode == "pilot" and prior_output_dir is not None:
        _stage_prior_attempt_artifacts(
            Path(prior_output_dir),
            pkt_inputs_dir,
            inputs_list,
            ledger,
        )
    ledger_path = pkt_inputs_dir / "verified-ledger.json"
    ledger_sha = write_atomic_with_sha256(
        ledger_path,
        lambda tmp: tmp.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
        write_complete_marker=True,
    )
    write_json(pkt / "verified-ledger.json", ledger)
    inputs_list.append(
        {
            "path": str(ledger_path),
            "dest": "verified-ledger.json",
            "sha256": ledger_sha,
        }
    )

    extra_env: dict[str, str] = {}
    canary_block: dict[str, Any] | None = None
    pinned_hardware: str | None = str(gpu).strip() if gpu is not None else None

    if mode == "smoke":
        # Registered Step B1b-Smoke: short-canary <= 25 min, cap 3.0 A100 u (11.25 min base /
        # 22.5 min 2x timeout = 3.0 A100 u at 8.0 u/h ceiling rate).
        est_minutes = 11.25
        timeout_minutes = 22.5
        a100_unit_cap = 3.0
        job_id = f"t34-gpu-sitl-b1b-smoke-20261109-{attempt}"
        required_exports = list(SMOKE_REQUIRED_EXPORTS)
        resume_fields: dict[str, Any] = {
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    "Step B1b-Smoke registered throughput and CUDA row-subset parity canary "
                    "(<= 25 min, cap 3.0 A100 u); no scoring look"
                ),
            }
        }
        canary_block = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": FAMILY_SCRIPT,
        }
        extra_env["SNAKE_COLAB_CANARY"] = "1"
        extra_env["SNAKE_COLAB_CANARY_PACKAGE"] = FAMILY_PACKAGE
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
    elif mode == "shakeout":
        # Cheap dev-only L4 CUDA shake-out: short-canary <= 20 min, cap <= 1.0 L4 u at 1.54 u/h
        # (7.5 min base / 15.0 min 2x timeout = 0.385 L4 u).
        if pinned_hardware is not None and pinned_hardware != "L4":
            raise ValueError(
                f"--mode shakeout requires requested_gpu='L4', got --gpu={pinned_hardware!r}"
            )
        pinned_hardware = "L4"
        est_minutes = 7.5
        timeout_minutes = 15.0
        a100_unit_cap = round((timeout_minutes * L4_RATE_UNITS_PER_HOUR) / 60.0 + 1e-9, 4)
        job_id = f"t34-gpu-sitl-b1b-shakeout-20261110-{attempt}"
        required_exports = list(SHAKEOUT_REQUIRED_EXPORTS)
        resume_fields = {
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    "Step B1b dev-only L4 CUDA shake-out canary (<= 20 min, cap <= 1.0 L4 u); "
                    "NOT EVIDENCE, no registered namespace read or scored"
                ),
            }
        }
        canary_block = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": FAMILY_SCRIPT,
        }
        extra_env["SNAKE_COLAB_CANARY"] = "1"
        extra_env["SNAKE_COLAB_CANARY_PACKAGE"] = FAMILY_PACKAGE
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
    else:
        # Step B1b (`B1-A100-Pilot`): requires measured smoke_result_json or explicit base_seconds
        if smoke_result_json is not None:
            smoke_res_path = Path(smoke_result_json).resolve()
            if "shakeout" in smoke_res_path.name.lower():
                raise ValueError(
                    f"Refusing shake-out result {smoke_res_path} for --mode pilot: "
                    "shake-out outputs are NOT EVIDENCE and cannot size or gate the pilot"
                )
            smoke_data = json.loads(smoke_res_path.read_text(encoding="utf-8"))
            if (
                smoke_data.get("schema") != "t34-gpu-sitl-b1b-smoke-result/v1"
                or bool(smoke_data.get("shakeout"))
                or bool(smoke_data.get("not_evidence"))
                or smoke_data.get("evidence_class") == "SHAKEOUT_NOT_EVIDENCE"
                or smoke_data.get("mode") == "shakeout"
                or smoke_data.get("train_namespace") != TRAIN_NAMESPACE
            ):
                raise ValueError(
                    f"Refusing non-smoke or NOT EVIDENCE result {smoke_res_path} for --mode pilot "
                    f"(schema={smoke_data.get('schema')!r}, "
                    f"evidence_class={smoke_data.get('evidence_class')!r}, "
                    f"not_evidence={smoke_data.get('not_evidence')!r})"
                )
            rg = smoke_data["pilot_runaway_guard"]
            base_sec = float(rg["base_estimate_seconds"])
            timeout_minutes = float(rg["runaway_guard_timeout_minutes_2x"])
            est_minutes = float(rg["base_estimate_minutes"])
            a100_unit_cap = float(rg["runaway_guard_a100_units_2x_at_8p00_broker"])
            if "runaway_guard_max_collection_steps_2x" in rg:
                extra_env["GPU_SITL_MAX_COLLECTION_STEPS"] = str(
                    int(rg["runaway_guard_max_collection_steps_2x"])
                )
            if "runaway_guard_timeout_seconds_2x" in rg:
                extra_env["GPU_SITL_MAX_WALL_SECONDS"] = str(
                    float(rg["runaway_guard_timeout_seconds_2x"])
                )
            if canary_receipt is None:
                for cand_rcpt in (
                    smoke_res_path.parent / "receipt.json",
                    smoke_res_path.parent.parent / "receipt.json",
                ):
                    if cand_rcpt.is_file():
                        canary_receipt = cand_rcpt
                        break
        elif base_seconds is not None and math.isfinite(base_seconds) and base_seconds > 0:
            base_sec = float(base_seconds)
            est_minutes = base_sec / 60.0
            timeout_minutes = float(math.ceil(2.0 * est_minutes))
            a100_unit_cap = (timeout_minutes / 60.0) * 8.0
        else:
            raise ValueError(
                "Preparing --mode pilot requires either --smoke-result or positive --base-seconds"
            )
        eff_sec = timeout_minutes * 60.0
        job_id = f"t34-gpu-sitl-b1b-pilot-20261109-{attempt}"
        required_exports = list(PILOT_REQUIRED_EXPORTS)
        resume_fields = {
            "resume": {
                "unit_kind": "update",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        }
        if canary_receipt is not None:
            rcpt_path = Path(canary_receipt).resolve()
            if not rcpt_path.is_file():
                raise FileNotFoundError(f"Canary receipt not found: {rcpt_path}")
            rcpt_data = json.loads(rcpt_path.read_text(encoding="utf-8"))
            if "shakeout" in str(rcpt_data.get("job_id", "")).lower():
                raise ValueError(f"Refusing shake-out canary receipt {rcpt_path} for --mode pilot")
            saved_mf_path = rcpt_path.parent / "manifest.json"
            if saved_mf_path.is_file():
                saved_mf = json.loads(saved_mf_path.read_text(encoding="utf-8"))
                if (saved_mf.get("env") or {}).get("GPU_SITL_MODE") == "shakeout" or bool(
                    saved_mf.get("not_evidence")
                ):
                    raise ValueError(
                        f"Refusing shake-out canary manifest in {rcpt_path.parent} for --mode pilot"
                    )
            rcpt_hw = str((rcpt_data.get("canary_evidence") or {}).get("hardware") or "").strip()
            if not rcpt_hw and (rcpt_path.parent / "hardware.json").is_file():
                hw_data = json.loads(
                    (rcpt_path.parent / "hardware.json").read_text(encoding="utf-8")
                )
                rcpt_hw = str(hw_data.get("kind") or "").strip()
            if not rcpt_hw:
                rcpt_hw = pinned_hardware or "A100"
            if pinned_hardware is not None and pinned_hardware != rcpt_hw:
                raise ValueError(
                    f"--gpu {pinned_hardware!r} does not match canary receipt hardware {rcpt_hw!r}"
                )
            pinned_hardware = rcpt_hw
            canary_block = {
                "receipt": {
                    "path": str(rcpt_path),
                    "sha256": digest(rcpt_path),
                }
            }

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": resolved_commit,
        "script": FAMILY_SCRIPT,
        "device": "cuda",
        "source_paths": runtime_source_paths(resolved_commit),
        "requirements": {
            "gpu": "H100",
            "allowed_gpus": ["H100"],
            "prefer": ["H100", "A100"],
            "min_vcpu": 2,
            "est_minutes": est_minutes,
        },
        "timeout_minutes": timeout_minutes,
        "output_dir": str(out_path),
        "max_export_bytes": 2 * 1024**3,
        "inputs": inputs_list,
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/gpu_sitl/requirements.txt",
            "GPU_SITL_SOURCE_COMMIT": resolved_commit,
            "GPU_SITL_MODE": mode,
            "SNAKE_DQN_DEVICE": "cuda",
            "MAX_JOBS": "4",
            **extra_env,
        },
        "required_exports": required_exports,
        "budget": {
            "rate_units_per_hour": 12.0,
            "unit_cap": a100_unit_cap,
            "reserve_units": 0,
            "other_assignments": {},
        },
        **resume_fields,
        "requirements_file": "research/gpu_sitl/requirements.txt",
        "registration_binding": (
            {
                "doc": REGISTRATION_DOC,
                "sha256": reg_sha,
                "not_evidence": True,
                "evidence_class": "SHAKEOUT_NOT_EVIDENCE",
                "shakeout_smoke_namespace": SHAKEOUT_SMOKE_NAMESPACE,
                "shakeout_probe_namespace": SHAKEOUT_PROBE_NAMESPACE,
                "shakeout_train_namespace": SHAKEOUT_TRAIN_NAMESPACE,
            }
            if mode == "shakeout"
            else {
                "doc": REGISTRATION_DOC,
                "sha256": reg_sha,
                "train_namespace": TRAIN_NAMESPACE,
                "probe32_namespace": PROBE32_NAMESPACE,
                "preflight_namespace": PREFLIGHT_NAMESPACE,
            }
        ),
        "technical_envelope": {
            "base_seconds": base_sec,
            "effective_seconds": eff_sec,
            "registered_a100_unit_cap": a100_unit_cap,
            "mode": mode,
        },
    }
    if mode == "shakeout":
        manifest["not_evidence"] = True
        manifest["evidence_class"] = "SHAKEOUT_NOT_EVIDENCE"
    if canary_block is not None:
        manifest["canary"] = canary_block

    if pinned_hardware is not None:
        # Pin to the explicit hardware (L4 for shakeout, or A100/H100 from smoke canary receipt).
        if pinned_hardware == "L4":
            rate_uph = L4_RATE_UNITS_PER_HOUR
        elif pinned_hardware == "H100":
            rate_uph = 12.0
        else:
            rate_uph = 8.0
        manifest["requirements"] = {
            "gpu": pinned_hardware,
            "allowed_gpus": [pinned_hardware],
            "min_vcpu": 2,
            "est_minutes": est_minutes,
        }
        manifest["budget"]["rate_units_per_hour"] = rate_uph
        manifest["budget"]["unit_cap"] = round((timeout_minutes * rate_uph) / 60.0 + 1e-9, 4)
    else:
        prepare_preference(manifest)

    if canary_receipt is not None and prior_output_dir is None:
        colab_canary.check(manifest, ROOT)

    manifest_path = pkt / "dispatch-manifest.json"
    write_json(manifest_path, manifest)
    return {
        "packet_dir": str(pkt),
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "mode": mode,
        "job_id": job_id,
        "commit": resolved_commit,
        "gpu": manifest["requirements"]["gpu"],
        "timeout_minutes": timeout_minutes,
        "unit_cap": manifest["budget"]["unit_cap"],
        "a100_unit_cap": a100_unit_cap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Target packet directory.")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=("smoke", "shakeout", "pilot"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--smoke-result", type=Path, default=None)
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--gpu", choices=("H100", "A100", "L4"), default=None)
    parser.add_argument("--base-seconds", type=float, default=None)
    parser.add_argument("--prior-output-dir", type=Path, default=None)
    parser.add_argument("--attempt", type=str, default="attempt-01")
    args = parser.parse_args()
    info = prepare_b1b_packet(
        args.packet,
        commit=args.commit,
        mode=args.mode,
        output_dir=args.output_dir,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        smoke_result_json=args.smoke_result,
        canary_receipt=args.canary_receipt,
        gpu=args.gpu,
        base_seconds=args.base_seconds,
        prior_output_dir=args.prior_output_dir,
        attempt=args.attempt,
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
