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

from research.compute.gpu_preference import prepare_preference
from research.gpu_sitl.preflight import resolve_product_checkpoints
from research.gpu_sitl.probe_bank import verify_file_sha256, write_atomic_with_sha256
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    PROBE32_NAMESPACE,
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    train_seeds,
)
from web.backend.served_checkpoint import B8_S3_SHA256 as B8_SHA
from web.backend.served_checkpoint import FRP3_S12_SHA256 as FRP3_SHA

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_DOC = "docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md"


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
    base_seconds: float | None = None,
    prior_output_dir: Path | str | None = None,
    attempt: str = "attempt-01",
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Step B1b-Smoke or Step B1b Pilot (no dispatch)."""
    if mode not in ("smoke", "pilot"):
        raise ValueError(f"mode must be 'smoke' or 'pilot', got {mode!r}")

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_commit, "origin/main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if ancestor_check.returncode != 0:
        raise ValueError(
            f"Refusing to prepare packet from unmerged commit {resolved_commit}: "
            "commit is not an ancestor of origin/main"
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

    inputs_list: list[dict[str, str]] = [
        {"path": str(b8_path), "dest": "b8-s3.pth", "sha256": B8_SHA},
        {"path": str(opp_path), "dest": "frp3-s12.pth", "sha256": FRP3_SHA},
    ]

    extra_env: dict[str, str] = {}
    if mode == "smoke":
        # Registered Step B1b-Smoke: short-canary <= 25 min, cap 3.0 A100 u (base 12.5 min / 1.5 u)
        est_minutes = 12.5
        timeout_minutes = 25.0
        a100_unit_cap = 3.0
        job_id = f"t34-gpu-sitl-b1b-smoke-20261109-{attempt}"
        required_exports = [
            "smoke_result.json",
            "smoke_result.json.sha256",
            "smoke_result.json.complete",
        ]
        resume_fields: dict[str, Any] = {
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    "Step B1b-Smoke registered throughput and CUDA row-subset parity canary "
                    "(<= 25 min, cap 3.0 A100 u); no scoring look"
                ),
            }
        }
        base_sec = est_minutes * 60.0
        eff_sec = timeout_minutes * 60.0
    else:
        # Step B1b (`B1-A100-Pilot`): requires measured smoke_result_json or explicit base_seconds
        if smoke_result_json is not None:
            smoke_data = json.loads(Path(smoke_result_json).read_text(encoding="utf-8"))
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
        pkt_inputs_dir = pkt / "inputs"
        pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
        ledger: dict[str, Any] = {
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
        if prior_output_dir is not None:
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
        # Also write top-level copy for backwards-compatible inspection
        write_json(pkt / "verified-ledger.json", ledger)
        inputs_list.append(
            {
                "path": str(ledger_path),
                "dest": "verified-ledger.json",
                "sha256": ledger_sha,
            }
        )
        required_exports = [
            "probe32_bank.pt",
            "probe32_bank.pt.sha256",
            "probe32_bank.pt.complete",
            "b8_sitl_student_final.pth",
            "b8_sitl_student_final.pth.sha256",
            "b8_sitl_student_final.pth.complete",
            "sitl_pilot_result.json",
            "sitl_pilot_result.json.sha256",
            "sitl_pilot_result.json.complete",
        ]
        resume_fields = {
            "resume": {
                "unit_kind": "update",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        }

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": resolved_commit,
        "script": "research/gpu_sitl/job.sh",
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
        "registration_binding": {
            "doc": REGISTRATION_DOC,
            "sha256": reg_sha,
            "train_namespace": TRAIN_NAMESPACE,
            "probe32_namespace": PROBE32_NAMESPACE,
            "preflight_namespace": PREFLIGHT_NAMESPACE,
        },
        "technical_envelope": {
            "base_seconds": base_sec,
            "effective_seconds": eff_sec,
            "registered_a100_unit_cap": a100_unit_cap,
            "mode": mode,
        },
    }
    prepare_preference(manifest)

    manifest_path = pkt / "dispatch-manifest.json"
    write_json(manifest_path, manifest)
    return {
        "packet_dir": str(pkt),
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "mode": mode,
        "job_id": job_id,
        "commit": resolved_commit,
        "timeout_minutes": timeout_minutes,
        "unit_cap": manifest["budget"]["unit_cap"],
        "a100_unit_cap": a100_unit_cap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Target packet directory.")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--smoke-result", type=Path, default=None)
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
        base_seconds=args.base_seconds,
        prior_output_dir=args.prior_output_dir,
        attempt=args.attempt,
    )
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
