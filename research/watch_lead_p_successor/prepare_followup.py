"""Seal a fresh, bounded L4 engineering follow-up; never dispatch."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest, verify_export
from research.watch_lead_p_successor.host import FOLLOWUP_SCRIPT
from research.watch_lead_p_successor.prepare import ROOT, audit_bank, bank_record, write


def prepare(pin: str, packet: Path, preflight: Path, phase: str) -> None:
    """Reuse only engineering input identities, never a used manifest/output."""
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != pin:
        raise ValueError("full current committed source pin required")
    if phase not in ("contention", "bare", "tail"):
        raise ValueError("foreign follow-up phase")
    summary = json.loads((preflight / "followup.json").read_text())
    if not (
        summary.get("complete") is True
        and summary.get("smoke") is True
        and summary.get("repeated_window_identity") is True
        and summary.get("phase") == phase
    ):
        raise ValueError("real follow-up smoke required")
    files = sorted(preflight.glob("*.json"))
    if {p.name for p in preflight.iterdir()} != {
        p.name + suffix for p in files for suffix in ("", ".sha256", ".complete")
    }:
        raise ValueError("orphan preflight export")
    for path in files:
        verify_export(path)
    paths = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    paths += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "research/watch_lead_p_successor").iterdir())
        if p.suffix in (".py", ".sh")
    ]
    paths += ["tools/sandbox/SANDBOX_INPUT_REGISTRY.json"]
    paths = sorted(set(paths))
    import hashlib

    for name in paths:
        blob = subprocess.check_output(["git", "show", f"{pin}:{name}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest(ROOT / name):
            raise ValueError(f"runtime source drift: {name}")
    packet.mkdir(parents=True, exist_ok=False)
    write(packet / "profile-bank.json", bank_record())
    write(packet / "exclusion-audit.json", audit_bank())
    checks = ROOT.parent / "snake-dqn/saved_snakes"
    inputs = [
        (checks / "b8_s3_t7_20261016.pth", "b8-s3.pth"),
        (checks / "frp3_m3_s12_u60000_20261005.pth", "frp3-s12.pth"),
        (packet / "profile-bank.json", "profile-bank.json"),
    ]
    manifest = {
        "job_id": f"p-successor-missing-v4-{phase}-l4",
        "p_profile_source_delivery": "background-sparse-v1",
        "commit": pin,
        "script": FOLLOWUP_SCRIPT,
        "device": "cpu",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 15},
        "timeout_minutes": 30,
        "output_dir": str(packet / "l4/output"),
        "source_paths": paths,
        "max_export_bytes": 16 * 1024 * 1024,
        "lease_seconds": 180,
        "inputs": [{"path": str(p), "dest": dest, "sha256": digest(p)} for p, dest in inputs],
        "env": {
            "SNAKE_DQN_DEVICE": "cpu",
            "P_PROFILE_FOLLOWUP_PHASE": phase,
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/watch_lead_p_successor",
            "COLAB_CLI_REQUIREMENTS": "research/watch_lead_p/requirements.txt",
        },
        "requirements_file": "research/watch_lead_p/requirements.txt",
        "budget": {
            "rate_units_per_hour": 1.54,
            "unit_cap": 0.78,
            "reserve_units": 0.156,
            "other_assignments": {},
        },
        "resume_exemption": {
            "kind": "short-canary",
            "reason": "Excluded timing only; missing contention and natural CPU-tail "
            "stress; whole-session science units are never produced",
        },
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": FOLLOWUP_SCRIPT},
        "required_exports": ["profile/host.json", "profile/followup.json"]
        + (
            ["profile/missing-contention-w8.json", "profile/missing-contention-w11.json"]
            if phase == "contention"
            else ["profile/scan-summary.json", f"profile/scan-{bank_record()['seeds'][0]}.json"]
            + (
                ["profile/tail-peak-w4.json", "profile/tail-late-w4.json"]
                if phase == "tail"
                else []
            )
        ),
    }

    write(packet / "l4-manifest.json", manifest)
    write(packet / "source-hashes.json", {p: digest(ROOT / p) for p in paths})
    write(
        packet / "preflight.json",
        {"path": str(preflight), "summary_sha256": verify_export(preflight / "followup.json")},
    )
    write(
        packet / "envelopes.json",
        {
            "base_timeout_minutes": 15,
            "effective_timeout_minutes": 30,
            "base_work_seconds": 600,
            "effective_work_seconds": 1200,
            "base_cap_units": 0.39,
            "effective_cap_units": 0.78,
            "base_reserve_units": 0.078,
            "effective_reserve_units": 0.156,
            "whole_timeout_units": 0.77,
            "rate_ceiling_units_per_hour": 1.54,
            "no_automatic_retry": True,
            "status": "Bounded engineering guard; not a full-search-session forecast",
        },
    )
    (packet / "DISPATCH.txt").write_text(
        "Engineering timing only. Operator executes once; no automatic retry.\n"
        "No old P payload input and no scientific successor dispatch.\n"
        "Native 30-minute probe, upload route unchanged; no assumed free account slot.\n"
    )
    command = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "
    command += "VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 "
    command += "SNAKE_DQN_DEVICE=cpu nice -n 15 " + shlex.join(
        [
            str(ROOT / "tools/ops/dispatch.sh"),
            manifest["job_id"],
            str(packet / "l4-manifest.json"),
            "--broker",
            str(ROOT),
            "--cwd",
            str(ROOT),
            "--state-dir",
            str(packet / "l4/state"),
            "--dispatch-txt",
            str(packet / "DISPATCH.txt"),
        ]
    )
    (packet / "canary-command.sh").write_text(command + "\n")
    write(
        packet / "seal.json", {p.name: digest(p) for p in sorted(packet.iterdir()) if p.is_file()}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--phase", choices=("contention", "bare", "tail"), required=True)
    args = parser.parse_args()
    prepare(args.pin, args.packet.resolve(), args.preflight.resolve(), args.phase)


if __name__ == "__main__":
    main()
