"""Create-only preparation and sealing of the P engineering probes; never dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p.protocol import FIXTURE_SEED, SEEDS
from tools.sandbox.registry import DEFAULT_REGISTRY, load_registry, world_seed

ROOT = Path(__file__).resolve().parents[2]
STUDY = "p-successor-profile-v1"


def bank_record() -> dict:
    """Canonical engineering allocation, burned for any successor inference."""
    return {
        "study": STUDY,
        "namespace": f"a100-dev/{STUDY}/",
        "indices": list(range(4)),
        "seeds": [world_seed(STUDY, i) for i in range(4)],
        "successor_eligible": False,
    }


def audit_bank() -> dict:
    """Compare public seed metadata only; never discover seeds through payloads."""
    registry = load_registry()
    excluded = set(registry["excluded_seeds"]) | set(SEEDS) | {FIXTURE_SEED}
    excluded.update(world_seed("t34-watch-lead-1028", i) for i in range(32, 36))
    for start, count in ((400_000_000, 288), (500_000_000, 768)):
        for i in range(start, start + count):
            raw = hashlib.sha256(f"a100-dev/sandbox-a-train01/{i}".encode()).digest()[:8]
            excluded.add(int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF)
    bank = bank_record()
    seeds = bank["seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) & excluded:
        raise ValueError("fresh profile bank collision; reject entire allocation")
    return {
        "bank": bank,
        "registry_sha256": digest(DEFAULT_REGISTRY),
        "excluded_count": len(excluded),
        "collisions": [],
        "coverage": (
            "registry + P32/engineering32-35 + TRAIN01 monitor/nomination; "
            "no exhaustive historical claim"
        ),
        "old_P_payloads_read": False,
    }


def write(path: Path, value: object) -> None:
    """Refuse overwriting any prepared packet or evidence."""
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def prepare(pin: str, packet: Path) -> None:
    """Pin scripts/inputs, bound engineering time, record whole-timeout pricing."""
    if (
        subprocess.check_output(
            ["git", "rev-parse", f"{pin}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
        != pin
    ):
        raise ValueError("full source commit required")
    paths = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    paths += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "research/watch_lead_p_successor").glob("*.py"))
    ]
    paths += [
        "research/watch_lead_p_successor/profile_job.sh",
        "research/watch_lead_p_successor/launch.sh",
        "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
    ]
    paths = sorted(set(paths))
    for path in paths:
        blob = subprocess.check_output(["git", "show", f"{pin}:{path}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest(ROOT / path):
            raise ValueError(f"uncommitted runtime drift: {path}")
    packet.mkdir(parents=True, exist_ok=False)
    write(packet / "profile-bank.json", bank_record())
    write(packet / "exclusion-audit.json", audit_bank())
    help_result = subprocess.run(
        ["colab", "new", "--help"], text=True, capture_output=True, check=True, timeout=30
    )
    (packet / "colab-new-help.txt").write_text(help_result.stdout + help_result.stderr)
    versions = subprocess.check_output(["colab", "version"], text=True, timeout=30)
    (packet / "colab-version.txt").write_text(versions)
    checks = ROOT.parent / "snake-dqn/saved_snakes"
    inputs = [
        (checks / "b8_s3_t7_20261016.pth", "b8-s3.pth"),
        (checks / "frp3_m3_s12_u60000_20261005.pth", "frp3-s12.pth"),
        (packet / "profile-bank.json", "profile-bank.json"),
    ]
    envelope = {}
    for gpu, rate in (("L4", 1.54), ("A100", 5.30)):
        name = f"p-successor-profile-v1-{gpu.lower()}"
        base_cap = math.ceil(15 / 60 * rate * 100) / 100
        manifest = {
            "job_id": name,
            "p_profile_source_delivery": "background-sparse-v1",
            "commit": pin,
            "script": "research/watch_lead_p_successor/profile_job.sh",
            "device": "cpu",
            "requirements": {"gpu": gpu, "allowed_gpus": [gpu], "min_vcpu": 12, "est_minutes": 15},
            "timeout_minutes": 30,
            "source_paths": paths,
            "output_dir": str(packet / gpu.lower() / "output"),
            "max_export_bytes": 16 * 1024 * 1024,
            "lease_seconds": 180,
            "inputs": [
                {"path": str(path), "sha256": digest(path), "dest": dest} for path, dest in inputs
            ],
            "env": {
                "SNAKE_DQN_DEVICE": "cpu",
                "COLAB_CLI_REQUIREMENTS": "research/watch_lead_p/requirements.txt",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/watch_lead_p_successor",
            },
            "requirements_file": "research/watch_lead_p/requirements.txt",
            "budget": {
                "rate_units_per_hour": rate,
                "unit_cap": 2 * base_cap,
                "reserve_units": 2 * base_cap * 0.2,
                "other_assignments": {},
            },
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    "Fresh excluded fixed CPU profiling windows; "
                    "no scientific look or automatic completion"
                ),
            },
            "timing_calibration": {"no_automatic_retry": True},
            "canary": {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": "research/watch_lead_p_successor/profile_job.sh",
            },
            "required_exports": ["profile/host.json", "profile/profile.json"]
            + [f"profile/sweep-w{w}.json" for w in (1, 4, 8, 11)],
        }
        write(packet / f"{gpu.lower()}-manifest.json", manifest)
        envelope[gpu] = {
            "base_minutes": 15,
            "effective_minutes": 30,
            "base_unit_cap": base_cap,
            "effective_unit_cap": 2 * base_cap,
            "base_reserve_units": base_cap * 0.2,
            "effective_reserve_units": 2 * base_cap * 0.2,
            "whole_timeout_units": 0.5 * rate,
            "rate_ceiling": rate,
            "status": "finite engineering guard, NOT a measured completion forecast",
        }
        command = (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "
            "VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 "
            "SNAKE_DQN_DEVICE=cpu nice -n 15 "
            + shlex.join(
                [
                    str(ROOT / "tools/ops/dispatch.sh"),
                    name,
                    str(packet / f"{gpu.lower()}-manifest.json"),
                    "--broker",
                    str(ROOT),
                    "--cwd",
                    str(ROOT),
                    "--state-dir",
                    str(packet / gpu.lower() / "state"),
                    "--dispatch-txt",
                    str(packet / "DISPATCH.txt"),
                ]
            )
        )
        (packet / ("canary-command.sh" if gpu == "L4" else "compare-command.sh")).write_text(
            command + "\n"
        )
    (packet / "DISPATCH.txt").write_text(
        "Engineering CPU profile only; operator executes once per host, sequentially.\n"
        "No old P exports are inputs. No successor scientific manifest exists.\n"
        "L4 primary; A100 optional CPU-host comparison, no higher core count promised.\n"
        "No automatic allocation retry or completion. Require verified release before next host.\n"
    )
    write(packet / "envelopes.json", envelope)
    write(packet / "source-hashes.json", {p: digest(ROOT / p) for p in paths})
    write(
        packet / "seal.json", {p.name: digest(p) for p in sorted(packet.iterdir()) if p.is_file()}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pin, args.packet.resolve())


if __name__ == "__main__":
    main()
