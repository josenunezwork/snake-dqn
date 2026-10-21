"""L1 D0 manifest: one L4 job under the qualified Colab CLI supervisor (never dispatched here).

Inputs: the pinned public catalog (B8 s3 + league checkpoints for replay) and the
hash-bound excluded-A5 tarball from :mod:`research.t34_claim_a_loss.inputs`
(supervisor upload, ``inputs/d0-exports.tar.gz``).  Budget: priced at 2.0 u/h
over the whole-session timeout, capped inside the L1 lane cap of 12 u
(JUDGMENT §2).  The builder is pure: it never allocates; ``--validate`` runs only
the supervisor's local validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.t34_claim_a import make_manifest as claim_a
from research.t34_exit_b8 import make_manifest as ratified
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport

from . import inputs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_claim_a_loss"
SOURCE_PATHS = sorted(set(claim_a.SOURCE_PATHS) | {PACKAGE})
LANE_CAP_UNITS = 12.0
L4_RATE_UNITS_PER_HOUR = 2.0
# Bound: bootstrap 8 + unpack 2 + cache 32 (Mac smoke: one A5 world replays in 3.5 s on one
# core; 39 worlds on 11 L4 workers at a 2-3x slower vCPU is minutes, bounded generously)
# + free-Q 5 + cells (~43k CUDA updates + held-out predictions) <= 45 + decide 1 = ~93.
WORKLOAD_BOUND_MINUTES = 95
SETUP_EXPORT_MARGIN_MINUTES = 25
CLAIM = "D0_DESIGN_DIAGNOSTIC_EXCLUDED_A5_NOT_EVIDENCE"
MAX_TARBALL_BYTES = 128 * 1024**2


class ManifestRefused(ValueError):
    """The D0 draft cannot become a supervisor job."""


def make_d0_manifest(
    *,
    commit: str,
    output_dir: Path,
    tarball: Path,
    est_minutes: int = 100,
    timeout_minutes: int = 240,
    unit_cap: float | None = None,
) -> dict[str, Any]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    timeout = int(timeout_minutes)
    if timeout < 2 * int(est_minutes):
        raise ManifestRefused("timeout must be at least twice the estimate")
    if timeout < WORKLOAD_BOUND_MINUTES + SETUP_EXPORT_MARGIN_MINUTES:
        raise ManifestRefused("timeout must cover the bounded workload plus 25 minutes")
    priced = timeout * L4_RATE_UNITS_PER_HOUR / 60
    cap = float(unit_cap) if unit_cap is not None else priced
    if priced > cap:
        raise ManifestRefused("whole-session timeout priced above the unit cap")
    if cap > LANE_CAP_UNITS:
        raise ManifestRefused(f"D0 job cap {cap} u exceeds the L1 lane cap {LANE_CAP_UNITS} u")
    tarball = Path(tarball).expanduser().resolve(strict=True)
    if tarball.stat().st_size > MAX_TARBALL_BYTES:
        raise ManifestRefused("D0 tarball exceeds the bounded upload size")
    tar_sha = inputs.sha256_file(tarball)
    catalog = ratified.PUBLIC_CATALOG_PATH.resolve(strict=True)
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc
    return {
        "job_id": f"t34-claim-a-d0-{commit[:8]}",
        "commit": commit,
        "script": f"{PACKAGE}/d0.sh",
        "claim": CLAIM,
        "transport": {"mode": "public_release_urls", "dispatch_ready": True},
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_D0_TARBALL_SHA256": tar_sha,
            "T34_D0_DEVICE": "cuda",
            "T34_D0_ANCHOR_STRIDE": "1",
            "T34_D0_EXPECT_WORLDS": str(inputs.EXPECTED_A5_WORLDS),
            "T34_D0_THREADS": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": int(est_minutes),
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir),
        "inputs": [
            {
                "path": str(catalog),
                "sha256": ratified.sha256_file(catalog),
                "dest": "public-inputs.json",
            },
            {"path": str(tarball), "sha256": tar_sha, "dest": "d0-exports.tar.gz"},
        ],
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
            "reserve_units": round(cap * 0.2, 3),
            # a mapping (or absence) is required; a list is refused at admission
            "other_assignments": {},
        },
        "d0": {
            "lane": "L1",
            "lane_cap_units": LANE_CAP_UNITS,
            "spec": "docs/research/specs/2026-10-21_t34_claim_a_amendment.md",
            "data": "excluded A5 v2 exports only (39 worlds, B8 s3 carrier and rollout)",
            "rule": f"{PACKAGE}/decide.py",
        },
        "workload_bound_minutes": WORKLOAD_BOUND_MINUTES,
        "lease_seconds": 600,
        "max_export_bytes": 256 * 1024**2,
        "source_paths": SOURCE_PATHS,
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    return claim_a.validate_manifest(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--est-minutes", type=int, default=100)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    parser.add_argument("--unit-cap", type=float)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    manifest = make_d0_manifest(
        commit=args.commit or ratified.git_head(),
        output_dir=args.output_dir,
        tarball=args.tarball,
        est_minutes=args.est_minutes,
        timeout_minutes=args.timeout_minutes,
        unit_cap=args.unit_cap,
    )
    if args.validate:
        validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.out), "job_id": manifest["job_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
