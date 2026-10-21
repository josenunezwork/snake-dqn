"""Prepare one of three local-validated, non-dispatched O2 L4 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from research.t34_exit_b8.public_inputs import _load as load_public_catalog
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport
from research.t34_o2_harvest.registration import CAPS, require_registration

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/t34_o2_harvest/launch.sh"
CHUNKS = ((0, 32, "scripted"), (32, 32, "frozen"), (64, 32, "mixed"))
L4_RATE = 2.0
MAX_EXPORT_BYTES = 64 * 1024**2


class ManifestRefused(ValueError):
    """A proposed O2 transport plan failed a local admission condition."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_sha(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _preflight(receipt_path: Path, registration: Path) -> dict[str, Any]:
    """Bind a completed excluded CPU timing preflight; never invent L4 speed."""
    try:
        from research.t34_o2_harvest.preflight import validate_preflight

        checked = validate_preflight(receipt_path, registration)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid O2 timing preflight: {exc}") from exc
    forecast = checked.get("forecast", {})
    seconds, minutes = forecast.get("worst_chunk_seconds"), forecast.get("estimated_minutes")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or not 0 < seconds <= 7200
        or type(minutes) is not int
        or not 3 <= minutes <= 120
        or forecast.get("fits_twofold_timeout_and_24_unit_envelope") is not True
    ):
        raise ManifestRefused("preflight forecast does not admit a <=120-minute O2 chunk")
    return checked


def make_manifest(
    *,
    registration: Path,
    catalog: Path,
    preflight_receipt: Path,
    output_dir: Path,
    commit: str,
    chunk: int,
) -> dict[str, Any]:
    """Build a single O2 chunk manifest without allocation, auth, or download."""
    if not _full_sha(commit):
        raise ManifestRefused("commit must be a full lowercase SHA")
    if chunk not in range(len(CHUNKS)):
        raise ManifestRefused("chunk must be 0, 1, or 2")
    if not output_dir.is_absolute() or output_dir.exists():
        raise ManifestRefused("output_dir must be a fresh absolute path")
    try:
        frozen = require_registration(registration)
        load_public_catalog(catalog)
        require_qualified_transport(ROOT, commit)
    except (OSError, ValueError, json.JSONDecodeError, QualificationRefused) as exc:
        raise ManifestRefused(str(exc)) from exc
    if frozen["source_commit"] != commit:
        raise ManifestRefused("manifest commit must equal registration source_commit")
    try:
        timing = _preflight(preflight_receipt, registration)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestRefused(str(exc)) from exc
    start, count, mix = CHUNKS[chunk]
    timeout_minutes = int(CAPS["max_job_seconds"] // 60)
    estimate_minutes = timing["forecast"]["estimated_minutes"]
    if timeout_minutes >= 24 * 60 or timeout_minutes * L4_RATE / 60 > CAPS["units_per_job"]:
        raise ManifestRefused("registered per-job cap is unsafe")
    return {
        "job_id": f"t34-o2-harvest-{chunk:02d}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "SNAKE_DQN_DEVICE": "cpu",
            "T34_O2_MODE": "campaign",
            "T34_O2_WORKERS": str(CAPS["workers_per_job"]),
            "T34_O2_START": str(start),
            "T34_O2_COUNT": str(count),
            "T34_O2_TIMING_REQUIRED": "1",
            "T34_O2_MAX_FORECAST_SECONDS": str(CAPS["ordinary_job_seconds"]),
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": estimate_minutes,
        },
        "timeout_minutes": timeout_minutes,
        "output_dir": str(output_dir),
        "inputs": [
            {
                "path": str(registration.resolve()),
                "sha256": _sha(registration),
                "dest": "registration.json",
            },
            {"path": str(catalog.resolve()), "sha256": _sha(catalog), "dest": "public-inputs.json"},
            {"path": timing["path"], "sha256": timing["sha256"], "dest": "preflight-receipt.json"},
        ],
        "budget": {
            "unit_cap": CAPS["units_per_job"],
            "rate_units_per_hour": L4_RATE,
            "reserve_units": CAPS["reserve_units_per_job"],
        },
        "lease_seconds": 600,
        "max_export_bytes": MAX_EXPORT_BYTES,
        "source_paths": sorted(frozen["source_hashes"]),
        "o2": {
            "chunk": chunk,
            "start": start,
            "count": count,
            "mix": mix,
            "total_worlds": 96,
            "concurrency_cap": 3,
            "cpu_only_worker": True,
            "workers": CAPS["workers_per_job"],
            "aggregate_unit_cap": CAPS["units"],
            "aggregate_reserve_units": CAPS["reserve_units"],
            "preflight": timing,
            "l4_rate_claim": "none; CPU preflight only",
            "timing_before_study": {"required": True, "excluded_rows": 3, "horizon": 5000},
        },
    }


def _verify_sources(manifest: dict[str, Any], registration: dict[str, Any]) -> None:
    for name, digest in registration["source_hashes"].items():
        try:
            payload = subprocess.check_output(
                ["git", "show", f"{manifest['commit']}:{name}"], cwd=ROOT, timeout=20
            )
        except subprocess.SubprocessError as exc:
            raise ManifestRefused(f"registered source absent from pinned commit: {name}") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ManifestRefused(f"registered source differs from pinned commit: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        manifest = make_manifest(
            registration=args.registration,
            catalog=args.catalog,
            preflight_receipt=args.preflight_receipt,
            output_dir=args.output_dir,
            commit=args.commit,
            chunk=args.chunk,
        )
        registration = require_registration(args.registration)
        _verify_sources(manifest, registration)
        from research.colab_cli import source
        from research.colab_cli.supervisor import validate

        validate(manifest, repo=ROOT)
        with tempfile.TemporaryDirectory(prefix="t34-o2-source-") as directory:
            source.prepare(ROOT, manifest, Path(directory))
    except (ManifestRefused, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        parser.error(f"manifest output already exists: {args.out}")


if __name__ == "__main__":
    main()
