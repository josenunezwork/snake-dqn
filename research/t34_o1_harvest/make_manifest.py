"""Build the qualified, no-dispatch Colab manifest for the O1 harvest census."""

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
from research.t34_o1_harvest.registration import require_registration
from research.t34_o1_harvest.timing import TIMING_FIXTURE, validate_timing

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/t34_o1_harvest/launch.sh"
L4_RATE_UNITS_PER_HOUR = 2.0
UNIT_CAP = 8.0
RESERVE_UNITS = 1.6
RUNTIME_CAP_MINUTES = 240
ORDINARY_RUNTIME_CAP_MINUTES = 192
ESTIMATE_MAX_MINUTES = 120
TIMING_FIXTURE_UNIT_CAP = 1.0
MIN_VCPU = 12
WORKERS = 11


class ManifestRefused(ValueError):
    """The O1 manifest cannot be safely prepared."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _full_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _require_catalog(catalog: Path) -> None:
    try:
        load_public_catalog(catalog)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid public-input catalog: {catalog}") from exc


def _preflight_sizing(
    receipt_path: Path, registration: Path, bank: dict[str, Any]
) -> dict[str, Any]:
    """Bind the production estimate to the excluded real-entrypoint fixture."""
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise ManifestRefused("preflight receipt must be an existing absolute file")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused("preflight receipt is not JSON") from exc
    if not isinstance(receipt, dict):
        raise ManifestRefused("preflight receipt must be a JSON object")
    if (
        receipt.get("schema") != "t34-o1-run/v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("mode") != "preflight"
        or receipt.get("registration_sha256") != _sha(registration)
        or receipt.get("timing_fixture") != TIMING_FIXTURE
    ):
        raise ManifestRefused("preflight receipt is not the completed registered O1 fixture")
    try:
        entries = receipt["worlds"]
        if len(entries) != 2:
            raise ValueError("expected two timing worlds")
        worlds = []
        for i, entry in enumerate(entries):
            if entry["path"] != f"world-{i:03d}.json":
                raise ValueError("preflight payload name differs")
            path = receipt_path.parent / entry["path"]
            if _sha(path) != entry["sha256"]:
                raise ValueError("preflight payload digest differs")
            worlds.append(json.loads(path.read_text()))
        validate_timing(worlds, bank[TIMING_FIXTURE["rows"]])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestRefused(f"preflight timing payloads differ: {exc}") from exc
    forecast = receipt.get("forecast")
    if not isinstance(forecast, dict):
        raise ManifestRefused("preflight receipt lacks its L4 forecast")
    from research.t34_o1_harvest.runner import fixture_forecast

    if forecast != fixture_forecast(worlds, WORKERS):
        raise ManifestRefused("preflight forecast differs from verified timing payloads")
    seconds = forecast.get("l4_forecast_seconds_including_setup_export")
    units = forecast.get("l4_forecast_units_at_2_per_hour")
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (seconds, units)
        )
        or seconds <= 0
        or units <= 0
    ):
        raise ManifestRefused("preflight L4 forecast must contain finite positive time and units")
    estimate = math.ceil(seconds / 60)
    if estimate > ESTIMATE_MAX_MINUTES:
        raise ManifestRefused("preflight forecast exceeds the 120-minute manifest estimate ceiling")
    if seconds > ORDINARY_RUNTIME_CAP_MINUTES * 60:
        raise ManifestRefused(
            "preflight forecast exceeds the 192-minute ordinary runtime allowance"
        )
    if units > UNIT_CAP - RESERVE_UNITS:
        raise ManifestRefused("preflight forecast exceeds the non-reserve unit allowance")
    return {
        "path": str(receipt_path),
        "sha256": _sha(receipt_path),
        "estimate_minutes": estimate,
        "l4_forecast_seconds_including_setup_export": seconds,
        "l4_forecast_units_at_2_per_hour": units,
    }


def make_manifest(
    *,
    registration: Path,
    catalog: Path,
    output_dir: Path,
    commit: str,
    preflight_receipt: Path | None = None,
) -> dict[str, Any]:
    """Return the one-lane O1 manifest without allocating or fetching inputs."""
    if not _full_sha(commit):
        raise ManifestRefused("commit must be a full lowercase SHA")
    if not output_dir.is_absolute() or output_dir.exists():
        raise ManifestRefused("output_dir must be fresh and absolute")
    try:
        frozen = require_registration(registration)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestRefused(f"invalid frozen registration: {exc}") from exc
    if frozen.get("source_commit") != commit:
        raise ManifestRefused("manifest commit must equal registration source_commit")
    _require_catalog(catalog)
    # The callable remains useful for pure shape tests.  The command-line
    # production path requires this receipt, so no manifest published through
    # this module can retain the provisional ceiling-only estimate.
    preflight = (
        _preflight_sizing(preflight_receipt, registration, frozen["bank"])
        if preflight_receipt is not None
        else None
    )
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc

    manifest = {
        "job_id": "t34-o1-harvest",
        "commit": commit,
        "script": SCRIPT,
        "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "SNAKE_DQN_DEVICE": "cpu",
            "T34_O1_CATALOG": "public-inputs.json",
            "T34_O1_MODE": "campaign",
            "T34_O1_SOURCE_COMMIT": commit,
            "T34_O1_WORKERS": str(WORKERS),
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": MIN_VCPU,
            "est_minutes": preflight["estimate_minutes"] if preflight else ESTIMATE_MAX_MINUTES,
        },
        "timeout_minutes": RUNTIME_CAP_MINUTES,
        "output_dir": str(output_dir),
        "inputs": [
            {
                "path": str(registration.resolve()),
                "sha256": _sha(registration),
                "dest": "registration.json",
            },
            {
                "path": str(catalog.resolve()),
                "sha256": _sha(catalog),
                "dest": "public-inputs.json",
            },
        ],
        "budget": {
            "unit_cap": UNIT_CAP,
            "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
            "reserve_units": RESERVE_UNITS,
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        # Registration's AST-derived closure is the authoritative finite
        # transport inventory.  It includes the read-only frozen machinery,
        # public catalog, and this entrypoint, while excluding Mac assets.
        "source_paths": sorted(frozen["source_hashes"]),
        "o1_budget": {
            "schema": "t34-o1-harvest-budget/v1",
            "unit_cap": UNIT_CAP,
            "runtime_cap_minutes": RUNTIME_CAP_MINUTES,
            "ordinary_runtime_cap_minutes": ORDINARY_RUNTIME_CAP_MINUTES,
            "reserve_fraction": 0.20,
            "reserve_units": RESERVE_UNITS,
            "timing_fixture": {
                "worlds": 2,
                "roots_per_world": 2,
                "unit_cap": TIMING_FIXTURE_UNIT_CAP,
                "must_admit_full_bank": True,
            },
            "campaign": {"worlds": 48, "max_roots_per_world": 8, "horizon": 500},
            "workers": WORKERS,
            "precision": "cpu-fp32",
        },
        "preflight": preflight,
    }
    if preflight:
        manifest["inputs"].append(
            {
                "path": preflight["path"],
                "sha256": preflight["sha256"],
                "dest": "preflight-receipt.json",
            }
        )
    return manifest


def _verify_pinned_sources(manifest: dict[str, Any], registration: dict[str, Any]) -> None:
    """Ensure the registered executable hashes name the exact pinned tree."""
    source_hashes = registration.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ManifestRefused("registration source_hashes are absent")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ManifestRefused("registration source inventory is malformed")
        try:
            pinned = subprocess.check_output(
                ["git", "show", f"{manifest['commit']}:{name}"], cwd=ROOT, timeout=20
            )
        except subprocess.SubprocessError as exc:
            raise ManifestRefused(f"registered source absent from pinned commit: {name}") from exc
        if hashlib.sha256(pinned).hexdigest() != digest:
            raise ManifestRefused(f"registered source differs from pinned commit: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args()
    commit = args.commit or _commit(ROOT)
    try:
        manifest = make_manifest(
            registration=args.registration,
            catalog=args.catalog,
            output_dir=args.output_dir,
            commit=commit,
            preflight_receipt=args.preflight_receipt,
        )
        registration = require_registration(args.registration)
        _verify_pinned_sources(manifest, registration)
        # Both calls are pure local checks.  They deliberately do not call the
        # Colab executable, allocate a session, or download the public inputs.
        from research.colab_cli import source
        from research.colab_cli.supervisor import validate

        validate(manifest, repo=ROOT)
        with tempfile.TemporaryDirectory(prefix="t34-o1-source-") as directory:
            source.prepare(ROOT, manifest, Path(directory))
    except (ManifestRefused, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        parser.error(f"manifest output already exists: {args.out}")


if __name__ == "__main__":
    main()
