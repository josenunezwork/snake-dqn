"""Prepare one blocked, public-input-only L4 fit-preflight manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = Path(__file__).with_name("INPUTS_REQUIRED.json")
CATALOG_PATH = Path(__file__).with_name("PUBLIC_INPUTS.json")
SCRIPT = "research/t34_exit_b8_fit/fit.sh"


class ManifestRefused(ValueError):
    """A job has not met the portable P6 admission contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text())["inputs"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestRefused(f"invalid input catalog: {path}") from exc
    if not isinstance(rows, list) or not rows:
        raise ManifestRefused("input catalog must contain rows")
    return rows


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("name"),
        row.get("dest"),
        row.get("size_bytes"),
        row.get("sha256"),
        json.dumps(row.get("members", []), sort_keys=True),
    )


def _public_ready(catalog: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> bool:
    if {_identity(row) for row in catalog} != {_identity(row) for row in inventory}:
        raise ManifestRefused("public catalog identities differ from INPUTS_REQUIRED")
    for row in catalog:
        if type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0:
            return False
        digest, url = row.get("sha256"), row.get("url")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
        if not isinstance(url, str) or not url.startswith("https://") or len(url) <= 8:
            return False
    return True


def validate_binding(value: Any) -> dict[str, Any]:
    """Validate the portable engine contract before local or runtime execution."""
    rows = value.get("inputs") if isinstance(value, dict) else None
    required = ("source_factory", "probe_factory", "preregistration_sha256")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or value.get("approved") is not True
        or any(not isinstance(value.get(key), str) or not value[key] for key in required)
    ):
        raise ManifestRefused(
            "binding needs one public payload, factories, preregistration SHA, approval"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", value["preregistration_sha256"]):
        raise ManifestRefused("preregistration_sha256 must be a complete SHA-256")
    if value.get("arm", "S1b") not in {"S1a", "S1b"}:
        raise ManifestRefused("arm must be S1a or S1b")
    for key in ("source_factory", "probe_factory"):
        if not re.fullmatch(r"[a-zA-Z_][\w.]*:[a-zA-Z_]\w*", value[key]):
            raise ManifestRefused(f"{key} must name a pinned module:function")
    row = rows[0]
    if not isinstance(row, dict) or row.get("complete_marker") is not True:
        raise ManifestRefused("engine payload must request verified completion markers")
    if row.get("dest") != "engine-export/payload.json" or not _public_ready(rows, rows):
        raise ManifestRefused("engine payload public identity/destination is incomplete")
    return value


def _binding(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ManifestRefused(f"invalid engine binding: {path}") from exc
    return validate_binding(value)


def make_manifest(
    *,
    commit: str,
    output_dir: Path,
    inventory_path: Path = INVENTORY_PATH,
    public_catalog_path: Path = CATALOG_PATH,
    engine_binding_path: Path | None = None,
    mode: str = "preflight",
) -> dict[str, Any]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase SHA")
    if not output_dir.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    inventory, catalog = _rows(inventory_path), _rows(public_catalog_path)
    # Checkpoints are published, but the P1 export and the preregistration are
    # independent admission inputs.  A draft cannot claim allocator readiness.
    if not _public_ready(catalog, inventory):
        raise ManifestRefused("required public checkpoint URLs or identities are unavailable")
    binding = _binding(engine_binding_path)
    if mode not in {"preflight", "fit"}:
        raise ManifestRefused("mode must be preflight or fit")
    resources = {
        "est_minutes": 15,
        "timeout_minutes": 30,
        "unit_cap": 1.0,
        "rate_units_per_hour": 2.0,
        "reserve_units": 0.1,
    }
    if mode == "fit":
        required = set(resources)
        supplied = (binding or {}).get("fit_resources")
        if not isinstance(supplied, dict) or not required.issubset(supplied):
            raise ManifestRefused(
                "fit mode requires measured, approved fit_resources in the binding"
            )
        resources = {key: supplied[key] for key in required}
        if type(binding.get("worlds")) is not int or binding["worlds"] < 1:
            raise ManifestRefused("fit mode requires the preregistered positive world count")
    ready = binding is not None
    inputs = [
        {
            "path": str(public_catalog_path.resolve()),
            "sha256": sha256_file(public_catalog_path),
            "dest": "public-inputs.json",
        }
    ]
    if engine_binding_path is not None:
        inputs.append(
            {
                "path": str(engine_binding_path.resolve()),
                "sha256": sha256_file(engine_binding_path),
                "dest": "engine-binding.json",
            }
        )
    return {
        "job_id": (binding or {}).get("job_id", "t34-exit-b8-fit-" + mode),
        "commit": commit,
        "script": SCRIPT,
        "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
        "transport": {
            "mode": "public_release_urls",
            "status": (
                "READY_FOR_SUPERVISOR_VALIDATION"
                if ready
                else "ENGINE_EXPORT_AND_PREREGISTRATION_PENDING"
            ),
            "dispatch_ready": ready,
        },
        "source_pins": {
            "configs/free_space_v2.yaml": (
                "41c590461b3179ddbf9ce0ed7fdfb754fc84b188c38790077b7e1cd159335624"
            )
        },
        "required_runtime_handoff": {
            "engine_export": ["payload.json", "payload.json.sha256", "payload.json.complete"]
        },
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_FIT_INPUT_MODE": "public",
            "T34_FIT_DEVICE": "cuda",
            "T34_FIT_MODE": mode,
            "T34_FIT_EPOCHS": str((binding or {}).get("epochs", 5)),
            "T34_FIT_WORLDS": str(2 if mode == "preflight" else binding["worlds"]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 2,
            "est_minutes": resources["est_minutes"],
        },
        "timeout_minutes": resources["timeout_minutes"],
        "output_dir": str(output_dir),
        "inputs": inputs,
        "budget": {
            key: resources[key] for key in ("unit_cap", "rate_units_per_hour", "reserve_units")
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024 * 1024,
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    if not manifest.get("transport", {}).get("dispatch_ready"):
        raise ManifestRefused("public input publication is pending; manifest is not dispatch-ready")
    from research.colab_cli.supervisor import validate

    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--engine-binding", type=Path)
    parser.add_argument("--mode", choices=("preflight", "fit"), default="preflight")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    manifest = make_manifest(
        commit=args.commit or git_head(),
        output_dir=args.output_dir,
        engine_binding_path=args.engine_binding,
        mode=args.mode,
    )
    if args.validate:
        validate_manifest(manifest)
    with args.out.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
