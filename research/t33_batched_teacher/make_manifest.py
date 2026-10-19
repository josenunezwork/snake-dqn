"""Prepare, but never dispatch, T33's one-L4 throughput benchmark manifest.

The normal transport is a portable public-release catalog.  Until its URLs are
published, it emits an explicitly blocked handoff rather than an allocatable
job.  The six-file local-upload route remains available only for local
supervisor preflight; this module never publishes or downloads inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = Path(__file__).with_name("INPUTS_REQUIRED.json")
PUBLIC_CATALOG_PATH = Path(__file__).with_name("PUBLIC_INPUTS.json")
JOB_ID = "t33-batched-teacher-l4-throughput"
SCRIPT = "research/t33_batched_teacher/bench.sh"
L4_RATE_UNITS_PER_HOUR = 2.0
ESTIMATE_MINUTES = 30
TIMEOUT_MINUTES = 60
MAX_EXPORT_BYTES = 16 * 1024**2


class ManifestRefused(ValueError):
    """An inventory or output path cannot safely produce a transport manifest."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path = ROOT) -> str:
    """Read the commit that will be pinned by a draft manifest."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_inventory(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid input inventory: {path}") from exc
    rows = value.get("inputs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 6:
        raise ManifestRefused("T33 inventory must contain exactly six raw checkpoint files")
    return rows


def _load_public_catalog(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid public input catalog: {path}") from exc
    rows = value.get("inputs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 6:
        raise ManifestRefused("public input catalog must contain exactly six checkpoint files")
    return rows


def upload_inputs(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Verify inventory bytes and return only supervisor upload fields."""
    inputs: list[dict[str, str]] = []
    destinations: set[str] = set()
    for row in rows:
        name, raw_path, dest, expected = (
            row.get("name"),
            row.get("local_path"),
            row.get("dest"),
            row.get("sha256"),
        )
        if not all(isinstance(item, str) and item for item in (name, raw_path, dest, expected)):
            raise ManifestRefused("each inventory row requires name, local_path, dest, and sha256")
        if dest in destinations or Path(dest).is_absolute() or ".." in Path(dest).parts:
            raise ManifestRefused(f"unsafe or duplicate upload destination: {dest!r}")
        destinations.add(dest)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ManifestRefused(f"missing local input: {path}")
        if path.stat().st_size != row.get("size_bytes"):
            raise ManifestRefused(f"input size drift: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ManifestRefused(f"input hash drift: {path}")
        inputs.append({"path": str(path), "sha256": actual, "dest": dest})
    expected_destinations = {
        "B8_s3_final.pth",
        "checkpoints/champion_a5_freespace_20260621.pth",
        "checkpoints/best_apex_fs.pth",
        "checkpoints/best_apex_stage1_fs.pth",
        "checkpoints/best_apex_pre_fs.pth",
        "checkpoints/frp3_m3_s12_u60000_20261005.pth",
    }
    if destinations != expected_destinations:
        raise ManifestRefused("inventory does not bind T33's B8 and five frozen checkpoints")
    return inputs


def _catalog_matches_inventory(
    catalog: list[dict[str, Any]], inventory: list[dict[str, Any]]
) -> bool:
    """Accept URLs only when every immutable identity matches the local inventory."""
    public = {
        row.get("name"): (row.get("dest"), row.get("size_bytes"), row.get("sha256"))
        for row in catalog
    }
    local = {
        row.get("name"): (row.get("dest"), row.get("size_bytes"), row.get("sha256"))
        for row in inventory
    }
    return public == local and len(public) == 6


def _valid_public_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("https://") and len(value) > len("https://")


def public_catalog_input(catalog_path: Path) -> dict[str, str]:
    """Bind the one portable catalog the supervisor uploads for public mode."""
    path = catalog_path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path), "dest": "public-inputs.json"}


def make_manifest(
    *,
    commit: str,
    output_dir: Path,
    inventory_path: Path = INVENTORY_PATH,
    public_catalog_path: Path = PUBLIC_CATALOG_PATH,
    local_upload_preflight: bool = False,
) -> dict[str, Any]:
    """Build a public blocked/ready draft or explicit local-upload preflight manifest."""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    output = output_dir.expanduser()
    if not output.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    inventory = _load_inventory(inventory_path)
    if local_upload_preflight:
        inputs = upload_inputs(inventory)
        claim = "LOCAL_UPLOAD_PREFLIGHT_ONLY"
        transport = {
            "mode": "local_upload_preflight",
            "status": "LOCAL_ONLY_NOT_A_PRODUCTION_CLAIM",
            "dispatch_ready": False,
        }
        input_mode = "local-preflight"
    else:
        catalog = _load_public_catalog(public_catalog_path)
        if not _catalog_matches_inventory(catalog, inventory):
            raise ManifestRefused(
                "public catalog identities must exactly match the authoritative inventory"
            )
        ready = all(_valid_public_url(row.get("public_url")) for row in catalog)
        inputs = [public_catalog_input(public_catalog_path)]
        claim = "DRAFT_NO_ALLOCATION_AUTHORIZED"
        transport = {
            "mode": "public_release_urls",
            "status": "PUBLICATION_READY" if ready else "PUBLICATION_BLOCKED",
            "dispatch_ready": ready,
            "publishing_pending": not ready,
        }
        input_mode = "public"
    return {
        "job_id": JOB_ID,
        "commit": commit,
        "script": SCRIPT,
        "claim": claim,
        "transport": transport,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T33_INPUT_MODE": input_mode,
            "T33_DEVICE": "cuda",
            "T33_WORLDS": "256,1024,4096",
            "T33_FALLBACK_WORLDS": "64",
            "T33_SECONDS": "1200",
            "T33_HORIZON": "128",
            "T33_ROLLOUTS": "8",
            "T33_THREADS": "12",
            "NUMBA_NUM_THREADS": "12",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": ESTIMATE_MINUTES,
        },
        "timeout_minutes": TIMEOUT_MINUTES,
        "output_dir": str(output),
        "inputs": inputs,
        "budget": {
            "unit_cap": 2.0,
            "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
            "reserve_units": 0.4,
        },
        "lease_seconds": 180,
        "max_export_bytes": MAX_EXPORT_BYTES,
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Use the local supervisor's pure validation function; never call its CLI."""
    transport = manifest.get("transport", {})
    if transport.get("mode") == "public_release_urls" and not transport.get("dispatch_ready"):
        raise ManifestRefused("public input publication is pending; manifest is not dispatch-ready")
    from research.colab_cli.supervisor import validate

    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", default=None, help="full published commit; defaults to current HEAD"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--public-catalog", type=Path, default=PUBLIC_CATALOG_PATH)
    parser.add_argument(
        "--local-upload-preflight",
        action="store_true",
        help="use six local uploads for local validation only; never a production claim",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--validate", action="store_true", help="call supervisor.validate locally only"
    )
    args = parser.parse_args()
    manifest = make_manifest(
        commit=args.commit or git_head(),
        output_dir=args.output_dir,
        inventory_path=args.inventory,
        public_catalog_path=args.public_catalog,
        local_upload_preflight=args.local_upload_preflight,
    )
    if args.validate:
        validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.out), "validated": args.validate}, sort_keys=True))


if __name__ == "__main__":
    main()
