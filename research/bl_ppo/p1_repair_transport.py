"""Convert ruled P1 origin inputs to the qualified public HTTPS transport.

This module only changes how already-pinned bytes enter the qualified Colab
runtime.  It refuses alternate repositories, tags, paths, byte identities, or
destinations, leaving the origin importer's independent digest checks intact.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from research.bl_ppo.exports import digest, export_json

RELEASE_REPOSITORY = "josenunezwork/snake-dqn"
RELEASE_TAG = "p1-y3-origin-20261023"
RELEASE_BASE_URL = f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}"


def _asset_map(assets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate the planned immutable release inventory by release asset name."""
    if not isinstance(assets, list) or not assets:
        raise ValueError("release transport requires a nonempty selected asset inventory")
    selected: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("release asset must be an object")
        name, size, checksum = asset.get("name"), asset.get("size_bytes"), asset.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or name in selected
        ):
            raise ValueError("release asset inventory is malformed")
        selected[name] = {"name": name, "size_bytes": size, "sha256": checksum}
    return selected


def public_rows(
    manifest: dict[str, Any], assets: list[dict[str, Any]], release_base_url: str
) -> dict[str, Any]:
    """Return a copy whose selected origin files use immutable public HTTPS rows.

    ``assets`` names the bytes selected by the residual-upload forecast.  Each
    selected source must be a current local input with the same flat runtime
    destination and exact planned size and SHA-256.  The resulting URL points
    only at the ruled repository and immutable release tag.
    """
    if release_base_url != RELEASE_BASE_URL:
        raise ValueError("release URL must name the ruled immutable repository and tag")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("manifest requires an input list")
    selected = _asset_map(assets)
    converted: set[str] = set()
    result = copy.deepcopy(manifest)
    rows: list[dict[str, Any]] = []
    for row in result["inputs"]:
        if not isinstance(row, dict):
            raise ValueError("manifest input must be an object")
        dest = row.get("dest")
        if dest not in selected:
            rows.append(row)
            continue
        asset = selected[dest]
        source_name = row.get("path")
        if not isinstance(source_name, str) or "url" in row:
            raise ValueError("selected release input must be a local source row")
        source = Path(source_name)
        is_origin_payload = source.parent.name == "origin-inputs"
        is_control_bundle = source.name == dest == "p1-inputs.tar.gz"
        if (
            source.name != asset["name"]
            or not (is_origin_payload or is_control_bundle)
            or row.get("sha256") != asset["sha256"]
            or not source.is_file()
            or source.stat().st_size != asset["size_bytes"]
            or digest(source) != asset["sha256"]
        ):
            raise ValueError("selected release input differs from its pinned source bytes")
        rows.append(
            {
                "url": f"{RELEASE_BASE_URL}/{quote(asset['name'], safe='')}",
                "sha256": asset["sha256"],
                "size_bytes": asset["size_bytes"],
                "dest": dest,
            }
        )
        converted.add(dest)
    if converted != set(selected):
        raise ValueError("selected release asset has no matching origin input destination")
    result["inputs"] = rows
    return result


def public_input_preflight(*, manifest_path: Path, output: Path) -> dict[str, Any]:
    """Create a receipt for the broker's pre-allocation public-input fetch check.

    This is deliberately an operator-only post-publication command.  It calls
    the qualified supervisor's Mac-side check, creates evidence once, and
    never allocates a runtime or changes a release.
    """
    from research.colab_cli.supervisor import input_preflight

    manifest_path, output = manifest_path.resolve(), output.resolve()
    if not manifest_path.is_file():
        raise ValueError("public-input preflight manifest does not exist")
    if output.exists():
        raise FileExistsError("public-input preflight receipt must be fresh")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("public-input preflight manifest must be an object")
    rows = [row for row in manifest.get("inputs", []) if isinstance(row, dict) and "url" in row]
    if not rows:
        raise ValueError("public-input preflight manifest has no HTTPS rows")
    started = time.monotonic()
    verified = input_preflight(manifest)
    elapsed = time.monotonic() - started
    if len(verified) != len(rows):
        raise ValueError("public-input preflight did not return every manifest HTTPS row")
    total_bytes = sum(row["size_bytes"] for row in verified)
    receipt = {
        "schema": "bl-p1-repair-public-input-preflight/v1",
        "status": "PASS",
        "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        "inputs": verified,
        "total_bytes": total_bytes,
        "elapsed_seconds": elapsed,
        "bytes_per_second": total_bytes / max(elapsed, 1e-9),
        "provider_allocations": 0,
        "provider_calls": 0,
    }
    export_json(output, receipt)
    return receipt


def main() -> None:
    """Run the non-allocating post-publication preflight receipt command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight", help="verify manifest HTTPS rows before allocation"
    )
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(public_input_preflight(manifest_path=args.manifest, output=args.output)))


if __name__ == "__main__":
    main()
