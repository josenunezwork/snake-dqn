"""Convert ruled P1 origin inputs to the qualified public HTTPS transport.

This module only changes how already-pinned bytes enter the qualified Colab
runtime.  It refuses alternate repositories, tags, paths, byte identities, or
destinations, leaving the origin importer's independent digest checks intact.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export

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


TRANSPORT_ADDENDUM_SHA256 = "2fb9e3914a5eb86e9cd5c42495a1d8213fa1475aa6cf0c72d14c83350de5bc5e"
DRIVE_URL = re.compile(
    r"https://drive\.usercontent\.google\.com/download\?id=([A-Za-z0-9_-]{10,})"
    r"&export=download&confirm=t"
)
DRIVE_MANIFESTS = (
    "canary-manifest.json",
    "repair-manifest.json",
    "replacement-held-manifest.json",
    "qualification-held-manifest.json",
)


def drive_rows(manifest: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    """Swap precisely the nine origin URLs, preserving all other manifest values."""
    rows = manifest.get("inputs", [])
    public = [row for row in rows if "url" in row]
    names = {row["dest"] for row in public}
    if len(public) != 9 or len(names) != 9 or set(urls) != names:
        raise ValueError("Drive mapping must name exactly the nine pinned public assets")
    ids = set()
    result = copy.deepcopy(manifest)
    for row in result["inputs"]:
        if "url" not in row:
            continue
        if row["url"] != f"{RELEASE_BASE_URL}/{quote(row['dest'], safe='')}":
            raise ValueError("source pack must retain the ruled GitHub fallback URLs")
        url = urls[row["dest"]]
        match = DRIVE_URL.fullmatch(url) if isinstance(url, str) else None
        if match is None or "PLACEHOLDER" in url or "CLAUDE" in url:
            raise ValueError("Drive URL must use the exact direct-download form and a real file ID")
        if match[1] in ids:
            raise ValueError("Drive assets require distinct file IDs")
        ids.add(match[1])
        row["url"] = url
    return result


def _attempt_payloads(source: Path, output: Path) -> tuple[dict, dict, bytes, list]:
    """Derive only the existing root-scoped attempt identity and bundle closure."""
    import gzip
    import hashlib
    import io
    import tarfile

    bundle = source / "p1-inputs.tar.gz"
    verify_export(bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        members = {item.name: archive.extractfile(item).read() for item in archive.getmembers()}
    old_members = dict(members)
    tag = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    namespaces = {}
    for control_name, ledger_name, namespace in (
        ("p1-control.json", "verified-ledger.json", f"bl_p1_y3_pack04_repair06_{tag}"),
        ("canary-control.json", "canary-verified-ledger.json", f"bl_p1_canary_repair06_{tag}"),
    ):
        control = json.loads(members[control_name])
        ledger = json.loads(members[ledger_name])
        old_namespace = control["resume"]["namespace"]
        if ledger["namespace"] != old_namespace or ledger["units"] != []:
            raise ValueError("source attempt requires matching empty ledgers")
        if ledger["allocation"] != control["resume"]["allocation"]:
            raise ValueError("source allocation differs from its ledger")
        if old_namespace == namespace:
            raise ValueError("Drive pack requires a new attempt-scoped namespace")
        namespaces[old_namespace] = namespace
        control["resume"]["namespace"] = namespace
        ledger["namespace"] = namespace
        for name, value in ((control_name, control), (ledger_name, ledger)):
            members[name] = (json.dumps(value, sort_keys=True) + "\n").encode()
            members[name + ".sha256"] = (
                f"{hashlib.sha256(members[name]).hexdigest()}  {name}\n".encode()
            )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, body in sorted(members.items()):
                header = tarfile.TarInfo(name)
                header.size = len(body)
                archive.addfile(header, io.BytesIO(body))
    payload = buffer.getvalue()
    templates = {}
    for name in DRIVE_MANIFESTS:
        path = source / name
        verify_export(path)
        value = json.loads(path.read_text())
        if value.get("scientific_admission") is not False:
            raise ValueError("source manifests must remain held")
        if Path(value["output_dir"]).exists():
            raise ValueError("source attempt already has outputs; return to judge")
        old_tag = value["job_id"].rsplit("-", 1)[1]
        value["job_id"] = value["job_id"].removesuffix(old_tag) + tag
        value["output_dir"] = str(output / Path(value["output_dir"]).name)
        value["resume"]["namespace"] = namespaces[value["resume"]["namespace"]]
        for row in value["inputs"]:
            if row["dest"] == "p1-inputs.tar.gz":
                row["sha256"] = hashlib.sha256(payload).hexdigest()
                row["size_bytes"] = len(payload)
        templates[name] = value
    changed_members = [
        {
            "name": name,
            "before_sha256": hashlib.sha256(old_members[name]).hexdigest(),
            "after_sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in sorted(members.items())
        if old_members[name] != body
    ]
    return templates, members, payload, changed_members


def _manifest_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """Enumerate exact value differences, including the identity-only exception."""
    if isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys():
        return [
            entry
            for key in before
            for entry in _manifest_diff(before[key], after[key], f"{path}/{key}")
        ]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [
            entry
            for i, (a, b) in enumerate(zip(before, after))
            for entry in _manifest_diff(a, b, f"{path}/{i}")
        ]
    return [] if before == after else [{"field": path, "before": before, "after": after}]


def drive_forecast(source: Path, repo: Path) -> dict[str, Any]:
    """Recompute residual staging from the unchanged local inputs and source pin."""
    from research.bl_upload import qualification as q

    manifest = json.loads((source / "canary-manifest.json").read_text())
    inputs = q._input_rows(manifest, repo)
    bootstrap = q._pinned_bootstrap_bytes(repo, manifest["commit"])
    prior = json.loads((source / "upload-preflight.json").read_text())
    verify_export(source / "upload-preflight.json")
    plan, fallback_bytes, plan_sha = q._source_plan(
        repo, manifest, q._pinned_source_module(repo, manifest["commit"])
    )
    # The pinned gzip fallback has a current-time header; compare every stable
    # source-file hash and the byte size rather than that volatile archive hash.
    stable_plan = {k: v for k, v in plan.items() if k != "fallback_sha256"}
    stable_prior = {k: v for k, v in prior["source_plan"].items() if k != "fallback_sha256"}
    if prior["commit"] != manifest["commit"] or stable_plan != stable_prior:
        raise ValueError("pinned source fallback plan changed")
    total = sum(row["bytes"] for row in inputs + bootstrap)
    seconds = q._forecast(total + fallback_bytes, len(inputs) + 1)
    if seconds > 150:
        raise ValueError("residual upload forecast exceeds 150 seconds")
    forecast = json.loads((source / "measured-forecast-03.json").read_text())
    verify_export(source / "measured-forecast-03.json")
    forecast.update(
        transport_ruling="JUDGMENT_P1_REPAIR_TRANSPORT.md Addendum A",
        transport_ruling_sha256=TRANSPORT_ADDENDUM_SHA256,
        status="HOLD_OWNER_DRIVE_EXPOSURE_AND_MAC_PREFLIGHT",
        exact_upload_bytes=total,
        fallback_upload_bytes=total + fallback_bytes,
        residual_upload_files=len(inputs),
        upload_forecast_seconds=seconds,
        public_fetch_measured_seconds=None,
        public_fetch_measured_bytes_per_second=None,
        source_plan_sha256=plan_sha,
        fill_after_publication=(
            "Create public-fetch-preflight-drive-04.json; never edit this receipt"
        ),
    )
    forecast["forecast_minutes"] = (
        200 + 300 + 150 + 960 + forecast["download_seconds_median"]
    ) / 60
    forecast["twice_forecast_minutes"] = 2 * forecast["forecast_minutes"]
    if forecast["twice_forecast_minutes"] > 200:
        raise ValueError("twice forecast exceeds the frozen 200-minute ceiling")
    return forecast


def prepare_drive_pack(*, source: Path, output: Path, repo: Path) -> dict[str, Any]:
    """Freeze attempt-specific bytes and templates before Drive IDs exist."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or output.is_relative_to(source):
        raise FileExistsError("Drive preparation requires a fresh sibling pack")
    templates, members, bundle, changed = _attempt_payloads(source, output)
    plan_path = source / "release-plan.json"
    verify_export(plan_path)
    assets = _asset_map(json.loads(plan_path.read_text())["assets"])
    if len(assets) != 9:
        raise ValueError("requires exactly nine assets")
    for asset in assets.values():
        path = source / "release-assets" / asset["name"]
        if path.stat().st_size != asset["size_bytes"] or digest(path) != asset["sha256"]:
            raise ValueError("source release asset differs from pinned size/SHA")
    forecast = drive_forecast(source, repo)
    output.mkdir(parents=True)
    for name, value in templates.items():
        export_json(output / "templates" / name, value)
    for name, body in members.items():
        # Members already contain their own original/recomputed completion sidecars.
        path = output / "inputs" / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as handle:
            handle.write(body)
    export_bytes(output / "p1-inputs.tar.gz", bundle)
    export_bytes(output / "release-assets" / "p1-inputs.tar.gz", bundle)
    bundle_row = next(
        row
        for row in templates["canary-manifest.json"]["inputs"]
        if row["dest"] == "p1-inputs.tar.gz"
    )
    assets["p1-inputs.tar.gz"].update(
        sha256=bundle_row["sha256"], size_bytes=bundle_row["size_bytes"]
    )
    for asset in assets.values():
        asset["source_path"] = str(
            (output if asset["name"] == "p1-inputs.tar.gz" else source)
            / "release-assets"
            / asset["name"]
        )
    export_json(output / "drive-urls.template.json", {name: None for name in sorted(assets)})
    export_json(
        output / "drive-asset-plan.json",
        {
            "assets": list(assets.values()),
            "total_asset_bytes": sum(a["size_bytes"] for a in assets.values()),
            "transport_ruling_sha256": TRANSPORT_ADDENDUM_SHA256,
        },
    )
    export_json(output / "measured-forecast-04.json", forecast)
    receipt = {
        "schema": "bl-p1-drive-attempt-preparation/v1",
        "status": "PREPARED_HELD_AWAITING_CLAUDE_URL_JSON",
        "source": str(source),
        "output": str(output),
        "transport_ruling_sha256": TRANSPORT_ADDENDUM_SHA256,
        "source_manifest_sha256": {name: digest(source / name) for name in DRIVE_MANIFESTS},
        "template_sha256": {name: digest(output / "templates" / name) for name in DRIVE_MANIFESTS},
        "bundle_members_changed": changed,
        "bundle": {
            "before_sha256": digest(source / "p1-inputs.tar.gz"),
            "after_sha256": digest(output / "p1-inputs.tar.gz"),
            "size_bytes": len(bundle),
            "regenerated": True,
        },
        "manifest_identity_changes": {
            name: _manifest_diff(json.loads((source / name).read_text()), value)
            for name, value in templates.items()
        },
        "attempt_rule": "New root-derived namespace/job/output; allocation and pin unchanged",
        "unchanged_local_dependencies": "Immutable -03 local input paths preserved",
        "provider_calls": 0,
    }
    export_json(output / "attempt-identity-03-to-04.json", receipt)
    return receipt


def build_drive_pack(*, source: Path, urls_path: Path, output: Path) -> dict[str, Any]:
    """Finalize the prepared -04 exactly once from Claude's asset-to-URL JSON."""
    import hashlib

    source, output = source.resolve(), output.resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Drive pack must be a sibling of the immutable fallback")
    urls = json.loads(urls_path.read_text())
    if not isinstance(urls, dict):
        raise ValueError("Drive URLs JSON must be an asset-to-URL object")
    receipt_path = output / "attempt-identity-03-to-04.json"
    verify_export(receipt_path)
    preparation = json.loads(receipt_path.read_text())
    expected, _, bundle, _ = _attempt_payloads(source, output)
    if preparation["source"] != str(source) or preparation["output"] != str(output):
        raise ValueError("preparation must bind this exact source/output pair")
    if hashlib.sha256(bundle).hexdigest() != verify_export(output / "p1-inputs.tar.gz"):
        raise ValueError("prepared attempt bundle changed")
    for name in [*DRIVE_MANIFESTS, "byte-diff-03-to-04.json"]:
        if any((output / (name + suffix)).exists() for suffix in ("", ".sha256", ".complete")):
            raise FileExistsError("Drive pack already contains immutable output")
    payloads, changes = {}, {}
    for name in DRIVE_MANIFESTS:
        path = output / "templates" / name
        verify_export(path)
        if digest(source / name) != preparation["source_manifest_sha256"][name]:
            raise ValueError("immutable fallback manifest changed")
        body = path.read_bytes()
        manifest = json.loads(body)
        if manifest != expected[name] or digest(path) != preparation["template_sha256"][name]:
            raise ValueError("prepared manifest differs from the identity-only derivation")
        if Path(manifest["output_dir"]).exists():
            raise ValueError("attempt output is not fresh")
        updated = drive_rows(manifest, urls)
        for row in manifest["inputs"]:
            if "url" in row:
                old = json.dumps(row["url"]).encode()
                if body.count(old) != 1:
                    raise ValueError("URL must occur exactly once for byte-only substitution")
                body = body.replace(old, json.dumps(urls[row["dest"]]).encode(), 1)
        if json.loads(body) != updated:
            raise ValueError("unexpected non-URL manifest change")
        payloads[name] = body
        changes[name] = _manifest_diff(json.loads((source / name).read_text()), updated)
    receipt = {
        "schema": "bl-p1-drive-url-byte-diff/v1",
        "status": "URL_SWAP_PLUS_REQUIRED_ATTEMPT_IDENTITY_NOT_ADMITTED",
        "transport_ruling_sha256": TRANSPORT_ADDENDUM_SHA256,
        "source": str(source),
        "output": str(output),
        "urls_json_sha256": digest(urls_path),
        "preparation_sha256": digest(receipt_path),
        "source_manifest_sha256": preparation["source_manifest_sha256"],
        "manifest_sha256": {
            name: hashlib.sha256(body).hexdigest() for name, body in payloads.items()
        },
        "changes": changes,
        "url_changes_per_manifest": 9,
        "non_url_changes": "Exactly attempt-identity-03-to-04.json; no other changes",
        "bundle": preparation["bundle"],
        "source_pack_modified": False,
        "provider_calls": 0,
    }
    for name, body in payloads.items():
        export_bytes(output / name, body)
    export_json(output / "byte-diff-03-to-04.json", receipt)
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
    build = commands.add_parser("build-drive", help="freeze URL-only -04 from Claude's JSON")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--urls", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    prepare = commands.add_parser(
        "prepare-drive", help="prepare attempt bytes and URL placeholders"
    )
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(public_input_preflight(manifest_path=args.manifest, output=args.output)))
    elif args.command == "prepare-drive":
        receipt = prepare_drive_pack(source=args.source, output=args.output, repo=args.repo)
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    elif args.command == "build-drive":
        receipt = build_drive_pack(source=args.source, urls_path=args.urls, output=args.output)
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
