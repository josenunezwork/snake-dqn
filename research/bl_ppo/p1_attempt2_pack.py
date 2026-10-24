"""Create a fresh attempt-2 pack from the consumed, immutable -04 evidence.

This is a Mac-only preparer. Publication, final fetch proof and operator admission
are separate steps. It never changes the frozen executable or dispatches work.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_ppo.p1_repair_transport import DRIVE_URL

PIN = "03e0af75c64538f53a6971e6fec5d9a173bc921b"
TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"
ATTEMPT2_RULING = "55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13"
RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_P1_REPAIR_ATTEMPT2_20261023.md"
)
FINALISATION = "fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc"
FAILURE_RECEIPT = "b675b84e98d155251f43d1b44d00640240dbf2769cde2f37465edf6dc08b66b1"
FAILURE_REVIEW = "b40f8abb4d9a6b942da17680778811dd3ecd1e3eb7a9a5a482e7c529c36392d5"
MANIFESTS = ("canary-manifest.json", "repair-manifest.json")
SOURCE_MANIFEST_SHA256 = {
    "canary-manifest.json": "fa398a8eb52009da34c8c05cc69a913c71eeaa2b38dd4a0573cf37d0b79c726e",
    "repair-manifest.json": "233a77e0e3305fbfd946f3f7aeda9c44f7e40f7f4957112196d245cd38c81f61",
}


def manifest_diff(before: Any, after: Any, path: str = "") -> list[dict]:
    """Enumerate leaf changes, including added operational metadata."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                changes.append(
                    {"field": f"{path}/{key}", "before": before.get(key), "after": after.get(key)}
                )
            else:
                changes.extend(manifest_diff(before[key], after[key], f"{path}/{key}"))
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [
            change
            for i, (a, b) in enumerate(zip(before, after))
            for change in manifest_diff(a, b, f"{path}/{i}")
        ]
    return [] if before == after else [{"field": path, "before": before, "after": after}]


def read_export(path: Path) -> dict[str, Any]:
    """Read only a complete immutable object."""
    verify_export(path)
    return json.loads(path.read_text())


def attempt_payloads(source: Path, output: Path) -> tuple[dict, bytes, dict, list]:
    """Derive new controls/empty ledgers without pretending -04 is unconsumed."""
    if digest(RULING) != ATTEMPT2_RULING:
        raise ValueError("attempt-2 ruling changed")
    if digest(source / "repair-output/receipt.json") != FAILURE_RECEIPT:
        raise ValueError("not the ruled consumed -04 repair receipt")
    if verify_export(source / "P1_REPAIR_FAILURE_CODEX_V_20261023.json") != FAILURE_REVIEW:
        raise ValueError("not the independently verified -04 failure")
    failed = json.loads((source / "repair-output/receipt.json").read_text())
    if (
        failed.get("status") != "INCOMPLETE"
        or failed.get("failure_class") != "UPLOAD_CAP"
        or failed.get("stop_verified") is not True
        or failed.get("cleanup_error") is not None
        or failed.get("terminal") is not None
        or failed.get("exports") != []
    ):
        raise ValueError("consumed attempt lacks the ruled failure and verified release")
    old_bundle_sha = verify_export(source / "p1-inputs.tar.gz")
    for name in MANIFESTS:
        source_value = read_export(source / name)
        if digest(source / name) != SOURCE_MANIFEST_SHA256[name]:
            raise ValueError("source manifest differs from the independently reviewed -04 bytes")
        bundles = [r for r in source_value["inputs"] if r["dest"] == "p1-inputs.tar.gz"]
        if (
            len(bundles) != 1
            or bundles[0]["sha256"] != old_bundle_sha
            or bundles[0]["size_bytes"] != (source / "p1-inputs.tar.gz").stat().st_size
        ):
            raise ValueError("old bundle differs from the pinned -04 manifest input")
    with tarfile.open(source / "p1-inputs.tar.gz", "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError("unsafe bundle member")
            if member.name in members:
                raise ValueError("duplicate bundle member")
            members[member.name] = archive.extractfile(member).read()
    old_members = dict(members)
    tag = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    namespaces = {}
    for control_name, ledger_name, prefix in (
        ("p1-control.json", "verified-ledger.json", "bl_p1_y3_pack04_repair06_"),
        ("canary-control.json", "canary-verified-ledger.json", "bl_p1_canary_repair06_"),
    ):
        control, ledger = (json.loads(members[n]) for n in (control_name, ledger_name))
        old_ns = control["resume"]["namespace"]
        new_ns = prefix + tag
        if (
            ledger["namespace"] != old_ns
            or ledger["allocation"] != control["resume"]["allocation"]
            or ledger["units"] != []
            or new_ns == old_ns
        ):
            raise ValueError("fresh attempt requires matching empty ledger and new namespace")
        namespaces[old_ns] = new_ns
        control["resume"]["namespace"] = ledger["namespace"] = new_ns
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
    bundle = buffer.getvalue()
    templates = {}
    for name in MANIFESTS:
        value = read_export(source / name)
        if digest(source / name) != SOURCE_MANIFEST_SHA256[name]:
            raise ValueError("source manifest differs from the independently reviewed -04 bytes")
        if value["commit"] != PIN or value["scientific_admission"] is not False:
            raise ValueError("-04 must retain the exact held executable pin")
        value = copy.deepcopy(value)
        value["job_id"] = value["job_id"].rsplit("-", 1)[0] + "-" + tag
        value["output_dir"] = str(output / Path(value["output_dir"]).name)
        if Path(value["output_dir"]).exists():
            raise FileExistsError("new attempt output is already consumed")
        value["resume"]["namespace"] = namespaces[value["resume"]["namespace"]]
        value["resume"]["max_outcome_blind_retries"] = 2
        value["p1_attempt2"] = {
            "ruling_sha256": digest(RULING),
            "qualified_transport_tree": TREE,
            "canary_attempts": 1,
            "repair_attempts": 1,
            "further_repair_retries": 0,
            "canary_retries": 0,
            "required_control_nice": 0,
            "factory_drained_through_verified_release": True,
            "renicer_excludes_complete_control_tree": True,
            "warning": "WARNING_UNEXPLAINED_DEBIT",
        }
        for row in value["inputs"]:
            if row["dest"] == "p1-inputs.tar.gz":
                row["sha256"] = hashlib.sha256(bundle).hexdigest()
                row["size_bytes"] = len(bundle)
                # Never invent a usable URL for unpublished new bytes.
                row["url"] = None
        templates[name] = value
    changed = [name for name in sorted(members) if members[name] != old_members[name]]
    return templates, bundle, members, changed


def prepare(*, source: Path, output: Path) -> dict:
    """Write a fresh sibling; all old outputs remain immutable."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or output.is_relative_to(source):
        raise FileExistsError("attempt-2 needs a fresh sibling directory")
    templates, bundle, members, changed = attempt_payloads(source, output)
    urls = json.loads((source / "drive-urls.json").read_text())
    urls["p1-inputs.tar.gz"] = None
    for name, value in templates.items():
        export_json(output / "templates" / name, value)
    for name, body in members.items():
        path = output / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(body)
    export_bytes(output / "p1-inputs.tar.gz", bundle)
    export_json(output / "drive-urls.template.json", urls)
    receipt = {
        "schema": "bl-p1-attempt2-preparation/v1",
        "status": "HOLD_NEW_BUNDLE_PUBLICATION_AND_EXACT_FETCH_PROOF",
        "source": str(source),
        "output": str(output),
        "executable_commit": PIN,
        "ruling_sha256": FINALISATION,
        "attempt2_ruling_sha256": digest(RULING),
        "source_manifest_sha256": {n: digest(source / n) for n in MANIFESTS},
        "template_sha256": {n: digest(output / "templates" / n) for n in MANIFESTS},
        "bundle_sha256": digest(output / "p1-inputs.tar.gz"),
        "bundle_members_changed": changed,
        "changes": {
            n: manifest_diff(read_export(source / n), value) for n, value in templates.items()
        },
        "unchanged_origin_urls": 8,
        "provider_calls": 0,
    }
    export_json(output / "preparation.json", receipt)
    return receipt


def finalize(*, output: Path, urls_path: Path) -> dict:
    """Freeze only the new bundle URL; preserve all eight selected origin URLs."""
    prep = read_export(output / "preparation.json")
    source = Path(prep["source"])
    expected, bundle, _, _ = attempt_payloads(source, output)
    if hashlib.sha256(bundle).hexdigest() != verify_export(output / "p1-inputs.tar.gz"):
        raise ValueError("prepared bundle changed")
    urls = json.loads(urls_path.read_text())
    prior = json.loads((source / "drive-urls.json").read_text())
    if set(urls) != set(prior) or len(urls) != 9:
        raise ValueError("mapping requires the exact nine assets")
    ids = set()
    for name, url in urls.items():
        match = DRIVE_URL.fullmatch(url) if isinstance(url, str) else None
        if match is None or match[1] in ids:
            raise ValueError("mapping requires nine unique qualified Drive URLs")
        ids.add(match[1])
        if name != "p1-inputs.tar.gz" and url != prior[name]:
            raise ValueError("the eight scientific origin URLs are immutable")
    if urls["p1-inputs.tar.gz"] == prior["p1-inputs.tar.gz"]:
        raise ValueError("new bundle requires a new immutable Drive object")
    for name in MANIFESTS:
        if (
            digest(source / name) != prep["source_manifest_sha256"][name]
            or digest(output / "templates" / name) != prep["template_sha256"][name]
        ):
            raise ValueError("source or prepared template binding changed")
        if read_export(output / "templates" / name) != expected[name]:
            raise ValueError("template differs from deterministic ruled derivation")
        if (output / name).exists():
            raise FileExistsError("final manifest already exists")
    for name, value in expected.items():
        next(row for row in value["inputs"] if row["dest"] == "p1-inputs.tar.gz")["url"] = urls[
            "p1-inputs.tar.gz"
        ]
        export_json(output / name, value)
    receipt = {
        "schema": "bl-p1-attempt2-final-byte-diff/v1",
        "status": "FINAL_HELD_REQUIRES_MAC_PREFLIGHT_CODEX_V_AND_OPERATOR",
        "preparation_sha256": digest(output / "preparation.json"),
        "mapping_sha256": digest(urls_path),
        "manifest_sha256": {n: digest(output / n) for n in MANIFESTS},
        "changes": {
            n: manifest_diff(read_export(source / n), value) for n, value in expected.items()
        },
        "provider_calls": 0,
    }
    export_json(output / "byte-diff-04-to-05.json", receipt)
    return receipt


def main() -> None:
    """Prepare or finalize without provider access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urls", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(source=args.source, output=args.output)
    else:
        result = finalize(output=args.output, urls_path=args.urls)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
