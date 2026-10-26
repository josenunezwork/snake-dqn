"""Metadata/publication and exact-allowlist gates for the F4 exception only."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_residual/f4_postmortem_20261025"
RULING_SHA = "8c4480a5e80f63ec500abd7baa6062adc757866937ce560291cbbaab68b5eeec"
OUTPUT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/"
    "postmortem-harm-20261025-01"
)
RA_LEDGER = "research/bl_residual/ledger/F4_R_A_20261025"


def git(*args: str, root: Path = ROOT) -> bytes:
    """Bound each Git operation and never expose its captured error payload."""
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, timeout=60, check=False
    )
    if completed.returncode:
        raise ValueError("post-mortem Git publication check failed")
    return completed.stdout


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity(
    path: Path, expected: dict[str, Any], *, sealed: bool = True, sidecar: bool = True
) -> dict[str, Any]:
    """Verify path, size, SHA and completion before any deserialization."""
    if str(path) != expected.get("path", str(path)):
        raise ValueError("foreign input path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected["size_bytes"]:
        raise ValueError("unsafe input or size mismatch")
    observed = verify_export(path) if sealed else digest(path)
    if observed != expected["sha256"]:
        raise ValueError("input hash differs from ruling/manifest")
    if not sealed and sidecar:
        sidecar = Path(str(path) + ".sha256")
        if sidecar.is_symlink() or sidecar.read_text().split() != [observed, path.name]:
            raise ValueError("metadata/ruling SHA sidecar mismatch")
    return {"path": str(path), "size": info.st_size, "sha256": observed}


def read_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Open only prepublished implementation metadata, never a real payload."""
    path = root / PACKAGE / "manifest.json"
    verify_export(path)
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema") != "bl-residual-f4-harm-postmortem-manifest/v1"
        or manifest.get("ruling", {}).get("sha256") != RULING_SHA
        or manifest.get("output_root") != str(OUTPUT)
        or manifest.get("source_pin") != "f9c08082b4e298a3d6666e7820e49c7973a16b99"
        or manifest.get("census")
        != dict(total_rows=5936, natural_rows=560, eligible_lives=71, reserve_lives=64, parents=29)
    ):
        raise ValueError("foreign post-mortem manifest")
    ruling = root / manifest["ruling"]["path"]
    if verify_export(ruling) != RULING_SHA:
        raise ValueError("ruling differs from ratified exception")
    return manifest


def provenance(manifest: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Reconcile published packet/fit/census metadata; no model/table access."""
    ledgers: dict[str, Any] = {}
    for name, record in manifest["allowed_metadata_provenance"].items():
        identity(root / name, record, sealed=not name.endswith(".md"))
        if name.endswith(".json"):
            ledgers[name] = json.loads((root / name).read_bytes())
    packet = ledgers[f"{RA_LEDGER}/command-packet.json"]
    inputs = manifest["allowed_payload_inputs"]
    if set(inputs) != {
        "--approval",
        "--archive-receipt",
        "--b8",
        *(
            f"--d{dose}-{part}"
            for dose in (256, 1024)
            for part in ("allocation", "checkpoint", "control", "table")
        ),
    } or any(
        {k: record[k] for k in ("path", "sha256")} != packet["input_identities"].get(option)
        for option, record in inputs.items()
    ):
        raise ValueError("allowlist differs from the ruling-bound R-A packet")
    if (
        inputs["--d256-table"]["path"] != inputs["--d1024-table"]["path"]
        or inputs["--d256-table"]["sha256"] != inputs["--d1024-table"]["sha256"]
        or manifest["immutable_joint_claim"]["path"] != packet["lock_path"]
    ):
        raise ValueError("shared table/original claim substitution")
    for dose, phase in (
        (256, "F4_D256_FIT_PHASE_20261023.json"),
        (1024, "F4_D1024_FIT_PHASE_20261024.json"),
    ):
        units = ledgers[f"research/bl_residual/ledger/{phase}"]["units"]
        record = inputs[f"--d{dose}-checkpoint"]
        if (
            len(units) != 1
            or any(
                units[0][key] != record[other]
                for key, other in (("path", "path"), ("size", "size_bytes"), ("sha256", "sha256"))
            )
            or units[0]["index"] != (0 if dose == 256 else 1)
        ):
            raise ValueError("checkpoint differs from published fit-phase ledger")
    excluded_path = root / manifest["development_exclusion_ledger"]
    verify_export(excluded_path)
    excluded = json.loads(excluded_path.read_bytes())
    census = ledgers["research/bl_residual/ledger/F4_R_A_ARCHIVE_CENSUS_20261024.json"]
    if (
        census["counts"]["rows"] != 5936
        or census["counts"]["natural_rows"] != 560
        or census["counts"]["eligible_lives"] != 71
        or census["counts"]["parents"] != 29
        or len(packet["lock"]["reserve"]["reserve_life_ids"]) != 64
        or excluded["parent_worlds"] != sorted(set(census["parent_worlds"]))
        or len(excluded["parent_worlds"]) != 29
        or excluded["designation"] != "DEVELOPMENT_FOREVER"
        or excluded["ruling_sha256"] != RULING_SHA
        or census["table_sha256"] != inputs["--d256-table"]["sha256"]
    ):
        raise ValueError("census/permanent exclusion mismatch")
    funding = ledgers[f"{RA_LEDGER}/funding-look.json"]
    if funding["verdict"] != "STOP_FUNDING" or funding["harm_screen_d1024_pass"] is not False:
        raise ValueError("foreign retained funding disposition")
    return {"parents": excluded, "census": census, "harm_pass": False, "packet": packet}


def publication(pin: str, root: Path = ROOT) -> dict[str, Any]:
    """Fresh origin/main byte proof before the once-only claim or payload access."""
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("a full published implementation commit is required")
    git("fetch", "--no-tags", "origin", "main", root=root)
    if git("rev-parse", "HEAD", root=root).decode().strip() != pin:
        raise ValueError("executing HEAD differs from publication pin")
    git("merge-base", "--is-ancestor", pin, "origin/main", root=root)
    if git("status", "--porcelain", "--untracked-files=all", root=root).strip():
        raise ValueError("post-mortem source worktree must be clean")
    manifest = read_manifest(root)
    names = (
        git(
            "ls-files",
            "research/bl_residual",
            "research/bl_ppo/exports.py",
            "research/mac_lease.py",
            "research/compute/thermal_guard.py",
            "src/model",
            "tools/ops/dispatch.sh",
            "tools/ops/mac_run.sh",
            "tests/test_bl_residual_postmortem*",
            root=root,
        )
        .decode()
        .splitlines()
    )
    # Check all named publication metadata and relevant executable source bytes.
    source = {}
    for name in names:
        if name.endswith(".py") or name.endswith(".sh") or name.startswith(PACKAGE + "/"):
            payload = (root / name).read_bytes()
            if payload != git("show", f"{pin}:{name}", root=root):
                raise ValueError("local implementation differs from published pin")
            source[name] = _sha(payload)
    for name, expected_sha in manifest["source_sha256"].items():
        reference = git("show", f"{manifest['source_pin']}:{name}", root=root)
        if _sha(reference) != expected_sha:
            raise ValueError("reference harm code differs from ruling")
        if name not in ("research/bl_residual/fit.py", "research/bl_residual/heldout.py"):
            if (root / name).read_bytes() != reference:
                raise ValueError("frozen computation source changed")
    for name, expected_sha in manifest["implementation_source_sha256"].items():
        if source.get(name) != expected_sha:
            raise ValueError("implementation source differs from the frozen manifest")
    evidence = root / PACKAGE / "synthetic-verification.json"
    verify_export(evidence)
    if json.loads(evidence.read_bytes()).get("status") != "PASS_SYNTHETIC_ONLY":
        raise ValueError("synthetic verification not published")
    metadata = provenance(manifest, root)
    return {
        "implementation_commit": pin,
        "origin_main": git("rev-parse", "origin/main", root=root).decode().strip(),
        "source_sha256": source,
        "manifest": manifest,
        "metadata": metadata,
    }


def verify_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    """After publication AND the spent claim, verify only the fixed allowlist."""
    verified = {}
    deduplicated = {}
    for name, record in {
        **manifest["allowed_payload_inputs"],
        "--original-joint-claim": manifest["immutable_joint_claim"],
    }.items():
        path = Path(record["path"])
        if path not in deduplicated:
            if name in ("--b8", "--approval"):
                # Legacy raw copies have no adjacent producer seals. Their
                # detached identity seals are frozen and published with the
                # manifest; they do not claim original producer completion.
                seal = record["detached_identity_seal"]
                checksum = ROOT / seal["sha256_path"]
                completion = ROOT / seal["complete_path"]
                if (
                    checksum.is_symlink()
                    or completion.is_symlink()
                    or checksum.read_text().split() != [record["sha256"], path.name]
                    or completion.read_bytes() != b"complete\n"
                ):
                    raise ValueError("detached input identity seal mismatch")
                deduplicated[path] = identity(path, record, sealed=False, sidecar=False)
            else:
                deduplicated[path] = identity(path, record)
        if any(
            deduplicated[path][k] != record[other]
            for k, other in (("size", "size_bytes"), ("sha256", "sha256"))
        ):
            raise ValueError("conflicting aliases for one immutable input")
        verified[name] = deduplicated[path]
    for record in manifest["allowed_external_metadata_receipts"].values():
        # Receipts/manifests are the only extra external files permitted by the ruling.
        identity(
            Path(record["path"]),
            record,
            sealed=record["seal_mode"] == "export",
            sidecar=False,
        )
    return verified


def require_independent_ancestry(original_parents: list[str | None], root: Path = ROOT) -> None:
    """Future audits must supply resolved original roots, including every ancestor.

    Namespace/index/life names are never an independence certificate. An auditor
    unable to resolve an original world must pass None and fail closed.
    """
    manifest = read_manifest(root)
    burned = provenance(manifest, root)["parents"]["parent_worlds"]
    if not original_parents or any(not parent or parent in burned for parent in original_parents):
        raise ValueError("exposed or unresolved original ancestry is development data")


def validate_original_claim(original: dict[str, Any], packet: dict[str, Any]) -> None:
    """The old claim is provenance, never fresh permission or a replaceable claim."""
    if any(
        original.get(key) != packet["lock"].get(key)
        for key in ("schema", "lock_root", "reserve", "d256", "d1024")
    ):
        raise ValueError("original R-A claim differs from its published packet")
