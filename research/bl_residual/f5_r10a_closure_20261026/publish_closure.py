"""Create a sibling HOLD annex from explicit metadata reports, never labels.

The evidence list is a closed list supplied on the command line. This publisher
hashes those review artifacts only; it does not follow paths inside a report.
It cannot issue extractor acceptance, a custodian certificate or an adoption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

NAME = "F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2"
BASE_SHA = "8ee06291b23a1ef8c657b59a4f4990349ae4c524a680c47f06785a47d227f764"
RULING_SHA = "d6e2f029da689bbbe44e6c9c337ad989202a36af1fc26bcadad2f6f3be7e4fa6"
DRAFT_SHA = "6aeda9e66fbe67da41e0abf3549880a16f33bd52b448964b933234bbfb94f34d"


def encoded(value: object) -> bytes:
    """Return canonical metadata bytes."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def checked(path: Path) -> bytes:
    """Read a bounded regular review artifact, without symlinks or protected paths."""
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink artifact refused")
    lower = str(path.absolute()).lower()
    if "4096" in lower or "t34-round0-study-01" in lower:
        raise ValueError("protected path refused")
    if path.suffix not in (".py", ".md", ".json", ".sha256", ".complete"):
        raise ValueError("review artifact type refused")
    if not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError("review artifact bound refused")
    return path.read_bytes()


def create(path: Path, raw: bytes) -> None:
    """Publish atomically without replacing historical evidence."""
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink publication refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".f5-closure-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main() -> None:
    """Bind explicit recovery reports to the unchanged annex and ruling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--publish", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    base = root / "research/bl_residual/f5_r10a_20261025/F5_R10A_METADATA_ANNEX_20261025.json"
    authority = Path(__file__).parent / "JUDGMENT_F5_R10A_GAP_CLOSURE_20261026T003745Z.md"
    for path, expected in ((base, BASE_SHA), (authority, RULING_SHA)):
        if hashlib.sha256(checked(path)).hexdigest() != expected:
            raise ValueError("authority binding mismatch")
    summary = json.loads(checked(args.summary))
    if summary["schema"] != "bl-f5-gap-closure-summary/v1":
        raise ValueError("summary schema refused")
    if summary["disposition"] != "HOLD_BEFORE_EXPOSURE":
        raise ValueError("publisher cannot grant acceptance or adoption")
    evidence = []
    for path in args.evidence:
        raw = checked(path)
        evidence.append(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        )
    if len({row["path"] for row in evidence}) != len(evidence):
        raise ValueError("duplicate evidence")
    annex = {
        "schema": "bl-f5-r10a-gap-closure-annex/v1",
        "original_annex_sha256": BASE_SHA,
        "ruling_sha256": RULING_SHA,
        "draft_sha256": DRAFT_SHA,
        "publisher_sha256": hashlib.sha256(checked(Path(__file__))).hexdigest(),
        "disposition": "HOLD_BEFORE_EXPOSURE",
        "adopted": False,
        "real_extractor_run": False,
        "summary": summary,
        "evidence": evidence,
        "evidence_inventory_sha256": hashlib.sha256(encoded(evidence)).hexdigest(),
        "access_attestation": {
            "label_or_outcome_payloads_opened_or_hashed": False,
            "real_status_extraction": False,
            "d4096_accessed_including_metadata": False,
            "protected_study_accessed": False,
            "postmortem_executed": False,
            "scientific_execution": False,
            "colab_or_dispatch": False,
            "current_environment_used_as_historical_identity": False,
            "original_annex_or_draft_modified": False,
        },
    }
    raw = encoded(annex)
    digest = hashlib.sha256(raw).hexdigest()
    lines = [
        "# F5 R10a sibling gap-closure annex",
        "",
        (
            "**HOLD before exposure.** This is a source-bound preparation record; "
            "no adoption is asserted."
        ),
        "",
        f"Original annex JSON SHA-256: `{BASE_SHA}`",
        "",
        f"Ruling SHA-256: `{RULING_SHA}`",
        "",
        f"F5 v2 SHA-256: `{DRAFT_SHA}`",
        "",
        f"This annex JSON SHA-256: `{digest}`",
        "",
        *summary["human_sections"],
        "",
        "No real label/outcome payload was opened or hashed. No D4096 metadata or protected study",
        "was accessed. No post-mortem, inference, replay, fit, coverage completion, canary, Colab,",
        "dispatch, serving change or release was performed. Original documents remain immutable.",
        "",
        "The machine annex binds each review artifact by exact path, size and SHA-256.",
        "Independent extractor acceptance must precede any real scan; its metadata export must",
        "then be sealed before independent adoption. Missing historical evidence cannot be adopted",
        "as unknown or supplied from the current environment.",
        "",
    ]
    human = "\n".join(lines).encode()
    files = {f"{NAME}.json": raw, f"{NAME}.md": human}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}
    for name, sha in hashes.items():
        files[f"{name}.sha256"] = f"{sha}  {name}\n".encode()
    files[f"{NAME}.complete.json"] = encoded(
        {"schema": "bl-f5-closure-publication/v1", "files": hashes}
    )
    for directory in args.publish:
        if any((directory / name).exists() for name in files):
            raise ValueError("create-only publication already exists")
    for directory in args.publish:
        for name, payload in files.items():
            create(directory / name, payload)
    print(json.dumps(hashes, sort_keys=True))


if __name__ == "__main__":
    main()
