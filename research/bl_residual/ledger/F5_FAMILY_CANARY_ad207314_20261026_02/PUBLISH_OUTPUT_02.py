"""Publish the F5 family-canary output as create-only operational evidence.

This publisher intentionally reads only the saved ``output-02`` directory.  It
does not read or publish the supervisor state directory, account records, or
any provider credential/lease material.  The result is an operational
INCOMPLETE record, never scientific or timing-qualification evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "f5-timing-recovery-offline-20261026/canary-02"
)
ACCEPTANCE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "astra-20261024/ACCEPT_F5_FAMILY_CANARY_02.md"
)
DESTINATION = Path(__file__).resolve().parent
OUTPUT_NAME = "output-02"
FORBIDDEN_KEY_PARTS = (
    "access_token",
    "refresh_token",
    "client_secret",
    "api_key",
    "authorization",
    "cookie",
    "bearer",
    "lease_token",
    "lease_private",
    "lease_key",
)


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    """Encode one canonical publication JSON document."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def create(path: Path, raw: bytes) -> None:
    """Write one new file, refusing any overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def create_or_verify(path: Path, raw: bytes) -> None:
    """Create exact bytes or verify a prior interrupted publication wrote them."""
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f"existing publication differs: {path}")
        return
    create(path, raw)


def publish(path: Path, raw: bytes) -> None:
    """Publish exact bytes and sidecars, verifying interrupted prior writes."""
    create_or_verify(path, raw)
    create_or_verify(Path(f"{path}.sha256"), f"{sha256(raw)}  {path.name}\n".encode())
    create_or_verify(Path(f"{path}.complete"), b"complete\n")


def key_paths(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    """Yield JSON object-key paths without returning scalar values."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            current = (*prefix, key_text)
            yield current
            yield from key_paths(child, current)
    elif isinstance(value, list):
        for child in value:
            yield from key_paths(child, prefix)


def reject_private_json_keys(source: Path, raw: bytes) -> None:
    """Refuse a saved output that names known credential or private lease fields."""
    if source.suffix != ".json":
        return
    value = json.loads(raw)
    lowered_paths = [".".join(part.lower() for part in parts) for parts in key_paths(value)]
    forbidden = sorted(
        path
        for path in lowered_paths
        if any(part in path.replace("-", "_") for part in FORBIDDEN_KEY_PARTS)
    )
    if forbidden:
        raise ValueError(f"credential or private lease field(s) in {source.name}: {forbidden}")


def source_entries(source: Path) -> list[Path]:
    """Return the exact regular-file output set, refusing path surprises."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source output directory must be a real directory")
    entries = list(source.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("source output must not contain symlinks")
    files = sorted(path for path in entries if path.is_file())
    if len(files) + sum(path.is_dir() for path in entries) != len(entries):
        raise ValueError("source output directory must contain only regular files and directories")
    if not files:
        raise ValueError("source output directory is empty")
    return files


def main() -> None:
    """Copy exact saved output bytes and publish an INCOMPLETE operational index."""
    source = ARTIFACT_ROOT / OUTPUT_NAME
    receipt_path = source / "receipt.json"
    terminal_path = source / "terminal.json"
    if not receipt_path.is_file() or not terminal_path.is_file():
        raise ValueError("receipt.json and terminal.json are required saved output evidence")
    if (DESTINATION / "OUTPUT_PUBLICATION_02.json").exists():
        raise FileExistsError("output publication already exists")

    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("status") != "INCOMPLETE":
        raise ValueError("publication is reserved for the recorded INCOMPLETE refusal")
    if receipt.get("stop_verified") is not True:
        raise ValueError("recorded stop must be verified")
    if receipt.get("exports") != [{"path": "environment/cross-assignment-refusal.json", "sha256": "ae157f89cce650d73bab192acb324987f97a50cc29fed44ee83bc8a09faf48b9", "size": 13420754}]:
        raise ValueError("recorded refusal must have exactly the pinned non-admissible diagnostic")
    if receipt.get("cleanup_error") is not None:
        raise ValueError("recorded refusal cleanup must be clean")

    entries: list[dict[str, object]] = []
    for item in source_entries(source):
        raw = item.read_bytes()
        relative_path = item.relative_to(source)
        # The staged source closure is retained as exact opaque output evidence.
        # Do not parse its archive or source-plan artifact references as live
        # runtime state; the credential boundary is the supervisor's saved state
        # and account directories, both of which are outside ``output-02``.
        if "staging" not in relative_path.parts and len(raw) <= 2_000_000:
            reject_private_json_keys(item, raw)
        destination = DESTINATION / OUTPUT_NAME / relative_path
        if len(raw) <= 2_000_000:
            publish(destination, raw)
        entries.append(
            {
                "relative_path": str(relative_path),
                "source_path": str(item),
                "size": len(raw),
                "sha256": sha256(raw),
                "published_relative_path": str(destination.relative_to(DESTINATION)) if len(raw) <= 2_000_000 else None,
                "publication": "EXACT_BYTES" if len(raw) <= 2_000_000 else "EXTERNAL_HASH_BOUND_ARTIFACT",
            }
        )

    acceptance_raw = ACCEPTANCE.read_bytes()
    publish(DESTINATION / ACCEPTANCE.name, acceptance_raw)
    publisher = Path(__file__)
    # The publisher itself is already the create-only repository source.  Add
    # only its verification sidecars; a second copy would be an overwrite.
    publisher_raw = publisher.read_bytes()
    create_or_verify(
        Path(f"{publisher}.sha256"), f"{sha256(publisher_raw)}  {publisher.name}\n".encode()
    )
    create_or_verify(Path(f"{publisher}.complete"), b"complete\n")

    prior_packet = DESTINATION / "ARTIFACT_PACKET.json"
    if not prior_packet.is_file():
        raise ValueError("sealed preparation packet is required")
    report = {
        "schema": "bl-f5-family-canary-output-publication/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "publication_scope": "SAVED_OUTPUT_02_OPERATIONAL_REFUSAL_ONLY",
        "admissibility": {
            "scientific_evidence": False,
            "timing_qualification_evidence": False,
            "reason": "INCOMPLETE v5 cross-assignment environment refusal; one non-admissible diagnostic, zero timing/scientific units",
        },
        "source": {
            "artifact_root": str(ARTIFACT_ROOT),
            "output_root": str(source),
            "runtime_source_commit": receipt.get("commit"),
            "sealed_preparation_packet": {
                "published_relative_path": str(prior_packet.relative_to(DESTINATION)),
                "sha256": sha256(prior_packet.read_bytes()),
                "size": prior_packet.stat().st_size,
            },
        },
        "acceptance": {
            "external_path": str(ACCEPTANCE),
            "published_relative_path": ACCEPTANCE.name,
            "sha256": sha256(acceptance_raw),
            "size": len(acceptance_raw),
        },
        "recorded_terminal_state": {
            "job_id": receipt.get("job_id"),
            "status": receipt.get("status"),
            "stop_verified": receipt.get("stop_verified"),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
            "cleanup_error": receipt.get("cleanup_error"),
            "export_count": len(receipt["exports"]),
            "failure_classification": "F5 cross-assignment invariant environment differs",
        },
        "credential_and_state_exclusion": {
            "copied_source_scope": str(source),
            "not_copied": [
                str(ARTIFACT_ROOT / "state-02"),
                str(ARTIFACT_ROOT / "accounts"),
                "any runtime token, credential, or private lease state",
            ],
            "json_key_scan": (
                "refuses known credential and private lease key names in top-level operational JSON; "
                "external diagnostic is hash-bound only; staging source closure, opaque archive, and hashed artifact-path references are retained "
                "without parsing"
            ),
        },
        "all_original_files": len(entries),
        "entries": entries,
        "provider_calls": 0,
        "dispatches": 0,
        "payloads_deserialized": False,
        "opaque_bytes_only": True,
    }
    publish(DESTINATION / "OUTPUT_PUBLICATION_02.json", encode(report))
    print(
        json.dumps(
            {
                "output_files": len(entries),
                "receipt_sha256": sha256(receipt_path.read_bytes()),
                "status": receipt["status"],
                "stop_verified": receipt["stop_verified"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
