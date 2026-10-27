"""Create-only, lossless publication of preliminary attempt-03 evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from research.bl_ppo.exports import export_bytes, export_json, verify_export

DEST = Path(__file__).resolve().parent
PACKAGE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "sol-20261026/f5-hold-corrections-v5/step3-package"
)
ASTRA = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024")


def main() -> None:
    """Preserve all original files, including original sidecars, without rewriting."""
    output = PACKAGE / "qualification-output-attempt03"
    state = PACKAGE / "qualification-state-attempt03/f5-step3-qualification-v5-attempt03"
    files = [
        ("output/" + str(p.relative_to(output)), p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    files += [("state/" + p.name, p) for p in sorted(state.glob("*.json*")) if p.is_file()]
    files += [
        ("package/" + name, PACKAGE / name)
        for name in (
            "qualification-manifest-attempt03.json",
            "PACKAGE_SEAL.json",
            "envelope.json",
            "source-inventory.json",
            "MACHINE_DIFF_VS_V4.json",
        )
    ]
    files += [
        ("authority/" + name, ASTRA / name)
        for name in (
            "ACCEPT_F5_HOLD_ENGINEERING_V4.md",
            "ACCEPT_F5_HOLD_ENGINEERING_V5.md",
        )
    ]
    assert not (DEST / "PUBLICATION.json").exists()
    entries = []
    for index, (relative, source) in enumerate(files):
        assert source.is_file() and not source.is_symlink(), str(source)
        raw = source.read_bytes()
        encoded = gzip.compress(raw, mtime=0) if len(raw) > 1_500_000 else raw
        encoding = "gzip" if len(raw) > 1_500_000 else "identity"
        parts = []
        for part_index, offset in enumerate(range(0, max(1, len(encoded)), 1_500_000)):
            part = DEST / "payloads" / f"{index:04d}-{part_index:03d}.part"
            export_bytes(part, encoded[offset : offset + 1_500_000])
            verify_export(part)
            parts.append(str(part.relative_to(DEST)))
        restored = b"".join((DEST / p).read_bytes() for p in parts)
        restored = gzip.decompress(restored) if encoding == "gzip" else restored
        assert restored == raw
        entries.append(
            {
                "source_relative_path": relative,
                "original_path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "encoding": encoding,
                "parts": parts,
            }
        )
    receipt = json.loads((output / "receipt.json").read_bytes())
    export_json(
        DEST / "PUBLICATION.json",
        {
            "schema": "f5-step1-attempt03-complete-evidence/v1",
            "attempt": 3,
            "publication_kind": "COMPLETE_SAVED_EVIDENCE_PENDING_POST_PUBLICATION_VERIFICATION",
            "entries": entries,
            "original_file_count": len(entries),
            "original_total_bytes": sum(e["size"] for e in entries),
            "saved_receipt_status": receipt["status"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "dispatches_by_this_task": 0,
            "provider_calls_by_this_task": 0,
            "scientific_look_performed": False,
            "scientific_units_completed": 0,
            "release_evidence_scope": (
                "Saved receipt and broker assignment disappearance; no fresh provider query"
            ),
        },
    )
    print(
        json.dumps(
            {
                "files": len(entries),
                "bytes": sum(e["size"] for e in entries),
                "publication_sha256": verify_export(DEST / "PUBLICATION.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
