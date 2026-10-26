"""Materialize only allowed closed-F0 lives; never open excluded life payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import digest, load_registry, safe_path, validate_rows, verify_sources


def prepare(dose: str, output: Path, registry: dict | None = None) -> dict:
    """Verify each selected immutable F0 file, then copy all its training rows."""
    import numpy as np

    from research.bl_round30.materialize import _npz_bytes, _table

    registry = registry or load_registry()
    verify_sources(registry)
    output = safe_path(output)
    if output.exists():
        raise ValueError("create-only output already exists")
    chosen = set(registry["datasets"][dose]["life_ids"])
    parts = []
    for life in registry["lives"]:
        if life["source_life_id"] not in chosen:
            continue
        path = safe_path(life["payload_path"])
        if path.stat().st_size != life["payload_size"] or digest(path) != life["payload_sha256"]:
            raise ValueError("closed F0 payload changed")
        if (
            Path(str(path) + ".complete").read_text() != "complete\n"
            or Path(str(path) + ".sha256").read_text().split()[0] != life["payload_sha256"]
        ):
            raise ValueError("closed F0 completion changed")
        unit = json.loads(path.read_text())
        if (
            unit["life_id"] != life["source_life_id"]
            or unit["parent_world"] != life["ultimate_parent_id"]
            or unit["status"] != "COMPLETE"
        ):
            raise ValueError("closed F0 ancestry changed")
        part = _table([(life, row) for row in unit["rows"]])
        records = [
            dict(zip(("row_id", "life_id", "parent_world"), r))
            for r in zip(
                part["row_id"].tolist(), part["life_id"].tolist(), part["parent_world"].tolist()
            )
        ]
        validate_rows(records, registry, dose)
        parts.append(part)
    table = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_npz_bytes(table))
    result = {
        "path": str(output),
        "sha256": digest(output),
        "row_count": len(table["q"]),
        "life_count": len(chosen),
        "parent_count": len(set(table["parent_world"].tolist())),
    }
    Path(str(output) + ".sha256").write_text(result["sha256"] + "  " + output.name + "\n")
    Path(str(output) + ".complete").write_text("complete\n")
    return result


def main() -> None:
    """Print a provenance binding for insertion into the committed registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", choices=["D256", "D1024"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dose, args.output)))


if __name__ == "__main__":
    main()
