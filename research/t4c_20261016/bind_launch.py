"""Freeze the prepared T4c inputs after data and CPU verification complete."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"


def sha256(path: Path) -> str:
    """Stream artifact hashing without loading checkpoints into a device."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Validate historical identities, then write a create-only launch binding."""
    output = ROOT / "research/t4c_20261016/launch_inputs.json"
    if output.exists():
        raise SystemExit(f"Refusing to replace frozen launch binding: {output}")
    old_path = ROOT / "research/t4a_two_sided_20261016/launch_inputs.json"
    original = json.loads(old_path.read_text())
    files = dict(original["files"])
    for name, expected in files.items():
        if sha256(Path(name)) != expected:
            raise SystemExit(f"Historical registered input changed: {name}")
    for part, expected_schema in (
        ("bank-v3", "t4c-derived-label-bank/v3"),
        ("outside-bank", "t4c-outside-counter-bank/v1"),
    ):
        directory = ARTIFACTS / "t4c-data-20261016" / part
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != expected_schema:
            raise SystemExit(f"Wrong data manifest schema: {manifest_path}")
        files[str(manifest_path)] = sha256(manifest_path)
        for name, digest in manifest["shards"].items():
            shard = directory / name
            if sha256(shard) != digest:
                raise SystemExit(f"Bank payload changed: {shard}")
            files[str(shard)] = digest
    for name in (
        "docs/research/specs/t4c_prereg_2026-10-16.md",
        "research/redesign_m3c_20261008/train_m3c.py",
        "research/redesign_m3c_20261008/learner_m3c.py",
        "research/t3_rollout_labels_20261015/bank.py",
        "research/t4c_20261016/lane0.sh",
        "research/t4c_20261016/lane1.sh",
    ):
        path = ROOT / name
        files[str(path)] = sha256(path)
    receipt = {
        "schema": "t4c-launch-inputs/v1",
        "authority": "prepared CPU-only launch; no training started",
        "historical_binding": {"path": str(old_path), "sha256": sha256(old_path)},
        "files": files,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Bound {len(files)} inputs in {output}")


if __name__ == "__main__":
    main()
