"""Create the immutable T11 input binding before its bounded MPS smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
PACKAGE = ROOT / "research/t11_local30_20261017"
RUNS = ARTIFACTS / "redesign-t11-mac-20261017"


def sha256(path: Path) -> str:
    """Hash an artifact without materializing its model or arrays."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind() -> dict:
    """Verify historical identities and publish create-only input hashes."""
    output = PACKAGE / "launch_inputs.json"
    if output.exists():
        raise ValueError(f"Refusing existing immutable binding: {output}")
    historical = ROOT / "research/t4a_two_sided_20261016/launch_inputs.json"
    files = dict(json.loads(historical.read_text())["files"])
    for name, expected in files.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Historical input changed: {name}")
    bank = RUNS / "bank-local30"
    manifest = json.loads((bank / "manifest.json").read_text())
    if {p.name for p in bank.glob("*.npz")} != set(manifest["shards"]):
        raise ValueError("LOCAL30 bank shard set mismatch")
    files[str(bank / "manifest.json")] = sha256(bank / "manifest.json")
    for name, expected in manifest["shards"].items():
        path = bank / name
        if sha256(path) != expected:
            raise ValueError(f"LOCAL30 shard changed: {path}")
        files[str(path)] = expected
    audit = json.loads((PACKAGE / "seed_audit.json").read_text())
    if not (
        audit.get("schema") == "t11-local30-seed-audit/v1"
        and audit.get("pass") is True
        and audit.get("status") == "PASS"
    ):
        raise ValueError("T11 seed audit did not pass")
    sources = [
        ROOT / "docs/research/specs/t11_prereg_2026-10-17.md",
        ROOT / "docs/research/specs/t11_local30_spec_2026-10-17.md",
        ROOT / "research/redesign_m3c_20261008/train_m3c.py",
        ROOT / "research/redesign_m3c_20261008/learner_m3c.py",
        ROOT / "research/t3_rollout_labels_20261015/bank.py",
        historical,
    ]
    sources += sorted(PACKAGE.glob("*.py"))
    sources += [PACKAGE / name for name in ("lane0.sh", "lane1.sh", "seed_audit.json")]
    for path in sources:
        files[str(path)] = sha256(path)
    result = {
        "schema": "t11-launch-inputs/v1",
        "authority": "200k MPS smoke only; prepared 8M lanes remain unstarted",
        "historical_binding": {"path": str(historical), "sha256": sha256(historical)},
        "files": files,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    """Freeze inputs only; never launch training."""
    result = bind()
    print(
        json.dumps({"binding": str(PACKAGE / "launch_inputs.json"), "files": len(result["files"])})
    )


if __name__ == "__main__":
    main()
