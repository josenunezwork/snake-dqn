"""Fresh, narrow seed registration for the T12B Check-4 rerun.

The prior T12 audit remains the broad registered-bank scanner.  This wrapper
uses its audit under a tightly bounded namespace/purpose patch, then adds
explicit exclusions that a general JSON scan cannot infer: all original T12
seeds and the two fixed diagnostic probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t12_b8_serving_20261017 import seed_audit as prior  # noqa: E402

NAMESPACE = "t12b-check4-20261017/v1"
SCHEMA = "t12b-check4-seed-audit/v1"
PURPOSE_COUNTS = {"latency": 4}
PRIOR_AUDIT = Path("research/t12_b8_serving_20261017/SEED_AUDIT_T12_V2.json")


def sha256(path: Path) -> str:
    """Return a SHA-256 file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    """Return the canonical digest for a seed set."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def seed_for(index: int, namespace: str = NAMESPACE) -> int:
    """Allocate one ordered latency seed using the registered recipe."""
    if not 0 <= index < PURPOSE_COUNTS["latency"]:
        raise ValueError("latency seed index is outside the four-case roster")
    material = f"{namespace}|latency|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generate_banks(namespace: str = NAMESPACE) -> dict[str, Any]:
    """Return the four fixed Check-4 latency identities without filesystem I/O."""
    seeds = [seed_for(index, namespace) for index in range(PURPOSE_COUNTS["latency"])]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("T12B latency allocation collides internally")
    return {
        "namespace": namespace,
        "recipe": "uint32_be(sha256(f'{namespace}|latency|{i}')[:4])",
        "banks": {"latency": {"count": 4, "seeds": seeds, "seed_digest": seed_digest(seeds)}},
        "count": 4,
        "seed_digest": seed_digest(seeds),
    }


def _prior_and_diagnostic_seeds(repo: Path) -> set[int]:
    document = json.loads((repo / PRIOR_AUDIT).read_text(encoding="utf-8"))
    original = {int(seed) for row in document["banks"].values() for seed in row.get("seeds", [])}
    # These fixed diagnostic seeds are intentionally not inferred from an
    # output artifact: the audit must reject them even before diagnosis runs.
    from research.t12b_check4_20261017.diagnose import DIAGNOSTIC_SEEDS

    return original | {int(seed) for seed in DIAGNOSTIC_SEEDS.values()}


def _unexcluded_new_artifact_hits(repo: Path, candidates: set[int]) -> dict[str, Any]:
    """Scan the T12B artifact tree that the inherited audit excludes broadly."""
    artifact_root = repo.parent / "snake-dqn-artifacts" / "t12b-check4-20261017"
    if not artifact_root.exists():
        return {"root": str(artifact_root), "exists": False, "hits": []}
    command = [
        "rg",
        "--hidden",
        "--no-ignore",
        "--no-filename",
        "--fixed-strings",
        "--word-regexp",
        "--only-matching",
    ]
    command.extend(item for seed in sorted(candidates) for item in ("-e", str(seed)))
    command.append(str(artifact_root))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"T12B artifact scan failed: {result.stderr}")
    return {
        "root": str(artifact_root),
        "exists": True,
        "hits": sorted({int(token) for token in result.stdout.splitlines() if token.isdigit()}),
        "command": command[:-1] + ["<t12b-artifact-root>"],
    }


def audit(repo: str | Path = ".") -> dict[str, Any]:
    """Run prior broad coverage plus T12/T12B explicit collision checks."""
    repo_path = Path(repo).resolve()
    allocation = generate_banks()
    candidates = set(allocation["banks"]["latency"]["seeds"])
    # The prior audit supplies its complete T7/rejection/artifact inventory.
    # Patch only its allocation globals; do not modify any old research files.
    with (
        patch.object(prior, "NAMESPACE", NAMESPACE),
        patch.object(prior, "PURPOSE_COUNTS", PURPOSE_COUNTS),
        patch.object(prior, "generate_banks", side_effect=generate_banks),
        patch.object(prior, "ARTIFACT_OUTPUT_DIRNAME", "t12b-check4-20261017"),
    ):
        inherited = prior.audit(repo_path)
    explicit = _prior_and_diagnostic_seeds(repo_path)
    explicit_collisions = sorted(candidates & explicit)
    new_artifact_scan = _unexcluded_new_artifact_hits(repo_path, candidates)
    passed = (
        bool(inherited.get("pass")) and not explicit_collisions and not new_artifact_scan["hits"]
    )
    return {
        "schema": SCHEMA,
        "pass": passed,
        "namespace": NAMESPACE,
        "allocation": allocation,
        "prior_audit": inherited,
        "explicit_exclusions": {
            "prior_t12_allocation_path": str(repo_path / PRIOR_AUDIT),
            "prior_t12_allocation_sha256": sha256(repo_path / PRIOR_AUDIT),
            "prior_t12_and_diagnostic_count": len(explicit),
            "collision_seeds": explicit_collisions,
        },
        "unexcluded_t12b_artifact_scan": new_artifact_scan,
        "limitations": inherited.get("limitations", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite seed audit: {args.out}")
    result = audit(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=1)
        stream.write("\n")
    print(json.dumps({"pass": result["pass"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
