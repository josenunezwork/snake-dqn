"""Seed-collision audit for the Claim A label/probe namespaces.

Seeds are exactly the ones the ratified world generator uses
(``dev_screen.screen_seeds(count, "t34-exit-b8-20261019/v1", namespace)`` via
``trajectory.make_world_rows``).  Two independent checks, both fail-closed:

1. Explicit intersection with the registered banks: Round 0
   (``research/t34_round0/BANK_ROUND0.json``), Round 0-D
   (``research/t34_round0d/BANK_ROUND0D.json``) and the Path B retention bank
   (``research/t34_pathb/RETENTION_BANK_V2.json``), plus their excluded
   preflight/A5 fixture rows.
2. The ratified freshness scan ``research.t34_pathb.bank._audit_candidates``:
   every seed is searched as a whole word over docs/research/src/scripts and the
   whole ``snake-dqn-artifacts`` history (which holds every strict-gate,
   serving and screen bank receipt) plus the concrete FRP training exclusions.

The receipt never lists raw seeds (only per-namespace SHA-256 digests), so it
cannot collide with itself in a later scan.  Run it from the main checkout:
the scan needs ``<repo>/../snake-dqn-artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

DOMAIN = "t34-exit-b8-20261019/v1"
REGISTERED_BANKS = {
    "round0": "research/t34_round0/BANK_ROUND0.json",
    "round0d": "research/t34_round0d/BANK_ROUND0D.json",
    "retention_v2": "research/t34_pathb/RETENTION_BANK_V2.json",
}
QUALIFICATION_NAMESPACES = {
    "t34-claim-a/qualification/label": 24,
    "t34-claim-a/qualification/probe": 32,
}


class SeedAuditFailed(RuntimeError):
    """A Claim A namespace collides with a registered bank or visible history."""


def namespace_seeds(namespace: str, count: int) -> list[int]:
    from research.apex_safety_20260926 import dev_screen

    return [int(seed) for seed in dev_screen.screen_seeds(int(count), DOMAIN, namespace)]


def seed_digest(seeds: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(s) for s in seeds).encode()).hexdigest()


def _bank_seeds(value: Any) -> set[int]:
    """Every integer stored under a ``*seed*`` key anywhere in a bank document."""
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "seed" in str(key).lower():
                if type(item) is int:
                    found.add(item)
                elif isinstance(item, list):
                    found.update(x for x in item if type(x) is int)
            found |= _bank_seeds(item)
    elif isinstance(value, list):
        for item in value:
            found |= _bank_seeds(item)
    return found


def audit(
    namespaces: Mapping[str, int],
    *,
    repo: Path,
    scan: bool = True,
    scan_repo: Path | None = None,
) -> dict[str, Any]:
    """Audit all namespaces at once; raise :class:`SeedAuditFailed` on any collision."""
    from research.t34_pathb.bank import _audit_candidates

    per_namespace = {}
    everything: list[int] = []
    for namespace, count in sorted(namespaces.items()):
        if not namespace.startswith("t34-claim-a/"):
            raise SeedAuditFailed(f"not a Claim A namespace: {namespace}")
        seeds = namespace_seeds(namespace, count)
        per_namespace[namespace] = {"count": int(count), "sha256": seed_digest(seeds)}
        everything.extend(seeds)
    if len(set(everything)) != len(everything):
        raise SeedAuditFailed("internal collision between Claim A namespaces")
    banks = {}
    for name, relative in REGISTERED_BANKS.items():
        path = repo / relative
        if not path.is_file():
            raise SeedAuditFailed(f"registered bank missing: {path}")
        seeds = _bank_seeds(json.loads(path.read_text()))
        if not seeds:
            raise SeedAuditFailed(f"no seeds parsed from registered bank {path}")
        hits = sorted(set(everything) & seeds)
        if hits:
            raise SeedAuditFailed(f"collision with {name}: {len(hits)} seeds")
        banks[name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bank_seeds": len(seeds),
            "collisions": 0,
        }
    history = None
    if scan:
        try:
            history = _audit_candidates(scan_repo or repo, everything)
        except RuntimeError as exc:
            raise SeedAuditFailed(f"freshness scan rejected the allocation: {exc}") from exc
        history = {
            key: history[key]
            for key in ("method", "candidate_count", "scan_roots", "coverage", "limits")
        } | {
            "concrete_hits": {root: len(hits) for root, hits in history["concrete_hits"].items()},
            "historical_frp_exclusions": len(history["historical_frp_exclusions"]),
        }
    return {
        "schema": "t34-claim-a-seed-audit/v1",
        "status": "PASS",
        "domain": DOMAIN,
        "namespaces": per_namespace,
        "registered_banks": banks,
        "history_scan": history,
        "note": "raw seeds deliberately omitted; recompute with namespace_seeds()",
    }


def verify_receipt(receipt: Mapping[str, Any], namespace: str, count: int) -> None:
    """A manifest's namespace/count must be covered by a PASS receipt (prefix-stable)."""
    if receipt.get("schema") != "t34-claim-a-seed-audit/v1" or receipt.get("status") != "PASS":
        raise SeedAuditFailed("seed audit receipt is not a PASS Claim A audit")
    if receipt.get("history_scan") is None:
        raise SeedAuditFailed("seed audit receipt lacks the full-history scan")
    entry = receipt.get("namespaces", {}).get(namespace)
    if not entry or int(entry["count"]) < int(count):
        raise SeedAuditFailed(f"namespace {namespace} x{count} is not covered by the audit")
    if seed_digest(namespace_seeds(namespace, int(entry["count"]))) != entry["sha256"]:
        raise SeedAuditFailed(f"audited seeds for {namespace} do not recompute")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--scan-repo", type=Path, help="checkout whose ../snake-dqn-artifacts is scanned"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--namespace",
        action="append",
        help="NAMESPACE=COUNT (default: the qualification label/probe namespaces)",
    )
    args = parser.parse_args()
    namespaces = dict(QUALIFICATION_NAMESPACES)
    if args.namespace:
        namespaces = {}
        for item in args.namespace:
            name, _, count = item.rpartition("=")
            namespaces[name] = int(count)
    receipt = audit(
        namespaces,
        repo=args.repo.resolve(),
        scan_repo=args.scan_repo.resolve() if args.scan_repo else None,
    )
    receipt["scan_repo"] = str((args.scan_repo or args.repo).resolve())
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({"status": receipt["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
