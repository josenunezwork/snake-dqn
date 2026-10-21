"""Fail-closed E6/E7 allocation and collision audit.

The visible-history scan reuses T4e's bounded, partitioned ripgrep helper.  It
passes an exact exclusion for the sealed Round-0 study; that tree is never
walked or opened.  Its published bank and matching audit are the only allowed
coverage proof for its world identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.apex_safety_20260926 import dev_screen
from research.bl_ppo.exports import export_json
from research.t4e_return_20261016 import seed_audit as t4e_audit

DOMAIN = "bl-prefix-e6e7-development/20261021/v2"
SEED_OFFSET = 1 << 40
NAMESPACES = {
    "source": "source-world",
    "future": "future-rng",
    "exploration": "exploration",
    "preflight": "preflight",
}
SOURCE_WORLD_COUNT, FUTURE_RNG_COUNT, EXPLORATION_COUNT, PREFLIGHT_COUNT = 4096, 3000, 4000, 128
FROZEN_COUNTS = {
    "source": SOURCE_WORLD_COUNT,
    "future": FUTURE_RNG_COUNT,
    "exploration": EXPLORATION_COUNT,
    "preflight": PREFLIGHT_COUNT,
}
SEALED_REL = Path("t34-round0-study-01")
TEXT_GLOBS = ("*.json", "*.jsonl", "*.md", "*.py", "*.yaml", "*.yml", "*.sh", "*.csv", "*.tsv")


class SeedAuditFailed(RuntimeError):
    """The complete allocation cannot be certified and must not be used."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(seed) for seed in seeds).encode()).hexdigest()


def namespace_seeds(kind: str, count: int, *, round_index: int = 0) -> list[int]:
    """Derive a deterministic namespace.  Never replace a collision."""
    if kind not in NAMESPACES:
        raise ValueError(f"unknown BL prefix namespace: {kind}")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    namespace = NAMESPACES[kind] if round_index == 0 else f"{NAMESPACES[kind]}/round{round_index}"
    return [SEED_OFFSET + int(seed) for seed in dev_screen.screen_seeds(count, DOMAIN, namespace)]


def frozen_allocation(*, round_index: int = 0) -> dict[str, list[int]]:
    allocation = {
        kind: namespace_seeds(kind, count, round_index=round_index)
        for kind, count in FROZEN_COUNTS.items()
    }
    values = [seed for group in allocation.values() for seed in group]
    if len(values) != len(set(values)):
        raise SeedAuditFailed("whole allocation rejected: internal namespace collision")
    return allocation


def _collect_ints(value: Any) -> set[int]:
    """Collect numeric identities from a JSON object, including arbitrary lists."""
    if type(value) is int:
        return {value}
    if isinstance(value, list):
        return set().union(*(_collect_ints(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(_collect_ints(item) for item in value.values())) if value else set()
    return set()


def _bound_bank(
    repo: Path, bank_rel: str, audit_rel: str, statuses: set[str], *, require_binding: bool = False
) -> tuple[dict[str, Any], set[int]]:
    bank_path, audit_path = repo / bank_rel, repo / audit_rel
    if not bank_path.is_file() or not audit_path.is_file():
        raise SeedAuditFailed(f"missing registered inventory: {bank_rel} or {audit_rel}")
    bank, receipt = json.loads(bank_path.read_text()), json.loads(audit_path.read_text())
    status_ok = receipt.get("status") in statuses or receipt.get("pass") is True
    if not status_ok:
        raise SeedAuditFailed(f"registered inventory not complete: {audit_rel}")
    expected = receipt.get("bank_sha256")
    if require_binding and not isinstance(expected, str):
        raise SeedAuditFailed(f"registered bank audit lacks required binding: {audit_rel}")
    if expected is not None and expected != sha256(bank_path):
        raise SeedAuditFailed(f"registered bank bytes differ from audit: {bank_rel}")
    values = _collect_ints(bank)
    if not values:
        raise SeedAuditFailed(f"registered inventory has no numeric identities: {bank_rel}")
    return (
        {
            "bank": bank_rel,
            "audit": audit_rel,
            "bank_sha256": sha256(bank_path),
            "count": len(values),
        },
        values,
    )


def _registered_history(repo: Path) -> tuple[dict[str, Any], set[int]]:
    """Reuse T7/T12 history plus published banks without opening sealed outputs."""
    from research.redesign_t7_phase_r_20261016 import seed_audit as t7_audit
    from research.t12_b8_serving_20261017 import seed_audit as t12_audit

    registered = set(t7_audit._registered(repo))
    t12_summary, t12_values = t12_audit._t7_allocations(repo)
    registered.update(t12_values)
    evidence: dict[str, Any] = {
        "t7_generated_history": {
            "count": len(registered),
            "seed_digest": seed_digest(sorted(registered)),
        },
        "t12_t7_allocations": t12_summary,
    }
    for label, bank, receipt, statuses, require_binding in (
        (
            "round0",
            "research/t34_round0/BANK_ROUND0.json",
            "research/t34_round0/SEED_AUDIT_ROUND0.json",
            {"CLEAR"},
            True,
        ),
        (
            "round0d",
            "research/t34_round0d/BANK_ROUND0D.json",
            "research/t34_round0d/SEED_AUDIT_ROUND0D.json",
            {"CLEAR"},
            False,
        ),
        (
            "pathb",
            "research/t34_pathb/RETENTION_BANK_V2.json",
            "research/t34_pathb/SEED_AUDIT_RETENTION_V2.json",
            {"PASS"},
            False,
        ),
    ):
        row, values = _bound_bank(repo, bank, receipt, statuses, require_binding=require_binding)
        evidence[label] = row
        registered.update(values)
    evidence["sealed_round0_coverage"] = {
        "sealed_rel": str(SEALED_REL),
        "proof": "published BANK_ROUND0 plus matching CLEAR audit",
        "covered": True,
    }
    artifacts = repo.parent / "snake-dqn-artifacts"
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise SeedAuditFailed("historical FRP exclusion inventories incomplete")
    rows = []
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            raise SeedAuditFailed(f"malformed historical exclusion inventory: {path}")
        registered.update(values)
        rows.append({"path": str(path), "sha256": sha256(path), "count": len(values)})
    evidence["historical_frp_exclusions"] = rows
    # Published Exit-B8 receipts sometimes bind only namespace/count/digest;
    # reconstruct their identities before comparing rather than treating a
    # missing literal seed list as evidence of freshness.
    from research.t34_claim_a.seed_audit import _bank_seeds
    from research.t34_claim_a.seed_audit import namespace_seeds as claim_a_seeds
    from research.t34_claim_a.seed_audit import seed_digest as claim_a_digest

    published = []
    for path in sorted((repo / "research").rglob("*SEED_AUDIT*.json")):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repo / "research") or SEALED_REL.name in resolved.parts:
            raise SeedAuditFailed(f"unsafe published inventory path: {path}")
        value = json.loads(resolved.read_text())
        derived = set()
        if value.get("domain") == "t34-exit-b8-20261019/v1":
            for namespace, entry in value.get("namespaces", {}).items():
                values = claim_a_seeds(namespace, int(entry["count"]))
                if claim_a_digest(values) != entry["sha256"]:
                    raise SeedAuditFailed(f"published seed namespace digest mismatch: {path}")
                derived.update(values)
        registered.update(_bank_seeds(value))
        registered.update(derived)
        published.append(
            {
                "path": str(path.relative_to(repo)),
                "resolved_path": str(resolved.relative_to(repo)),
                "sha256": sha256(path),
                "derived_count": len(derived),
            }
        )
    evidence["published_seed_audits"] = published
    return evidence, registered


def _sealed_symlink_exclusions(root: Path, sealed: Path) -> tuple[Path, ...]:
    """Find links resolving into the sealed tree without ever following them."""
    result: list[Path] = []
    pending = [root]
    sealed_resolved = sealed.resolve()
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path == sealed:
                    continue
                if entry.is_symlink():
                    resolved = path.resolve()
                    if resolved == sealed_resolved or sealed_resolved in resolved.parents:
                        result.append(path)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
    return tuple(result)


def _visible_text_scan(
    repo: Path, artifacts: Path, candidates: set[int], excludes: tuple[Path, ...]
) -> dict[str, Any]:
    roots = tuple(
        path
        for path in (repo / "docs", repo / "research", repo / "src", repo / "scripts", artifacts)
        if path.is_dir()
    )
    missing = [
        str(path) for path in (repo / "docs", repo / "research", artifacts) if not path.is_dir()
    ]
    if missing:
        raise SeedAuditFailed(f"missing mandatory visible-history roots: {missing}")
    arguments = ["--hidden", "--no-ignore", "--fixed-strings", "--word-regexp", "--only-matching"]
    for glob in TEXT_GLOBS:
        arguments.extend(["--glob", glob])
    for seed in sorted(candidates):
        arguments.extend(["-e", str(seed)])
    scans = {
        str(root): t4e_audit._partitioned_rg(root, arguments, excludes, "bl-prefix-identities")
        for root in roots
    }
    return {
        "method": "t4e-partitioned-rg/fixed-word/threads1/timeout60",
        "roots": [str(root) for root in roots],
        "hits": {
            name: sorted(int(value) for value in found if value.isdigit())
            for name, (found, _) in scans.items()
        },
        "progress": {name: progress for name, (_, progress) in scans.items()},
    }


def audit(
    names: Mapping[str, int] | None = None,
    *,
    repo: Path = Path("."),
    out: Path | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    """Return a PASS receipt or raise; a partial audit is never a scientific PASS."""
    repo = Path(repo).resolve()
    requested = dict(FROZEN_COUNTS if names is None else names)
    if not requested or set(requested) - set(NAMESPACES):
        raise ValueError("requested allocation must contain known namespaces")
    allocation = {
        kind: namespace_seeds(kind, count, round_index=round_index)
        for kind, count in requested.items()
    }
    candidates = {seed for values in allocation.values() for seed in values}
    if len(candidates) != sum(len(values) for values in allocation.values()):
        raise SeedAuditFailed("whole allocation rejected: internal namespace collision")
    artifacts = repo.parent / "snake-dqn-artifacts"
    sealed = artifacts / SEALED_REL
    exclusions = (sealed, *_sealed_symlink_exclusions(artifacts, sealed)) + (
        (Path(out).resolve(),) if out is not None else ()
    )
    history, registered = _registered_history(repo)
    registered_hits = sorted(candidates & registered)
    if registered_hits:
        raise SeedAuditFailed(f"whole allocation rejected: registered={registered_hits}")
    scan = _visible_text_scan(repo, artifacts, candidates, exclusions)
    visible_hits = sorted({seed for values in scan["hits"].values() for seed in values})
    if registered_hits or visible_hits:
        raise SeedAuditFailed(
            f"whole allocation rejected: registered={registered_hits}, visible={visible_hits}"
        )
    return {
        "schema": "bl-prefix-seed-audit/v2",
        "status": "PASS",
        "domain": DOMAIN,
        "round_index": round_index,
        "allocation": {
            kind: {
                "namespace": (
                    NAMESPACES[kind]
                    if round_index == 0
                    else f"{NAMESPACES[kind]}/round{round_index}"
                ),
                "count": len(values),
                "seeds": values,
                "seed_digest": seed_digest(values),
            }
            for kind, values in allocation.items()
        },
        "candidate_count": len(candidates),
        "candidate_digest": seed_digest(sorted(candidates)),
        "registered_history": history,
        "registered_collision_seeds": registered_hits,
        "visible_text_scan": scan,
        "sealed_study": {
            "path": str(sealed),
            "excluded_exactly": True,
            "opened_or_traversed": False,
        },
        "limitations": (
            "No claim about opaque binaries, unrecorded external work, or inaccessible history."
        ),
    }


def verify_receipt(
    receipt: Mapping[str, Any], *, preflight_only: bool = False, round_index: int = 0
) -> None:
    """Reconstruct an export's allocation and reject incomplete provenance."""
    if receipt.get("schema") != "bl-prefix-seed-audit/v2" or receipt.get("status") != "PASS":
        raise SeedAuditFailed("seed receipt is not a PASS v2 receipt")
    if receipt.get("domain") != DOMAIN:
        raise SeedAuditFailed("seed receipt domain differs from frozen registration")
    if receipt.get("round_index", 0) != round_index:
        raise SeedAuditFailed("seed receipt round differs from requested allocation")
    allocation = receipt.get("allocation")
    expected = {"preflight": PREFLIGHT_COUNT} if preflight_only else FROZEN_COUNTS
    if not isinstance(allocation, Mapping) or set(allocation) != set(expected):
        raise SeedAuditFailed("seed receipt namespace set differs from requested allocation")
    all_values: list[int] = []
    for kind, count in expected.items():
        row = allocation.get(kind)
        namespace = (
            NAMESPACES[kind] if round_index == 0 else f"{NAMESPACES[kind]}/round{round_index}"
        )
        if not isinstance(row, Mapping) or row.get("namespace") != namespace:
            raise SeedAuditFailed(f"seed receipt namespace metadata invalid: {kind}")
        values = row.get("seeds")
        if not isinstance(values, list) or values != namespace_seeds(
            kind, count, round_index=round_index
        ):
            raise SeedAuditFailed(f"seed receipt allocation does not reconstruct: {kind}")
        if row.get("count") != count or row.get("seed_digest") != seed_digest(values):
            raise SeedAuditFailed(f"seed receipt count/digest invalid: {kind}")
        all_values.extend(values)
    if len(all_values) != len(set(all_values)):
        raise SeedAuditFailed("seed receipt has an internal collision")
    history = receipt.get("registered_history")
    if not isinstance(history, Mapping) or not isinstance(
        history.get("historical_frp_exclusions"), list
    ):
        raise SeedAuditFailed("seed receipt lacks mandatory historical inventories")
    coverage = history.get("sealed_round0_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("covered") is not True:
        raise SeedAuditFailed("seed receipt lacks sealed-study coverage proof")
    if receipt.get("registered_collision_seeds") != []:
        raise SeedAuditFailed("seed receipt records registered collisions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--round-index", type=int, default=0)
    args = parser.parse_args()
    receipt = audit(
        {"preflight": PREFLIGHT_COUNT} if args.preflight_only else None,
        repo=args.repo,
        out=args.out,
        round_index=args.round_index,
    )
    print(
        json.dumps({"status": receipt["status"], **export_json(args.out, receipt)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
