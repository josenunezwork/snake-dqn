"""E16 finite natural-world banks and a sealed-root-pruned freshness audit.

This is a producer of identity metadata only. Its consumers must use the finite
critic/actor vectors, never substitute a PRNG-derived reset seed. Evaluation,
timing and preflight worlds have separate namespaces. No game is played here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
DOMAIN = "bl-e16-20261021/v1"
SCHEMA = "bl-e16-bank/v1"
AUDIT_SCHEMA = "bl-e16-seed-audit/v1"
SEALED_DIRECTORY = "t34-round0-study-01"
MIX_COUNTS = dict.fromkeys(("scripted", "frozen", "mixed"), 96)
PHASE_COUNTS = {"critic": 4096, "actor": 4096, "timing": 3, "preflight": 1, "evaluation": 96}
SEED_OFFSET = 1 << 48
SUFFIXES = {".json", ".jsonl", ".md", ".py", ".yaml", ".yml", ".sh", ".csv", ".tsv", ".txt"}


class SeedAuditFailed(RuntimeError):
    """The allocation is unusable; no individual seed may be replaced."""


def canonical_sha(value: Any) -> str:
    """Hash portable JSON semantics and reject nonfinite values."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    """Hash a regular file without loading a large artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def namespace_seeds(phase: str, mix: str, count: int, round_index: int = 0) -> list[int]:
    """Derive the entire frozen 48-bit SHA family inside signed int64."""
    if phase not in PHASE_COUNTS or mix not in MIX_COUNTS:
        raise ValueError("unknown E16 phase or mix")
    if type(count) is not int or not 1 <= count <= PHASE_COUNTS[phase]:
        raise ValueError("invalid namespace count")
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    namespace = f"{DOMAIN}/round-{round_index}/{phase}/{mix}"
    return [
        SEED_OFFSET
        + int.from_bytes(hashlib.sha256(f"{namespace}|{i}".encode()).digest()[:6], "big")
        for i in range(count)
    ]


def _rows(phase: str, round_index: int) -> list[dict[str, Any]]:
    # Reuse only the read-only established roster builder, replacing its seed
    # family before any simulation or policy consumes a row.
    from research.t34_round0.bank import _rows as roster_rows

    namespace = f"{DOMAIN}/round-{round_index}/{phase}"
    horizon = 40 if phase == "preflight" else 5000
    rows = roster_rows(namespace, dict.fromkeys(MIX_COUNTS, PHASE_COUNTS[phase]), horizon)
    groups = {
        mix: namespace_seeds(phase, mix, PHASE_COUNTS[phase], round_index) for mix in MIX_COUNTS
    }
    offsets = dict.fromkeys(MIX_COUNTS, 0)
    for row in rows:
        mix = row["mix"]
        row["world_seed"] = groups[mix][offsets[mix]]
        offsets[mix] += 1
        row["natural_start"] = True
    return rows


def materialize(round_index: int = 0) -> dict[str, Any]:
    """Construct a deterministic metadata bank; this never samples outcomes."""
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    value = {
        "schema": SCHEMA,
        "domain": DOMAIN,
        "round_index": round_index,
        "counts": dict(MIX_COUNTS),
        "phase_counts_per_mix": dict(PHASE_COUNTS),
        "profile": {
            "horizon": 5000,
            "hero_mode": "terminal",
            "hero_slot": 0,
            "hero_veto": "none",
            "frozen_veto": "v8",
            "mechanics_version": 2,
            "engine": "GridBatchSim",
            "natural_start": True,
        },
        "rows": _rows("evaluation", round_index),
        "excluded_timing_rows": _rows("timing", round_index),
        "excluded_preflight_rows": _rows("preflight", round_index),
        "phase_seeds": {
            phase: {mix: namespace_seeds(phase, mix, count, round_index) for mix in MIX_COUNTS}
            for phase, count in PHASE_COUNTS.items()
            if phase in {"critic", "actor"}
        },
    }
    flattened = list(iter_seeds(value))
    if len(flattened) != len(set(flattened)):
        raise SeedAuditFailed("whole allocation rejected: internal collision")
    value["allocation_sha256"] = canonical_sha(value)
    return value


def iter_seeds(bank: Mapping[str, Any]) -> Iterable[int]:
    """Yield every seed, preserving duplicates so validation can reject them."""
    for key in ("rows", "excluded_timing_rows", "excluded_preflight_rows"):
        for row in bank[key]:
            yield row["world_seed"]
    for groups in bank["phase_seeds"].values():
        for seeds in groups.values():
            yield from seeds


def all_seeds(bank: Mapping[str, Any]) -> set[int]:
    """Return the complete mutually disjoint identity set."""
    return set(iter_seeds(bank))


def validate_bank(bank: Mapping[str, Any]) -> None:
    """Reject changes to any ordered vector, roster, count or profile."""
    if bank.get("schema") != SCHEMA or bank != materialize(bank.get("round_index")):
        raise SeedAuditFailed("bank differs from frozen deterministic allocation")


def training_seed(
    bank: Mapping[str, Any], phase: str, world_id: int, episode: int, worlds: int = 128
) -> int:
    """Map each collector world/episode to one finite equal-mix natural seed.

    The slot's mix is world_id modulo three. A/G deliberately share the same
    vector. Within an arm no identity repeats; exhaustion fails before reset.
    """
    if phase not in {"critic", "actor"} or type(worlds) is not int or worlds < 3:
        raise ValueError("invalid finite-bank collector phase/size")
    if type(world_id) is not int or not 0 <= world_id < worlds:
        raise ValueError("invalid collector world_id")
    if type(episode) is not int or episode < 0:
        raise ValueError("invalid collector episode")
    mixes = tuple(MIX_COUNTS)
    mix = mixes[world_id % 3]
    stride = len(range(world_id % 3, worlds, 3))
    index = world_id // 3 + episode * stride
    values = bank["phase_seeds"][phase][mix]
    if index >= len(values):
        raise SeedAuditFailed(f"finite {phase}/{mix} bank exhausted before reset")
    return int(values[index])


def _safe_files(
    roots: Iterable[Path],
    excludes: Iterable[Path] = (),
    aliases: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Prune forbidden names before descent; never resolve or follow symlinks."""
    excluded = {Path(os.path.abspath(path)) for path in excludes}
    result = []
    for root in roots:
        if root.is_symlink() or SEALED_DIRECTORY in root.parts:
            raise SeedAuditFailed("unsafe inventory root")
        for current, directories, names in os.walk(
            root, followlinks=False, onerror=lambda error: (_ for _ in ()).throw(error)
        ):
            directory = Path(current)
            if aliases is not None:
                for name in [*directories, *names]:
                    path = directory / name
                    if name != SEALED_DIRECTORY and path.is_symlink():
                        aliases.append(
                            {
                                "path": str(path),
                                "target": os.readlink(path),
                                "directory": name in directories,
                            }
                        )
            directories[:] = [
                name
                for name in directories
                if name not in {SEALED_DIRECTORY, ".git", "__pycache__"}
                and not (directory / name).is_symlink()
            ]
            for name in names:
                path = directory / name
                if path in excluded or path.suffix.lower() not in SUFFIXES or path.is_symlink():
                    continue
                if not path.is_file():
                    raise SeedAuditFailed(f"nonregular historical inventory: {path}")
                result.append(path)
    return sorted(set(result))


def _public_s1_alias(target: Path) -> tuple[Path, Path, Path] | None:
    """Cover the removed S1 worktree via its published, internally bound inventory."""
    if target.parts[-3:] != ("research", "s1_mature_20261018", "SEED_BANK.json"):
        return None
    package = ROOT / "research/s1_mature_20261018"
    paths = (package / "SEED_BANK.json", package / "SEED_AUDIT.json", package / "WORLD_BANK.json")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise SeedAuditFailed("published S1 alias coverage incomplete")
    value, receipt = (json.loads(path.read_text()) for path in paths[:2])
    if (
        value.get("schema") != "s1-seed-bank/v1"
        or receipt.get("pass") is not True
        or value.get("seeds") != receipt.get("training_seeds")
        or receipt.get("world_bank_file", {}).get("sha256") != sha256(paths[2])
        or value.get("collision_audit") != "research/s1_mature_20261018/SEED_AUDIT.json"
    ):
        raise SeedAuditFailed("published S1 alias coverage binding invalid")
    return paths


def _cover_aliases(
    aliases: list[dict[str, Any]], files: list[Path], roots: list[Path], excludes: Iterable[Path]
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Cover lexical alias targets via explicit ordinary paths, never alias traversal.

    External repository aliases contribute only their ordinary source/inventory
    roots. Runtime environments contain dependencies, not experiment banks.
    """
    covered = set(files)
    records = []
    pending = list(aliases)
    seen = set()
    while pending:
        alias = pending.pop()
        if alias["path"] in seen:
            continue
        seen.add(alias["path"])
        path = Path(alias["path"])
        target = Path(alias["target"])
        target = Path(os.path.abspath(target if target.is_absolute() else path.parent / target))
        if SEALED_DIRECTORY in target.parts:
            raise SeedAuditFailed("alias target requires forbidden live-study coverage")
        if not alias["directory"] and path.suffix.lower() not in SUFFIXES:
            continue
        if alias["directory"] and path.name in {"venv", ".venv", "node_modules"}:
            records.append({**alias, "coverage": "non-study dependency environment"})
            continue
        # Check components without resolving the alias. Any chain requires an
        # explicit public inventory rather than implicit link traversal.
        if any(part.is_symlink() for part in [target, *target.parents]):
            raise SeedAuditFailed(f"uncovered chained inventory alias: {path}")
        if alias["directory"]:
            if any(target.is_relative_to(root) for root in roots):
                records.append({**alias, "coverage": "ordinary target inventoried independently"})
                continue
            source_roots = [
                target / name
                for name in ("docs", "research", "src", "scripts")
                if (target / name).is_dir()
            ]
            if not source_roots:
                raise SeedAuditFailed(f"external directory alias lacks public inventory: {path}")
            next_aliases = []
            covered.update(_safe_files(source_roots, excludes, next_aliases))
            pending.extend(next_aliases)
            records.append(
                {
                    **alias,
                    "coverage": "explicit ordinary source roots",
                    "roots": list(map(str, source_roots)),
                }
            )
        else:
            if not target.is_file():
                fallback = _public_s1_alias(target)
                if fallback is None:
                    raise SeedAuditFailed(f"inventory alias target missing: {path}")
                public_bank, public_audit, public_world_bank = fallback
                covered.update(fallback)
                records.append(
                    {
                        **alias,
                        "coverage": "published S1 training seed and bound world inventories",
                        "public_bank": str(public_bank),
                        "public_bank_sha256": sha256(public_bank),
                        "public_audit_sha256": sha256(public_audit),
                        "public_world_bank_sha256": sha256(public_world_bank),
                    }
                )
                continue
            covered.add(target)
            records.append(
                {
                    **alias,
                    "coverage": "explicit ordinary target file",
                    "target_sha256": sha256(target),
                }
            )
    return sorted(covered), sorted(records, key=lambda row: row["path"])


def _scan_history(candidates: set[int], files: list[Path]) -> dict[str, Any]:
    """Scan explicit files with bounded arguments; missing/unreadable input fails."""
    found: set[int] = set()
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="bl-e16-seeds-", suffix=".patterns"
    ) as patterns:
        patterns.write("\n".join(map(str, sorted(candidates))) + "\n")
        patterns.flush()
        command = [
            "rg",
            "--threads",
            "1",
            "--hidden",
            "--no-ignore",
            "--fixed-strings",
            "--word-regexp",
            "--only-matching",
            "--no-heading",
            "--no-filename",
            "--no-line-number",
            "-f",
            patterns.name,
            "--",
        ]
        for start in range(0, len(files), 400):
            try:
                result = subprocess.run(
                    [*command, *map(str, files[start : start + 400])],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise SeedAuditFailed("historical literal scan failed or timed out") from error
            if result.returncode not in (0, 1):
                raise SeedAuditFailed(f"historical literal scan failed: {result.stderr[:500]}")
            found.update(map(int, result.stdout.split()))
    return {
        "method": "explicit-safe-files-fixed-word-threads1-timeout60/v1",
        "file_count": len(files),
        "file_list_sha256": canonical_sha(list(map(str, files))),
        "hits": sorted(found & candidates),
    }


def _registered_history(
    repo: Path, files: list[Path], artifacts: Path
) -> tuple[dict[str, Any], set[int]]:
    """Reconstruct omitted namespace seeds and bind published sealed-bank proof."""
    from research.redesign_t7_phase_r_20261016.seed_audit import _registered
    from research.t12_b8_serving_20261017.seed_audit import _t7_allocations
    from research.t34_claim_a.seed_audit import namespace_seeds as claim_seeds
    from research.t34_claim_a.seed_audit import seed_digest as claim_digest

    registered = set(_registered(repo))
    t7_record, t7_values = _t7_allocations(repo)
    registered.update(t7_values)
    bank_path = repo / "research/t34_round0/BANK_ROUND0.json"
    audit_path = repo / "research/t34_round0/SEED_AUDIT_ROUND0.json"
    if bank_path not in files or audit_path not in files:
        raise SeedAuditFailed("published sealed-bank coverage proof missing")
    proof = json.loads(audit_path.read_text())
    if proof.get("status") != "CLEAR" or proof.get("bank_sha256") != sha256(bank_path):
        raise SeedAuditFailed("published sealed-bank coverage proof is stale")
    reconstructed = []
    for path in files:
        if path.suffix != ".json" or "SEED_AUDIT" not in path.name.upper():
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise SeedAuditFailed(f"malformed published seed audit: {path}")
        if value.get("domain") == "t34-exit-b8-20261019/v1":
            for namespace, entry in value.get("namespaces", {}).items():
                seeds = claim_seeds(namespace, int(entry["count"]))
                if claim_digest(seeds) != entry["sha256"]:
                    raise SeedAuditFailed(f"published namespace digest mismatch: {path}")
                registered.update(seeds)
                reconstructed.append(
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "namespace": namespace,
                        "count": len(seeds),
                    }
                )
    exclusions = [
        path
        for path in files
        if path.name == "training_exclusions.json"
        and path.parent.name.startswith("namespaces")
        and path.parent.parent.name.startswith("frp-")
        and path.is_relative_to(artifacts)
    ]
    if len(exclusions) < 7:
        raise SeedAuditFailed("historical FRP exclusion inventories incomplete")
    records = []
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise SeedAuditFailed(f"malformed FRP exclusion inventory: {path}")
        registered.update(values)
        records.append({"path": str(path), "sha256": sha256(path), "count": len(values)})
    return {
        "historical_generated_count": len(registered),
        "t7_allocations": t7_record,
        "reconstructed_receipts": reconstructed,
        "frp_exclusions": records,
        "sealed_coverage": {
            "bank_path": str(bank_path),
            "bank_sha256": sha256(bank_path),
            "audit_path": str(audit_path),
            "audit_sha256": sha256(audit_path),
            "opened_or_traversed": False,
        },
    }, registered


def audit(
    bank: Mapping[str, Any],
    repo: Path = ROOT,
    artifacts: Path = ARTIFACTS,
    excludes: Iterable[Path] = (),
) -> dict[str, Any]:
    """Reject a whole allocation on any collision, missing proof, or scan error."""
    validate_bank(bank)
    repo, artifacts = Path(os.path.abspath(repo)), Path(os.path.abspath(artifacts))
    mandatory = (repo / "docs", repo / "research", artifacts)
    if any(not path.is_dir() for path in mandatory):
        raise SeedAuditFailed("mandatory historical scan roots unavailable")
    roots = [
        repo / name for name in ("docs", "research", "src", "scripts") if (repo / name).is_dir()
    ] + [artifacts]
    aliases: list[dict[str, Any]] = []
    files = _safe_files(roots, excludes, aliases)
    files, alias_coverage = _cover_aliases(aliases, files, roots, excludes)
    history, known = _registered_history(repo, files, artifacts)
    history["alias_coverage"] = alias_coverage
    candidates = all_seeds(bank)
    registered_hits = sorted(candidates & known)
    # Positive controls must be present in literal safe history too.
    controls = {min(known)}
    scan = _scan_history(candidates | controls, files)
    if not controls.issubset(scan["hits"]):
        raise SeedAuditFailed("historical scan failed positive control")
    hits = sorted(set(scan["hits"]) & candidates)
    if registered_hits or hits:
        raise SeedAuditFailed(
            f"whole allocation rejected: registered={registered_hits}, visible={hits}"
        )
    scan["hits"] = hits
    scan["positive_controls"] = sorted(controls)
    return {
        "schema": AUDIT_SCHEMA,
        "status": "CLEAR",
        "domain": DOMAIN,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "candidate_count": len(candidates),
        "seeds_sha256": canonical_sha(sorted(candidates)),
        "collisions": [],
        "history": history,
        "literal_scan": scan,
        "sealed_directory": {"name": SEALED_DIRECTORY, "opened_or_traversed": False},
        "limitations": [
            "Opaque binaries, unrecorded runs and inaccessible external history are not covered.",
            "Symlinks are skipped without following them; "
            "published sealed bank covers its identities.",
        ],
    }


def check_audit(bank: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    """Require a complete matching audit before a consumer resets any world."""
    validate_bank(bank)
    scan, history = receipt.get("literal_scan", {}), receipt.get("history", {})
    if (
        receipt.get("schema") != AUDIT_SCHEMA
        or receipt.get("status") != "CLEAR"
        or receipt.get("domain") != DOMAIN
        or receipt.get("bank_semantic_sha256") != bank["allocation_sha256"]
        or receipt.get("candidate_count") != len(all_seeds(bank))
        or receipt.get("seeds_sha256") != canonical_sha(sorted(all_seeds(bank)))
        or receipt.get("collisions") != []
        or scan.get("hits") != []
        or not scan.get("positive_controls")
        or scan.get("file_count", 0) <= 0
        or len(history.get("frp_exclusions", [])) < 7
        or not history.get("sealed_coverage", {}).get("bank_sha256")
    ):
        raise SeedAuditFailed("missing, stale or incomplete E16 seed audit")


def require_committed_registration(repo: Path = ROOT) -> str:
    """Refuse real allocation until the exact registration bytes are committed."""
    relative = "research/bl_e16/REGISTRATION.md"
    try:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=repo, capture_output=True, check=True
        ).stdout
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise SeedAuditFailed("registration must be committed before allocation") from error
    if (repo / relative).read_bytes() != committed:
        raise SeedAuditFailed("registration differs from committed bytes")
    return head


def main(argv: list[str] | None = None) -> int:
    """Allocate only after the registration commit; export create-only sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args(argv)
    commit = require_committed_registration()
    value = materialize(args.round)
    names = [args.out / name for name in ("BANK.json", "SEED_AUDIT.json")]
    if any(path.exists() for path in names):
        raise FileExistsError("allocation exports are create-only")
    receipt = audit(value, excludes=names)
    receipt["registration_commit"] = commit
    from research.bl_ppo.exports import export_json

    args.out.mkdir(parents=True, exist_ok=True)
    export_json(names[0], value)
    export_json(names[1], receipt)
    print(
        json.dumps({"bank": str(names[0]), "audit": "CLEAR", "seed_count": len(all_seeds(value))})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
