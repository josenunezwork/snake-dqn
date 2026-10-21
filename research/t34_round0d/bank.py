"""Allocate every Round 0-D identity before data; any collision rejects the whole allocation.

Rows are materialized with Round 0's own roster builder (imported, unchanged), so
the 96/72/72 design, rosters, frozen-slot rotation, v8 opponents and H5000
profile are identical to Round 0 apart from the fresh namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from research.t34_round0.bank import _rows, ordered_rows
from research.t34_round0d.common import COUNTS, PACKAGE, ROOT, read_json, sha256, write_once

__all__ = ["ordered_rows"]

NAMESPACE = "t34-round0d-dense-20261021/v1"
PREFLIGHT_NAMESPACE = "t34-round0d-preflight-20261021/v1"
A5D_NAMESPACE = "t34-round0d-a5d-20261021/v1"
PREFLIGHT_COUNTS = {"scripted": 1, "frozen": 1}
A5D_COUNTS = {"scripted": 12, "frozen": 9, "mixed": 9}
BANK_SCHEMA = "t34-round0d-bank/v1"
AUDIT_SCHEMA = "t34-round0d-seed-audit/v1"
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
if not ARTIFACTS.is_dir():  # worktrees live under <repo>/.claude/worktrees/<name>
    ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
# Live Round 0 outputs are never opened (no-peek rule); its identities are
# checked from BANK_ROUND0.json instead.
LIVE_EXCLUSIONS = ("t34-round0-study-01",)
SCAN_GLOBS = ("*.json", "*.jsonl", "*.md", "*.py", "*.yaml", "*.yml", "*.sh", "*.csv", "*.tsv")


def materialize() -> dict[str, list[dict[str, Any]]]:
    """Return selection, excluded preflight, and excluded A5-D rows (deterministic)."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    return {
        "rows": _rows(NAMESPACE, COUNTS, 5000),
        "excluded_preflight_rows": _rows(PREFLIGHT_NAMESPACE, PREFLIGHT_COUNTS, 500),
        "excluded_a5d_rows": _rows(A5D_NAMESPACE, A5D_COUNTS, 5000),
    }


def _seeds(rows: Iterable[dict[str, Any]], key: str = "world_seed") -> set[int]:
    return {int(row[key]) for row in rows}


def known_banks(repo: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Explicit registered banks that Round 0-D must be disjoint from (P7 §2)."""
    banks: dict[str, dict[str, Any]] = {}
    path = repo / "research/t34_round0/BANK_ROUND0.json"
    value = read_json(path)
    banks["round0"] = {
        "path": str(path.relative_to(repo)),
        "sha256": sha256(path),
        "seeds": _seeds(value["rows"]) | _seeds(value.get("excluded_preflight_rows", [])),
    }
    path = repo / "research/t34_pathb/RETENTION_BANK_V2.json"
    value = read_json(path)
    rows = [*value["terminal_rows"], *value["product_rows"], value["timing_row"]]
    banks["pathb_retention_v2"] = {
        "path": str(path.relative_to(repo)),
        "sha256": sha256(path),
        "seeds": _seeds(rows, "seed"),
    }
    path = repo / "research/phase0_search_20261019/WORLD_BANK.json"
    value = read_json(path)
    phase0 = list(value["selection"])
    for group in ("banks", "fixtures"):
        for item in value.get(group, {}).values():
            phase0.extend(item if isinstance(item, list) else item.get("rows", []))
    banks["phase0"] = {
        "path": str(path.relative_to(repo)),
        "sha256": sha256(path),
        "seeds": {int(row["world_seed"]) for row in phase0 if isinstance(row, dict)},
    }
    return banks


def t3_census_seeds() -> set[int]:
    """The 32 scripted census worlds reused by T3 dense (E1); Mac-only context."""
    from research.t34_pathb.precheck_v2 import _original_worlds

    seeds, _ = _original_worlds()
    return {int(seed) for seed in seeds}


def scan_history(
    candidates: list[int], repo: Path = ROOT, artifacts: Path = ARTIFACTS
) -> dict[str, Any]:
    """One fail-closed literal scan of text history, mirroring Path B's audit.

    Differences from ``research.t34_pathb.bank._audit_candidates`` (not modified):
    CSV/TSV are included in the same pass (Round 0 needed a second tabular audit),
    and the live Round 0 study root is excluded so no Round 0 outcome byte is read.
    """
    roots = [repo / "docs", repo / "research", repo / "src", repo / "scripts", artifacts]
    mandatory = [repo / "docs", repo / "research", artifacts]
    if any(not path.is_dir() for path in mandatory):
        raise RuntimeError("mandatory Round 0-D freshness scan roots unavailable")
    scanned = [path for path in roots if path.is_dir()]
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
    ]
    for pattern in SCAN_GLOBS:
        command.extend(["--glob", pattern])
    for name in LIVE_EXCLUSIONS:
        command.extend(["--glob", f"!{name}/**"])
    for seed in candidates:
        command.extend(["-e", str(seed)])
    hits: dict[str, list[int]] = {}
    for root in scanned:
        result = subprocess.run([*command, str(root)], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"freshness scan failed: {result.stderr[:500]}")
        hits[str(root)] = sorted({int(line.strip()) for line in result.stdout.split()})
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise RuntimeError("historical FRP training exclusion inventories incomplete")
    present = []
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            raise RuntimeError(f"malformed historical exclusions: {path}")
        present.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "count": len(values),
                "collisions": sorted(set(candidates) & set(values)),
            }
        )
    return {
        "method": "deterministic-all-at-once-sha256-v1 (Round 0 _rows)",
        "candidate_count": len(candidates),
        "scan_roots": [str(path) for path in scanned],
        "scan_globs": list(SCAN_GLOBS),
        "excluded_live_study_roots": list(LIVE_EXCLUSIONS),
        "concrete_hits": hits,
        "historical_frp_exclusions": present,
        "coverage": "visible text source/docs/artifacts incl. CSV/TSV + historical exclusions",
        "limits": (
            "opaque binaries, unrecorded external runs, inaccessible history, and the live "
            "Round 0 study root (identities checked via BANK_ROUND0.json) excluded"
        ),
    }


def audit(allocation: dict[str, list[dict[str, Any]]], *, scan: Any = scan_history) -> dict:
    """Return a CLEAR audit or raise; never replaces or rerolls a world."""
    rows = [
        *allocation["rows"],
        *allocation["excluded_preflight_rows"],
        *allocation["excluded_a5d_rows"],
    ]
    seeds = [int(row["world_seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("whole Round 0-D allocation rejected: internal collision")
    banks = known_banks()
    bank_hits = {name: sorted(set(seeds) & item["seeds"]) for name, item in banks.items()}
    try:
        census = t3_census_seeds()
        census_record: dict[str, Any] = {"count": len(census), "available": True}
    except Exception as exc:  # recorded coverage limit, never a silent pass
        census, census_record = set(), {"available": False, "error": type(exc).__name__}
    bank_hits["t3_dense_census"] = sorted(set(seeds) & census)
    # Positive controls: one known seed per explicit bank must be found by the
    # same scan, so a broken scanner cannot certify CLEAR by finding nothing.
    controls = {name: min(item["seeds"]) for name, item in banks.items()}
    history = scan([*seeds, *controls.values()])
    found = {value for values in history["concrete_hits"].values() for value in values}
    missing_controls = {name: seed for name, seed in controls.items() if seed not in found}
    if missing_controls:
        raise RuntimeError(f"freshness scan failed its positive controls: {missing_controls}")
    for root, values in history["concrete_hits"].items():
        history["concrete_hits"][root] = sorted(set(values) - set(controls.values()))
    history["positive_controls"] = controls
    history["candidate_count"] = len(seeds)
    for item in history["historical_frp_exclusions"]:
        item["collisions"] = sorted(set(item["collisions"]) & set(seeds))
    history_hits = {root: values for root, values in history["concrete_hits"].items() if values}
    exclusion_hits = [x for x in history["historical_frp_exclusions"] if x["collisions"]]
    collisions = {
        **{f"bank:{k}": v for k, v in bank_hits.items() if v},
        **{f"text:{k}": v for k, v in history_hits.items()},
        **{f"exclusions:{x['path']}": x["collisions"] for x in exclusion_hits},
    }
    if collisions:
        raise RuntimeError(f"whole Round 0-D allocation rejected: {collisions}")
    return {
        **history,
        "schema": AUDIT_SCHEMA,
        "status": "CLEAR",
        "seeds": seeds,
        "collisions": [],
        "selection_count": len(allocation["rows"]),
        "excluded_preflight_count": len(allocation["excluded_preflight_rows"]),
        "excluded_a5d_count": len(allocation["excluded_a5d_rows"]),
        "explicit_banks": {
            name: {"path": item["path"], "sha256": item["sha256"], "count": len(item["seeds"])}
            for name, item in banks.items()
        },
        "t3_dense_census": census_record,
        "bank_collisions": bank_hits,
    }


def validate_bank(bank: dict[str, Any]) -> None:
    """Refuse any bank that is not the deterministic frozen Round 0-D allocation."""
    if bank.get("schema") != BANK_SCHEMA or bank.get("namespace") != NAMESPACE:
        raise ValueError("wrong Round 0-D bank authority")
    expected = materialize()
    for key, rows in expected.items():
        if bank.get(key) != rows:
            raise ValueError(f"bank {key} differs from the deterministic frozen allocation")
    rows = bank["rows"]
    if len({r["world_seed"] for r in rows}) != 240 or len({r["world_id"] for r in rows}) != 240:
        raise ValueError("bank seeds/world ids are not unique")
    for mix, count in COUNTS.items():
        if sum(r["mix"] == mix for r in rows) != count:
            raise ValueError("bank mix balance differs from registration")


def allocate(bank_path: Path, audit_path: Path) -> dict[str, Any]:
    """Create-only allocation and audit; a collision leaves no bank behind."""
    if bank_path.exists() or audit_path.exists():
        raise FileExistsError("allocation is create-only")
    allocation = materialize()
    record = audit(allocation)
    payload = {
        "schema": BANK_SCHEMA,
        "namespace": NAMESPACE,
        "counts": COUNTS,
        "profile": {
            "base": "promotion-v2-watch-rect",
            "horizon": 5000,
            "hero": "terminal",
            "hero_slot": 0,
            "snakes": 6,
            "hero_veto": "none",
            "frozen_veto": "v8",
            "mechanics_version": 1,
            "roster_builder": "research.t34_round0.bank._rows (unchanged)",
        },
        **allocation,
    }
    write_once(bank_path, payload)
    write_once(audit_path, {**record, "bank_sha256": sha256(bank_path)})
    return payload


def check_audit(bank_path: Path, audit_path: Path) -> dict[str, Any]:
    """Validate the frozen bank, its Mac audit, and current explicit-bank disjointness."""
    bank, record = read_json(bank_path), read_json(audit_path)
    validate_bank(bank)
    seeds = {
        int(r["world_seed"])
        for key in ("rows", "excluded_preflight_rows", "excluded_a5d_rows")
        for r in bank[key]
    }
    for name, item in known_banks().items():
        if seeds & item["seeds"]:
            raise ValueError(f"Round 0-D allocation collides with the current {name} bank")
    if (
        record.get("schema") != AUDIT_SCHEMA
        or record.get("status") != "CLEAR"
        or record.get("collisions") != []
        or record.get("bank_sha256") != sha256(bank_path)
        or any(record.get("concrete_hits", {}).values())
        or set(record.get("seeds", [])) != seeds
        or len(record.get("historical_frp_exclusions", [])) < 7
    ):
        raise ValueError("missing, incomplete, or mismatched Round 0-D collision audit")
    return bank


def fixture_bank(bank: dict[str, Any], key: str) -> dict[str, Any]:
    """Derive an excluded preflight/A5-D bank view; never selection evidence."""
    if key not in ("excluded_preflight_rows", "excluded_a5d_rows"):
        raise ValueError("unknown excluded fixture set")
    return {
        "schema": BANK_SCHEMA,
        "namespace": PREFLIGHT_NAMESPACE if key == "excluded_preflight_rows" else A5D_NAMESPACE,
        "excluded": True,
        "rows": list(bank[key]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=PACKAGE / "BANK_ROUND0D.json")
    parser.add_argument("--audit", type=Path, default=PACKAGE / "SEED_AUDIT_ROUND0D.json")
    parser.add_argument(
        "--preflight-inputs", type=Path, help="write excluded A6 fixture inputs (fresh dir)"
    )
    args = parser.parse_args()
    if args.preflight_inputs is not None:
        bank = check_audit(args.bank, args.audit)
        args.preflight_inputs.mkdir(parents=True, exist_ok=False)
        write_once(
            args.preflight_inputs / "bank.json", fixture_bank(bank, "excluded_preflight_rows")
        )
        write_once(args.preflight_inputs / "audit.json", read_json(args.audit))
        print(json.dumps({"preflight_inputs": str(args.preflight_inputs)}))
        return
    payload = allocate(args.bank, args.audit)
    print(json.dumps({"namespace": payload["namespace"], "rows": len(payload["rows"])}))


if __name__ == "__main__":
    main()
