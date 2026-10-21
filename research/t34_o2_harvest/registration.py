"""Create and validate the immutable O2 registration before any study data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.t34_o2_harvest.bank import PACKAGE, ROOT, validate_bank

O1_DECISION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/"
    "census-02-extracted/decision.json"
)
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
CAPS: dict[str, Any] = {
    "units": 24.0,
    "reserve_units": 4.8,
    "ordinary_units": 19.2,
    "l4_rate_units_per_hour": 2.0,
    "jobs": 3,
    "units_per_job": 8.0,
    "reserve_units_per_job": 1.6,
    "ordinary_units_per_job": 6.4,
    "max_job_seconds": 14400,
    "ordinary_job_seconds": 11520,
    "workers_per_job": 11,
    "export_bytes_per_job": 67108864,
}
CONTROLLERS = {
    "bare": "B8_s3",
    "d": "rescue_only_existing_t34_pathb_v2_ranking",
    "d_plus_o": "rescue_priority_then_o1_top_harvest_branch",
    "o1_threshold": {"delta_A500": 5.0, "death_drop_attributed": True},
    "branch_budget": "same_total_budget_as_d",
    "branch_horizon": 500,
    "time_roots": [1000, 2000, 3000, 4000],
    "online_event_limit": 4,
    "event_dedupe_decisions": 64,
    "event_manhattan_radius": 30,
    "max_branch_rows_per_assisted_world": 24000,
    "harvest_rank": "survivor_mass_desc_then_bare_then_action; top_must_qualify",
}


def sha256(path: Path) -> str:
    """Return the full SHA-256 digest for a regular file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other JSON roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    """Write canonical JSON once; registrations must never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def source_paths(repo: Path = ROOT) -> list[str]:
    """Resolve the executable closure from AST imports plus declared rule inputs."""
    from research.t34_pathb.source_binding_v2 import _imports, _resolve_import

    pending = [
        f"research.t34_o2_harvest.{name}"
        for name in (
            "runner",
            "analyze",
            "oc",
            "preflight",
            "registration",
            "make_manifest",
            "export",
        )
    ]
    modules: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        modules.add(module)
        for path in _resolve_import(repo, module):
            if path not in paths:
                paths.add(path)
                pending.extend(_imports(repo, path))
    for name in (
        "research/t34_o2_harvest/PREREGISTRATION_O2.md",
        "research/t34_o2_harvest/BANK_O2.json",
        "research/t34_o2_harvest/SEED_AUDIT_O2.json",
        "research/t34_o2_harvest/SEED_AUDIT_MAIN_UPDATE.json",
        "research/t34_o2_harvest/SEED_AUDIT_FINAL_MAIN.json",
        "research/t34_o2_harvest/OC.json",
        "research/t34_o2_harvest/launch.sh",
        "research/survival_band_v2_20261006/survival_pools_20261006.json",
        "research/t34_o1_harvest/REGISTRATION_O1.json",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "research/colab_cli/QUALIFICATION.md",
        "research/apex_safety_20260926/deployment.yaml",
        "docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
    ):
        path = repo / name
        if not path.is_file():
            raise ValueError(f"required registration input is absent: {name}")
        paths.add(path)
    paths.update(path for path in repo.glob("configs/**/*.yaml") if path.is_file())
    return sorted(path.relative_to(repo).as_posix() for path in paths)


def _require_o1_go(path: Path = O1_DECISION) -> dict[str, Any]:
    """Bind the external O1 terminal decision without displaying its payload."""
    if not path.is_file():
        raise ValueError("O1 decision artifact is absent")
    decision = read_json(path)
    if decision.get("schema") != "t34-o1-decision/v1" or decision.get("status") != "GO":
        raise ValueError("O1 decision artifact is not GO")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "schema": "t34-o1-decision/v1",
        "status": "GO",
    }


def _git_blob(repo: Path, commit: str, name: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo, timeout=20)


def freeze(path: Path, *, source_commit: str, repo: Path = ROOT) -> None:
    """Create the pre-data O2 registration from a committed implementation tree."""
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("source_commit must be a full lowercase SHA")
    bank_path, audit_path, oc_path = (
        PACKAGE / "BANK_O2.json",
        PACKAGE / "SEED_AUDIT_O2.json",
        PACKAGE / "OC.json",
    )
    bank, audit, oc = read_json(bank_path), read_json(audit_path), read_json(oc_path)
    validate_bank(bank)
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != sha256(bank_path):
        raise ValueError("O2 seed audit is missing, non-clear, or does not bind its bank")
    if oc.get("schema") != "t34-o2-oc/v1":
        raise ValueError("O2 operating-characteristics result is missing")
    names = source_paths(repo)
    hashes = {name: sha256(repo / name) for name in names}
    # The complete closure (rules and data as well as Python) must be present
    # at the pinned commit because source.prepare ships that exact sparse tree.
    for name in names:
        if hashlib.sha256(_git_blob(repo, source_commit, name)).hexdigest() != hashes[name]:
            raise ValueError(f"implementation differs from source_commit: {name}")
    write_once(
        path,
        {
            "schema": "t34-o2-harvest-registration/v1",
            "frozen_before_data": True,
            "authority": "O1 GO; JUDGMENT O2 conditional design",
            "source_commit": source_commit,
            "source_hashes": hashes,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "o1_go_decision": _require_o1_go(),
            "bank": bank,
            "bank_sha256": sha256(bank_path),
            "seed_audit": {
                "path": "research/t34_o2_harvest/SEED_AUDIT_O2.json",
                "sha256": sha256(audit_path),
            },
            "oc": {"path": "research/t34_o2_harvest/OC.json", "sha256": sha256(oc_path)},
            "horizon": 5000,
            "study_worlds": 96,
            "mix_counts": {"scripted": 32, "frozen": 32, "mixed": 32},
            "one_look": {"looks": 1, "multiplicity": "Holm over D+O-D and D+O-B8"},
            "controllers": CONTROLLERS,
            "caps": CAPS,
            "dispatch_authorized": False,
        },
    )


def require_registration(path: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    """Fail closed if O2 source, O1 authority, bank, OC, or frozen rules drift."""
    value = read_json(path)
    required = {
        "schema": "t34-o2-harvest-registration/v1",
        "frozen_before_data": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "horizon": 5000,
        "study_worlds": 96,
        "mix_counts": {"scripted": 32, "frozen": 32, "mixed": 32},
        "one_look": {"looks": 1, "multiplicity": "Holm over D+O-D and D+O-B8"},
        "controllers": CONTROLLERS,
        "caps": CAPS,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("not the frozen O2 authority")
    validate_bank(value["bank"])
    audit = read_json(PACKAGE / "SEED_AUDIT_O2.json")
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != sha256(
        PACKAGE / "BANK_O2.json"
    ):
        raise ValueError("O2 seed audit differs")
    if value.get("bank_sha256") != sha256(PACKAGE / "BANK_O2.json"):
        raise ValueError("registration bank digest differs")
    if value.get("seed_audit", {}).get("sha256") != sha256(PACKAGE / "SEED_AUDIT_O2.json"):
        raise ValueError("registration seed audit differs")
    if value.get("oc", {}).get("sha256") != sha256(PACKAGE / "OC.json"):
        raise ValueError("registration operating-characteristics result differs")
    # The O1 artifact is a Mac-side external input and deliberately unavailable
    # in a Colab runtime.  Freeze verifies its bytes; runtime validates only
    # its immutable identity recorded in this registration.
    o1 = value.get("o1_go_decision")
    if (
        not isinstance(o1, dict)
        or o1.get("schema") != "t34-o1-decision/v1"
        or o1.get("status") != "GO"
    ):
        raise ValueError("registration lacks frozen O1 GO authority")
    if not isinstance(o1.get("sha256"), str) or len(o1["sha256"]) != 64:
        raise ValueError("registration O1 GO digest is malformed")
    names = source_paths(repo)
    if set(value.get("source_hashes", {})) != set(names):
        raise ValueError("source inventory differs")
    for name, digest in value["source_hashes"].items():
        if sha256(repo / name) != digest:
            raise ValueError(f"source hash differs: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PACKAGE / "REGISTRATION_O2.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        require_registration(args.out)
    else:
        if not args.source_commit:
            parser.error("--source-commit is required to freeze the registration")
        freeze(args.out, source_commit=args.source_commit)


if __name__ == "__main__":
    main()
