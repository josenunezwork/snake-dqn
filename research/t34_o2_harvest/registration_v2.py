"""Freeze a prospective O2 resize without changing any v1 bytes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from research.t34_o2_harvest.bank_v2 import PACKAGE, ROOT, validate_bank
from research.t34_o2_harvest.plan_v2 import caps, optimal_capacity
from research.t34_o2_harvest.registration import (
    CHECKPOINT_SHA256,
    CONTROLLERS,
    _git_blob,
    _require_o1_go,
    read_json,
    sha256,
    write_once,
)


def source_paths(repo: Path = ROOT) -> list[str]:
    """Bind the complete executable closure and all prospective rule inputs."""
    from research.t34_pathb.source_binding_v2 import _imports, _resolve_import

    pending = [
        f"research.t34_o2_harvest.{n}"
        for n in (
            "runner_v2",
            "analyze_v2",
            "registration_v2",
            "bank_v2",
            "power_v2",
            "preflight_v2",
            "make_manifest_v2",
            "export",
            "upload_readiness",
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
    names = (
        "research/t34_o2_harvest/REGISTRATION_O2_V2.md",
        "research/t34_o2_harvest/AMENDMENT_O2_V2.md",
        "research/t34_o2_harvest/REGISTRATION_O2.json",
        "research/t34_o2_harvest/BANK_O2.json",
        "research/t34_o2_harvest/SEED_AUDIT_O2.json",
        "research/t34_o2_harvest/BANK_O2_V2.json",
        "research/t34_o2_harvest/SEED_AUDIT_O2_V2.json",
        "research/t34_o2_harvest/OC_V2.json",
        "research/t34_o2_harvest/launch_v2.sh",
        "research/t34_o1_harvest/REGISTRATION_O1.json",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "research/colab_cli/QUALIFICATION.md",
        "research/apex_safety_20260926/deployment.yaml",
        "docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
    )
    for name in names:
        path = repo / name
        if not path.is_file():
            raise ValueError(f"required V2 input missing: {name}")
        paths.add(path)
    paths.update(path for path in repo.glob("configs/**/*.yaml") if path.is_file())
    # Power evidence is bound by its explicit repository file identity too.
    oc = read_json(repo / "research/t34_o2_harvest/OC_V2.json")
    for item in [oc.get("source", {})]:
        if isinstance(item, dict) and "path" in item:
            path = repo / item["path"]
            if path.is_file():
                paths.add(path)
    return sorted(p.relative_to(repo).as_posix() for p in paths)


def _authority() -> dict[str, Any]:
    plan = optimal_capacity()
    return {
        "schema": "t34-o2-harvest-registration/v2",
        "frozen_before_data": True,
        "horizon": 5000,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "study_worlds": plan["study_worlds"],
        "mix_counts": plan["mix_counts"],
        "chunks": plan["chunks"],
        "caps": caps(),
        "controllers": CONTROLLERS,
        "one_look": {
            "looks": 1,
            "multiplicity": "Holm over D+O-D and D+O-B8",
            "family_alpha": 0.05,
        },
        "point_bar_b8": 40.0,
        "dispatch_authorized": False,
    }


def freeze(path: Path, *, source_commit: str, repo: Path = ROOT) -> None:
    """Require committed exact sources, collision receipt, and O1 GO."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("full lowercase source commit required")
    bank = read_json(PACKAGE / "BANK_O2_V2.json")
    validate_bank(bank)
    audit = read_json(PACKAGE / "SEED_AUDIT_O2_V2.json")
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != sha256(
        PACKAGE / "BANK_O2_V2.json"
    ):
        raise ValueError("missing V2 collision audit")
    hashes = {name: sha256(repo / name) for name in source_paths(repo)}
    for name, digest in hashes.items():
        if hashlib.sha256(_git_blob(repo, source_commit, name)).hexdigest() != digest:
            raise ValueError(f"source differs from commit: {name}")
    write_once(
        path,
        {
            **_authority(),
            "authority": "Owner prospective resize request 2026-10-21; no dispatch",
            "supersedes": {
                "path": "research/t34_o2_harvest/REGISTRATION_O2.json",
                "sha256": sha256(PACKAGE / "REGISTRATION_O2.json"),
            },
            "source_commit": source_commit,
            "source_hashes": hashes,
            "bank": bank,
            "bank_sha256": sha256(PACKAGE / "BANK_O2_V2.json"),
            "seed_audit": {
                "path": "research/t34_o2_harvest/SEED_AUDIT_O2_V2.json",
                "sha256": sha256(PACKAGE / "SEED_AUDIT_O2_V2.json"),
            },
            "oc": {
                "path": "research/t34_o2_harvest/OC_V2.json",
                "sha256": sha256(PACKAGE / "OC_V2.json"),
            },
            "o1_go_decision": _require_o1_go(),
        },
    )


def require_registration(path: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    """Fail closed on source/rule/bank/OC/receipt drift; do not read O2 outcomes."""
    value = read_json(path)
    if any(value.get(k) != v for k, v in _authority().items()):
        raise ValueError("not the frozen O2 V2 authority")
    validate_bank(value["bank"])
    for key, name in (("bank_sha256", "BANK_O2_V2.json"),):
        if value[key] != sha256(PACKAGE / name):
            raise ValueError("V2 bank digest differs")
    for key, name in (("seed_audit", "SEED_AUDIT_O2_V2.json"), ("oc", "OC_V2.json")):
        if value[key]["sha256"] != sha256(PACKAGE / name):
            raise ValueError(f"V2 {key} digest differs")
    audit = read_json(PACKAGE / "SEED_AUDIT_O2_V2.json")
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != value["bank_sha256"]:
        raise ValueError("V2 seed audit not clear")
    if value.get("o1_go_decision", {}).get("status") != "GO":
        raise ValueError("O1 GO missing")
    if set(value["source_hashes"]) != set(source_paths(repo)):
        raise ValueError("V2 source inventory differs")
    for name, digest in value["source_hashes"].items():
        if sha256(repo / name) != digest:
            raise ValueError(f"source hash differs: {name}")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PACKAGE / "REGISTRATION_O2_V2.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        require_registration(args.out)
    else:
        if not args.source_commit:
            parser.error("--source-commit required")
        freeze(args.out, source_commit=args.source_commit)
