"""Create-only pre-data O1 registration and executable/source identity checks."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from research.t34_round0.common import B8_SHA, head, read_json, sha256, write_once

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research/t34_o1_harvest"
__all__ = ["ROOT", "PACKAGE", "B8_SHA", "sha256", "read_json", "write_once", "require_registration"]
CAPS = {
    "units": 8,
    "runtime_seconds": 14400,
    "reserve_units": 1.6,
    "ordinary_seconds": 11520,
    "runner_seconds": 10320,
    "timing_seconds": 1800,
    "timing_units": 1,
    "workers": 11,
    "export_bytes": 67108864,
    "rate_units_per_hour": 2,
}


def source_paths(repo: Path = ROOT) -> list[str]:
    """Reuse the retained AST resolver for an O1-rooted transitive closure."""
    from research.t34_pathb.source_binding_v2 import _imports, _resolve_import

    pending = [
        "research.t34_o1_harvest." + name
        for name in ("runner", "analyze", "registration", "make_manifest", "export")
    ]
    modules, paths = set(), set()
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        modules.add(module)
        for path in _resolve_import(repo, module):
            if path not in paths:
                paths.add(path)
                pending.extend(_imports(repo, path))
    paths.update(repo.glob("configs/**/*.yaml"))
    paths.update(repo.glob("configs/**/*.yml"))
    paths.update((repo / "research/colab_cli").glob("*.py"))
    paths.update((repo / "research/colab_cli").glob("*.sh"))
    for name in (
        "research/t34_o1_harvest/PREREGISTRATION_O1.md",
        "research/t34_o1_harvest/BANK_O1.json",
        "research/t34_o1_harvest/SEED_AUDIT_O1.json",
        "research/t34_o1_harvest/SEED_AUDIT_MAIN_UPDATE.json",
        "research/t34_o1_harvest/launch.sh",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "research/colab_cli/QUALIFICATION.md",
        "research/apex_safety_20260926/deployment.yaml",
        "docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
    ):
        paths.add(repo / name)
    return sorted(path.relative_to(repo).as_posix() for path in paths)


def freeze(path: Path) -> None:
    """Bind published code, the audited bank, rule and caps before gameplay."""
    from research.t34_o1_harvest.bank import validate_bank

    bank = read_json(PACKAGE / "BANK_O1.json")
    validate_bank(bank)
    audit = read_json(PACKAGE / "SEED_AUDIT_O1.json")
    if audit.get("status") != "CLEAR" or audit["bank_sha256"] != sha256(PACKAGE / "BANK_O1.json"):
        raise ValueError("seed audit missing or mismatched")
    commit = head(ROOT)
    names = source_paths()
    for name in names:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        import hashlib

        if hashlib.sha256(blob).hexdigest() != sha256(ROOT / name):
            raise ValueError(f"uncommitted source: {name}")
    write_once(
        path,
        {
            "schema": "t34-o1-registration/v1",
            "frozen_before_data": True,
            "authority": "owner O1 instruction; JUDGMENT adopted O1",
            "source_commit": commit,
            "source_hashes": {n: sha256(ROOT / n) for n in names},
            "checkpoint_sha256": B8_SHA,
            "bank": bank,
            "caps": CAPS,
            "one_look": True,
            "study_worlds": 48,
            "branch_horizon": 500,
            "death_attribution": "death_delta_mass_sum >= delta_mass_sum >= 2500",
            "decision": {"worlds": 8, "frozen_mixed": 4},
            "dispatch_authorized": False,
        },
    )


def require_registration(path: Path) -> dict[str, Any]:
    """Refuse altered registration, source, bank, checkpoint identity or rules."""
    value = read_json(path)
    if (
        value.get("schema") != "t34-o1-registration/v1"
        or value.get("frozen_before_data") is not True
        or value.get("caps") != CAPS
        or value.get("checkpoint_sha256") != B8_SHA
        or value.get("one_look") is not True
        or value.get("study_worlds") != 48
        or value.get("branch_horizon") != 500
        or value.get("decision") != {"worlds": 8, "frozen_mixed": 4}
        or value.get("death_attribution") != "death_delta_mass_sum >= delta_mass_sum >= 2500"
    ):
        raise ValueError("not the frozen O1 authority")
    expected = source_paths()
    if set(value["source_hashes"]) != set(expected):
        raise ValueError("source inventory differs")
    for name, digest in value["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"source hash differs: {name}")
    if value["bank"] != read_json(PACKAGE / "BANK_O1.json"):
        raise ValueError("registration bank differs")
    audit = read_json(PACKAGE / "SEED_AUDIT_O1.json")
    if audit["status"] != "CLEAR" or audit["bank_sha256"] != sha256(PACKAGE / "BANK_O1.json"):
        raise ValueError("seed audit differs")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PACKAGE / "REGISTRATION_O1.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        require_registration(args.out)
    else:
        freeze(args.out)


if __name__ == "__main__":
    main()
