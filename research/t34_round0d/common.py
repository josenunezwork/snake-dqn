"""Portable identities for Round 0-D.

Generic helpers are imported from the hash-bound Round 0 package; nothing in
``research.t34_round0`` is modified. Only study identities (paths, schema and
tag prefixes, the executable closure) are Round 0-D specific.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from research.t34_round0.common import B8_SHA, COUNTS, head, read_json, sha256, write_once

__all__ = [
    "B8_SHA",
    "BINDING_DOCS",
    "COUNTS",
    "PACKAGE",
    "ROOT",
    "TAG_PREFIX",
    "head",
    "read_json",
    "sha256",
    "source_hashes",
    "write_once",
]

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research/t34_round0d"
# Must never match Round 0's ``refs/tags/t34-round0/*`` admission glob.
TAG_PREFIX = "t34-round0d"
LEDGER_DIR = "research/t34_round0d/ledger"
HORIZON = 5000
BINDING_DOCS = (
    "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
    "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
    "docs/research/final_run_20261019/P7_STATISTICS.md",
    "docs/research/final_run_20261019/FINAL_PLAN.md",
    "docs/research/governance_amendment_survival_band_v2_2026-10-06.md",
    "docs/research/specs/2026-10-20_t34_nogo_branch_plan.md",
    "docs/research/t3_ceiling_test_2026-10-15.md",
)
# Round 0 rule documents whose bars Round 0-D copies verbatim; bound read-only.
ROUND0_RULES = (
    "research/t34_round0/PREREGISTRATION_ROUND0.md",
    "research/t34_round0/PREREGISTRATION_ROUND0_FIXES1.md",
    "research/t34_round0/ERRATUM_FIXES1.md",
    "research/t34_round0/ERRATUM_INTEGRITY_SIZING.md",
    "research/t34_round0/OC_ROUND0.json",
    "research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json",
    "docs/research/specs/2026-10-20_t34_round0_boost_floor_amendment.md",
    "docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md",
    "docs/research/advice/2026-10-20_colab_lanes/RULING_round0_go_file.md",
)
PACKAGE_RULES = (
    "research/t34_round0d/PREREGISTRATION_ROUND0D.md",
    "research/t34_round0d/BANK_ROUND0D.json",
    "research/t34_round0d/SEED_AUDIT_ROUND0D.json",
)


def source_hashes(repo: Path = ROOT) -> dict[str, str]:
    """Conservatively bind the tracked executable closure and every governing rule."""
    names = (
        subprocess.run(
            ["git", "ls-files", "-z", "src", "configs", "web/backend", "research"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    selected = {n for n in names if n and Path(n).suffix in {".py", ".sh", ".yaml", ".yml"}}
    selected.update(BINDING_DOCS)
    selected.update(ROUND0_RULES)
    selected.update(PACKAGE_RULES)
    selected.add("research/t34_pathb/RETENTION_BANK_V2.json")
    selected.add("research/t34_round0/BANK_ROUND0.json")
    selected.add("research/phase0_search_20261019/WORLD_BANK.json")
    selected.add("research/colab_cli/QUALIFICATION.md")
    selected.add("research/t34_exit_b8/PUBLIC_INPUTS.json")
    missing = sorted(name for name in selected if not (repo / name).is_file())
    if missing:
        raise FileNotFoundError(f"Round 0-D source closure is incomplete: {missing}")
    return {name: sha256(repo / name) for name in sorted(selected)}
