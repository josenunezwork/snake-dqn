"""Regression guard: DENSITY-2 may change allocation, never the round-14 science."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

BASELINE = "ec7c8a35"
ROOT = Path(__file__).resolve().parents[2]


def _baseline(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)


@pytest.mark.parametrize(
    "path",
    [
        "research/bl_density/METRICS.md",
        "research/bl_density/METRICS_AMENDMENT_01.md",
        "research/bl_density/METRICS_AMENDMENT_02.md",
        "research/bl_density/METRICS_AMENDMENT_03.md",
        "research/bl_density/analysis.py",
        "research/bl_density/statistics.py",
    ],
)
def test_round14_rules_and_thresholds_are_byte_identical_to_spare_slot_baseline(path: str) -> None:
    """The decision implementation remains exactly the ruled round-14 version."""
    assert (ROOT / path).read_bytes() == _baseline(path).encode()


@pytest.mark.parametrize(
    "function",
    [
        "cheap_prefilter",
        "effective_q_features",
        "legal_alternatives",
        "retrospective_doom",
        "_profile_override",
        "_root_geometry",
        "_inject_p1_long",
        "_rollout_continuation",
        "_build",
        "_step",
    ],
)
def test_density_two_retains_frozen_runner_science(function: str) -> None:
    """Permit orchestration edits while pinning the rule-bearing helper ASTs."""
    current = ast.parse((ROOT / "research/bl_density/runner.py").read_text())
    baseline = ast.parse(_baseline("research/bl_density/runner.py"))

    def lookup(tree: ast.Module) -> ast.FunctionDef:
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function
        )

    assert ast.dump(lookup(current), include_attributes=False) == ast.dump(
        lookup(baseline), include_attributes=False
    )
