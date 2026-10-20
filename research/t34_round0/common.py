"""Portable identities and create-only artifacts for Round 0."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"scripted": 96, "frozen": 72, "mixed": 72}
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
PACKAGE = ROOT / "research/t34_round0"
BINDING_DOCS = (
    "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
    "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
    "docs/research/final_run_20261019/P7_STATISTICS.md",
    "docs/research/final_run_20261019/FINAL_PLAN.md",
    "docs/research/governance_amendment_survival_band_v2_2026-10-06.md",
    "docs/research/specs/2026-10-20_t34_round0_prereg.md",
)


def sha256(path: Path) -> str:
    """Hash immutable file bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read finite, unique-key JSON objects only."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON {value}")

    result = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def write_once(path: Path, payload: Any) -> None:
    """Create an artifact without replacing prior evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes(repo: Path = ROOT) -> dict[str, str]:
    """Conservatively bind the tracked executable closure and campaign rules."""
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
    selected.add("research/t34_round0/PREREGISTRATION_ROUND0.md")
    selected.add("research/t34_round0/ERRATUM_A5_ADDENDUM_B.md")
    selected.add("research/t34_round0/BANK_ROUND0.json")
    selected.add("research/t34_round0/SEED_AUDIT_ROUND0.json")
    selected.add("research/t34_round0/TABULAR_SEED_AUDIT.json")
    selected.add("research/t34_pathb/RETENTION_BANK_V2.json")
    selected.add("research/t34_round0/OC_ROUND0.json")
    selected.add("research/colab_cli/QUALIFICATION.md")
    selected.add("research/t34_exit_b8/PUBLIC_INPUTS.json")
    return {name: sha256(repo / name) for name in sorted(selected)}


def head(repo: Path = ROOT) -> str:
    """Return the full current source commit."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
