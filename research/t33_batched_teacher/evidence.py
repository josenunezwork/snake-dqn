"""Local source bindings for reproducible T33 development receipts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def identity() -> dict:
    """Bind tracked and new Python dependency bytes, without remote operations."""
    roots = [
        ROOT / "src",
        *(
            ROOT / "research" / name
            for name in (
                "t33_batched_teacher",
                "phase0_search_20261019",
                "t3_rollout_labels_20261015",
                "b8_death_census_20261014",
                "redesign_scope_20261007",
                "apex_safety_20260926",
            )
        ),
    ]
    files = sorted(p for root in roots for p in root.rglob("*.py"))
    bindings = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "source_sha256": bindings,
    }
