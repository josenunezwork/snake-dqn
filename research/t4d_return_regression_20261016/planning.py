"""Bounded single-thread ripgrep freshness audit for T4d collection namespaces."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


def concrete_hits(root: Path, tokens: Sequence[str], *, words: bool) -> set[str]:
    """Search visible non-binary, non-ignored files, failing closed on I/O/timeout."""
    if not root.is_dir() or not tokens:
        raise ValueError("audit requires an existing root and nonempty tokens")
    command = [
        "rg",
        "--no-config",
        "--no-ignore-global",
        "--no-require-git",
        "--threads",
        "1",
        "--fixed-strings",
        "--only-matching",
        "--no-filename",
        "--no-heading",
        "--color",
        "never",
    ]
    if words:
        command.append("--word-regexp")
    for token in tokens:
        command.extend(["-e", token])
    command.extend(["--", str(root)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"T4d freshness scan failed: {result.stderr[-2000:]}")
    return set(result.stdout.splitlines())


def plan(
    student: Path,
    out: Path,
    namespace: str,
    worlds: int,
    mixes: list[str],
    *,
    smoke_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Bind the student, fresh seeds and authoritative evaluation rosters before play."""
    from research.b8_death_census_20261014 import census
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters, sha256
    from research.t4_expert_iteration_20261015.collect_for_student import excluded_seeds

    student, out = student.resolve(), out.resolve()
    if not student.is_file() or worlds < 1 or not mixes:
        raise ValueError("existing student, positive world count and mixes required")
    if out.exists() and any(out.iterdir()):
        raise ValueError("fresh collection output must be empty")
    roots = (Path(__file__).resolve().parents[2], census.ARTIFACTS)
    for root in (() if smoke_seeds is not None else roots):
        if concrete_hits(root, [namespace], words=False):
            raise ValueError(f"stale T4d namespace under {root}")
    exclusions = excluded_seeds(worlds)
    if smoke_seeds is not None and (
        len(smoke_seeds) != worlds or len(set(smoke_seeds)) != worlds or worlds > 2
    ):
        raise ValueError("smoke seeds must specify at most two unique fixture worlds")
    seeds: list[int] = list(smoke_seeds or [])
    rejected: dict[str, list[str]] = {}
    cursor = 0
    while len(seeds) < worlds:
        if cursor >= max(64, worlds * 8):
            raise RuntimeError("fresh-seed audit budget exhausted; no collection started")
        batch_size = max(16, worlds)
        candidates = list(
            dict.fromkeys(
                int.from_bytes(hashlib.sha256(f"{namespace}:{index}".encode()).digest()[:4], "big")
                for index in range(cursor, cursor + batch_size)
            )
        )
        cursor += batch_size
        hits = {
            str(root): concrete_hits(root, [str(s) for s in candidates], words=True)
            for root in roots
        }
        for seed in candidates:
            reasons = [root for root, found in hits.items() if str(seed) in found]
            reasons.extend(
                f"generated:{key}" for key, values in exclusions.items() if seed in values
            )
            if reasons:
                rejected[str(seed)] = reasons
            elif seed not in seeds:
                seeds.append(seed)
                if len(seeds) == worlds:
                    break
    chosen_mixes = [mixes[i % len(mixes)] for i in range(worlds)]
    value = {
        "schema": "t4d-return-round/v1",
        "namespace": namespace,
        "student": {"path": str(student), "sha256": sha256(student)},
        "seed_audit": {
            "method": (
                "explicit-smoke-fixture-no-freshness-claim"
                if smoke_seeds is not None
                else "sha256-namespace-counter + rg-fixed-word/threads1/timeout60"
            ),
            "roots": [str(root) for root in roots],
            "selected": seeds,
            "rejected": rejected,
            "excluded_namespace_overlap": {
                name: sorted(set(seeds) & values) for name, values in exclusions.items()
            },
        },
        "plans": planned_rosters(gid._context(1), seeds, chosen_mixes),
        "played_policy": "student",
        "continuation_policy": "student",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "namespace.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value
