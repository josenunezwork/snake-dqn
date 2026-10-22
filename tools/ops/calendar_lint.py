"""Reject calendar windows embedded in executable research launchers."""

from __future__ import annotations

import re
from pathlib import Path

CALENDAR_LITERAL = re.compile(
    r"(?:(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)|\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[0-2]?\d:[0-5]\d\b)"
)

# This is intentionally explicit. Historical prose is outside the executable
# launcher scope; adding a document to a broader future scan requires review.
HISTORICAL_DOC_ALLOWLIST = frozenset(
    {
        "research/colab_cli/REQUALIFICATION.md",
    }
)

# A source exception identifies only an inert historical reference, never an
# entire launcher. New calendar literals in this file (or any other launcher)
# still fail the lint.
HISTORICAL_SOURCE_ALLOWLIST = frozenset(
    {
        (
            "research/perf_compile_20261013/run_bench.sh",
            2,
            "2026-10-13",
            "# perf-compile (2026-10-13): re-runnable torch.compile benchmarks for the ego2s "
            "trainer",
        ),
        (
            "research/t11_local30_20261017/bind_launch.py",
            58,
            "2026-10-17",
            'ROOT / "docs/research/specs/t11_prereg_2026-10-17.md",',
        ),
        (
            "research/t11_local30_20261017/bind_launch.py",
            59,
            "2026-10-17",
            'ROOT / "docs/research/specs/t11_local30_spec_2026-10-17.md",',
        ),
        (
            "research/t4c_20261016/bind_launch.py",
            51,
            "2026-10-16",
            '"docs/research/specs/t4c_prereg_2026-10-16.md",',
        ),
        (
            "research/t4e_return_20261016/run_collection.py",
            40,
            "2026-10-16",
            'PREREG = Path("docs/research/specs/t4e_prereg_2026-10-16.md")',
        ),
        (
            "research/t4e_return_20261016/run_collection.py",
            307,
            "2026-10-16",
            'namespace = f"t4e-return-collection/2026-10-16/{root.name}"',
        ),
    }
)


def launcher_paths(repo: Path) -> list[Path]:
    """Return executable launcher sources, excluding documentation by construction."""
    research = repo / "research"
    candidates = list(research.glob("**/run_*.sh"))
    candidates.extend(research.glob("**/run_*.py"))
    candidates.extend(research.glob("**/*launch*.py"))
    candidates.extend(research.glob("**/launcher*.py"))
    return sorted({path for path in candidates if path.is_file()})


def calendar_literals(path: Path) -> list[tuple[int, str, str]]:
    """Return line-numbered calendar literals and their stripped source lines."""
    findings: list[tuple[int, str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in CALENDAR_LITERAL.finditer(line):
            findings.append((number, match.group(0), line.strip()))
    return findings


def lint(repo: Path) -> list[str]:
    """Return every launcher calendar literal as a stable, displayable diagnostic."""
    findings: list[str] = []
    for path in launcher_paths(repo):
        relative = path.relative_to(repo).as_posix()
        if relative in HISTORICAL_DOC_ALLOWLIST:
            continue
        findings.extend(
            f"{relative}:{line}: hard-coded calendar literal {literal!r}"
            for line, literal, source_line in calendar_literals(path)
            if (relative, line, literal, source_line) not in HISTORICAL_SOURCE_ALLOWLIST
        )
    return findings
