"""Keep time-sensitive schedules out of executable research launchers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINT_PATH = REPO / "tools/ops/calendar_lint.py"
SPEC = importlib.util.spec_from_file_location("ops_calendar_lint", LINT_PATH)
assert SPEC and SPEC.loader
calendar_lint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calendar_lint
SPEC.loader.exec_module(calendar_lint)


def test_repository_launchers_contain_no_calendar_windows() -> None:
    assert calendar_lint.lint(REPO) == []


def test_detects_date_and_weekday_windows_in_launchers(tmp_path: Path) -> None:
    launcher = tmp_path / "research/example/run_job.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# deadline 2026-10-29T20:00\n# Thu 20:00\n", encoding="utf-8")
    python_launcher = tmp_path / "research/example/bind_launch.py"
    python_launcher.write_text("window = 'Mon 09:30'\n", encoding="utf-8")
    python_runner = tmp_path / "research/example/run_job.py"
    python_runner.write_text("deadline = '2026-10-28'\n", encoding="utf-8")

    findings = calendar_lint.lint(tmp_path)

    assert "research/example/run_job.sh:1: hard-coded calendar literal '2026-10-29'" in findings
    assert "research/example/run_job.sh:2: hard-coded calendar literal 'Thu 20:00'" in findings
    assert "research/example/bind_launch.py:1: hard-coded calendar literal 'Mon 09:30'" in findings
    assert "research/example/run_job.py:1: hard-coded calendar literal '2026-10-28'" in findings


def test_historical_docs_are_not_launcher_targets(tmp_path: Path) -> None:
    document = tmp_path / "research/colab_cli/REQUALIFICATION.md"
    document.parent.mkdir(parents=True)
    document.write_text("Historical: 2026-10-22 Thu 20:00\n", encoding="utf-8")

    assert calendar_lint.lint(tmp_path) == []


def test_historical_source_allowlist_is_limited_to_one_literal(tmp_path: Path) -> None:
    launcher = tmp_path / "research/perf_compile_20261013/run_bench.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "# historical benchmark\n"
        "# perf-compile (2026-10-13): re-runnable torch.compile benchmarks for the ego2s trainer\n"
        "# Thu 20:00\n",
        encoding="utf-8",
    )

    assert calendar_lint.lint(tmp_path) == [
        "research/perf_compile_20261013/run_bench.sh:3: hard-coded calendar literal 'Thu 20:00'"
    ]


def test_historical_source_allowlist_rejects_executable_replacement(tmp_path: Path) -> None:
    launcher = tmp_path / "research/perf_compile_20261013/run_bench.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# historical benchmark\nDEADLINE='2026-10-13'\n", encoding="utf-8")

    assert calendar_lint.lint(tmp_path) == [
        "research/perf_compile_20261013/run_bench.sh:2: hard-coded calendar literal '2026-10-13'"
    ]


def test_historical_document_path_allowlist_is_exact(tmp_path: Path) -> None:
    launcher = tmp_path / "research/t11_local30_20261017/bind_launch.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "\n" * 57
        + 'ROOT / "docs/research/specs/t11_prereg_2026-10-17.md",\n'
        + "other = '2026-10-18'\n",
        encoding="utf-8",
    )

    assert calendar_lint.lint(tmp_path) == [
        "research/t11_local30_20261017/bind_launch.py:59: hard-coded calendar literal '2026-10-18'"
    ]
