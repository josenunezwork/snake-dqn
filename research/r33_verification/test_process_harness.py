"""Read-only table coverage for the current interim process-policy callers."""

from pathlib import Path

import pytest

from research.r33_verification.process_harness import exercise, table


def test_current_callers_share_frozen_process_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One frozen node exercises every finite case through all three callers."""
    for case in table()["cases"]:
        case_state = tmp_path / case["id"]
        case_state.mkdir()
        exercise(case, monkeypatch, case_state)
