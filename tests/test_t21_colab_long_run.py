"""Static contracts for the owner-run T21 recovery notebook."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_long_run.ipynb"
DOC = REPO / "research" / "t16_colab" / "long_run.md"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_long_run_notebook_is_valid_and_separate_from_smoke() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert NOTEBOOK.name != "colab_throughput_smoke.ipynb"
    assert "TO BE PRE-REGISTERED" in _source()
    assert "PRE_REGISTERED_RECIPE_ARGS = [" in _source()


def test_long_run_uses_isolated_py312_and_capability_selected_torch() -> None:
    source = _source()
    assert "['uv', 'venv', '--seed', '--python', '3.12'" in source
    assert "--query-gpu=compute_cap" in source
    assert "compute_capability >= 10.0" in source
    assert "https://download.pytorch.org/whl/cu128" in source
    assert "https://download.pytorch.org/whl/cu126" in source


def test_long_run_auto_resumes_only_verified_drive_snapshots() -> None:
    source = _source()
    assert "drive.mount('/content/drive')" in source
    assert "discover_snapshots(Path(" in source
    assert "verify=True" in source
    assert "--resume-full-state" in source
    assert "resume_args = ['--resume-full-state', str(durable_ckpt_dir)] if snapshots else []" in source
    assert "A named Drive snapshot exists but none verify; recovery is required." in source


def test_long_run_passes_the_async_full_state_contract() -> None:
    source = _source()
    for flag in (
        "--full-state-every",
        "--full-state-replay",
        "--ckpt-local-dir",
        "--ckpt-durable-dir",
    ):
        assert flag in source
    assert "FULL_STATE_REPLAY not in {'none', 'full'}" in source
    assert "REQUESTED_TRANSITIONS must be a positive registered integer." in source


def test_long_run_never_releases_while_work_or_durability_is_unconfirmed() -> None:
    source = _source()
    assert "if completed.returncode:" in source
    assert "summary.get('transitions', -1) < REQUESTED_TRANSITIONS" in source
    assert "summary.get('halt') is not None" in source
    assert "summary.get('stopped_for_time')" in source
    assert "summary.get('full_state_complete') is not True" in source
    assert "No verified durable snapshot after successful training" in source
    assert "manifests[0].get('transitions', -1) >= summary['transitions']" in source
    assert "manifests[0].get('transitions') >= REQUESTED_TRANSITIONS" in source
    assert "if RELEASE_AFTER_SUCCESS:" in source
    assert source.index("if RELEASE_AFTER_SUCCESS:") > source.index("durable final snapshot confirmed:")
    assert source.count("runtime.unassign()") == 1


def test_long_run_documentation_explains_replay_size_and_no_replay_resume() -> None:
    text = DOC.read_text()
    assert "resumed-without-replay" in text
    assert "files[].size_bytes" in text
    assert "--full-state-replay none|full" in text
