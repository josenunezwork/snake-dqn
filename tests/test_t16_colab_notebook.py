"""Static contract checks for the owner-run, unexecuted Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_throughput_smoke.ipynb"


def notebook_source() -> str:
    """Return the concatenated code and markdown source for static contracts."""
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_notebook_is_valid_unexecuted_python() -> None:
    """Keep the published notebook safe to open and run from a clean Colab."""
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_notebook_has_pinned_isolated_and_resumable_contract() -> None:
    """Pin code and inputs rather than relying on Colab's mutable kernel state."""
    source = notebook_source()
    assert "PINNED_COMMIT" in source
    import re

    pins = re.findall(r"PINNED_COMMIT = '([0-9a-f]{40})'", source)
    assert len(pins) == 1
    assert "drive.mount" in source
    assert "create_environment(venv, run_owned)" in source
    assert "--system-site-packages" not in source
    assert "verify_and_extract" in source
    assert "--budget-seconds" in source
    assert "--scratch-dir" in source  # trainer outputs stay off Drive during timing
    assert re.search(r"MATRIX = '(smoke02|smoke03)'", source) and "'--matrix', MATRIX" in source
    assert "files.download" in source
    assert "RUN_NAME = 'smoke-03'" in source
    assert "TOTAL_BUDGET_SECONDS = 4500" in source
    assert "TIMING_SECONDS = 300" in source
    assert "bench_learner.py" in source
    assert "p0_learner_bench" in source
    assert "p0_dir = results_dir / 'p0_learner_bench' / PINNED_COMMIT" in source
    assert "p0_identity_path" in source
    assert "p0_receipt_path = p0_dir / 'p0_receipt.json'" in source
    assert "t17b-p0-learner-bench-receipt/v1" in source
    assert "completed_with_failures" in source
    assert "continuing to smoke matrix" in source
    assert "p0_runtime" in source and "get_device_name(0)" in source
    assert "df', '-h', '/dev/shm'" in source
    assert "t1_k2_c_ro,t1_bf16_k2_c_ro,t1_k4_c_ro,t1_bf16_k4_c_ro" in source
    assert "('--obs-pack', 'packed'), ('--raw-rings', 'raw')" in source
    assert "for gather_threads in (0, 3)" in source
    assert "local_ring_dir.mkdir(parents=True, exist_ok=True)" in source
    assert "p0_deadline = time.monotonic() + 300" in source
    assert "deadline_at=p0_deadline" in source
    assert "remaining_budget = min(3600" in source
    assert "--omit-smoke03-row5" in source
    assert "runner_identity_path = results_dir / 'identity.json'" in source
    assert "prior_identity.get('omitted_smoke03_row5')" in source
    assert "full_matrix_estimated_seconds = 3785" in source
    assert "six_row_estimated_seconds = 3335" in source
    assert "startswith('2.9.')" in source  # preserve the CUDA torch 2.9.x contract


def test_notebook_bounds_and_cleans_only_its_own_subprocess_groups() -> None:
    """All setup commands share the post-auth deadline and private process groups."""
    source = notebook_source()
    assert "deadline = started_after_drive_auth + TOTAL_BUDGET_SECONDS" in source
    assert "timeout = active_deadline - time.monotonic() - cleanup_grace" in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source
    assert "os.killpg(process.pid, signal.SIGKILL)" in source
    assert "cleanup_owned_group(process, cleanup_grace)" in source
    assert "invocation_path.unlink(missing_ok=True)" in source
    assert "run_owned(run, cwd=checkout, check=False, cleanup_grace=12)" in source
