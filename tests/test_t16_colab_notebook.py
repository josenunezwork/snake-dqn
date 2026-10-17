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
    assert "venv', '--system-site-packages" in source
    assert "verify_and_extract" in source
    assert "--budget-seconds" in source
    assert "--scratch-dir" in source  # trainer outputs stay off Drive during timing
    assert "files.download" in source
    assert "TOTAL_BUDGET_SECONDS = 3000" in source
    assert "startswith('2.9.')" in source  # any CUDA torch 2.9.x; no forced 2.5 GB reinstall


def test_notebook_bounds_and_cleans_only_its_own_subprocess_groups() -> None:
    """All setup commands share the post-auth deadline and private process groups."""
    source = notebook_source()
    assert "deadline = started_after_drive_auth + TOTAL_BUDGET_SECONDS" in source
    assert "timeout = deadline - time.monotonic() - cleanup_grace" in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source
    assert "os.killpg(process.pid, signal.SIGKILL)" in source
    assert "cleanup_owned_group(process, cleanup_grace)" in source
    assert "invocation_path.unlink(missing_ok=True)" in source
    assert "run_owned(run, cwd=checkout, check=False, cleanup_grace=12)" in source
