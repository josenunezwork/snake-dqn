"""Packet-13 review regressions for JSON prose and optional uv diagnostics."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_resume


@pytest.mark.parametrize("prefix", ["note ", "note {a} ", 'note {"elapsed":1} '])
@pytest.mark.parametrize("suffix", ["", " tail", ' tail {"elapsed":2}'])
def test_every_embedded_object_healthy_fault_twins(
    tmp_path: Path, prefix: str, suffix: str
) -> None:
    path = tmp_path / "supervisor.log"
    for payload in ('{"support_status":"VALID"}', '{"nested":{"scientific_fault":false}}'):
        path.write_text(prefix + payload + suffix + "\n")
        ct_resume._scan_supervisor_log(path, job_id=None, generation=None)
    for payload in (
        '{"support_status":"INVALID"}',
        '{"nested":{"scientific_fault":true}}',
        '{"x":NaN}',
        '{"x":1e999}',
        '{"x":1,"x":2}',
        '{"x":{"LookupError: injected":0}}',
    ):
        path.write_text(prefix + payload + suffix + "\n")
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_supervisor_log(path, job_id=None, generation=None)


@pytest.mark.parametrize("uv_status", [None, 0, 1])
def test_uv_version_is_optional_but_child_failure_is_preserved(
    tmp_path: Path, uv_status: int | None
) -> None:
    """Missing/failing uv must not prevent the actual adapter invoking its worker."""
    binary = tmp_path / "bin"
    binary.mkdir()
    if uv_status is not None:
        uv = binary / "uv"
        uv.write_text(f'#!/bin/sh\necho "uv 0.13.6"\nexit {uv_status}\n')
        uv.chmod(0o755)
    python = binary / "python"
    python.write_text('#!/bin/sh\nif [ "$1" = "-m" ]; then echo ENTRYPOINT; exit 73; fi\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(Path(__file__).with_name("ct_job.sh"))],
        env={
            **os.environ,
            "PATH": f"{binary}:/usr/bin:/bin",
            "COLAB_CLI_ROOT": str(tmp_path),
            "COLAB_INPUT_DIR": str(tmp_path / "inputs"),
            "SANDBOX_A_SOURCE_COMMIT": "a" * 40,
            "SANDBOX_CT_CONTROL_JSON": "{}",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 73, result.stderr
    assert "ENTRYPOINT" in result.stdout
