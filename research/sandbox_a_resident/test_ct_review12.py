"""Packet-11 Opus corrections: VM uv liveness and prefixed host JSON faults."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_resume
from research.sandbox_a_resident.ct_cuda_observed import BOOTSTRAP_LINES
from research.sandbox_a_resident.ct_uv import UV_VERSION_PATTERN

# Current upstream Display snapshots include a target triple, with optional
# hash/date. These Linux pip-wheel shapes are not Mac Homebrew captures.
# https://github.com/astral-sh/uv/blob/0.13.6/crates/uv-cli/src/version.rs
LINUX_PIP_LINES = (
    "uv 0.13.6 (x86_64-unknown-linux-gnu)",
    "uv 0.13.6 (53b0f5d92 2026-10-27 x86_64-unknown-linux-gnu)",
    "uv 0.13.6 (x86_64-unknown-linux-musl)",
)
VERSION_LINES = LINUX_PIP_LINES + (
    "uv 0.13.1",
    "uv 0.13.6 (53b0f5d92 2026-10-27)",
    "uv 0.13.1 (Homebrew 2026-10-08 aarch64-apple-darwin)",
    "uv 0.13.6 (aarch64-apple-darwin)",
)


@pytest.mark.parametrize("line", VERSION_LINES)
def test_uv_version_closed_lines_healthy_fault_twins(line: str) -> None:
    assert UV_VERSION_PATTERN.fullmatch(line)
    ct_resume._scan_job_tail(line + "\n", label="uv-version")
    ct_resume._scan_json_faults({"snapshot": {"job_log_tail": line}}, label="uv-poll")
    for bad in (
        line + " LookupError: injected",
        line + ' {"scientific_fault":true}',
        line + "\nLookupError: injected",
        line.replace("uv ", "uv warning: "),
    ):
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_job_tail(bad, label="uv-version-fault")


@pytest.mark.parametrize(
    "line",
    [
        "uv 0.13.6 (LookupError: injected)",
        "uv 0.13.6 (x86_64-unknown-linux-gnu LookupError)",
        "uv 0.13.6 (53b0f5d92 2026-10-27 arbitrary-text)",
        "uv 0.13.6 (x86_64-unknown-linux-gnu) trailing",
        "uv 0.13.6 (Homebrew arbitrary-text)",
        "uv 0.13.6+unqualified (x86_64-unknown-linux-gnu)",
        "uv 0.13.6+24 (53b0f5d92 2026-10-27 x86_64-unknown-linux-gnu)",
    ],
)
def test_unknown_uv_version_metadata_refuses(line: str) -> None:
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_job_tail(line, label="uv-metadata")


def test_uv_warning_admission_stays_literal() -> None:
    warnings = [line for line in BOOTSTRAP_LINES if line.startswith("warning:")]
    assert len(warnings) == 2
    for line in warnings:
        ct_resume._scan_job_tail(line, label="literal-uv-warning")
        for bad in (line + " LookupError: injected", line.replace("warning:", "warning: changed")):
            with pytest.raises(ct_resume.CTResumeRefused):
                ct_resume._scan_job_tail(bad, label="changed-uv-warning")


@pytest.mark.parametrize("prefix", ["note ", "log ", "2026-10-28T12:00:00Z info: "])
def test_word_prefixed_host_json_healthy_fault_twins(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "supervisor.log"
    for healthy in ('{"elapsed":1.5}', '{"support_status":"VALID"}', '{"scientific_fault":false}'):
        path.write_text(prefix + healthy + "\n")
        ct_resume._scan_supervisor_log(path, job_id=None, generation=None)
    for bad in (
        '{"support_status":"INVALID"}',
        '{"scientific_fault":true}',
        '{"nested":{"scientific_fault":true}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
        '{"LookupError: injected":0}',
        '{"x":1,"x":2}',
        "{not JSON} LookupError: injected",
    ):
        path.write_text(prefix + bad + "\n")
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_supervisor_log(path, job_id=None, generation=None)


@pytest.mark.parametrize("prefix", ["note ", "LookupError: injected "])
def test_host_prefix_is_fault_scanned_even_with_valid_object(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "supervisor.log"
    path.write_text(prefix + '{"elapsed":1.5}\n')
    if "LookupError" in prefix:
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_supervisor_log(path, job_id=None, generation=None)
    else:
        ct_resume._scan_supervisor_log(path, job_id=None, generation=None)


@pytest.mark.parametrize("version", LINUX_PIP_LINES)
def test_real_job_adapter_logs_linux_uv_without_version_gate(tmp_path: Path, version: str) -> None:
    """Execute the actual Bash adapter; stub only its unavailable VM executables."""
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "uv").write_text('#!/bin/sh\nprintf "%s\\n" "$FIXTURE_UV_VERSION"\n')
    (binary / "python").write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ]; then printf "ENTRYPOINT %s\\n" "$2"; fi\n'
    )
    for path in binary.iterdir():
        path.chmod(0o755)
    env = os.environ.copy()
    env.pop("CT_QUALIFIED_UV_VERSION", None)
    env.update(
        PATH=f"{binary}:/usr/bin:/bin",
        COLAB_CLI_ROOT=str(tmp_path),
        COLAB_INPUT_DIR=str(tmp_path / "inputs"),
        SANDBOX_A_SOURCE_COMMIT="a" * 40,
        SANDBOX_CT_CONTROL_JSON="{}",
        FIXTURE_UV_VERSION=version,
    )
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("ct_job.sh"))],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.splitlines() == [
        version,
        "ENTRYPOINT research.sandbox_a_resident.ct_entrypoint",
    ]
    ct_resume._scan_job_tail(version, label="actual-adapter-version")
