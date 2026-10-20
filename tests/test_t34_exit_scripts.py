"""Failure-order checks for the portable Bash 3 worker supervision loop."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ("bench.sh", "label.sh"))
def test_second_worker_failure_terminates_first_without_sequential_wait(
    tmp_path: Path, script: str
) -> None:
    """A fast lane 1 failure must not wait for lane 0's long running child."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pid_file = tmp_path / "lane0.pid"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$*\" == *'public_inputs'* ]]; then\n"
        "  for ((i=1; i<=$#; i++)); do [[ \"${!i}\" == --out-dir ]] && j=$((i+1)) && out=\"${!j}\"; done\n"
        "  mkdir -p \"${out}/checkpoints\"; : > \"${out}/B8_s3_final.pth\"; exit 0\n"
        "fi\n"
        "for arg in \"$@\"; do [[ \"${arg}\" == *lane1* ]] && { [[ -n \"${T34_TEST_ALL_SUCCESS:-}\" ]] && exit 0 || exit 7; }; done\n"
        "[[ -n \"${T34_TEST_ALL_SUCCESS:-}\" ]] && exit 0\n"
        f"echo $$ > {pid_file}\n"
        "trap 'exit 143' TERM INT\n"
        "sleep 30\n"
    )
    fake_python.chmod(0o755)
    runtime, inputs = tmp_path / "runtime", tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "public-inputs.json").write_text("{}")
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", COLAB_CLI_ROOT=str(runtime), COLAB_INPUT_DIR=str(inputs), T34_SECONDS="120", T34_WORLDS="2", T34_MAX_WORLDS="2", T34_ROUND="a6", T34_SHARD="0", T34_ATTEMPT="0", T34_NAMESPACE="t34-test")
    started = time.monotonic()
    result = subprocess.run(["bash", str(Path("research/t34_exit_b8") / script)], env=env, capture_output=True, text=True, timeout=8)
    assert result.returncode != 0
    assert time.monotonic() - started < 5
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("script", ("bench.sh", "label.sh"))
def test_all_success_workers_are_reaped_without_ps_notfound_errexit(tmp_path: Path, script: str) -> None:
    """A reaped child can disappear between polls without failing the launcher."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *'public_inputs'* ]]; then\n"
        "  for ((i=1; i<=$#; i++)); do [[ \"${!i}\" == --out-dir ]] && j=$((i+1)) && out=\"${!j}\"; done\n"
        "  mkdir -p \"${out}/checkpoints\"; : > \"${out}/B8_s3_final.pth\"; exit 0\n"
        "fi\nexit 0\n"
    )
    fake_python.chmod(0o755)
    runtime, inputs = tmp_path / "runtime", tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "public-inputs.json").write_text("{}")
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", COLAB_CLI_ROOT=str(runtime), COLAB_INPUT_DIR=str(inputs), T34_SECONDS="120", T34_WORLDS="2", T34_MAX_WORLDS="2", T34_ROUND="a6", T34_SHARD="0", T34_ATTEMPT="0", T34_NAMESPACE="t34-test")
    result = subprocess.run(["bash", str(Path("research/t34_exit_b8") / script)], env=env, capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
