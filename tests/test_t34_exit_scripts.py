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
        '  for ((i=1; i<=$#; i++)); do [[ "${!i}" == --out-dir ]] && '
        'j=$((i+1)) && out="${!j}"; done\n'
        '  mkdir -p "${out}/checkpoints"; : > "${out}/B8_s3_final.pth"; exit 0\n'
        "fi\n"
        'for arg in "$@"; do [[ "${arg}" == *lane1* ]] && { '
        '[[ -n "${T34_TEST_ALL_SUCCESS:-}" ]] && exit 0 || exit 7; }; done\n'
        '[[ -n "${T34_TEST_ALL_SUCCESS:-}" ]] && exit 0\n'
        f"echo $$ > {pid_file}\n"
        "exec sleep 30\n"
    )
    fake_python.chmod(0o755)
    runtime, inputs = tmp_path / "runtime", tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "public-inputs.json").write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        COLAB_CLI_ROOT=str(runtime),
        COLAB_INPUT_DIR=str(inputs),
        T34_SECONDS="120",
        T34_WORLDS="2",
        T34_MAX_WORLDS="2",
        T34_ROUND="a6",
        T34_SHARD="0",
        T34_ATTEMPT="0",
        T34_NAMESPACE="t34-test",
    )
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(Path("research/t34_exit_b8") / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode != 0
    assert time.monotonic() - started < 5
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("script", ("bench.sh", "label.sh"))
def test_all_success_workers_are_reaped_without_ps_notfound_errexit(
    tmp_path: Path, script: str
) -> None:
    """A reaped child can disappear between polls without failing the launcher."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *'public_inputs'* ]]; then\n"
        '  for ((i=1; i<=$#; i++)); do [[ "${!i}" == --out-dir ]] && '
        'j=$((i+1)) && out="${!j}"; done\n'
        '  mkdir -p "${out}/checkpoints"; : > "${out}/B8_s3_final.pth"; exit 0\n'
        "fi\nexit 0\n"
    )
    fake_python.chmod(0o755)
    runtime, inputs = tmp_path / "runtime", tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "public-inputs.json").write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        COLAB_CLI_ROOT=str(runtime),
        COLAB_INPUT_DIR=str(inputs),
        T34_SECONDS="120",
        T34_WORLDS="2",
        T34_MAX_WORLDS="2",
        T34_ROUND="a6",
        T34_SHARD="0",
        T34_ATTEMPT="0",
        T34_NAMESPACE="t34-test",
    )
    result = subprocess.run(
        ["bash", str(Path("research/t34_exit_b8") / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ("bench.sh", "label.sh"))
@pytest.mark.parametrize("runtime", (False, True))
def test_policy_and_trigger_arguments_reach_both_launch_paths(
    tmp_path: Path, script: str, runtime: bool
) -> None:
    """Inspect actual child argv, including quoting of the optional adapter."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_log = tmp_path / "argv"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *public_inputs* ]]; then exit 0; fi\n'
        f'printf "%s\\n" "$@" >> "{argv_log}"\n'
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        T34_INPUT_DIR=str(tmp_path / "inputs"),
        T34_OUTPUT_DIR=str(tmp_path / "outputs"),
        T34_POLICY_FACTORY="future.module:factory",
        T34_STUDENT_CHECKPOINT="student weights.pth",
        T34_CHECKPOINT="other.pth",
        T34_CONFIRM_INTERVAL="4",
        T34_CONFIRM_MIN_LENGTH="500",
        T34_LOW_LENGTH_PROBABILITY="0.04",
        T34_DEATH_WINDOW="75",
        T34_MAX_REWINDS="1",
        T34_BETA="0.5",
        T34_WORLDS="2",
        T34_ROUND="r2",
        T34_SHARD="0",
        T34_ATTEMPT="1",
        T34_NAMESPACE="test",
    )
    if runtime:
        env.update(
            COLAB_CLI_ROOT=str(tmp_path / "runtime"), COLAB_INPUT_DIR=str(tmp_path / "catalog")
        )
    else:
        env.pop("COLAB_CLI_ROOT", None)
    result = subprocess.run(
        ["bash", str(Path("research/t34_exit_b8") / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr
    args = argv_log.read_text().splitlines()
    for flag, value in (
        ("--policy-factory", "future.module:factory"),
        ("--confirm-interval", "4"),
        ("--confirm-min-length", "500"),
        ("--low-length-probability", "0.04"),
        ("--death-window", "75"),
        ("--max-rewinds", "1"),
    ):
        assert args[args.index(flag) + 1] == value
    assert args[args.index("--student-checkpoint") + 1].endswith("/student weights.pth")
    assert args[args.index("--checkpoint") + 1].endswith("/other.pth")
    if script == "label.sh":
        assert args[args.index("--beta") + 1] == "0.5"


@pytest.mark.parametrize(
    "round_id,interval,beta", (("r0", "4", "1"), ("r2", "8", "0.5"), ("r3", "8", "0"))
)
def test_label_shell_round_defaults_reach_python(
    tmp_path: Path, round_id: str, interval: str, beta: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_log = tmp_path / "argv"
    fake_python = fake_bin / "python"
    fake_python.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_log}"\n')
    fake_python.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        T34_INPUT_DIR=str(tmp_path / "inputs"),
        T34_OUTPUT_DIR=str(tmp_path / "outputs"),
        T34_ROUND=round_id,
    )
    for key in ("COLAB_CLI_ROOT", "T34_CONFIRM_INTERVAL", "T34_BETA"):
        env.pop(key, None)
    subprocess.run(
        ["bash", "research/t34_exit_b8/label.sh"],
        env=env,
        check=True,
        timeout=8,
        capture_output=True,
    )
    args = argv_log.read_text().splitlines()
    assert args[args.index("--confirm-interval") + 1] == interval
    assert args[args.index("--beta") + 1] == beta
