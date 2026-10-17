"""Mocked interpreter selection independent of Colab, downloads, and CUDA."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from research.t16_colab import environment


@pytest.mark.parametrize("kernel_version", [(3, 10, 0), (3, 12, 0), (3, 13, 15)])
def test_uv_selects_managed_312_without_kernel_packages(tmp_path, monkeypatch, kernel_version):
    monkeypatch.setattr(environment.sys, "executable", "/kernel/python")
    monkeypatch.setattr(environment.sys, "version_info", kernel_version)
    python = str(tmp_path / "venv" / "bin" / "python")
    info = {"python": "3.12.12", "python_executable": python, "isolated": True}
    run_owned = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(info)))

    selected, actual = environment.create_environment(tmp_path / "venv", run_owned)

    assert (selected, actual) == (python, info)
    commands = [call.args[0] for call in run_owned.call_args_list]
    assert commands[:3] == [
        ["/kernel/python", "-m", "pip", "install", "uv"],
        ["/kernel/python", "-m", "uv", "python", "install", "3.12"],
        [
            "/kernel/python",
            "-m",
            "uv",
            "venv",
            "--python",
            "3.12",
            "--managed-python",
            "--seed",
            str(tmp_path / "venv"),
        ],
    ]
    assert commands[3][0] == python
    assert run_owned.call_args.kwargs == {"capture_output": True}
    assert all("--system-site-packages" not in command for command in commands)


@pytest.mark.parametrize(
    "version,isolated", [("3.13.15", True), ("3.11.14", True), ("3.12.12", False)]
)
def test_wrong_interpreter_is_refused(tmp_path, version, isolated):
    info = {"python": version, "python_executable": "wrong", "isolated": isolated}
    run_owned = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(info)))
    with pytest.raises(RuntimeError, match="Expected isolated uv Python 3.12"):
        environment.create_environment(tmp_path, run_owned)


def test_uv_failure_does_not_fall_back_to_kernel(tmp_path):
    run_owned = Mock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
    with pytest.raises(subprocess.CalledProcessError):
        environment.create_environment(tmp_path, run_owned)
    assert run_owned.call_count == 1


@pytest.mark.parametrize(
    "p0_failure,prior_omission",
    [
        (None, None),
        ("timeout_all", None),
        ("called_all", None),
        ("called_once", None),
        ("tempdir_once", None),
        (None, False),
        (None, True),
    ],
)
def test_notebook_routes_workload_to_selected_interpreter(
    tmp_path, monkeypatch, p0_failure, prior_omission
):
    """Execute the actual orchestration cell with mocked Colab and all external commands."""
    import sys
    from types import ModuleType

    notebook = Path(__file__).resolve().parents[1] / "notebooks/colab_throughput_smoke.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = "".join(cells[2]["source"])
    # Retain the real orchestration after definitions, but replace only its process wrapper.
    import ast

    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef) or node.name not in {"run_owned", "sha256"}
    ]
    colab = ModuleType("google.colab")
    colab.drive = Mock()
    colab.files = Mock()
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The smoke notebook probes the host GPU before installing its interpreter.
    # Supply a deterministic capability so this fixture also runs on CPU hosts.
    real_subprocess_run = subprocess.run

    def fake_subprocess_run(command, *args, **kwargs):
        if command[:2] == ["nvidia-smi", "--query-gpu=compute_cap"]:
            return subprocess.CompletedProcess(command, 0, "8.0\n", "")
        return real_subprocess_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    selected = "/managed-312/bin/python"
    select = Mock(return_value=(selected, {"python": "3.12.12"}))
    monkeypatch.setattr(environment, "create_environment", select)

    def local_path(value):
        value = str(value)
        if value == "/dev/shm":
            return tmp_path / "dev-shm"
        return tmp_path / value.removeprefix("/content/")

    monkeypatch.setattr("pathlib.Path", local_path)
    archive = local_path("/content/archive")
    archive.write_bytes(b"archive")
    local_path("/dev/shm").mkdir(parents=True)
    commit = "a" * 40
    calls = []
    if prior_omission is not None:
        output = local_path("/content/results")
        output.mkdir()
        (output / "identity.json").write_text(json.dumps({"omitted_smoke03_row5": prior_omission}))

    p0_attempts = 0
    tempdir_attempts = 0
    original_temporary_directory = tempfile.TemporaryDirectory

    if p0_failure == "tempdir_once":

        def fail_first_temporary_directory(*args, **kwargs):
            nonlocal tempdir_attempts
            tempdir_attempts += 1
            if tempdir_attempts == 1:
                raise OSError("P0 test ring directory unavailable")
            return original_temporary_directory(*args, **kwargs)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", fail_first_temporary_directory)

    def fake_run(command, **kwargs):
        nonlocal p0_attempts
        calls.append(command)
        if any(str(part).endswith("bench_learner.py") for part in command):
            p0_attempts += 1
            owned_ring = Path(command[command.index("--ring-dir") + 1])
            (owned_ring / "m3c-bench-test.bin").write_bytes(b"owned mmap stand-in")
            assert kwargs["deadline_at"] <= namespace["deadline"]
            if p0_failure == "timeout_all":
                raise TimeoutError("P0 test timeout after owned group cleanup")
            if p0_failure == "called_all" or (p0_failure == "called_once" and p0_attempts == 1):
                raise subprocess.CalledProcessError(1, command)
        if "research/t16_colab/run_smoke.py" in command:
            (local_path("/content/results") / "invocation_result.json").write_text("{}")
        if command[:2] == [selected, "-c"] and "get_device_name(0)" in command[2]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"torch": "2.9.1", "cuda": "12.6", "gpu": "test", "capability": [8, 0]}),
            )
        if command[:3] == ["df", "-h", "/dev/shm"]:
            return subprocess.CompletedProcess(command, 0, "Filesystem  Size\n")
        return subprocess.CompletedProcess(command, 0, commit + "\n")

    namespace = {
        "DRIVE_ROOT": "/content/drive",
        "RESULTS_DIR": "/content/results",
        "INPUT_ARCHIVE": "/content/archive",
        "EXPECTED_ARCHIVE_SHA256": "hash",
        "TOTAL_BUDGET_SECONDS": 3000,
        "PINNED_COMMIT": commit,
        "REPOSITORY": "repo",
        "CUDA_WHEEL_INDEX": "https://download.pytorch.org/whl/cu126",
        "TIMING_SECONDS": 150,
        "MATRIX": "smoke03",
        "SCRATCH_DIR": "/content/scratch",
        "run_owned": fake_run,
        "sha256": lambda path: "hash",
        "display": Mock(),
    }
    exec(compile(tree, str(notebook), "exec"), namespace)
    select.assert_called_once()
    workloads = [command for command in calls if command[0] == selected]
    if p0_failure:
        assert any("research/t16_colab/run_smoke.py" in command for command in calls)
        assert not list(local_path("/dev/shm").glob("t17b-p0-*"))
        assert not list(local_path("/content/scratch/p0-rings").glob("t17b-p0-*"))
        receipt = json.loads(
            (
                local_path("/content/results") / "p0_learner_bench" / commit / "p0_receipt.json"
            ).read_text()
        )
        assert receipt["status"] == "completed_with_failures"
        assert len(receipt["benchmarks"]) == 8
        expected_failures = 1 if p0_failure in {"called_once", "tempdir_once"} else 8
        failures = [row for row in receipt["benchmarks"] if row["status"] == "failed"]
        successes = [row for row in receipt["benchmarks"] if row["status"] == "completed"]
        assert len(failures) == expected_failures
        assert len(successes) == 8 - expected_failures
        if p0_failure == "timeout_all":
            assert all("P0 test timeout" in row["error"] for row in failures)
        elif p0_failure == "tempdir_once":
            assert all("P0 test ring directory unavailable" in row["error"] for row in failures)
        else:
            assert all("CalledProcessError" in row["error"] for row in failures)
    else:
        receipt = json.loads(
            (
                local_path("/content/results") / "p0_learner_bench" / commit / "p0_receipt.json"
            ).read_text()
        )
        assert receipt["status"] == "completed"
        assert len(receipt["benchmarks"]) == 8
        assert all(row["status"] == "completed" for row in receipt["benchmarks"])
    expected_workloads = 13 if p0_failure == "tempdir_once" else 14
    assert len(workloads) == expected_workloads  # setup, P0 runtime probe, P0 benches, smoke runner
    assert all(command[0] == selected for command in workloads)
    assert all(
        pin in workloads[0] for pin in ["numpy==1.26.4", "numba==0.68.0", "llvmlite==0.50.0"]
    )
    assert "torch==2.9.1" in workloads[1]
    assert any("get_device_name(0)" in command[2] for command in workloads if command[1] == "-c")
    p0_calls = [
        command
        for command in workloads
        if any(str(part).endswith("bench_learner.py") for part in command)
    ]
    expected_p0_calls = 7 if p0_failure == "tempdir_once" else 8
    assert len(p0_calls) == expected_p0_calls
    assert {command[command.index("--gather-threads") + 1] for command in p0_calls} == {"0", "3"}
    ring_dirs = [Path(command[command.index("--ring-dir") + 1]) for command in p0_calls]
    expected_shm_calls = 3 if p0_failure == "tempdir_once" else 4
    assert sum(path.parent == local_path("/dev/shm") for path in ring_dirs) == expected_shm_calls
    assert sum(path.parent == local_path("/content/scratch/p0-rings") for path in ring_dirs) == 4
    assert all(path.name.startswith("t17b-p0-") for path in ring_dirs)
    assert all(command[command.index("--devices") + 1] == "cuda" for command in p0_calls)
    assert ["df", "-h", "/dev/shm"] in calls
    runner = workloads[-1]
    assert "--matrix" in runner and runner[runner.index("--matrix") + 1] == "smoke03"
    assert "--scratch-dir" in runner and "--budget-seconds" in runner
    assert ("--omit-smoke03-row5" in runner) == (prior_omission is not False)
    assert (
        local_path("/content/results") / "p0_learner_bench" / commit / "identity.json"
    ).is_file()
    assert (
        local_path("/content/results") / "p0_learner_bench" / commit / "p0_receipt.json"
    ).is_file()
    assert not list(local_path("/dev/shm").glob("t17b-p0-*"))
    assert not list(local_path("/content/scratch/p0-rings").glob("t17b-p0-*"))
    colab.files.download.assert_not_called()
