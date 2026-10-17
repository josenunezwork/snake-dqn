"""Mocked interpreter selection independent of Colab, downloads, and CUDA."""

import json
import subprocess
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


def test_notebook_routes_workload_to_selected_interpreter(tmp_path, monkeypatch):
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
    selected = "/managed-312/bin/python"
    select = Mock(return_value=(selected, {"python": "3.12.12"}))
    monkeypatch.setattr(environment, "create_environment", select)

    def local_path(value):
        value = str(value)
        return tmp_path / value.removeprefix("/content/")

    monkeypatch.setattr("pathlib.Path", local_path)
    archive = local_path("/content/archive")
    archive.write_bytes(b"archive")
    commit = "a" * 40
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "research/t16_colab/run_smoke.py" in command:
            (local_path("/content/results") / "invocation_result.json").write_text("{}")
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
        "SCRATCH_DIR": "/content/scratch",
        "run_owned": fake_run,
        "sha256": lambda path: "hash",
        "display": Mock(),
    }
    exec(compile(tree, str(notebook), "exec"), namespace)
    select.assert_called_once()
    workloads = [command for command in calls if command[0] != "git"]
    assert len(workloads) == 5  # dependencies, CUDA wheel, CUDA probe, bundle, runner
    assert all(command[0] == selected for command in workloads)
    assert all(
        pin in workloads[0] for pin in ["numpy==1.26.4", "numba==0.68.0", "llvmlite==0.50.0"]
    )
    assert "torch==2.9.1" in workloads[1]
    assert "--scratch-dir" in workloads[-1] and "--budget-seconds" in workloads[-1]
    colab.files.download.assert_called_once()
