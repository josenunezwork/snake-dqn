"""Offline create-only tests for the qualified Colab control root wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "colab_control_launch", ROOT / "tools/ops/colab_control_launch.py"
)
assert SPEC is not None and SPEC.loader is not None
launch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launch
SPEC.loader.exec_module(launch)


def admission(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    broker = tmp_path / "broker"
    launcher = broker / "research/colab_cli/launch.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/bash\n")
    manifest = tmp_path / "live.json"
    manifest.write_text("{}\n")
    supervisor_args = [
        "--state-dir",
        str(tmp_path / "state"),
        "--colab",
        "/tmp/colab",
        "run",
        str(manifest),
    ]
    argv = [str(launcher), "--control-nice0", *supervisor_args]
    path = tmp_path / "lane.json"
    path.write_text(
        json.dumps(
            {
                "policy_id": launch.POLICY_ID,
                "ruling_sha256": "2ec03d85be07d4d9264e8aba105567ee23b40e364313c0da91fd57c215b0d64a",
                "operational_pin": "a" * 40,
                "qualified_broker_source_closure": "b" * 40,
                "qualification_record": str(tmp_path / "qualified.json"),
                "qualification_record_sha256": "b" * 64,
                "operational_root": str(tmp_path),
                "dispatch_source_closure": {"/tmp/dispatch.py": "b" * 64},
                "pane_argv": ["/bin/bash", "-c", "frozen", "colab-control-nice0"],
                "bootstrap_argv": [sys.executable, "/tmp/control-helper.py"],
                "source_closure": {"research/colab_cli/supervisor.py": "c" * 64},
                "broker": str(broker),
                "launcher": str(launcher),
                "module": "research.colab_cli.supervisor",
                "executable": "/usr/bin/caffeinate",
                "interpreter": sys.executable,
                "import_root": str(broker),
                "argv": [
                    "/usr/bin/caffeinate",
                    "-is",
                    sys.executable,
                    "-m",
                    "research.colab_cli.supervisor",
                    *supervisor_args,
                ],
                "cli_executable": "/tmp/colab",
                "cli_interpreter": "/tmp/python",
                "cli_sha256": "d" * 64,
                "cli_package_closure": {"/tmp/colab": "d" * 64},
                "oauth_interpreter": sys.executable,
                "mint_program_sha256": "e" * 64,
                "manifest": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "job_id": "lane",
                "state_path": str(tmp_path / "state"),
                "output_path": str(tmp_path / "output"),
                "overlap_status": "PREPARED_NOT_QUALIFIED_FOR_OVERLAP",
            }
        )
    )
    return path, broker, launcher, argv


def test_binds_complete_root_record_before_exec(monkeypatch, tmp_path):
    path, broker, launcher, argv = admission(tmp_path)
    monkeypatch.setattr(launch, "_nice", lambda pid: 0)
    monkeypatch.setattr(
        launch,
        "_identity",
        lambda pid: {
            "birth": "birth-1",
            "birth_precision": {"unit": "fixture"},
            "executable": "python",
        },
    )
    native = json.loads(path.read_text())
    monkeypatch.setattr(
        launch,
        "native_argv",
        lambda pid: tuple(
            native["bootstrap_argv"] if pid == launch.os.getpid() else native["pane_argv"]
        ),
    )
    monkeypatch.setattr(launch.colab_control, "native_cwd", lambda pid: str(broker.resolve()))
    monkeypatch.setattr(launch.colab_control, "_source_matches", lambda *args: True)
    original_identity = launch._identity
    monkeypatch.setattr(
        launch,
        "_identity",
        lambda pid: (
            original_identity(pid)
            if pid == launch.os.getpid()
            else {"birth": "pane-1", "executable": "/bin/bash"}
        ),
    )
    root = launch.bind_root(path, broker, launcher, argv)
    record = json.loads(root.read_text())
    assert root.name == "lane.root.json"
    assert record["pid"] > 0 and record["birth"] == "birth-1"
    assert record["argv"][0] == "/usr/bin/caffeinate"
    with pytest.raises(launch.Refused, match="already exists"):
        launch.bind_root(path, broker, launcher, argv)


def test_refuses_nonzero_nice_without_a_root_record(monkeypatch, tmp_path):
    path, broker, launcher, argv = admission(tmp_path)
    monkeypatch.setattr(launch, "_nice", lambda pid: 15)
    with pytest.raises(launch.Refused, match="absolute nice 0"):
        launch.bind_root(path, broker, launcher, argv)
    assert not path.with_suffix(".root.json").exists()
