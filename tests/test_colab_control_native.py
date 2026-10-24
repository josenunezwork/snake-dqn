"""Darwin-only native identity fixtures for the Colab control boundary.

These fixtures use a temporary Git broker and inert local Python/shell children.
They do not invoke the provider CLI, network, tmux, a factory, or any research
payload.  They are intentionally run only through the serial nice-15 test
route.  The separate authorized nice-0 observation is described at the end of
this module and is not a pytest fixture, so collecting this file never starts a
tmux session or changes priority.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.bl_labels.factory_persistence import _native_identity
from research.compute import colab_control as control
from research.compute.factory_companion_argv import native_argv

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Darwin native process identity")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wait(path: Path, seconds: float = 5.0) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text().strip()
        time.sleep(0.02)
    raise AssertionError(f"fixture did not publish {path}")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _broker(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "broker"
    package = root / "research/colab_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    ready = root / "ready.txt"
    cli_pid = root / "cli.pid"
    cli = root / "colab"
    cli.write_text(
        "#!"
        + sys.executable
        + "\nimport os,pathlib,time\n"
        + "pathlib.Path(os.environ['COLAB_CONTROL_CLI_PID']).write_text(str(os.getpid()))\n"
        + "time.sleep(30)\n"
    )
    cli.chmod(0o755)
    supervisor = package / "supervisor.py"
    supervisor.write_text(
        "import os, pathlib, subprocess, time\n"
        "cli=os.environ['COLAB_CONTROL_FAKE_CLI']\n"
        "p=subprocess.Popen([cli, '--auth', 'oauth2', '--config',\n"
        " os.environ['COLAB_CONTROL_CONFIG'], 'sessions'])\n"
        "pathlib.Path(os.environ['COLAB_CONTROL_READY']).write_text(str(p.pid))\n"
        "p.wait(timeout=30)\n"
    )
    (package / "source.py").write_text("# inert source closure\n")
    (package / "release.py").write_text("# inert release closure\n")
    (root / "dispatch.py").write_text("# inert dispatch provenance\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Native Fixture")
    _git(root, "config", "user.email", "native@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "inert native fixture")
    return root, ready, cli_pid


def _record(root: Path, supervisor_pid: int, cli: Path, state: Path) -> control.AdmissionRecord:
    identity = _native_identity(supervisor_pid)
    argv = list(native_argv(supervisor_pid))
    tree = _git(root, "rev-parse", "HEAD:research/colab_cli")
    qualification = root / "qualification.json"
    qualification.write_text(
        json.dumps({"schema": "snake-dqn-colab-cli-qualified/v1", "tree_hash": tree})
    )
    manifest = root / "manifest.json"
    manifest.write_text("{}\n")
    admission = root / "lane.json"
    source_closure = {
        str(path.relative_to(root)): _sha(path)
        for path in (root / "research/colab_cli").glob("*.py")
    }
    dispatch = root / "dispatch.py"
    dispatch.write_text("# inert dispatch provenance\n")
    raw = {
        "policy_id": control.POLICY_ID,
        "ruling_sha256": control.RULING_SHA256,
        "operational_pin": _git(root, "rev-parse", "HEAD"),
        "qualified_broker_source_closure": tree,
        "qualification_record": str(qualification),
        "qualification_record_sha256": _sha(qualification),
        "operational_root": str(root),
        "dispatch_source_closure": {str(dispatch): _sha(dispatch)},
        "broker": str(root),
        "launcher": str(root / "research/colab_cli/launch.sh"),
        "module": control.SUPERVISOR_MODULE,
        "pid": supervisor_pid,
        "birth": identity["birth"],
        "executable": identity["executable"],
        "interpreter": sys.executable,
        "interpreter_native": control.interpreter_native(Path(sys.executable)),
        "cli_interpreter_native": control.interpreter_native(Path(sys.executable)),
        "import_root": str(root),
        "argv": argv,
        "source_closure": source_closure,
        "cli_executable": str(cli),
        "cli_interpreter": sys.executable,
        "cli_sha256": _sha(cli),
        "cli_package_closure": {str(cli): _sha(cli)},
        "oauth_interpreter": sys.executable,
        "mint_program_sha256": "a" * 64,
        "manifest": str(manifest),
        "manifest_sha256": _sha(manifest),
        "job_id": "native-fixture",
        "state_path": str(state),
        "output_path": str(root / "output"),
    }
    helper = str(root / "tools/ops/colab_control_launch.py")
    launcher = raw["launcher"]
    raw["argv"][0] = sys.executable
    raw["bootstrap_argv"] = [
        sys.executable,
        helper,
        "--admission",
        str(admission),
        "--broker",
        str(root),
        "--launcher",
        launcher,
        "--",
        launcher,
        "--control-nice0",
        *argv[3:],
    ]
    admission.write_text(json.dumps(raw, sort_keys=True))
    raw["bootstrap"] = {
        "pid": supervisor_pid,
        "birth": identity["birth"],
        "cwd": str(root),
        "executable": identity["executable"],
        "argv": raw["bootstrap_argv"],
    }

    raw["admission_path"] = str(admission)
    raw["admission_sha256"] = _sha(admission)
    root_record = root / "lane.root.json"
    root_record.write_text(json.dumps(raw, sort_keys=True))
    return control._record(root_record, raw)


def test_actual_native_supervisor_and_shebang_cli_identity(tmp_path: Path) -> None:
    """Exercise native argv, cwd, executable, birth, and default source checks."""
    broker, ready, _ = _broker(tmp_path)
    state = broker / "state"
    state.mkdir()
    cli = broker / "colab"
    environment = dict(
        os.environ,
        COLAB_CONTROL_FAKE_CLI=str(cli),
        COLAB_CONTROL_READY=str(ready),
        COLAB_CONTROL_CLI_PID=str(broker / "cli.pid"),
        COLAB_CONTROL_CONFIG=str(state / "native-fixture/sessions.json"),
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "research.colab_cli.supervisor",
            "--state-dir",
            str(state),
            "--colab",
            str(cli),
            "run",
            str(broker / "manifest.json"),
        ],
        cwd=broker,
        env=environment,
    )
    cli_pid = None
    try:
        cli_pid = int(_wait(ready))
        record = _record(broker, process.pid, cli, state)
        supervisor = _native_identity(process.pid)
        cli_identity = _native_identity(cli_pid)
        rows = [
            {
                "pid": process.pid,
                "ppid": os.getpid(),
                "birth": supervisor["birth"],
                "executable": supervisor["executable"],
            },
            {
                "pid": cli_pid,
                "ppid": process.pid,
                "birth": cli_identity["birth"],
                "executable": cli_identity["executable"],
            },
        ]
        admitted = control.annotate(rows, {(record.pid, record.birth): record})
        assert admitted[process.pid].role == "supervisor"
        assert admitted[cli_pid].role == "cli"
        observed_cli_argv = native_argv(cli_pid)
        assert observed_cli_argv[0] in {sys.executable, record.cli_interpreter_native}
        assert observed_cli_argv[1] == str(cli)
        assert control.native_cwd(process.pid) == str(broker.resolve())
        parent_initializer = broker / "research/__init__.py"
        parent_initializer.write_text("raise RuntimeError('unapproved parent')\n")
        assert not control._import_resolution_matches(record)
        assert not control._source_matches(record, str(broker.resolve()))
        parent_initializer.unlink()
    finally:
        if cli_pid is not None:
            try:
                os.kill(cli_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


# Authorized nice-0 observation plan (not executed by this test module):
# 1. Root/V obtain an exclusive quiet lease and identify an existing verified
#    nice-0 tmux server.  Do not create or reprioritize a server.
# 2. Create one owned transient pane (<=120 s), using the prospective qualified
#    broker revision, an inert local fake transport, and a root record whose
#    expected pane/bootstrap/caffeinate/supervisor/CLI/helper argv vectors are
#    frozen before start.  No factory, provider command, network, science input,
#    or allocation is permitted.
# 3. Sample each live PID with native argv, native birth/executable, cwd, and
#    nice.  Require absolute 0 for pane, bootstrap, caffeinate, supervisor, and
#    active fake CLI/helper; classify through the normal source checker.
# 4. Confirm the fake transport never receives a provider verb; wait for bounded
#    cleanup and record only redacted identity/nice/timing evidence.  This is an
#    operational fixture, never overlap-canary or qualification proof.
