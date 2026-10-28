"""Real startup/registry joins in disposable host-only processes; no dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from research.bl_paced_identity.record import read_export
from research.bl_served_search import supervision_dispatch
from research.bl_served_search import supervision_io as io

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def prospective_root(tmp_path, *, cwd=None, invocation="exact", relative_paths=False):
    """Pause before module execution, using real psutil argv and Collector I/O.

    sitecustomize is only a test entry harness: it installs the real collector
    and waits, so the supervisor never enters any operational/provider path.
    """
    tmp_path = tmp_path.resolve()
    harness = tmp_path / "harness"
    harness.mkdir()
    state = tmp_path / "state"
    lane = state / "offline-binding"
    hook = lane / "cli-hook/sitecustomize.py"
    hook.parent.mkdir(parents=True)
    guard = str(ROOT / "research/colab_cli/cli_guard.py")
    hook.write_text(
        "import os,runpy\ntry:\n    runpy.run_path("
        + repr(guard)
        + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
    )
    (lane / "assignment.json").write_text(
        json.dumps({"endpoint": "offline", "session": "offline", "generation": "offline"})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "offline-binding",
                "output_dir": str(tmp_path / "output"),
                "sandbox_admission": {"schema": "offline-fixture"},
                "sandbox_supervision": "collector-v2",
            }
        )
    )
    cli = tmp_path / "colab"
    cli.write_text("#!" + str(Path(sys.executable).resolve()) + "\n# never executed\n")
    ready = tmp_path / "ready.json"
    directory = tmp_path / "output.supervision"
    (harness / "sitecustomize.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "from tools.sandbox.collector_v2 import Collector\n"
        "try:\n"
        f"    Collector(Path({str(directory)!r}), "
        f"manifest_bytes=Path({str(manifest)!r}).read_bytes())\n"
        "except Exception as exc:\n"
        f"    Path({str(ready)!r}).write_text(json.dumps({{'error': str(exc)}}))\n"
        "else:\n"
        f"    Path({str(ready)!r}).write_text('{{}}')\n"
        "sys.stdin.buffer.read(1)\nos._exit(0)\n"
    )
    args = [sys.executable]
    if invocation == "flag":
        args.append("-u")
    args += (
        [str(ROOT / "tools/sandbox/supervisor_v2.py")]
        if invocation == "script"
        else ["-m", "tools.sandbox.supervisor_v2"]
    )

    def argument(path):
        return os.path.relpath(path, ROOT) if relative_paths else str(path)

    flags = ["--state-dir", argument(state), "--colab", argument(cli)]
    if invocation == "swapped":
        flags = flags[2:] + flags[:2]
    elif invocation == "abbreviated":
        flags[0], flags[2] = "--state", "--col"
    args += [*flags, "run", argument(manifest)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(harness), str(ROOT)])}
    child = subprocess.Popen(
        args,
        cwd=cwd or ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "offline startup harness did not reach collector"
        yield manifest, directory, cli, json.loads(ready.read_text())
    finally:
        try:
            child.communicate(b"x", timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize("symlinked", [False, True])
def test_real_collector_registry_join_and_post_startup_cli_mutation(tmp_path, symlinked):
    cwd = ROOT
    if symlinked:
        cwd = tmp_path / "symlinked-checkout"
        cwd.symlink_to(ROOT, target_is_directory=True)
    with prospective_root(tmp_path, cwd=cwd) as (manifest, directory, cli, result):
        assert result == {}
        context = read_export(directory / "collector.json")
        startup = context["startup"]
        assert startup["cwd"] == str(ROOT)
        assert startup["colab"] == str(cli.resolve())
        assert startup["colab_sha256"] == io.file_digest(cli)
        registry = supervision_dispatch.capture_dispatch_registry(manifest)
        root = registry["roots"][0]
        expected_paths = {str((ROOT / name).resolve()) for name in context["sources"]}
        hook = Path(startup["argv"][4]) / "offline-binding/cli-hook/sitecustomize.py"
        assert root["source_hashes"].keys() == expected_paths | {str(cli.resolve()), str(hook)}
        assert root["source_hashes"][str(cli.resolve())] == startup["colab_sha256"]
        assert root["source_hashes"][str(ROOT / "tools/sandbox/collector_v2.py")] == (
            context["sources"]["tools/sandbox/collector_v2.py"]
        )
        cli.write_text(cli.read_text() + "# post-startup mutation\n")
        with pytest.raises(ValueError, match="differs from prospective startup"):
            supervision_dispatch.capture_dispatch_registry(manifest)
        cli.write_text(cli.read_text().removesuffix("# post-startup mutation\n"))
        hook.write_text(hook.read_text() + "# changed hook\n")
        with pytest.raises(ValueError, match="differs from prospective startup"):
            supervision_dispatch.capture_dispatch_registry(manifest)


def test_loaded_checkout_must_match_real_process_cwd(tmp_path):
    with prospective_root(tmp_path, cwd=tmp_path) as (_, directory, _, result):
        assert "cwd differs from loaded collector checkout" in result["error"]
        assert not (directory / "collector.json").exists()


@pytest.mark.parametrize("invocation", ["flag", "script", "swapped", "abbreviated"])
def test_manifest_bytes_require_exact_real_process_argv(tmp_path, invocation):
    with prospective_root(tmp_path, invocation=invocation) as (_, directory, _, result):
        assert "manifest requires exact supervisor argv" in result["error"]
        assert not (directory / "collector.json").exists()


def test_relative_startup_paths_are_resolved_in_root_cwd(tmp_path, monkeypatch):
    with prospective_root(tmp_path, relative_paths=True) as (manifest, directory, cli, result):
        assert result == {}
        startup = read_export(directory / "collector.json")["startup"]
        assert not Path(startup["argv"][4]).is_absolute()
        assert not Path(startup["argv"][6]).is_absolute()
        assert not Path(startup["argv"][-1]).is_absolute()
        # Registry capture runs elsewhere than the root; use actual psutil
        # identity/source checks and the real immutable collector export.
        monkeypatch.chdir(tmp_path)
        registry = supervision_dispatch.capture_dispatch_registry(manifest)
        root = registry["roots"][0]
        assert root["cwd"] == str(ROOT)
        assert root["state_dir"] == str(tmp_path / "state")
        assert root["colab"] == str(cli.resolve())
        assert root["manifest"] == str(manifest.resolve())
        assert root["source_hashes"][str(cli.resolve())] == io.file_digest(cli)
        assert registry["assignment_path"] == str(
            tmp_path / "state/offline-binding/assignment.json"
        )
