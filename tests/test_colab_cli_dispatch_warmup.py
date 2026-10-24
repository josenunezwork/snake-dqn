"""Production dispatch regression for a timed-out first Colab kernel probe.

The test deliberately crosses the process boundaries which the warmup unit
tests cannot cover: ``dispatch.sh`` starts a fixture tmux command, the broker
starts the supervisor, and every fake CLI invocation imports the real
``sitecustomize`` guard.  The only replaced provider pieces are the official
CLI package and its HTTP boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DISPATCH = REPO / "tools" / "ops" / "dispatch.sh"
FAKE_COLAB = REPO / "tests" / "fixtures" / "fake_colab.py"


def _wait_for(predicate: object, seconds: float = 25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("fixture did not publish expected production-path evidence")


def _write_installed_cli_stub(root: Path) -> Path:
    """Provide only the installed CLI state and HTTP APIs used by the guard.

    ``fake_colab.py`` is still the CLI executable.  These modules are loaded
    by the supervisor-created sitecustomize hook and by its private guard
    helper process, which makes the regression exercise persisted session
    metadata rather than a monkeypatched ``kernel_idle`` boolean.
    """
    package = root / "colab_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "client.py").write_text(
        "class Client:\n"
        "    def assign(self, *args, **kwargs): return None\n"
        "    def _issue_request(self, *args, **kwargs): return None\n"
    )
    (package / "state.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "class State:\n"
        "    def __init__(self, session, value):\n"
        "        self.session = session\n"
        "        self.endpoint = value['endpoint']\n"
        "        self.url = value['url']\n"
        "        self.token = value['token']\n"
        "        self.kernel_id = value.get('kernel_id')\n"
        "class StateStore:\n"
        "    def __init__(self, path): self.path = Path(path)\n"
        "    def get(self, session):\n"
        "        if not self.path.exists(): return None\n"
        "        value = json.loads(self.path.read_text()).get(session)\n"
        "        if not isinstance(value, dict): return None\n"
        "        needed = ('endpoint', 'url', 'token')\n"
        "        valid = all(isinstance(value.get(k), str) and value[k] for k in needed)\n"
        "        if not valid: return None\n"
        "        return State(session, value)\n"
        "    def add(self, state):\n"
        "        value = json.loads(self.path.read_text()) if self.path.exists() else {}\n"
        "        row = value[state.session]\n"
        "        row['kernel_id'] = state.kernel_id\n"
        "        self.path.write_text(json.dumps(value, sort_keys=True))\n"
        "        record = {'session': state.session, 'kernel_id': state.kernel_id}\n"
        "        with Path(__import__('os').environ['WARMUP_STATE_LOG']).open('a') as out:\n"
        "            out.write(json.dumps(record) + '\\n')\n"
    )
    (package / "contents.py").write_text(
        "class ContentsClient:\n"
        "    def __init__(self, state): self.state = state\n"
        "    def _request(self, method, path, params=None):\n"
        "        assert (method, path, params) == ('GET', '/content', {'content': '0'})\n"
        "        return {'type': 'directory'}\n"
    )
    (root / "requests.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "class HTTPError(Exception): pass\n"
        "class _Session:\n"
        "    def request(self, *args, **kwargs): return None\n"
        "class sessions: Session = _Session\n"
        "class _Response:\n"
        "    def raise_for_status(self): return None\n"
        "    def json(self): return [{'id': 'recovered-kernel', 'execution_state': 'idle'}]\n"
        "def get(url, **kwargs):\n"
        "    record = {'url': url, 'timeout': kwargs.get('timeout')}\n"
        "    Path(os.environ['WARMUP_HTTP_LOG']).open('a').write(json.dumps(record) + '\\n')\n"
        "    return _Response()\n"
    )
    return root


def _write_oauth_wrapper(root: Path) -> Path:
    """Run real guard helpers while keeping credential minting offline."""
    wrapper = root / "oauth-python"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json, runpy, sys, time\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('cli_guard.py'):\n"
        "    guard = sys.argv[1]\n"
        "    sys.argv = [guard, *sys.argv[2:]]\n"
        "    runpy.run_path(guard, run_name='__main__')\n"
        "elif '-c' in sys.argv:\n"
        "    result = {'access_token': 'offline-release-token', 'expires_at': time.time() + 3600}\n"
        "    result['scopes'] = json.loads(sys.argv[-1])\n"
        "    print(json.dumps(result))\n"
        "else:\n"
        "    raise SystemExit('unexpected offline oauth invocation')\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def _fixture_broker(tmp_path: Path) -> tuple[Path, str]:
    """Make a clean, commit-pinned broker whose launch entry is test-safe."""
    broker = tmp_path / "broker"
    shutil.copytree(REPO / "research" / "colab_cli", broker / "research" / "colab_cli")
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        source = REPO / name
        if source.is_file():
            shutil.copy2(source, broker / name)
    launch = broker / "research" / "colab_cli" / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(cd "$(dirname "$0")/../.." && pwd)"\n'
        'exec "${SNAKE_DQN_PYTHON:?}" -m research.colab_cli.supervisor "$@"\n'
    )
    launch.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=broker, check=True)
    subprocess.run(["git", "add", "."], cwd=broker, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=warmup-test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture broker",
        ],
        cwd=broker,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=broker, text=True).strip()
    return broker, commit


def test_dispatch_recovers_missing_kernel_metadata_before_first_upload(tmp_path: Path) -> None:
    """An ambiguous first readiness exec needs a GET-discovered persisted kernel ID."""
    broker, commit = _fixture_broker(tmp_path)
    home = tmp_path / "home"
    binaries = home / ".local" / "bin"
    binaries.mkdir(parents=True)
    fake = binaries / "colab"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, runpy, sys\n"
        "from colab_cli.client import Client\n"
        "# The wrapper takes a read-only account snapshot before Broker installs its hook.\n"
        "account_read = sys.argv[1:] in (['--auth', 'oauth2', 'sessions'], "
        "['--auth', 'oauth2', 'usage'])\n"
        "if not account_read and not getattr(Client, '_snake_guard_installed', False):\n"
        "    raise SystemExit('sitecustomize guard hook was not installed')\n"
        "if account_read:\n"
        "    config = os.path.join(os.environ['FAKE_COLAB_REMOTE'], 'account-snapshot.json')\n"
        "    sys.argv[1:] = ['--auth', 'oauth2', '--config', config, sys.argv[-1]]\n"
        "args = sys.argv[1:]\n"
        "if len(args) > 4 and args[4] == 'upload':\n"
        "    sessions = json.loads(open(args[3]).read())\n"
        "    if not any(row.get('kernel_id') == 'recovered-kernel' for row in sessions.values()):\n"
        "        raise SystemExit('upload preceded recovered kernel identity')\n"
        f"runpy.run_path({str(FAKE_COLAB)!r}, run_name='__main__')\n"
    )
    fake.chmod(0o755)
    tmux = binaries / "tmux"
    tmux.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        "if sys.argv[1] == 'has-session': raise SystemExit(0)\n"
        "subprocess.Popen(['bash', '-c', sys.argv[-1]], stdin=subprocess.DEVNULL, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
    )
    tmux.chmod(0o755)
    stub_root = _write_installed_cli_stub(tmp_path / "installed-cli")
    oauth_python = _write_oauth_wrapper(tmp_path)
    fake_state, fake_log, remote = (
        tmp_path / "state.json",
        tmp_path / "calls.jsonl",
        tmp_path / "remote",
    )
    fake_state.write_text(
        json.dumps({"balance": 50.0, "rate": 0.0, "sessions": {}, "exec_count": {}})
    )
    fake_log.touch()
    remote.mkdir()
    http_log = tmp_path / "kernel-http.jsonl"
    state_log = tmp_path / "kernel-state.jsonl"
    ops = tmp_path / "ops"
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=broker, text=True
    ).strip()
    ops.mkdir()
    (ops / "colab-cli-qualified.json").write_text(
        json.dumps({"schema": "snake-dqn-colab-cli-qualified/v1", "tree_hash": tree})
    )
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"warmup input")
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "warmup-dispatch",
                "commit": commit,
                "script": "research/colab_cli/acceptance_probe.sh",
                "requirements": {"gpu": "cpu", "min_vcpu": 1, "est_minutes": 1.0},
                "timeout_minutes": 3.0,
                "budget": {
                    "rate_units_per_hour": 0.5,
                    "unit_cap": 1.5,
                    "reserve_units": 0.0,
                    "other_assignments": {},
                },
                "output_dir": str(output),
                "env": {"WARMUP_DISPATCH_TEST": "1"},
                "inputs": [
                    {
                        "path": str(payload),
                        "dest": "input.bin",
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ],
                "lease_seconds": 180,
                "max_export_bytes": 1024 * 1024,
                "resume_exemption": {
                    "kind": "acceptance",
                    "reason": "offline transport regression",
                },
            }
        )
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "PYTHONPATH": str(stub_root),
        "SNAKE_DQN_PYTHON": sys.executable,
        "SNAKE_COLAB_OAUTH_PYTHON": str(oauth_python),
        "FAKE_COLAB_STATE": str(fake_state),
        "FAKE_COLAB_LOG": str(fake_log),
        "FAKE_COLAB_REMOTE": str(remote),
        "FAKE_COLAB_MODE": "busy_exec",
        "FAKE_COLAB_OWN_RATE": "0.5",
        "WARMUP_HTTP_LOG": str(http_log),
        "WARMUP_STATE_LOG": str(state_log),
        "OPS_DIR": str(ops),
        "OPS_ALLOC_POLL_INTERVAL_SECONDS": "0.02",
        "OPS_ALLOC_LOCK_TIMEOUT_SECONDS": "20",
    }
    result = subprocess.run(
        [
            str(DISPATCH),
            "warmup-path",
            str(manifest),
            "--broker",
            str(broker),
            "--cwd",
            str(broker),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    _wait_for(lambda: (output / "receipt.json").is_file())
    receipt = json.loads((output / "receipt.json").read_text())
    # The fixture stops after the first useful uploads because its sparse
    # temporary broker has no GitHub source remote.  The receipt still proves
    # orderly cleanup and the completed upload boundary under production code.
    assert receipt["status"] == "INCOMPLETE" and receipt["stop_verified"] is True
    assert receipt["warmup_seconds"] > 0
    events = [json.loads(line) for line in fake_log.read_text().splitlines()]
    assert sum(event["argv"][4] == "new" for event in events) == 1
    assert json.loads(fake_state.read_text())["exec_count"][receipt["session"]] >= 2
    uploads = [event for event in events if event["argv"][4] == "upload"]
    assert uploads, "first Contents upload must occur only after recovered idle proof"
    assert receipt["contents_transfers"][0]["status"] == "completed"
    kernel_reads = [json.loads(line) for line in http_log.read_text().splitlines()]
    assert kernel_reads == [
        {"url": "https://fake-runtime.invalid/api/kernels", "timeout": [30, 120]}
    ]
    config = next((ops / "state" / "warmup-path" / "warmup-dispatch").glob("sessions.json"))
    saved = json.loads(config.read_text())[receipt["session"]]
    assert [json.loads(line) for line in state_log.read_text().splitlines()] == [
        {"session": receipt["session"], "kernel_id": "recovered-kernel"}
    ]
    assert saved["kernel_id"] == "recovered-kernel"
    assert saved["url"] == "https://fake-runtime.invalid/" and saved["token"] == "fake-proxy-token"
