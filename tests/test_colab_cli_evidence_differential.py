"""Fake-clock behavioral differential for the L4 transport evidence change.

The ruling permits transport evidence additions only when the transport still
behaves like the reviewed tree.  These tests execute the real supervisor and
transfer implementations in isolated child interpreters, once from the
reviewed ``research/colab_cli`` tree object and once from this checkout.  The
comparison removes evidence-only fields (identities, timestamps, response
digests and source locations) while retaining commands, code hashes, retries,
deadlines, sleeps, exceptions, fences and transfer decisions.

No Colab command is made.  The child process replaces only the subprocess,
clock, UUID and idle-probe boundaries with deterministic fakes.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from research.rpc_qualification.behavior_trace import strip_evidence as _strip_evidence

REPO = Path(__file__).resolve().parents[1]
TRANSPORT_TREE = "4eefe6b10345f0a7b53c22f3158ce1c131093bfe"


_WORKER = r"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from research.colab_cli import cli_guard, policy, supervisor, transfers


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


class FakeCLIProcess:
    def __init__(self, clock, scenario):
        self.clock = clock
        self.scenario = scenario
        self.cli = None
        self.calls = []
        self.exec_count = 0
        self.session_count = 0
        self.heartbeat_done = False

    @staticmethod
    def _request(wrapped):
        match = re.search(r"# SNAKE_REQUEST_SHA256 ([0-9a-f]{64})", wrapped)
        return match.group(1) if match else None

    @staticmethod
    def _nonce(wrapped):
        match = re.search(r"print\((['\"])(.*?)\1 \+", wrapped)
        if not match:
            match = re.search(r"print\((['\"])(.*?)\1", wrapped)
        return match.group(2) if match else ""

    @staticmethod
    def _args(argv):
        args = list(argv)
        command = args[args.index("--config") + 2]
        semantic = []
        skip = False
        for index, value in enumerate(args[args.index("--config") + 2 :]):
            if skip:
                skip = False
                continue
            if value in {"-f", "--config"}:
                semantic.append(value)
                skip = True
                semantic.append("<file>")
                continue
            if value == "-s":
                semantic.extend((value, "<session>"))
                skip = True
                continue
            semantic.append(value)
        return command, semantic

    def __call__(self, argv, **kwargs):
        command, semantic = self._args(argv)
        row = {
            "kind": "external",
            "command": command,
            "args": semantic,
            "timeout": float(kwargs.get("timeout", 0)),
        }
        if command == "exec":
            path = Path(argv[argv.index("-f") + 1])
            wrapped = path.read_text()
            row["envelope_sha256"] = hashlib.sha256(wrapped.encode()).hexdigest()
            row["code_sha256"] = self._request(wrapped)
            self.exec_count += 1
            nonce = self._nonce(wrapped)
            if self.scenario == "heartbeat" and not self.heartbeat_done:
                self.heartbeat_done = True
                self.cli._put_lease("/content/root", "generation", 60)
            if self.scenario == "exhaustion" or (
                self.scenario == "recovery" and self.exec_count == 1
            ):
                timeout = float(kwargs["timeout"])
                self.clock.now += timeout
                self.calls.append({**row, "outcome": "timeout", "timeout": timeout})
                raise subprocess.TimeoutExpired(argv, timeout)
            if self.scenario == "permanent":
                output = nonce + json.dumps({"ok": False, "error": "kernel permanent"})
            elif self.scenario == "tails":
                output = "\n".join(
                    (
                        nonce + json.dumps({"ok": True, "pending": True}),
                        nonce + json.dumps({"ok": True, "value": {"ok": True}}),
                    )
                )
            elif self.exec_count == 1 and self.scenario == "pending":
                output = nonce + json.dumps({"ok": True, "pending": True})
            else:
                output = nonce + json.dumps({"ok": True, "value": {"ok": True}})
            self.calls.append({**row, "outcome": "completed"})
            return subprocess.CompletedProcess(argv, 0, output, "")
        if command == "sessions" and self.scenario == "reads":
            self.session_count += 1
            if self.session_count == 1:
                self.calls.append({**row, "outcome": "503"})
                raise supervisor.Refused("HTTP 503")
            self.calls.append({**row, "outcome": "completed"})
            return subprocess.CompletedProcess(argv, 0, "No active sessions found on server.\n", "")
        if command == "sessions" and self.scenario == "stop":
            self.session_count += 1
            self.calls.append({**row, "outcome": "completed"})
            if self.session_count == 1:
                return subprocess.CompletedProcess(
                    argv, 0, "[session] endpoint | Hardware: L4\n", ""
                )
            return subprocess.CompletedProcess(argv, 0, "No active sessions found on server.\n", "")
        if command == "upload":
            self.calls.append({**row, "outcome": "completed"})
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.calls.append({**row, "outcome": "completed"})
        return subprocess.CompletedProcess(argv, 0, "", "")


def _stable_ids():
    counter = 0

    class Value:
        hex = ""

        def __init__(self, value):
            self.hex = value

    def make():
        nonlocal counter
        counter += 1
        return Value(f"{counter:032x}")

    return make


def _exception(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _run_rpc(scenario):
    clock = Clock()
    events = []
    supervisor.time.monotonic = clock.monotonic
    supervisor.time.sleep = clock.sleep
    transfers.time.monotonic = clock.monotonic
    transfers.time.sleep = clock.sleep
    policy.poll_delay = lambda _attempt: 2.0
    supervisor.uuid.uuid4 = _stable_ids()
    transfers.uuid.uuid4 = supervisor.uuid.uuid4
    process = FakeCLIProcess(clock, scenario)
    supervisor.subprocess.run = process
    probe_count = 0

    def kernel_idle(*args, **kwargs):
        nonlocal probe_count
        probe_count += 1
        return scenario == "recovery" and probe_count >= 2

    cli_guard.kernel_idle = kernel_idle
    cli = supervisor.CLI("fake-colab", Path("/tmp/fake-config"), "session")
    process.cli = cli
    original_rpc = cli.rpc

    def traced_rpc(code, *args, **kwargs):
        events.append({"phase": "before", "fence": cli.exec_uncertain})
        try:
            result = original_rpc(code, *args, **kwargs)
        except BaseException:
            events.append({"phase": "after", "fence": cli.exec_uncertain})
            raise
        events.append({"phase": "after", "fence": cli.exec_uncertain})
        return result

    cli.rpc = traced_rpc
    try:
        if scenario == "exhaustion" or scenario == "recovery":
            result = cli.poll_rpc("result = {'ok': True}", timeout=150)
        elif scenario == "reads":
            result = cli.sessions(timeout=20)
        else:
            result = cli.rpc("result = {'ok': True}", timeout=20)
        outcome = {"kind": "return", "value": result}
    except BaseException as exc:
        outcome = {"kind": "exception", "error": _exception(exc)}
    return {
        "outcome": outcome,
        "external": process.calls,
        "call_records": cli.call_records,
        "sleeps": clock.sleeps,
        "clock": clock.now,
        "fence": cli.exec_uncertain,
        "fence_transitions": events,
        "release": {"decision": "none", "fence": cli.exec_uncertain},
    }


def _run_probe_policy():
    '''Compare the ruling's authorized legacy hardened-probe normalization.'''
    cases = (
        cli_guard.GuardError("earlier context timed out\nGuardError: OAuth credentials rejected"),
        cli_guard.GuardError("read helper operation budget exhausted"),
        cli_guard.GuardError(
            "read helper operation budget exhausted\nGuardError: quota denied"
        ),
        ConnectionError("connection reset"),
        TimeoutError("timed out"),
        subprocess.TimeoutExpired([], 1),
    )
    return {
        "probe_policy": [policy.recovery_probe_retryable(error) for error in cases],
        "release": {"decision": "none", "fence": False},
    }


def _run_stop(root):
    clock = Clock()
    supervisor.time.monotonic = clock.monotonic
    supervisor.time.sleep = clock.sleep
    process = FakeCLIProcess(clock, "stop")
    supervisor.subprocess.run = process
    cli = supervisor.CLI("fake-colab", Path("/tmp/fake-config"), "session")
    process.cli = cli
    events = [{"phase": "before", "fence": cli.exec_uncertain}]
    try:
        supervisor.stop_verified(cli, "endpoint")
        outcome = {"kind": "return", "value": True}
    except BaseException as exc:
        outcome = {"kind": "exception", "error": _exception(exc)}
    events.append({"phase": "after", "fence": cli.exec_uncertain})
    return {
        "outcome": outcome,
        "external": process.calls,
        "sleeps": clock.sleeps,
        "fence": cli.exec_uncertain,
        "fence_transitions": events,
        "release": {"decision": "verified", "session_reads": process.session_count},
    }


class TransferCLI:
    session = "transfer-session"

    def __init__(self, root, scenario):
        self.root = Path(root)
        self.scenario = scenario
        (self.root / "content").mkdir(parents=True)
        self.transient_failures = 1 if "retry" in self.scenario else 0
        self.calls = []
        self.rpc_calls = []
        self.transfer_records = []

    def path(self, value):
        return self.root / str(value).lstrip("/")

    def call(self, *args, timeout=45):
        self.calls.append({"command": args[0], "args": list(args), "timeout": timeout})
        if self.transient_failures:
            self.transient_failures -= 1
            raise RuntimeError("ContentsClient connection reset")
        if args[0] == "upload":
            source, remote = Path(args[-2]), self.path(args[-1])
            remote.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, remote)
        elif args[0] == "download":
            remote, local = self.path(args[-2]), Path(args[-1])
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote, local)
        else:
            raise AssertionError(args)
        return ""

    def rpc(self, code, timeout=20):
        operation = re.search(r"# (SNAKE_TRANSFER_[A-Z_]+)", code).group(1)
        self.rpc_calls.append(
            {
                "operation": operation,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "timeout": timeout,
            }
        )
        spec = json.loads(re.search(r"spec = (\{.*\})", code).group(1))
        if operation == "SNAKE_TRANSFER_CREATE_DIR":
            self.path(spec["path"]).mkdir(parents=True, exist_ok=True)
            return True
        if operation == "SNAKE_TRANSFER_REMOVE_TREE":
            shutil.rmtree(self.path(spec["path"]), ignore_errors=True)
            return True
        if operation == "SNAKE_TRANSFER_VERIFY_FILE":
            data = self.path(spec["path"]).read_bytes()
            return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if operation == "SNAKE_TRANSFER_UPLOAD_ASSEMBLE":
            data = b"".join(
                self.path(f"{spec['parts']}/{index:08d}.part").read_bytes()
                for index in range(spec["chunks"])
            )
            target = self.path(spec["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if operation == "SNAKE_TRANSFER_DOWNLOAD_SPLIT":
            data = self.path(spec["source"]).read_bytes()
            parts = self.path(spec["parts"])
            parts.mkdir(parents=True, exist_ok=True)
            for index in range(0, len(data), spec["chunk_bytes"]):
                (parts / f"{index // spec['chunk_bytes']:08d}.part").write_bytes(
                    data[index : index + spec["chunk_bytes"]]
                )
            return {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "chunks": (len(data) + spec["chunk_bytes"] - 1) // spec["chunk_bytes"],
            }
        raise AssertionError(operation)


def _normal_arg(value):
    value = str(value)
    if value.startswith("/content/"):
        return value
    if value.startswith("/"):
        return "<local>"
    return value


def _run_transfers(root, scenario):
    clock = Clock()
    transfers.time.monotonic = clock.monotonic
    transfers.time.sleep = clock.sleep
    transfers.uuid.uuid4 = _stable_ids()
    transfers.UPLOAD_CHUNK_BYTES = 4
    transfers.DOWNLOAD_CHUNK_BYTES = 4
    cli = TransferCLI(root, scenario)
    source = Path(root) / "source.bin"
    source.write_bytes(b"0123456789")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        transfers.upload_file(cli, source, "/content/repeated.bin", lambda: None)
        if "retry" in cli.scenario:
            source.write_bytes(b"abcdefghij")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        transfers.upload_file(cli, source, "/content/repeated.bin", lambda: None)
        target = Path(root) / "out.bin"
        transfers.download_file(
            cli,
            "/content/repeated.bin",
            target,
            source.stat().st_size,
            lambda: None,
            expected_sha256=digest,
        )
        outcome = {
            "kind": "return",
            "value": digest,
            "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    except BaseException as exc:
        outcome = {"kind": "exception", "error": _exception(exc)}
    remaining_staging = sorted(
        str(path.relative_to(root))
        for path in (Path(root) / "content").glob(".snake-transfer-*")
    )
    return {
        "outcome": outcome,
        "calls": [
            {
                "command": row["command"],
                "args": [_normal_arg(v) for v in row["args"]],
                "timeout": row["timeout"],
            }
            for row in cli.calls
        ],
        "rpc_calls": cli.rpc_calls,
        "transfers": [
            {key: row.get(key) for key in ("direction", "remote", "size_bytes", "status", "sha256")}
            for row in cli.transfer_records
        ],
        "sleeps": clock.sleeps,
        "release": {
            "cleanup_verified": not remaining_staging,
            "remaining_staging": remaining_staging,
        },
    }


def main():
    scenario, root = sys.argv[1], Path(sys.argv[2])
    if scenario in {"chunks", "chunks_retry"}:
        result = _run_transfers(root, scenario)
    elif scenario == "probe_policy":
        result = _run_probe_policy()
    elif scenario == "stop":
        result = _run_stop(root)
    else:
        result = _run_rpc(scenario)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def _materialize_baseline(destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", TRANSPORT_TREE],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    package = destination / "research" / "colab_cli"
    package.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(package)


def _run_worker(source_root: Path, workspace_root: Path, scenario: str) -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    workdir = workspace_root / f"work-{scenario}"
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, scenario, str(workdir)],
        cwd=source_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"worker {scenario} failed in {source_root}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"worker did not emit JSON: {result.stdout!r}\n{result.stderr}"
        ) from exc


def _differential_trace(tmp_path: Path) -> dict:
    scenarios = (
        "pending",
        "tails",
        "recovery",
        "exhaustion",
        "permanent",
        "heartbeat",
        "reads",
        "probe_policy",
        "stop",
        "chunks",
        "chunks_retry",
    )
    with tempfile.TemporaryDirectory(prefix="colab-differential-") as baseline_dir:
        baseline_root = Path(baseline_dir)
        _materialize_baseline(baseline_root)
        current_root = tmp_path / "current"
        current_root.mkdir()
        for root in (baseline_root, current_root):
            (root / "work").mkdir()
        trace = {"transport_tree": TRANSPORT_TREE, "scenarios": {}}
        for scenario in scenarios:
            baseline = _run_worker(baseline_root, baseline_root, scenario)
            current = _run_worker(REPO, current_root, scenario)
            trace["scenarios"][scenario] = {
                "baseline": baseline,
                "current": current,
                "canonical": _strip_evidence(baseline),
            }
            if scenario == "probe_policy":
                # The ruling explicitly authorizes folding the hardened probe
                # predicate into policy.py. The frozen tree accepts terminal
                # GuardError text as retryable; current policy rejects those
                # terminal faults while retaining helper/transport retryability.
                expected_legacy = [True, True, True, True, True, True]
                expected_hardened = [False, True, False, True, True, True]
                assert baseline["probe_policy"] == expected_legacy
                assert current["probe_policy"] == expected_hardened
                trace["scenarios"][scenario]["authorized_exception"] = {
                    "kind": "probe_policy_hardening",
                    "baseline": "legacy_terminal_guarderror_retry",
                    "current": "folded_hardened_probe_policy",
                    "baseline_vector": expected_legacy,
                    "current_vector": expected_hardened,
                }
            elif _strip_evidence(current) != _strip_evidence(baseline):
                raise AssertionError(
                    scenario
                    + "\nCURRENT="
                    + json.dumps(_strip_evidence(current), sort_keys=True)
                    + "\nBASELINE="
                    + json.dumps(_strip_evidence(baseline), sort_keys=True)
                )
        return trace


@pytest.fixture(scope="module")
def differential_trace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _differential_trace(tmp_path_factory.mktemp("colab-differential"))


@pytest.mark.parametrize(
    "scenario",
    [
        "pending",
        "tails",
        "recovery",
        "exhaustion",
        "permanent",
        "heartbeat",
        "reads",
        "probe_policy",
        "stop",
        "chunks",
        "chunks_retry",
    ],
)
def test_current_transport_matches_reviewed_tree_under_fake_clock(
    differential_trace: dict, scenario: str
) -> None:
    """Each ruling scenario has identical behavior after evidence is stripped."""
    trace = differential_trace
    row = trace["scenarios"][scenario]
    if scenario == "probe_policy":
        assert row["authorized_exception"]["kind"] == "probe_policy_hardening"
        assert row["current"]["probe_policy"] == [False, True, False, True, True, True]
        assert row["baseline"]["probe_policy"] == [True, True, True, True, True, True]
    else:
        assert _strip_evidence(row["current"]) == _strip_evidence(row["baseline"])
    if scenario in {"chunks", "chunks_retry"}:
        assert row["current"]["release"] == {
            "cleanup_verified": True,
            "remaining_staging": [],
        }


def test_differential_trace_can_be_published_as_machine_evidence(
    differential_trace: dict,
) -> None:
    """The optional output is create-only and contains both raw and canonical traces."""
    trace = differential_trace
    destination = os.environ.get("SNAKE_COLAB_DIFFERENTIAL_TRACE")
    if destination:
        path = Path(destination)
        assert not path.exists(), "differential trace publication is create-only"
        path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    assert trace["transport_tree"] == TRANSPORT_TREE
