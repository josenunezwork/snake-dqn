"""Offline checks for worker and queue-worker notebook contracts."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_queue import core, worker

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "research" / "colab_queue" / "worker.sh"


@pytest.mark.parametrize(
    "name",
    [
        "colab_queue_worker.ipynb",
        "colab_queue_worker_a.ipynb",
        "colab_queue_worker_b.ipynb",
        "colab_queue_worker_c.ipynb",
    ],
)
def test_worker_notebook_has_four_pinned_bootstrap_and_keepalive_cells(name):
    path = ROOT / "notebooks" / name
    assert path.is_file(), path
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] >= 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(notebook["cells"]) == 4
    for cell in cells:
        assert cell["execution_count"] is None and cell["outputs"] == []
        compile("".join(cell["source"]), str(path), "exec")
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "drive.mount" in source
    assert "worker.sh" in source
    assert "nohup" in source or "Popen" in source
    assert "REF=" in source or "ref" in source.lower()
    # The bootstrap must pin its fetched source; an unqualified raw URL is
    # precisely what would make old runtimes silently run new code.
    assert any(token in source for token in ("COMMIT", "PIN", "SHA", "refs/heads"))
    pins = re.findall(r"[0-9a-f]{40}", source)
    assert pins and all(set(pin) != {"0"} for pin in pins)
    assert "KEEP_RUNTIME = True" in "".join(cells[0]["source"])
    assert "--worker-id" in "".join(cells[2]["source"])
    assert "# Keep-alive" in "".join(cells[-1]["source"])
    assert "proc.wait(timeout=600)" in "".join(cells[-1]["source"])


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("colab_queue_worker*.ipynb")))
@pytest.mark.parametrize("keep_runtime", [True, False])
def test_notebook_idle_setting_precedes_mount_and_launch(path, keep_runtime, monkeypatch):
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][0]["source"])
    source = source.replace("KEEP_RUNTIME = True", f"KEEP_RUNTIME = {keep_runtime}")
    monkeypatch.setenv("IDLE_MINUTES", "100000")
    mounted = []

    def mount(target):
        mounted.append(target)
        assert os.environ.get("IDLE_MINUTES", "10") == ("100000" if keep_runtime else "10")

    monkeypatch.setitem(
        sys.modules, "google.colab", SimpleNamespace(drive=SimpleNamespace(mount=mount))
    )
    exec(compile(source, str(path), "exec"), {})
    assert mounted == ["/content/drive"]


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("colab_queue_worker*.ipynb")))
@pytest.mark.parametrize("missing_status", [False, True])
def test_notebook_keepalive_reports_own_job_and_ends_on_exit(
    path, missing_status, tmp_path, capsys
):
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][-1]["source"])
    log_path = tmp_path / "worker.log"
    heartbeat_dir = tmp_path / "queue" / "heartbeats"
    heartbeat_dir.mkdir(parents=True)
    (heartbeat_dir / "other-worker.json").write_text(json.dumps({"job_id": "other-job"}))
    if not missing_status:
        log_path.write_text("old line\nlatest line\n")
        (heartbeat_dir / "this-worker.json").write_text(json.dumps({"job_id": "own-job"}))

    class FakeProcess:
        waits = 0

        def poll(self):
            return 0 if self.waits == 2 else None

        def wait(self, timeout):
            assert timeout == 600
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

    proc = FakeProcess()
    exec(
        compile(source, str(path), "exec"),
        {
            "Path": Path,
            "DRIVE_ROOT": str(tmp_path),
            "worker_id": "this-worker",
            "proc": proc,
            "log_path": log_path,
            "subprocess": subprocess,
        },
    )
    output = capsys.readouterr().out
    assert proc.waits == 2
    assert output.count("worker alive: yes") == 2
    assert output.count("worker alive: no") == 1
    assert "other-job" not in output
    assert "+00:00" in output
    if missing_status:
        assert "log unavailable" in output and "heartbeat unavailable" in output
    else:
        assert "current queue job: own-job" in output
        assert "last worker.log line: latest line" in output


def test_worker_shell_is_syntactically_valid_and_has_no_network_requirement():
    if not WORKER.exists():
        pytest.fail(f"worker script missing: {WORKER}")
    result = subprocess.run(["bash", "-n", str(WORKER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = WORKER.read_text() + "\n" + (WORKER.parent / "worker.py").read_text()
    assert "IDLE_MINUTES" in source
    assert "heartbeat" in source.lower()
    assert "failed" in source.lower() and "done" in source.lower()
    assert "runtime.unassign" in source or "google.colab" in source


def _make_repo(tmp_path: Path, script: str) -> tuple[Path, str]:
    """Create a clean commit-matching checkout for ``--local-test``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "queue@test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Queue Test"], check=True)
    (repo / "job.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + script)
    (repo / "job.sh").chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "add", "job.sh"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "queue test"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return repo, commit


def _spec(job_id: str, commit: str, output: str) -> dict:
    spec = {
        "id": job_id,
        "commit": commit,
        "script": "job.sh",
        "env": {},
        "requirements": {"gpu": "cpu", "min_vcpu": 1, "est_minutes": 1},
        "depends_on": [],
        "output_dir": output,
        "priority": 0,
    }
    return spec


def _worker_cmd(root: Path, repo: Path, worker_id: str, lock: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "research" / "colab_queue" / "worker.py"),
        "--root",
        str(root),
        "--repo",
        str(repo),
        "--worker-id",
        worker_id,
        "--local-test",
        "--lock-path",
        str(lock),
        "--settle-seconds",
        "0.08",
        "--poll-seconds",
        "0.03",
        "--heartbeat-seconds",
        "0.03",
        "--idle-minutes",
        "0.04",
    ]


def _start_workers(root: Path, repo: Path, count: int = 3) -> list[subprocess.Popen]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return [
        subprocess.Popen(
            _worker_cmd(root, repo, f"worker-{index}", root / f"lock-{index}"),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for index in range(count)
    ]


def _stop_workers(processes: list[subprocess.Popen]) -> list[str]:
    outputs = []
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=5)
        outputs.append(output)
    return outputs


def _wait_for_receipt(root: Path, job_id: str, bucket: str, timeout: float = 10) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipts = []
        for path in (root / bucket / job_id).glob("*.json"):
            receipts.append(json.loads(path.read_text()))
        if receipts:
            return receipts
        time.sleep(0.03)
    return []


def test_three_local_workers_execute_each_job_once(tmp_path):
    repo, commit = _make_repo(
        tmp_path,
        'printf "%s:%s\\n" "$QUEUE_JOB_ID" "$QUEUE_WORKER_ID" >> "$QUEUE_OUTPUT_DIR/effects.txt"\n',
    )
    drive = tmp_path / "drive"
    root = drive / "queue"
    queue = core.Queue(root)
    queue.ensure_layout()
    for job_id in ("job-a", "job-b", "job-c"):
        output = drive / "outputs" / job_id
        output.mkdir(parents=True)
        queue.enqueue(_spec(job_id, commit, f"outputs/{job_id}"))

    processes = _start_workers(drive, repo)
    try:
        for job_id in ("job-a", "job-b", "job-c"):
            assert _wait_for_receipt(root, job_id, "done"), job_id
    finally:
        logs = _stop_workers(processes)
    effects = []
    for job_id in ("job-a", "job-b", "job-c"):
        effects.extend((drive / "outputs" / job_id / "effects.txt").read_text().splitlines())
    assert sorted(line.split(":", 1)[0] for line in effects) == ["job-a", "job-b", "job-c"]
    assert all("queue unavailable" not in log for log in logs)


def test_nonzero_job_exit_is_recorded_failed_with_log_and_exit_code(tmp_path):
    repo, commit = _make_repo(tmp_path, 'echo "boom"; exit 7\n')
    drive = tmp_path / "drive"
    root = drive / "queue"
    queue = core.Queue(root)
    queue.ensure_layout()
    (drive / "outputs" / "failed-job").mkdir(parents=True)
    queue.enqueue(_spec("failed-job", commit, "outputs/failed-job"))
    processes = _start_workers(drive, repo, count=1)
    try:
        receipts = _wait_for_receipt(root, "failed-job", "failed")
        assert receipts
        assert receipts[0]["exit_code"] == 7
        assert "boom" in (drive / receipts[0]["log"]).read_text()
    finally:
        _stop_workers(processes)


def test_missing_output_fails_one_job_then_next_valid_job_runs(tmp_path):
    repo, commit = _make_repo(tmp_path, 'echo "$QUEUE_JOB_ID" > "$QUEUE_OUTPUT_DIR/result.txt"\n')
    drive = tmp_path / "drive"
    root = drive / "queue"
    queue = core.Queue(root)
    queue.ensure_layout()
    queue.enqueue(_spec("bad-output", commit, "outputs/bad-output"))
    queue.enqueue(_spec("good-output", commit, "outputs/good-output"))
    (drive / "outputs" / "bad-output").rmdir()
    processes = _start_workers(drive, repo, count=1)
    try:
        failed = _wait_for_receipt(root, "bad-output", "failed")
        done = _wait_for_receipt(root, "good-output", "done")
        assert failed and "output directory missing" in failed[0]["error"]
        assert done
        assert (
            drive / "outputs" / "good-output" / "result.txt"
        ).read_text().strip() == "good-output"
    finally:
        _stop_workers(processes)


def test_timeout_kills_owned_process_group(tmp_path):
    instance = worker.Worker.__new__(worker.Worker)
    instance.args = SimpleNamespace(poll_seconds=0.02)
    instance.assert_owner = lambda: None
    instance.heartbeat = lambda: None
    log_path = tmp_path / "timeout.log"
    with log_path.open("wb") as log:
        with pytest.raises(TimeoutError, match="exceeded"):
            instance.run_process(["bash", "-c", "while true; do sleep 1; done"], log, 0.1)


def test_cancelled_job_is_fenced_and_worker_reaps_child(tmp_path):
    repo, commit = _make_repo(
        tmp_path,
        'echo "$$" > "$QUEUE_OUTPUT_DIR/child.pid"\nwhile true; do sleep 1; done\n',
    )
    drive = tmp_path / "drive"
    root = drive / "queue"
    queue = core.Queue(root)
    queue.ensure_layout()
    (drive / "outputs" / "cancel-job").mkdir(parents=True)
    queue.enqueue(_spec("cancel-job", commit, "outputs/cancel-job"))
    processes = _start_workers(drive, repo, count=1)
    try:
        deadline = time.monotonic() + 5
        running = False
        while time.monotonic() < deadline:
            heartbeat = root / "heartbeats" / "worker-0.json"
            pid_file = drive / "outputs" / "cancel-job" / "child.pid"
            if heartbeat.exists():
                data = json.loads(heartbeat.read_text())
                running = (
                    data.get("job_id") == "cancel-job"
                    and data.get("phase") == "running"
                    and pid_file.exists()
                )
            if running:
                break
            time.sleep(0.03)
        assert running, "worker did not reach running phase before cancellation"
        queue.cancel("cancel-job")
        receipts = _wait_for_receipt(root, "cancel-job", "failed", timeout=8)
        assert receipts
        assert "OwnershipLost" in receipts[0]["error"]
        pid = int((drive / "outputs" / "cancel-job" / "child.pid").read_text())
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.03)
        else:
            pytest.fail(f"cancelled child process {pid} remained alive")
    finally:
        _stop_workers(processes)


def test_idle_release_calls_colab_hook_only_through_explicit_function(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    worker.release_runtime("python-under-test")
    assert calls
    assert calls[0][0][0] == "python-under-test"
    assert "runtime.unassign()" in calls[0][0][-1]
    assert calls[0][1]["check"] is True


def test_release_retries_one_transient_unassign_failure(monkeypatch):
    calls = []

    def flaky_release(python):
        calls.append(python)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, [python])

    instance = worker.Worker.__new__(worker.Worker)
    instance.args = SimpleNamespace(colab_python="python-under-test")
    monkeypatch.setattr(worker, "release_runtime", flaky_release)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    instance.release()
    assert calls == ["python-under-test", "python-under-test"]


def test_failed_setup_clears_cached_commit_before_retry(monkeypatch, tmp_path):
    commit_a = "a" * 40
    commit_b = "b" * 40
    instance = worker.Worker.__new__(worker.Worker)
    instance.args = SimpleNamespace(
        local_test=False,
        setup_timeout_seconds=10,
    )
    instance.caps = {"gpu": "cpu", "vcpu": 1}
    instance.repo = tmp_path
    instance.prepared_commit = commit_a
    instance.phase = "idle"
    instance.heartbeat = lambda force=False: None
    commands = []
    statuses = iter([0, 1, 0, 0])

    def fake_run_process(command, log, timeout, env=None, cwd=None):
        commands.append(command)
        return next(statuses)

    git_outputs = iter([commit_a, "", commit_a, ""])

    def fake_check_output(command, **kwargs):
        return next(git_outputs)

    monkeypatch.setattr(instance, "run_process", fake_run_process)
    monkeypatch.setattr(worker.subprocess, "check_output", fake_check_output)
    log = (tmp_path / "setup.log").open("ab")
    try:
        instance.prepare({"commit": commit_a}, log)
        with pytest.raises(RuntimeError, match="setup exited 1"):
            instance.prepare({"commit": commit_b}, log)
        assert instance.prepared_commit is None
        instance.prepare({"commit": commit_a}, log)
    finally:
        log.close()
    setup_commands = [command for command in commands if command and command[0] == "bash"]
    assert [command[-1] for command in setup_commands] == [commit_b, commit_a]
