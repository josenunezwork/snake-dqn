"""Drive conflict, publication, and generation regressions; temporary files only."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_queue import core


def spec(job_id: str = "job", **overrides) -> dict:
    return {
        "id": job_id,
        "commit": "a" * 40,
        "script": "research/job.sh",
        "env": {},
        "requirements": {"gpu": "any", "min_vcpu": 1, "est_minutes": 1},
        "depends_on": [],
        "output_dir": f"results/{job_id}",
        "priority": 1,
        **overrides,
    }


@pytest.fixture
def queue(tmp_path):
    instance = core.Queue(tmp_path / "drive" / "queue")
    instance.ensure_layout()
    return instance


def claim(queue, job, worker_id="worker", token="token"):
    return queue.write_claim(job, worker_id, token, {"gpu": "L4", "vcpu": 8})


def test_enqueue_publishes_only_after_worker_paths_exist(queue, monkeypatch):
    original = core.write_json

    def observed(path, value):
        if path.parent == queue.pending_dir:
            assert (queue.root.parent / value["output_dir"]).is_dir()
            assert (queue.claimed_dir / value["id"]).is_dir()
            assert (queue.done_dir / value["id"]).is_dir()
            assert (queue.failed_dir / value["id"]).is_dir()
        return original(path, value)

    monkeypatch.setattr(core, "write_json", observed)
    job = queue.enqueue(spec())

    def forbid_mkdir(*args, **kwargs):
        raise AssertionError("runtime must not mkdir on Drive")

    monkeypatch.setattr(Path, "mkdir", forbid_mkdir)
    claim(queue, job)
    queue.write_heartbeat("worker", {"phase": "running"})
    queue.write_receipt("job", 0, "token", "done", {})
    assert queue.job_status("job")["state"] == "done"


def test_claims_in_duplicate_queue_and_claimed_directories_participate(queue):
    job = queue.enqueue(spec())
    claim(queue, job, "worker-z", "z")
    other = queue.root.parent / "queue (1)" / "claimed (2)" / "job (1)"
    other.mkdir(parents=True)
    lower = {**queue.claims("job")[0], "worker_id": "worker-a", "token": "a"}
    (other / "whatever (1).json").write_text(json.dumps(lower))
    assert queue.winner("job")["token"] == "a"
    assert not queue.is_winner(job, "worker-z", "z")


def test_stale_cached_job_cannot_claim_new_generation(queue):
    job = queue.enqueue(spec())
    claim(queue, job)
    queue.cancel("job")
    queue.requeue("job", stale_owner_ack=True)
    with pytest.raises(core.QueueError):
        claim(queue, job, "stale", "stale")
    assert queue.claims("job", 1) == []


def test_partial_requeue_sync_never_admits_work(queue):
    job = queue.enqueue(spec())
    core.write_json(queue.pending_path("job"), {**job, "queue_generation": 1, "attempt": 1})
    assert queue.eligible_jobs({"gpu": "L4", "vcpu": 8}) == []
    with pytest.raises(core.QueueError):
        claim(queue, queue.load_jobs()["job"])


def test_losing_receipt_does_not_fence_winner_or_unblock_dependency(queue):
    job = queue.enqueue(spec())
    queue.enqueue(spec("dependent", depends_on=["job"]))
    claim(queue, job, "a", "winner")
    claim(queue, job, "z", "loser")
    queue.write_receipt("job", 0, "loser", "done", {})
    assert queue.job_status("job")["state"] == "claimed"
    assert queue.eligible_jobs({"gpu": "L4", "vcpu": 8}) == []
    queue.write_receipt("job", 0, "winner", "done", {})
    assert [job["id"] for job in queue.eligible_jobs({"gpu": "NVIDIA H100", "vcpu": 8})] == [
        "dependent"
    ]
    queue.cancel("job")
    assert queue.eligible_jobs({"gpu": "L4", "vcpu": 8}) == []


def test_duplicate_terminal_receipts_dedupe_but_disagreement_fails_closed(queue):
    job = queue.enqueue(spec())
    claim(queue, job)
    queue.write_receipt("job", 0, "token", "done", {"exit_code": 0})
    original = queue.receipt_path("done", "job", 0, "token")
    duplicate = original.with_name("duplicate (1).json")
    shutil.copyfile(original, duplicate)
    assert queue.job_status("job")["state"] == "done"
    data = json.loads(duplicate.read_text())
    data["exit_code"] = 99
    duplicate.write_text(json.dumps(data))
    with pytest.raises(core.QueueConflict):
        queue.job_status("job")


def test_conflicting_success_failure_never_unblocks_dependency(queue):
    job = queue.enqueue(spec())
    dependent = queue.enqueue(spec("dependent", depends_on=["job"]))
    claim(queue, job)
    queue.write_receipt("job", 0, "token", "done", {})
    queue.write_receipt("job", 0, "token", "failed", {})
    with pytest.raises(core.QueueConflict):
        queue.dependencies_done(dependent)


def test_missing_dependency_does_not_block_independent_job(queue):
    queue.enqueue(spec("dependent", depends_on=["later"]))
    queue.enqueue(spec("ready"))
    assert [job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 8})] == ["ready"]


def test_duplicate_control_directory_cancellation_is_respected(queue):
    queue.enqueue(spec())
    alternate = queue.root / "control (1)"
    alternate.mkdir()
    (alternate / "job (1).json").write_text(
        json.dumps(
            {
                "id": "job",
                "generation": 0,
                "cancelled": True,
            }
        )
    )
    assert queue.eligible_jobs({"gpu": "L4", "vcpu": 8}) == []


@pytest.mark.parametrize("output", [".", "../escape", "/tmp/escape", "queue", "queue/done"])
def test_unsafe_output_paths_rejected(queue, output):
    with pytest.raises(core.QueueError):
        queue.enqueue(spec(output_dir=output))


def test_cli_lifecycle_is_local_and_machine_readable(tmp_path):
    drive = tmp_path / "drive"
    source = tmp_path / "job.json"
    source.write_text(json.dumps(spec()))
    cli = Path(__file__).resolve().parents[1] / "research/colab_queue/q.py"

    def run(*args):
        result = subprocess.run(
            [sys.executable, str(cli), "--drive-root", str(drive), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    assert run("enqueue", str(source))["enqueued"]["id"] == "job"
    assert run("list")["jobs"][0]["state"] == "pending"
    assert run("cancel", "job")["control"]["cancelled"]
    assert run("status", "job")["state"] == "cancelled"
    assert run("requeue", "job", "--stale-owner-stopped")["control"]["generation"] == 1
    assert run("watch", "--once")["jobs"][0]["generation"] == 1


@pytest.mark.parametrize("terminal", [False, True])
def test_cached_eligible_job_cannot_join_after_execution_started(queue, terminal):
    job = queue.enqueue(spec())
    claim(queue, job, "z", "owner")
    if terminal:
        queue.write_receipt("job", 0, "owner", "done", {})
    else:
        queue.write_heartbeat("z", {"job_id": "job", "token": "owner", "phase": "running"})
    with pytest.raises(core.QueueError):
        claim(queue, job, "a", "late")
    assert queue.winner("job")["token"] == "owner"


def test_malformed_claim_fails_closed_instead_of_crashing_worker(queue):
    queue.enqueue(spec())
    path = queue.claimed_dir / "job" / "broken.json"
    path.write_text(json.dumps({"id": "job", "token": "broken", "generation": 0}))
    with pytest.raises(core.QueueConflict):
        queue.eligible_jobs({"gpu": "L4", "vcpu": 8})


def test_failed_notebook_download_cannot_start_old_bootstrap(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "notebooks/colab_queue_worker.ipynb").read_text())
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "worker.sh").write_text("old bootstrap")
    source = "".join(notebook["cells"][1]["source"])
    source = source.replace("/content/colab-queue", str(bootstrap)).replace("0" * 40, "a" * 40)

    def failed_fetch(*args, **kwargs):
        raise subprocess.CalledProcessError(22, "curl")

    monkeypatch.setattr(subprocess, "run", failed_fetch)
    namespace = {"WORKER_READY_REF": "old-success"}
    with pytest.raises(subprocess.CalledProcessError):
        exec(compile(source, "fetch-cell", "exec"), namespace)
    assert namespace["WORKER_READY_REF"] is None
    with pytest.raises(AssertionError, match="fetch cell"):
        exec("".join(notebook["cells"][2]["source"]), namespace)
    assert (bootstrap / "worker.sh").read_text() == "old bootstrap"


def test_eval_adapter_hashes_selected_archive_not_sidecar_path(tmp_path):
    import hashlib
    import os

    repo = Path(__file__).resolve().parents[1]
    source = (repo / "research/colab_queue/examples/t23_eval_shard.sh").read_text()
    source = source.replace("py=/content/t16-colab-venv/bin/python", f"py='{sys.executable}'")
    source = source.replace("/content/t23-queue-eval.XXXXXX", str(tmp_path / "stage.XXXXXX"))
    adapter = tmp_path / "adapter.sh"
    adapter.write_text(source)
    original = tmp_path / "original.tar.gz"
    original.write_bytes(b"original")
    seed = tmp_path / "copied" / "seed-7101"
    seed.mkdir(parents=True)
    archive = seed / "t23-seed-7101.tar.gz"
    archive.write_bytes(b"altered-copy")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    archive.with_suffix(".gz.sha256").write_text(f"{digest}  {original}\n")
    output = tmp_path / "output"
    output.mkdir()
    env = {
        **os.environ,
        "QUEUE_OUTPUT_DIR": str(output),
        "T23_ROOT": str(seed.parent),
        "T23_QUEUE_EXAMPLE_ALLOW": "1",
    }
    result = subprocess.run(["bash", str(adapter)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "archive SHA-256 mismatch" in result.stderr
    assert not list(output.iterdir())
