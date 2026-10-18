"""Local contracts for the Drive-backed Colab queue."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.colab_queue import core

Q_CLI = Path(__file__).resolve().parents[1] / "research" / "colab_queue" / "q.py"


def _job(
    job_id: str,
    *,
    gpu: str = "any",
    min_vcpu: int = 1,
    depends_on: list[str] | None = None,
    priority: int = 0,
) -> dict:
    return {
        "id": job_id,
        "commit": "a" * 40,
        "script": "research/example.py",
        "env": {"SNAKE_TEST": job_id},
        "requirements": {"gpu": gpu, "min_vcpu": min_vcpu, "est_minutes": 1},
        "depends_on": depends_on or [],
        "output_dir": f"outputs/{job_id}",
        "priority": priority,
    }


def _queue(path: Path) -> core.Queue:
    queue = core.Queue(path)
    queue.ensure_layout()
    return queue


def _claim(queue: core.Queue, job: dict, worker_id: str, token: str) -> dict:
    return queue.write_claim(job, worker_id, token, {"gpu": "L4", "vcpu": 8})


def test_layout_precreates_all_shared_directories_and_enqueue_schema(tmp_path):
    queue = _queue(tmp_path)
    for dirname in core.QUEUE_DIRS:
        assert (tmp_path / dirname).is_dir(), dirname

    stored = queue.enqueue(_job("train-1", gpu="L4", min_vcpu=4, priority=7))
    assert json.loads((tmp_path / "pending" / "train-1.json").read_text()) == stored
    assert stored["requirements"] == {"gpu": "L4", "min_vcpu": 4, "est_minutes": 1}
    assert queue.load_jobs() == {"train-1": stored}


def test_duplicate_drive_directory_same_identity_is_deduplicated(tmp_path):
    queue = _queue(tmp_path)
    stored = queue.enqueue(_job("same"))
    duplicate = tmp_path / "pending (1)"
    duplicate.mkdir()
    (duplicate / "same.json").write_text(json.dumps(stored))
    assert queue.load_jobs() == {"same": stored}


def test_duplicate_drive_directory_divergent_identity_is_a_conflict(tmp_path):
    queue = _queue(tmp_path)
    stored = queue.enqueue(_job("same", priority=1))
    duplicate = tmp_path / "pending (1)"
    duplicate.mkdir()
    (duplicate / "same.json").write_text(json.dumps({**stored, "priority": 2}))
    with pytest.raises(core.QueueConflict):
        queue.load_jobs()


def test_three_workers_claim_same_job_and_lexical_winner_is_unique(tmp_path):
    queue = _queue(tmp_path)
    job = queue.enqueue(_job("race"))
    barrier = threading.Barrier(3)

    def claim(worker_id: str):
        barrier.wait(timeout=2)
        return _claim(queue, job, worker_id, f"token-{worker_id}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(claim, ("worker-c", "worker-a", "worker-b")))
    assert len(results) == 3
    claims = list((tmp_path / "claimed" / "race").glob("*.json"))
    assert len(claims) == 3
    assert {json.loads(path.read_text())["worker_id"] for path in claims} == {
        "worker-a",
        "worker-b",
        "worker-c",
    }
    assert queue.winner("race")["worker_id"] == "worker-a"
    assert (
        sum(
            queue.is_winner(job, worker_id, f"token-{worker_id}")
            for worker_id in ("worker-a", "worker-b", "worker-c")
        )
        == 1
    )


def test_duplicate_claim_filename_is_not_atomic_or_a_second_winner(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue(_job("race"))
    claim_dir = tmp_path / "claimed" / "race"
    claim_dir.mkdir(parents=True, exist_ok=True)
    first = {
        "id": "race",
        "worker_id": "worker-a",
        "token": "a",
        "generation": 0,
        "attempt": 0,
        "created_at": core.utc_now(),
        "capabilities": {"gpu": "L4", "vcpu": 8},
    }
    second = {**first, "worker_id": "worker-b", "token": "b"}
    (claim_dir / "g0.worker-a.a.json").write_text(json.dumps(first))
    (claim_dir / "g0.worker-a.a (1).json").write_text(json.dumps(first))
    (claim_dir / "g0.worker-b.b.json").write_text(json.dumps(second))
    assert queue.winner("race")["worker_id"] == "worker-a"
    assert len(queue.claims("race")) == 2


def test_dependencies_and_gpu_vcpu_matching(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue(_job("base", gpu="A100", min_vcpu=8))
    queue.enqueue(_job("dependent", depends_on=["base"], gpu="any", min_vcpu=4))
    queue.enqueue(_job("cpu", gpu="cpu", min_vcpu=2))
    queue.enqueue(_job("l4", gpu="L4", min_vcpu=8))

    assert {job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 8})} == {"l4"}
    base = queue.load_jobs()["base"]
    _claim(queue, base, "worker-base", "done-token")
    queue.write_receipt("base", 0, "done-token", "done", {})
    assert {job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 8})} == {
        "dependent",
        "l4",
    }
    assert "cpu" in {job["id"] for job in queue.eligible_jobs({"gpu": "cpu", "vcpu": 2})}
    assert "cpu" not in {job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 8})}
    assert "dependent" not in {job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 2})}
    assert "l4" not in {job["id"] for job in queue.eligible_jobs({"gpu": "A100", "vcpu": 8})}


def test_cancel_fences_job_and_requeue_invalidates_stale_claim(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue(_job("fenced"))
    job = queue.load_jobs()["fenced"]
    _claim(queue, job, "worker-a", "old")
    queue.cancel("fenced")
    assert "fenced" not in {job["id"] for job in queue.eligible_jobs({"gpu": "L4", "vcpu": 8})}

    queue.requeue("fenced", stale_owner_ack=True)
    job = queue.load_jobs()["fenced"]
    _claim(queue, job, "worker-b", "new")
    assert queue.winner("fenced")["token"] == "new"
    assert queue.is_winner(job, "worker-b", "new")


def test_stale_heartbeat_is_detectable_by_age(tmp_path):
    queue = _queue(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()
    queue.write_heartbeat("old-worker", {"phase": "running"})
    queue.write_heartbeat("new-worker", {"phase": "idle"})
    old_path = tmp_path / "heartbeats" / "old-worker.json"
    old_record = json.loads(old_path.read_text())
    old_record["updated_at"] = old
    core.write_json(old_path, old_record)
    new_path = tmp_path / "heartbeats" / "new-worker.json"
    new_record = json.loads(new_path.read_text())
    new_record["updated_at"] = fresh
    core.write_json(new_path, new_record)
    status = queue.heartbeat_status(stale_seconds=600, now=time.time())
    assert status["old-worker"]["stale"]
    assert not status["new-worker"]["stale"]


def test_q_cli_enqueue_status_cancel_requeue_and_watch_once(tmp_path):
    drive = tmp_path / "drive"
    spec_path = tmp_path / "job.json"
    spec_path.write_text(json.dumps(_job("cli-job", gpu="cpu")))

    def run(*args: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(Q_CLI), "--drive-root", str(drive), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    assert run("enqueue", str(spec_path))["enqueued"]["id"] == "cli-job"
    assert run("status", "cli-job")["state"] == "pending"
    assert run("cancel", "cli-job")["control"]["cancelled"] is True
    assert run("status", "cli-job")["state"] == "cancelled"
    assert run("requeue", "cli-job", "--stale-owner-stopped")["control"]["generation"] == 1
    assert run("status", "cli-job")["state"] == "pending"
    snapshot = run("watch", "--once")
    assert snapshot["jobs"][0]["id"] == "cli-job"
