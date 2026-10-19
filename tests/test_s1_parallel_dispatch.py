"""Offline scheduler coverage for S1's parallel dispatcher."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, run


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tripwire: bool = False):
    root = tmp_path / "artifacts"
    root.mkdir()
    amendment = tmp_path / "amendment.md"
    amendment.write_text("approved")
    rows, liabilities = [], {}
    # Registered ordering is deliberately interleaved.
    for arm, seed in (("n3", 1), ("n1", 1), ("n3", 2), ("n1", 2)):
        job = f"s1-mature-{arm}-{seed}"
        path = tmp_path / f"{job}.json"
        admission.write_once(
            path,
            {
                "job_id": job,
                "output_dir": str(root / "training" / job),
                "env": {"S1_ARM": arm, "S1_SEED": str(seed)},
            },
        )
        rows.append((job, path))
        liabilities[job] = dict(
            units=1, gpu_allocation_hours=1, mac_worker_hours=0, elapsed_hours=1
        )
    plan = {
        "parallel_amendment": {"path": str(amendment), "sha256": admission.sha256(amendment)},
        "run_rules": {"lanes": 3, "serialized": False},
        "liabilities": liabilities,
    }

    class FakeLedger:
        live: set[str] = set()
        claimed: list[str] = []
        settled: list[str] = []
        configured = False
        lock = threading.Lock()

        def __init__(self, *_):
            pass

        def initialize(self, **_):
            return {}

        def configure_parallel(self, *_args, **_kwargs):
            self.configured = True

        def claim_next(self, ordered, _liabilities, _paths, **_kwargs):
            with self.lock:
                for job in ordered:
                    if job not in self.claimed:
                        self.claimed.append(job)
                        self.live.add(job)
                        return job
            return None

        def settle(self, job, *_):
            with self.lock:
                self.live.remove(job)
                self.settled.append(job)

        def halt_arm(self, *_):
            return {}

    fake = FakeLedger()
    monkeypatch.setattr(run.ledger, "Ledger", lambda *_: fake)

    def snapshot(path: Path, **_):
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs = [str(root / "training" / job) for job in fake.live]
        admission.write_once(
            path,
            {
                "active_assignments": len(outputs),
                "sessions": {},
                "active_records": [{"output": x} for x in outputs],
            },
        )
        return path

    monkeypatch.setattr(run, "_snapshot", snapshot)
    monkeypatch.setattr(run.admission, "delivery_preflight", lambda *_: None)
    monkeypatch.setattr(run.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_finish_evaluation", lambda *_: None)

    def recover(path: Path, _root: Path):
        arm = run._read(path)["env"]["S1_ARM"]
        return (
            (
                {"checkpoint": "fixture"},
                {"arm": "N3", "tripwire": True, "evidence": {"summary": {}}},
            )
            if tripwire and arm == "n3"
            else ({"checkpoint": "fixture"}, None)
        )

    monkeypatch.setattr(run, "_recover_training", recover)
    args = argparse.Namespace(
        lanes=3,
        ledger=tmp_path / "ledger.json",
        parallel_handoff=None,
        previous_plan=None,
        plan=tmp_path / "plan.json",
        snapshots=tmp_path / "snapshots",
        colab=tmp_path / "never",
        account_config=tmp_path / "config",
        acceptance_review=tmp_path / "review",
        history=[f"{name}={tmp_path / name}" for name in ("q1", "a2", "a3")],
        state_root=tmp_path / "states",
        artifact_root=root,
        supervisor=tmp_path / "supervisor",
    )
    for name in ("q1", "a2", "a3"):
        (tmp_path / name).touch()
    return args, plan, rows, fake


def test_parallel_dispatch_claims_each_registered_job_once_and_exhausts(tmp_path, monkeypatch):
    args, plan, rows, fake = _fixture(tmp_path, monkeypatch)
    first_wave = threading.Barrier(3, timeout=3)

    def boundary(command, **_):
        if any(command[-1] == str(path) for _, path in rows[:3]):
            first_wave.wait()

    monkeypatch.setattr(run.subprocess, "run", boundary)
    run._parallel_dispatch(args, plan, rows, {"status": "PASS"})
    assert fake.claimed == [job for job, _ in rows]
    assert set(fake.settled) == set(fake.claimed)


def test_parallel_dispatch_tripwire_stops_future_arm_claims(tmp_path, monkeypatch):
    args, plan, rows, fake = _fixture(tmp_path, monkeypatch, tripwire=True)
    args.lanes = plan["run_rules"]["lanes"] = 2
    run._parallel_dispatch(args, plan, rows, {"status": "PASS"})
    assert set(fake.claimed) == {job for job, _ in rows if job != "s1-mature-n3-2"}
    assert len(fake.claimed) == 3


def test_parallel_dispatch_requires_hash_bound_amendment(tmp_path, monkeypatch):
    args, plan, rows, _ = _fixture(tmp_path, monkeypatch)
    plan.pop("parallel_amendment")
    with pytest.raises(admission.AdmissionRefused, match="hash-bound amendment"):
        run._parallel_dispatch(args, plan, rows, {"status": "PASS"})


def test_parallel_lane_failure_does_not_cancel_already_claimed_healthy_lane(tmp_path, monkeypatch):
    args, plan, rows, fake = _fixture(tmp_path, monkeypatch)

    def boundary(command, **_):
        if command[-1].endswith("s1-mature-n3-1.json"):
            time.sleep(0.05)
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run.subprocess, "run", boundary)
    with pytest.raises(subprocess.CalledProcessError):
        run._parallel_dispatch(args, plan, rows, {"status": "PASS"})
    # The other concurrently claimed lane settles; no cancellation is issued.
    assert fake.settled


def test_account_failure_fences_new_claims_while_submitted_work_finishes(tmp_path, monkeypatch):
    args, plan, rows, fake = _fixture(tmp_path, monkeypatch)
    original_snapshot = run._snapshot
    child_started = threading.Event()
    account_failed = threading.Event()

    def boundary(*_, **__):
        child_started.set()
        assert account_failed.wait(3)

    def snapshot(path, **kwargs):
        if fake.claimed:
            assert child_started.wait(3)
            account_failed.set()
            raise admission.AdmissionRefused("unknown assignment")
        return original_snapshot(path, **kwargs)

    monkeypatch.setattr(run.subprocess, "run", boundary)
    monkeypatch.setattr(run, "_snapshot", snapshot)
    with pytest.raises(admission.AdmissionRefused, match="unknown assignment"):
        run._parallel_dispatch(args, plan, rows, {"status": "PASS"})
    assert len(fake.claimed) == 1
    assert fake.settled == fake.claimed


@pytest.mark.parametrize("defect", [None, "runtime_source", "seed", "liability", "mac"])
def test_partial_handoff_recovers_original_finals_and_refuses_science_changes(
    tmp_path, monkeypatch, defect
):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_rows, new_rows, liabilities = [], [], {}
    for seed in range(5):
        for arm in ("n3", "n1"):
            job = f"s1-mature-{arm}-{seed}"
            before = {
                "job_id": job,
                "env": {"S1_ARM": arm, "S1_SEED": str(seed)},
                "inputs": [],
                "output_dir": str(old_root / "training" / job),
            }
            after = {**before, "output_dir": str(new_root / "training" / job)}
            if defect == "seed" and not new_rows:
                after["env"] = {**before["env"], "S1_SEED": "999"}
            for root, value, rows in ((old_root, before, old_rows), (new_root, after, new_rows)):
                path = root / "jobs" / f"{job}.json"
                admission.write_once(path, value)
                rows.append({"job_id": job, "path": str(path), "sha256": admission.sha256(path)})
            liabilities[job] = dict(
                units=2, gpu_allocation_hours=1, mac_worker_hours=0, elapsed_hours=1
            )
    liabilities["mac-final-evaluation"] = dict(
        units=0, gpu_allocation_hours=0, mac_worker_hours=2, elapsed_hours=1
    )
    plans = []
    for root, rows, serial in ((old_root, old_rows, True), (new_root, new_rows, False)):
        mac = root / "jobs" / "mac.json"
        admission.write_once(mac, {"workers": 2})
        plans.append(
            {
                "schema": admission.SCHEMA,
                "status": "PREPARED_NOT_DISPATCHED",
                "run_rules": {"serialized": serial},
                "source_closure": {"runtime.py": "same"},
                "manifests": rows,
                "liabilities": liabilities,
                "mac_manifest": {"path": str(mac), "sha256": admission.sha256(mac)},
            }
        )
    old_plan = old_root / "plan.json"
    admission.write_once(old_plan, plans[0])
    if defect == "runtime_source":
        plans[1]["source_closure"]["runtime.py"] = "changed"
    if defect == "mac":
        Path(plans[1]["mac_manifest"]["path"]).write_text('{"workers":3}')
    completed = old_rows[0]["job_id"]
    receipt = old_root / "receipt.json"
    admission.write_once(receipt, {"status": "PASS"})
    state = {
        "outstanding": {},
        "settled": {
            completed: {"receipt": {"path": str(receipt)}},
            "qualification-failure:historical": {"preserved": True},
        },
        "planned": {job: bound.copy() for job, bound in liabilities.items() if job != completed},
    }
    if defect == "liability":
        state["planned"]["mac-final-evaluation"]["elapsed_hours"] = 0
    ledger_path = tmp_path / "ledger.json"
    admission.write_once(ledger_path, state)
    original = ledger_path.read_bytes()
    recovered_paths = []

    class ReceiptValidator:
        def _receipt(self, path, binding):
            assert path == receipt
            assert binding["path"] == old_rows[0]["path"]

    def recover(path, root):
        recovered_paths.append(path)
        assert root == new_root
        return {"checkpoint": "original-final"}, None

    monkeypatch.setattr(run, "_recover_training", recover)
    args = argparse.Namespace(previous_plan=old_plan, artifact_root=new_root, ledger=ledger_path)
    if defect is not None:
        with pytest.raises(admission.AdmissionRefused):
            run._parallel_resume(args, plans[1], ReceiptValidator())
        assert not recovered_paths
    else:
        recovered, outcomes, halted = run._parallel_resume(args, plans[1], ReceiptValidator())
        assert recovered == {completed: {"checkpoint": "original-final"}}
        assert outcomes == [] and halted == set()
        assert recovered_paths == [Path(old_rows[0]["path"])]
    assert json.loads(ledger_path.read_text()) == state
    assert ledger_path.read_bytes() == original
