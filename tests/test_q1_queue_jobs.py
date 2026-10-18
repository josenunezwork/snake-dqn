"""Queue graph and hard-admission contracts for the Q0/Q1 registration."""
from __future__ import annotations

import pytest

from research.q0q1_20261017 import queue_jobs as q
from research.q0q1_20261017 import run_queue_job as runner


PIN = "a" * 40
SHA = "b" * 64
BINDS = q.Bindings(
    q0b_l4_receipt="/content/drive/MyDrive/t16_colab/q0b/l4.json", q0b_l4_receipt_sha256=SHA,
    study_plan="/content/drive/MyDrive/t16_colab/q1/study-plan.json", study_plan_sha256=SHA,
    runtime_parity_receipt="/content/drive/MyDrive/t16_colab/q1/runtime-parity.json", runtime_parity_receipt_sha256=SHA,
    eval_parity_receipt="/content/drive/MyDrive/t16_colab/q1/eval-parity.json", eval_parity_receipt_sha256=SHA,
    parity_owner_admission="/content/drive/MyDrive/t16_colab/q1/parity-owner.json", parity_owner_admission_sha256=SHA,
)


def test_q1_graph_uses_final_pin_shared_b8_and_all_required_dependencies():
    jobs = q.build_jobs(PIN, BINDS, eval_shards=2)
    assert len(jobs) == 2 + 20 + (5 * 5 * 2)
    assert {job["commit"] for job in jobs} == {PIN}
    assert all(job["env"]["Q0Q1_REGISTRATION_STATUS"] == q.REGISTRATION_STATUS for job in jobs)
    b8 = next(job for job in jobs if job["id"] == "q1-eval-b8-261701-s0of2")
    candidate = next(job for job in jobs if job["id"] == "q1-eval-d-261701-s0of2")
    assert b8["depends_on"] == ["q0-audit-t23-mass", "q0-audit-b8-init"]
    assert set(candidate["depends_on"]) == {"q0-audit-t23-mass", "q0-audit-b8-init", "q1-train-d-261701", b8["id"]}
    assert candidate["env"]["Q0Q1_PARITY_OWNER_ADMISSION"] == BINDS.parity_owner_admission
    assert all(job["requirements"]["gpu"] == "L4" for job in jobs if job["id"].startswith("q1-train"))


def test_generator_rejects_placeholder_or_unbound_evidence():
    with pytest.raises(ValueError, match="immutable"):
        q.build_jobs("0" * 40, BINDS)
    with pytest.raises(ValueError, match="placeholder"):
        q.build_jobs(PIN, q.Bindings(BINDS.q0b_l4_receipt, "0" * 64))


def test_draft_graph_can_be_materialized_without_future_evidence_pins():
    jobs = q.build_jobs(PIN, q.Bindings())
    assert len(jobs) == 122
    assert "Q0Q1_Q0B_L4_RECEIPT" not in jobs[2]["env"]
    assert jobs[2]["env"]["Q0Q1_REGISTRATION_STATUS"] == q.REGISTRATION_STATUS


def test_train_command_pins_the_registered_t23_control_and_q1_treatments(tmp_path):
    env = {
        "Q0Q1_JOB_KIND": "q1_train", "Q0Q1_INPUT_ROOT": "/inputs", "Q1_SEED": "261701",
        "Q1_TRANSITIONS": "8000000", "Q1_ARM": "D", "Q1_ACTORS": "3",
    }
    command = runner.command(env, tmp_path)
    for flag, value in (("--worlds", "128"), ("--checkpoint-every", "500000"),
                        ("--reward", "mass"), ("--kill-scale", "0"), ("--obs-version", "bv8"),
                        ("--batch-scale", "2"), ("--actor-queue-frames", "2"),
                        ("--collection-profile", "task-aligned-n1"), ("--learner-seats", "1")):
        index = command.index(flag)
        assert command[index + 1] == value
    assert "--onpolicy-labels" in command and "--learner-fast" in command


def test_draft_guard_runs_before_job_admission_or_output_access():
    with pytest.raises(RuntimeError, match=q.REGISTRATION_STATUS):
        runner.require_owner_release({"Q0Q1_REGISTRATION_STATUS": q.REGISTRATION_STATUS})


def test_future_release_still_requires_explicit_owner_token():
    with pytest.raises(RuntimeError, match="owner release"):
        runner.require_owner_release({"Q0Q1_REGISTRATION_STATUS": "OWNER_REVIEWED"})
