"""Queue graph and hard-admission contracts for the Q0/Q1 registration."""
from __future__ import annotations

import json

import pytest

from research.q0q1_20261017 import queue_jobs as q
from research.q0q1_20261017 import run_queue_job as runner


PIN = "a" * 40
SHA = "b" * 64
BINDS = q.Bindings(
    q0b_l4_receipt="/content/drive/MyDrive/t16_colab/q0b/l4.json", q0b_l4_receipt_sha256=SHA,
    study_plan="/content/drive/MyDrive/t16_colab/q1/study-plan.json", study_plan_sha256=SHA,
    runtime_parity_receipt="/content/drive/MyDrive/t16_colab/q1/runtime-parity.json", runtime_parity_receipt_sha256=SHA,
    mac_parity_reference="/content/drive/MyDrive/t16_colab/q1/mac-parity.json", mac_parity_reference_sha256=SHA,
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
    init = next(job for job in jobs if job["id"] == "q0-audit-b8-init")
    assert init["depends_on"] == ["q0-audit-t23-mass"]
    assert set(candidate["depends_on"]) == {"q0-audit-t23-mass", "q0-audit-b8-init", "q1-train-d-261701", b8["id"]}
    assert candidate["env"]["Q0Q1_PARITY_OWNER_ADMISSION"] == BINDS.parity_owner_admission
    assert all(job["requirements"]["gpu"] == "L4" for job in jobs if job["id"].startswith("q1-train"))
    assert {job["timeout_minutes"] for job in jobs if job["id"].startswith("q1-train")} == {90}


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


def test_out_dir_writes_one_queue_schema_object_per_job(tmp_path):
    assert q.main(["--commit", PIN, "--out-dir", str(tmp_path)]) == 0
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) == 122
    value = json.loads(paths[0].read_text())
    assert isinstance(value, dict) and value["id"] == paths[0].stem


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
    from research.redesign_m3c_20261008 import train_m3c
    parsed = train_m3c.build_parser().parse_args(command[3:])
    assert parsed.max_wall_seconds == 5100
    assert parsed.namespace == q.q1_bank.training_namespace("D")


def test_b8_fidelity_command_consumes_the_mass_observation_receipt(tmp_path):
    command = runner.command({"Q0Q1_JOB_KIND": "q0_b8_init", "Q0Q1_INPUT_ROOT": "/inputs",
                              "Q0Q1_T23_ROOT": "/t23", "Q0Q1_MASS_OBSERVATION_RECEIPT": "/mass.json"}, tmp_path)
    assert command[command.index("--observation-receipt") + 1] == "/mass.json"


def test_training_outcome_only_kills_registered_scientific_tripwires():
    assert runner.classify_training_outcome({"halt": "non-finite loss"}) == "KILL"
    assert runner.classify_training_outcome({"halt": "action collapse [99, 0]"}) == "KILL"
    assert runner.classify_training_outcome({"halt": "RuntimeError: Drive disconnected"}) == "INCOMPLETE"


def test_draft_guard_runs_before_job_admission_or_output_access():
    with pytest.raises(RuntimeError, match=q.REGISTRATION_STATUS):
        runner.require_owner_release({"Q0Q1_REGISTRATION_STATUS": q.REGISTRATION_STATUS})


def test_future_release_still_requires_explicit_owner_token():
    with pytest.raises(RuntimeError, match="owner release"):
        runner.require_owner_release({"Q0Q1_REGISTRATION_STATUS": "OWNER_REVIEWED"})
