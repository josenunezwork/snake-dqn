"""Exercise the complete coordinator flow with allocation/evaluation boundaries faked."""

import argparse
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, run


@pytest.mark.parametrize("tripwire", [False, True])
@pytest.mark.parametrize("continuation", [False, True, "prior_retry"])
def test_serial_dispatch_completes_the_registered_look_or_skips_only_killed_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tripwire: bool, continuation: bool | str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    seeds = (11, 12, 13, 14, 15)
    monkeypatch.setattr(run.bank, "TRAINING_SEEDS", seeds)
    manifests, liabilities = [], {}
    for seed in seeds:
        for arm in ("n3", "n1"):
            job = f"s1-mature-{arm}-{seed}"
            path = tmp_path / f"{job}.json"
            admission.write_once(
                path,
                {
                    "job_id": job,
                    "output_dir": str(root / "training" / job),
                    "env": {"S1_ARM": arm, "S1_SEED": str(seed)},
                    "inputs": [{"path": str(tmp_path / "b8"), "dest": "B8_s3_final.pth"}],
                },
            )
            manifests.append({"job_id": job, "path": str(path), "sha256": admission.sha256(path)})
            liabilities[job] = {
                "units": 1,
                "gpu_allocation_hours": 1,
                "mac_worker_hours": 0,
                "elapsed_hours": 1,
            }
    mac_path = tmp_path / "mac.json"
    admission.write_once(mac_path, {"kind": "mac"})
    liabilities["mac-final-evaluation"] = {
        "units": 0,
        "gpu_allocation_hours": 0,
        "mac_worker_hours": 2,
        "elapsed_hours": 1,
    }
    bank_path = tmp_path / "WORLD_BANK.json"
    admission.write_once(bank_path, {})
    admission.write_once(tmp_path / "SEED_AUDIT.json", {})
    plan = {
        "manifests": manifests,
        "liabilities": liabilities,
        "mac_manifest": {"path": str(mac_path), "sha256": admission.sha256(mac_path)},
        "world_bank": {"path": str(bank_path)},
    }
    plan_path = tmp_path / "plan.json"
    admission.write_once(plan_path, plan)
    monkeypatch.setattr(run, "_verify_plan", lambda *_: plan)
    events = []

    class FakeLedger:
        def __init__(self, *_):
            self.active = None

        def status(self):
            return {}

        def reserve_manual_retry(self, source, job, path):
            assert source == manifests[1]["job_id"]
            self.reserve(job, None, path)

        def initialize(self, **_):
            events.append("initialize")

        def record_account_snapshot(self, _):
            events.append("snapshot")

        def reserve(self, job, *_):
            assert self.active is None
            self.active = job
            events.append(("reserve", job))

        def settle(self, job, *_):
            assert self.active == job
            self.active = None
            events.append(("settle", job))

        settle_mac = settle

    monkeypatch.setattr(run.ledger, "Ledger", FakeLedger)
    monkeypatch.setattr(run, "_snapshot", lambda path, **_: path)
    monkeypatch.setattr(
        run.audit,
        "validated_halts",
        lambda value, _: {item["arm"] for item in value.get("training_outcomes", [])},
    )

    recovery_jobs = []

    def recover(path, _):
        manifest = json.loads(path.read_text())
        recovery_jobs.append(manifest["job_id"])
        env = manifest["env"]
        outcome = {"arm": "N3"} if tripwire and env["S1_ARM"] == "n3" else None
        return {"checkpoint": "fixture.pth", "provenance": "fixture.json"}, outcome

    monkeypatch.setattr(run, "_recover_training", recover)
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append((kwargs["arm"], kwargs["seed"]))
        target = root / f"eval-{len(evaluations)}.json"
        admission.write_once(target, {"fixture": True})
        return target

    monkeypatch.setattr(run.evaluate, "run", evaluate)

    def boundary(command, **_):
        events.append(("boundary", str(command[0])))

    monkeypatch.setattr(run.subprocess, "run", boundary)

    class EvalProcess:
        returncode = 0

        def __init__(self, command, *, start_new_session):
            assert start_new_session is True
            run._evaluate_all(plan, root, bank_path)

        def wait(self, **_):
            return 0

    monkeypatch.setattr(run.subprocess, "Popen", EvalProcess)
    history = []
    for name in ("q1", "a2", "a3"):
        target = tmp_path / name
        target.touch()
        history.append(f"{name}={target}")
    args = argparse.Namespace(
        plan=plan_path,
        repo=admission.ROOT,
        ledger=tmp_path / "ledger.json",
        acceptance_review=tmp_path / "review.json",
        snapshots=tmp_path / "snapshots",
        history=history,
        state_root=tmp_path / "states",
        artifact_root=root,
        supervisor=admission.ROOT / "research/colab_cli/launch.sh",
        colab=tmp_path / "never-invoked",
        account_config=tmp_path / "config.json",
    )
    if continuation:
        source_id = manifests[1]["job_id"]
        retry_path = tmp_path / "retry.json"
        retry = json.loads(Path(manifests[1]["path"]).read_text())
        retry["job_id"] = "retry-n1"
        retry["output_dir"] = str(root / "training/retry-n1")
        admission.write_once(retry_path, retry)
        record = {
            "source_job_id": source_id,
            "retry_job_id": "retry-n1",
            "retry_manifest": {"path": str(retry_path)},
            "source_manifest": {"path": manifests[1]["path"]},
            "operational_failure_review": {"sha256": "review"},
            "state_dir": str(tmp_path / "retry-state"),
        }
        prior_entry = {"receipt": {}}
        if continuation == "prior_retry":
            prior_path = tmp_path / "prior-retry.json"
            prior_manifest = json.loads(Path(manifests[0]["path"]).read_text())
            prior_manifest["job_id"] = "retry-n3"
            admission.write_once(prior_path, prior_manifest)
            prior_entry["operational_failure_review"] = {"sha256": "prior-review"}
        admission.write_once(
            args.ledger,
            {
                "settled": {
                    manifests[0]["job_id"]: prior_entry,
                    source_id: {"operational_failure_review": {"sha256": "review"}},
                },
                "planned": {"retry-n1": {}},
            },
        )
        if continuation == "prior_retry":
            state = json.loads(args.ledger.read_text())
            state["settled"]["retry-n3"] = {
                "retry_of": manifests[0]["job_id"],
                "source_manifest": {
                    "path": str(prior_path),
                    "sha256": admission.sha256(prior_path),
                },
            }
            args.ledger.write_text(json.dumps(state))
        monkeypatch.setattr(run, "_retry_record", lambda *_: record)
        args.retry_record = tmp_path / "record.json"
        run.continue_dispatch(args)
        assert "initialize" not in events
        if continuation == "prior_retry":
            assert "retry-n3" in recovery_jobs
            assert manifests[0]["job_id"] not in recovery_jobs
    else:
        run.dispatch(args)
    reserves = [event[1] for event in events if isinstance(event, tuple) and event[0] == "reserve"]
    assert len(reserves) == (7 if tripwire else 11) - int(bool(continuation))
    assert reserves[-1] == "mac-final-evaluation"
    assert len(evaluations) == (10 if tripwire else 15)
    assert {seed for arm, seed in evaluations if arm == "B8"} == set(seeds)
    if tripwire:
        assert not any(arm == "N3" for arm, _ in evaluations)
    assert json.loads((root / "mac-final-evaluation-receipt.json").read_text())["status"] == "PASS"


def test_final_recovery_is_idempotent_and_rejects_changed_bytes(tmp_path: Path) -> None:
    from research.s1_mature_20261018 import train

    source = tmp_path / "run"
    source.mkdir()
    for name in ("final.pth", "target.pth", "summary.json"):
        (source / name).write_bytes(b"bound fixture")
    archive = tmp_path / "source.tar.gz"
    train._archive_final(source, archive)
    exports = tmp_path / "exports"
    train._publish_chunked(archive, exports, "s1-final")
    recovered_archive = tmp_path / "recovered.tar.gz"
    destination = tmp_path / "recovered"
    for _ in range(2):
        run._reassemble_final(exports / "s1-final.index.json", exports, recovered_archive)
        checkpoint, target, summary = run._safe_extract_final(recovered_archive, destination)
        assert checkpoint.name == "final.pth"
        assert target.name == "target.pth"
    summary.write_bytes(b"changed fixture")
    with pytest.raises(admission.AdmissionRefused, match="existing recovered final bytes changed"):
        run._safe_extract_final(recovered_archive, destination)
