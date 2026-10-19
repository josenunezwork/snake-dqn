"""Focused non-allocating tests for the S1 admission boundary."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, final_artifacts, run


def _final_transport(tmp_path: Path) -> dict:
    from tests.test_colab_cli_admission import passing_result

    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(passing_result.__wrapped__(tmp_path)))
    return final_artifacts.acceptance_transport(acceptance)


def test_preregistration_requires_prospective_approval_line(tmp_path: Path) -> None:
    draft = tmp_path / "PREREGISTRATION.md"
    draft.write_text("**Status: DRAFT_PENDING_OWNER_REVIEW.**\n")
    with pytest.raises(admission.AdmissionRefused):
        admission.require_approved_preregistration(draft)
    # The old draft text is intentionally retained in the frozen history.
    draft.write_text(
        "**Status: APPROVED_S1_MATURE_20261018.** Approved by planner before data.\n"
        "old DRAFT_PENDING_OWNER_REVIEW bytes\n"
    )
    assert admission.require_approved_preregistration(draft)["sha256"] == admission.sha256(draft)


def test_liabilities_reject_wrong_arm_timing() -> None:
    good = []
    for arm, minutes, timeout in (("n3", 30, 60), ("n1", 68, 136)):
        for seed in range(5):
            good.append(
                {
                    "job_id": f"{arm}-{seed}",
                    "env": {"S1_ARM": arm},
                    "requirements": {"est_minutes": minutes},
                    "timeout_minutes": timeout,
                }
            )
    bound = admission.liabilities(good)
    assert len(bound) == 10
    assert not set(bound).intersection({"retry-n3", "retry-n1"})
    assert bound["n3-0"]["gpu_allocation_hours"] == pytest.approx(65 / 60)
    assert bound["n1-0"]["units"] == pytest.approx(141 / 30)
    good[0]["timeout_minutes"] = 136
    with pytest.raises(admission.AdmissionRefused):
        admission.liabilities(good)


def test_final_export_recovery_rejects_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("bad")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../outside")
    with pytest.raises(admission.AdmissionRefused):
        run._safe_extract_final(archive, tmp_path / "extract")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_pricing_rejects_nonfinite_bounds(tmp_path: Path, number: float) -> None:
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "status": "PASS",
                "counter_admission": "PASS",
                "identity": {
                    "source_manifest": {
                        relative: admission.sha256(admission.ROOT / relative)
                        for relative in (
                            "research/redesign_m3c_20261008/train_m3c.py",
                            "research/s1_mature_20261018/train.py",
                        )
                    }
                },
            }
        )
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema": "s1-eval-timing/v1",
                "status": "completed",
                "episode_count": 10,
                "identity": {"source_manifest": {}},
            }
        )
    )
    receipt = tmp_path / "pricing.json"
    receipt.write_text(
        __import__("json").dumps(
            {
                "status": "PASS",
                "evidence": {
                    name: {"path": str(item), "sha256": admission.sha256(item)}
                    for name, item in (("training", training), ("evaluation", evaluation))
                },
                "complete_upper_bound": {"mac_worker_hours": number, "elapsed_hours": 1},
                "mac_upper_bound": {"elapsed_hours": 1},
                "final_artifact_transport": _final_transport(tmp_path),
            }
        )
    )
    with pytest.raises(admission.AdmissionRefused):
        admission.priced_mac_bound(receipt)


def test_pricing_requires_current_counter_sources_and_bound_final_transfer(tmp_path: Path) -> None:
    sources = {
        relative: admission.sha256(admission.ROOT / relative)
        for relative in (
            "research/redesign_m3c_20261008/train_m3c.py",
            "research/s1_mature_20261018/train.py",
        )
    }
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"status": "PASS", "counter_admission": "PASS"}))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema": "s1-eval-timing/v1",
                "status": "completed",
                "episode_count": 10,
                "identity": {"source_manifest": {}},
            }
        )
    )

    def priced() -> Path:
        path = tmp_path / "pricing.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "evidence": {
                        name: {"path": str(item), "sha256": admission.sha256(item)}
                        for name, item in (("training", training), ("evaluation", evaluation))
                    },
                    "complete_upper_bound": {"mac_worker_hours": 1, "elapsed_hours": 1},
                    "mac_upper_bound": {"elapsed_hours": 1},
                    "final_artifact_transport": _final_transport(tmp_path),
                }
            )
        )
        return path

    with pytest.raises(admission.AdmissionRefused, match="source identity"):
        admission.priced_mac_bound(priced())
    training.write_text(
        json.dumps(
            {
                "status": "PASS",
                "counter_admission": "PASS",
                "identity": {"source_manifest": sources},
            }
        )
    )
    bound = admission.priced_mac_bound(priced())
    assert bound["per_arm_gpu_hours"]["n3"] > 30 / 60
    assert bound["per_arm_gpu_hours"]["n1"] > 68 / 60


def test_primary_forecast_fits_caps_without_pre_funding_optional_reruns() -> None:
    from research.s1_mature_20261018 import ledger, price

    root = admission.ROOT / "research/s1_mature_20261018"
    forecast = price.forecast(
        root / "EVAL_PREFLIGHT_FINAL.json", root / "TRAIN_PREFLIGHT_FINAL_ONLY.json"
    )
    hard = forecast["primary_hard_reservation"]
    assert hard["gpu_allocation_hours"] == pytest.approx(5 * (65 + 141) / 60 + 2)
    assert hard["units"] == pytest.approx(38.333333333333336)
    assert all(hard[key] <= cap for key, cap in ledger.USABLE_CAPS.items())
    assert forecast["conditional_reruns"]["pre_funded"] is False
    assert forecast["conditional_planning_forecast"]["known_base_plus_cleanup_gpu_hours"] == 12.8
    assert forecast["complete_upper_bound"] is None  # Missing live transfer qualification.
    assert forecast["status"] == "REFUSED"
    assert not any("exceeds usable" in reason for reason in forecast["refusal_reasons"])


def test_prepare_matches_pricing_and_uploads_bound_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from research.s1_mature_20261018 import ledger, manifests, price, review_jobs

    root = admission.ROOT / "research/s1_mature_20261018"
    transport = _final_transport(tmp_path)
    acceptance = Path(transport["cli_acceptance_receipt"]["path"])
    forecast = price.forecast(
        root / "EVAL_PREFLIGHT_FINAL.json",
        root / "TRAIN_PREFLIGHT_FINAL_ONLY.json",
        acceptance_path=acceptance,
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps(forecast))
    seed = tmp_path / "seed-bank.json"
    seed.write_text(json.dumps({"seeds": list(admission.bank.TRAINING_SEEDS)}))
    inputs = [manifests.ExternalInput("seed-bank", seed, "seed-bank.json")]
    for dest in (
        "B8_s3_final.pth",
        "demo.tar.gz",
        "league.tar.gz",
        "mac-parity-reference.json",
        "parity-bank.json",
        "q0b-l4.json",
    ):
        path = tmp_path / dest
        path.write_text("fixture")
        inputs.append(manifests.ExternalInput(dest, path, dest))
    monkeypatch.setattr(review_jobs, "verified_inputs", lambda: inputs)
    monkeypatch.setattr(admission, "published_commit", lambda repo, ref: "a" * 40)
    monkeypatch.setattr(admission, "source_closure", lambda repo, commit: {"fixture": "bound"})
    args = argparse.Namespace(
        repo=admission.ROOT,
        published_ref="origin/t31-colab-safety-20261018",
        out=tmp_path / "plan.json",
        preregistration=root / "PREREGISTRATION.md",
        seed_bank=seed,
        world_bank=root / "WORLD_BANK.json",
        seed_audit=root / "SEED_AUDIT.json",
        input=[],
        artifact_root=tmp_path / "artifacts",
        pricing=pricing,
        manifest_dir=tmp_path / "jobs",
    )
    plan = json.loads(admission.prepare(args).read_text())
    assert plan["status"] == "PREPARED_NOT_DISPATCHED"
    assert plan["published_ref"] == args.published_ref
    assert plan["complete_upper_bound"] == pytest.approx(forecast["complete_upper_bound"])
    assert all(plan["complete_upper_bound"][key] <= cap for key, cap in ledger.USABLE_CAPS.items())
    for item in plan["manifests"]:
        row = json.loads(Path(item["path"]).read_text())
        assert [item for item in row["inputs"] if item["dest"] == "cli-acceptance-result.json"] == [
            {"dest": "cli-acceptance-result.json", **final_artifacts.binding(acceptance)}
        ]
    with pytest.raises(admission.AdmissionRefused, match="already exists"):
        admission.prepare(args)


def test_published_source_requires_clean_exact_remote_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_git(repo: Path, *args: str) -> str:
        calls.append(args)
        return "" if args == ("status", "--porcelain") else "a" * 40

    monkeypatch.setattr(admission, "git", fake_git)
    ref = "origin/t31-colab-safety-20261018"
    assert admission.published_commit(tmp_path, ref) == "a" * 40
    assert ("show-ref", "--verify", f"refs/remotes/{ref}") in calls
    assert ("rev-parse", f"refs/remotes/{ref}") in calls
    with pytest.raises(admission.AdmissionRefused, match="remote-tracking"):
        admission.published_commit(tmp_path, "HEAD")
    monkeypatch.setattr(
        admission, "git", lambda repo, *args: " M dirty" if args[0] == "status" else "a" * 40
    )
    with pytest.raises(admission.AdmissionRefused, match="clean"):
        admission.published_commit(tmp_path, ref)
    monkeypatch.setattr(
        admission,
        "git",
        lambda repo, *args: "b" * 40 if args == ("rev-parse", f"refs/remotes/{ref}") else "a" * 40,
    )
    with pytest.raises(admission.AdmissionRefused, match="HEAD to equal"):
        admission.published_commit(tmp_path, ref)
