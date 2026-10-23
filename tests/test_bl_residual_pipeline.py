"""F3 allocation, immutable-unit and actual-entrypoint plumbing contracts."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import digest, verify_export
from research.bl_residual import job
from research.bl_residual.data import allocate_lives, validate_rows, world_fold
from research.bl_residual.fit import FitRecipe
from research.bl_residual.make_pack import prepare_inputs
from research.bl_residual.protocol import (
    F4_REGISTRATION_SHA,
    ROUND_26_JUDGMENT_SHA,
    allocation_hash,
    build_ledger,
    canonical_sha,
    read_ledger,
    require_judgment,
)

B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


def catalog(n: int) -> list[dict]:
    return [
        {
            "life_id": f"life-{i}",
            "parent_world": f"parent-{i // 2}",
            "stream": "S-Watch6" if i % 3 == 0 else "S-gate",
            "rescue_bearing": True,
            "allocation_index": i,
        }
        for i in range(n)
    ]


def test_nested_doses_and_append_only_holdout() -> None:
    first = allocate_lives(catalog(1500))
    second = allocate_lives(catalog(6000), prior=first)
    for dose in (256, 1024):
        assert first["doses"][str(dose)] == second["doses"][str(dose)]
    d = second["doses"]
    assert len(d["256"]["heldout_lives"]) == 64
    assert d["1024"]["train_lives"] == d["4096"]["train_lives"][:960]
    train_parents = {second["parents"][life] for life in d["4096"]["train_lives"]}
    hold_parents = {second["parents"][life] for life in d["256"]["heldout_lives"]}
    assert not train_parents & hold_parents
    assert all(world_fold(p) == 0 for p in hold_parents)
    changed = catalog(6000)
    changed[0]["parent_world"] = "foreign"
    with pytest.raises(ValueError, match="changed"):
        allocate_lives(changed, prior=first)


def test_insufficient_holdout_never_shrinks_N() -> None:
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        allocate_lives(catalog(10))
    value = allocate_lives(catalog(500))
    assert value["doses"]["256"]["ready"]
    assert not value["doses"]["4096"]["ready"]
    assert value["doses"]["4096"]["train_lives"] == []


def test_only_exact_judged_registration_can_authorize_factory_fit(tmp_path: Path) -> None:
    registration = Path(__file__).resolve().parents[1] / "research/bl_residual/F4_REGISTRATION.md"
    assert digest(registration) == F4_REGISTRATION_SHA
    approval = tmp_path / "judgment.json"
    approval.write_text(
        json.dumps(
            {
                "schema": "bl-residual-f4-judgment/v1",
                "decision": "APPROVE",
                "judgment_sha256": ROUND_26_JUDGMENT_SHA,
                "registration_sha256": F4_REGISTRATION_SHA,
                "recipe_sha256": "f" * 64,
            }
        )
    )
    assert (
        require_judgment(approval, registration, "f" * 64)["registration_sha256"]
        == F4_REGISTRATION_SHA
    )
    copied = tmp_path / "F4_COPY.md"
    copied.write_bytes(registration.read_bytes())
    with pytest.raises(ValueError, match="UNRATIFIED"):
        require_judgment(approval, copied, "f" * 64)
    approval.write_text(json.dumps({"schema": "bl-residual-f4-judgment/v1"}))
    with pytest.raises(ValueError, match="UNRATIFIED"):
        require_judgment(approval, registration, "f" * 64)


@pytest.mark.skipif(not B8.exists(), reason="local pinned B8 fixture unavailable")
def test_real_synthetic_entrypoint_resume_skips_exact_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    control = prepare_inputs(inputs, B8, doses=(256,))
    first = tmp_path / "first"
    result = job.run(inputs, first, mode="smoke")
    assert result["done"] == result["expected_units"] == 1
    assert result["heldout_outcomes_opened"] is False
    ledger = tmp_path / "ledger.json"
    verified = build_ledger([first], ledger, control["namespace"], control["allocation"])
    assert len(verified["units"]) == 1
    # Resume only changes the separately bound ledger, never the fit allocation.
    (inputs / "verified-ledger.json").write_bytes(ledger.read_bytes())
    control["ledger_sha256"] = digest(ledger)
    assert allocation_hash(control) == control["allocation"]
    (inputs / "control.json").write_text(json.dumps(control))
    monkeypatch.setattr(job, "synthetic_table", lambda *_: pytest.fail("verified unit recomputed"))
    resumed = job.run(inputs, tmp_path / "second", mode="smoke")
    assert resumed["skipped"] == [0]
    assert resumed["completed"] == []
    with pytest.raises(ValueError, match="foreign"):
        read_ledger(ledger, "foreign", control["allocation"])
    with pytest.raises(ValueError, match="duplicate"):
        build_ledger(
            [first, first],
            tmp_path / "duplicates.json",
            control["namespace"],
            control["allocation"],
        )


@pytest.mark.skipif(not B8.exists(), reason="local pinned B8 fixture unavailable")
def test_fit_fault_seals_terminal_receipt_and_stops_later_doses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    prepare_inputs(inputs, B8, doses=(256, 1024))
    from research.bl_residual.fit import FitFault

    calls: list[int] = []

    def fault(*_args: object, **_kwargs: object) -> object:
        calls.append(1)
        raise FitFault("synthetic numerical tripwire")

    monkeypatch.setattr(job, "fit_residual", fault)
    output = tmp_path / "fault"
    with pytest.raises(FitFault, match="synthetic numerical tripwire"):
        job.run(inputs, output, mode="smoke")
    assert calls == [1]
    fault_receipt = output / "fault-0000.json"
    assert verify_export(fault_receipt) == digest(fault_receipt)
    assert json.loads(fault_receipt.read_text())["kind"] == "KILL_FIT"
    assert not (output / "fit-0000.pth").exists()


def test_all_scientific_controls_are_allocation_bound() -> None:
    control = {"namespace": "test", "doses": [256], "train_lives": {"256": ["a"]}}
    original = allocation_hash(control)
    control["train_lives"]["256"] = ["b"]
    assert original != allocation_hash(control)


def test_d4096_staging_refuses_without_prior_fits_and_receipts(tmp_path: Path) -> None:
    control = {
        "synthetic": False,
        "namespace": "f4-series",
        "stage_receipts": {},
        "input_hashes": {},
    }
    with pytest.raises(ValueError, match="D4096 requires passed D256 and D1024"):
        job._validate_staging(tmp_path, control, [4096], "a" * 64, set())


def test_pinned_entrypoint_device_resolves_without_shell_execution(tmp_path: Path) -> None:
    from research.colab_cli.canary import job_device

    script = Path(__file__).resolve().parents[1] / "research/bl_residual/colab.sh"
    target = tmp_path / "research/bl_residual/colab.sh"
    target.parent.mkdir(parents=True)
    target.write_bytes(script.read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Device fixture",
        ],
        check=True,
    )
    manifest = {
        "commit": "HEAD",
        "script": "research/bl_residual/colab.sh",
        "device": "cuda",
        "env": {"F3_DEVICE": "cuda", "F3_MODE": "canary"},
    }
    assert job_device(manifest, tmp_path) == "cuda"


def test_heldout_refuses_synthetic_foreign_and_changed_reserve(tmp_path: Path) -> None:
    from research.bl_residual.heldout import validate_candidate

    allocation = tmp_path / "allocation.json"
    allocation.write_text("frozen allocator fixture")
    control = {
        "synthetic": False,
        "doses": [256],
        "namespace": "series",
        "recipe_sha256": canonical_sha(dataclasses.asdict(FitRecipe())),
        "input_hashes": {"life-allocation.json": digest(allocation)},
    }
    control["allocation"] = allocation_hash(control)
    meta = {
        "schema": "bl-residual-fit/v1",
        "synthetic": False,
        "dose": 256,
        "index": 0,
        "updates": 2000,
        "namespace": "series",
        "recipe_sha256": control["recipe_sha256"],
        "allocation": control["allocation"],
    }
    validate_candidate(meta, control, allocation)
    with pytest.raises(ValueError, match="foreign/synthetic"):
        validate_candidate({**meta, "synthetic": True}, control, allocation)
    with pytest.raises(ValueError, match="foreign/synthetic"):
        validate_candidate({**meta, "updates": 3}, control, allocation)
    allocation.write_text("other reserve")
    with pytest.raises(ValueError, match="held-out allocation"):
        validate_candidate(meta, control, allocation)


@pytest.mark.skipif(not B8.exists(), reason="local pinned B8 fixture unavailable")
def test_raw_adapter_refuses_unknown_status_and_cross_parent_life() -> None:
    from src.model.ego2s_network import load_ego2s_checkpoint

    table = job.synthetic_table(load_ego2s_checkpoint(str(B8)))
    validate_rows(table)
    fp64 = {**table, "q": table["q"].astype(np.float64)}
    with pytest.raises(ValueError, match="fp32"):
        validate_rows(fp64)
    table["status"][0, 0] = "BAD"
    with pytest.raises(ValueError, match="status"):
        validate_rows(table)
    table["status"][0, 0] = "EXACT"
    table["parent_world"][0] = "foreign"
    with pytest.raises(ValueError, match="ancestry"):
        validate_rows(table)
    assert table["local"].dtype == np.uint8
