"""FLOOD host package checks against the actual reviewed broker receipt layout."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.ops import flood_sparse_main as main

pytestmark = pytest.mark.skipif(
    not main.CANARY.is_dir() or not main.BROKER.is_dir(),
    reason="requires retained Mac canary artifacts and qualified frozen broker",
)


def test_real_canary_without_receipt_sidecars_builds_registered_main(tmp_path: Path) -> None:
    assert not Path(str(main.CANARY / "receipt.json") + ".complete").exists()
    job = main.main_manifest(main.CANARY, tmp_path, main.BROKER)
    probe = main.load(main.CANARY / "manifest.json")
    assert job["commit"] == main.PIN
    assert job["source_paths"] == probe["source_paths"]
    assert job["inputs"] == probe["inputs"]
    assert job["native_cuda_compile"] == probe["native_cuda_compile"]
    assert job["canary"]["receipt"]["sha256"] == main.sha(main.CANARY / "receipt.json")
    assert job["timeout_minutes"] == 160
    assert job["requirements"] == {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 80,
    }
    assert job["budget"] == {
        "unit_cap": 80 / 3,
        "reserve_units": 16 / 3,
        "rate_units_per_hour": 10.0,
        "other_assignments": {},
    }
    assert job["env"]["SPARSE_BENCH_CANARY"] == "0"
    assert job["env"]["SPARSE_BENCH_SECONDS"] == "4800"
    assert job["env"]["GPU_SIM_SPIKE_SECONDS"] == "4800"
    assert "SNAKE_COLAB_CANARY" not in job["env"]
    assert "SNAKE_COLAB_CANARY_PACKAGE" not in job["env"]
    assert job["resume"]["ledger_path"] == job["env"]["SPARSE_BENCH_LEDGER_PATH"]
    assert job["resume"]["skip_verified"] is True
    assert job["resume"]["max_outcome_blind_retries"] == 2


@pytest.mark.parametrize("mutation", ["review-scope", "review-route", "receipt", "pin", "export"])
def test_review_or_saved_evidence_drift_refuses(tmp_path: Path, monkeypatch, mutation: str) -> None:
    original = main.load

    def changed(path: Path) -> dict:
        value = copy.deepcopy(original(path))
        if path.name == "CODEX_V_MAC_REVIEW.json":
            if mutation == "review-scope":
                value["verdict_scope"] = "ROUTE_PASS"
            if mutation == "review-route":
                value["route_decision"] = "PASS"
        if path.name == "receipt.json":
            if mutation == "receipt":
                value["stop_verified"] = False
            if mutation == "pin":
                value["commit"] = "a" * 40
            if mutation == "export":
                value["exports"][0]["sha256"] = "a" * 64
        return value

    monkeypatch.setattr(main, "load", changed)
    with pytest.raises(ValueError):
        main.main_manifest(main.CANARY, tmp_path, main.BROKER)


def test_foreign_main_ledger_refuses(tmp_path: Path, monkeypatch) -> None:
    from research import metadata_archive
    from research.gpu_sim_spike import sparse_units

    unpack = metadata_archive.unpack

    def foreign(archive: Path, root: Path, expected: str) -> dict:
        result = unpack(archive, root, expected)
        ledger = root / "inputs/verified-ledger.json"
        value = main.load(ledger)
        value["namespace"] = "foreign"
        for suffix in ("", ".sha256", ".complete"):
            Path(str(ledger) + suffix).unlink()
        sparse_units.publish_json(ledger, value)
        return result

    monkeypatch.setattr(metadata_archive, "unpack", foreign)
    with pytest.raises(ValueError, match="foreign sparse verified ledger"):
        main.main_manifest(main.CANARY, tmp_path, main.BROKER)


def test_publication_is_create_only_and_completion_checked(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    main.publish(path, {"value": 1})
    assert main.verified(path) == {"value": 1}
    with pytest.raises(FileExistsError):
        main.publish(path, {"value": 2})
    path.write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="digest/name differs"):
        main.verified(path)


@pytest.mark.parametrize("changed_prior", [False, True])
def test_live_dispatch_requires_light_acceptance_before_ops(
    tmp_path: Path, monkeypatch, changed_prior: bool
) -> None:
    job = {"job_id": "unit-test"}
    for name, value in (
        ("sparse-bench-main.json", job),
        ("dispatch-main.sh", "shell\n"),
        ("DISPATCH.txt", "instructions\n"),
        ("UPLOAD_AUDIT.json", {}),
    ):
        main.publish(tmp_path / name, value)
    main.publish(
        tmp_path / "PREPARATION.json",
        {
            "implementation": main.implementation(),
            "manifest_sha256": main.sha(tmp_path / "sparse-bench-main.json"),
            "dispatch_sha256": main.sha(tmp_path / "dispatch-main.sh"),
            "dispatch_txt_sha256": main.sha(tmp_path / "DISPATCH.txt"),
            "upload_audit_sha256": main.sha(tmp_path / "UPLOAD_AUDIT.json"),
            "canary_review_sha256": main.sha(main.CANARY / "CODEX_V_MAC_REVIEW.json"),
            "prior_spend": {},
        },
    )
    monkeypatch.setattr(main, "main_manifest", lambda *args: job)
    monkeypatch.setattr(main, "prior_spend", lambda: {"new-receipt": 1} if changed_prior else {})
    monkeypatch.setattr(main.subprocess, "run", lambda *args, **kwargs: pytest.fail("ops reached"))
    with pytest.raises(ValueError if changed_prior else FileNotFoundError):
        main.dispatch(main.CANARY, tmp_path, main.BROKER, False)


def test_known_prior_spend_plus_contracted_main_fits_series(monkeypatch) -> None:
    # Freeze this test's eight reviewed receipts: subsequent real attempts must
    # invalidate a dispatch preparation, but must not break this historical test.
    exp = main.BASE / "snake-dqn-artifacts/big-lever/exp"
    relative = [
        "gpu-engine-sprint-20261023-02/sparse-cap160/canary-output/receipt.json",
        "gpu-engine-sprint-20261023-03/sparse-cap160-exactness/canary-output/receipt.json",
        "gpu-engine-sprint-20261023-03/sparse-cap160-exactness-fresh-02/canary-output/receipt.json",
        "gpu-engine-sprint-20261023-03/sparse-cap160-exactness-nvccfix-01/"
        "native-compile/output/receipt.json",
        "gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03/"
        "native-compile/output/receipt.json",
        "gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01/"
        "native-compile/output/receipt.json",
        "gpu-sparse-metadata-20261024-02/packet/canary-output/receipt.json",
        "gpu-sparse-metadata-20261024-02/packet/native-compile/output/receipt.json",
    ]
    monkeypatch.setattr(Path, "glob", lambda root, pattern: iter(exp / name for name in relative))
    accounting = main.prior_spend()
    assert len(accounting["receipts"]) == 8
    assert accounting["units"] == pytest.approx(5.461173721516194)
    assert accounting["elapsed_minutes"] == pytest.approx(52.37891746598373)
    assert accounting["committed_wall_minutes"] <= 216
    assert accounting["committed_units"] <= 36


def test_unknown_prior_accounting_refuses(monkeypatch) -> None:
    original = main.load

    def changed(path: Path) -> dict:
        value = copy.deepcopy(original(path))
        if path.name == "receipt.json":
            value["accounting"] = None
            value["elapsed_seconds"] = 1
        return value

    monkeypatch.setattr(main, "load", changed)
    with pytest.raises(ValueError, match="unknown spend"):
        main.prior_spend()
