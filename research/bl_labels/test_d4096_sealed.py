"""Metadata-only adversarial tests for the D4096 early-collection seals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels import d4096_sealed as sealed
from research.bl_labels import w12_runtime
from research.bl_labels.exports import _verify, publish_json

PIN = "1" * 40


def _terminal(path: Path, elapsed: float, *, status: str = "COMPLETE", stage="d1024") -> Path:
    from research.bl_labels.frozen_factory import OLD_PIN

    publish_json(
        path,
        {
            "schema": "bl-labels-f0-mac-driver/v1",
            "status": status,
            "cumulative_elapsed_seconds": elapsed,
            "workers": 6,
            "collector_source_commit": OLD_PIN,
            "namespace": f"fixture-mac-{stage}",
        },
    )
    return path


def _fit_receipt(root: Path, job_id: str) -> Path:
    directory = root / job_id
    directory.mkdir()
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "commit": PIN,
                "script": "research/bl_residual/f4_d256_colab.sh",
                "env": {"F4_DOSE": "1024", "F4_MODE": "canary" if "canary" in job_id else "main"},
            },
            sort_keys=True,
        )
    )
    receipt = directory / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "commit": PIN,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "status": "PASS",
                "stop_verified": True,
            },
            sort_keys=True,
        )
    )
    return receipt


def _ready(tmp_path: Path) -> tuple[Path, Path, Path]:
    from research.bl_labels.frozen_factory import OLD_PIN

    control = tmp_path / "d1024-control.json"
    allocation = [{"allocation_id": "p0", "stream": "watch6"}]
    publish_json(
        control,
        {
            "namespace": "fixture-mac-d1024",
            "source_commit": PIN,
            "allocation": allocation,
            "mac_gate": {"stage": "d1024"},
        },
    )
    start = tmp_path / "attempt-01.start.json"
    publish_json(start, {"namespace": "fixture-mac-d1024", "source_commit": PIN, "workers": 6})
    d1024 = tmp_path / "attempt-01.terminal.json"
    publish_json(
        d1024,
        {
            "schema": "bl-labels-f0-mac-driver/v1",
            "status": "COMPLETE",
            "workers": 6,
            "namespace": "fixture-mac-d1024",
            "collector_source_commit": OLD_PIN,
            "w12_calibration_source_commit": OLD_PIN,
            "cumulative_elapsed_seconds": 3600,
            "start_receipt": start.name,
            "start_sha256": _verify(start),
            "attempts_ledger": [d1024.name],
            "completed_parents": ["p0"],
            "slowdown_state_by_stream": {"watch6": {"width_wall_seconds": {"6": 3600}}},
        },
    )
    throughput = tmp_path / "d1024-w6.json"
    w12_runtime.seal_d1024_old_w6_throughput(control, [d1024], throughput)
    canary = _fit_receipt(tmp_path, "f4-d1024-canary-01")
    main = _fit_receipt(tmp_path, "f4-d1024-main-01")
    readiness = tmp_path / "readiness.json"
    sealed.seal_readiness(
        readiness,
        d1024,
        {"f4-d1024-canary-01": canary, "f4-d1024-main-01": main},
        throughput,
        factory_exited=True,
        mac_lease_released=True,
        all_fit_jobs_listed=True,
    )
    return readiness, d1024, throughput


def test_metadata_only_readiness_and_fixed1024_projection(tmp_path) -> None:
    readiness, d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256-terminal.json", 7200, stage="d256")
    projection = tmp_path / "projection.json"
    value = sealed.seal_mac_h_projection(projection, readiness, d256, d1024)
    assert value["projected_mac_hours"] == pytest.approx(10.8)
    assert sealed.validate_projection(projection, readiness)["verdict"] == "PASS"


def test_readiness_refuses_implicit_operator_attestation(tmp_path) -> None:
    d1024 = _terminal(tmp_path / "d1024.json", 1)
    throughput = tmp_path / "throughput.json"
    publish_json(
        throughput,
        {
            "schema": "bl-labels-f0-d1024-old-w6-throughput/v1",
            "stage": "d1024",
            "workers": 6,
            "source_commit": PIN,
            "d1024_terminal": {"path": str(d1024.resolve()), "sha256": _verify(d1024)},
            "streams": {"watch6": {"completed_parents": 1, "elapsed_seconds": 1}},
        },
    )
    fit = tmp_path / "fit.json"
    publish_json(fit, {"status": "HOLD"})
    with pytest.raises(ValueError, match="explicit"):
        sealed.seal_readiness(
            tmp_path / "readiness.json",
            d1024,
            {"main": fit},
            throughput,
            factory_exited=False,
            mac_lease_released=True,
            all_fit_jobs_listed=True,
        )


def test_projection_refuses_a_d1024_terminal_not_bound_by_readiness(tmp_path) -> None:
    readiness, _d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256.json", 1, stage="d256")
    other = _terminal(tmp_path / "other-d1024.json", 1)
    with pytest.raises(ValueError, match="differs"):
        sealed.seal_mac_h_projection(tmp_path / "projection.json", readiness, d256, other)


def test_discard_seal_blocks_funding_and_malformed_discard_holds(tmp_path, monkeypatch) -> None:
    readiness, _d1024, _ = _ready(tmp_path)
    stop = tmp_path / "ra-stop.json"
    publish_json(stop, {"schema": "bl-residual-funding-look/v1", "verdict": "STOP_FUNDING"})
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    discard = tmp_path / "d4096-not-run-by-rule.json"
    sealed.seal_not_run_by_rule(discard, readiness, stop)
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        sealed.require_not_discarded(discard)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path / "other")
    bad = tmp_path / "bad.json"
    publish_json(bad, {"schema": sealed.DISCARD_SCHEMA})
    with pytest.raises(ValueError, match="malformed"):
        sealed.require_not_discarded(bad)


def test_funding_requires_a_pass_receipt(tmp_path) -> None:
    no = tmp_path / "no.json"
    publish_json(no, {"schema": "bl-residual-funding-look/v1", "verdict": "STOP_FUNDING"})
    with pytest.raises(ValueError, match="STOP_FUNDING"):
        sealed.require_funding_receipt(None)
    with pytest.raises(ValueError, match="STOP_FUNDING"):
        sealed.require_funding_receipt(no)
    from dataclasses import asdict

    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import F4_REGISTRATION_SHA, NAMESPACE, canonical_sha

    digest = "a" * 64
    yes = tmp_path / "yes.json"
    publish_json(
        yes,
        {
            "schema": "bl-residual-funding-look/v1",
            "verdict": "PASS",
            "series_id": NAMESPACE,
            "registration_sha256": F4_REGISTRATION_SHA,
            "recipe_sha256": canonical_sha(asdict(FitRecipe())),
            "doses": [256, 1024],
            "reserve_allocation_sha256": digest,
            "d256_fit_sha256": digest,
            "d1024_fit_sha256": digest,
            "inputs": {"d256_checkpoint_sha256": digest, "d1024_checkpoint_sha256": digest},
        },
    )
    assert sealed.require_funding_receipt(yes)["verdict"] == "PASS"


def test_projection_above_130_hours_never_publishes(tmp_path):
    readiness, d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256.json", 130 * 3600, stage="d256")
    out = tmp_path / "too-expensive.json"
    with pytest.raises(ValueError, match="exceeds 130"):
        sealed.seal_mac_h_projection(out, readiness, d256, d1024)
    assert not out.exists()


@pytest.mark.parametrize("label,stage", [("INSUFFICIENT", "d1024-fit"), ("INVALID", "r-a")])
def test_stop_labels_without_existing_receipt_are_bound_create_only(
    tmp_path, monkeypatch, label, stage
):
    readiness, _, _ = _ready(tmp_path)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    out = tmp_path / "d4096-not-run-by-rule.json"
    sealed.seal_not_run_by_rule(out, readiness, terminal_label=label, terminal_stage=stage)
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        sealed.require_not_discarded(out)
    with pytest.raises(FileExistsError):
        sealed.seal_not_run_by_rule(out, readiness, terminal_label=label, terminal_stage=stage)


def test_recoverable_hold_and_partial_seal_refuse(tmp_path, monkeypatch):
    readiness, _, _ = _ready(tmp_path)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    out = tmp_path / "d4096-not-run-by-rule.json"
    with pytest.raises(ValueError, match="explicitly unrecoverable"):
        sealed.seal_not_run_by_rule(out, readiness, terminal_label="HOLD", terminal_stage="r-a")
    Path(str(out) + ".sha256").write_text("partial")
    with pytest.raises(ValueError, match="partial"):
        sealed.require_not_discarded(out)


def test_fit_and_public_union_refuse_discard_before_payloads(tmp_path, monkeypatch):
    from research.bl_labels import d4096_final, d4096_resume, mac_gates
    from research.bl_residual import job

    readiness, d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256.json", 7200, stage="d256")
    projection = tmp_path / "projection.json"
    sealed.seal_mac_h_projection(projection, readiness, d256, d1024)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    out = tmp_path / "d4096-not-run-by-rule.json"
    sealed.seal_not_run_by_rule(
        out, readiness, terminal_label="INSUFFICIENT", terminal_stage="d1024-fit"
    )
    monkeypatch.setattr(job, "D4096_DISCARD_SEAL", out)
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        job._validate_staging(
            tmp_path, {"synthetic": False, "stage_receipts": {}}, [4096], "a" * 64, set()
        )
    control = {
        "namespace": "fixture-mac-d4096",
        "allocation": [],
        "mac_gate": {
            "stage": "d4096",
            "bindings": {
                "sealed_collection": {
                    "readiness": {"path": str(readiness)},
                    "projection": {"path": str(projection)},
                }
            },
        },
    }
    monkeypatch.setattr(d4096_final, "require_pinned_source", lambda _: "a" * 40)
    monkeypatch.setattr(d4096_resume, "original_control", lambda _: control)
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        d4096_final.publish_final(
            tmp_path, tmp_path / "final/ledger.json", "a" * 40, tmp_path / "funding"
        )
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        mac_gates.publish_ledger(
            tmp_path / "foreign-pack", [], tmp_path / "public/ledger.json", control=control
        )
    assert not (tmp_path / "final").exists()
    assert not (tmp_path / "public").exists()


def test_plain_supervisor_receipt_is_bound_and_active_job_refuses(tmp_path):
    readiness, _, _ = _ready(tmp_path)
    value = sealed.validate_readiness(readiness)
    receipt = Path(value["d1024_fit_jobs"][0]["receipt"]["path"])
    changed = json.loads(receipt.read_text())
    changed["stop_verified"] = False
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="terminal/HOLD"):
        sealed.validate_readiness(readiness)


def test_incomplete_final_union_never_publishes_final_marker(tmp_path, monkeypatch):
    from research.bl_labels import mac_gates

    monkeypatch.setattr(sealed, "require_funding_receipt", lambda *_: {})
    control = {
        "namespace": "fixture-mac-d4096",
        "allocation": [{"allocation_id": "p0", "seed": 1}],
        "mac_gate": {"stage": "d4096"},
    }
    out = tmp_path / "final/verified-ledger.json"
    with pytest.raises(ValueError, match="parents pending"):
        mac_gates.publish_ledger(
            tmp_path,
            [],
            out,
            control=control,
            funding_receipt=tmp_path / "funding",
            require_complete=True,
        )
    assert not out.exists()
    assert not Path(str(out) + ".complete").exists()


def test_alternate_l4_union_entrypoint_cannot_bypass_pending_funding(tmp_path):
    from research.bl_labels import l4_import

    with pytest.raises(ValueError, match="PASS funding"):
        l4_import.union_close(
            {"namespace": "fixture-mac-d4096", "allocation": [], "mac_gate": {"stage": "d4096"}},
            [],
            tmp_path / "delegation",
            tmp_path / "union",
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("unrecoverable", [False, True])
def test_receipt_hold_requires_explicit_unrecoverable(tmp_path, monkeypatch, unrecoverable):
    readiness, _, _ = _ready(tmp_path)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    terminal = tmp_path / "hold.json"
    publish_json(terminal, {"schema": "bl-residual-funding-look/v1", "verdict": "HOLD"})
    out = tmp_path / "d4096-not-run-by-rule.json"
    if not unrecoverable:
        with pytest.raises(ValueError, match="explicitly unrecoverable"):
            sealed.seal_not_run_by_rule(out, readiness, terminal)
        assert not out.exists()
    else:
        value = sealed.seal_not_run_by_rule(out, readiness, terminal, unrecoverable=True)
        assert value["terminal_stage"] == "r-a"
        with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
            sealed.require_not_discarded(out)


def test_reader_refuses_legacy_recoverable_receipt_hold(tmp_path, monkeypatch):
    readiness, _, _ = _ready(tmp_path)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    terminal = tmp_path / "hold.json"
    publish_json(terminal, {"schema": "bl-residual-funding-look/v1", "verdict": "HOLD"})
    out = tmp_path / "d4096-not-run-by-rule.json"
    publish_json(
        out,
        {
            "schema": sealed.DISCARD_SCHEMA,
            "judgment_sha256": sealed.JUDGMENT_SHA256,
            "readiness": {"path": str(readiness), "sha256": _verify(readiness)},
            "terminal": {"path": str(terminal), "sha256": _verify(terminal)},
            "terminal_label": "HOLD",
            "terminal_stage": "r-a",
            "unrecoverable": False,
        },
    )
    with pytest.raises(ValueError, match="explicitly unrecoverable"):
        sealed.require_not_discarded(out)


def test_receipt_stage_must_match_schema_on_write_and_read(tmp_path, monkeypatch):
    readiness, _, _ = _ready(tmp_path)
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    terminal = tmp_path / "stop.json"
    publish_json(terminal, {"schema": "bl-residual-funding-look/v1", "verdict": "STOP_FUNDING"})
    out = tmp_path / "d4096-not-run-by-rule.json"
    with pytest.raises(ValueError, match="stage differs"):
        sealed.seal_not_run_by_rule(out, readiness, terminal, terminal_stage="d1024-fit")
    publish_json(
        out,
        {
            "schema": sealed.DISCARD_SCHEMA,
            "judgment_sha256": sealed.JUDGMENT_SHA256,
            "readiness": {"path": str(readiness), "sha256": _verify(readiness)},
            "terminal": {"path": str(terminal), "sha256": _verify(terminal)},
            "terminal_label": "STOP_FUNDING",
            "terminal_stage": "d1024-fit",
            "unrecoverable": False,
        },
    )
    with pytest.raises(ValueError, match="malformed"):
        sealed.require_not_discarded(out)


@pytest.mark.parametrize("suffix", ["", ".sha256", ".complete"])
def test_dangling_discard_symlink_is_present_for_canonical_explicit_and_create_only(
    tmp_path, monkeypatch, suffix
):
    monkeypatch.setattr(sealed, "CANONICAL_PACK", tmp_path)
    out = tmp_path / "d4096-not-run-by-rule.json"
    dangling = Path(str(out) + suffix)
    dangling.symlink_to(tmp_path / "missing-target")
    assert not dangling.exists() and dangling.is_symlink()
    for candidate in (None, out, tmp_path / "alternate-discard.json", tmp_path / "missing-target"):
        with pytest.raises(ValueError, match="HOLD"):
            sealed.require_not_discarded(candidate)
    with pytest.raises(FileExistsError, match="create-only"):
        sealed._fresh_receipt(out)
    assert dangling.is_symlink() and not (tmp_path / "missing-target").exists()


def test_prospective_disclosure_keeps_actual_placement_separate(tmp_path):
    from research.bl_labels.d4096_design import COLLECTION_DESIGN, OWNER_DECISION_SHA256

    readiness, d1024, _ = _ready(tmp_path)
    d256 = _terminal(tmp_path / "d256.json", 7200, stage="d256")
    b2 = sealed.seal_mac_h_projection(tmp_path / "b2.json", readiness, d256, d1024)
    for receipt in (sealed.validate_readiness(readiness), b2):
        assert receipt["collection_design"] == COLLECTION_DESIGN
        assert receipt["owner_decision"]["sha256"] == OWNER_DECISION_SHA256
        assert receipt["actual_execution_placement"]["d4096"] == "NOT_STARTED"


def test_release_l4_inventory_refuses_omitted_failed_setup_and_identity_attempts(tmp_path):
    from research.bl_residual.protocol import F4_REGISTRATION_SHA, NAMESPACE

    receipts = []
    for name in ("completed", "failed", "setup", "identity"):
        path = tmp_path / f"{name}-receipt.json"
        path.write_text(json.dumps({"job_id": name, "status": name}))
        receipts.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    identity = {"series_id": NAMESPACE, "registration_sha256": F4_REGISTRATION_SHA, "dose": 4096}
    inventory = tmp_path / "inventory.json"
    publish_json(
        inventory,
        {
            **identity,
            "schema": "bl-labels-d4096-l4-attempt-inventory/v1",
            "all_attempts_listed": True,
            "attempts": [{"receipt": row} for row in receipts],
        },
    )
    for omitted in (1, 2, 3):
        charges = tmp_path / f"charges-missing-{omitted}.json"
        publish_json(
            charges,
            {
                **identity,
                "schema": "bl-residual-d4096-l4-accounting/v1",
                "attempts": [
                    {"receipt_sha256": row["sha256"], "charged_units": 1}
                    for i, row in enumerate(receipts)
                    if i != omitted
                ],
            },
        )
        with pytest.raises(ValueError, match="omits"):
            sealed._release_l4_accounting(
                charges, inventory, series_id=NAMESPACE, no_l4_attempts=False
            )
    charges = tmp_path / "charges-complete.json"
    publish_json(
        charges,
        {
            **identity,
            "schema": "bl-residual-d4096-l4-accounting/v1",
            "attempts": [{"receipt_sha256": row["sha256"], "charged_units": 1} for row in receipts],
        },
    )
    result = sealed._release_l4_accounting(
        charges, inventory, series_id=NAMESPACE, no_l4_attempts=False
    )
    assert result["attempt_count"] == 4 and result["charged_units"] == 4
    assert result["inventory"]["sha256"] == _verify(inventory)
    Path(receipts[1]["path"]).write_text("changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        sealed._release_l4_accounting(charges, inventory, series_id=NAMESPACE, no_l4_attempts=False)


def test_absent_l4_ledger_requires_explicit_no_attempts_attestation():
    with pytest.raises(ValueError, match="no-L4 attestation"):
        sealed._release_l4_accounting(None, None, series_id="fixture", no_l4_attempts=False)
    assert sealed._release_l4_accounting(None, None, series_id="fixture", no_l4_attempts=True) == {
        "status": "ABSENT_NO_L4_ATTEMPTS_ATTESTED",
        "ledger": None,
        "inventory": None,
    }
