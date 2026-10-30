"""Pre-look reporting and conditional hardware admission regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.doom_auc_smoke.aggregate import analyze
from research.doom_auc_smoke.evidence import digest, publish
from research.doom_auc_smoke.prelook import attach_kinds, census, read_prelook
from research.doom_auc_smoke.prepare import admit, manifest, unavailable_h100
from research.doom_auc_smoke.test_smoke import make_prelook, receipt_for


def test_census_zeros_boundary_and_no_label_requirement() -> None:
    rows = [
        dict(
            root_index=i,
            root_uid=str(i),
            world_id=i // 2,
            partition="assessment",
            kind="ordinary" if i % 2 else "death_lookback",
            hero_length=599 + i,
        )
        for i in range(4)
    ]
    report = census(rows, ["assessment"] * 3)
    assert report["late_roots"] == 3 and report["eligible_parent_worlds"] == 2
    assert report["per_assessment_world"]["2"]["roots"] == 0
    assert report["per_assessment_world"]["0"]["ordinary"] == 1
    assert not report["class_support_verified"] and not report["necessary_world_count_met"]
    rows[-1]["root_uid"] = "0"
    with pytest.raises(ValueError, match="duplicate"):
        census(rows, ["assessment"] * 3)


def test_descriptive_kinds_do_not_change_primary_and_empty_is_defined() -> None:
    n = 24
    doom = (torch.arange(n) % 2).bool()[:, None, None].expand(n, 6, 4)
    held = dict(
        world=torch.arange(n),
        length=torch.full((n,), 600),
        doom=doom,
        legal=torch.ones((n, 6), dtype=torch.bool),
        action=torch.zeros(n).long(),
        kind=["ordinary" if i % 2 else "death_lookback" for i in range(n)],
    )
    scores = dict(spatial=doom[:, :, 0].float(), b8=doom[:, :, 0].float())
    report = analyze(held, scores, resamples=100)
    assert report["verdict"] == "GO"
    assert report["endpoints"]["late/ordinary/b8_action"]["arms"]["spatial"]["auc"] is None
    assert report["endpoints"]["late/death_lookback/per_action"]["support"]["events"] == 288
    held["kind"] = ["ordinary"] * n
    other = analyze(held, scores, resamples=100)
    assert other["endpoints"]["late/b8_action"] == report["endpoints"]["late/b8_action"]
    assert other["endpoints"]["all/death_lookback/b8_action"]["arms"]["b8"]["auc"] is None
    assert report["interpretation"]["sampled_training_set_equivalents"] == 46.875


def h100_failure(root: Path, item: dict) -> Path:
    root.mkdir()
    probe = copy.deepcopy(item)
    probe["canary"] = dict(mode="probe")
    (root / "manifest.json").write_text(json.dumps(probe))
    path = root / "receipt.json"
    path.write_text(
        json.dumps(
            dict(
                manifest_sha256=digest(root / "manifest.json"),
                commit=item["commit"],
                job_id=item["job_id"],
                status="INCOMPLETE",
                endpoint=None,
                terminal=None,
                exports=[],
                stop_verified=False,
                cleanup_error="ALLOCATION_UNRESOLVED",
                cli_calls=[dict(command="new", status="failed")],
                error="accelerator=H100: Service Unavailable (503)",
            )
        )
    )
    return path


def test_unavailable_receipt_binding_and_non503_refusal(tmp_path: Path) -> None:
    item = manifest(
        dict(commit="a" * 40, registration_sha256="b" * 64),
        [],
        [],
        tmp_path / "output",
        "test-doom",
        "fit",
    )
    path = h100_failure(tmp_path / "h100", item)
    unavailable_h100(path, item)
    row = json.loads(path.read_text())
    for change in (
        dict(error="timeout"),
        dict(endpoint="allocated"),
        dict(exports=[{}]),
        dict(manifest_sha256="0" * 64),
        dict(commit="c" * 40),
        dict(cli_calls=[]),
        dict(status="PASS"),
    ):
        path.write_text(json.dumps(dict(row, **change)))
        with pytest.raises(ValueError, match="503/unavailable"):
            unavailable_h100(path, item)


def test_a100_only_admission_validates_and_keeps_envelopes(tmp_path: Path, monkeypatch) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    config = dict(
        identity="id",
        commit="a" * 40,
        registration_sha256="b" * 64,
        assessment_sha256="c" * 64,
        mac_look_file=str(tmp_path / "look.json"),
    )
    (packet / "config.json").write_text(json.dumps(config))
    item = manifest(config, [], [], packet / "fit-output", "test-doom", "fit")
    (packet / "fit.pending.json").write_text(json.dumps(item))
    h100 = h100_failure(tmp_path / "h100", item)
    a100 = tmp_path / "a100"
    for arm in ("spatial", "b8"):
        publish(
            a100 / "exports/doom",
            arm + "-prediction",
            "canary-id",
            dict(steps=3, measured_step_seconds=[1.0, 0.1, 0.11]),
        )
    receipt_for(a100)
    prelook = make_prelook(tmp_path / "prelook.json", config, [str(i) for i in range(24)])
    called = []
    monkeypatch.setattr("research.colab_cli.supervisor.validate", lambda m, r: called.append(m))
    dest = tmp_path / "admission"
    admit(packet, h100, a100 / "receipt.json", a100_only=True, admission_dir=dest, prelook=prelook)
    fit = json.loads((dest / "fit.json").read_text())
    assert len(called) == 1
    assert fit["requirements"]["gpu"] == "A100"
    assert fit["requirements"]["allowed_gpus"] == ["A100"]
    assert "prefer" not in fit["requirements"] and "hardware_canaries" not in fit["requirements"]
    assert fit["commit"] == item["commit"] and fit["inputs"] == item["inputs"]
    assert fit["timeout_minutes"] == 120
    assert fit["technical_envelope"]["useful_worker_seconds"] == 2700
    assert fit["budget"]["unit_cap"] == 16 and fit["budget"]["reserve_units"] == 4
    assert not json.loads((dest / "ADMISSION.json").read_text())["h100_stop_verified"]

    def refuse(*_args):
        raise ValueError("genuine proof missing")

    monkeypatch.setattr("research.colab_cli.supervisor.validate", refuse)
    with pytest.raises(ValueError, match="genuine proof"):
        admit(
            packet,
            h100,
            a100 / "receipt.json",
            a100_only=True,
            admission_dir=tmp_path / "refused",
            prelook=prelook,
        )
    assert not (tmp_path / "refused").exists()


def test_forecast_cap_prevents_manifest_publication(tmp_path: Path, monkeypatch) -> None:
    # Exercise the real admission path above with a slow measured step.
    original = __import__("research.doom_auc_smoke.prepare", fromlist=["fit_forecast"])
    monkeypatch.setattr(original, "fit_forecast", lambda timings: 1400.0)
    with pytest.raises(ValueError, match="45-minute"):
        test_a100_only_admission_validates_and_keeps_envelopes(tmp_path, monkeypatch)
    assert not (tmp_path / "admission/fit.json").exists()


def test_prelook_foreign_amendment_and_metadata_join(tmp_path: Path) -> None:
    config = dict(identity="id", assessment_sha256="c" * 64)
    path = make_prelook(tmp_path / "prelook.json", config, ["0", "1"])
    report = read_prelook(path, config)
    held = dict(root_uids=["0", "1"], world=torch.arange(2), length=torch.full((2,), 600))
    assert attach_kinds(held, report)["kind"] == ["ordinary"] * 2
    held["world"][0] = 2
    with pytest.raises(ValueError, match="join"):
        attach_kinds(held, report)
    report["documents"] = {}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="binding"):
        read_prelook(path, config)
