"""D4096-only width admission and exact immutable rebinding refusals."""

from __future__ import annotations

import pytest

from research.bl_labels import d1024_w12, d4096_pack, mac_gates, w12_admission
from research.bl_labels.exports import publish_json


@pytest.mark.parametrize(
    "stage,owner,expected", [("d1024", True, 6), ("d4096", False, 6), ("d4096", True, 12)]
)
def test_old_width_is_stage_specific(tmp_path, monkeypatch, stage, owner, expected):
    control = {
        "source_commit": d4096_pack.SCIENCE,
        "mac_gate": {"stage": stage, "status": "HOLD", "w12_calibration_source_commit": "old"},
    }
    path = tmp_path / "control"
    publish_json(path, control)
    speed = tmp_path / "speed"
    publish_json(
        speed,
        {
            "schema": "bl-labels-f0-speed-equiv/v1",
            "status": "FAIL",
            "old": {"commit": "old"},
            "new": {"commit": d4096_pack.SCIENCE},
            "f1_bank_sha256": mac_gates.F1_BANK_SHA256,
        },
    )
    cal = tmp_path / "cal"
    publish_json(cal, {"schema": w12_admission.SCHEMA})
    measured = tmp_path / "measured"
    publish_json(measured, {"b8_sha256": "digest", "frp3_sha256": "digest"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(d1024_w12.w12_calibration, "sha", lambda _: "digest")
    monkeypatch.setattr(w12_admission, "CALIBRATION_RECEIPT", measured)
    monkeypatch.setattr(mac_gates, "_future_source_pin", lambda _: d4096_pack.SCIENCE)
    monkeypatch.setattr(mac_gates, "_future_readiness", lambda *_: {})
    monkeypatch.setattr(mac_gates, "_d4096_sealed_bindings", lambda *_: {})
    monkeypatch.setattr(d1024_w12, "_calibration_workers", lambda *_: (12, ["--owner-w12-ok"]))
    monkeypatch.delenv("F0_COMMIT", raising=False)
    monkeypatch.setenv("F1_COMMIT", "")
    monkeypatch.setenv("F0_COMMIT", d4096_pack.SCIENCE)
    _, workers, flags, fallback = d1024_w12._validate_inputs(
        stage,
        path,
        speed,
        cal,
        checkpoint,
        checkpoint,
        tmp_path / "old",
        tmp_path / "f4",
        tmp_path,
        owner_w12_ok=owner,
    )
    assert workers == expected and fallback
    assert bool(flags) == (expected == 12)


def test_unsealed_d4096_still_refuses_before_science_or_admission(tmp_path):
    path = tmp_path / "control"
    control = {"mac_gate": {"stage": "d4096", "status": "HOLD"}}
    publish_json(path, control)
    with pytest.raises(ValueError, match="HOLD"):
        d1024_w12._prepare_science_root(path, "d4096")
    with pytest.raises(ValueError, match="sealed readiness/projection"):
        mac_gates.admit_future(
            path,
            speed_equivalence=path,
            calibration=path,
            out=tmp_path / "out",
            f4_control=path,
            f4_inputs=tmp_path,
        )
    assert not (tmp_path / "out").exists()


def test_sealed_path_rejects_unruled_control_before_receipts(tmp_path):
    value = {
        "source_commit": d4096_pack.OLD_SCIENCE,
        "allocation": [],
        "mac_gate": {"stage": "d4096", "deadline_seconds": 468000},
    }
    with pytest.raises(ValueError, match="exact ruled OLD rebinding"):
        mac_gates._d4096_sealed_bindings(value, tmp_path / "r", tmp_path / "p")


def test_real_rebinding_has_only_two_control_byte_changes():
    """Read only the frozen control, allocation and empty ledger metadata."""
    source = d4096_pack.ORIGINAL
    pack = d4096_pack.PACK
    if not pack.is_dir():
        pytest.skip("operator artifact pack is not installed")
    old = (source / "d4096-control.json").read_bytes()
    new = (pack / "d4096-control.json").read_bytes()
    needle = f'"source_commit":"{d4096_pack.OLD_SCIENCE}"'.encode()
    assert old.count(needle) == 2
    assert new == old.replace(needle, f'"source_commit":"{d4096_pack.SCIENCE}"'.encode())
    for name in ("d4096-seed-audit.json", "d4096-verified-ledger.json", "d1024-control.json"):
        assert (pack / name).read_bytes() == (source / name).read_bytes()
    assert mac_gates._read(pack / "d4096-verified-ledger.json")["units"] == []


def test_pack_create_only_and_judgment_identity(tmp_path):
    judgment = tmp_path / "judgment"
    judgment.write_text("foreign")
    with pytest.raises(ValueError, match="judgment sha"):
        d4096_pack.create(tmp_path, tmp_path / "new", judgment, tmp_path / "binding")
    assert not (tmp_path / "new").exists()
