"""Independent offline admission, immutable export and exact-resume contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research.gpu_sim_spike import move_bench, move_pack, move_units
from research.gpu_sim_spike.make_manifests import ManifestRefused

PIN = "313758255ee5b9f1422a0202c14a80cc32554857"


def test_move_allocation_covers_three_rosters_and_scales() -> None:
    assert move_units.allocation() == tuple(
        f"cell-{mix}-e{e}" for e in (64, 256, 1024) for mix in ("scripted", "frozen", "mixed")
    )


def test_move_namespace_rejects_non_sha() -> None:
    with pytest.raises(ValueError):
        move_units.namespace("bad")


def test_immutable_unit_and_verified_resume(tmp_path: Path) -> None:
    cells, ns = move_units.allocation(smoke=True), move_units.namespace(PIN, smoke=True)
    move_units.publish_unit(tmp_path, ns, cells, cells[0], {"sentinel": 13})
    with pytest.raises(FileExistsError):
        move_units.publish_unit(tmp_path, ns, cells, cells[0], {"sentinel": 14})
    ledger = tmp_path / "verified-ledger.json"
    move_units.build_ledger(tmp_path, ledger, commit=PIN, smoke=True)
    assert move_units.load_ledger(ledger, ns, cells) == {cells[0]: {"sentinel": 13}}
    with pytest.raises(ValueError, match="foreign"):
        move_units.load_ledger(ledger, move_units.namespace("a" * 40, smoke=True), cells)


def test_corruption_and_foreign_allocation_refused(tmp_path: Path) -> None:
    cells, ns = move_units.allocation(), move_units.namespace(PIN)
    payload = move_units.publish_unit(tmp_path, ns, cells, cells[0], {"sentinel": 1})
    ledger = tmp_path / "verified-ledger.json"
    move_units.build_ledger(tmp_path, ledger, commit=PIN)
    payload.write_text("corrupt")
    with pytest.raises(ValueError):
        move_units.load_ledger(ledger, ns, cells)
    with pytest.raises(ValueError):
        move_units.publish_unit(tmp_path, ns, cells, "cell-foreign-e1024", {})


def test_incomplete_unit_never_enters_ledger(tmp_path: Path) -> None:
    cells, ns = move_units.allocation(), move_units.namespace(PIN)
    payload = move_units.publish_unit(tmp_path, ns, cells, cells[0], {})
    Path(str(payload) + ".complete").unlink()
    result = move_units.build_ledger(tmp_path, tmp_path / "verified-ledger.json", commit=PIN)
    assert result["units"] == []


def test_invalid_cpu_preflight_refuses_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "mac-preflight.json"
    move_units.publish_json(path, {"status": "INVALID", "commit": PIN, "device": "cpu"})
    with pytest.raises(ManifestRefused, match="CPU_EXACTNESS_REQUIRED"):
        move_pack.require_preflight(path, PIN)


def test_fixture_recovery_and_checksum(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "recovered"
    cells, ns = move_units.allocation(), move_units.namespace(PIN)
    move_units.publish_unit(source, ns, cells, cells[0], {"fixture": "test"})
    fixture = source / "recorded-states/cell-scripted-e64.npz"
    move_units.publish_bytes(fixture, b"fixture-bytes")
    ledger = destination / "verified-ledger.json"
    result = move_units.build_ledger(source, ledger, commit=PIN)
    assert len(result["fixtures"]) == 1
    assert move_units.load_ledger(ledger, ns, cells)[cells[0]] == {"fixture": "test"}
    (destination / "recorded-states/cell-scripted-e64.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        move_units.load_ledger(ledger, ns, cells)


def test_diagnostic_restores_methods_after_exception(monkeypatch: Any) -> None:
    class Broken:
        move_backend = "reference"

        def _respawn(self) -> None:
            pass

        def _collision_flags_reference(self) -> None:
            pass

        def _resolve_collisions_reference(self) -> None:
            pass

        def step(self, actions: Any) -> None:
            raise RuntimeError("deliberate")

    sim = Broken()
    original = sim._respawn
    with pytest.raises(RuntimeError, match="deliberate"):
        move_bench._stage_once(sim, None, "cpu")
    assert sim._respawn == original


def test_manifest_envelopes_and_family(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(move_pack, "checkpoint_inputs", lambda: [])
    for canary, wall, units in ((True, 24, 4), (False, 192, 32)):
        ledger = tmp_path / f"{canary}/verified-ledger.json"
        move_units.publish_json(
            ledger,
            move_units.empty_ledger(
                move_units.namespace(PIN, canary=canary), move_units.allocation(canary=canary)
            ),
        )
        job = move_pack.manifest(
            commit=PIN, output_dir=tmp_path / f"out-{canary}", ledger=ledger, canary=canary, rate=10
        )
        assert job["timeout_minutes"] == wall
        assert job["budget"]["unit_cap"] == units
        assert job["source_paths"] and "src" in job["source_paths"]
        assert job["resume"]["skip_verified"] is True
        if canary:
            assert job["canary"]["family"] == move_pack.SCRIPT
        else:
            assert "receipt" not in job["canary"]
        with pytest.raises(ManifestRefused):
            move_pack.manifest(
                commit=PIN, output_dir=tmp_path / "bad", ledger=ledger, canary=canary, rate=11
            )


def test_passing_header_cannot_hide_failed_cpu_proof(tmp_path: Path) -> None:
    receipt = tmp_path / "move-bench.json"
    from research.gpu_sim_spike.benchmark import source_hashes

    move_units.publish_json(
        receipt, {"status": "COMPLETE", "commit": PIN, "source_sha256": source_hashes()}
    )
    move_units.publish_json(tmp_path / "cpu-parity.json", {"status": "INVALID", "commit": PIN})
    pre = tmp_path / "mac-preflight.json"
    move_units.publish_json(
        pre,
        {
            "status": "PREFLIGHT",
            "commit": PIN,
            "device": "cpu",
            "receipt_path": str(receipt),
            "receipt_sha256": move_units._verify(receipt),
        },
    )
    with pytest.raises(ManifestRefused, match="CPU byte proof failed"):
        move_pack.require_preflight(pre, PIN)


def test_templates_are_explicitly_non_admitted(tmp_path: Path, monkeypatch: Any) -> None:
    import json

    monkeypatch.setattr(move_pack.a1_prof_pack, "require_source_pin", lambda commit: None)
    monkeypatch.setattr(move_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(move_pack, "convert_public_checkpoints", lambda value, **kwargs: value)
    monkeypatch.setattr(move_pack, "assert_upload_budget", lambda *args, **kwargs: None)
    move_pack.prepare(tmp_path, PIN, None, 10, templates_only=True)
    for path in tmp_path.glob("*-template.json"):
        assert json.loads(path.read_text())["scientific_admission"] is False
    assert len(list(tmp_path.glob("*-template.json"))) == 2


def test_same_commit_different_source_proof_is_refused(tmp_path: Path) -> None:
    receipt = tmp_path / "move-bench.json"
    move_units.publish_json(receipt, {"status": "COMPLETE", "commit": PIN, "source_sha256": {}})
    pre = tmp_path / "mac-preflight.json"
    move_units.publish_json(
        pre,
        {
            "status": "PREFLIGHT",
            "commit": PIN,
            "device": "cpu",
            "receipt_path": str(receipt),
            "receipt_sha256": move_units._verify(receipt),
        },
    )
    with pytest.raises(ManifestRefused, match="source hashes differ"):
        move_pack.require_preflight(pre, PIN)


def test_template_dispatch_refused_before_operations(tmp_path: Path, monkeypatch: Any) -> None:
    import argparse
    import json

    from tools.ops import dispatch

    ops = tmp_path / "untouched-ops"
    monkeypatch.setenv("OPS_DIR", str(ops))
    manifest = tmp_path / "blocked.json"
    manifest.write_text(
        json.dumps(
            {
                "scientific_admission": False,
                "resume": {
                    "unit_kind": "timing-cell",
                    "ledger_path": "inputs/verified-ledger.json",
                    "skip_verified": True,
                    "max_outcome_blind_retries": 2,
                },
            }
        )
    )
    args = argparse.Namespace(
        name="move-blocked",
        manifest=manifest,
        broker=tmp_path,
        cwd=tmp_path,
        state_dir=None,
        dispatch_txt=tmp_path / "no-dispatch.txt",
    )
    with pytest.raises(dispatch.Refused, match="scientific_admission=false"):
        dispatch.dispatch(args)
    assert not ops.exists()
