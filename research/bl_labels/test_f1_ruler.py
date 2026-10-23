"""Focused offline contracts for the F1 root-bank and placement receipt."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.bl_labels import f1_ruler
from research.bl_labels.exports import publish_json


def _root(index: int) -> dict:
    return {
        "root_id": hashlib.sha256(f"root-{index}".encode()).hexdigest(),
        "initial_allocation": {
            "allocation_id": "gate-frozen-000",
            "seed": 1,
            "stream": "gate-frozen",
            "horizon": 5000,
            "plan": {},
        },
        "root_decision": 1,
        "root_digest": "a" * 64,
        "tape_end": 1,
    }


def _session(index: int) -> dict:
    root = _root(index)
    return {
        "schema": "bl-labels-f0-session/v1",
        "status": "COMPLETE",
        "namespace": "test-source",
        "allocation": root["initial_allocation"],
        "audit_roots": [root],
        "audit_tape": [{"decision": 0, "actions": [0, 0, 0, 0, 0, 0]}],
        "rows": [{"decisive": True}],
    }


def test_build_bank_uses_metadata_only_and_lowest_hashes(tmp_path: Path) -> None:
    paths = []
    for index in range(70):
        path = tmp_path / f"session-{index}.json"
        publish_json(path, _session(index))
        paths.append(path)
    out = tmp_path / "bank.json"
    value = f1_ruler.build_bank(paths, out)
    assert value["root_count"] == 64
    assert value["outcome_blind"] is True
    assert "rows" not in value["roots"][0]
    assert f1_ruler.load_bank(out)["roots"] == value["roots"]


def test_bank_refuses_incomplete_or_duplicate_sources(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    value = _session(1)
    value["status"] = "INCOMPLETE"
    publish_json(bad, value)
    with pytest.raises(ValueError, match="complete"):
        f1_ruler.build_bank([bad], tmp_path / "bank.json")


def test_load_bank_requires_frozen_hash_order(tmp_path: Path) -> None:
    roots = [
        {k: v for k, v in _root(i).items() if k != "tape_end"} | {"tape": []} for i in range(64)
    ]
    value = {"schema": f1_ruler.SCHEMA, "root_count": 64, "roots": roots[::-1]}
    path = tmp_path / "bank.json"
    publish_json(path, value)
    with pytest.raises(ValueError, match="selection order"):
        f1_ruler.load_bank(path)


def test_restore_failure_publishes_durable_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from research.bl_labels.f1_units import empty_ledger
    from research.bl_labels.factory import FactoryInvalid

    roots = sorted(
        [_root(i) for i in range(64)],
        key=lambda r: hashlib.sha256(r["root_id"].encode()).hexdigest(),
    )
    bank = tmp_path / "bank.json"
    publish_json(bank, {"schema": f1_ruler.SCHEMA, "root_count": 64, "roots": roots})
    from research.bl_labels.exports import _verify

    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(_verify(bank), "cpu"))

    def fail(*_args, **_kwargs):
        raise FactoryInvalid("injected local restore digest mismatch")

    monkeypatch.setattr(f1_ruler, "run", fail)
    out = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f1_ruler",
            "run",
            "--bank",
            str(bank),
            "--ledger",
            str(ledger),
            "--b8",
            "/unused-b8",
            "--frp3",
            "/unused-frp3",
            "--platform",
            "cpu",
            "--output",
            str(out),
            "--deadline-seconds",
            "10",
            "--probe",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        f1_ruler.main()
    assert exit_info.value.code == 2
    assert _verify(out / "INVALID.json")
    assert _verify(out / "runtime-result.json")


@pytest.mark.parametrize("audit_selected", [False, True])
def test_label_workload_uses_natural_bare_and_only_registered_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audit_selected: bool
) -> None:
    import pickle
    import time
    from types import SimpleNamespace

    from research.bl_labels import factory

    natural = {
        "survives": False,
        "frames_executed": 300,
        "death_decision": 399,
        "final_length": 0,
        "mass_integral": 2.0,
    }
    entry = factory._TapeEntry(
        100, 0, 100, 20, 0, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (0, 1, 2), "a" * 64, {}
    )
    value = {
        "root": {
            "root_id": "b" * 64,
            "root_digest": "c" * 64,
            "source_namespace": "source",
            "initial_allocation": {"allocation_id": "parent", "stream": "gate-scripted"},
        },
        "entry": entry,
        "natural": natural,
        "anchor": 100,
        "snapshot": b"anchor",
        "prefix": {},
        "checkpoint": "d" * 64,
        "restore": None,
        "death_endpoint": 400,
    }
    path = tmp_path / "root.pickle"
    path.write_bytes(pickle.dumps(value))
    monkeypatch.setattr(
        factory,
        "_restore_root",
        lambda *_args, **_kwargs: SimpleNamespace(snapshot=lambda: b"restored"),
    )
    monkeypatch.setattr(
        factory, "_selection_hash", lambda *_args: ("0" if audit_selected else "1") * 64
    )
    calls = []

    def branch(_snapshot, action, depth, **_kwargs):
        calls.append((action, depth))
        if action == 0:
            return dict(natural)
        return {
            "survives": True,
            "frames_executed": 300,
            "death_decision": None,
            "final_length": 30,
            "mass_integral": 30.0,
            "extended": {"survives": True},
        }

    monkeypatch.setattr(factory, "_branch", branch)
    record = f1_ruler._label_root((str(path), time.monotonic() + 10))
    assert calls == ([(1, 300), (2, 300), (0, 300), (1, 1440)] if audit_selected else [(1, 300)])
    assert record["bare_audits"] == int(audit_selected)
    assert record["bare_audit_mismatches"] == 0
    assert not {"decisive", "outcomes", "a_star", "mass", "survives"} & record.keys()


def test_verified_worker_schedules_are_not_recomputed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    from research.bl_labels.f1_units import LABELS, build_ledger

    roots = sorted(
        [_root(i) for i in range(64)],
        key=lambda r: hashlib.sha256(r["root_id"].encode()).hexdigest(),
    )
    monkeypatch.setattr(f1_ruler, "cpu_model", lambda: "fixture CPU")
    monkeypatch.setattr(f1_ruler.os, "cpu_count", lambda: 2)
    original = tmp_path / "original"
    for label, workers in zip(LABELS, (1, 1, 2)):
        publish_json(
            original / "units" / f"config-{label}.json",
            {
                "source_commit": "local-functional",
                "bank_sha256": "a" * 64,
                "platform": "cpu",
                "schedule_label": label,
                "status": "COMPLETE",
                "transport_probe": False,
                "root_count": 64,
                "workers": workers,
                "cpu": {"nproc": 2, "model": "fixture CPU"},
                "records": [{"root_id": root["root_id"]} for root in roots],
            },
        )
    ledger = tmp_path / "verified/ledger.json"
    build_ledger([original], ledger, "a" * 64, "cpu")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("verified schedule was recomputed")

    monkeypatch.setattr(f1_ruler, "_prepare_source", forbidden)
    result = f1_ruler.run(
        {"roots": roots, "source_sessions": []},
        "a" * 64,
        ledger,
        Path("/unused"),
        Path("/unused"),
        "cpu",
        tmp_path / "output",
        time.monotonic() + 10,
    )
    assert result["skipped_verified"] == ["all", "half", "one"]
    assert result["status"] == "COMPLETE"
    assert not list((tmp_path / "output").rglob("config-*.json"))


def test_stage_caps_double_once_and_charge_full_session() -> None:
    from research.bl_labels.f1_manifests import STAGES, envelope

    assert sum(stage["units"] for stage in STAGES.values()) == pytest.approx(1.04)
    effective = [envelope(stage)["effective"] for stage in STAGES]
    assert sum(stage["units"] for stage in effective) == pytest.approx(2.08)
    assert sum(stage["units"] for stage in effective) <= 2.5
    for name, limits in zip(STAGES, effective):
        rate = 0.08 if name.startswith("cpu") else 1.54
        assert rate * limits["minutes"] / 60 + limits["reserve"] <= limits["units"]
        assert limits["minutes"] == 2 * STAGES[name]["minutes"]


def test_lossless_transport_preserves_frozen_bank_identity(tmp_path: Path) -> None:
    import json
    import lzma

    from research.bl_labels.exports import _write_once

    raw = (json.dumps({"frozen": "bank payload"}) + "\n").encode()
    path = tmp_path / "bank.json.xz"
    _write_once(path, lzma.compress(raw, preset=6))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_once(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
    _write_once(Path(str(path) + ".complete"), b"complete\n")
    assert f1_ruler.bank_bytes(path) == raw
    assert f1_ruler.bank_digest(path) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("platform", ["cpu", "l4"])
def test_triplet_upload_preserves_verifier_basename(tmp_path: Path, platform: str) -> None:
    import shutil

    from research.bl_labels.exports import _verify
    from research.bl_labels.f1_manifests import triplet
    from research.bl_labels.f1_units import empty_ledger, load_ledger

    old = tmp_path / f"f1-{platform}-verified-ledger.json"
    publish_json(old, empty_ledger("a" * 64, platform))
    remote = tmp_path / "remote/inputs/f1-verified-ledger.json"
    remote.parent.mkdir(parents=True)
    # Pack-08 verified the local triplet, then renamed only its upload paths.
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(str(old) + suffix, str(remote) + suffix)
    with pytest.raises(ValueError, match="F0 export digest mismatch"):
        load_ledger(remote, "a" * 64, platform)
    with pytest.raises(ValueError, match="preserve basename"):
        triplet(old, "inputs/f1-verified-ledger.json")
    fixed = tmp_path / platform / "f1-verified-ledger.json"
    publish_json(fixed, empty_ledger("a" * 64, platform))
    for row in triplet(fixed, "inputs/f1-verified-ledger.json"):
        shutil.copyfile(row["path"], tmp_path / "remote" / row["dest"])
    assert _verify(remote) == hashlib.sha256(old.read_bytes()).hexdigest()
    assert load_ledger(remote, "a" * 64, platform) == {}


def test_real_canary05_exports_and_failed_pack_destination_regression(tmp_path: Path) -> None:
    """Opt-in offline reproduction using immutable real F0 artifacts, no model jobs."""
    import json
    import os
    import shutil

    from research.bl_labels.exports import _verify
    from research.bl_labels.f1_manifests import charge_failed_probe, triplet
    from research.bl_labels.f1_units import empty_ledger, load_ledger

    artifact_root = os.environ.get("F1_REAL_ARTIFACT_ROOT")
    if not artifact_root:
        pytest.skip("set F1_REAL_ARTIFACT_ROOT to the immutable bl-labels-f0 artifact root")
    artifacts = Path(artifact_root)
    bank = tmp_path / "f1-root-bank.json"
    sessions = sorted((artifacts / "canary-output-05/exports/lives").glob("*--session.json"))
    assert len(sessions) == 31
    f1_ruler.build_bank(sessions, bank)
    packed = artifacts / "f1-pack-08/f1-root-bank.json.xz"
    assert bank.read_bytes() == f1_ruler.bank_bytes(packed)
    digest = f1_ruler.bank_digest(bank)
    assert digest == "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"
    for platform in ("cpu", "l4"):
        manifest = json.loads((artifacts / f"f1-pack-08/f1-{platform}-probe.json").read_text())
        runtime = tmp_path / f"runtime-{platform}"
        for row in manifest["inputs"]:
            if not row["dest"].startswith("inputs/"):
                continue
            source, target = Path(row["path"]), runtime / row["dest"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        assert f1_ruler.bank_digest(runtime / "inputs/f1-root-bank.json.xz") == digest
        remote = runtime / "inputs/f1-verified-ledger.json"
        assert _verify(
            Path(
                next(
                    r["path"]
                    for r in manifest["inputs"]
                    if r["dest"] == "inputs/f1-verified-ledger.json"
                )
            )
        )
        with pytest.raises(ValueError, match="F0 export digest mismatch"):
            load_ledger(remote, digest, platform, manifest["commit"])
        fixed = tmp_path / platform / "f1-verified-ledger.json"
        publish_json(fixed, empty_ledger(digest, platform, source_commit=manifest["commit"]))
        for row in triplet(fixed, "inputs/f1-verified-ledger.json"):
            shutil.copyfile(row["path"], runtime / row["dest"])
        assert load_ledger(remote, digest, platform, manifest["commit"]) == {}
        estimate = manifest["requirements"]["est_minutes"]
        charged = charge_failed_probe(
            manifest, artifacts / f"f1-output-08/{platform}-probe/receipt.json"
        )
        assert manifest["requirements"]["est_minutes"] == estimate
        assert manifest["timeout_minutes"] < 2 * estimate
        assert charged["headroom_admitted"] is False
        assert manifest["budget"]["unit_cap"] + charged["units_charged"] == pytest.approx(
            0.04 if platform == "cpu" else 0.36
        )
        with pytest.raises(ValueError, match="verified stopped packaging failure"):
            charge_failed_probe(manifest, artifacts / f"f1-output-08/{platform}-probe/receipt.json")
