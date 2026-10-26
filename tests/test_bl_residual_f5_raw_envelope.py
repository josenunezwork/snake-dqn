"""Offline raw-byte transport and real F5 skip-loop contracts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import lzma
import shutil
from pathlib import Path
from typing import Any

import pytest

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f5_raw_envelope as envelope
from research.bl_residual import f5_timing_resume as resume
from tests.test_bl_residual_f5_timing_resume import _attested, _binding, _fixture


def _metadata(path: Path, value: Any) -> dict[str, Any]:
    export_json(path, value)
    return resume.metadata_reference(path)


def _prior(
    root: Path,
    binding: dict[str, str],
    key: int = 672,
    *,
    assignment: str = "attempt-0",
    raw: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exports = root / "exports"
    payload = _attested(exports, {**_fixture(key), "timing_attempt_id": assignment})
    if raw is not None:
        from research.bl_residual import f5_environment

        for label in ("environment_before", "environment_after"):
            path = exports / payload[label]["path"]
            _rewrite(path, raw)
            payload[label].update(size=path.stat().st_size, sha256=digest(path))
        payload["kernel_provenance"] = {
            "assignment_raw_sha256": f5_environment.receipt_sha256(raw),
            "os_kernel": dict(raw["os_kernel"]),
            "platform": raw["platform"],
            "gpu_uuid": raw["gpu_uuid"],
            "before": payload["environment_before"],
            "after": payload["environment_after"],
        }
    unit = resume.publish_unit(exports, binding, key, payload)
    control = _metadata(
        root / "control.json",
        {
            "timing_attempt_id": assignment,
            "identity": {
                "source_sha256": binding["source_sha256"],
                "environment_sha256": binding["environment_sha256"],
                "allocation_sha256": binding["allocation"],
            },
            "timing_input_sha256": binding["input_sha256"],
        },
    )
    manifest = _metadata(
        root / "manifest.json",
        {
            "job_id": assignment,
            "commit": "1" * 40,
            "inputs": [{"dest": "control.json", "sha256": control["sha256"]}],
        },
    )
    receipt = _metadata(
        root / "receipt.json",
        {
            "job_id": assignment,
            "manifest_sha256": manifest["sha256"],
            "commit": "1" * 40,
            "stop_verified": True,
            "cleanup_error": None,
            "status": "INCOMPLETE",
            "exports": [
                {field: unit[field] for field in ("path", "size", "sha256")},
                payload["environment_before"],
                payload["environment_after"],
            ],
        },
    )
    return unit, {
        "exports_root": str(exports.absolute()),
        "broker_receipt": receipt,
        "manifest": manifest,
        "control": control,
    }


def _package(tmp_path: Path) -> tuple[Path, list[dict], dict, dict, Path]:
    binding = _binding()
    unit, attempt = _prior(tmp_path / "prior", binding)
    ledger = tmp_path / "prior" / "verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(binding), "units": [unit]})
    package = tmp_path / "package"
    _, inventory = envelope.package_inputs(package, ledger, binding, [attempt])
    return package / "inputs/verified-ledger.json", inventory, unit, attempt, ledger


def _rewrite(path: Path, value: Any) -> None:
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    export_json(path, value)


def test_exact_round_trip_kernel_links_and_original_receipt(tmp_path: Path) -> None:
    ledger, inventory, unit, attempt, original = _package(tmp_path)
    materialized = tmp_path / "materialized"
    result = envelope.load_ledger(ledger, _binding(), inventory, materialized)
    assert set(result) == {672}
    assert ledger.read_bytes() == original.read_bytes()
    assert verify_export(ledger) == verify_export(original)
    source = Path(attempt["exports_root"])
    reconstructed = next(materialized.rglob("timing-units/fixture-672.json"))
    for relative in (
        unit["path"],
        "environment/before.json",
        "environment/after-fixture-672.json",
    ):
        for suffix in ("", ".sha256", ".complete"):
            assert (
                Path(str(reconstructed.parent.parent / relative) + suffix).read_bytes()
                == Path(str(source / relative) + suffix).read_bytes()
            )
    original_payload = json.loads((source / unit["path"]).read_text())["payload"]
    assert result[672]["kernel_provenance"] == original_payload["kernel_provenance"]
    assert inventory[0]["units"] == [unit]
    assert inventory[0]["broker_receipts"] == [attempt["broker_receipt"]]
    assert any(
        p.is_file() and p.read_bytes() == Path(attempt["broker_receipt"]["path"]).read_bytes()
        for p in materialized.rglob("*")
    )
    with pytest.raises((ValueError, FileExistsError)):
        envelope.load_ledger(ledger, _binding(), inventory, materialized)


@pytest.mark.parametrize(
    "kind",
    [
        "outer",
        "raw",
        "kernel",
        "receipt",
        "foreign",
        "duplicate",
        "trailing",
        "truncated",
        "bomb",
        "unused-payload",
        "unused-blob",
    ],
)
def test_tampering_refuses_even_with_resealed_outer_export(tmp_path: Path, kind: str) -> None:
    ledger, inventory, _, _, _ = _package(tmp_path)
    path = ledger.parent / inventory[0]["path"]
    value = json.loads(path.read_text())
    if kind == "outer":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        compressed = base64.b64decode(value["archive"]["data_base64"])
        archive = json.loads(lzma.decompress(compressed))
        if kind in {"raw", "kernel", "receipt", "unused-payload", "unused-blob"}:
            if kind == "raw":
                blob = archive["blobs"][archive["units"][0]["after"]["blob"]]
                archive["payloads"][blob["payload"]] = {"text_utf8": "{}"}
            elif kind == "kernel":
                blob = archive["blobs"][archive["units"][0]["unit_blob"]]
                original = json.loads(archive["payloads"][blob["payload"]]["text_utf8"])
                original["payload"]["kernel_provenance"]["gpu_uuid"] = "foreign-gpu"
                archive["payloads"][blob["payload"]] = {"text_utf8": json.dumps(original)}
            elif kind == "receipt":
                archive["attempts"][0]["receipt"]["text_utf8"] = "{}"
            elif kind == "unused-payload":
                archive["payloads"][envelope.digest_bytes(b"foreign")] = {"text_utf8": "foreign"}
            else:
                parts = {"payload": b"foreign", "sha256": b"foreign", "complete": b"complete\n"}
                archive["payloads"][envelope.digest_bytes(b"foreign")] = {"text_utf8": "foreign"}
                archive["blobs"][envelope._blob_id(parts)] = {
                    "payload": envelope.digest_bytes(b"foreign"),
                    "sha256": base64.b64encode(b"foreign").decode(),
                    "complete": base64.b64encode(b"complete\n").decode(),
                }
            raw = json.dumps(archive).encode()
            value["archive"].update(
                data_base64=base64.b64encode(lzma.compress(raw)).decode(),
                decoded_size=len(raw),
                decoded_sha256=envelope.digest_bytes(raw),
            )
        elif kind == "foreign":
            value["identity"]["source_sha256"] = "0" * 64
        elif kind == "duplicate":
            inventory.append(copy.deepcopy(inventory[0]))
        elif kind == "trailing":
            value["archive"]["data_base64"] = base64.b64encode(compressed + b"extra").decode()
        elif kind == "truncated":
            value["archive"]["data_base64"] = base64.b64encode(compressed[:-3]).decode()
        else:
            monkey_limit = 16
            # A small declared size must bound inflation even when the frozen
            # hard limit permits more bytes.
            value["archive"]["decoded_size"] = monkey_limit
        _rewrite(path, value)
        inventory[0]["sha256"] = digest(path)
        inventory[0]["size"] = path.stat().st_size
    with pytest.raises(ValueError):
        envelope.load_ledger(ledger, _binding(), inventory, tmp_path / "materialized")


@pytest.mark.parametrize("kind", ["raw-link", "duplicate-export", "manifest", "stop", "source"])
def test_original_broker_provenance_is_required_before_packaging(tmp_path: Path, kind: str) -> None:
    binding = _binding()
    unit, attempt = _prior(tmp_path / "prior", binding)
    receipt_path = Path(attempt["broker_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    if kind == "raw-link":
        receipt["exports"].pop()
    elif kind == "duplicate-export":
        receipt["exports"].append(copy.deepcopy(receipt["exports"][0]))
    elif kind == "manifest":
        receipt["manifest_sha256"] = "0" * 64
    elif kind == "stop":
        receipt["stop_verified"] = False
    else:
        binding = {**binding, "source_sha256": "0" * 64}
    _rewrite(receipt_path, receipt)
    attempt["broker_receipt"] = resume.metadata_reference(receipt_path)
    ledger = tmp_path / "prior/verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(binding), "units": [unit]})
    with pytest.raises(ValueError):
        envelope.package_inputs(tmp_path / "package", ledger, binding, [attempt])


def test_duplicate_ledger_and_foreign_paths_refuse(tmp_path: Path) -> None:
    unit, attempt = _prior(tmp_path / "prior", _binding())
    for name, units in (
        ("duplicate", [unit, unit]),
        ("escape", [{**unit, "path": "../timing-units/fixture-672.json"}]),
        ("foreign-role", [{**unit, "path": "timing-units/fixture-673.json"}]),
    ):
        ledger = tmp_path / name / "verified-ledger.json"
        export_json(ledger, {**resume.empty_ledger(_binding()), "units": units})
        with pytest.raises(ValueError):
            envelope.package_inputs(tmp_path / (name + "-package"), ledger, _binding(), [attempt])


@pytest.mark.parametrize("kind", ["omitted", "replaced", "duplicate-attempt"])
def test_receipt_completed_unit_closure_cannot_be_selected_away(tmp_path: Path, kind: str) -> None:
    unit, attempt = _prior(tmp_path / "prior", _binding())
    receipt_path = Path(attempt["broker_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    if kind == "omitted":
        receipt["exports"].append(
            {"path": "timing-units/fixture-673.json", "size": 1, "sha256": "0" * 64}
        )
    elif kind == "replaced":
        receipt["exports"][0]["sha256"] = "0" * 64
    _rewrite(receipt_path, receipt)
    attempt["broker_receipt"] = resume.metadata_reference(receipt_path)
    ledger = tmp_path / "prior/verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(_binding()), "units": [unit]})
    attempts = [attempt, copy.deepcopy(attempt)] if kind == "duplicate-attempt" else [attempt]
    with pytest.raises(ValueError):
        envelope.package_inputs(tmp_path / "package", ledger, _binding(), attempts)


@pytest.mark.parametrize("kind", ["kernel", "foreign-assignment"])
def test_scanner_reopens_raw_links_after_all_outer_hashes_are_resealed(
    tmp_path: Path, kind: str
) -> None:
    from research.bl_residual import f5_environment

    unit, attempt = _prior(tmp_path / "prior", _binding())
    source = Path(attempt["exports_root"])
    unit_path = source / unit["path"]
    record = json.loads(unit_path.read_text())
    payload = record["payload"]
    if kind == "kernel":
        payload["kernel_provenance"]["gpu_uuid"] = "foreign"
    else:
        after_path = source / payload["environment_after"]["path"]
        after = json.loads(after_path.read_text())
        after["gpu_uuid"] = "00000000-0000-0000-0000-000000000099"
        after["gpu_driver_query"]["uuid"] = "GPU-" + after["gpu_uuid"]
        after["gpu_driver_query_raw"] = f"NVIDIA L4, GPU-{after['gpu_uuid']}, 550, 8.9, 23034"
        f5_environment.validate_environment(after)
        _rewrite(after_path, after)
        payload["environment_after"].update(
            size=after_path.stat().st_size, sha256=digest(after_path)
        )
        payload["kernel_provenance"]["after"] = copy.deepcopy(payload["environment_after"])
    _rewrite(unit_path, record)
    unit.update(sha256=digest(unit_path), size=unit_path.stat().st_size)
    receipt_path = Path(attempt["broker_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["exports"] = [
        {field: unit[field] for field in ("path", "size", "sha256")},
        payload["environment_before"],
        payload["environment_after"],
    ]
    _rewrite(receipt_path, receipt)
    attempt["broker_receipt"] = resume.metadata_reference(receipt_path)
    ledger = tmp_path / "prior/verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(_binding()), "units": [unit]})
    with pytest.raises(ValueError, match="provenance|drift"):
        envelope.package_inputs(tmp_path / "package", ledger, _binding(), [attempt])


def test_empty_ledger_route_does_not_create_materialization(tmp_path: Path) -> None:
    ledger = tmp_path / "verified-ledger.json"
    export_json(ledger, resume.empty_ledger(_binding()))
    materialized = tmp_path / "materialized"
    assert envelope.load_ledger(ledger, _binding(), None, materialized) == {}
    assert not materialized.exists()


def test_runtime_is_self_contained_and_refuses_symlink_materialization(tmp_path: Path) -> None:
    ledger, inventory, _, _, _ = _package(tmp_path)
    shutil.rmtree(tmp_path / "prior")
    assert set(envelope.load_ledger(ledger, _binding(), inventory, tmp_path / "fresh")) == {672}
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        envelope.load_ledger(ledger, _binding(), inventory, alias / "materialized")
    assert list(outside.iterdir()) == []


def test_multiple_original_assignments_keep_isolated_before_paths(tmp_path: Path) -> None:
    from tests.test_bl_residual_f5_environment import _receipt

    first, a = _prior(tmp_path / "first", _binding())
    second, b = _prior(
        tmp_path / "second",
        _binding(),
        673,
        assignment="attempt-1",
        raw=_receipt(uuid="B", node="host-b", root="/venv-b"),
    )
    ledger = tmp_path / "verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(_binding()), "units": [first, second]})
    _, inventory = envelope.package_inputs(tmp_path / "package", ledger, _binding(), [a, b])
    result = envelope.load_ledger(
        tmp_path / "package/inputs/verified-ledger.json",
        _binding(),
        inventory,
        tmp_path / "materialized",
    )
    assert set(result) == {672, 673}
    assert result[672]["kernel_provenance"]["gpu_uuid"] == "A"
    assert result[673]["kernel_provenance"]["gpu_uuid"] == "B"
    assert result[672]["environment_before"]["path"] == result[673]["environment_before"]["path"]


def test_carrier_deduplicates_large_native_evidence_within_unchanged_cap(tmp_path: Path) -> None:
    from research.bl_residual import f5_environment
    from tests.test_bl_residual_f5_environment import _receipt

    raw = _receipt()
    contents = {
        f"fixture/file-{i:05d}.py": {
            "raw_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "canonical_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
        }
        for i in range(35_000)
    }
    package = raw["installed_packages"]["torch"]
    package["installed_contents"] = contents
    package["installed_contents_sha256"] = hashlib.sha256(
        f5_environment.canonical_json(
            {key: item["canonical_sha256"] for key, item in contents.items()}
        ).encode()
    ).hexdigest()
    raw["installed_numeric_packages"]["torch"] = copy.deepcopy(package)
    f5_environment.validate_environment(raw)
    raw_bytes = (json.dumps(raw, sort_keys=True) + "\n").encode()
    assert len(raw_bytes) > 13_419_976
    unit, attempt = _prior(tmp_path / "prior", _binding(), raw=raw)
    control_path = Path(attempt["control"]["path"])
    control = json.loads(control_path.read_text())
    # Match the historical control's nested base64 expansion without opening
    # the real pinned receipt or any runtime artifact.
    control["environment_reference"] = {"raw_v5_base64": base64.b64encode(raw_bytes).decode()}
    _rewrite(control_path, control)
    attempt["control"] = resume.metadata_reference(control_path)
    manifest_path = Path(attempt["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"][0]["sha256"] = attempt["control"]["sha256"]
    _rewrite(manifest_path, manifest)
    attempt["manifest"] = resume.metadata_reference(manifest_path)
    receipt_path = Path(attempt["broker_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest_sha256"] = attempt["manifest"]["sha256"]
    _rewrite(receipt_path, receipt)
    attempt["broker_receipt"] = resume.metadata_reference(receipt_path)
    ledger = tmp_path / "prior/verified-ledger.json"
    export_json(ledger, {**resume.empty_ledger(_binding()), "units": [unit]})
    rows, inventory = envelope.package_inputs(tmp_path / "package", ledger, _binding(), [attempt])
    carrier = json.loads(Path(rows[-3]["path"]).read_text())
    archive = json.loads(lzma.decompress(base64.b64decode(carrier["archive"]["data_base64"])))
    # Sidecar names differ between before and after, but their payload bytes
    # remain identical. Unit-independent metadata occurs once in the archive.
    assert len(archive["attempts"]) == 1
    assert len(archive["blobs"]) == 3
    assert len(archive["payloads"]) == 2
    transferred = sum(row["size_bytes"] for row in rows)
    assert transferred + control_path.stat().st_size + 15_000_000 < 45_397_643
    assert set(
        envelope.load_ledger(
            tmp_path / "package/inputs/verified-ledger.json",
            _binding(),
            inventory,
            tmp_path / "materialized",
        )
    ) == {672}


def test_real_qualification_loop_skips_enveloped_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_residual import f5_job, f5_qualification
    from tests.test_bl_residual_f5_resume_integration import _binding as loop_binding
    from tests.test_bl_residual_f5_resume_integration import (
        _control,
        _inputs,
        _install_outer_seams,
    )

    executed: list[int] = []
    calibrations: list[str] = []
    _install_outer_seams(monkeypatch, executed, calibrations)
    binding = loop_binding()
    unit, attempt = _prior(tmp_path / "prior", binding)
    original = tmp_path / "prior/verified-ledger.json"
    export_json(original, {**resume.empty_ledger(binding), "units": [unit]})
    package = tmp_path / "package"
    _, inventory = envelope.package_inputs(package, original, binding, [attempt])
    control = {**_control(), "timing_raw_envelopes": inventory}
    monkeypatch.setattr(f5_job, "_control", lambda inputs: control)
    inputs = _inputs(tmp_path)
    report = f5_qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        tmp_path / "run",
        ledger=package / "inputs/verified-ledger.json",
    )
    assert executed == list(range(673, 684))
    assert calibrations == ["whole-cw"]
    assert (
        report["samples"][0]["kernel_provenance"]
        == json.loads((Path(attempt["exports_root"]) / unit["path"]).read_text())["payload"][
            "kernel_provenance"
        ]
    )
    canary_root = tmp_path / "matching-canary"
    canary = f5_qualification.run(
        inputs,
        inputs / "unused-fixtures.json",
        canary_root,
        ledger=package / "inputs/verified-ledger.json",
        transport_only=True,
        canary_tiny_workload=True,
    )
    assert (
        canary_root / "resume/raw-envelope-imports/unit-672/timing-units/fixture-672.json"
    ).is_file()
    assert [entry["key"] for entry in canary["canary_only"]] == [672, 673]
    assert [sample["allocation_index"] for sample in canary["samples"]] == [672, 673]
    assert canary["status"] == "COMPLETE_TRANSPORT_ONLY_NOT_TIMING_QUALIFIED"


def test_changed_source_inventory_includes_the_envelope_module() -> None:
    from research.bl_residual.f5_package import ROOT, source_inventory

    sources = source_inventory(ROOT)
    relative = "research/bl_residual/f5_raw_envelope.py"
    assert sources[relative] == digest(ROOT / relative)


def test_builder_seals_envelope_inventory_through_existing_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_residual import f5_timing_recovery_package as package
    from research.colab_cli.transfer_policy import validate_f5_timing_inventory

    # Replace only Git/archive and historic checkpoint metadata. The real
    # builder seals control, ledger, manifests and authority into fresh paths.
    sources = {"research/bl_residual/f5_fit.py": "a" * 64}
    monkeypatch.setattr(
        package, "_git", lambda repo, *args: "" if args[0] == "status" else "1" * 40
    )
    monkeypatch.setattr(package, "_git_bytes", lambda *args: b"offline-source")
    sources["research/bl_residual/f5_fit.py"] = package.hashlib.sha256(
        b"offline-source"
    ).hexdigest()
    monkeypatch.setattr(package.f5_package, "source_inventory", lambda repo: sources)
    monkeypatch.setattr(
        package.subprocess, "check_output", lambda *args, **kwargs: b"source-archive"
    )
    monkeypatch.setattr(package, "_metadata_manifest", lambda path: {})
    monkeypatch.setattr(package, "_prior_environment_control", lambda prior: {"offline": True})
    private = tmp_path / "private"
    private.mkdir()
    rows = []
    for name in (
        "timing-fixtures.json",
        "champion_a5_freespace_20260621.pth",
        "best_apex_fs.pth",
        "best_apex_stage1_fs.pth",
        "best_apex_pre_fs.pth",
        "frp3_m3_s12_u60000_20261005.pth",
    ):
        path = private / name
        path.write_bytes(b"offline-checkpoint-metadata")
        rows.append(
            {
                "dest": name,
                "path": str(path),
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    rows.append(
        {
            "dest": "b8.pth",
            "url": package.B8_URL,
            "sha256": package.B8_SHA256,
            "size_bytes": 6_562_005,
        }
    )
    monkeypatch.setattr(package, "_source_inputs", lambda prior: rows)
    empty = tmp_path / "empty"
    package.build(empty, tmp_path / "unused.json")
    empty_control = json.loads((empty / "inputs/control.json").read_text())
    assert "timing_raw_envelopes" not in empty_control
    binding = json.loads((empty / "inputs/verified-ledger.json").read_text())["identity"]
    unit, attempt = _prior(tmp_path / "prior", binding)
    original = tmp_path / "prior/verified-ledger.json"
    ledger_ref = _metadata(original, {**resume.empty_ledger(binding), "units": [unit]})
    spec = tmp_path / "resume-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "bl-f5-raw-envelope-resume-spec/v1",
                "ledger": ledger_ref,
                "attempts": [attempt],
                "attempt_index": 1,
            }
        )
    )
    nonempty = tmp_path / "nonempty"
    package.build(nonempty, tmp_path / "unused.json", resume_spec=spec)
    control = json.loads((nonempty / "inputs/control.json").read_text())
    assert control["timing_input_sha256"] == empty_control["timing_input_sha256"]
    assert control["identity"] == empty_control["identity"]
    assert control["allocation"] != empty_control["allocation"]
    assert control["timing_attempt_id"].endswith("-02")
    assert (nonempty / "inputs/verified-ledger.json").read_bytes() == original.read_bytes()
    for name in ("timing-operation.draft.json", "family-canary.draft.json"):
        manifest = json.loads((nonempty / name).read_text())
        authority = manifest["f5_timing_synthetic_inventory"]
        validate_f5_timing_inventory(authority)
        assert {row["dest"] for row in manifest["inputs"]} == {
            row["dest"] for row in authority["inputs"]
        }
        for row in manifest["inputs"]:
            if "path" in row:
                assert row["sha256"] == digest(Path(row["path"]))
                assert row["size_bytes"] == Path(row["path"]).stat().st_size
        names = {row["dest"] for row in manifest["inputs"]}
        assert all("environment/" not in name for name in names)
        assert {unit["path"] + suffix for suffix in ("", ".sha256", ".complete")} <= names
    assert set(
        envelope.load_ledger(
            nonempty / "inputs/verified-ledger.json",
            binding,
            control["timing_raw_envelopes"],
            tmp_path / "materialized",
        )
    ) == {672}
