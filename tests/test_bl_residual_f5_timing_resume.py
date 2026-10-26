"""Offline contracts for F5 immutable timing-unit resume and Mac reduction."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_timing_resume as subject
from research.bl_residual.f5_r10a_annex import allocate


def _binding() -> dict[str, str]:
    return subject.identity(
        namespace="f5-timing-recovery-20261026",
        allocation="a" * 64,
        source_sha256="b" * 64,
        environment_sha256="c" * 64,
        input_sha256="d" * 64,
    )


def _fixture(index: int) -> dict[str, Any]:
    row = next(dict(item) for item in allocate(0) if item.get("allocation_index") == index)
    return {
        **row,
        "elapsed_seconds": 1.0,
        "parity": "PASS",
        "q_shape": [1, 6],
        "full_h300": True,
        "long_body": True,
        "full_h5000_natural": True,
        "watch_fixture_frames": 14_400,
        "t_root": 0.1,
        "t_branch": 0.2,
        "t_nat": 0.3,
        "watch_seconds": 0.4,
    }


def _calibration() -> dict[str, Any]:
    return {
        "schema": "bl-f5-disjoint-calibration/v1",
        "elapsed_seconds": 2.0,
        "arms": {
            arm: {
                "serialized_continuation_exact": True,
                "full_seconds": 1.0,
                "serialized_continuation_seconds": 1.5,
            }
            for arm in ("C", "W")
        },
    }


def _attested(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Supply typed synthetic raw evidence, without mocking the production scanner."""
    from research.bl_residual import f5_environment
    from tests.test_bl_residual_f5_environment import _receipt

    raw = _receipt()
    links = {}
    after = (
        f"after-fixture-{payload['allocation_index']}.json"
        if "allocation_index" in payload
        else "after-calibration.json"
    )
    for field, name in (("environment_before", "before.json"), ("environment_after", after)):
        path = root / "environment" / name
        if not path.exists():
            export_json(path, raw)
        else:
            assert json.loads(path.read_text()) == raw
        links[field] = {
            "path": f"environment/{name}",
            "size": path.stat().st_size,
            "sha256": subject.verify_export(path),
        }
    return {
        **payload,
        **links,
        "kernel_provenance": {
            "assignment_raw_sha256": f5_environment.receipt_sha256(raw),
            "os_kernel": dict(raw["os_kernel"]),
            "platform": raw["platform"],
            "gpu_uuid": raw["gpu_uuid"],
            "before": links["environment_before"],
            "after": links["environment_after"],
        },
    }


def _publish(root, binding, key, payload, **kwargs):
    return subject.publish_unit(root, binding, key, _attested(root, payload), **kwargs)


def _entries(root: Path, calibration: bool = False) -> list[dict[str, Any]]:
    result = [_publish(root, _binding(), index, _fixture(index)) for index in range(672, 684)]
    if calibration:
        result.append(_publish(root, _binding(), subject.CALIBRATION_KEY, _calibration()))
    return result


def _attempt(name: str = "attempt-01") -> dict[str, Any]:
    return {
        "attempt_id": name,
        "identity": _binding(),
        "setup_seconds": 1.0,
        "io_seconds": 2.0,
        "imbalance_seconds": 3.0,
    }


def _complete_ledger(root: Path) -> Path:
    subject.build_ledger(root, _binding(), _entries(root, calibration=True))
    return root / "verified-ledger.json"


def test_mid_fixture_failure_exports_completed_prefix_then_missing_only_reruns(
    tmp_path: Path,
) -> None:
    prefix = [_publish(tmp_path, _binding(), index, _fixture(index)) for index in (672, 673, 674)]
    subject.build_ledger(tmp_path, _binding(), prefix)
    scanned = subject.load_ledger(tmp_path / "verified-ledger.json", _binding())
    assert set(scanned) == {672, 673, 674}
    assert [index for index in range(672, 684) if index not in scanned] == list(range(675, 684))


def test_refuses_markerless_hash_drift_foreign_duplicate_and_inline_ledger(tmp_path: Path) -> None:
    entry = _publish(tmp_path, _binding(), 672, _fixture(672))
    (tmp_path / "timing-units/fixture-672.json.complete").unlink()
    with pytest.raises(subject.ResumeRefusal):
        subject.build_ledger(tmp_path, _binding(), [entry])
    with pytest.raises(subject.ResumeRefusal):
        subject.load_ledger(tmp_path / "missing.json", _binding())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["arms"].pop("C"),
        lambda value: value["arms"]["W"].__setitem__("serialized_continuation_exact", False),
    ],
)
def test_calibration_is_one_complete_coupled_unit(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    value = _calibration()
    mutate(value)
    with pytest.raises(subject.ResumeRefusal):
        _publish(tmp_path, _binding(), subject.CALIBRATION_KEY, value)


@pytest.mark.parametrize(
    "arm,key,value",
    [("C", "full_seconds", float("nan")), ("W", "serialized_continuation_seconds", float("inf"))],
)
def test_calibration_timing_comparisons_must_be_finite(arm: str, key: str, value: float) -> None:
    calibration = _calibration()
    calibration["arms"][arm][key] = value
    with pytest.raises(subject.ResumeRefusal):
        subject._validate_payload(subject.CALIBRATION_KEY, calibration)


def test_mac_reducer_requires_thirteen_canonical_units_and_holds_legacy_attempt_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    subject.build_ledger(tmp_path, _binding(), _entries(tmp_path))
    with pytest.raises(subject.ResumeRefusal, match="thirteen"):
        subject.reduce_ledger(tmp_path / "verified-ledger.json", _binding(), attempts=[_attempt()])

    complete = _complete_ledger(tmp_path / "complete")
    reduced = subject.reduce_ledger(
        complete, _binding(), attempts=[_attempt(), _attempt("attempt-02")]
    )
    assert reduced["unit_count"] == 13
    assert reduced["registered_coverage_passed"] is True
    assert reduced["max_seconds"] == {
        "t_root": 0.1,
        "t_branch": 0.2,
        "t_nat": 0.3,
        "watch_seconds": 0.4,
    }
    assert reduced["coupled_calibration_max_seconds"] == 1.5
    assert reduced["status"] == "HOLD_UNVERIFIED_ATTEMPT_ACCOUNTING"
    assert reduced["accounting_verified"] is False
    assert reduced["archive_registered_conservative_hours"] is None
    assert reduced["all_attempt_setup_io_imbalance_seconds"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_id", "foreign-parent"),
        ("group", "foreign-group"),
        ("seed", 0),
    ],
)
def test_ledger_scanner_refuses_canonical_fixture_allocation_drift(
    tmp_path: Path, field: str, value: Any
) -> None:
    source = tmp_path / "source"
    _complete_ledger(source)
    drift = tmp_path / "drift"
    shutil.copytree(source / "timing-units", drift / "timing-units")
    shutil.copytree(source / "environment", drift / "environment")
    path = drift / "timing-units/fixture-672.json"
    payload = json.loads(path.read_text())
    payload["payload"][field] = value
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    export_json(path, payload)
    entries = []
    for key in range(672, 684):
        item = drift / "timing-units" / f"fixture-{key}.json"
        entries.append(
            {
                "key": key,
                "path": item.relative_to(drift).as_posix(),
                "sha256": subject.verify_export(item),
                "size": item.stat().st_size,
            }
        )
    calibration = drift / "timing-units/calibration.json"
    entries.append(
        {
            "key": subject.CALIBRATION_KEY,
            "path": calibration.relative_to(drift).as_posix(),
            "sha256": subject.verify_export(calibration),
            "size": calibration.stat().st_size,
        }
    )
    with pytest.raises(subject.ResumeRefusal, match="allocation identity"):
        subject.build_ledger(drift, _binding(), entries)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda attempt: attempt.__setitem__("setup_seconds", float("nan")),
        lambda attempt: attempt.__setitem__("io_seconds", float("inf")),
        lambda attempt: attempt.__setitem__("imbalance_seconds", -0.1),
        lambda attempt: attempt.__setitem__("identity", {"foreign": "identity"}),
        lambda attempt: attempt.__setitem__("attempt_id", ""),
    ],
)
def test_mac_reducer_requires_finite_typed_accounting_for_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger = _complete_ledger(tmp_path)
    attempt = _attempt()
    mutate(attempt)
    with pytest.raises(subject.ResumeRefusal):
        subject.reduce_ledger(ledger, _binding(), attempts=[attempt])
    reduced = subject.reduce_ledger(ledger, _binding(), attempts=[])
    assert reduced["status"] == "HOLD_UNVERIFIED_ATTEMPT_ACCOUNTING"
    assert reduced["accounting_verified"] is False
    assert reduced["archive_registered_conservative_hours"] is None
    assert reduced["all_attempt_setup_io_imbalance_seconds"] is None


def test_mac_reducer_refuses_non_darwin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Linux")
    with pytest.raises(subject.ResumeRefusal, match="Mac-only"):
        subject.reduce_ledger(_complete_ledger(tmp_path), _binding(), attempts=[_attempt()])


def test_completed_unit_schema_and_nonfinite_duration_are_refused(tmp_path: Path) -> None:
    entry = _publish(tmp_path, _binding(), 672, _fixture(672))
    record = json.loads((tmp_path / entry["path"]).read_text())
    root = tmp_path / "foreign-schema"
    changed = copy.deepcopy(record)
    changed["schema"] = "foreign-schema"
    path = root / "timing-units/fixture-672.json"
    export_json(path, changed)
    broken = {
        "key": 672,
        "path": "timing-units/fixture-672.json",
        "sha256": subject.verify_export(path),
        "size": path.stat().st_size,
    }
    with pytest.raises(subject.ResumeRefusal):
        subject.build_ledger(root, _binding(), [broken])

    fixture = _fixture(672)
    fixture["elapsed_seconds"] = float("nan")
    with pytest.raises(subject.ResumeRefusal):
        subject._validate_payload(672, fixture)


def test_canary_only_unit_uses_same_export_but_cannot_enter_full_ledger(tmp_path: Path) -> None:
    entry = _publish(tmp_path, _binding(), 673, _fixture(673), canary_only=True)
    assert entry["path"] == "timing-canary/fixture-673.json"
    with pytest.raises(subject.ResumeRefusal):
        subject.build_ledger(tmp_path, _binding(), [entry])
    subject.build_ledger(tmp_path, _binding(), [entry], name="canary-ledger.json", canary_only=True)
    assert set(
        subject.load_ledger(tmp_path / "canary-ledger.json", _binding(), canary_only=True)
    ) == {673}


@pytest.mark.parametrize("key", [672, subject.CALIBRATION_KEY])
@pytest.mark.parametrize(
    "field", ["gpu_uuid", "platform", "os_kernel", "assignment_raw_sha256", "before", "after"]
)
def test_scanner_recomputes_every_kernel_provenance_component(tmp_path, key, field):
    payload = _fixture(key) if isinstance(key, int) else _calibration()
    entry = _publish(tmp_path, _binding(), key, payload)
    path = tmp_path / entry["path"]
    record = json.loads(path.read_text())
    record["payload"]["kernel_provenance"][field] = "forged"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    exported = export_json(path, record)
    entry.update(sha256=exported["sha256"], size=exported["size"])
    with pytest.raises(subject.ResumeRefusal, match="provenance"):
        subject.build_ledger(tmp_path, _binding(), [entry])


@pytest.mark.parametrize("canary", [False, True])
def test_scanner_reopens_raw_links_before_skip_and_refuses_missing_marker(tmp_path, canary):
    entry = _publish(tmp_path, _binding(), 672, _fixture(672), canary_only=canary)
    subject.build_ledger(tmp_path, _binding(), [entry], canary_only=canary)
    (tmp_path / "environment/after-fixture-672.json.complete").unlink()
    with pytest.raises(subject.ResumeRefusal, match="native v6"):
        subject.load_ledger(tmp_path / "verified-ledger.json", _binding(), canary_only=canary)


def test_publish_refuses_even_coherently_rehashed_within_assignment_kernel_drift(tmp_path):
    from tests.test_f5_kernel_provenance_v6 import native

    payload = _attested(tmp_path, _fixture(672))
    path = tmp_path / "environment/after-fixture-672.json"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    exported = export_json(path, native(1))
    payload["environment_after"].update(sha256=exported["sha256"], size=exported["size"])
    payload["kernel_provenance"]["after"] = dict(payload["environment_after"])
    with pytest.raises(subject.ResumeRefusal, match="within-assignment"):
        subject.publish_unit(tmp_path, _binding(), 672, payload)


def test_unit_cannot_be_published_without_kernel_evidence(tmp_path):
    with pytest.raises(subject.ResumeRefusal, match="environment link"):
        subject.publish_unit(tmp_path, _binding(), 672, _fixture(672))


def test_before_file_cannot_substitute_for_post_unit_attestation(tmp_path):
    payload = _attested(tmp_path, _fixture(672))
    payload["environment_after"] = dict(payload["environment_before"])
    payload["kernel_provenance"]["after"] = dict(payload["environment_after"])
    with pytest.raises(subject.ResumeRefusal, match="foreign scope"):
        subject.publish_unit(tmp_path, _binding(), 672, payload)


def test_kernel_strata_preserve_unit_measurements_and_use_slowest_rate(tmp_path):
    from research.bl_residual import f5_environment
    from tests.test_f5_kernel_provenance_v6 import native

    entries = []
    for key in [*range(672, 684), subject.CALIBRATION_KEY]:
        second = not isinstance(key, int) or key >= 678
        root = tmp_path / "attempts" / ("b" if second else "a")
        raw = native(int(second))
        payload = _fixture(key) if isinstance(key, int) else _calibration()
        if isinstance(key, int) and second:
            for name in ("t_root", "t_branch", "t_nat"):
                payload[name] *= 5
        after = f"after-fixture-{key}.json" if isinstance(key, int) else "after-calibration.json"
        links = {}
        for field, name in (("environment_before", "before.json"), ("environment_after", after)):
            path = root / "environment" / name
            if not path.exists():
                export_json(path, raw)
            links[field] = {
                "path": f"environment/{name}",
                "size": path.stat().st_size,
                "sha256": subject.verify_export(path),
            }
        payload.update(links)
        payload["kernel_provenance"] = {
            "assignment_raw_sha256": f5_environment.receipt_sha256(raw),
            "os_kernel": raw["os_kernel"],
            "platform": raw["platform"],
            "gpu_uuid": raw["gpu_uuid"],
            "before": links["environment_before"],
            "after": links["environment_after"],
        }
        entry = subject.publish_unit(root, _binding(), key, payload)
        entry["path"] = (root / entry["path"]).relative_to(tmp_path).as_posix()
        entries.append(entry)
    subject.build_ledger(tmp_path, _binding(), entries)
    reduced = subject.reduce_ledger(tmp_path / "verified-ledger.json", _binding())
    assert len(reduced["kernel_strata"]) == 2
    assert sum(len(row["units"]) for row in reduced["kernel_strata"]) == 13
    assert reduced["max_seconds"]["t_root"] == 0.5
    assert reduced["max_seconds"]["t_branch"] == 1.0
    assert reduced["max_seconds"]["t_nat"] == 1.5
    assert (
        reduced["forecast_kernel_rule"]
        == "SLOWEST_OBSERVED_QUALIFYING_PRIMITIVE_NO_KERNEL_AVERAGING"
    )
