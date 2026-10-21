"""Fail-closed authorization and transport-device tests for L1' probes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_paced_identity.authorization import (
    MODE,
    SCHEMA,
    SCOPE,
    ProbeAuthorization,
    load_probe_authorization,
)
from research.bl_paced_identity.events import EventValidationError, canonical_bytes
from research.bl_paced_identity.fixtures import fixture_identity
from research.bl_paced_identity.transport import THREAD_ENV, ProcessTransport, _pin_worker


def _authorization(
    path: Path, *, workers: int = 4, pending: int = 8, devices: list[str] | None = None
) -> Path:
    path.write_bytes(
        canonical_bytes(
            {
                "schema": SCHEMA,
                "authorized": True,
                "scope": SCOPE,
                "mode": MODE,
                "allowed_workers": workers,
                "allowed_max_pending": pending,
                "allowed_devices": ["cpu", "mps"] if devices is None else devices,
            }
        )
    )
    return path


def test_probe_authorization_is_file_bound_and_permits_only_its_envelope(tmp_path):
    path = _authorization(tmp_path / "probe.json", workers=3, pending=6, devices=["cpu", "mps"])
    authorization = load_probe_authorization(path)
    assert authorization.path == str(path.resolve())
    assert len(authorization.sha256) == 64
    assert authorization.permits(3, 6, "mps")
    assert not authorization.permits(4, 6, "mps")
    assert not authorization.permits(3, 7, "mps")
    assert not authorization.permits(3, 6, "cuda")
    assert authorization.evidence()["scope"] == "L1_PRIME"
    assert authorization.evidence()["mode"] == "PROBE_ONLY"


@pytest.mark.parametrize(
    "value, message",
    [
        ({}, "schema keys"),
        (
            {
                "schema": SCHEMA,
                "authorized": False,
                "scope": SCOPE,
                "mode": MODE,
                "allowed_workers": 4,
                "allowed_max_pending": 8,
                "allowed_devices": ["cpu"],
            },
            "authorized",
        ),
        (
            {
                "schema": SCHEMA,
                "authorized": True,
                "scope": "SERVING",
                "mode": MODE,
                "allowed_workers": 4,
                "allowed_max_pending": 8,
                "allowed_devices": ["cpu"],
            },
            "wrong schema",
        ),
        (
            {
                "schema": SCHEMA,
                "authorized": True,
                "scope": SCOPE,
                "mode": MODE,
                "allowed_workers": 4,
                "allowed_max_pending": 8,
                "allowed_devices": [["mps"]],
            },
            "allowed_devices",
        ),
    ],
)
def test_probe_authorization_fails_closed_for_wrong_or_missing_scope(tmp_path, value, message):
    path = tmp_path / "invalid.json"
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(EventValidationError, match=message):
        load_probe_authorization(path)


def test_expanded_probe_and_mps_need_matching_explicit_authorization(tmp_path):
    identity = fixture_identity(4, 8)
    with pytest.raises(EventValidationError, match="owner"):
        identity.require_authorized(mode="probe", device="mps")
    with pytest.raises(EventValidationError, match="owner"):
        identity.require_authorized(mode="probe", owner_multi_worker=True, device="cpu")
    cpu_only = _authorization(tmp_path / "cpu-only.json", devices=["cpu"])
    with pytest.raises(EventValidationError, match="owner"):
        identity.require_authorized(mode="probe", probe_authorization_file=cpu_only, device="mps")
    mps = _authorization(tmp_path / "mps.json")
    identity.require_authorized(mode="probe", probe_authorization_file=mps, device="mps")
    with pytest.raises(EventValidationError, match="owner"):
        identity.require_authorized(
            mode="serving", owner_multi_worker=True, probe_authorization_file=mps, device="mps"
        )
    fixture_identity().require_authorized(mode="probe", device="cpu")


def test_missing_and_malformed_probe_files_fail_closed(tmp_path):
    with pytest.raises(EventValidationError, match="regular"):
        load_probe_authorization(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not JSON", encoding="utf-8")
    with pytest.raises(EventValidationError, match="unreadable"):
        load_probe_authorization(malformed)


def test_serving_cannot_present_probe_authorization_even_with_serving_owner_flag(
    monkeypatch, tmp_path
):
    path = _authorization(tmp_path / "probe.json")
    controller = SimpleNamespace(identity=fixture_identity(4, 8), _emit=lambda *_args, **_kw: None)
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(sysname="Darwin"))
    with pytest.raises(EventValidationError, match="PROBE_ONLY"):
        ProcessTransport(
            controller,
            lambda _request, _payload: {},
            mode="serving",
            owner_multi_worker=True,
            probe_authorization_file=path,
        )


def test_mps_child_initializer_checks_torch_availability_without_real_accelerator(monkeypatch):
    unavailable_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        set_num_interop_threads=lambda _value: None,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", unavailable_torch)
    monkeypatch.setattr(os, "getpriority", lambda *_args: 10)
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    for name in THREAD_ENV:
        monkeypatch.setenv(name, "2")
    with pytest.raises(RuntimeError, match="MPS transport requested"):
        _pin_worker("mps")


def test_probe_transport_records_file_evidence_and_passes_mps_only_to_child(monkeypatch, tmp_path):
    path = _authorization(tmp_path / "probe.json")
    events: list[tuple[str, dict[str, object]]] = []
    controller = SimpleNamespace(
        identity=fixture_identity(4, 8),
        _emit=lambda kind, **data: events.append((kind, data)),
    )
    captured: dict[str, object] = {}

    class Pool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(sysname="Darwin"))
    monkeypatch.setattr(
        "research.bl_paced_identity.transport.multiprocessing.get_context",
        lambda name: "spawn-context",
    )
    monkeypatch.setattr(
        "research.bl_paced_identity.transport.concurrent.futures.ProcessPoolExecutor", Pool
    )
    monkeypatch.delenv("SNAKE_DQN_DEVICE", raising=False)

    transport = ProcessTransport(
        controller,
        lambda _request, _payload: {},
        mode="probe",
        probe_authorization_file=path,
        device="mps",
    )

    assert transport.device == "mps"
    assert captured["max_workers"] == 4
    assert captured["mp_context"] == "spawn-context"
    assert captured["initargs"] == ("mps",)
    assert "SNAKE_DQN_DEVICE" not in os.environ
    assert events == [
        (
            "transport_authorization",
            {
                "mode": "probe",
                "device": "mps",
                "authorization": {
                    "path": str(path.resolve()),
                    "sha256": load_probe_authorization(path).sha256,
                    "scope": "L1_PRIME",
                    "mode": "PROBE_ONLY",
                    "allowed_workers": 4,
                    "allowed_max_pending": 8,
                    "allowed_devices": ["cpu", "mps"],
                },
            },
        )
    ]


def test_probe_transport_refuses_toctou_narrowed_second_authorization_before_pool(monkeypatch):
    wide = ProbeAuthorization(
        path="/authorization.json",
        sha256="a" * 64,
        allowed_workers=4,
        allowed_max_pending=8,
        allowed_devices=("cpu", "mps"),
    )
    narrowed = ProbeAuthorization(
        path="/authorization.json",
        sha256="b" * 64,
        allowed_workers=1,
        allowed_max_pending=1,
        allowed_devices=("cpu",),
    )
    controller = SimpleNamespace(identity=fixture_identity(4, 8), _emit=lambda *_args, **_kw: None)
    allocated = False

    class Pool:
        def __init__(self, **_kwargs):
            nonlocal allocated
            allocated = True

    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(sysname="Darwin"))
    monkeypatch.setattr(
        "research.bl_paced_identity.authorization.load_probe_authorization", lambda _path: wide
    )
    monkeypatch.setattr(
        "research.bl_paced_identity.transport.load_probe_authorization", lambda _path: narrowed
    )
    monkeypatch.setattr(
        "research.bl_paced_identity.transport.concurrent.futures.ProcessPoolExecutor", Pool
    )

    with pytest.raises(EventValidationError, match="no longer permits"):
        ProcessTransport(
            controller,
            lambda _request, _payload: {},
            mode="probe",
            probe_authorization_file="/authorization.json",
            device="mps",
        )
    assert not allocated
