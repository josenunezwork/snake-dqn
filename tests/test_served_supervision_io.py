"""Immutable binding and unavailable-live-observer gates; no live process control."""

from __future__ import annotations

from copy import deepcopy

import pytest

from research.bl_paced_identity.record import export_json
from research.bl_served_search import remote_supervision, served_tick_probe
from research.bl_served_search import supervision_io as io


def test_registry_triplet_never_admits_a_caller_capability_flag(tmp_path, monkeypatch):
    registry = {
        "policy": io.POLICY,
        "ruling_sha256": io.RULING_SHA256,
        "roots": [{"id": "root"}],
        "live_accounting_capability": "COMPLETE",
        "loaded_source_verified": True,
    }
    monkeypatch.setattr(remote_supervision, "validate_registry", lambda _: {"valid": True})
    monkeypatch.setattr(io, "verify_root", lambda _: None)
    path = tmp_path / "registry.json"
    export_json(path, registry)
    assert io.read_registry(path) == registry
    with pytest.raises(RuntimeError, match="admission unavailable"):
        io.read_registry(path, live=True)


@pytest.mark.parametrize("key", ["policy", "ruling_sha256"])
def test_registry_rejects_policy_ruling_tamper(tmp_path, key):
    value = {"policy": io.POLICY, "ruling_sha256": io.RULING_SHA256}
    value[key] = "foreign"
    path = tmp_path / "registry.json"
    export_json(path, value)
    with pytest.raises(ValueError, match="policy/ruling"):
        io.read_registry(path)


def test_registry_rejects_incomplete_triplet(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        io.read_registry(path)


@pytest.mark.parametrize("kind", ["host", "panel", "check"])
def test_supervised_commands_refuse_before_outputs_lease_or_game_imports(
    tmp_path, monkeypatch, kind
):
    calls = []

    def refuse(path, *, live=False):
        calls.append((path, live))
        raise RuntimeError(io.BLOCKER)

    monkeypatch.setattr(io, "read_registry", refuse)
    output = tmp_path / "must-not-exist"
    argv = [
        "--classified-host-load",
        "--remote-supervision-registry",
        str(tmp_path / "registry.json"),
        "--output",
        str(output),
    ]
    if kind == "host":
        argv += ["--host-only-seconds", "600"]
    elif kind == "panel":
        argv += ["--qualification-receipt", str(tmp_path / "missing-report.json")]
    else:
        argv += ["--check"]
    with pytest.raises(RuntimeError, match="admission unavailable"):
        served_tick_probe.main(argv)
    assert calls == [(tmp_path / "registry.json", True)]
    assert not output.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--host-only-seconds", "600"],
        ["--host-only-seconds", "601"],
        ["--qualification-receipt", "foreign.json"],
        ["--remote-supervision-registry", "r.json", "--smoke"],
        ["--remote-supervision-registry", "r.json", "--reduce", "raw.json"],
    ],
)
def test_malformed_supervision_command_refused(argv):
    with pytest.raises(SystemExit) as exc:
        served_tick_probe.main(argv)
    assert exc.value.code == 2


def _host_intent():
    registry = {
        "policy": io.POLICY,
        "ruling_sha256": io.RULING_SHA256,
        "ruling_hash": io.RULING_SHA256,
        "policy_source_hash": io.digest(io.policy_sources()),
    }
    host = {
        "load_policy": io.POLICY,
        "supervisor_registry": registry,
        "supervision_journal": [],
        "supervision_cpu_evidence": {},
        "supervision_identity_observations": [],
        "supervision_point_screens": [],
        "cpu_samples": [],
        "supervision_replay": {"valid": True, "errors": []},
    }
    intent = {
        "load_policy": io.POLICY,
        "ruling_sha256": io.RULING_SHA256,
        "registry_sha256": io.digest(registry),
        "policy_source_sha256": io.policy_sources(),
    }
    return host, intent


@pytest.mark.parametrize(
    "key", ["load_policy", "ruling_sha256", "registry_sha256", "policy_source_sha256"]
)
def test_independent_reducer_refuses_intent_mismatch(key):
    host, intent = _host_intent()
    intent[key] = "mismatch"
    with pytest.raises(ValueError, match="mismatch"):
        io.replay_host(host, intent)


def test_independent_reducer_recomputes_and_rejects_summary_mismatch(monkeypatch):
    host, intent = _host_intent()
    monkeypatch.setattr(remote_supervision, "replay", lambda *a, **kw: {"valid": True})
    with pytest.raises(ValueError, match="differs from independent replay"):
        io.replay_host(host, intent)


def test_even_consistent_synthetic_replay_cannot_certify_a_live_panel(monkeypatch):
    host, intent = _host_intent()
    monkeypatch.setattr(
        remote_supervision, "replay", lambda *a, **kw: deepcopy(host["supervision_replay"])
    )
    with pytest.raises(ValueError, match="admission unavailable"):
        io.replay_host(host, intent)


def test_registry_digest_is_the_actual_payload_digest(tmp_path):
    value = {"one": "two"}
    digest = export_json(tmp_path / "value.json", value)
    assert io.digest(value) == digest
