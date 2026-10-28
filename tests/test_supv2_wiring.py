"""Offline dispatch wiring and fail-closed live boundaries; no provider or tmux."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from research.bl_paced_identity.record import export_json
from research.bl_served_search import served_tick_probe, supervision_dispatch
from research.bl_served_search import supervision_io as io
from research.bl_served_search import supervision_qualification
from tools.ops import dispatch
from tools.sandbox import supervisor_v2
from tools.sandbox.collector_v2 import Collector

ROOT = Path(__file__).resolve().parents[1]


def packet(tmp_path):
    return {
        "job_id": "future-v2",
        "output_dir": str(tmp_path / "output"),
        "sandbox_admission": {"schema": "fixture"},
        "sandbox_supervision": "collector-v2",
    }


def test_selectors_keep_both_legacy_defaults_and_choose_explicit_v2(tmp_path):
    value = packet(tmp_path)
    assert dispatch.supervisor_launcher(ROOT, value) == ROOT / "tools/sandbox/launch_v2.sh"
    del value["sandbox_supervision"]
    assert dispatch.supervisor_launcher(ROOT, value) == ROOT / "tools/sandbox/launch.sh"
    assert dispatch.supervisor_launcher(ROOT, {}) == ROOT / "research/colab_cli/launch.sh"
    with pytest.raises(dispatch.Refused, match="broker"):
        dispatch.supervisor_launcher(tmp_path, packet(tmp_path))


@pytest.mark.parametrize("opt", [None, False, {}, "", "v2", "collector-v3"])
def test_unknown_opt_ins_refused(tmp_path, opt):
    value = packet(tmp_path)
    value["sandbox_supervision"] = opt
    with pytest.raises(dispatch.Refused, match="collector-v2"):
        dispatch.sandbox_collector_directory(value)


@pytest.mark.parametrize("output", ["relative", "", "/tmp/../tmp/job"])
def test_collector_requires_canonical_absolute_output(tmp_path, output):
    value = packet(tmp_path)
    value["output_dir"] = output
    with pytest.raises(dispatch.Refused):
        dispatch.sandbox_collector_directory(value)


def test_non_sandbox_cannot_select_v2(tmp_path):
    value = packet(tmp_path)
    del value["sandbox_admission"]
    with pytest.raises(dispatch.Refused):
        dispatch.supervisor_launcher(ROOT, value)


@pytest.mark.parametrize("bound", [True, False])
def test_admission_binds_the_opt_in_before_preflight(tmp_path, monkeypatch, bound):
    from tools.sandbox.tests.test_admission import _bind_preflight, _packet

    value = _packet(tmp_path, monkeypatch)
    original = deepcopy(value)
    value["sandbox_supervision"] = "collector-v2"
    _bind_preflight(monkeypatch, value, tmp_path, original=value if bound else original)
    if bound:
        dispatch.require_sandbox_admission(value, ROOT)
    else:
        with pytest.raises(dispatch.Refused, match="committed preflight binding"):
            dispatch.require_sandbox_admission(value, ROOT)


def test_retry_has_a_separate_evidence_directory(tmp_path):
    value = packet(tmp_path)
    first = dispatch.sandbox_collector_directory(value)
    value["output_dir"] += "-retry-1"
    second = dispatch.sandbox_collector_directory(value)
    assert first == tmp_path / "output.supervision"
    assert second == tmp_path / "output-retry-1.supervision"


def test_validation_selects_v2_without_creating_a_collector(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(packet(tmp_path)))
    calls = []
    monkeypatch.setattr(dispatch, "command", lambda argv, **kw: calls.append(argv))
    dispatch.validate(ROOT, ROOT, tmp_path / "state", path, tmp_path / "validate.log")
    assert calls[0][0] == str(ROOT / "tools/sandbox/launch_v2.sh")
    assert calls[0][-2:] == ["validate", str(path)]
    assert calls[0][4] == str((Path.home() / ".local/bin/colab").resolve())
    assert not (tmp_path / "output.supervision").exists()


def test_start_propagates_host_collector_path_and_quotes_it(tmp_path, monkeypatch):
    value = packet(tmp_path)
    value["output_dir"] = str(tmp_path / "output with spaces '$()'")
    manifest = tmp_path / "packet.json"
    manifest.write_text(json.dumps(value))
    log = tmp_path / "logs" / "job.log"
    log.parent.mkdir()
    calls = []
    monkeypatch.setattr(dispatch, "command", lambda argv, **kw: calls.append(argv))
    dispatch.start(ROOT, ROOT, tmp_path / "state", manifest, log, "fixture", 1e20)
    assert len(calls) == 1
    body = calls[0][-1]
    import shlex

    # start() nests the environment command inside a quoted bash -c body.
    inner = shlex.split(body)[-1]
    assert (
        shlex.quote("SNAKE_SANDBOX_COLLECTOR_DIR=" + value["output_dir"] + ".supervision") in inner
    )
    assert "launch_v2.sh" in body
    assert str((Path.home() / ".local/bin/colab").resolve()) in inner
    assert "SNAKE_SANDBOX_COLLECTOR_DIR" not in json.loads(manifest.read_text()).get("env", {})


@pytest.mark.parametrize("kind", ["existing", "dangling", "control", "bench"])
def test_start_refuses_before_log_or_registration(tmp_path, monkeypatch, kind):
    value = packet(tmp_path)
    manifest = tmp_path / "packet.json"
    manifest.write_text(json.dumps(value))
    directory = dispatch.sandbox_collector_directory(value)
    if kind == "existing":
        directory.mkdir()
    if kind == "dangling":
        directory.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(dispatch, "command", lambda *a, **kw: pytest.fail("command executed"))
    log = tmp_path / "must-not-exist"
    with pytest.raises(dispatch.Refused):
        dispatch.start(
            ROOT,
            ROOT,
            tmp_path / "state",
            manifest,
            log,
            "fixture",
            1e20,
            control_nice0=kind == "control",
            small_bench_config=tmp_path if kind == "bench" else None,
        )
    assert not log.exists()


def test_validate_entrypoint_does_not_install_collector(tmp_path, monkeypatch):
    from tools.sandbox import supervisor

    calls = []
    monkeypatch.setattr(
        sys, "argv", ["module", "--state-dir", "state", "--colab", "cli", "validate", "manifest"]
    )
    monkeypatch.setattr(supervisor, "main", lambda: calls.append("validate"))
    monkeypatch.setattr(supervisor_v2, "Collector", lambda _: pytest.fail("collector created"))
    supervisor_v2.main()
    assert calls == ["validate"]


def test_run_entrypoint_refuses_unbound_environment_before_collector(tmp_path, monkeypatch):
    value = packet(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(
        sys, "argv", ["module", "--state-dir", "state", "--colab", "cli", "run", str(manifest)]
    )
    monkeypatch.setenv("SNAKE_SANDBOX_COLLECTOR_DIR", str(tmp_path / "foreign"))
    monkeypatch.setattr(supervisor_v2, "Collector", lambda _: pytest.fail("collector created"))
    with pytest.raises(ValueError, match="opted-in"):
        supervisor_v2.main()


def dispatch_fixture(tmp_path, monkeypatch):
    value = packet(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value))
    argv = [
        "/python",
        "-m",
        "tools.sandbox.supervisor_v2",
        "--state-dir",
        str(tmp_path / "state"),
        "--colab",
        "/cli",
        "run",
        str(manifest),
    ]
    startup = {
        "pid": 501,
        "uid": 501,
        "birth_ns": 1234000,
        "argv": argv,
        "cwd": str(ROOT),
        "interpreter": "/python",
        "manifest": str(manifest),
        "manifest_sha256": io.file_digest(manifest),
    }
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps({"endpoint": "endpoint", "session": "session", "generation": "generation"})
    )
    context = {
        "version": "sandbox-child-collector/v2",
        "policy": io.POLICY,
        "sources": Collector.sources(),
        "startup": startup,
        "root_process_id": "501@1234000",
    }
    directory = dispatch.sandbox_collector_directory(value)
    directory.mkdir()
    export_json(directory / "collector.json", context)
    root = {**startup, "assignment": "endpoint", "session": "session", "generation": "generation"}
    registry = {"roots": [root], "assignment_path": str(assignment_path)}
    monkeypatch.setattr(io, "capture_registry", lambda _: deepcopy(registry))
    monkeypatch.setattr(io, "verify_root", lambda _: None)
    return manifest, directory, registry, context


def test_dispatch_registry_binds_startup_manifest_assignment_and_actual_source(
    tmp_path, monkeypatch
):
    manifest, directory, _, context = dispatch_fixture(tmp_path, monkeypatch)
    registry = supervision_dispatch.capture_dispatch_registry(manifest)
    binding = registry["dispatch_binding"]
    assert binding["manifest_sha256"] == io.file_digest(manifest)
    assert binding["collector_sha256"] == io.digest(context)
    assert binding["collector_directory"] == str(directory)
    assert binding["job_id"] == "future-v2"


@pytest.mark.parametrize("change", ["pid", "birth_ns", "argv", "assignment", "terminal", "packet"])
def test_registry_refuses_restarts_foreign_assignments_and_changed_packets(
    tmp_path, monkeypatch, change
):
    manifest, directory, registry, _ = dispatch_fixture(tmp_path, monkeypatch)
    if change in {"pid", "birth_ns", "argv", "assignment"}:
        registry["roots"][0][change] = "foreign"
    elif change == "terminal":
        (directory / "status.json").write_text("{}")
    else:
        manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError):
        supervision_dispatch.capture_dispatch_registry(manifest)


def test_combined_parser_reaches_the_unchanged_closed_live_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        io, "read_registry", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError(io.BLOCKER))
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="admission unavailable"):
        served_tick_probe.main(
            [
                "--load-policy",
                "classified-desktop-supervised-v2",
                "--classified-host-load",
                "--desktop-app-registry",
                "desktop.json",
                "--remote-supervision-registry",
                "remote.json",
                "--host-only-seconds",
                "600",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_qualification_cannot_consume_a_flag_as_observer_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(
        supervision_qualification,
        "capture_dispatch_registry",
        lambda _: {
            "live_accounting_capability": "QUALIFIED_LIVE_V2",
            "loaded_source_verified": True,
        },
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="admission unavailable"):
        supervision_qualification.main(
            [
                "--dispatch-manifest",
                "packet.json",
                "--desktop-app-registry",
                "desktop.json",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_qualification_check_reports_missing_capability_without_output(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(supervision_qualification, "capture_dispatch_registry", lambda _: {})
    output = tmp_path / "not-created"
    assert (
        supervision_qualification.main(
            [
                "--dispatch-manifest",
                "packet.json",
                "--desktop-app-registry",
                "desktop.json",
                "--output",
                str(output),
                "--check",
            ]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "NOT_ADMISSIBLE"
    assert report["qualification_published"] is False
    assert report["requested_seconds"] == 600
    assert not output.exists()


def test_registry_rechecks_assignment_after_source_capture(tmp_path, monkeypatch):
    manifest, _, registry, _ = dispatch_fixture(tmp_path, monkeypatch)

    def change(_):
        Path(registry["assignment_path"]).write_text("{}")

    monkeypatch.setattr(io, "verify_root", change)
    with pytest.raises(ValueError, match="changed or became terminal"):
        supervision_dispatch.capture_dispatch_registry(manifest)
