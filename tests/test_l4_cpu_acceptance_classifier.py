"""CPU acceptance corrupt-export evidence reaches the v2 classifier intact."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels.l4_faults_v2 import classify_session
from research.colab_cli import (
    acceptance_faults,
    cli_guard,
    injection_manifest,
    policy,
    supervisor,
    transfers,
)
from research.colab_cli.acceptance_faults import AcceptanceObserver
from tests.test_rpc_qualification_matrix import _Clock, _FakeRuntime


def _receipt(cli: supervisor.CLI) -> dict[str, object]:
    return {
        "job_id": "cpu-acceptance-classifier",
        "status": "PASS",
        "failure_class": None,
        "error": None,
        "stop_verified": True,
        "cleanup_error": None,
        "cli_calls": copy.deepcopy(cli.call_records),
        "transfers": copy.deepcopy(cli.transfer_records),
        "contents_transfers": copy.deepcopy(cli.contents_transfers),
        "marker_census": copy.deepcopy(cli.marker_census),
    }


class _AcceptanceProvider:
    """Map the fake provider's isolated filesystem for ordinary observer RPCs."""

    def __init__(self, runtime: _FakeRuntime, root: Path) -> None:
        self.runtime = runtime
        self.root = root

    def __call__(self, *args: str, timeout: float) -> str:
        if args[0] == "download" and not self.runtime._remote(args[-2]).is_file():
            raise supervisor.Refused("HTTP 404 Not Found")
        if args[0] != "exec":
            return self.runtime(*args, timeout=timeout)
        source = Path(args[args.index("-f") + 1])
        original = source.read_text()
        if "SNAKE_MARKER_CENSUS" in original:
            rewritten = original.replace(
                "path = Path(name)", f"path = Path({str(self.root)!r}) / name.lstrip('/')"
            )
        else:
            rewritten = original.replace(
                "/content/cpu-acceptance", str(self.root / "content/cpu-acceptance")
            )
        source.write_text(rewritten)
        try:
            return self.runtime(*args, timeout=timeout)
        finally:
            source.write_text(original)


def test_real_cpu_corrupt_export_producer_is_a_clean_planned_classifier_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the observer's real export/download path against only a fake provider."""
    clock = _Clock()
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    monkeypatch.setattr(transfers.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(transfers.time, "sleep", clock.sleep)
    monkeypatch.setattr(policy, "poll_delay", lambda _attempt: 0.0)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_args, **_kwargs: True)

    work = tmp_path / "cpu-acceptance"
    output = work / "output"
    output.mkdir(parents=True)
    cli = supervisor.CLI("fake-colab", work / "config", "cpu-acceptance-classifier")
    runtime = _FakeRuntime(work / "remote", clock, "clean")
    runtime.cli = cli
    monkeypatch.setattr(cli, "_call_once", _AcceptanceProvider(runtime, work / "remote"))

    observer = AcceptanceObserver("unused")
    observer._corrupt_export_rejection(cli, "/content/cpu-acceptance", output)
    assert observer.records["failure-and-corrupt-export"]["corrupt_export_rejected"] is True

    # These are the same deferred marker observations the busy-client path
    # records before final cleanup census; use the real Contents calls and
    # supervisor binding rather than manufacturing transfer rows.
    for name in ("client-start.json", "client-finished.json"):
        remote = f"/content/cpu-acceptance/{name}"
        payload = ('{"request":"cpu"}\n').encode()
        target = work / "remote" / remote.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        local = output / name
        first_call = len(cli.call_records)
        cli.call("download", "-s", cli.session, remote, str(local), timeout=policy.READ_SECONDS)
        cli.record_client_marker_download(
            first_call=first_call,
            remote=remote,
            local_size=local.stat().st_size,
            local_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
        )
    cli.collect_marker_census()
    manifest = {
        "schema": injection_manifest.SCHEMA,
        "session_job_id": "cpu-acceptance-classifier",
        "purpose": "real acceptance corrupt-export classifier regression",
        "written_before_dispatch": True,
        "entries": injection_manifest.acceptance_entries("cpu"),
    }
    result = classify_session(_receipt(cli), manifest)

    assert result["verdict"] == "PASS", result
    assert result["counted"] == []
    assert result["evidence_missing"] == []
    excluded = result["excluded_injections"]
    assert len(excluded) == 2
    assert {row["injection_id"] for row in excluded} == {"corrupt-export-rejection"}
    assert {row["type"] for row in excluded} == {"deliberate-hash-mismatch-export"}
    assert sum(row.get("parent_failure") is True for row in excluded) == 1
    assert sum("transfers_index" in row for row in excluded) == 1

    missing = _receipt(cli)
    missing["marker_census"].pop("client_files")
    assert classify_session(missing, manifest)["verdict"] == "EVIDENCE_MISSING"
    mismatched = _receipt(cli)
    remote = "/content/cpu-acceptance/client-start.json"
    mismatched["marker_census"]["client_files"][remote]["sha256"] = "0" * 64
    assert classify_session(mismatched, manifest)["verdict"] == "FAIL"


def test_real_cpu_ready_acceptance_sequence_through_classifier(tmp_path, monkeypatch):
    """Execute the CPU ready observer, including its owned-client loss boundary.

    The fake implements only the provider's kernel/Contents/account responses
    and the intentionally terminated external CLI process. Observer methods,
    RPC envelopes, exports, reconnect, deferred census, and classifier are real.
    """
    clock = _Clock()
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    monkeypatch.setattr(policy, "poll_delay", lambda _attempt: 0.1)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_k: True)
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    output = tmp_path / "output"
    output.mkdir()
    remote_base = tmp_path / "remote"
    remote_root = remote_base / "content/cpu-acceptance"
    remote_root.mkdir(parents=True)
    payload = b"tiny real export\n"
    digest = hashlib.sha256(payload).hexdigest()
    exports = remote_root / "output"
    exports.mkdir()
    (exports / "busy-snapshot.bin").write_bytes(payload)
    (exports / "busy-snapshot.bin.sha256").write_text(digest + "  busy-snapshot.bin\n")
    (exports / "busy-snapshot.bin.complete").write_text("complete\n")
    probe = {
        "phase": "transfer_ready",
        "pid": 812,
        "execution_id": "cpu-execution",
        "snapshot": {"path": "busy-snapshot.bin", "bytes": len(payload), "sha256": digest},
    }
    state = remote_root / "acceptance-state.json"
    state.write_text(json.dumps(probe))
    (remote_root / "runtime.py").write_text(
        "def read_state(root, generation, renew=False):\n    return {'terminal': None}\n"
    )
    runtime = _FakeRuntime(remote_base, clock, "clean")
    boundary = _AcceptanceProvider(runtime, remote_base)
    clients = []
    pending_silence = [0]

    def provider(*args, timeout):
        if args[0] == "sessions":
            return "[cpu-acceptance-classifier] endpoint | Hardware: CPU\n"
        if args[0] == "usage":
            return "Current balance: 1000\nUsage rate: 0.08\nActive assignments: 1\n"
        if args[0] == "url":
            return "https://example.invalid/cpu\n"
        if args[0] == "upload" and args[-1].endswith("/client-release.json"):
            result = boundary(*args, timeout=timeout)
            request = json.loads(Path(args[-2]).read_text())["request"]
            (remote_root / "client-finished.json").write_text(
                json.dumps({"request": request, "finished_at": 100.0})
            )
            state.write_text(json.dumps({**probe, "phase": "transfer_finished"}))
            return result
        if args[0] == "exec":
            source_path = Path(args[args.index("-f") + 1])
            source = source_path.read_text()
            if "time.sleep(65)" in source:
                pending_silence[0] += 1
                if pending_silence[0] <= 5:
                    clock.now += 13
                    return runtime._reply(source, pending=True)
                source_path.write_text(source.replace("time.sleep(65)", "pass"))
                try:
                    return boundary(*args, timeout=timeout)
                finally:
                    source_path.write_text(source)
        return boundary(*args, timeout=timeout)

    original_cli = supervisor.CLI

    class ObservedCLI(original_cli):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._call_once = provider
            self.accounting = SimpleNamespace(check=lambda *_a, **_k: None)
            self.release_credential = None
            runtime.cli = self

    class OwnedClient:
        def __init__(self, argv, **kwargs):
            assert argv[0] == "fake-colab"
            assert kwargs["umask"] == 0o077
            source = Path(argv[argv.index("-f") + 1]).read_text()
            request = ast.literal_eval(re.search(r"^request=(.+)$", source, re.M).group(1))
            assert "client-start.json" in source and "client-finished.json" in source
            (remote_root / "client-start.json").write_text(
                json.dumps({"request": request, "started_at": 0.0})
            )
            self.pid = 32123
            self.returncode = None
            self.terminated = 0
            clients.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated += 1
            self.returncode = -signal.SIGTERM

        def wait(self, timeout):
            assert timeout == 5
            return self.returncode

    monkeypatch.setattr(supervisor, "CLI", ObservedCLI)
    monkeypatch.setattr(acceptance_faults.subprocess, "Popen", OwnedClient)
    cli = ObservedCLI("fake-colab", tmp_path / "config", "cpu-acceptance-classifier")
    manifest = {
        "schema": injection_manifest.SCHEMA,
        "session_job_id": "cpu-acceptance-classifier",
        "purpose": "actual CPU ready acceptance sequence",
        "written_before_dispatch": True,
        "entries": injection_manifest.acceptance_entries("cpu"),
    }
    observer = AcceptanceObserver("unused", injections={"cpu-acceptance-classifier": manifest})
    context = {
        "cli": cli,
        "root": "/content/cpu-acceptance",
        "output": output,
        "generation": "generation-cpu",
        "manifest": {
            "job_id": "cpu-acceptance-classifier",
            "requirements": {"gpu": "cpu"},
            "budget": {"other_assignments": {}},
            "max_export_bytes": 1024,
        },
    }
    observer("ready", context)
    reconnected = context["cli"]
    assert reconnected is not cli
    assert len(clients) == 1 and clients[0].terminated == 1
    assert reconnected.marker_census_attempted is False
    assert set(observer.records) == {
        "url-no-second-allocation",
        "silence-timeout",
        "failure-and-corrupt-export",
        "client-loss-reconnect",
    }
    # The final observer RPC has completed before the one authorized census.
    assert reconnected.call_records is cli.call_records
    assert reconnected.transfer_records is cli.transfer_records
    assert reconnected.client_marker_downloads is cli.client_marker_downloads
    # Broker.run's finally retains the original client after the reconnect.
    cli.collect_marker_census()
    receipt = _receipt(cli)
    result = classify_session(receipt, manifest)
    assert result["verdict"] == "PASS", result
    assert result["counted"] == result["evidence_missing"] == []
    assert len(reconnected.client_marker_downloads) == 2
    assert any(
        row["injection_id"] == "client-loss-finished-absent"
        for row in result["excluded_injections"]
    )
    destination = os.environ.get("SNAKE_CPU_ACCEPTANCE_TRACE")
    if destination:
        with Path(destination).open("x") as stream:
            json.dump(
                {
                    "receipt": receipt,
                    "injection_manifest": manifest,
                    "classification": result,
                    "observer_gates": observer.records,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    for change in ("missing", "malformed", "mismatch"):
        altered = copy.deepcopy(receipt)
        files = altered["marker_census"]["client_files"]
        remote = "/content/cpu-acceptance/client-start.json"
        if change == "missing":
            files.pop(remote)
        elif change == "malformed":
            files[remote] = {}
        else:
            files[remote]["sha256"] = "0" * 64
        classified = classify_session(altered, manifest)
        assert classified["verdict"] == ("FAIL" if change == "mismatch" else "EVIDENCE_MISSING")


def test_real_noninjected_corrupt_suffix_failure_is_counted(tmp_path, monkeypatch):
    """A provider failure in the planned export is never the planned rejection."""
    clock = _Clock()
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_k: True)
    cli = supervisor.CLI("fake-colab", tmp_path / "config", "cpu-acceptance-classifier")
    runtime = _FakeRuntime(tmp_path / "remote", clock, "clean")
    runtime.cli = cli
    boundary = _AcceptanceProvider(runtime, tmp_path / "remote")

    def provider(*args, timeout):
        if args[0] == "download":
            raise supervisor.Refused("HTTP 503 Service Unavailable")
        return boundary(*args, timeout=timeout)

    monkeypatch.setattr(cli, "_call_once", provider)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(supervisor.Refused, match="transfer failure"):
        AcceptanceObserver("unused")._corrupt_export_rejection(
            cli, "/content/cpu-acceptance", output
        )
    manifest = {
        "schema": injection_manifest.SCHEMA,
        "session_job_id": "cpu-acceptance-classifier",
        "purpose": "noninjected provider failure on planned export",
        "written_before_dispatch": True,
        "entries": injection_manifest.acceptance_entries("cpu"),
    }
    result = classify_session(_receipt(cli), manifest)
    assert result["verdict"] == "FAIL"
    assert any(row["class"] == "C6" for row in result["counted"])
    assert any(
        row["remote"].endswith("-fault/output/acceptance-corrupt.bin")
        for row in cli.transfer_records
    )
    assert not result["excluded_injections"]
