"""Tiny future-only subprocess smokes; never control a remote/live job."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from research.bl_served_search.supervision_identity import verify_identity
from tests.test_served_supervision_identity import fixture
from tools.sandbox import supervision_accounting as subject
from tools.sandbox.collector_v2 import Collector, collector_route


def _journal(tmp_path: Path) -> subject.ImmutableJournal:
    return subject.ImmutableJournal(tmp_path / "journal")


def _launch(journal, oid="poll", code="pass"):
    return subject.launch_operation(
        journal,
        oid,
        [sys.executable, "-c", code],
        payload=subject.operation_payload("usage", b""),
        parent_identity="900@100",
    )


def test_payload_binding_and_exact_64_kib_cap() -> None:
    assert subject.operation_payload("read_state", b"a" * 65535, b"b")["application_bytes"] == 65536
    with pytest.raises(subject.AccountingError, match="65536"):
        subject.operation_payload("read_state", b"a" * 65536, b"b")
    for template in ("status", "terminal_state", "train", "pytest"):
        with pytest.raises(subject.AccountingError, match="not permitted"):
            subject.operation_payload(template, b"")


def test_wait4_child_entirely_between_samples_and_flat_lifecycle(tmp_path) -> None:
    journal = _journal(tmp_path)
    child, tracked = _launch(journal, code="sum(i*i for i in range(10000))")
    subject.record_exec(journal, tracked)
    result = subject.wait4_exit(journal, child, tracked)
    subject.record_operation_end(journal, tracked.operation_id)
    journal.seal()
    events = subject.read_journal(journal.directory)
    assert [row["kind"] for row in events] == [
        "operation_start",
        "launch",
        "exec",
        "exit",
        "operation_end",
    ]
    assert result["returncode"] == 0
    assert result["final_cpu_ns"] == result["user_cpu_ns"] + result["system_cpu_ns"] > 0
    assert result["final_scope"] == "subtree"
    assert result["descendants_complete"] is False
    assert type(tracked.birth_ns) is int and tracked.birth_ns != tracked.launch_monotonic_ns
    assert "payload" not in events[0]


def test_external_reaping_never_forges_success(tmp_path) -> None:
    journal = _journal(tmp_path)
    child, tracked = _launch(journal, code="raise SystemExit(23)")
    os.waitpid(child.pid, 0)
    with pytest.raises(subject.AccountingError, match="already reaped"):
        subject.wait4_exit(journal, child, tracked)
    assert child.returncode != 0
    assert child.final_rusage is None


def test_poll_and_communicate_preserve_final_rusage(tmp_path) -> None:
    child = subject.Wait4Popen([sys.executable, "-c", "print('ok')"], stdout=subprocess.PIPE)
    stdout, _ = child.communicate(timeout=5)
    assert stdout == b"ok\n"
    assert child.poll() == 0
    before = child.final_rusage.copy()
    assert child.wait() == 0 and child.final_rusage == before


def test_journal_concurrency_seal_and_tamper_refusal(tmp_path) -> None:
    journal = _journal(tmp_path)
    errors = []

    def write(index):
        try:
            journal.append("root_identity", {"index": index})
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors
    journal.seal()
    rows = subject.read_journal(journal.directory)
    assert [row["seq"] for row in rows] == list(range(1, 9))
    with pytest.raises(subject.AccountingError, match="after journal seal"):
        journal.append("late", {})
    with pytest.raises(FileExistsError):
        subject.ImmutableJournal(journal.directory)
    (journal.events_dir / "00000002-root_identity.json.complete").unlink()
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        subject.read_journal(journal.directory)


def test_reserved_fields_time_regression_and_nonfinite_payload_refused(tmp_path) -> None:
    journal = _journal(tmp_path)
    journal.append("one", {}, monotonic_ns=10)
    with pytest.raises(subject.AccountingError, match="strictly increase"):
        journal.append("two", {}, monotonic_ns=10)
    journal.append("three", {}, monotonic_ns=11)
    journal.seal()
    assert [row["seq"] for row in subject.read_journal(journal.directory)] == [1, 2]
    for key in ("seq", "kind", "mono_ns", "previous_sha256"):
        with pytest.raises(subject.AccountingError, match="reserved"):
            _journal(tmp_path / key).append("one", {key: 3})
    with pytest.raises(ValueError):
        subject._canonical_json({"x": float("nan")})


@pytest.mark.parametrize("payload", [{"seq": 3}, {"path": Path("payload")}, {"x": float("nan")}])
def test_journal_prewrite_rejection_does_not_poison_or_advance_chain(tmp_path, payload):
    journal = _journal(tmp_path)
    journal.append("before", {}, monotonic_ns=10)
    with pytest.raises(subject.AccountingError):
        journal.append("invalid", payload, monotonic_ns=11)
    assert len(list(journal.events_dir.glob("*.json"))) == 1
    journal.append("after", {}, monotonic_ns=11)
    journal.seal()
    rows = subject.read_journal(journal.directory)
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[1]["previous_sha256"] == rows[0]["sha256"]


def test_journal_partial_write_failure_still_prevents_append_and_seal(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    original = subject.export_json

    def partial_write(path, record):
        path.write_text("partial")
        raise OSError("simulated interrupted triplet")

    monkeypatch.setattr(subject, "export_json", partial_write)
    with pytest.raises(subject.AccountingError, match="immutable journal write failed"):
        journal.append("one", {})
    monkeypatch.setattr(subject, "export_json", original)
    with pytest.raises(subject.AccountingError, match="after journal failure"):
        journal.append("two", {})
    with pytest.raises(subject.AccountingError, match="seal failed journal"):
        journal.seal()


@pytest.mark.parametrize("shebang", [False, True])
def test_collector_pathlike_arguments_preserve_complete_journal(tmp_path, shebang):
    script = tmp_path / "echo-path.py"
    script.write_text(
        "#!" + str(Path(sys.executable).resolve()) + "\nimport sys\nprint(sys.argv[1])\n"
    )
    script.chmod(0o700)
    argument = tmp_path / "payload"
    args = [script, argument] if shebang else [Path(sys.executable), script, argument]
    collector = Collector(tmp_path / "collector")
    result = collector.run(args, capture_output=True, text=True, timeout=5)
    assert result.stdout == str(argument) + "\n"
    assert collector.run([sys.executable, "-c", "pass"], timeout=5).returncode == 0
    assert not collector.seal()["errors"]
    rows = subject.read_journal(collector.journal.directory)
    assert [row["kind"] for row in rows] == [
        "operation_start",
        "launch",
        "exec",
        "exit",
        "operation_end",
    ] * 2
    for row in rows:
        if row["kind"] in {"launch", "exec"}:
            assert all(isinstance(arg, str) for arg in row["identity"]["argv"])


def test_native_counter_rejects_float_or_missing_final(tmp_path) -> None:
    journal = _journal(tmp_path)
    child, tracked = _launch(journal)
    child.wait()
    child.final_rusage = {"user_cpu_ns": 0.1, "system_cpu_ns": 0}
    with pytest.raises(subject.AccountingError, match="exact finite"):
        subject.wait4_exit(journal, child, tracked)


def test_schema_records_roundtrip_through_identity_verifier(tmp_path) -> None:
    registry, events, observations = fixture("usage")
    journal = _journal(tmp_path)
    for row in events:
        payload = {
            key: value for key, value in row.items() if key not in {"kind", "seq", "mono_ns"}
        }
        journal.append(row["kind"], payload, monotonic_ns=row["mono_ns"])
    journal.seal()
    result = verify_identity(registry, subject.read_journal(journal.directory), observations)
    assert result["valid"], result["errors"]


def test_collector_records_short_children_nonzero_exit_and_no_admission(tmp_path) -> None:
    collector = Collector(tmp_path / "collector")
    for code, expected in (("print('tiny')", 0), ("raise SystemExit(23)", 23)):
        result = collector.run([sys.executable, "-c", code], capture_output=True, timeout=5)
        assert result.returncode == expected
    status = collector.seal()
    assert status["capability"] == "NOT_ADMISSIBLE" and not status["descendants_complete"]
    rows = subject.read_journal(collector.journal.directory)
    assert [row["kind"] for row in rows] == [
        "operation_start",
        "launch",
        "exec",
        "exit",
        "operation_end",
    ] * 2
    exits = [row for row in rows if row["kind"] == "exit"]
    assert all(row["final_cpu_ns"] > 0 for row in exits)
    assert exits[1]["returncode"] == 23
    assert all(row["template"] == "ordinary" for row in rows if row["kind"] == "operation_start")
    assert not status["errors"], status["errors"]


def test_collector_concurrent_calls_have_unique_operations_and_final_counters(tmp_path) -> None:
    collector = Collector(tmp_path / "collector")
    outputs = []

    def run():
        outputs.append(
            collector.run([sys.executable, "-c", "print('ok')"], capture_output=True, timeout=5)
        )

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    collector.seal()
    rows = subject.read_journal(collector.journal.directory)
    starts = [row["operation_id"] for row in rows if row["kind"] == "operation_start"]
    assert len(set(starts)) == 2 and len(outputs) == 2
    for oid in starts:
        kinds = [row["kind"] for row in rows if row.get("operation_id") == oid]
        assert kinds == ["operation_start", "launch", "exec", "exit", "operation_end"]


def test_collector_timeout_preserves_owned_child_cleanup_and_final(tmp_path) -> None:
    collector = Collector(tmp_path / "collector")
    with pytest.raises(subprocess.TimeoutExpired):
        collector.run(
            [sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.02, capture_output=True
        )
    collector.seal()
    rows = subject.read_journal(collector.journal.directory)
    final = next(row for row in rows if row["kind"] == "exit")
    assert final["returncode"] < 0 and type(final["final_cpu_ns"]) is int
    assert rows[-1]["kind"] == "operation_end"


def test_accounting_failure_does_not_prevent_safety_call_or_restore(tmp_path, monkeypatch) -> None:
    from research.colab_cli import supervisor

    original = supervisor.subprocess
    collector = Collector(tmp_path / "collector")

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(collector.journal, "append", fail)
    with collector_route(collector):
        result = supervisor.subprocess.run([sys.executable, "-c", "pass"], timeout=5)
        assert result.returncode == 0
    assert supervisor.subprocess is original and subprocess.Popen is not subject.Wait4Popen
    assert collector.errors


def test_collector_reader_binds_context_status_and_current_source(tmp_path) -> None:
    from research.bl_paced_identity.record import export_json
    from tools.sandbox.collector_v2 import read_collector

    collector = Collector(tmp_path / "collector")
    collector.run([sys.executable, "-c", "pass"], timeout=5)
    collector.seal()
    loaded = read_collector(collector.journal.directory)
    assert len(loaded["supervision_journal"]) == 5
    # Even correctly re-exported metadata must agree with the terminal binding.
    context = collector.journal.directory / "collector.json"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(context) + suffix).unlink()
    export_json(context, {"version": "foreign"})
    with pytest.raises(subject.AccountingError, match="source/context/status mismatch"):
        read_collector(collector.journal.directory)


@pytest.mark.parametrize(
    "template", ["usage", "sessions", "read_state", "renew_lease", "heartbeat_upload"]
)
def test_qualified_callsite_template_recognition_without_provider(tmp_path, monkeypatch, template):
    import json
    from types import SimpleNamespace

    from research.bl_served_search.supervision_identity import (
        CALLSITES,
        heartbeat_request,
        rpc_request,
    )
    from research.colab_cli import supervisor
    from tools.sandbox import collector_v2

    binary = tmp_path / "colab"
    binary.write_text("#!" + str(Path(sys.executable).resolve()) + "\n")
    config = tmp_path / "lane/sessions.json"
    config.parent.mkdir()
    cli = supervisor.CLI(str(binary), config, "session")
    (config.parent / "assignment.json").write_text(
        json.dumps({"session": "session", "endpoint": "endpoint", "generation": "generation"})
    )
    cli.assignment_record.write_text(json.dumps({"endpoint": "endpoint"}))
    hook = config.parent / "cli-hook/sitecustomize.py"
    hook.parent.mkdir()
    hook.write_text("# synthetic hook; not admission\n")
    args = [str(binary), "--auth", "oauth2", "--config", str(config)]
    request_path = tmp_path / "request"
    if template in {"usage", "sessions"}:
        args += [template]
        callsite = template
    elif template == "heartbeat_upload":
        request_path.write_text(heartbeat_request("generation"))
        args += [
            "upload",
            "-s",
            "session",
            str(request_path),
            "/content/snake-cli/generation/lease-heartbeat.json",
        ]
        callsite = "_put_lease"
    else:
        request_path.write_text(rpc_request(template, "generation", "SNAKE_RPC_" + "a" * 32))
        args += ["exec", "-s", "session", "-f", str(request_path), "--timeout", "5"]
        callsite = CALLSITES[template].removeprefix("CLI.")
    frames = [
        SimpleNamespace(
            function="_call_once",
            frame=SimpleNamespace(
                f_globals={
                    "__name__": "research.colab_cli.supervisor",
                    "__file__": supervisor.__file__,
                },
                f_locals={"self": cli},
            ),
        ),
        SimpleNamespace(
            function=callsite,
            frame=SimpleNamespace(f_globals={"__name__": "research.colab_cli.supervisor"}),
        ),
    ]
    monkeypatch.setattr(collector_v2.inspect, "stack", lambda **kwargs: frames)
    collector = Collector(tmp_path / "collector")
    metadata, request = collector.metadata(args, 5)
    assert metadata["template"] == template
    assert metadata["request_digest"] == subject.sha256_bytes(request)
    assert metadata["helper_sources"][str(hook)] == metadata["helper_hook"]
    # Matching names from an unrelated module are not qualified callsites.
    frames[1].frame.f_globals["__name__"] = "unqualified_module"
    assert collector.metadata(args, 5)[0]["template"] == "ordinary"
    frames[1].frame.f_globals["__name__"] = "research.colab_cli.supervisor"
    if template in {"read_state", "renew_lease"}:
        frames[1].function = "read_state"  # Not a method in the pinned CLI.
        assert collector.metadata(args, 5)[0]["template"] == "ordinary"
        frames[1].function = callsite
        request_path.write_text("result=mod.read_state()")
        rejected, request = collector.metadata(args, 5)
        assert rejected["template"] == "ordinary" and request == b""
        assert "payload_path" not in rejected and "nonce" not in rejected
    # A generic CLI caller or changed assignment never inherits a template.
    frames[1].function = "local_training"
    assert collector.metadata(args, 5)[0]["template"] == "ordinary"
    cli.assignment_record.write_text(json.dumps({"endpoint": "foreign"}))
    assert collector.metadata(args, 5)[0]["template"] == "ordinary"
    collector.seal()


@pytest.mark.parametrize("force,template", [(True, "read_state"), (False, "renew_lease")])
def test_real_qualified_cli_rpc_shapes_without_provider(tmp_path, monkeypatch, force, template):
    """Exercise renew_runtime_lease -> poll_rpc -> rpc -> call -> _call_once.

    The pinned CLI emits read_state via force=True; it has no CLI.read_state.
    Only the subprocess boundary is replaced, so stack, argv, and temporary RPC
    bytes all come from the unchanged qualified source.
    """
    import json
    import re
    from types import SimpleNamespace

    from research.bl_served_search import supervision_identity as identity
    from research.colab_cli import supervisor

    binary = tmp_path / "colab"
    binary.write_text("#!" + str(Path(sys.executable).resolve()) + "\n")
    config = tmp_path / "lane/sessions.json"
    config.parent.mkdir()
    cli = supervisor.CLI(str(binary), config, "session")
    (config.parent / "assignment.json").write_text(
        json.dumps({"session": "session", "endpoint": "endpoint", "generation": "generation"})
    )
    cli.assignment_record.write_text(json.dumps({"endpoint": "endpoint"}))
    collector = Collector(tmp_path / "collector")
    observed = []

    def intercept(args, **kwargs):
        assert args[:9] == [
            str(binary),
            "--auth",
            "oauth2",
            "--config",
            str(config),
            "exec",
            "-s",
            "session",
            "-f",
        ]
        assert args[10] == "--timeout" and len(args) == 12
        request = Path(args[9]).read_text()
        nonce = re.search(r"SNAKE_RPC_[0-9a-f]{32}", request).group()
        assert request == identity.rpc_request(template, "generation", nonce)
        metadata, raw = collector.metadata(args, kwargs["timeout"])
        assert metadata["template"] == template
        assert metadata["provenance"]["callsite"] == "CLI.renew_runtime_lease"
        assert metadata["rpc_timeout_seconds"] == float(args[-1])
        assert 0 < kwargs["timeout"] <= float(args[-1])
        assert raw == request.encode()
        assert metadata["request_digest"] == subject.sha256_bytes(raw)
        observed.append(metadata)
        return subprocess.CompletedProcess(
            args, 0, nonce + json.dumps({"ok": True, "value": {}}), ""
        )

    monkeypatch.setattr(supervisor, "subprocess", SimpleNamespace(run=intercept))
    assert cli.renew_runtime_lease("/content/snake-cli/generation", "generation", force=force) == {}
    assert len(observed) == 1
    collector.seal()


def test_primitive_postspawn_identity_failure_preserves_child_for_cleanup(tmp_path, monkeypatch):
    import psutil

    journal = _journal(tmp_path)
    monkeypatch.setattr(
        psutil, "Process", lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid))
    )
    with pytest.raises(subject.LaunchAccountingError) as raised:
        _launch(journal, code="pass")
    assert raised.value.child.wait(timeout=5) == 0
    assert raised.value.child.final_rusage is not None


def test_collector_missing_exec_stays_invalid_and_never_forges_a_final(tmp_path):
    collector = Collector(tmp_path / "collector")
    with pytest.raises(FileNotFoundError):
        collector.run([str(tmp_path / "missing-program")], timeout=5)
    status = collector.seal()
    assert any("spawn/exec failed" in error for error in status["errors"])
    rows = subject.read_journal(collector.journal.directory)
    assert [row["kind"] for row in rows] == ["operation_start", "operation_end"]
    assert not any(row["kind"] == "exit" for row in rows)


def test_ordinary_bulk_response_keeps_completion_and_exact_raw_text_bytes(tmp_path):
    collector = Collector(tmp_path / "collector")
    payload = b"x" * 65537 + b"\r\n"
    result = collector.run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*65537+b'\\r\\n')"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout == "x" * 65537 + "\n"
    collector.seal()
    rows = subject.read_journal(collector.journal.directory)
    assert rows[-1]["kind"] == "operation_end"
    assert rows[-1]["response_bytes"] == len(payload)
    assert rows[-1]["response_digest"] == subject.sha256_bytes(payload)


def test_lost_rusage_is_explicitly_missing_never_zero(tmp_path, monkeypatch):
    original = subject.Wait4Popen.communicate

    def lose(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.final_rusage = None
        return result

    monkeypatch.setattr(subject.Wait4Popen, "communicate", lose)
    collector = Collector(tmp_path / "collector")
    collector.run([sys.executable, "-c", "pass"], timeout=5)
    status = collector.seal()
    assert "missing final rusage" in status["errors"]
    rows = subject.read_journal(collector.journal.directory)
    assert next(row for row in rows if row["kind"] == "exit")["final_cpu_ns"] is None


def test_root_cli_guard_helpers_are_collected_and_all_routes_restored(tmp_path):
    from research.colab_cli import cli_guard, release, supervisor
    from tools.ops import dispatch

    originals = [module.subprocess for module in (cli_guard, release, supervisor, dispatch)]
    collector = Collector(tmp_path / "collector")
    with collector_route(collector):
        assert (
            cli_guard.subprocess
            is release.subprocess
            is supervisor.subprocess
            is dispatch.subprocess
        )
        assert cli_guard.subprocess.run([sys.executable, "-c", "pass"], timeout=5).returncode == 0
    assert [module.subprocess for module in (cli_guard, release, supervisor, dispatch)] == originals
    collector.seal()
    assert len(subject.read_journal(collector.journal.directory)) == 5
