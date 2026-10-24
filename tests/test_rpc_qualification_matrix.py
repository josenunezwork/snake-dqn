"""Producer-to-checker matrix for the L4 v2 hold-resolution ruling.

Each receipt in this file starts with a real ``supervisor.CLI`` and the real
``transfers`` adapter.  The fake runtime replaces only the provider boundary
(``CLI._call_once``), the kernel-idle probe, and the clock.  The checker then
consumes the resulting call/transfer/census records.  The negative cases alter
one recorded fact after production to model a provider or evidence failure;
they do not construct synthetic call records by hand.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.bl_labels.l4_faults_v2 import classify_session
from research.colab_cli import cli_guard, policy, supervisor, transfers
from research.colab_cli.failures import ClassifiedFailure
from research.rpc_qualification import checker
from research.rpc_qualification.broker import InjectionObserver


def _kernel(source: str, runtime_root: Path) -> str:
    """Execute the real RPC envelope against an isolated fake /content root."""
    source = source.replace("/content/.snake-rpc-", str(runtime_root / ".snake-rpc-"))
    source = source.replace("threading.Thread(target=worker, daemon=True).start()", "worker()")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "<matrix-fake-kernel>", "exec"), {})
    return output.getvalue()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


class _FakeRuntime:
    """Provider-side fake used behind the real supervisor and transfer code."""

    def __init__(self, root: Path, clock: _Clock, mode: str) -> None:
        self.root = root
        self.clock = clock
        self.mode = mode
        self.exec_counts: dict[str, int] = {}
        self.calls: list[tuple[str, ...]] = []
        self.timeout_injected = False
        self.upload_failures = 0

    def _remote(self, value: str) -> Path:
        return self.root / value.lstrip("/")

    @staticmethod
    def _request(source: str) -> tuple[str, str]:
        request = re.search(r"# SNAKE_REQUEST_ID ([0-9a-f]{32})", source)
        digest = re.search(r"# SNAKE_REQUEST_SHA256 ([0-9a-f]{64})", source)
        assert request and digest
        return request.group(1), digest.group(1)

    @staticmethod
    def _transfer_path(source: str) -> str:
        """Read the generated transfer spec without depending on its repr.

        ``transfers._rpc`` embeds the operation payload as a Python literal;
        parsing its AST keeps this fake coupled to the real adapter rather
        than to whether the implementation happens to use single quotes or
        JSON-style double quotes.
        """
        module = ast.parse(source)
        # ``request_code`` wraps the transfer body in a worker function, so
        # the assignment is nested under that function in the final source.
        for statement in ast.walk(module):
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "spec" for target in statement.targets
            ):
                continue
            spec = ast.literal_eval(statement.value)
            if isinstance(spec, dict) and isinstance(spec.get("path"), str):
                return spec["path"]
        raise AssertionError("transfer verification source lacks a literal spec.path")

    def _reply(self, source: str, *, pending: bool = False, value: Any = True) -> str:
        request_id, digest = self._request(source)
        nonce = "SNAKE_RPC_" + request_id
        reply = {
            "ok": True,
            "pending": pending,
            "request_id": request_id,
            "sha256": digest,
            "value": value,
        }
        return nonce + json.dumps(reply, sort_keys=True)

    def _exec(self, argv: tuple[str, ...], timeout: float) -> str:
        source = Path(argv[argv.index("-f") + 1]).read_text()
        if "SNAKE_MARKER_CENSUS" in source:
            census = _kernel(source, self.root)
            return census

        request_id, _digest = self._request(source)
        count = self.exec_counts.get(request_id, 0) + 1
        self.exec_counts[request_id] = count

        if self.mode in {"recover", "recover-pending"} and not self.timeout_injected:
            self.timeout_injected = True
            _kernel(source, self.root)
            self.clock.now += float(timeout)
            raise subprocess.TimeoutExpired(argv, timeout)
        if self.mode == "pending" and count == 1:
            return self._reply(source, pending=True)
        if self.mode == "recover-pending" and count == 2:
            final = _kernel(source, self.root)
            match = re.search(r"(SNAKE_RPC_[0-9a-f]{32})(\{[^\n]*\})", final)
            assert match
            reply = json.loads(match.group(2))
            reply["pending"] = True
            return match.group(1) + json.dumps(reply, sort_keys=True)
        if self.mode == "503":
            self.clock.now += min(float(timeout), 1.0)
            raise supervisor.Refused("503 Service Unavailable")
        if "SNAKE_TRANSFER_VERIFY_FILE" in source:
            remote = self._transfer_path(source)
            path = self._remote(remote)
            # Execute the real envelope and only remap the fake provider's
            # remote file path into the isolated runtime root.
            return _kernel(source.replace(remote, str(path)), self.root)
        return _kernel(source, self.root)

    def __call__(self, *args: str, timeout: float) -> str:
        self.calls.append(tuple(args))
        command = args[0]
        if command == "exec":
            return self._exec(tuple(args), timeout)
        if command == "upload":
            source, remote = Path(args[-2]), args[-1]
            if self.mode == "flaky-transfer" and self.upload_failures == 0:
                self.upload_failures += 1
                raise OSError("connection reset by fake ContentsClient")
            target = self._remote(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return ""
        if command == "download":
            remote, destination = args[-2], Path(args[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._remote(remote), destination)
            return ""
        return ""


def _receipt(cli: supervisor.CLI, *, status: str = "PASS", **extra: Any) -> dict[str, Any]:
    return {
        "job_id": "matrix-job",
        "status": status,
        "failure_class": None,
        "error": None,
        "stop_verified": True,
        "cleanup_error": None,
        "cli_calls": copy.deepcopy(cli.call_records),
        "transfers": copy.deepcopy(cli.transfer_records),
        "contents_transfers": copy.deepcopy(cli.contents_transfers),
        "marker_census": copy.deepcopy(cli.marker_census),
        **extra,
    }


def _produce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "clean"
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = _Clock()
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    monkeypatch.setattr(transfers.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(transfers.time, "sleep", clock.sleep)
    monkeypatch.setattr(policy, "poll_delay", lambda _attempt: 0.0)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_args, **_kwargs: True)

    cli = supervisor.CLI("fake-colab", tmp_path / "config", "matrix-session")
    runtime_mode = "503" if mode == "503-injected" else mode
    runtime = _FakeRuntime(tmp_path / "remote", clock, runtime_mode)
    runtime.cli = cli
    monkeypatch.setattr(cli, "_call_once", runtime)

    try:
        if mode == "503-injected":
            request_id = "a" * 32
            restore, state = InjectionObserver._arm_post_send_loss(cli, request_id)
            try:
                cli.rpc("result = {'matrix': True}", timeout=30, request_id=request_id)
            except (ClassifiedFailure, supervisor.Refused):
                pass
            finally:
                restore()
            assert state["post_send_loss"] is False
        elif mode in {"recover", "recover-pending"}:
            cli.poll_rpc("result = {'matrix': True}", timeout=300)
        else:
            cli.rpc("result = {'matrix': True}", timeout=30)
    except (ClassifiedFailure, supervisor.Refused):
        # The 503 case intentionally leaves the original request unresolved.
        pass

    if mode in {
        "clean",
        "pending",
        "recover",
        "recover-pending",
        "transfer",
        "flaky-transfer",
    }:
        source = tmp_path / "input.json"
        source.write_text('{"matrix": true}\n')
        transfers.upload_file(cli, source, "/content/matrix-input.json", renew=lambda: None)

    # A failed request leaves the supervisor fence set.  Clearing it here is a
    # test-only cleanup step so marker collection can be exercised independently;
    # it does not alter the failed call record or its checker classification.
    cli.exec_uncertain = False
    try:
        cli.collect_marker_census()
    except (ClassifiedFailure, supervisor.Refused):
        pass
    return _receipt(cli)


def _classes(result: dict[str, Any]) -> set[str]:
    return {row["class"] for row in result["counted"]}


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("clean", "PASS"),
        ("pending", "PASS"),
        ("503", "FAIL"),
        ("stale", "FAIL"),
        ("missing-markers", "EVIDENCE_MISSING"),
        ("corrupt-markers", "FAIL"),
        ("151s", "FAIL"),
        ("fifth-attempt", "FAIL"),
        ("changed-request-hash", "FAIL"),
        ("child-only-exit", "FAIL"),
    ],
)
def test_real_producer_to_checker_matrix(
    case: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if case == "pending":
        mode = "recover-pending"
    elif case == "503":
        mode = "503-injected"
    elif case in {"151s", "fifth-attempt"}:
        mode = "recover"
    elif case == "stale":
        mode = "transfer"
    else:
        mode = "clean"
    receipt = _produce(tmp_path, monkeypatch, mode)
    injection_manifest: dict[str, Any] | None = None
    if case == "503":
        failed = next(
            row
            for row in receipt["cli_calls"]
            if row.get("command") == "exec" and row.get("status") == "failed"
        )
        injection_manifest = {
            "schema": "l4-label-faults-injection-manifest/v1",
            "session_job_id": "matrix-job",
            "written_before_dispatch": True,
            "entries": [
                {
                    "injection_id": "matrix-response-loss",
                    "type": "post-send-response-loss",
                    "command": "exec",
                    "request_id": failed["request_id"],
                    "request_sha256": failed["request_sha256"],
                    "failure_type": "post_send_response_loss",
                    "max_calls": 1,
                }
            ],
        }
    elif case == "stale":
        receipt["transfers"][0]["sha256"] = "0" * 64
    elif case == "missing-markers":
        receipt["marker_census"] = None
    elif case == "corrupt-markers":
        marker = next(iter(receipt["marker_census"]["markers"].values()))
        marker["done_corrupt"] = True
    elif case == "151s":
        summary = next(row for row in receipt["cli_calls"] if row.get("command") == "recovery")
        summary.update(recovery_seconds=151.0, elapsed_seconds=151.0)
    elif case == "fifth-attempt":
        summary = next(row for row in receipt["cli_calls"] if row.get("command") == "recovery")
        summary["attempts"] = 5
    elif case == "changed-request-hash":
        finals = [
            row
            for row in receipt["cli_calls"]
            if row.get("command") == "exec" and row.get("reply_kind") == "final"
        ]
        finals[-1]["request_sha256"] = "f" * 64
    elif case == "child-only-exit":
        receipt.update(status="INCOMPLETE", failure_class="CHILD_NONZERO_EXIT")

    result = classify_session(receipt, injection_manifest)
    assert result["verdict"] == expected, result
    if case == "clean":
        consumed_calls = [
            row for row in receipt["cli_calls"] if row.get("transfer_role") == "consumed"
        ]
        assert consumed_calls
        assert all(row.get("parent_transfer_id") for row in consumed_calls)
        assert all(row.get("transfer_id") for row in consumed_calls)
        assert receipt["marker_census"]
        assert list((tmp_path / "remote").glob(".snake-rpc-*/*.started"))
        assert list((tmp_path / "remote").glob(".snake-rpc-*/*.done"))
    if case == "missing-markers":
        assert result["evidence_missing"]
        assert not result["counted"]
    if case == "corrupt-markers":
        assert "C4" in _classes(result)
    if case == "503":
        assert result["excluded_injections"] == []
    if case == "child-only-exit":
        assert "C3" not in _classes(result)


def test_optional_cpu_fault_closes_shared_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real checker attempt verdict aggregates a dispatched CPU fault."""
    l4 = classify_session(_produce(tmp_path / "l4", monkeypatch))
    cpu = _produce(tmp_path / "cpu", monkeypatch)
    cpu["cli_calls"][0]["elapsed_seconds"] = cpu["cli_calls"][0]["timeout_seconds"] + 3.0
    cpu_result = classify_session(cpu)

    jobs = [
        {"job_id": "cpu", "platform": "cpu", "mode": "loop", "manifest": "cpu.json"},
        {"job_id": "l4", "platform": "l4", "mode": "loop", "manifest": "l4.json"},
        {
            "job_id": "injection",
            "platform": "cpu",
            "mode": "injection",
            "manifest": "injection.json",
        },
    ]
    paths: dict[str, tuple[Path, Path, Path, Path]] = {}
    for job in jobs:
        output = tmp_path / "checker" / job["job_id"]
        output.mkdir(parents=True)
        manifest = output / job["manifest"]
        manifest.write_text("{}")
        paths[job["job_id"]] = (
            manifest,
            output,
            output / "receipt.json",
            output / "runtime.json",
        )

    plan = tmp_path / "plan.json"
    fake = tmp_path / "fake.json"
    fake.write_text("{}")
    plan.write_text(
        json.dumps(
            {
                "fake_proof": {"path": str(fake), "sha256": checker._sha(fake)},
                "source_commit": "0" * 40,
                "repo": str(tmp_path),
            }
        )
    )
    publication = tmp_path / "publication.json"
    publication.write_text(
        json.dumps(
            {
                "schema": checker.PUBLICATION_SCHEMA,
                "plan_sha256": checker._sha(plan),
                "ledgers": {job["job_id"]: {"evidence": {}} for job in jobs},
            }
        )
    )

    monkeypatch.setattr(checker, "_profile", lambda _plan: (json.loads(_plan.read_text()), jobs))
    monkeypatch.setattr(
        checker,
        "require_review",
        lambda **_kwargs: {"artifact_sha256": {"plan": checker._sha(plan)}},
    )
    monkeypatch.setattr(checker, "_verify_remote_publication", lambda *_args: None)
    monkeypatch.setattr(checker, "validate_fake_proof", lambda *_args: {})
    monkeypatch.setattr(checker, "_sealed_evidence", lambda *_args: {})
    monkeypatch.setattr(checker, "_ledger_inventory", lambda *_args: {})
    monkeypatch.setattr(checker, "_job_paths", lambda _plan, job: paths[job["job_id"]])

    def loop_result(_plan: Path, job: dict[str, Any]) -> dict[str, Any]:
        value = cpu_result if job["job_id"] == "cpu" else l4
        return {
            "session": job["job_id"],
            "platform": job["platform"],
            "status": value["verdict"],
            "faults": [{"source": "v2:" + row["class"], "record": row} for row in value["counted"]],
        }

    monkeypatch.setattr(checker, "_check_loop", loop_result)
    monkeypatch.setattr(
        checker,
        "_check_injection",
        lambda _plan, job: {
            "session": job["job_id"],
            "platform": "cpu",
            "status": "PASS",
            "faults": [],
            "no_duplicate": True,
        },
    )
    result = checker.verdict(plan, publication, tmp_path / "verdict.json")
    assert result["attempt_status"] == "FAIL"
    assert result["attempt_counted_faults"] >= 1
    assert any(fault["source"] == "v2:C1" for fault in result["attempt_faults"])


def test_real_contents_retry_retains_parent_child_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient chunk retry remains a verified C6 transfer, not a new RPC recovery."""
    receipt = _produce(tmp_path, monkeypatch, "flaky-transfer")
    result = classify_session(receipt)

    assert result["verdict"] == "PASS", result
    consumed = [call for call in receipt["cli_calls"] if call.get("transfer_role") == "consumed"]
    assert consumed
    assert all(call["parent_transfer_id"] for call in consumed)
    assert all(call["transfer_id"] for call in consumed)
    assert any(call["status"] == "failed" for call in consumed)
    assert any(call["status"] == "completed" for call in consumed)
    parent = receipt["transfers"][0]
    assert parent["status"] == "verified"
    assert parent["children"][0]["attempts"] == 2
    assert parent["children"][0]["verified_sha256"] == parent["verified_sha256"]
