"""l4-label-faults/v2 (JUDGMENT_L4_FAULT_DEFINITION_20261024 §2.4, §4 step 1).

Mandatory classifier sanity: attempt 1's L4 arm stays FAIL (call 117 C2, the
session end C3, call 58 benign-recovered); its CPU arm is PASS with its
deliberate faults excluded only by manifest identity; synthetic C1-C6 cases;
and fake-CLI tests of the one recovery-path fix.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.bl_labels import l4_faults_v2 as v2
from research.colab_cli import cli_guard, failures, injection_manifest, policy, supervisor

FIXTURES = Path(__file__).parent / "fixtures"
SANITY = json.loads((FIXTURES / "l4_faults_v2_acceptance_1792762502.json").read_text())
FROZEN = json.loads((FIXTURES / "colab_control_faults_acceptance_1792762502.json").read_text())
L4_SOURCE_SHA = "78fae4b656429f034c1a95d41bb6a9bc8614acd963b3230f5044169bb31ee20b"
CPU_SOURCE_SHA = "c4528d37a249bd0688d778f5a464c9678d4edbc08515857fb74f56749325e26d"


def _ok(**extra: Any) -> dict[str, Any]:
    return {"status": "completed", "elapsed_seconds": 1.0, "timeout_seconds": 30.0, **extra}


def _receipt(calls: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "status": "PASS",
        "failure_class": None,
        "error": None,
        "stop_verified": True,
        "cleanup_error": None,
        "cli_calls": calls,
        "contents_transfers": [],
        "transfers": [],
        **extra,
    }


def _classes(result: dict[str, Any]) -> list[tuple[str, int | None]]:
    return [(row["class"], row.get("index")) for row in result["counted"]]


# --- §2.4 mandatory sanity -------------------------------------------------


def test_fixture_is_bound_to_the_attempt_one_receipts() -> None:
    assert SANITY["arms"]["l4"]["source_receipt_sha256"] == L4_SOURCE_SHA
    assert SANITY["arms"]["cpu"]["source_receipt_sha256"] == CPU_SOURCE_SHA
    assert FROZEN["source_receipt_sha256"] == L4_SOURCE_SHA
    assert SANITY["arms"]["l4"]["cli_calls"] == [
        {k: v for k, v in row.items()} for row in SANITY["arms"]["l4"]["cli_calls"]
    ]
    assert len(SANITY["arms"]["l4"]["cli_calls"]) == len(FROZEN["cli_calls"]) == 122


def test_attempt_one_l4_still_fails_call_117_c2_end_c3_call_58_recovered() -> None:
    result = v2.classify_session(SANITY["arms"]["l4"])
    assert result["verdict"] == "FAIL"
    assert ("C2", 117) in _classes(result)
    assert any(row["class"] == "C3" for row in result["counted"])
    assert result["counted_by_class"] == {"C1": 0, "C2": 1, "C3": 1, "C4": 0, "C5": 0, "C6": 0}
    recovered = {row["first_failure_index"]: row for row in result["recovered"]}
    assert recovered[58]["resolved_index"] == 59 and recovered[58]["attempts"] == 1
    assert result["carve_outs"] == [{"index": 7, "rule": "pre-runtime heartbeat PUT"}]


def test_frozen_projection_named_by_the_judgment_also_fails() -> None:
    """The judgment names the older projection; it lacks transfer rows, so it is stricter."""
    result = v2.classify_session({**FROZEN["receipt"], "cli_calls": FROZEN["cli_calls"]})
    assert result["verdict"] == "FAIL"
    assert ("C2", 117) in _classes(result)
    assert any(row["class"] == "C3" for row in result["counted"])
    assert 58 in {row["first_failure_index"] for row in result["recovered"]}


def _retroactive_cpu_manifest(receipt: dict[str, Any]) -> dict[str, Any]:
    """Identity of attempt 1's deliberate CPU faults from the reviewed harness.

    Attempt 1 predates manifests; this sanity-only fixture reuses the harness's
    planned entries and binds the silence exec to its realized request ID.
    """
    entries = copy.deepcopy(injection_manifest.acceptance_entries("cpu"))
    silence = [
        c for c in receipt["cli_calls"] if c.get("request_sha256") == entries[0]["request_sha256"]
    ]
    entries[0]["request_id"] = silence[0]["request_id"]
    return {
        "schema": v2.INJECTION_SCHEMA,
        "session_job_id": receipt["job_id"],
        "written_before_dispatch": False,
        "provenance": "retroactive-sanity-fixture",
        "entries": entries,
    }


def test_attempt_one_cpu_passes_with_injections_excluded_by_identity() -> None:
    receipt = SANITY["arms"]["cpu"]
    manifest = _retroactive_cpu_manifest(receipt)
    result = v2.classify_session(receipt, manifest, allow_retroactive_manifest=True)
    assert result["verdict"] == "PASS", result["counted"]
    excluded = {row["injection_id"] for row in result["excluded_injections"]}
    assert excluded == {
        "client-loss-start-absent",
        "client-loss-finished-absent",
        "corrupt-export-rejection",
    } | ({"silence-timeout"} & excluded)
    assert result["carve_outs"] == [{"index": 7, "rule": "pre-runtime heartbeat PUT"}]


def test_cpu_without_manifest_fails_injections_are_never_inferred() -> None:
    result = v2.classify_session(SANITY["arms"]["cpu"])
    assert result["verdict"] == "FAIL"
    assert result["counted_by_class"]["C6"] == 3


def test_retroactive_manifest_refused_outside_the_sanity_fixture() -> None:
    receipt = SANITY["arms"]["cpu"]
    with pytest.raises(ValueError, match="pre-dispatch"):
        v2.classify_session(receipt, _retroactive_cpu_manifest(receipt))


# --- synthetic C1-C6 ---------------------------------------------------------


def test_c1_bound_failure_and_missing_bound() -> None:
    result = v2.classify_session(
        _receipt(
            [
                _ok(command="sessions", elapsed_seconds=32.5, timeout_seconds=30.0),
                {"command": "usage", "status": "completed", "elapsed_seconds": 1.0},
                _ok(command="usage", elapsed_seconds=31.9, timeout_seconds=30.0),
            ]
        )
    )
    assert _classes(result) == [("C1", 0), ("C1", 1)]


def _exec(status: str, rid: str | None = "r1", sha: str = "h", elapsed: float = 30.0) -> dict:
    row = {"command": "exec", "status": status, "elapsed_seconds": elapsed, "timeout_seconds": 30.0}
    if rid is not None:
        row.update(request_id=rid, request_sha256=sha)
    if status == "failed":
        row["failure_class"] = "CONTROL_RPC_TIMEOUT"
    return row


def test_c2_recovered_within_limits_is_recorded_not_counted() -> None:
    result = v2.classify_session(
        _receipt([_exec("failed"), _exec("failed"), _exec("completed", elapsed=2)])
    )
    assert result["verdict"] == "PASS"
    assert result["recovered_events"] == 1 and result["max_attempts"] == 2


@pytest.mark.parametrize(
    "calls",
    [
        [_exec("failed")],  # never resolved
        [_exec("failed", rid=None)],  # no durable identity
        [_exec("failed")] * 5 + [_exec("completed")],  # 5 attempts > 4
        [_exec("failed")]
        + [_ok(command="usage", elapsed_seconds=29.0)] * 5
        + [_exec("completed")],  # 145 + 30 s > 150 s
    ],
)
def test_c2_unrecovered_requests(calls: list[dict[str, Any]]) -> None:
    result = v2.classify_session(_receipt(calls))
    assert [row["class"] for row in result["counted"]] == ["C2"]


def test_c2_recovery_record_is_authoritative() -> None:
    summary = _ok(
        command="recovery",
        request_id="r1",
        request_sha256="h",
        attempts=4,
        recovery_seconds=150.5,
        elapsed_seconds=150.5,
        timeout_seconds=150.0,
    )
    result = v2.classify_session(_receipt([_exec("failed"), _exec("completed"), summary]))
    assert [row["class"] for row in result["counted"]] == ["C2"]


def test_c2_reads_resolve_by_later_completed_read() -> None:
    failed = {**_ok(command="sessions"), "status": "failed"}
    assert v2.classify_session(_receipt([failed, _ok(command="sessions")]))["verdict"] == "PASS"
    assert _classes(v2.classify_session(_receipt([failed])))[0] == ("C2", 0)


@pytest.mark.parametrize(
    "terminal",
    [
        {"status": "INCOMPLETE", "failure_class": "UPLOAD_CAP"},
        {"stop_verified": False},
        {"cleanup_error": "release unverified"},
        {"status": "INCOMPLETE", "failure_class": "CHILD_NONZERO_EXIT"},
    ],
)
def test_c3_non_success_terminals(terminal: dict[str, Any]) -> None:
    result = v2.classify_session(_receipt([_ok(command="version")], **terminal))
    assert [row["class"] for row in result["counted"]] == ["C3"]


def _manifest(entries: list[dict[str, Any]], job: str = "job-1") -> dict[str, Any]:
    return {
        "schema": v2.INJECTION_SCHEMA,
        "session_job_id": job,
        "written_before_dispatch": True,
        "entries": entries,
    }


def test_designed_watchdog_release_excludes_c3_only_with_verified_stop() -> None:
    entry = injection_manifest.acceptance_entries("L4")[2]
    base = dict(
        status="INCOMPLETE",
        failure_class="ACCOUNTING",
        error="AccountingError('supervised assignment disappeared from sessions')",
    )
    clean = v2.classify_session(_receipt([_ok(command="version")], **base), _manifest([entry]))
    assert clean["verdict"] == "PASS"
    dirty = v2.classify_session(
        _receipt([_ok(command="version")], **base, stop_verified=False), _manifest([entry])
    )
    assert [row["class"] for row in dirty["counted"]] == ["C3"]
    assert v2.classify_session(_receipt([_ok(command="version")], **base))["verdict"] == "FAIL"


def test_c4_code_hash_differs_and_marker_counts() -> None:
    result = v2.classify_session(_receipt([_exec("failed", sha="a"), _exec("completed", sha="b")]))
    assert "C4" in [row["class"] for row in result["counted"]]
    for markers in (
        {"r": {"started": 2, "done": 1}},
        {"r": {"started": 0, "done": 1}},
        {"r": {"started": 1, "done": 2}},
    ):
        result = v2.classify_session(_receipt([_ok(command="version")]), markers=markers)
        assert [row["class"] for row in result["counted"]] == ["C4"]
    ok = v2.classify_session(
        _receipt([_ok(command="version")]), markers={"r": {"started": 1, "done": 1}}
    )
    assert ok["verdict"] == "PASS"


def test_c5_unit_integrity() -> None:
    good = {
        "unit_id": "u1",
        "runtime_sha256": "a",
        "mac_sha256": "a",
        "complete": True,
        "sha256_sidecar_ok": True,
    }
    cases = [
        {"committed": [good, good]},
        {"committed": [{**good, "mac_sha256": "b"}]},
        {"committed": [{**good, "complete": False}]},
        {"committed": [good], "allocated": ["u1", "u2"], "pending": []},
        {"committed": [good], "integrity": {"bare_audit_mismatches": 1}},
        {"committed": [good], "integrity": {"worker_invariance": False}},
    ]
    for units in cases:
        result = v2.classify_session(_receipt([_ok(command="version")]), units=units)
        assert result["verdict"] == "FAIL" and {r["class"] for r in result["counted"]} == {"C5"}
    closed = {"committed": [good], "allocated": ["u1", "u2"], "pending": ["u2"]}
    assert (
        v2.classify_session(_receipt([_ok(command="version")]), units=closed)["verdict"] == "PASS"
    )


def _upload(status: str, remote: str) -> dict[str, Any]:
    return {
        "command": "upload",
        "status": status,
        "elapsed_seconds": 1.0,
        "timeout_seconds": 60.0,
        "remote": remote,
    }


def test_c6_transfers_and_the_single_heartbeat_carveout() -> None:
    hb = "/content/snake-cli/g/lease-heartbeat.json"
    release = "/content/snake-cli-g-release.json"
    calls = [
        _upload("completed", release),
        _upload("failed", hb),
        _upload("completed", hb),
        _upload("completed", "/content/snake-cli/g/bootstrap.sh"),
        _upload("failed", hb),
    ]
    result = v2.classify_session(_receipt(calls))
    assert result["carve_outs"] == [{"index": 1, "rule": "pre-runtime heartbeat PUT"}]
    assert _classes(result) == [("C6", 4)]  # a later heartbeat failure is not carved out
    staged = "/content/snake-cli/g/inputs/a.json"
    recovered = v2.classify_session(
        _receipt(
            [_upload("failed", staged), _upload("completed", staged)],
            transfers=[{"direction": "upload", "remote": staged, "status": "verified"}],
        )
    )
    assert recovered["verdict"] == "PASS"
    unverified = v2.classify_session(
        _receipt(
            [_ok(command="version")],
            transfers=[{"direction": "download", "remote": "/x/y.bin", "status": "failed"}],
        )
    )
    assert [row["class"] for row in unverified["counted"]] == ["C6"]


def test_injection_identity_is_exact() -> None:
    entry = {
        "injection_id": "drop-1",
        "type": "post_send_response_loss",
        "command": "exec",
        "request_id": "planned",
        "max_calls": 1,
    }
    tagged = {**_exec("failed", rid="other"), "injected": True}
    result = v2.classify_session(_receipt([tagged]), _manifest([entry]))
    assert [row["class"] for row in result["counted"]] == ["C2"]  # a tag is not identity
    assert result["unrealized_injections"] == ["drop-1"]
    planned = v2.classify_session(
        _receipt([_exec("failed", rid="planned")] * 2), _manifest([entry])
    )
    assert [row["class"] for row in planned["counted"]] == ["C2"]  # max_calls bounds absorption
    with pytest.raises(ValueError, match="different session"):
        v2.classify_session(_receipt([_ok(command="version")]), _manifest([entry], job="other"))
    with pytest.raises(ValueError, match="request ID"):
        v2.validate_injection_manifest(_manifest([{**entry, "request_id": None}]))


def test_planned_acceptance_manifest_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "cpu.injection.json"
    digest = injection_manifest.write(
        path, "job-1", injection_manifest.acceptance_entries("cpu"), purpose="test"
    )
    loaded = v2.load_injection_manifest(path, job_id="job-1")
    assert injection_manifest.load(path, digest) == loaded
    with pytest.raises(FileExistsError):
        injection_manifest.write(path, "job-1", [], purpose="test")


# --- the one recovery-path fix (fake CLI) -----------------------------------


def _kernel(code: str, tmp_path: Path) -> str:
    code = code.replace("/content/.snake-rpc-", str(tmp_path / ".snake-rpc-"))
    code = code.replace("threading.Thread(target=worker, daemon=True).start()", "worker()")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<fake-kernel>", "exec"), {})
    return output.getvalue()


def _fake_clock(monkeypatch: pytest.MonkeyPatch, delay: float = 2.0) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda d: now.__setitem__(0, now[0] + d))
    monkeypatch.setattr(policy, "poll_delay", lambda _attempt: delay)
    return now


def _done_receipt(cli: supervisor.CLI) -> dict[str, Any]:
    return _receipt(copy.deepcopy(cli.call_records))


def test_double_tail_event_recovers_once_under_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = _fake_clock(monkeypatch)
    target = tmp_path / "mutation"
    operation = (
        "from pathlib import Path\n"
        f"with Path({str(target)!r}).open('x') as handle: handle.write('once')\n"
        "result={'committed':True}"
    )
    requests: list[str] = []

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert 0 < kwargs["timeout"] <= 30
        code = Path(argv[argv.index("-f") + 1]).read_text()
        requests.append(re.search(r"# SNAKE_REQUEST_ID (\w+)", code).group(1))
        output = _kernel(code, tmp_path)  # the kernel always runs the claim logic
        if len(requests) <= 2:  # two consecutive lost responses (tail events)
            now[0] += kwargs["timeout"]
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *a, **k: True)
    assert cli.recoverable_rpc(operation) == {"committed": True}
    assert len(set(requests)) == 1 and len(requests) == 3
    assert target.read_text() == "once"  # the re-issue never re-executed
    assert len(list(tmp_path.glob(".snake-rpc-*/*.started"))) == 1
    assert len(list(tmp_path.glob(".snake-rpc-*/*.done"))) == 1
    summary = cli.call_records[-1]
    assert summary["command"] == "recovery" and summary["status"] == "completed"
    assert summary["attempts"] == 2 and summary["recovery_seconds"] <= policy.RECOVERY_SECONDS
    result = v2.classify_session(_done_receipt(cli))
    assert result["verdict"] == "PASS"
    assert result["recovered_events"] == 1 and result["max_attempts"] == 2
    # Both original failures stay recorded.
    assert sum(r["status"] == "failed" for r in cli.call_records if r["command"] == "exec") == 2


def test_exhaustion_at_four_attempts_is_c2_and_keeps_the_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = _fake_clock(monkeypatch, delay=1.0)
    execs = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        execs.append(kwargs["timeout"])
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    probes = []

    def busy(*args: Any, **kwargs: Any) -> bool:
        probes.append(kwargs["timeout"])
        now[0] += 5
        return False

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", busy)
    with pytest.raises(failures.ClassifiedFailure, match="busy/unknown") as raised:
        cli.poll_rpc("result=True")
    assert raised.value.failure_class is failures.FailureClass.CONTROL_RPC_TIMEOUT
    assert execs == [30] and len(probes) == policy.RECOVERY_ATTEMPTS
    assert cli.exec_uncertain
    assert cli.call_records[-1]["status"] == "failed" and cli.call_records[-1]["attempts"] == 4
    result = v2.classify_session(_done_receipt(cli))
    assert [row["class"] for row in result["counted"]] == ["C2"]


def test_exhaustion_at_the_150s_ceiling_never_overruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = _fake_clock(monkeypatch, delay=20.0)
    execs = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        execs.append(kwargs["timeout"])
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *a, **k: True)
    with pytest.raises(failures.ClassifiedFailure, match="deadline reached"):
        cli.poll_rpc("result=True")  # default 300 s operation budget: the 150 s ceiling binds
    assert now[0] == pytest.approx(30 + policy.RECOVERY_SECONDS)
    assert all(timeout <= 30 for timeout in execs)
    summary = cli.call_records[-1]
    assert summary["recovery_seconds"] == pytest.approx(policy.RECOVERY_SECONDS)
    assert v2.classify_session(_done_receipt(cli))["counted_by_class"]["C2"] == 1


def test_stalled_idle_helper_retries_inside_ceiling_then_c2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call 117's death mode: each idle GET exhausts its own 30 s bound."""
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = _fake_clock(monkeypatch)

    def invoke(argv: list[str], **kwargs: Any) -> None:
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    helper = []

    def stalled(*_args: Any, timeout: float) -> dict[str, Any]:
        helper.append(timeout)
        now[0] += timeout
        raise cli_guard.GuardError("guard helper failed: TimeoutExpired")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli_guard, "_run_helper", stalled)
    with pytest.raises(failures.ClassifiedFailure):
        cli.poll_rpc("result=True")
    probes = [r for r in cli.call_records if r["command"] == "kernel_idle"]
    assert 1 < len(probes) <= policy.RECOVERY_ATTEMPTS
    assert all(
        r["status"] == "failed" and r["elapsed_seconds"] <= r["timeout_seconds"] for r in probes
    )
    assert now[0] <= 30 + policy.RECOVERY_SECONDS
    assert cli.exec_uncertain


def test_non_retryable_probe_error_fails_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = _fake_clock(monkeypatch)

    def invoke(argv: list[str], **kwargs: Any) -> None:
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def refused(*args: Any, **kwargs: Any) -> bool:
        raise cli_guard.GuardError("saved OAuth token is required; interactive auth is forbidden")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", refused)
    with pytest.raises(cli_guard.GuardError, match="OAuth"):
        cli.poll_rpc("result=True")
    assert cli.call_records[-1]["command"] == "recovery"
    assert cli.call_records[-1]["status"] == "failed" and cli.call_records[-1]["attempts"] == 1
    assert cli.exec_uncertain
