"""Metadata emission and the single bounded session-close observation."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

from research.colab_cli import supervisor, transfers
from research.colab_cli.acceptance_faults import AcceptanceObserver


def test_account_retry_keeps_operation_and_parse_identity(tmp_path, monkeypatch):
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    outcomes = [supervisor.Refused("HTTP 503"), "No active sessions found on server."]

    def call(*args, **kwargs):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cli, "_call_once", call)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    assert cli.sessions() == {}
    failed, final = cli.call_records
    assert failed["operation_id"] == final["operation_id"]
    assert failed["argument_sha256"] == final["argument_sha256"]
    assert [r["attempt"] for r in cli.call_records] == [1, 2]
    assert final["parse_success"] is True
    assert len(final["response_sha256"]) == 64
    assert all(r["finished_monotonic"] >= r["started_monotonic"] for r in cli.call_records)


def test_census_exports_only_identity_and_observes_corruption(tmp_path):
    namespace = "a" * 32
    root = tmp_path / (".snake-rpc-" + namespace)
    root.mkdir()
    request_id, digest = "b" * 32, "c" * 64
    (root / (request_id + ".started")).write_text(
        json.dumps({"request_id": request_id, "sha256": digest})
    )
    (root / (request_id + ".done")).write_text(
        json.dumps({"request_id": request_id, "sha256": digest, "value": "PRIVATE_PAYLOAD"})
    )
    (root / ("d" * 32 + ".started")).write_text("corrupt")
    code = supervisor.marker_census_code(namespace, "CENSUS").replace(
        "/content/", str(tmp_path) + "/"
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<census>", "exec"), {})
    assert "PRIVATE_PAYLOAD" not in output.getvalue()
    rows = json.loads(output.getvalue().removeprefix("CENSUS"))
    assert rows[request_id] == {
        "started": 1,
        "done": 1,
        "started_request_id": request_id,
        "done_request_id": request_id,
        "started_sha256": digest,
        "done_sha256": digest,
    }
    assert rows["d" * 32]["started_corrupt"] is True


def test_census_single_read_and_no_busy_fence_change(tmp_path, monkeypatch):
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    commands = []

    def invoke(*args, **kwargs):
        commands.append((args, kwargs))
        source = Path(args[args.index("-f") + 1]).read_text()
        assert source.startswith("# SNAKE_MARKER_CENSUS")
        return "SNAKE_MARKER_CENSUS_" + hashlib.sha256(b"owned").hexdigest()[:32] + "{}"

    monkeypatch.setattr(cli, "_call_once", invoke)
    assert cli.collect_marker_census() == {}
    assert len(commands) == 1
    assert commands[0][1]["timeout"] <= 30
    assert cli.call_records[-1]["evidence_role"] == "marker_census"
    assert not cli.exec_uncertain
    assert cli.collect_marker_census() == {}
    assert len(commands) == 1
    cli = supervisor.CLI("fake", tmp_path / "busy-config", "busy")
    cli.exec_uncertain = True
    with pytest.raises(supervisor.Refused, match="fence"):
        cli.collect_marker_census()
    assert len(commands) == 1
    assert cli.exec_uncertain


def test_control_uploads_never_claim_consumed_verification():
    assert transfers.transfer_evidence("upload")["transfer_plane"] == "control"
    assert transfers.transfer_evidence("exec") == {}


@pytest.mark.parametrize(
    "error,expected",
    [
        ("HTTP 404 Not Found", True),
        ("404 Client Error: Not Found for url: /content/404/client-start.json", True),
        ("HTTP 503 for /content/404/client-start.json", False),
        ("HTTP 503 Service Unavailable; earlier not found", False),
    ],
)
def test_absence_evidence_uses_observed_status_not_path(tmp_path, error, expected):
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    remote = "/content/404/client-start.json"
    cli.call_records.append(
        {"command": "download", "remote": remote, "status": "failed", "error": error}
    )
    AcceptanceObserver._tag_observed_absence(cli, remote)
    assert (cli.call_records[0].get("failure_type") == "NOT_FOUND") is expected


def test_reconnect_operation_ids_do_not_collide(tmp_path, monkeypatch):
    first = supervisor.CLI("fake", tmp_path / "config", "owned")
    second = supervisor.CLI("fake", tmp_path / "config", "owned")
    for cli in (first, second):
        monkeypatch.setattr(cli, "_call_once", lambda *args, **kwargs: "ok")
        cli.call("version")
    assert first.call_records[0]["operation_id"] != second.call_records[0]["operation_id"]
    assert first.call_records[0]["attempt_id"] != second.call_records[0]["attempt_id"]
