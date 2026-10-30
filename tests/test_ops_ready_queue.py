"""Offline scheduling and mechanical-retry behavior; no real provider calls."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ops import auto_retry as ar
from tools.ops import ready_queue as rq
from tools.ops import seal_doom_ready


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def packet(tmp_path):
    manifest = tmp_path / "packet/manifest.json"
    value = {
        "job_id": "job0",
        "commit": "a" * 40,
        "script": "research/example/job.sh",
        "output_dir": str(tmp_path / "output"),
        "inputs": [],
        "timeout_minutes": 30,
        "env": {"UNCHANGED": "1"},
        "resume": {"max_outcome_blind_retries": 2},
        "technical_envelope": {"series_max_original_plus_completions": 3},
    }
    write(manifest, value)
    return manifest


def finish(manifest, log, **changes):
    value = ar.read(manifest)
    output = Path(value["output_dir"])
    executed = write(output / "manifest.json", value)
    receipt = {
        "job_id": value["job_id"],
        "commit": value["commit"],
        "manifest_sha256": ar.digest(executed),
        "stop_verified": True,
        "status": "INCOMPLETE",
        "exports": [],
        "cleanup_error": None,
        "terminal": None,
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "error": "Refused('kernel remains busy/unknown after telemetry hiccup')",
    }
    receipt.update(changes)
    write(output / "receipt.json", receipt)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("EXIT=1\n")
    return receipt


@pytest.mark.parametrize(
    "changes",
    [
        {"stop_verified": False},
        {"cleanup_error": "unknown"},
        {"exports": [{"path": "unit"}]},
        {"terminal": {"returncode": 1}},
        {"status": "PASS"},
        {"failure_class": "CHILD_NONZERO_EXIT"},
        {"failure_class": "ACCOUNTING"},
        {"error": "pytest timeout"},
        {"error": "transport timeout after numerical overflow"},
        {"error": "kernel transport timeout after CUDA out of memory"},
        {"error": "generic TimeoutExpired"},
        {"manifest_sha256": "0" * 64},
        {"commit": "b" * 40},
        {"job_id": "foreign"},
    ],
)
def test_retry_refuses_outputs_science_unknown_and_unbound(packet, tmp_path, changes):
    log = tmp_path / "job.log"
    finish(packet, log, **changes)
    with pytest.raises(ValueError):
        ar.prepare(packet, log, tmp_path / "retry", tmp_path)
    assert not (tmp_path / "retry").exists()


@pytest.mark.parametrize(
    "category,error",
    [
        ("RPC_EXEC_POLL_TIMEOUT", "kernel telemetry transport timeout"),
        ("TRANSPORT_TIMEOUT", "transport timeout"),
        ("UPLOAD_CAP", "total upload time cap"),
        ("ALLOCATION_REFUSED", "allocation refused"),
        ("ALLOCATION_UNAVAILABLE", "assignment unavailable 503"),
    ],
)
def test_retry_allowlist_preserves_inputs_caps_and_limit(packet, tmp_path, category, error):
    log = tmp_path / "job.log"
    finish(packet, log, failure_class=category, error=error)
    original = packet.read_bytes()
    current = packet
    for index in (1, 2):
        current = ar.prepare(current, log, tmp_path / f"retry-{index}", tmp_path)
        value = ar.read(current)
        assert value["completion_attempt"] == index
        assert value["inputs"] == [] and value["env"] == {"UNCHANGED": "1"}
        assert value["timeout_minutes"] == 30
        assert ar.read(current.parent / "RETRY.json")["completion_attempt"] == index
        finish(current, log)
    with pytest.raises(ValueError, match="limit exhausted"):
        ar.prepare(current, log, tmp_path / "retry-3", tmp_path)
    assert packet.read_bytes() == original


def test_retry_checks_actual_export_tree_and_shell_exit(packet, tmp_path):
    log = tmp_path / "job.log"
    finish(packet, log)
    log.write_text("not exited\n")
    with pytest.raises(ValueError):
        ar.eligible(ar.read(packet), Path(ar.read(packet)["output_dir"]), log)
    log.write_text("EXIT=1\n")
    write(Path(ar.read(packet)["output_dir"]) / "exports/partial", {})
    with pytest.raises(ValueError):
        ar.prepare(packet, log, tmp_path / "retry", tmp_path)


def command(tmp_path, manifest, name):
    return [
        str(tmp_path / "dispatch.sh"),
        name,
        str(manifest),
        "--broker",
        str(tmp_path),
        "--cwd",
        str(tmp_path),
    ]


def add_jobs(tmp_path, count):
    ops = tmp_path / "ops"
    for index in range(count):
        path = write(
            tmp_path / f"job{index}.json",
            {
                "job_id": f"job{index}",
                "commit": "a" * 40,
                "output_dir": str(tmp_path / f"output{index}"),
            },
        )
        rq.enqueue(ops, f"q{index}", command(tmp_path, path, f"job{index}"))
    return ops


def test_capacity_dry_run_and_duplicate_claims(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 4)
    before = (ops / "ready-queue.json").read_bytes()
    assert rq.tick(ops, dry_run=True)["would_dispatch"] == ["q0", "q1", "q2"]
    assert (ops / "ready-queue.json").read_bytes() == before
    calls = []
    monkeypatch.setattr(rq.subprocess, "Popen", lambda *a, **k: calls.append(a))
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: rq.tick(ops), range(2)))
    assert sum(len(result["dispatched"]) for result in results) == 3
    assert len(calls) == 3
    assert rq.tick(ops)["dispatched"] == []
    with pytest.raises(ValueError, match="already enqueued"):
        rq.enqueue(ops, "q0", command(tmp_path, tmp_path / "job0.json", "job0"))


def test_readiness_skips_blocked_head_and_uses_final_retry(tmp_path):
    ops = add_jobs(tmp_path, 3)
    with rq.locked(ops) as queue:
        queue["jobs"][0]["status"] = "LAUNCHED"
        queue["jobs"][1]["after"] = ["q0:pass"]
    assert rq.tick(ops, dry_run=True)["would_dispatch"] == ["q2"]
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="FAILED", retry_id="q2")
        assert not rq.ready(queue, queue["jobs"][1])
        queue["jobs"][2]["status"] = "PASS"
        assert rq.ready(queue, queue["jobs"][1])


def test_observer_never_frees_unverified_stop(packet, tmp_path):
    ops = tmp_path / "ops"
    manifest = write(ops / "manifests/job0.json", ar.read(packet))
    log = ops / "logs/job0.log"
    finish(manifest, log, stop_verified=False)
    assert rq.observations(ops)["job0"]["active"]
    finish(manifest, log)
    assert not rq.observations(ops)["job0"]["active"]


def test_integrated_tracking_retry_and_limit(packet, tmp_path, monkeypatch):
    ops = tmp_path / "ops"
    snapshot = write(ops / "manifests/job0.json", ar.read(packet))
    log = ops / "logs/job0.log"
    finish(snapshot, log)
    rq.enqueue(ops, "original", command(tmp_path, packet, "job0"), track=True, retry_limit=1)
    monkeypatch.setattr(rq.subprocess, "Popen", lambda *a, **k: None)
    rq.tick(ops)
    queue = ar.read(ops / "ready-queue.json")
    parent, retry = queue["jobs"]
    assert parent["retry_id"] == retry["id"] and retry["status"] == "DISPATCHING"
    assert "AUTO_RETRY_QUEUED" in (ops / "events.log").read_text()
    assert len(queue["jobs"]) == 2
    rq.tick(ops)
    assert len(ar.read(ops / "ready-queue.json")["jobs"]) == 2


def test_real_worker_claim_and_dispatch_smoke(tmp_path):
    ops = add_jobs(tmp_path, 1)
    script = tmp_path / "dispatch.sh"
    script.write_text("#!/bin/sh\n" 'printf called >> "$OPS_DIR/calls"\n' "exit 0\n")
    script.chmod(0o755)
    result = rq.tick(ops)
    assert result["dispatched"] == ["q0"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        queue = ar.read(ops / "ready-queue.json")
        if queue["jobs"][0]["status"] == "LAUNCHED":
            break
        time.sleep(0.02)
    assert queue["jobs"][0]["status"] == "LAUNCHED"
    assert (ops / "calls").read_text() == "called"
    row = queue["jobs"][0]
    with pytest.raises(ValueError, match="claim"):
        rq.worker(ops, row["id"], row["token"])


def test_prepare_gate_failure_and_changed_manifest_hold_without_submission(tmp_path):
    ops = add_jobs(tmp_path, 1)
    with rq.locked(ops) as queue:
        row = queue["jobs"][0]
        row.update(status="DISPATCHING", token="claim")
    write(tmp_path / "job0.json", {"tampered": True})
    rq.worker(ops, "q0", "claim")
    assert ar.read(ops / "ready-queue.json")["jobs"][0]["status"] == "HELD"
    assert not (ops / "calls").exists()


def test_refuse_shell_commands_and_explicit_canary_limit(packet, tmp_path):
    with pytest.raises(ValueError, match="dispatch.sh"):
        rq.dispatch_parts(["bash", "-c", "arbitrary"])
    value = ar.read(packet)
    value.pop("resume")
    with pytest.raises(ValueError, match="explicit"):
        ar.completion_limit(value)
    assert ar.completion_limit(value, 2) == (0, 2)


def test_live_inventory_counts_unmanaged_jobs_and_does_not_trust_old_receipts(packet, tmp_path):
    ops = add_jobs(tmp_path, 3)
    historical = ar.read(packet)
    snapshot = write(ops / "manifests/historical.json", historical)
    finish(snapshot, ops / "logs/historical.log", stop_verified=False)
    assert rq.tick(ops, dry_run=True, inventory={"external1", "external2"})["would_dispatch"] == [
        "q0"
    ]
    assert rq.tick(ops, dry_run=True, inventory={"a", "b", "c"})["would_dispatch"] == []
    assert rq.observations(ops)["historical"]["active"]  # remains ineligible for retry


def test_provider_probe_refuses_unknown_inventory_and_counts_all_assignments(monkeypatch):
    monkeypatch.setattr(
        rq.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout="[colab] update notice\n\n[?] endpoint1 | Hardware: L4 | Shape: Standard\n"
        ),
    )
    assert rq.provider_sessions() == {"endpoint1"}
    monkeypatch.setattr(rq.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="unknown"))
    with pytest.raises(ValueError, match="unrecognized"):
        rq.provider_sessions()


def test_package_helper_is_used_before_preparation(packet, tmp_path, monkeypatch):
    value = ar.read(packet)
    value["script"] = "research/doom_auc_smoke/job.sh"
    write(packet, value)
    log = tmp_path / "job.log"
    finish(packet, log)
    calls = []
    monkeypatch.setattr(ar.subprocess, "run", lambda argv, **kw: calls.append(argv))
    ar.prepare(packet, log, tmp_path / "retry", tmp_path)
    assert len(calls) == 1 and "build_ledger" in calls[0][2]


def test_original_completion_one_has_only_one_remaining_attempt(packet, tmp_path):
    value = ar.read(packet)
    value["completion_attempt"] = 1
    write(packet, value)
    log = tmp_path / "job.log"
    finish(packet, log)
    successor = ar.prepare(packet, log, tmp_path / "retry", tmp_path)
    assert ar.read(successor)["completion_attempt"] == 2
    finish(successor, log)
    with pytest.raises(ValueError, match="limit exhausted"):
        ar.prepare(successor, log, tmp_path / "retry-3", tmp_path)


def test_real_queue_automatically_dispatches_exactly_two_completions(packet, tmp_path):
    script = tmp_path / "dispatch.sh"
    script.write_text(f"#!{sys.executable}\n" + """
import hashlib,json,os,sys
from pathlib import Path
name=sys.argv[1]; source=Path(sys.argv[2]); m=json.loads(source.read_text())
ops=Path(os.environ['OPS_DIR']); out=Path(m['output_dir']); out.mkdir(parents=True)
for sub in ('logs','manifests','dispatch'):
    (ops/sub).mkdir(exist_ok=True)
(ops/'dispatch'/name).mkdir()
payload=source.read_bytes()
(ops/'manifests'/(name+'.json')).write_bytes(payload)
(out/'manifest.json').write_bytes(payload)
r=dict(job_id=m['job_id'],commit=m['commit'],manifest_sha256=hashlib.sha256(payload).hexdigest(),
       stop_verified=True,status='INCOMPLETE',exports=[],cleanup_error=None,terminal=None,
       failure_class='RPC_EXEC_POLL_TIMEOUT',error='kernel transport timeout')
(out/'receipt.json').write_text(json.dumps(r))
(ops/'logs'/(name+'.log')).write_text('EXIT=1\\n')
with (ops/'calls').open('a') as stream:
    stream.write(name+'\\n')
""")
    script.chmod(0o755)
    ops = tmp_path / "ops"
    rq.enqueue(ops, "original", command(tmp_path, packet, "job0"), retry_limit=2)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        rq.tick(ops, inventory=set())
        queue = ar.read(ops / "ready-queue.json")
        if len(queue["jobs"]) == 3 and all(r["status"] == "FAILED" for r in queue["jobs"]):
            break
        time.sleep(0.04)
    assert len(queue["jobs"]) == 3
    assert all(r["status"] == "FAILED" for r in queue["jobs"])
    assert len((ops / "calls").read_text().splitlines()) == 3
    assert (ops / "events.log").read_text().count('"event": "AUTO_RETRY_QUEUED"') == 2


def test_doom_seal_follows_canary_retries_and_refuses_unfinished(tmp_path):
    ops = add_jobs(tmp_path, 2)
    with pytest.raises(ValueError, match="admissible"):
        seal_doom_ready.receipt(ops, "q0")
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="FAILED", retry_id="q1")
        queue["jobs"][1]["status"] = "PASS"
    assert seal_doom_ready.receipt(ops, "q0") == tmp_path / "output1/receipt.json"
    with rq.locked(ops) as queue:
        queue["jobs"][1]["status"] = "UNAVAILABLE"
    assert seal_doom_ready.receipt(ops, "q0", h100=True) == tmp_path / "output1/receipt.json"
    with pytest.raises(ValueError):
        seal_doom_ready.receipt(ops, "q0")


def test_live_assignment_is_not_double_counted_before_terminal_receipt(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 3)
    with rq.locked(ops) as queue:
        queue["jobs"][0]["status"] = "LAUNCHED"
    value = ar.read(tmp_path / "job0.json")
    write(ops / "manifests/job0.json", value)
    log = ops / "logs/job0.log"
    log.parent.mkdir(parents=True)
    log.write_text("running\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write(
        tmp_path / "home/.config/colab-cli/snake/active-job0.json",
        {"output": value["output_dir"], "endpoint": "endpoint0"},
    )
    result = rq.tick(ops, dry_run=True, inventory={"endpoint0"})
    assert result == {"active": 1, "reserved": 0, "would_dispatch": ["q1", "q2"]}


def test_dispatch_fallback_and_accounting_retry_share_one_queue_slot():
    row = {"id": "q", "argv": ["/dispatch.sh", "job"], "status": "LAUNCHED"}
    for suffix in ("-a100-fallback", "-retry-1"):
        observed = {"job": {"active": False}, "job" + suffix: {"active": True, "endpoint": "live"}}
        assert rq.occupancy({"jobs": [row]}, observed, {"live"}) == (1, 0)
        assert rq.occupancy({"jobs": [row]}, observed, None) == (1, 0)
