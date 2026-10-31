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


@pytest.fixture(autouse=True)
def isolated_account(tmp_path, monkeypatch):
    """Offline queue tests never read the operator's live pending intents."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


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


@pytest.mark.parametrize(
    ("usage", "expected"),
    [("Current balance: 9\nActive assignments: 0\n", set()), ("Active assignments: 2\n", None)],
)
def test_provider_probe_accepts_silent_idle_only_with_zero_usage(monkeypatch, usage, expected):
    outputs = {"sessions": "[colab] update notice\n\n", "usage": usage}
    monkeypatch.setattr(
        rq.subprocess, "run", lambda argv, **k: SimpleNamespace(stdout=outputs[argv[-1]])
    )
    if expected is None:
        with pytest.raises(ValueError, match="empty account sessions inventory"):
            rq.provider_sessions()
    else:
        assert rq.provider_sessions() == expected


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


def test_reconciliation_failure_parks_head_with_backoff_and_advances_tail(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 2)
    with rq.locked(ops) as queue:
        queue["jobs"][0]["status"] = "LAUNCHED"
    state = {
        "active": True,
        "exited": True,
        "receipt": {"failure_class": "SUPERVISOR", "endpoint": None},
        "unavailable": False,
    }
    calls = []
    monkeypatch.setattr(rq, "observations", lambda _: {"job0": state})

    def refused(*args):
        calls.append(args)
        raise ValueError("account membership changed; fallback forbidden")

    monkeypatch.setattr(rq, "retire_unavailable", refused)
    monkeypatch.setattr(rq.subprocess, "Popen", lambda *a, **kw: None)
    result = rq.tick(ops, inventory=set())
    assert result["dispatched"] == ["q1"]
    head = ar.read(ops / "ready-queue.json")["jobs"][0]
    assert head["reconcile_failures"] == 1 and head["reconcile_retry_at"] > time.time()
    rq.tick(ops, inventory=set())
    assert len(calls) == 1  # no retry on each 15-second tick
    with rq.locked(ops) as queue:
        queue["jobs"][0]["reconcile_retry_at"] = 0
    rq.tick(ops, inventory=set())
    assert len(calls) == 2


def test_pending_fence_does_not_burn_other_ready_identities(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 2)
    write(Path.home() / ".config/colab-cli/snake/pending-foreign.json", {})
    monkeypatch.setattr(rq.subprocess, "Popen", lambda *a, **k: pytest.fail("must not dispatch"))
    result = rq.tick(ops, inventory=set())
    assert result["dispatched"] == [] and len(result["blocked_pending"]) == 1
    assert all(row["status"] == "READY" for row in ar.read(ops / "ready-queue.json")["jobs"])


def test_unallocated_retry_requires_live_bound_proof_and_backs_off(packet, tmp_path, monkeypatch):
    from tools.ops import dispatch

    ops = tmp_path / "ops"
    value = ar.read(packet)
    value["requirements"] = {"gpu": "H100", "prefer": ["H100", "A100"]}
    write(packet, value)
    snapshot = write(ops / "manifests/job0.json", value)
    log = ops / "logs/job0.log"
    receipt = finish(
        snapshot,
        log,
        stop_verified=False,
        failure_class="SUPERVISOR",
        cleanup_error="ALLOCATION_UNRESOLVED: retained pending record",
        endpoint=None,
        error="Refused('allocation unavailable')",
    )
    proof = write(
        ops / "dispatch/job0/unavailable.json",
        {
            "schema": "colab-allocation-unavailable/v1",
            "mode": "LIVE",
            "endpoint_ever_assigned": False,
            "vm_compute_started": False,
            "recovery": {"endpoint": None},
            "receipt": {
                "path": str(Path(value["output_dir"]) / "receipt.json"),
                "sha256": ar.digest(Path(value["output_dir"]) / "receipt.json"),
            },
        },
    )
    monkeypatch.setattr(dispatch, "allocation_unavailable", lambda *a: True)
    rq.enqueue(ops, "original", command(tmp_path, packet, "job0"), track=True, retry_limit=2)
    result = rq.tick(ops, inventory=set())
    assert result["dispatched"] == []
    parent, retry = ar.read(ops / "ready-queue.json")["jobs"]
    assert parent["status"] == "UNAVAILABLE" and parent["retry_id"] == retry["id"]
    assert retry["status"] == "READY" and retry["not_before"] > time.time()
    successor = ar.read(Path(retry["manifest"]))
    assert successor["completion_attempt"] == 1
    assert successor["job_id"] != value["job_id"] and successor["output_dir"] != value["output_dir"]
    assert successor["requirements"] == value["requirements"]
    write(proof, {**ar.read(proof), "mode": "DRY_RUN"})
    with pytest.raises(ValueError, match="live unallocated proof"):
        ar.eligible(value, Path(value["output_dir"]), log, proof)
    assert ar.read(Path(value["output_dir"]) / "receipt.json") == receipt


def test_queue_auto_reconciles_without_dispatch_when_intent_retires(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 1)
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="LAUNCHED", retry_limit=0)
    pending = write(Path.home() / ".config/colab-cli/snake/pending-own.json", {})
    state = {
        "active": True,
        "exited": True,
        "receipt": {"failure_class": "SUPERVISOR", "endpoint": None},
        "unavailable": False,
    }
    monkeypatch.setattr(rq, "observations", lambda _: {"job0": state})

    def retire(*args):
        pending.unlink()

    monkeypatch.setattr(rq, "retire_unavailable", retire)
    monkeypatch.setattr(rq, "no_assignment_proof", lambda *a: True)
    result = rq.tick(ops, inventory=set())
    assert result["dispatched"] == [] and not pending.exists()
    assert ar.read(ops / "ready-queue.json")["jobs"][0]["status"] == "UNAVAILABLE"


def pending_fence_receipt(manifest, log):
    value = ar.read(manifest)
    return finish(
        manifest,
        log,
        stop_verified=False,
        endpoint=None,
        session="snake-" + value["job_id"] + "-" + "a" * 12,
        failure_class="SUPERVISOR",
        accounting=None,
        usage_before=None,
        usage_after=None,
        cli_calls=[],
        control_operations=[],
        transfers=[],
        contents_transfers=[],
        error="Refused('unresolved allocation/stop: reconcile pending records first')",
    )


def test_queue_adopts_burned_pending_refusal_and_prepares_fresh_retry(tmp_path):
    ops = add_jobs(tmp_path, 1)
    manifest = write(ops / "manifests/job0.json", ar.read(tmp_path / "job0.json"))
    log = ops / "logs/job0.log"
    pending_fence_receipt(manifest, log)
    with rq.locked(ops) as queue:
        queue["jobs"][0]["retry_limit"] = 2
    result = rq.tick(ops, inventory=set())
    assert result["dispatched"] == []
    parent, retry = ar.read(ops / "ready-queue.json")["jobs"]
    assert parent["status"] == "FAILED" and parent["retry_id"] == retry["id"]
    assert retry["status"] == "READY" and retry["not_before"] > time.time()
    assert ar.read(Path(retry["manifest"]))["completion_attempt"] == 1
    assert "READY_ADOPTED" in (ops / "events.log").read_text()


@pytest.mark.parametrize("fault", ["new_call", "active", "pending", "error", "exports", "hash"])
def test_pending_fence_retry_refuses_ambiguous_or_allocated_receipts(packet, tmp_path, fault):
    log = tmp_path / "job.log"
    receipt = pending_fence_receipt(packet, log)
    value = ar.read(packet)
    output = Path(value["output_dir"])
    if fault in {"active", "pending"}:
        write(Path.home() / ".config/colab-cli/snake" / f"{fault}-{receipt['session']}.json", {})
    elif fault == "new_call":
        receipt["cli_calls"] = [{"command": "new", "status": "failed"}]
    elif fault == "error":
        receipt["error"] = "Refused('allocation identity uncertain')"
    elif fault == "exports":
        receipt["exports"] = [{"path": "completed"}]
    else:
        receipt["manifest_sha256"] = "0" * 64
    write(output / "receipt.json", receipt)
    with pytest.raises(ValueError):
        ar.eligible(value, output, log)


def test_single_a100_completion_does_not_reenter_h100_preference(packet, tmp_path):
    value = ar.read(packet)
    value["requirements"] = {"gpu": "A100", "allowed_gpus": ["A100"], "prefer": ["H100", "A100"]}
    write(packet, value)
    log = tmp_path / "job.log"
    finish(packet, log)
    successor = ar.read(ar.prepare(packet, log, tmp_path / "retry", tmp_path))
    assert successor["requirements"] == {"gpu": "A100", "allowed_gpus": ["A100"]}


def test_existing_dispatch_with_different_snapshot_is_parked(tmp_path):
    ops = add_jobs(tmp_path, 1)
    value = ar.read(tmp_path / "job0.json")
    value["changed"] = True
    manifest = write(ops / "manifests/job0.json", value)
    pending_fence_receipt(manifest, ops / "logs/job0.log")
    assert rq.tick(ops, inventory=set())["dispatched"] == []
    row = ar.read(ops / "ready-queue.json")["jobs"][0]
    assert row["status"] == "HELD" and "differs" in row["error"]


def test_worker_start_failure_parks_only_that_item(tmp_path, monkeypatch):
    ops = add_jobs(tmp_path, 2)
    calls = []

    def start(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise OSError("worker spawn refused")

    monkeypatch.setattr(rq.subprocess, "Popen", start)
    assert rq.tick(ops, inventory=set())["dispatched"] == ["q1"]
    head, tail = ar.read(ops / "ready-queue.json")["jobs"]
    assert head["status"] == "HELD" and tail["status"] == "DISPATCHING"


def test_long_retry_identity_follows_dispatch_hashed_fallback():
    from research.compute.gpu_preference import sibling_id

    name = "q" * 70
    fallback = sibling_id(name)
    state = {"active": True, "endpoint": "live"}
    assert rq.current_observation({fallback: state}, name) == (fallback, state)
    row = {"argv": ["/dispatch.sh", name], "status": "LAUNCHED"}
    assert rq.occupancy({"jobs": [row]}, {fallback: state}, {"live"}) == (1, 0)


def test_completed_predispatch_refusal_does_not_reserve_runtime_slot(tmp_path):
    ops = add_jobs(tmp_path, 1)
    script = tmp_path / "dispatch.sh"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="DISPATCHING", token="claim")
    rq.worker(ops, "q0", "claim")
    queue = ar.read(ops / "ready-queue.json")
    assert queue["jobs"][0]["status"] == "HELD"
    assert not queue["jobs"][0]["dispatch_attempted"]
    assert rq.occupancy(queue, {}, set()) == (0, 0)


@pytest.mark.parametrize("residue", [False, True])
def test_lock_refusal_backs_off_only_without_identity_residue(tmp_path, residue):
    ops = add_jobs(tmp_path, 1)
    script = tmp_path / "dispatch.sh"
    script.write_text(
        '#!/bin/sh\nprintf "REFUSE: account allocation lock is held: %s/alloc.lock\\n" '
        '"$OPS_DIR"\nexit 1\n'
    )
    script.chmod(0o755)
    if residue:
        (ops / "dispatch" / "job0").mkdir(parents=True)
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="DISPATCHING", token="claim")
    rq.worker(ops, "q0", "claim")
    queue = ar.read(ops / "ready-queue.json")
    row = queue["jobs"][0]
    assert row["status"] == ("HELD" if residue else "READY")
    assert not row["dispatch_attempted"]
    if not residue:
        assert row["lock_refusals"] == 1
        assert not rq.ready(queue, row)
        assert rq.occupancy(queue, {}, set()) == (0, 0)


@pytest.mark.parametrize("kind", ["lock", "capacity"])
def test_real_worker_retries_transient_refusal_with_fresh_log(tmp_path, kind):
    ops = add_jobs(tmp_path, 1)
    prepare = tmp_path / "prepare.py"
    prepare.write_text(
        "from pathlib import Path\n" f'Path({str(tmp_path / "prepared")!r}).open("x").close()\n'
    )
    with rq.locked(ops) as queue:
        queue["jobs"][0]["prepare_command"] = [sys.executable, str(prepare)]
    script = tmp_path / "dispatch.sh"
    refusal = (
        'printf "REFUSE: account allocation lock is held: %s/alloc.lock\\n" "$OPS_DIR"\n'
        if kind == "lock"
        else 'printf "REFUSE: account-wide assignment cap 3 (including notebooks)\\n"\n'
    )
    script.write_text(
        '#!/bin/sh\nif [ ! -f "$OPS_DIR/refused" ]; then\n'
        'touch "$OPS_DIR/refused"\n' + refusal + 'exit 1\nfi\nprintf called >> "$OPS_DIR/calls"\n'
    )
    script.chmod(0o755)
    assert rq.tick(ops)["dispatched"] == ["q0"]

    def wait_status(expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = ar.read(ops / "ready-queue.json")["jobs"][0]
            if row["status"] == expected:
                return row
            time.sleep(0.02)
        pytest.fail(f"worker never reached {expected}")

    row = wait_status("READY")
    assert row["prepared"]
    assert row["not_before"] > time.time()
    with rq.locked(ops) as queue:
        queue["jobs"][0]["not_before"] = 0
    assert rq.tick(ops)["dispatched"] == ["q0"]
    wait_status("LAUNCHED")
    assert (ops / "calls").read_text() == "called"
    assert len(list(ops.glob("queue-worker-q0*.log"))) == 2


def account_cap_receipt(manifest, log):
    receipt = pending_fence_receipt(manifest, log)
    receipt.update(
        failure_class="ACCOUNTING",
        error="Refused('account-wide assignment cap 3 (including notebooks)')",
        usage_before={"assignments": 3.0, "balance": 685.97, "rate": 10.61},
        cli_calls=[
            {"command": command, "operation": "control", "status": "completed", "attempt": 1}
            for command in ("version", "sessions", "usage")
        ],
    )
    write(Path(ar.read(manifest)["output_dir"]) / "receipt.json", receipt)
    return receipt


def test_account_cap_receipt_queues_fresh_completion_with_backoff(tmp_path):
    ops = add_jobs(tmp_path, 1)
    manifest = write(ops / "manifests/job0.json", ar.read(tmp_path / "job0.json"))
    account_cap_receipt(manifest, ops / "logs/job0.log")
    with rq.locked(ops) as queue:
        queue["jobs"][0]["retry_limit"] = 2
    assert rq.tick(ops, inventory=set())["dispatched"] == []
    parent, retry = ar.read(ops / "ready-queue.json")["jobs"]
    assert parent["status"] == "FAILED" and parent["retry_id"] == retry["id"]
    assert retry["status"] == "READY" and retry["not_before"] > time.time()
    successor = ar.read(Path(retry["manifest"]))
    original = ar.read(manifest)
    assert successor["job_id"] != original["job_id"]
    assert successor["output_dir"] != original["output_dir"]
    assert successor["completion_attempt"] == 1
    for key in original.keys() - {"job_id", "output_dir", "completion_attempt"}:
        assert successor[key] == original[key]
    assert rq.occupancy(ar.read(ops / "ready-queue.json"), rq.observations(ops), set()) == (0, 0)


@pytest.mark.parametrize(
    "fault",
    [
        "new_call",
        "failed_read",
        "missing_read",
        "active",
        "pending",
        "endpoint",
        "error",
        "exports",
        "export_tree",
        "export_file",
        "hardware",
        "terminal",
        "cleanup",
        "hash",
        "usage_after",
        "control_operation",
        "transfers",
        "contents_transfers",
        "no_exit",
    ],
)
def test_account_cap_retry_refuses_residue_and_ambiguous_evidence(packet, tmp_path, fault):
    log = tmp_path / "job.log"
    receipt = account_cap_receipt(packet, log)
    value = ar.read(packet)
    output = Path(value["output_dir"])
    if fault in {"active", "pending"}:
        write(Path.home() / ".config/colab-cli/snake" / f"{fault}-{receipt['session']}.json", {})
    elif fault == "new_call":
        receipt["cli_calls"].append({"command": "new", "status": "failed"})
    elif fault == "failed_read":
        receipt["cli_calls"][1]["status"] = "failed"
    elif fault == "missing_read":
        receipt["cli_calls"].pop()
    elif fault == "endpoint":
        receipt["endpoint"] = "allocated"
    elif fault == "error":
        receipt["error"] = "Refused('account-wide assignment cap exceeded during job')"
    elif fault in {"exports", "transfers", "contents_transfers", "control_operation"}:
        receipt["control_operations" if fault == "control_operation" else fault] = [{"path": "x"}]
    elif fault == "export_tree":
        write(output / "exports/unit", {})
    elif fault == "export_file":
        write(output / "exports", {})
    elif fault in {"hardware", "terminal"}:
        write(output / f"{fault}.json", {})
    elif fault == "cleanup":
        receipt["cleanup_error"] = "unknown"
    elif fault == "usage_after":
        receipt["usage_after"] = receipt["usage_before"]
    elif fault == "no_exit":
        log.write_text("still running\n")
    else:
        receipt["manifest_sha256"] = "0" * 64
    write(output / "receipt.json", receipt)
    with pytest.raises(ValueError):
        ar.eligible(value, output, log)


@pytest.mark.parametrize(
    "residue", [None, "dispatch", "manifest", "live_manifest", "output", "log"]
)
def test_capacity_worker_backs_off_only_without_identity_residue(tmp_path, residue):
    ops = add_jobs(tmp_path, 1)
    script = tmp_path / "dispatch.sh"
    script.write_text(
        '#!/bin/sh\nprintf "REFUSE: account-wide assignment cap 3 (including notebooks)\\n"\n'
        "exit 1\n"
    )
    script.chmod(0o755)
    if residue == "dispatch":
        (ops / "dispatch/job0").mkdir(parents=True)
    elif residue == "manifest":
        write(ops / "manifests/job0.json", {})
    elif residue == "live_manifest":
        write(ops / "manifests/job0.live.json", {})
    elif residue == "output":
        Path(ar.read(tmp_path / "job0.json")["output_dir"]).mkdir()
    elif residue == "log":
        (ops / "logs").mkdir()
        (ops / "logs/job0.log").write_text("INCOMPLETE\n")
    with rq.locked(ops) as queue:
        queue["jobs"][0].update(status="DISPATCHING", token="claim")
    rq.worker(ops, "q0", "claim")
    queue = ar.read(ops / "ready-queue.json")
    row = queue["jobs"][0]
    assert row["status"] == ("LAUNCHED" if residue == "log" else "HELD" if residue else "READY")
    if residue is None:
        assert row["capacity_refusals"] == 1 and not row["claimed"]
        assert not rq.ready(queue, row)
        assert rq.occupancy(queue, {}, set()) == (0, 0)
