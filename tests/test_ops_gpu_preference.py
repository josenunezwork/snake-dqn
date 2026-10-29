"""H100 preference admission and provider-free dispatch boundary proofs."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.compute.gpu_preference import prepare_preference, sibling_id, variants
from tests import test_ops_dispatch as shared

fixture = shared.fixture
wait_for = shared.wait_for
REAL_FAILURE = Path(__file__).parent / "fixtures/h100_assign_503_20261028"


def preferred(f: dict, mode: str = "gpu_success") -> None:
    packet = json.loads(f["manifest"].read_text())
    packet.update(
        test_mode=mode,
        timeout_minutes=10,
        requirements={
            "gpu": "H100",
            "allowed_gpus": ["H100"],
            "prefer": ["H100", "A100"],
            "est_minutes": 5,
            "min_vcpu": 2,
        },
        budget={
            "rate_units_per_hour": 12.0,
            "unit_cap": 2.0,
            "reserve_units": 0,
            "other_assignments": {},
        },
    )
    f["manifest"].write_text(json.dumps(packet))


def trace(f: dict, action: str) -> list[dict]:
    path = Path(f["env"]["TRACE"])
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["action"] == action
    ]


def test_h100_healthy_path_seals_both_before_submit(fixture):
    f = fixture
    preferred(f)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "requested_gpu=H100" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job"]
    all_rows = [json.loads(line) for line in Path(f["env"]["TRACE"]).read_text().splitlines()]
    first_run = next(i for i, row in enumerate(all_rows) if row["action"] == "run")
    assert {row["job"] for row in all_rows[:first_run]} == {"job", "job-a100-fallback"}
    admission = json.loads((f["ops"] / "dispatch/lane/gpu-preference.json").read_text())
    assert [row["gpu"] for row in admission["variants"]] == ["H100", "A100"]
    for row in admission["variants"]:
        assert hashlib.sha256(Path(row["manifest"]).read_bytes()).hexdigest() == row["sha256"]
    receipt_path = f["ops"] / "dispatch/lane/gpu-receipt.json"
    wait_for(receipt_path.exists)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["requested_gpu"] == "H100"
    assert receipt["observed_hardware"]["hardware"]["kind"] == "H100"
    assert not (f["tmp"] / "output-a100-fallback").exists()


@pytest.mark.parametrize("mode", ["gpu_unavailable", "gpu_service_unavailable"])
def test_allocation_unavailable_reconciles_then_uses_a100(fixture, mode):
    f = fixture
    preferred(f, mode)
    f["env"]["GPU_FAILURE_FIXTURE"] = str(REAL_FAILURE / "receipt.json")
    f["env"]["EXIT_DELAY"] = ".15"
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "requested_gpu=A100" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job", "job-a100-fallback"]
    before = json.loads((f["tmp"] / "output/receipt.json").read_text())
    assert before["stop_verified"] is False  # No invented stop of an unassigned endpoint.
    proof = json.loads((f["ops"] / "dispatch/lane/unavailable.json").read_text())
    assert proof["classification"] == "ALLOCATION_UNAVAILABLE"
    assert proof["endpoint_ever_assigned"] is False
    assert not Path(proof["pending"]["path"]).exists()
    fallback = json.loads((f["ops"] / "manifests/lane-a100-fallback.live.json").read_text())
    assert fallback["requirements"]["allowed_gpus"] == ["A100"]
    assert fallback["budget"]["rate_units_per_hour"] == 8.0
    assert fallback["budget"]["unit_cap"] == pytest.approx(10 * 8 / 60)
    receipt_path = f["ops"] / "dispatch/lane-a100-fallback/gpu-receipt.json"
    wait_for(receipt_path.exists)
    assert json.loads(receipt_path.read_text())["observed_hardware"]["hardware"]["kind"] == "A100"


@pytest.mark.parametrize("mode", ["gpu_endpoint_failure", "gpu_timeout", "gpu_entitlement"])
def test_failure_or_uncertainty_never_falls_back(fixture, mode):
    f = fixture
    preferred(f, mode)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert len(trace(f, "run")) == 1
    assert not (f["ops"] / "dispatch/lane/unavailable.json").exists()


def test_recovered_endpoint_keeps_pending_and_refuses_fallback(fixture):
    f = fixture
    preferred(f, "gpu_unavailable")
    f["env"]["RECOVER_ENDPOINT"] = "real-endpoint"
    result = f["invoke"]()
    assert result.returncode != 0
    assert "endpoint recovered" in result.stderr
    assert len(trace(f, "run")) == 1
    assert list((Path(f["env"]["HOME"]) / ".config/colab-cli/snake").glob("pending-*.json"))


def test_both_unavailable_never_attempts_a_third_allocation(fixture):
    f = fixture
    preferred(f, "gpu_both_unavailable")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job", "job-a100-fallback"]


def test_post_endpoint_accounting_failure_never_uses_legacy_retry(fixture):
    f = fixture
    preferred(f, "retry_success")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job"]


def test_fallback_existing_output_refuses_primary(fixture):
    f = fixture
    preferred(f)
    (f["tmp"] / "output-a100-fallback").mkdir()
    result = f["invoke"]()
    assert result.returncode != 0
    assert "identity already used" in result.stderr
    assert not Path(f["env"]["TRACE"]).exists()


@pytest.mark.parametrize(
    "field,value", [("unit_cap", 1.99), ("rate_units_per_hour", 7.5), ("unit_cap", True)]
)
def test_pricing_refuses_before_reservation(fixture, field, value):
    f = fixture
    preferred(f)
    packet = json.loads(f["manifest"].read_text())
    packet["budget"][field] = value
    f["manifest"].write_text(json.dumps(packet))
    result = f["invoke"]()
    assert result.returncode != 0
    assert not (f["ops"] / "dispatch").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "logs/lane-a100-fallback.log",
        "dispatch/lane-a100-fallback",
        "manifests/lane-a100-fallback.json",
    ],
)
def test_fallback_identity_reuse_refuses_primary(fixture, relative):
    f = fixture
    preferred(f)
    path = f["ops"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if path.suffix == ".log":
        path.write_text("EXIT=0\n")
    result = f["invoke"]()
    assert result.returncode != 0
    assert "identity already used" in result.stderr
    assert not Path(f["env"]["TRACE"]).exists()


def test_preference_dry_run_validates_both_without_reservation(fixture):
    f = fixture
    preferred(f)
    result = f["invoke"](dry_run=True)
    assert result.returncode == 0, result.stderr
    assert {row["job"] for row in trace(f, "validate")} == {"job", "job-a100-fallback"}
    assert not trace(f, "run")
    assert not (f["ops"] / "dispatch").exists()


def test_live_refresh_admits_calibrated_h100(fixture):
    f = fixture
    f["inventory"](
        {"h": "H100"},
        [
            {
                "session": "other",
                "endpoint": "h",
                "priced_rate_units_per_hour": 12,
                "observed_rate_units_per_hour": 10.3,
            }
        ],
    )
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert trace(f, "run")[0]["other"] == {"h": 10.3}


def test_preparation_changes_only_new_operational_envelope():
    packet = {
        "job_id": "x" * 64,
        "output_dir": "/tmp/never-dispatch/",
        "timeout_minutes": 20,
        "requirements": {"gpu": "A100", "est_minutes": 10},
        "budget": {"rate_units_per_hour": 8, "unit_cap": 3, "reserve_units": 2},
        "env": {"SEED": "unchanged"},
        "inputs": [],
    }
    prepare_preference(packet)
    h100, a100 = variants(packet)
    assert len(a100["job_id"]) == 64 and a100["job_id"] != h100["job_id"]
    assert sibling_id(packet["job_id"]) == a100["job_id"]
    assert a100["env"] == h100["env"] == {"SEED": "unchanged"}
    assert a100["output_dir"] == "/tmp/never-dispatch-a100-fallback"
    assert h100["budget"]["unit_cap"] == 4
    assert h100["technical_envelope"]["base_unit_cap"] == 2


@pytest.mark.parametrize("fail_finalize", [False, True])
def test_receipt_finalization_records_exit_and_diagnostics(fixture, fail_finalize):
    f = fixture
    preferred(f)
    f["env"]["EXIT_DELAY"] = ".6"
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    destination = f["ops"] / "dispatch/lane/gpu-receipt.json"
    if fail_finalize:
        # A create-only collision exercises the real finalizer's traceback.
        destination.write_text("preserved evidence\n")
    log = f["ops"] / "logs/lane.log"
    wait_for(lambda: "EXIT=" in log.read_text())
    if fail_finalize:
        assert "FileExistsError" in log.read_text()
        assert "Traceback" in log.read_text()
        assert "EXIT=1\n" in log.read_text()
        assert destination.read_text() == "preserved evidence\n"
    else:
        assert "EXIT=0\n" in log.read_text()
        assert json.loads(destination.read_text())["requested_gpu"] == "H100"
        assert "Traceback" not in log.read_text()


@pytest.mark.parametrize("custom_state", [False, True])
def test_long_fallback_name_uses_matching_default_or_explicit_state(fixture, custom_state):
    f = fixture
    preferred(f, "gpu_unavailable")
    name = "x" * 64
    state = f["tmp"] / "explicit state" if custom_state else None
    result = f["invoke"](name=name, state_dir=state)
    assert result.returncode == 0, result.stderr
    assert "requested_gpu=A100" in result.stdout
    expected = (
        Path(str(state) + "-a100-fallback")
        if custom_state
        else f["ops"] / "state" / sibling_id(name)
    )
    assert (expected / "job-a100-fallback").is_dir()
    if not custom_state:
        assert len(expected.name) == 64
        assert not (f["ops"] / "state" / (name + "-a100-fallback")).exists()


@pytest.mark.parametrize("custom_state", [False, True])
def test_existing_long_fallback_state_refuses_before_primary(fixture, custom_state):
    f = fixture
    preferred(f)
    name = "x" * 64
    state = f["tmp"] / "explicit state" if custom_state else None
    expected = (
        Path(str(state) + "-a100-fallback")
        if custom_state
        else f["ops"] / "state" / sibling_id(name)
    )
    expected.mkdir(parents=True)
    result = f["invoke"](name=name, state_dir=state)
    assert result.returncode != 0
    assert "identity already used" in result.stderr
    assert not Path(f["env"]["TRACE"]).exists()


@pytest.mark.parametrize("old_cap", [1.0, 4.0, 99.0])
def test_new_envelopes_price_each_hardware_forecast_not_inherited_cap(old_cap):
    packet = {
        "job_id": "new-packet",
        "output_dir": "/tmp/new-packet-output",
        "timeout_minutes": 20,
        "requirements": {"est_minutes": 10},
        "budget": {"unit_cap": old_cap, "reserve_units": 2},
    }
    prepare_preference(packet)
    for item, rate in zip(variants(packet), (12, 8)):
        envelope = item["technical_envelope"]
        assert item["budget"]["unit_cap"] == pytest.approx(20 * rate / 60)
        assert envelope["base_unit_cap"] == pytest.approx(10 * rate / 60)
        assert envelope["effective_unit_cap"] == item["budget"]["unit_cap"]
        assert envelope["effective_unit_cap"] == 2 * envelope["base_unit_cap"]
        assert envelope["base_timeout_minutes"] == 10
        assert envelope["effective_timeout_minutes"] == 20
        assert item["budget"]["reserve_units"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "healthy",
        "503",
        "400",
        "timeout",
        "url",
        "accelerator",
        "method",
        "suffix",
        "endpoint",
        "accounting_endpoint",
        "terminal",
        "exports",
        "hardware",
        "new_completed",
    ],
)
def test_real_producer_allocation_classification(tmp_path, mutation):
    from tools.ops.dispatch import allocation_unavailable

    receipt = json.loads((REAL_FAILURE / "receipt.json").read_text())
    detail = ast.literal_eval(receipt["error"][len("Refused(") : -1])
    replacements = {
        "503": ("Service Unavailable", "503"),
        "400": ("Service Unavailable", "Bad Request"),
        "timeout": ("Service Unavailable", "Read timed out"),
        "url": ("/tun/m/assign?", "/tun/m/other?"),
        "accelerator": ("accelerator=H100", "accelerator=A100"),
        "method": ("request POST", "request GET"),
        "suffix": ("Service Unavailable", "Service Unavailable extra"),
    }
    if mutation in replacements:
        detail = detail.replace(*replacements[mutation])
        receipt["error"] = "Refused(" + repr(detail) + ")"
    if mutation == "endpoint":
        receipt["endpoint"] = "assigned"
    if mutation == "accounting_endpoint":
        receipt["accounting"]["allocated_endpoint"] = "assigned"
    if mutation == "terminal":
        receipt["terminal"] = {"status": "FAILED"}
    if mutation == "exports":
        receipt["exports"] = [{"path": "payload"}]
    if mutation == "hardware":
        (tmp_path / "hardware.json").write_text("{}")
    if mutation == "new_completed":
        next(c for c in receipt["cli_calls"] if c["command"] == "new")["status"] = "completed"
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    log = tmp_path / "producer.log"
    log.write_bytes((REAL_FAILURE / "producer.log").read_bytes())
    assert allocation_unavailable(tmp_path, log) is (mutation in {"healthy", "503"})


def retained_pending(f):
    """Produce a failed attempt with the real error and a retained fixture intent."""
    preferred(f, "gpu_service_unavailable")
    f["env"]["GPU_FAILURE_FIXTURE"] = str(REAL_FAILURE / "receipt.json")
    f["env"]["RECOVER_ENDPOINT"] = "uncertain-endpoint"
    result = f["invoke"]()
    assert result.returncode != 0 and "endpoint recovered" in result.stderr
    del f["env"]["RECOVER_ENDPOINT"]
    account = Path(f["env"]["HOME"]) / ".config/colab-cli/snake"
    return next(account.glob("pending-*.json"))


def reconcile_command(f, pending, *, live=False):
    return [
        sys.executable,
        str(shared.ROOT / "tools/ops/reconcile_gpu_pending.py"),
        "--manifest",
        str(f["ops"] / "manifests/lane.live.json"),
        "--state-dir",
        str(f["ops"] / "state/lane"),
        "--log",
        str(f["ops"] / "logs/lane.log"),
        "--pending",
        str(pending),
        "--proof",
        str(f["tmp"] / "archive.json"),
        "--broker",
        str(f["broker"]),
        "--live" if live else "--dry-run",
    ]


@pytest.mark.parametrize("release", [True, False])
def test_fallback_waits_for_account_lock_or_retains_pending_at_deadline(fixture, release):
    f = fixture
    preferred(f, "gpu_service_unavailable")
    f["env"]["GPU_FAILURE_FIXTURE"] = str(REAL_FAILURE / "receipt.json")
    account = Path(f["env"]["HOME"]) / ".config/colab-cli/snake"
    account.mkdir(parents=True)
    proof = f["ops"] / "dispatch/lane/unavailable.json"
    with (account / "broker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = f["invoke"](capture=False)
        log = f["ops"] / "logs/lane.log"
        wait_for(lambda: log.exists() and "EXIT=1" in log.read_text())
        pending = next(account.glob("pending-*.json"))
        raw = pending.read_bytes()
        receipt = (f["tmp"] / "output/receipt.json").read_bytes()
        # A short contention must neither refuse nor retire the fence.
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        assert pending.read_bytes() == raw
        assert not proof.exists()
        assert len(trace(f, "run")) == 1
        with (f["ops"] / "alloc.lock").open("a") as outer:
            with pytest.raises(BlockingIOError):
                fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if release:
            fcntl.flock(lock, fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=10)
        assert (f["tmp"] / "output/receipt.json").read_bytes() == receipt
        if release:
            assert process.returncode == 0, stderr
            assert "requested_gpu=A100" in stdout
            assert [row["job"] for row in trace(f, "run")] == ["job", "job-a100-fallback"]
            assert (
                json.loads(proof.read_text())["pending"]["sha256"]
                == hashlib.sha256(raw).hexdigest()
            )
            assert not pending.exists()
        else:
            assert process.returncode != 0 and "account allocation lock is held" in stderr
            assert pending.read_bytes() == raw
            assert not proof.exists()
            assert len(trace(f, "run")) == 1
            assert not (f["tmp"] / "output-a100-fallback").exists()


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("release", [True, False])
def test_operator_waits_for_account_lock_or_retains_pending_at_deadline(fixture, live, release):
    f = fixture
    pending = retained_pending(f)
    raw = pending.read_bytes()
    active = pending.parent / "active-preexisting.json"
    active.write_text('{"session":"preexisting"}')
    before = active.read_bytes()
    argv = reconcile_command(f, pending, live=live) + ["--timeout-seconds", "2"]
    with (pending.parent / "broker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(
            argv, env=f["env"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.3)
        assert pending.read_bytes() == raw
        assert not (f["tmp"] / "archive.json").exists()
        if release:
            fcntl.flock(lock, fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=10)
    if release:
        assert process.returncode == 0, stderr
        assert json.loads(stdout)["pending"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert pending.exists() is not live
        assert (f["tmp"] / "archive.json").exists() is live
    else:
        assert process.returncode != 0 and "account allocation lock is held" in stderr
        assert pending.read_bytes() == raw
        assert not (f["tmp"] / "archive.json").exists()
    assert active.read_bytes() == before
    assert len(trace(f, "run")) == 1


@pytest.mark.parametrize("live", [False, True])
def test_operator_reconciliation_repeats_proof_and_preserves_active(fixture, live):
    f = fixture
    pending = retained_pending(f)
    raw = pending.read_bytes()
    active = pending.parent / "active-preexisting.json"
    active.write_text('{"session":"preexisting"}')
    before = active.read_bytes()
    result = subprocess.run(
        reconcile_command(f, pending, live=live),
        env=f["env"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["pending"]["record"] == json.loads(raw)
    assert evidence["pending"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["pending_action"] == ("unlink_after_archive" if live else "preserve")
    assert pending.exists() is not live
    archive = f["tmp"] / "archive.json"
    assert archive.exists() is live
    if live:
        assert json.loads(archive.read_text()) == evidence
    else:
        assert pending.read_bytes() == raw
    assert active.read_bytes() == before
    assert not (Path(f["env"]["HOME"]) / "real-colab-called").exists()
    assert len(trace(f, "run")) == 1  # Operator path never launches the fallback.


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "601"])
def test_operator_refuses_invalid_deadline_before_proof_or_mutation(fixture, seconds):
    f = fixture
    argv = reconcile_command(f, f["tmp"] / "pending.json", live=True)
    result = subprocess.run(
        argv + ["--timeout-seconds", seconds],
        env=f["env"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0 and "timeout must be positive and at most 600" in result.stderr
    assert not (f["tmp"] / "archive.json").exists()
    assert not Path(f["env"]["TRACE"]).exists()


@pytest.mark.parametrize(
    "fault",
    [
        "recovered",
        "sessions",
        "foreign",
        "archive",
        "pending_path",
        "assignment",
        "manifest",
        "proof_failure",
    ],
)
def test_operator_reconciliation_refuses_ambiguous_evidence(fixture, fault):
    f = fixture
    pending = retained_pending(f)
    raw = pending.read_bytes()
    selected = pending
    if fault == "recovered":
        f["env"]["RECOVER_ENDPOINT"] = "real-endpoint"
    elif fault == "sessions":
        f["env"]["INVENTORY_RAW"] = "[?] another | Hardware: L4 | Shape: Standard | Variant: GPU"
    elif fault == "foreign":
        (pending.parent / "pending-foreign.json").write_text("{}")
    elif fault == "archive":
        (f["tmp"] / "archive.json").write_text("preserved")
    elif fault == "pending_path":
        selected = pending.parent / "active-preexisting.json"
    elif fault == "assignment":
        lane = f["ops"] / "state/lane/job"
        (lane / "sessions.assignment.json").write_text('{"endpoint":"real-endpoint"}')
    elif fault == "manifest":
        (f["tmp"] / "output/manifest.json").write_text("{}")
    elif fault == "proof_failure":
        (f["tmp"] / "blocked").write_text("not a directory")
    argv = reconcile_command(f, selected, live=True)
    if fault == "proof_failure":
        argv[argv.index("--proof") + 1] = str(f["tmp"] / "blocked/archive.json")
    result = subprocess.run(argv, env=f["env"], text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "REFUSE:" in result.stderr
    assert pending.read_bytes() == raw
    assert len(trace(f, "run")) == 1
