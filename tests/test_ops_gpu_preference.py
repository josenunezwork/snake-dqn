"""H100 preference admission and provider-free dispatch boundary proofs."""

from __future__ import annotations

import argparse
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


@pytest.mark.parametrize("ct_packet", [False, True])
@pytest.mark.parametrize("fallback_seconds", [100, 181, 601])
def test_slow_fallback_validation_gets_fresh_bounded_window(
    fixture, monkeypatch, fallback_seconds, ct_packet
):
    """Simulate the 100s C/T preflight after 545s spent on the primary."""
    from tools.ops import dispatch as module

    f = fixture
    preferred(f, "gpu_unavailable")
    packet = json.loads(f["manifest"].read_text())
    packet["sandbox_admission"] = {}  # Select the bounded host-admission window.
    if ct_packet:
        packet["script"] = "research/sandbox_a_resident/ct_job.sh"
        if fallback_seconds > 420:
            fallback_seconds = 421
    bound = module.CT_OUTER_VALIDATE_TIMEOUT_SECONDS if ct_packet else 600
    initial_bound = bound if ct_packet else module.SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS
    monkeypatch.setenv("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "600")
    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module, "supervisor_launcher", lambda *_: Path("/fake/launch.sh"))
    starts, validations, windows = [], [], []
    receipt_paths, lease_paths = [], []

    def assert_locked():
        with (f["ops"] / "alloc.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def run(argv, *, timeout, **kwargs):
        assert argv[0] == "/fake/launch.sh"
        manifest = Path(argv[-1])
        live = manifest.name.endswith(".live.json")
        fallback = "a100-fallback" in manifest.name
        duration = fallback_seconds if live and fallback else 100
        if live:
            assert_locked()
        validations.append((live, fallback, timeout))
        env = kwargs["env"]
        receipt_paths.append((live, fallback, env["SNAKE_SANDBOX_TEST_RECEIPT"]))
        assert bool(env.get("SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT")) is live
        now[0] += min(duration, timeout)
        if duration > timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0)

    # Exercise validate() and command() themselves; only the subprocess is fake.
    monkeypatch.setattr(module, "run_process_tree", run)
    monkeypatch.setattr(module, "qualified", lambda *_args, **_kwargs: assert_locked())

    def refresh(path, evidence, deadline):
        assert_locked()
        windows.append(deadline - now[0])
        live = path.with_suffix(".live.json")
        live.write_bytes(path.read_bytes())
        evidence.write_text(
            json.dumps({"manifest_sha256": hashlib.sha256(live.read_bytes()).hexdigest()})
        )
        return live

    def start(_broker, _cwd, _lane, _live, log, name, deadline, **_kwargs):
        assert_locked()
        assert deadline > now[0]
        from tools.sandbox.admission import AllocationLease

        assert _kwargs["sandbox_test_receipt"] == Path(receipt_paths[-1][2])
        if lease_paths:
            assert not lease_paths[-1].exists()
        lease_path = _kwargs["sandbox_lease"]
        lease_paths.append(lease_path)
        lease = AllocationLease(lease_path, _live)
        lease.acquire()
        try:
            assert lease.remaining(9999) == deadline - now[0]
        finally:
            lease.close()
        starts.append(name)
        log.write_text("EXIT=1\n" if len(starts) == 1 else "EXIT=0\n")

    def wait(_output, _log, deadline, _poll):
        assert_locked()
        if len(starts) == 1:
            now[0] += 445
            assert deadline - now[0] == 55
            return "failed"
        return "allocated"

    def reconcile(*args):
        assert_locked()
        assert args[-1] - now[0] == 55

    monkeypatch.setattr(module, "refresh_assignments", refresh)
    monkeypatch.setattr(module, "start", start)
    monkeypatch.setattr(module, "wait_allocation", wait)
    monkeypatch.setattr(module, "allocation_unavailable", lambda *_: True)
    monkeypatch.setattr(module, "reconcile_unavailable", reconcile)
    monkeypatch.setattr(module, "ensure_watcher", lambda *_: None)
    args = argparse.Namespace(name="lane", state_dir=None)

    def call():
        return module.dispatch_preferred(
            args, packet, f["ops"], f["broker"], f["broker"], f["ops"] / "state/lane"
        )

    if fallback_seconds > bound:
        if ct_packet:
            with pytest.raises(module.Refused, match="validation failed:.*timed out") as error:
                call()
            assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
        else:
            with pytest.raises(subprocess.TimeoutExpired):
                call()
        assert starts == ["lane"]
    else:
        assert "requested_gpu=A100" in call()
        assert starts == ["lane", "lane-a100-fallback"]
    assert windows == [600, 600]
    assert validations == [
        (False, False, initial_bound),
        (False, True, initial_bound),
        (True, False, bound),
        (True, True, bound),
    ]
    assert receipt_paths[0][2] == receipt_paths[2][2]
    assert receipt_paths[1][2] == receipt_paths[3][2]
    assert receipt_paths[0][2] != receipt_paths[1][2]
    assert all(not path.exists() for path in lease_paths)
    # Exclusion is released after either success or validation timeout.
    with (f["ops"] / "alloc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


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
def test_pricing_thresholds_warn_and_continue(fixture, field, value):
    f = fixture
    preferred(f)
    packet = json.loads(f["manifest"].read_text())
    packet["budget"][field] = value
    f["manifest"].write_text(json.dumps(packet))
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    if value is not True:
        assert "WARNING" in result.stderr


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
@pytest.mark.parametrize("accelerator", ["H100", "A100", "L4", "G4", "B200"])
def test_real_producer_allocation_classification(tmp_path, mutation, accelerator):
    from tools.ops.dispatch import allocation_unavailable

    receipt = json.loads((REAL_FAILURE / "receipt.json").read_text())
    detail = ast.literal_eval(receipt["error"][len("Refused(") : -1])
    detail = detail.replace("accelerator=H100", f"accelerator={accelerator}")
    replacements = {
        "503": ("Service Unavailable", "503"),
        "400": ("Service Unavailable", "Bad Request"),
        "timeout": ("Service Unavailable", "Read timed out"),
        "url": ("/tun/m/assign?", "/tun/m/other?"),
        "accelerator": (f"accelerator={accelerator}", "accelerator=FOREIGN"),
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
    assert allocation_unavailable(tmp_path, log, accelerator) is (mutation in {"healthy", "503"})


def retained_pending(f, accelerator="H100"):
    """Produce a sealed single-GPU failure without a preference or fallback."""
    preferred(f, "gpu_retained_service_unavailable")
    packet = json.loads(f["manifest"].read_text())
    packet["requirements"].pop("prefer")
    packet["requirements"].update(gpu=accelerator, allowed_gpus=[accelerator])
    f["manifest"].write_text(json.dumps(packet))
    failure = json.loads((REAL_FAILURE / "receipt.json").read_text())
    failure["error"] = failure["error"].replace("accelerator=H100", f"accelerator={accelerator}")
    path = f["tmp"] / "failure.json"
    path.write_text(json.dumps(failure))
    f["env"]["GPU_FAILURE_FIXTURE"] = str(path)
    result = f["invoke"]()
    assert result.returncode == 0 and "allocation=failed" in result.stdout, result.stderr
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
@pytest.mark.parametrize("accelerator", ["H100", "A100"])
def test_operator_waits_for_account_lock_or_retains_pending_at_deadline(
    fixture, live, release, accelerator
):
    f = fixture
    pending = retained_pending(f, accelerator)
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
@pytest.mark.parametrize("accelerator", ["H100", "A100", "L4", "G4", "B200"])
def test_operator_reconciliation_repeats_proof_and_preserves_active(fixture, live, accelerator):
    f = fixture
    pending = retained_pending(f, accelerator)
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
    assert evidence["requested_accelerator"] == accelerator
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
@pytest.mark.parametrize("accelerator", ["H100", "A100"])
def test_operator_reconciliation_refuses_ambiguous_evidence(fixture, fault, accelerator):
    f = fixture
    pending = retained_pending(f, accelerator)
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


@pytest.mark.parametrize("error", ["Bad Request", "Read timed out", "Permission Denied"])
def test_a100_operator_refuses_other_errors_before_retiring(fixture, error):
    f = fixture
    pending = retained_pending(f, "A100")
    raw = pending.read_bytes()
    path = f["tmp"] / "output/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["error"] = receipt["error"].replace("Service Unavailable", error)
    path.write_text(json.dumps(receipt))
    result = subprocess.run(
        reconcile_command(f, pending, live=True),
        env=f["env"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0 and "unavailability is not proven" in result.stderr
    assert pending.read_bytes() == raw
    assert not (f["tmp"] / "archive.json").exists()
    assert len(trace(f, "run")) == 1


def test_a100_live_repeats_recovery_after_successful_dry_run(fixture):
    f = fixture
    pending = retained_pending(f, "A100")
    raw = pending.read_bytes()
    preview = subprocess.run(
        reconcile_command(f, pending), env=f["env"], text=True, capture_output=True, timeout=10
    )
    assert preview.returncode == 0, preview.stderr
    f["env"]["RECOVER_ENDPOINT"] = "now-assigned"
    live = subprocess.run(
        reconcile_command(f, pending, live=True),
        env=f["env"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert live.returncode != 0 and "endpoint recovered" in live.stderr
    assert pending.read_bytes() == raw
    assert not (f["tmp"] / "archive.json").exists()
    assert len(trace(f, "run")) == 1
