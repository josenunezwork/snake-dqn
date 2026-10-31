"""Subprocess integration tests for the Mac-side Colab CLI broker."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time as stdlib_time
import types
from collections import Counter
from pathlib import Path

import pytest

from research.colab_cli import supervisor

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "tests" / "fixtures" / "fake_colab.py"


def _fake_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "normal", assignments: int = 0
) -> str:
    state_path = tmp_path / "fake-state.json"
    log_path = tmp_path / "fake-calls.jsonl"
    remote_path = tmp_path / "remote"
    active = {}
    for index in range(assignments):
        active[f"gpu-notebook-{index + 1}"] = {
            "name": f"owner-notebook-{index + 1}",
            "gpu": "A100" if index == 0 else "L4",
        }
    state_path.write_text(
        json.dumps(
            {
                "balance": 1178.30,
                "rate": 6.84 if assignments else 0,
                "sessions": active,
                "exec_count": {},
            }
        )
    )
    log_path.touch()
    remote_path.mkdir()
    stub = tmp_path / "cli-stub" / "colab_cli"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "client.py").write_text(
        "class Client:\n    def assign(self,*a,**k): pass\n"
        "    def _issue_request(self,*a,**k): pass\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(stub.parent))
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state_path))
    monkeypatch.setenv("FAKE_COLAB_LOG", str(log_path))
    monkeypatch.setenv("FAKE_COLAB_REMOTE", str(remote_path))
    monkeypatch.setenv("FAKE_COLAB_MODE", mode)
    monkeypatch.setenv(
        "SNAKE_COLAB_OAUTH_PYTHON", str(REPO / "tests/fixtures/fake_colab_helper.py")
    )
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    return str(FAKE)


def _events(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]


def _assert_transport(tmp_path: Path) -> None:
    events = _events(tmp_path)
    assert events
    for event in events:
        assert event["auth"] == "oauth2"
        assert event["assign_retries"] is None
        assert event["request_timeout"] is None
        argv = event["argv"]
        assert isinstance(argv, list)
        assert argv[:4] == ["--auth", "oauth2", "--config", event["config"]]
        assert event["config"]


def _patch_virtual_clock(
    monkeypatch: pytest.MonkeyPatch, now: list[float], *, advance_sleep: bool
) -> None:
    """Install a module-local clock without mutating stdlib time for child waits."""
    from research.colab_cli import accounting, cli_guard, inputs, release, transfers

    clock = types.SimpleNamespace(**vars(stdlib_time))
    clock.monotonic = lambda: now[0]

    def sleep(seconds: float) -> None:
        if advance_sleep:
            now[0] += seconds

    clock.sleep = sleep
    for module in (supervisor, transfers, cli_guard, accounting, inputs, release):
        monkeypatch.setattr(module, "time", clock)


def _manifest(
    tmp_path: Path,
    *,
    output: Path | None = None,
    job_id: str = "fake-job",
    gpu: str = "cpu",
    rate: float = 6.84,
) -> dict[str, object]:
    commit = os.environ.get("SNAKE_TEST_PINNED_COMMIT")
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    os.environ["FAKE_COLAB_OWN_RATE"] = str(rate)
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"pinned-input")
    return {
        "job_id": job_id,
        "commit": commit,
        "script": "research/colab_cli/acceptance_probe.sh",
        "requirements": {"gpu": gpu, "min_vcpu": 1, "est_minutes": 1.0},
        "timeout_minutes": 3.0,
        "budget": {"rate_units_per_hour": rate, "unit_cap": 1.0, "reserve_units": 0.0},
        "output_dir": str(output or tmp_path / "job-output"),
        "env": {"FAKE_JOB": "1"},
        "inputs": [
            {
                "path": str(input_path),
                "dest": "input.bin",
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
        ],
        "lease_seconds": 180,
        "max_export_bytes": 256 * 1024 * 1024,
    }


@pytest.fixture(autouse=True)
def _isolate_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    # The implementation deliberately obtains runtime/bootstrap from the
    # manifest commit.  During this uncommitted test checkout, make an
    # unreachable temporary commit object containing those two files so the
    # integration tests exercise the same pinned-bundle path.
    index = tmp_path / "git-index"
    git_env = os.environ.copy()
    git_env["GIT_INDEX_FILE"] = str(index)
    subprocess.run(["git", "read-tree", "HEAD"], cwd=REPO, env=git_env, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "research/colab_cli/runtime.py",
            "research/colab_cli/bootstrap.sh",
            "research/colab_cli/release.py",
            "research/colab_cli/source.py",
            "research/colab_cli/acceptance_probe.sh",
        ],
        cwd=REPO,
        env=git_env,
        check=True,
    )
    subprocess.run(
        ["git", "update-index", "--chmod=+x", "research/colab_cli/acceptance_probe.sh"],
        cwd=REPO,
        env=git_env,
        check=True,
    )
    tree = subprocess.check_output(["git", "write-tree"], cwd=REPO, env=git_env, text=True).strip()
    commit = subprocess.check_output(
        [
            "git",
            "commit-tree",
            tree,
            "-p",
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        ],
        cwd=REPO,
        env=git_env,
        input="fake pinned test bundle\n",
        text=True,
    ).strip()
    monkeypatch.setenv("GIT_INDEX_FILE", str(index))
    monkeypatch.setenv("SNAKE_TEST_PINNED_COMMIT", commit)


def test_allocation_cap_counts_existing_notebook_assignments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=supervisor.ACCOUNT_ASSIGNMENT_CAP)
    manifest = _manifest(tmp_path)

    cap = supervisor.ACCOUNT_ASSIGNMENT_CAP
    with pytest.raises(
        supervisor.Refused,
        match=f"account concurrency cap {cap}|assignment cap {cap}",
    ):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    events = _events(tmp_path)
    assert not any(event["argv"][4] == "new" for event in events)
    _assert_transport(tmp_path)


def test_allocation_admits_sixth_assignment_below_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = supervisor.ACCOUNT_ASSIGNMENT_CAP - 1
    binary = _fake_env(monkeypatch, tmp_path, assignments=existing)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    manifest["budget"]["other_assignments"] = {
        f"gpu-notebook-{i + 1}": 6.84 / existing for i in range(existing)
    }

    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True


def test_normal_job_stops_and_verifies_backend_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)

    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    assert receipt["endpoint"]
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    assert any(event["argv"][4] == "stop" for event in _events(tmp_path))
    _assert_transport(tmp_path)


@pytest.mark.parametrize("slow_staging", [False, True])
def test_cold_first_call_is_warmed_before_any_upload_and_staging_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slow_staging: bool
) -> None:
    """A 134s startup is separate; a later 30s-bounded verify counts as staging."""
    from research.colab_cli import canary, cli_guard

    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_args, **_kwargs: now[0] >= 134.1)
    original = supervisor.CLI._call_once
    first = True
    verify_delayed = False
    uploads_at = []

    def cold_call(self, *args: str, timeout: float) -> str:
        nonlocal first, verify_delayed
        if args[0] == "exec":
            code = Path(args[args.index("-f") + 1]).read_text()
            if first:
                first = False
                assert "SNAKE_RUNTIME_READINESS" in code
                now[0] += timeout
                raise subprocess.TimeoutExpired("cold first exec", timeout)
            if slow_staging and "SNAKE_TRANSFER_VERIFY_FILE" in code and not verify_delayed:
                verify_delayed = True
                now[0] += min(120, timeout)  # Staging controls now have a 30s attempt bound.
                raise subprocess.TimeoutExpired("later verification", timeout)
        if args[0] == "upload":
            uploads_at.append(now[0])
            remote = args[-1]
            if slow_staging and remote.endswith(("bootstrap.sh", "source.py", "inputs/input.bin")):
                # Three actual staged payloads take 45s each.  The foreground
                # Contents renewal now adds 7s after each due payload, so the
                # measured transfer remains over 150s but under the shared cap.
                now[0] += 45
            elif (
                slow_staging
                and remote.endswith("lease-heartbeat.json")
                and threading.current_thread().name == "MainThread"
            ):
                # A due foreground renewal is real main-thread Contents work
                # inside staged transfer timing; the independent watchdog is
                # kept at its ordinary small control delay below.
                now[0] += 7
            else:
                # The release credential and independent watchdog remain real
                # Contents traffic, without borrowing the payload's slow path.
                now[0] += 2
        return original(self, *args, timeout=timeout)

    monkeypatch.setattr(supervisor.CLI, "_call_once", cold_call)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 2
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    output = tmp_path / "job-output"
    receipt = json.loads((output / "receipt.json").read_text())
    measurement = json.loads((output / "upload-measurement.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"]
    assert receipt["warmup_seconds"] == pytest.approx(135)
    assert min(uploads_at) >= 135  # Includes the private credential and lease PUTs.
    assert receipt["elapsed_seconds"] >= receipt["warmup_seconds"] + measurement["elapsed_seconds"]
    if slow_staging:
        assert measurement["transfer_seconds"] > 150
        with pytest.raises(canary.CanaryError, match="transfers must complete within 150"):
            canary.evidence(manifest, receipt, output, REPO)
    else:
        assert 0 < measurement["transfer_seconds"] < 150


@pytest.mark.parametrize("mode", ["normal", "github_unreachable"])
def test_staging_separates_verified_transfers_from_source_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """Source installation may exceed 150s without charging it to uploads."""
    binary = _fake_env(monkeypatch, tmp_path, mode=mode)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=False)
    original_call = supervisor.CLI._call_once
    original_rpc = supervisor.CLI.recoverable_rpc
    source_calls = []

    def timed_call(self, *args: str, timeout: float) -> str:
        if args[0] == "upload":
            now[0] += 7
        return original_call(self, *args, timeout=timeout)

    def timed_source(self, code: str, timeout: float = 120) -> object:
        if "SNAKE_SOURCE_DELIVERY" in code:
            source_calls.append(self.deadline)
            now[0] += 156.8 if mode == "normal" else 78.4
        return original_rpc(self, code, timeout=timeout)

    monkeypatch.setattr(supervisor.CLI, "_call_once", timed_call)
    monkeypatch.setattr(supervisor.CLI, "recoverable_rpc", timed_source)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 2
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    output = Path(manifest["output_dir"])
    measured = json.loads((output / "upload-measurement.json").read_text())
    receipt = json.loads((output / "receipt.json").read_text())
    transfers = 21 if mode == "normal" else 28
    assert measured["transfer_seconds"] == pytest.approx(transfers)
    source_delivery = 163.8 if mode == "normal" else 170.8
    assert measured["source_delivery_seconds"] == pytest.approx(source_delivery)
    assert measured["elapsed_seconds"] == pytest.approx(transfers + source_delivery)
    assert len(source_calls) == (1 if mode == "normal" else 2)
    staged = [
        row
        for row in receipt["transfers"]
        if row["direction"] == "upload"
        and row["remote"].rsplit("/", 1)[-1]
        in {"bootstrap.sh", "source.py", "input.bin", "source.tar.gz"}
    ]
    assert sum(row["elapsed_seconds"] for row in staged) == pytest.approx(transfers)
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True


def test_whole_staging_over_300_seconds_cannot_qualify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=False)
    original_rpc = supervisor.CLI.recoverable_rpc
    original_call = supervisor.CLI._call_once

    def timed_call(self, *args: str, timeout: float) -> str:
        if args[0] == "upload":
            now[0] += 1
        return original_call(self, *args, timeout=timeout)

    def slow_source(self, code: str, timeout: float = 120) -> object:
        if "SNAKE_SOURCE_DELIVERY" in code:
            now[0] += 301
        return original_rpc(self, code, timeout=timeout)

    monkeypatch.setattr(supervisor.CLI, "recoverable_rpc", slow_source)
    monkeypatch.setattr(supervisor.CLI, "_call_once", timed_call)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 2
    # Previously the broker completed staging after expiry and relied on the
    # canary's post-success bar. It must now refuse before another control RPC.
    with pytest.raises(supervisor.Refused, match="INCOMPLETE"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    output = Path(manifest["output_dir"])
    receipt = json.loads((output / "receipt.json").read_text())
    if (output / "upload-measurement.json").exists():
        measured = json.loads((output / "upload-measurement.json").read_text())
        assert measured["elapsed_seconds"] > 300
    assert receipt["stop_verified"] is True
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["terminal"] is None
    assert not receipt["exports"]


def test_probe_staging_cap_refusal_preserves_evidence_and_stops_before_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A probe saves its measured transfer evidence before its bar can refuse it."""
    from research.colab_cli import canary

    binary = _fake_env(monkeypatch, tmp_path)
    original_measurement = canary.staging_measurement
    original_rpc = supervisor.CLI.recoverable_rpc
    ready_controls = []
    calls = []

    def reject_probe(
        manifest: dict[str, object], measurement: dict[str, object]
    ) -> dict[str, float]:
        calls.append(measurement)
        original_measurement(manifest, measurement)
        raise canary.StagingCapError("transfers must complete within 150 seconds")

    def observe_rpc(self, code: str, timeout: float = 120) -> object:
        if "(root/'ready.json').write_text" in code:
            ready_controls.append(code)
        return original_rpc(self, code, timeout=timeout)

    monkeypatch.setattr(canary, "is_canary", lambda _manifest: True)
    monkeypatch.setattr(canary, "staging_measurement", reject_probe)
    monkeypatch.setattr(supervisor.CLI, "recoverable_rpc", observe_rpc)
    manifest = _manifest(tmp_path)

    with pytest.raises(supervisor.Refused, match="INCOMPLETE.*within 150"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    output = Path(manifest["output_dir"])
    measurement = json.loads((output / "upload-measurement.json").read_text())
    assert calls == [measurement]
    assert (output / "source-delivery.json").is_file()
    assert (output / "input-hashes.json").is_file()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["failure_class"] == "UPLOAD_CAP"
    assert receipt["stop_verified"] is True
    assert receipt["terminal"] is None
    assert not ready_controls
    _assert_transport(tmp_path)


def test_probe_staging_within_cap_releases_child_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A measured in-cap probe still reaches the ordinary child path."""
    from research.colab_cli import canary

    binary = _fake_env(monkeypatch, tmp_path)
    probe_checks = iter((True, False))
    monkeypatch.setattr(canary, "is_canary", lambda _manifest: next(probe_checks))
    manifest = _manifest(tmp_path)

    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((Path(manifest["output_dir"]) / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    assert (
        json.loads((Path(manifest["output_dir"]) / "upload-measurement.json").read_text())[
            "transfer_seconds"
        ]
        <= 150
    )
    _assert_transport(tmp_path)


@pytest.mark.parametrize(("input_elapsed", "refused_before_ready"), [(221.0, False), (301.0, True)])
def test_f5_scoped_measured_transfer_boundaries_stop_or_reach_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_elapsed: float,
    refused_before_ready: bool,
) -> None:
    """Exercise the broker's persisted, pre-ready F5 measurement path.

    The virtual clock advances only around real Contents uploads and source
    delivery.  It therefore proves the broker writes a genuine observed 221s
    transfer and, at 301 seconds, refuses before ``ready.json`` without sleeping.
    """
    from research.colab_cli import admission, canary, transfer_policy

    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_COLAB_PUBLIC_INPUT_BYTES", base64.b64encode(b"f5-input").decode())
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=False)
    pinned = os.environ["SNAKE_TEST_PINNED_COMMIT"]
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{pinned}^{{tree}}"], cwd=REPO, text=True
    ).strip()
    f5_inputs = []
    authority_rows = []
    for dest in sorted(transfer_policy._F5_TIMING_BASE_DESTINATIONS):
        payload = b"f5-control" if dest == "control.json" else b"f5-input"
        digest = hashlib.sha256(payload).hexdigest()
        if dest == "b8.pth":
            authority_rows.append(
                {
                    "dest": dest,
                    "size_bytes": len(payload),
                    "transport": "https",
                    "url": transfer_policy.B8_URL,
                    "sha256": digest,
                }
            )
            f5_inputs.append(
                {
                    "dest": dest,
                    "size_bytes": len(payload),
                    "url": transfer_policy.B8_URL,
                    "sha256": digest,
                }
            )
        else:
            path = tmp_path / dest
            path.write_bytes(payload)
            authority_rows.append(
                {"dest": dest, "size_bytes": len(payload), "transport": "upload", "sha256": digest}
            )
            f5_inputs.append(
                {"path": str(path), "dest": dest, "size_bytes": len(payload), "sha256": digest}
            )
    # Make the instrumented final upload the last operation inside the staging
    # block, so the measured 301 seconds reach canary admission rather than a
    # following upload precheck.
    f5_inputs.sort(key=lambda row: row["dest"] == "control.json")
    authority = {
        "schema": "f5-timing-recovery-synthetic-input-inventory/v1",
        "ruling_sha256": transfer_policy.F5_TIMING_RULING_SHA256,
        "namespace": transfer_policy.F5_TIMING_NAMESPACE,
        "source_commit": pinned,
        "source_tree": tree,
        "inputs": authority_rows,
    }
    authority["sha256"] = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "job_id": "f5-timing-canary-probe",
        "commit": pinned,
        "script": "research/bl_residual/f5_colab.sh",
        "source_paths": ["research/bl_residual/f5_colab.sh"],
        "device": "cuda:0",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 1},
        "timeout_minutes": 10,
        "budget": {"rate_units_per_hour": 10, "unit_cap": 100, "reserve_units": 0},
        "output_dir": str(tmp_path / "f5-output"),
        "env": {
            "F5_DEVICE": "cuda:0",
            "F5_MODE": "qualification-canary",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "inputs": f5_inputs,
        "required_exports": ["control.json"],
        "canary": {
            "schema": canary.SCHEMA,
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
        },
        "f5_timing_source_tree": tree,
        "f5_timing_synthetic_inventory": authority,
        "f5_timing_transfer_exception": transfer_policy.f5_timing_recovery_binding(
            authority["sha256"], "canary"
        ),
        "lease_seconds": 180,
        "max_export_bytes": 256 * 1024 * 1024,
    }
    monkeypatch.setattr(admission, "check_study_manifest", lambda *_args: None)
    monkeypatch.setattr(canary, "executable_identity", lambda *_args, **_kwargs: {"pinned": "ok"})
    monkeypatch.setattr(supervisor, "input_preflight", lambda _manifest: [])
    from research.colab_cli import transfers

    original_upload = transfers.upload_file
    original_rpc = supervisor.CLI.recoverable_rpc
    ready = []

    def timed_upload(cli: object, path: Path, dest: str, **kwargs: object) -> None:
        original_upload(cli, path, dest, **kwargs)
        if dest.endswith("/inputs/control.json"):
            now[0] += input_elapsed

    def timed_source(self, code: str, timeout: float = 120) -> object:
        if "SNAKE_SOURCE_DELIVERY" in code:
            now[0] += 16.0
            return {"transport": "github", "commit": pinned, "tree": tree}
        if "(root/'ready.json').write_text" in code:
            ready.append(code)
        return original_rpc(self, code, timeout=timeout)

    monkeypatch.setattr(transfers, "upload_file", timed_upload)
    monkeypatch.setattr(supervisor.CLI, "recoverable_rpc", timed_source)
    with pytest.raises((supervisor.Refused, ValueError)) as raised:
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    assert (tmp_path / "f5-output" / "upload-measurement.json").is_file(), str(raised.value)
    measured = json.loads((tmp_path / "f5-output" / "upload-measurement.json").read_text())
    assert measured["transfer_seconds"] == pytest.approx(input_elapsed)
    assert measured["input_transfer_seconds"]["control.json"] == pytest.approx(input_elapsed)
    assert bool(ready) is not refused_before_ready


def test_warmup_failure_stops_runtime_without_uploading_or_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import cli_guard

    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: False)
    with pytest.raises(supervisor.Refused, match="readiness warmup deadline"):
        supervisor.Broker(tmp_path / "lanes", binary).run(_manifest(tmp_path))
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"]
    assert receipt["warmup_seconds"] == pytest.approx(135, abs=0.002)  # Cleanup reserve.
    assert not any(event["argv"][4] in {"upload", "exec"} for event in _events(tmp_path))


@pytest.mark.parametrize(
    ("count", "outcome"),
    [
        (493, "pass"),
        (562, "pass"),
        (493, "running-then-terminal"),
        (130, "transfer-failure"),
        (2, "foreign"),
    ],
)
def test_finished_large_inventory_does_not_reenter_telemetry_per_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, count: int, outcome: str
) -> None:
    """Reproduce the two production sizes with a costly 120s full-state poll.

    The old loop forces another 120s poll per file, exhausting this 180s
    session after the first export despite already observing remote success.
    The fixed path needs one full poll and <=ceil(N/64) bounded batch GETs.
    """
    from research.colab_cli import export_batches

    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    if outcome == "running-then-terminal":
        manifest["timeout_minutes"] = 10
        manifest["budget"]["unit_cap"] = 2
    ready = False
    full_reads = 0
    original_poll = supervisor.CLI.poll_rpc
    original_batch = export_batches.export_batch
    batches = 0
    transferred = []

    def expensive_poll(self, code: str, timeout: float = 300) -> object:
        nonlocal full_reads
        if ready and "read_state" in code:
            full_reads += 1
            now[0] += 120
            self.remaining_seconds(1)  # Enforce the real first wall/unit cap.
        result = original_poll(self, code, timeout)
        if (
            ready
            and "read_state" in code
            and outcome == "running-then-terminal"
            and full_reads == 1
        ):
            return {
                "state": {"status": "running"},
                "terminal": None,
                "exports": result["exports"][:408],
            }
        return result

    def batch(*args: object, **kwargs: object) -> None:
        nonlocal batches
        batches += 1
        if outcome == "transfer-failure" and batches == 2:
            raise supervisor.Refused("injected batch transport failure")
        original_batch(*args, **kwargs)
        transferred.extend(entry["path"] for entry in args[2])

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", expensive_poll)
    monkeypatch.setattr(export_batches, "export_batch", batch)

    def observer(event: str, context: dict[str, object]) -> None:
        nonlocal ready
        if event != "ready":
            return
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        session = context["cli"].session
        remote = state["runtime"][session]
        output = tmp_path / "remote" / session / remote["root"].lstrip("/") / "output"
        inventory = []
        for index in range(count):
            payload = f"episode {index}".encode()
            name = f"episode-{index:04d}.json"
            digest = hashlib.sha256(payload).hexdigest()
            (output / name).write_bytes(payload)
            (output / (name + ".sha256")).write_text(digest + "\n")
            (output / (name + ".complete")).write_text("complete\n")
            inventory.append({"path": name, "sha256": digest, "size": len(payload)})
        remote["terminal"]["exports"] = inventory
        if outcome == "foreign":
            remote["terminal"]["generation"] = "foreign"
        state_path.write_text(json.dumps(state))
        ready = True

    broker = supervisor.Broker(tmp_path / "lanes", binary, observer=observer)
    success = outcome in {"pass", "running-then-terminal"}
    if success:
        broker.run(manifest)
    else:
        with pytest.raises(supervisor.Refused, match="INCOMPLETE"):
            broker.run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert full_reads == (2 if outcome == "running-then-terminal" else 1)
    assert now[0] == 120 * full_reads
    assert len(transferred) == len(set(transferred))
    if outcome == "foreign":
        assert batches == 0 and receipt["exports"] == []
        assert receipt["failure_class"] == "FOREIGN_GENERATION"
        assert receipt["terminal"] is None
    else:
        assert receipt["terminal"]["returncode"] == 0
        assert len(receipt["terminal"]["exports"]) == count
        expected = count if success else 64
        assert len(receipt["exports"]) == expected
        assert receipt["status"] == ("PASS" if success else "INCOMPLETE")
        downloads = [e for e in _events(tmp_path) if e["argv"][4] == "download"]
        expected_downloads = 9 if outcome == "running-then-terminal" else (count + 63) // 64
        assert len(downloads) == (expected_downloads if success else 1)
    _assert_transport(tmp_path)


def test_export_collection_skips_verified_entries_and_rejects_changed_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("unused", tmp_path / "config", "owned")
    entry = {"path": "episode.json", "sha256": "a" * 64, "size": 10}
    cli.verified_exports[entry["path"]] = dict(entry)
    monkeypatch.setattr(supervisor, "export_file", lambda *_args: pytest.fail("downloaded twice"))
    cli.renew = lambda: pytest.fail("polled unchanged inventory")
    supervisor.collect_exports(cli, "/content/snake-cli/g", [entry], tmp_path, 1024)
    with pytest.raises(supervisor.Refused, match="immutable snapshot metadata changed"):
        supervisor.collect_exports(
            cli, "/content/snake-cli/g", [{**entry, "sha256": "b" * 64}], tmp_path, 1024
        )


def test_observer_prefetched_export_is_validated_without_batch_overwrite(
    tmp_path: Path,
) -> None:
    payload = b"prefetched by acceptance"
    digest = hashlib.sha256(payload).hexdigest()
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "client.bin").write_bytes(payload)
    (exports / "client.bin.sha256").write_text(digest + "\n")
    (exports / "client.bin.complete").write_text("complete\n")
    cli = supervisor.CLI("unused", tmp_path / "config", "owned")
    entry = {"path": "client.bin", "sha256": digest, "size": len(payload)}
    supervisor.collect_exports(cli, "/content/snake-cli/g", [entry], tmp_path, 1024)
    assert cli.verified_exports == {"client.bin": entry}


def test_renewal_after_initial_staging_deadline_uses_fresh_contents_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A credential rotation after five minutes must not inherit staging's cap.

    This exercises the real broker through the subprocess-backed fake binary.
    ``ready`` is emitted only after the bounded initial input staging phase, so
    advancing the monotonic clock here reproduces an otherwise healthy runtime
    that needs a later credential renewal.
    """
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    monotonic = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10.0
    manifest["budget"]["unit_cap"] = 2.0
    renewed: list[bool] = []
    original = supervisor.CLI.call

    def delayed_credential(self, *args: str, timeout: float = 120) -> str:
        if args[0] == "upload" and args[-1].endswith("-release.json") and monotonic[0] >= 301:
            assert timeout >= 100  # The obsolete lease-minus-90 allowance failed here.
            result = original(self, *args, timeout=timeout)
            monotonic[0] += 100
            return result
        return original(self, *args, timeout=timeout)

    monkeypatch.setattr(supervisor.CLI, "call", delayed_credential)

    def observer(event: str, context: dict[str, object]) -> None:
        if event != "ready":
            return
        cli = context["cli"]
        # Stage uploads had their 300-second collective deadline at t=0. A
        # now-expired credential forces the later direct ContentsClient PUT.
        monotonic[0] = 301.0
        cli.release_credential["expires_at"] = 0
        renewed.append(True)

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    assert renewed == [True]
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    credential_puts = [
        row for row in receipt["contents_transfers"] if "-release.json" in row["remote"]
    ]
    assert len(credential_puts) >= 2  # initial credential plus the post-staging renewal
    assert credential_puts[-1]["status"] == "completed"
    assert credential_puts[-1]["bytes_per_second"] > 0
    logical_download = next(row for row in receipt["transfers"] if row["direction"] == "download")
    raw_download = next(
        row for row in receipt["contents_transfers"] if row["direction"] == "download"
    )
    assert logical_download["status"] == "verified"
    assert logical_download["bytes_per_second"] > 0
    assert raw_download["status"] == "completed"
    assert raw_download["bytes_per_second"] > 0
    assert all("total upload time cap" not in str(row) for row in receipt["transfers"])


@pytest.mark.parametrize("baseline", [False, True])
def test_l4_observed_rate_below_priced_ceiling_calibrates_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, baseline: bool
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=int(baseline))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    if baseline:
        manifest["budget"]["other_assignments"] = {"gpu-notebook-1": 6.84}
    monkeypatch.setenv("FAKE_COLAB_OWN_RATE", "1.54")
    # Account-wide samples lag before settling at baseline + own real rate.
    other = 6.84 if baseline else 0
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps([other, other + 1.54]))
    from research.colab_cli.accounting import AccountTracker

    original_read = AccountTracker.calibration_read

    def locked_read(self: AccountTracker, *args: object) -> float | None:
        # A second lane cannot read the temporary uncalibrated active record.
        with pytest.raises(supervisor.Refused, match="already supervised"):
            with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock", blocking=False):
                pass
        return original_read(self, *args)

    monkeypatch.setattr(AccountTracker, "calibration_read", locked_read)
    checked = []

    def observer(event: str, context: dict[str, object]) -> None:
        if event == "runtime_started":
            with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock", blocking=False):
                active = next(supervisor.ACCOUNT_ROOT.glob("active-*.json"))
                assert json.loads(active.read_text())[
                    "observed_rate_units_per_hour"
                ] == pytest.approx(1.54)
                checked.append(True)

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    assert checked == [True]
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    accounting = receipt["accounting"]
    assert accounting["priced_rate_units_per_hour"] == 2.0
    assert accounting["observed_rate_units_per_hour"] == pytest.approx(1.54)
    assert accounting["calibration"]["status"] == "PASS"
    assert len(accounting["calibration"]["samples"]) == supervisor.RATE_CALIBRATION_STABLE_READS + 1
    assert [
        sample["usage"]["rate"] for sample in accounting["calibration"]["samples"][-3:]
    ] == pytest.approx([other + 1.54] * 3)
    assert (
        accounting["last_crosscheck"]["own_elapsed_units"]
        >= accounting["last_crosscheck"]["observed_elapsed_units"]
    )


def _continue_past_first_poll(monkeypatch):
    """Delay terminal state once to prove warnings permit another poll."""
    ready = False
    reads = 0
    original = supervisor.CLI.poll_rpc

    def poll(self, code, timeout=300):
        nonlocal reads
        if ready and "read_state" in code:
            reads += 1
            if reads == 1:
                return {"state": {"status": "running"}, "terminal": None, "exports": []}
        return original(self, code, timeout)

    def observer(event, _context):
        nonlocal ready
        if event == "ready":
            ready = True

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", poll)
    return observer


@pytest.mark.parametrize(
    "rates,code",
    [
        ([2.06], "OBSERVED_RATE_ABOVE_PRICED_BOUND"),
        ([1.54, 1.7] * 6, "CALIBRATION_RATE_VARIATION"),
        ([1.54, 1.54, 1.54, 1.7], "PROVIDER_RATE_DRIFT"),
    ],
)
def test_rate_price_and_drift_warn_and_continue(monkeypatch, tmp_path, capsys, rates, code):
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_OWN_RATE", "1.54")
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps(rates))
    observer = _continue_past_first_poll(monkeypatch)
    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    assert f'"WARNING": "{code}"' in capsys.readouterr().err
    assert (tmp_path / "job-output/spend-000000.json").exists()
    assert (tmp_path / "job-output/poll-000001.json").exists()
    _assert_transport(tmp_path)


def test_provider_calibration_timeout_is_a_real_elapsed_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[0]")
    now = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    with pytest.raises(supervisor.Refused, match="did not stabilize"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert receipt["accounting"]["calibration"]["status"] == "INCOMPLETE"
    assert receipt["elapsed_seconds"] == pytest.approx(135, abs=0.002)  # Wall cap wins.
    # Another lane must never see an uncalibrated active record after lock release.
    with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock", blocking=False):
        assert not list(supervisor.ACCOUNT_ROOT.glob("active-*.json"))


def test_stop_success_exit_is_rejected_when_backend_keeps_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="stop_lie")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)

    with pytest.raises(supervisor.Refused, match="STOP_UNVERIFIED|INCOMPLETE"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"] is False
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions()
    _assert_transport(tmp_path)


def test_kernel_error_with_zero_cli_exit_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="kernel_error")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")

    with pytest.raises(supervisor.Refused, match="kernel error despite CLI exit zero"):
        cli.rpc("result=True")

    _assert_transport(tmp_path)


def test_busy_kernel_is_never_resubmitted_after_uncertain_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="busy_exec")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")

    with pytest.raises(supervisor.Refused, match="kernel completion not acknowledged"):
        cli.rpc("result=True")
    with pytest.raises(supervisor.Refused, match="uncertain/busy"):
        cli.rpc("result=True")

    events = _events(tmp_path)
    assert sum(event["argv"][4] == "exec" for event in events) == 1
    _assert_transport(tmp_path)


def test_corrupt_download_is_rejected_before_atomic_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="corrupt_download")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")
    cli.call("new", "-s", "fake-session")

    generation_root = "/content/snake-cli/fake-generation"
    remote_file = (
        Path(os.environ["FAKE_COLAB_REMOTE"])
        / "fake-session"
        / generation_root.lstrip("/")
        / "output"
    )
    payload = b"GOOD"
    remote_file.mkdir(parents=True)
    (remote_file / "checkpoint.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (remote_file / "checkpoint.bin.sha256").write_text(digest + "\n")
    (remote_file / "checkpoint.bin.complete").write_text("complete\n")
    entry = {"path": "checkpoint.bin", "size": len(payload), "sha256": digest}

    with pytest.raises(supervisor.Refused, match="download hash mismatch"):
        supervisor.export_file(cli, generation_root, entry, tmp_path / "published", 1024)

    assert not (tmp_path / "published" / "exports" / "checkpoint.bin").exists()
    assert (
        not list((tmp_path / "published").rglob("*.complete"))
        if (tmp_path / "published").exists()
        else True
    )
    _assert_transport(tmp_path)


def test_existing_output_directory_is_refused_as_nonfresh(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(supervisor.Refused, match="fresh Mac directory"):
        supervisor.validate(_manifest(tmp_path, output=output))


def test_h100_chain_allocates_once_stops_once_and_writes_release_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    first_output = tmp_path / "h100-first"
    second_output = tmp_path / "h100-second"
    manifests = [
        _manifest(tmp_path, output=first_output, job_id="h100-first", gpu="H100"),
        _manifest(tmp_path, output=second_output, job_id="h100-second", gpu="H100"),
    ]

    def change_second_export(event: str, context: dict[str, object]) -> None:
        if event != "ready" or context["output"] != second_output:
            return
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        session = context["cli"].session
        remote = state["runtime"][session]
        target = (
            tmp_path / "remote" / session / remote["root"].lstrip("/") / "output/checkpoint.bin"
        )
        target.write_bytes(b"SECOND")
        digest = hashlib.sha256(b"SECOND").hexdigest()
        Path(str(target) + ".sha256").write_text(digest + "\n")
        remote["terminal"]["exports"] = [{"path": "checkpoint.bin", "size": 6, "sha256": digest}]
        state_path.write_text(json.dumps(state))

    supervisor.Broker(tmp_path / "lanes", binary, observer=change_second_export).chain(
        manifests, retention_minutes=7, retention_units=1
    )

    events = _events(tmp_path)
    assert sum(event["argv"][4] == "new" for event in events) == 1
    assert sum(event["argv"][4] == "stop" for event in events) == 1
    first_receipt = json.loads((first_output / "receipt.json").read_text())
    second_receipt = json.loads((second_output / "receipt.json").read_text())
    assert first_receipt["status"] == "COMPLETED_SESSION_RETAINED"
    assert second_receipt["status"] == "COMPLETED_SESSION_RETAINED"
    assert first_receipt["session"] == second_receipt["session"]
    assert len(first_receipt["exports"]) == len(second_receipt["exports"]) == 1
    assert (first_output / "exports/checkpoint.bin").read_bytes() == b"GOOD"
    assert (second_output / "exports/checkpoint.bin").read_bytes() == b"SECOND"
    for output in (first_output, second_output):
        release = list(output.glob("session-release.json"))
        assert len(release) == 1
        evidence = json.loads(release[0].read_text())
        assert evidence["stop_verified"] is True
        assert evidence["error"] is None
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    _assert_transport(tmp_path)


@pytest.mark.parametrize("include_endpoint", [True, False])
def test_recover_pending_and_active_records_stops_owned_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_endpoint: bool
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    session = "orphan-session"
    config = tmp_path / "orphan-sessions.json"
    cli = supervisor.CLI(binary, config, session)
    cli.call("new", "-s", session)
    endpoint = next(iter(cli.sessions()))
    output = tmp_path / "recovered-output"
    output.mkdir()
    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    pending = supervisor.ACCOUNT_ROOT / f"pending-{session}.json"
    pending.write_text(
        json.dumps(
            {
                "session": session,
                "config": str(config),
                "endpoint": endpoint if include_endpoint else None,
                "output": str(output),
            }
        )
    )
    active = supervisor.ACCOUNT_ROOT / f"active-{session}.json"
    active.write_text(json.dumps({"session": session, "endpoint": endpoint}))

    supervisor.Broker(tmp_path / "lanes", binary).recover(pending)

    assert not pending.exists()
    assert not active.exists()
    evidence = list(output.glob("recovery-*.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text())["stop_verified"] is True
    assert supervisor.CLI(binary, config, session).sessions() == {}
    _assert_transport(tmp_path)


def test_partial_local_export_without_completion_marker_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "unused")
    published = tmp_path / "published" / "exports"
    target = published / "checkpoint.bin"
    target.parent.mkdir(parents=True)
    payload = b"already-downloaded"
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    Path(str(target) + ".sha256").write_text(digest + "\n")

    with pytest.raises(
        supervisor.Refused, match="immutable export changed|local export publication incomplete"
    ):
        supervisor.export_file(
            cli,
            "/content/snake-cli/generation",
            {"path": "checkpoint.bin", "size": len(payload), "sha256": digest},
            tmp_path / "published",
            1024,
        )

    assert not Path(str(target) + ".complete").exists()
    assert not _events(tmp_path)


def test_cli_error_redacts_token_bearing_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="error_url")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "unused")

    with pytest.raises(supervisor.Refused) as error:
        cli.call("version")

    message = str(error.value)
    assert "secret-token" not in message
    assert "refresh-secret" not in message
    _assert_transport(tmp_path)


def test_module_entrypoint_validate_status_and_run_with_fake_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    manifest = _manifest(tmp_path, output=tmp_path / "entry-output")
    manifest_path = tmp_path / "entry-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    state_dir = tmp_path / "entry-state"
    prefix = [
        str(Path(sys.executable)),
        "-m",
        "research.colab_cli.supervisor",
        "--state-dir",
        str(state_dir),
        "--colab",
        binary,
    ]
    env = os.environ.copy()

    validated = subprocess.run(
        [*prefix, "validate", str(manifest_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    assert not _events(tmp_path)

    status = subprocess.run(
        [*prefix, "status"], cwd=REPO, env=env, capture_output=True, text=True, check=False
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["sessions"] == {}

    ran = subprocess.run(
        [*prefix, "run", str(manifest_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    receipt = json.loads((tmp_path / "entry-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    _assert_transport(tmp_path)


def test_legacy_unit_cap_and_reserve_warn_but_keep_polling(monkeypatch, tmp_path, capsys):
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    manifest["budget"].update(unit_cap=0, reserve_units=2000)
    observer = _continue_past_first_poll(monkeypatch)
    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    output = tmp_path / "job-output"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    warnings = capsys.readouterr().err
    assert '"WARNING": "SESSION_UNIT_CAP_REACHED"' in warnings
    assert '"WARNING": "RESERVE_CROSSED"' in warnings
    assert '"WARNING": "BALANCE_BELOW_LEGACY_RESERVATIONS"' in warnings
    assert (output / "poll-000001.json").exists()
    assert (
        json.loads((output / "spend-000000.json").read_text())["unit_cap_status"]
        == "WARNING_UNIT_CAP_REACHED"
    )
    _assert_transport(tmp_path)


def test_optional_budget_runs_and_records_observed_usage(monkeypatch, tmp_path):
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    del manifest["budget"]
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    assert receipt["accounting"]["unit_cap"] is None
    assert receipt["accounting"]["observed_rate_units_per_hour"] > 0


def test_cli_call_caps_subprocess_timeout_at_session_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 100.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 99.5)
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="Colab CLI 0.7.4\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert cli.call("version", timeout=45) == "Colab CLI 0.7.4\n"
    assert seen["timeout"] == pytest.approx(0.5)
    assert seen["args"] == [
        "fake-colab",
        "--auth",
        "oauth2",
        "--config",
        str(tmp_path / "lane-sessions.json"),
        "version",
    ]
    assert seen["env"]["SNAKE_COLAB_NOTEBOOK_HASH"]


def test_cli_call_ignores_legacy_priced_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 1000.0
    cli.priced_deadline = 101.25
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="Colab CLI 0.7.4\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert cli.call("usage", timeout=45) == "Colab CLI 0.7.4\n"
    assert seen["timeout"] == pytest.approx(45)


def test_upload_phase_caps_outer_rpc_grace_and_restores_session_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 1000.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    seen = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout="no kernel acknowledgment", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    with pytest.raises(supervisor.Refused, match="not acknowledged"):
        with cli.limit_deadline(103.0):
            cli.rpc("result=True", timeout=60)
    assert seen == [3.0]
    assert cli.deadline == 1000.0


def test_backup_is_invoked_only_after_verified_remote_stop_and_receipt_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    manifest["backup_dir"] = str(tmp_path / "backup")
    order: list[str] = []

    def backup(output: Path, backup_dir: str, job: str, generation: str) -> None:
        assert supervisor.CLI(binary, tmp_path / "post-stop.json").sessions() == {}
        order.append("backup-after-stop")
        destination = Path(backup_dir) / job / generation
        destination.mkdir(parents=True)
        (destination / "snapshot.bin").write_bytes(b"verified")
        digest = hashlib.sha256(b"verified").hexdigest()
        (destination / "snapshot.bin.sha256").write_text(digest + "\n")
        (destination / "snapshot.bin.complete").write_text("complete\n")
        (output / "backup.json").write_text(
            json.dumps({"destination": str(destination), "hash_verified": True})
        )

    monkeypatch.setattr(supervisor.Broker, "backup", staticmethod(backup))
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    assert order == ["backup-after-stop"]
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    backup_record = json.loads((tmp_path / "job-output" / "backup.json").read_text())
    assert backup_record["hash_verified"] is True
    _assert_transport(tmp_path)


def test_stop_reissues_after_transient_lookup_error_with_long_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="stop_transient")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "owned-stop")
    cli.call("new", "-s", cli.session)
    endpoint = next(iter(cli.sessions()))
    timeouts = []
    original = cli.call

    def observe(*args: str, timeout: float = 45) -> str:
        if args[0] == "stop":
            timeouts.append(timeout)
        return original(*args, timeout=timeout)

    monkeypatch.setattr(cli, "call", observe)
    supervisor.stop_verified(cli, endpoint)
    assert len(timeouts) >= 2
    assert all(timeout >= 60 for timeout in timeouts)
    assert not cli.sessions()


def test_idempotent_poll_recovers_transport_hiccup_only_after_idle_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="poll_hiccup")
    cli = supervisor.CLI(binary, tmp_path / "session.json", "telemetry")
    assert cli.poll_rpc("result=True") is True
    assert sum(event["argv"][4] == "exec" for event in _events(tmp_path)) == 2


def test_other_notebook_spend_is_attributed_and_missing_rates_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=1)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    with pytest.raises(supervisor.Refused, match="unaccounted"):
        supervisor.Broker(tmp_path / "missing-lane", binary).run(manifest)
    assert not any(event["argv"][4] == "new" for event in _events(tmp_path))
    manifest = _manifest(tmp_path, job_id="accounted", output=tmp_path / "accounted-output")
    manifest["budget"]["other_assignments"] = {"gpu-notebook-1": 6.84}
    supervisor.Broker(tmp_path / "accounted-lane", binary).run(manifest)
    receipt = json.loads((tmp_path / "accounted-output/receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["accounting"]["baseline_rate_units_per_hour"] == 6.84
    assert list(supervisor.CLI(binary, tmp_path / "status.json").sessions().values()) == [
        "owner-notebook-1"
    ]


def test_recover_unknown_assignment_by_durable_notebook_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "interrupted")
    cli.call("new", "-s", cli.session)
    state = json.loads((tmp_path / "fake-state.json").read_text())
    endpoint = next(iter(state["sessions"]))
    state["sessions"][endpoint]["name"] = "?"
    (tmp_path / "fake-state.json").write_text(json.dumps(state))
    cli.config.unlink()
    output = tmp_path / "recovery-output"
    output.mkdir()
    supervisor.ACCOUNT_ROOT.mkdir(exist_ok=True)
    record = supervisor.ACCOUNT_ROOT / "pending-interrupted.json"
    record.write_text(
        json.dumps(
            {
                "session": cli.session,
                "config": str(cli.config),
                "output": str(output),
                "before": [],
                "notebook_hash": cli.notebook_hash,
            }
        )
    )
    supervisor.Broker(tmp_path / "recovery-state", binary).recover(record)
    assert cli.sessions() == {}
    assert not record.exists()
    assert sum(event["argv"][4] == "new" for event in _events(tmp_path)) == 1


def test_600_mib_snapshot_roundtrip_uses_bounded_fake_cli_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import transfers

    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "large-session.json", "large-transfer")
    # Initial staging completes at t=0.  The later snapshot transfer starts
    # after the old five-minute staging cap and takes another ten simulated
    # minutes; its own size-scaled deadline must govern it.
    clock = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    staging = tmp_path / "initial-staging.bin"
    staging.write_bytes(b"staged")
    transfers.upload_file(
        cli, staging, "/content/snake-cli/transfer-proof/staged.bin", lambda: None
    )
    clock[0] = 301.0
    real_call = cli.call

    def slow_upload(*args: str, **kwargs: object) -> str:
        result = real_call(*args, **kwargs)
        if args[0] == "upload":
            clock[0] += 2.0
        return result

    cli.call = slow_upload  # type: ignore[method-assign]
    source = tmp_path / "600mib.snapshot"
    size = 600 * 1024 * 1024
    with source.open("wb") as stream:
        stream.truncate(size)
    remote = "/content/snake-cli/transfer-proof/checkpoint.bin"
    digest = transfers.upload_file(cli, source, remote, renew=lambda: None)
    target = tmp_path / "downloaded.snapshot"
    transfers.download_file(cli, remote, target, size, expected_sha256=digest)
    assert target.stat().st_size == size
    assert supervisor.sha256(target) == digest
    later_upload = cli.transfer_records[-2]
    assert later_upload["direction"] == "upload"
    assert later_upload["elapsed_seconds"] == pytest.approx(600.0)
    assert later_upload["bytes_per_second"] > 0
    downloaded = cli.transfer_records[-1]
    assert downloaded["direction"] == "download"
    assert downloaded["bytes_per_second"] > 0
    assert any(
        record["direction"] == "download" and record["bytes_per_second"] > 0
        for record in cli.contents_transfers
    )
    uploads = [e for e in _events(tmp_path) if e["argv"][4] == "upload"]
    downloads = [e for e in _events(tmp_path) if e["argv"][4] == "download"]
    assert len(uploads) == 301  # One staged control file, then 300 snapshot chunks.
    assert len(downloads) == 75
    assert max(e["upload_bytes"] for e in uploads) <= 2 * 1024 * 1024
    assert not list((tmp_path / "remote").rglob("*.part"))


def test_chunk_retry_overwrites_partial_put_through_fake_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import transfers

    binary = _fake_env(monkeypatch, tmp_path, mode="upload_transient")
    cli = supervisor.CLI(binary, tmp_path / "session.json", "retry-transfer")
    source = tmp_path / "input.payload"
    source.write_bytes(b"immutable-input" * 1000)
    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 2048)
    digest = transfers.upload_file(cli, source, "/content/input.bin", renew=lambda: None)
    remote = tmp_path / "remote" / cli.session / "content/input.bin"
    assert remote.read_bytes() == source.read_bytes()
    assert supervisor.sha256(remote) == digest
    uploads = [event for event in _events(tmp_path) if event["argv"][4] == "upload"]
    assert len(uploads) == 9
    chunk_puts = Counter(event["argv"][-1] for event in uploads)
    assert len(chunk_puts) == 8
    assert sorted(chunk_puts.values()) == [1] * 7 + [2]
    assert all(path.endswith(".part") for path in chunk_puts)
    assert max(event["upload_bytes"] for event in uploads) <= 2048


@pytest.mark.parametrize(
    "mode,transport", [("normal", "github"), ("github_unreachable", "fallback")]
)
def test_runtime_source_delivery_with_fake_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, transport: str
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode=mode)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    output = Path(manifest["output_dir"])
    delivery = json.loads((output / "source-delivery.json").read_text())
    assert delivery["transport"] == transport
    assert delivery["commit"] == manifest["commit"]
    assert (
        delivery["tree"]
        == subprocess.check_output(
            ["git", "rev-parse", f"{manifest['commit']}^{{tree}}"], text=True
        ).strip()
    )
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert len(state["source_fetches"]) == (3 if mode == "github_unreachable" else 1)
    assert state["source_fetches"][0][-1] == manifest["commit"]
    assert state["source_fetches"][0][-2] == "https://github.com/josenunezwork/snake-dqn.git"
    archives = [
        event
        for event in _events(tmp_path)
        if event["argv"][4] == "upload" and event["argv"][-1].endswith("source.tar.gz")
    ]
    assert len(archives) == (1 if transport == "fallback" else 0)
    assert (output / "staging/source.tar.gz").stat().st_size < 2 * 1024 * 1024
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    _assert_transport(tmp_path)


def test_github_identity_mismatch_refuses_without_uploading_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="github_identity_mismatch")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    with pytest.raises(supervisor.Refused, match="identity"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((Path(manifest["output_dir"]) / "receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert not any(
        event["argv"][4] == "upload" and event["argv"][-1].endswith("source.tar.gz")
        for event in _events(tmp_path)
    )


def test_waiting_for_account_lock_renews_active_runtime_lease(tmp_path: Path) -> None:
    import threading

    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    renewed = threading.Event()
    acquired = threading.Event()
    cli.renew = renewed.set

    def wait_for_broker() -> None:
        with supervisor.account_locked(cli):
            acquired.set()

    with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock"):
        waiter = threading.Thread(target=wait_for_broker)
        waiter.start()
        assert renewed.wait(timeout=5)
        assert not acquired.is_set()
    waiter.join(timeout=5)
    assert acquired.is_set()
    assert not waiter.is_alive()


def test_waiting_for_calibration_lock_ignores_priced_unit_cap(monkeypatch, tmp_path):
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = 2000
    cli.priced_deadline = 1800
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 1800)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    calls = []

    def contended_once(handle, mode):
        calls.append(mode)
        if len(calls) == 1:
            raise BlockingIOError

    monkeypatch.setattr(supervisor.fcntl, "flock", contended_once)
    with supervisor.account_locked(cli):
        assert len(calls) == 2


def test_account_lock_uses_new_run_cap_instead_of_previous_retained_tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = 2100
    cli.priced_deadline = 2000
    cli.accounting = SimpleNamespace(started_monotonic=0, own_rate_units_per_hour=2, unit_cap=0.1)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 1800)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    calls = []

    def contended_once(handle: object, mode: int) -> None:
        calls.append(mode)
        if len(calls) == 1:
            raise BlockingIOError

    monkeypatch.setattr(supervisor.fcntl, "flock", contended_once)
    with supervisor.account_locked(cli):
        pass
    assert len(calls) == 3


def test_cleanup_lock_wait_ignores_expired_execution_and_uncertain_renewal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = -1

    def unsafe_renew() -> None:
        raise AssertionError("cleanup must not renew an uncertain kernel")

    cli.renew = unsafe_renew
    calls = []

    def contended_once(handle: object, mode: int) -> None:
        calls.append(mode)
        if len(calls) == 1:
            raise BlockingIOError

    monkeypatch.setattr(supervisor.fcntl, "flock", contended_once)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    with supervisor.account_locked(cli, cleanup=True):
        pass
    assert len(calls) == 3


def test_disconnect_acceptance_observes_fake_backend_endpoint_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from research.colab_cli.acceptance_faults import AcceptanceObserver

    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "disconnect")
    cli.call("new", "-s", cli.session)
    endpoint = next(iter(cli.sessions()))
    cli.accounting = SimpleNamespace(endpoint=endpoint)
    root = "/content/snake-cli/disconnect"
    remote = Path(os.environ["FAKE_COLAB_REMOTE"]) / cli.session / root.lstrip("/") / "output"
    remote.mkdir(parents=True)
    payload = b"checkpoint-on-lease-loss"
    (remote / "lease-checkpoint.pkl").write_bytes(payload)
    (remote / "lease-checkpoint.pkl.sha256").write_text(hashlib.sha256(payload).hexdigest())
    clock = [0.0]

    def disconnect_sleep(seconds: float) -> None:
        clock[0] += seconds
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        state["sessions"].pop(endpoint, None)
        state_path.write_text(json.dumps(state))

    monkeypatch.setattr(supervisor.time, "sleep", disconnect_sleep)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    observer = AcceptanceObserver("unused")
    monkeypatch.setattr(observer, "_state", lambda context: {})
    monkeypatch.setattr(cli, "poll_rpc", lambda code: clock.__setitem__(0, 180))
    output = tmp_path / "watchdog-output"
    output.mkdir()
    observer._watchdog(
        {
            "cli": cli,
            "root": root,
            "output": output,
            "generation": "g",
            "manifest": {"lease_seconds": 180},
        }
    )
    assert observer.records["mac-loss-watchdog-unassign"]["endpoint"] == endpoint
    assert not any(event["argv"][4] == "stop" for event in _events(tmp_path))


@pytest.mark.parametrize("command", ["sessions", "usage", "whoami", "url"])
def test_read_retries_transient_failures_within_one_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "state.json")
    clock = [0.0]
    calls = []
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )

    def invoke(*args: str, timeout: float) -> str:
        calls.append(timeout)
        if len(calls) < 3:
            raise supervisor.Refused("HTTP 503 temporarily unavailable")
        return "ok"

    monkeypatch.setattr(cli, "_call_once", invoke)
    assert cli.call(command) == "ok"
    assert calls == [120, 118, 113]
    assert clock[0] == 7


@pytest.mark.parametrize("command", ["new", "exec", "stop"])
def test_mutating_dispatch_is_never_blindly_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "state.json")
    calls = []

    def invoke(*args: str, timeout: float) -> str:
        calls.append(args)
        raise supervisor.Refused("HTTP 503 temporarily unavailable")

    monkeypatch.setattr(cli, "_call_once", invoke)
    with pytest.raises(supervisor.Refused):
        cli.call(command)
    assert len(calls) == 1


def test_sessions_respects_expired_job_caps_but_cleanup_can_observe_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "state.json", "owned")
    cli.deadline = 1.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 2.0)

    def invoke(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] > 0
        return subprocess.CompletedProcess(args, 0, "No active sessions found on server.", "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    with pytest.raises(supervisor.Refused, match="deadline"):
        cli.sessions()
    supervisor.stop_verified(cli, "owned-endpoint")
    assert cli.cleanup_mode is False


def test_contents_lease_runs_while_kernel_rpc_is_blocked_and_quiesces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    cli = supervisor.CLI("fake", tmp_path / "state.json")
    renewed = threading.Event()
    uploads = []

    def invoke(*args: str, timeout: float) -> str:
        assert args[0] == "upload"
        lease = json.loads(Path(args[-2]).read_text())
        uploads.append((lease, timeout))
        renewed.set()
        return "ok"

    monkeypatch.setattr(cli, "call", invoke)
    cli.start_lease("/content/snake-cli/owned", "generation", 180)
    assert renewed.wait(2)
    cli.stop_lease(wait=True)
    assert cli._lease_thread is not None and not cli._lease_thread.is_alive()
    assert uploads[0][0]["generation"] == "generation"
    assert uploads[0][1] <= 60


def test_lease_ignores_unit_deadline_and_keeps_wall_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "state.json")
    cli.priced_deadline = 1.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 2.0)
    assert cli.remaining_seconds(120) == 120
    cli.deadline = 1.0
    with pytest.raises(supervisor.Refused, match="deadline"):
        cli.remaining_seconds(120)


def test_export_sidecar_transient_failure_uses_contents_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: None)
    original = supervisor.CLI.call
    failures = []

    def invoke(self, *args: str, timeout: float = 120) -> str:
        if args[0] == "download" and args[-2].endswith(".sha256") and not failures:
            failures.append(args[-2])
            raise supervisor.Refused("HTTP 503 temporarily unavailable")
        return original(self, *args, timeout=timeout)

    monkeypatch.setattr(supervisor.CLI, "call", invoke)
    manifest = _manifest(tmp_path)
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((Path(manifest["output_dir"]) / "receipt.json").read_text())
    assert failures
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True


def test_calibration_allows_six_healthy_twenty_second_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    clock = [0.0]
    reads = []
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    original = supervisor.CLI.call

    def delayed(self, *args: str, timeout: float = 120) -> str:
        result = original(self, *args, timeout=timeout)
        if args[0] in {"sessions", "usage"} and hasattr(self, "accounting") and len(reads) < 6:
            assert timeout >= 20
            clock[0] += 20
            reads.append(args[0])
        return result

    monkeypatch.setattr(supervisor.CLI, "call", delayed)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 2
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((Path(manifest["output_dir"]) / "receipt.json").read_text())
    assert len(reads) == 6
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    assert receipt["accounting"]["calibration"]["samples"][-1]["elapsed_seconds"] >= 130


def test_poll_rpc_completion_grace_cannot_extend_per_attempt_budget(
    monkeypatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    cli.deadline = 500
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100)

    def invoke(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 120  # Not the ordinary mutation RPC's 130s outer grace.
        code = Path(argv[argv.index("-f") + 1]).read_text()
        nonce = "SNAKE_RPC_" + code.split("SNAKE_RPC_")[1][:32]
        return subprocess.CompletedProcess(argv, 0, nonce + '{"ok":true,"value":true}', "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    assert cli.poll_rpc("result=True") is True
    assert cli.deadline == 500


@pytest.mark.parametrize("fault", ["", "mismatch", "timeout"])
def test_public_release_broker_delivery_and_verified_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    """The fake provider runs pinned fetching; the real broker proves disappearance."""
    import io

    from research.colab_cli import inputs

    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    payload = b"release-input"
    manifest = _manifest(tmp_path)
    manifest["inputs"] = [
        {
            "url": "https://example.test/release/input.bin",
            "dest": "input.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    ]
    monkeypatch.setenv("FAKE_COLAB_PUBLIC_INPUT_BYTES", base64.b64encode(payload).decode())
    monkeypatch.setenv("FAKE_COLAB_PUBLIC_INPUT_FAULT", fault)

    def mac_preflight(row):
        return inputs.preflight(row["inputs"], opener=lambda *_args, **_kwargs: io.BytesIO(payload))

    monkeypatch.setattr(supervisor, "input_preflight", mac_preflight)
    if fault:
        with pytest.raises(supervisor.Refused, match="INCOMPLETE"):
            supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    else:
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert receipt["status"] == ("INCOMPLETE" if fault else "PASS")
    assert receipt["failure_class"] == ("CHILD_NONZERO_EXIT" if fault else None)
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    state = json.loads((tmp_path / "fake-state.json").read_text())
    if fault:
        assert not state.get("trainer_starts")
        assert receipt["exports"] == []
        failures = list(state["public_input_failures"].values())
        assert len(failures) == 1
        assert ("mismatch" if fault == "mismatch" else "timeout") in failures[0]
    else:
        assert state.get("trainer_starts") == [receipt["session"]]
        root = (
            tmp_path / "remote" / receipt["session"] / "content/snake-cli" / receipt["generation"]
        )
        assert (root / "inputs/input.bin").read_bytes() == payload
    _assert_transport(tmp_path)


def _write_broker_other_reservation(tmp_path: Path) -> None:
    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    (supervisor.ACCOUNT_ROOT / "active-other-lane.json").write_text(
        json.dumps(
            {
                "session": "other-lane",
                "endpoint": "gpu-notebook-1",
                "unit_cap": 1.0,
                "reserve_units": 0.0,
                "rate_units_per_hour": 6.84,
                "priced_rate_units_per_hour": 6.84,
                "observed_rate_units_per_hour": 6.84,
            }
        )
    )


@pytest.mark.parametrize(
    "kind",
    [
        "departure_usage_lags",
        "arrival_usage_lags",
        "departure_sessions_lag",
        "arrival_sessions_lag",
    ],
)
def test_fake_provider_membership_lag_warns_for_priced_broker_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """The subprocess fake reports usage first and sessions second at the poll boundary."""
    departure = kind.startswith("departure")
    binary = _fake_env(monkeypatch, tmp_path, assignments=int(departure))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    if departure:
        _write_broker_other_reservation(tmp_path)

    def observer(event: str, _context: dict[str, object]) -> None:
        if event != "ready":
            return
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        if departure:
            if kind == "departure_usage_lags":
                del state["sessions"]["gpu-notebook-1"]
                monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[2]")
            else:
                state["rate"] = 6.84
                monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[1]")
            (supervisor.ACCOUNT_ROOT / "active-other-lane.json").unlink()
        else:
            if kind == "arrival_usage_lags":
                state["sessions"]["broker-new"] = {"name": "other-lane", "gpu": "CPU"}
                monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[1]")
            else:
                state["rate"] = 13.68
                monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[2]")
            (supervisor.ACCOUNT_ROOT / "active-other-lane.json").write_text(
                json.dumps(
                    {
                        "session": "other-lane",
                        "endpoint": "broker-new",
                        "unit_cap": 1.0,
                        "reserve_units": 0.0,
                        "rate_units_per_hour": 6.84,
                        "priced_rate_units_per_hour": 6.84,
                        "observed_rate_units_per_hour": 6.84,
                    }
                )
            )
        state_path.write_text(json.dumps(state))

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    spend = json.loads((tmp_path / "job-output/spend-000000.json").read_text())
    assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"
    assert spend["provider_transition_mismatch_polls"] == 1
    _assert_transport(tmp_path)


def test_fake_provider_unknown_endpoint_refuses_before_count_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", "[1]")
    manifest = _manifest(tmp_path)

    def observer(event: str, _context: dict[str, object]) -> None:
        if event == "ready":
            state_path = tmp_path / "fake-state.json"
            state = json.loads(state_path.read_text())
            state["sessions"]["unknown"] = {"name": "foreign", "gpu": "CPU"}
            state_path.write_text(json.dumps(state))

    with pytest.raises(supervisor.Refused, match="unaccounted active assignment"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    _assert_transport(tmp_path)


def test_fake_provider_persistent_departure_lag_fails_on_third_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=1)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("FAKE_COLAB_USAGE_ASSIGNMENTS", json.dumps([2] * 4 + [2, 2, 2]))
    _write_broker_other_reservation(tmp_path)
    manifest = _manifest(tmp_path)
    original_poll = supervisor.CLI.poll_rpc
    reads = 0
    ready = False

    def delayed_terminal(self: supervisor.CLI, code: str, timeout: float = 30) -> object:
        nonlocal reads
        if ready and "read_state" in code:
            reads += 1
            # Accounting polls every fourth telemetry read, so retain eight
            # nonterminal states to exercise its third provider boundary.
            if reads < 9:
                return {"state": {"status": "running"}, "terminal": None, "exports": []}
        return original_poll(self, code, timeout)

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", delayed_terminal)

    def observer(event: str, _context: dict[str, object]) -> None:
        nonlocal ready
        if event == "ready":
            state_path = tmp_path / "fake-state.json"
            state = json.loads(state_path.read_text())
            del state["sessions"]["gpu-notebook-1"]
            state_path.write_text(json.dumps(state))
            (supervisor.ACCOUNT_ROOT / "active-other-lane.json").unlink()
            ready = True

    with pytest.raises(supervisor.Refused, match="count disagrees"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    assert reads == 8
    warnings = [
        json.loads((tmp_path / "job-output" / f"spend-{poll:06d}.json").read_text())[
            "provider_transition_mismatch_polls"
        ]
        for poll in (0, 4)
    ]
    assert warnings == [1, 2]
    assert not (tmp_path / "job-output/spend-000008.json").exists()
    _assert_transport(tmp_path)


@pytest.mark.parametrize("outcome", ["persists", "disappears", "relisted"])
def test_fake_provider_broker_departed_endpoint_carryover_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=1)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    _write_broker_other_reservation(tmp_path)
    manifest = _manifest(tmp_path)
    original_poll = supervisor.CLI.poll_rpc
    reads = 0
    ready = False

    def delayed_terminal(self: supervisor.CLI, code: str, timeout: float = 30) -> object:
        nonlocal reads
        if ready and "read_state" in code:
            reads += 1
            if reads == 8:
                # Resolve membership after two warnings, before the third
                # accounting poll; leave provider rate/count consistent.
                if outcome == "disappears":
                    state_path = tmp_path / "fake-state.json"
                    state = json.loads(state_path.read_text())
                    del state["sessions"]["gpu-notebook-1"]
                    state["rate"] -= 6.84
                    state_path.write_text(json.dumps(state))
                elif outcome == "relisted":
                    _write_broker_other_reservation(tmp_path)
            if reads < 9:
                return {"state": {"status": "running"}, "terminal": None, "exports": []}
        return original_poll(self, code, timeout)

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", delayed_terminal)

    def observer(event: str, _context: dict[str, object]) -> None:
        nonlocal ready
        if event == "ready":
            (supervisor.ACCOUNT_ROOT / "active-other-lane.json").unlink()
            ready = True

    broker = supervisor.Broker(tmp_path / "lanes", binary, observer=observer)
    if outcome == "persists":
        with pytest.raises(
            supervisor.Refused, match="unaccounted active assignment.*gpu-notebook-1"
        ):
            broker.run(manifest)
        assert reads == 8
        assert not (tmp_path / "job-output/spend-000008.json").exists()
    else:
        broker.run(manifest)
        receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
        assert receipt["stop_verified"] is True
        settled = json.loads((tmp_path / "job-output/spend-000008.json").read_text())
        assert settled["provider_transition_status"] == "OK"
        assert settled["provider_transition_mismatch_polls"] == 0
    for poll, expected in ((0, 1), (4, 2)):
        spend = json.loads((tmp_path / "job-output" / f"spend-{poll:06d}.json").read_text())
        assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"
        assert spend["provider_transition_mismatch_polls"] == expected
    _assert_transport(tmp_path)


@pytest.mark.parametrize("balance", [-1, 0])
def test_depleted_balance_warns_and_provider_allows_allocation(
    monkeypatch, tmp_path, capsys, balance
):
    binary = _fake_env(monkeypatch, tmp_path)
    state_path = tmp_path / "fake-state.json"
    state = json.loads(state_path.read_text())
    state["balance"] = balance
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    supervisor.Broker(tmp_path / "lanes", binary).run(_manifest(tmp_path))
    assert any(event["argv"][4] == "new" for event in _events(tmp_path))
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    assert '"WARNING": "BALANCE_DEPLETED"' in capsys.readouterr().err


@pytest.mark.parametrize("rates", [[0.0], [0.0, 0.5, 1.0] * 12])
def test_unsettled_calibration_uses_only_priced_informational_rate(
    monkeypatch, tmp_path, capsys, rates
):
    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    monkeypatch.setattr(supervisor, "RATE_CALIBRATION_SECONDS", 30)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps(rates))
    records = []

    def observer(event, context):
        if event == "runtime_started":
            records.append(
                json.loads(next(supervisor.ACCOUNT_ROOT.glob("active-*.json")).read_text())
            )

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    assert receipt["accounting"]["observed_rate_units_per_hour"] is None
    assert receipt["accounting"]["own_rate_units_per_hour"] == 2.0
    assert receipt["accounting"]["rate_measurement_unavailable"] is True
    assert records[0]["observed_rate_units_per_hour"] is None
    assert '"WARNING": "CALIBRATION_RATE_UNSTABLE"' in capsys.readouterr().err


def test_zero_and_mid_lag_rates_do_not_publish_before_settling(monkeypatch, tmp_path):
    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps([0, 0, 0, 0.5, 1, 1.54, 1.54, 1.54]))
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS" and receipt["stop_verified"]
    calibration = receipt["accounting"]["calibration"]
    assert calibration["status"] == "PASS"
    assert calibration["observed_rate_units_per_hour"] == 1.54
    assert len(calibration["samples"]) >= 6
    assert [x["usage"]["rate"] for x in calibration["samples"][-3:]] == [1.54] * 3


def test_retention_units_cli_default_is_unset(monkeypatch, tmp_path):
    manifest = tmp_path / "chain.json"
    manifest.write_text("{}")
    seen = []
    monkeypatch.setattr(
        supervisor.Broker, "chain", lambda self, jobs, minutes, units: seen.append(units)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervisor",
            "--state-dir",
            str(tmp_path / "state"),
            "chain",
            str(manifest),
            "--retention-minutes",
            "30",
        ],
    )
    supervisor.main()
    assert seen == [None]


def test_retained_chain_without_measured_rates_uses_informational_estimate(
    monkeypatch, tmp_path, capsys
):
    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    monkeypatch.setattr(supervisor, "RATE_CALIBRATION_SECONDS", 30)
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", "[0.0]")
    first = _manifest(tmp_path, job_id="chain-a", output=tmp_path / "chain-a", gpu="H100", rate=2)
    second = _manifest(tmp_path, job_id="chain-b", output=tmp_path / "chain-b", gpu="H100", rate=2)
    supervisor.Broker(tmp_path / "lanes", binary).chain([first, second], 10, None)
    for name in ("chain-a", "chain-b"):
        receipt = json.loads((tmp_path / name / "receipt.json").read_text())
        assert receipt["status"] == "COMPLETED_SESSION_RETAINED"
        assert receipt["accounting"]["observed_rate_units_per_hour"] is None
        assert receipt["accounting"]["accounting_rate_basis"] == "priced_informational"
        assert (
            json.loads((tmp_path / name / "session-release.json").read_text())["stop_verified"]
            is True
        )
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    assert '"WARNING": "RETENTION_UNIT_THRESHOLD"' not in capsys.readouterr().err
    assert sum(e["argv"][4] == "new" for e in _events(tmp_path)) == 1
