"""Offline D1024 receipt stall replays; no provider commands or fit payloads."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, cli_guard, policy, supervisor, transfers
from tests import test_colab_cli_fault_properties as harness

_isolate_account = harness._isolate_account
pinned_repo = harness.pinned_repo


@pytest.mark.parametrize(
    "receipt,selectors,latencies",
    [
        ("01", ["SNAKE_STAGING_PARENTS", "SNAKE_TRANSFER_CREATE_DIR"], []),
        (
            "02",
            ["synthetic-0.bin", "synthetic-1.bin"],
            [("/bootstrap.sh", 35.463554625), ("00000000.part", 9.4125765)],
        ),
    ],
)
def test_d1024_two_control_stalls_share_staging_deadline_on_fake_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_repo: Path,
    _isolate_account: harness._TransportClock,
    receipt: str,
    selectors: list[str],
    latencies: list[tuple[str, float]],
) -> None:
    """Both receipts have two 120s exec stalls plus slow Contents writes.

    01's two pre-transfer stalls are located at parent setup by control-flow
    reconstruction; 02's exact exec bodies are unrecorded. Replay 02 at the
    durable chunk-assembly boundary, including completed-but-unacknowledged
    mutation recovery. This deliberately tests the stronger safety case,
    rather than claiming those untagged historical bodies are identifiable.
    """
    schedule = [
        {
            "call": "upload",
            "session_contains": "-l4-",
            "contains": text,
            "fault": "latency",
            "latency_seconds": seconds,
        }
        for text, seconds in latencies
    ] + [
        {
            "call": "exec",
            "session_contains": "-l4-",
            "contains": text,
            "fault": "timeout",
            "phase": "post",
        }
        for text in selectors
    ]
    binary = harness._fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": int(receipt),
            "max_latency_seconds": 120,
            "schedule": schedule,
        },
    )
    plan = harness._pinned_plan(tmp_path, pinned_repo, profile="final-only")
    # Enough to exercise two independently UUID-fenced chunk assemblies; all
    # bytes are synthetic, never copied from a scientific input.
    for job in plan["jobs"]:
        if job["name"] != "l4":
            continue
        manifest_path = Path(job["manifest"])
        manifest = json.loads(manifest_path.read_text())
        for i in range(2):
            path = tmp_path / f"synthetic-{i}.bin"
            with path.open("wb") as handle:
                handle.truncate(transfers.UPLOAD_CHUNK_BYTES + 1)
            manifest["inputs"].append(
                {
                    "path": str(path),
                    "dest": path.name,
                    "sha256": supervisor.sha256(path),
                }
            )
        manifest_path.write_text(json.dumps(manifest))
        job["manifest_sha256"] = supervisor.sha256(manifest_path)

    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(harness.HELPER),
    )
    harness._assert_safe(tmp_path, plan)
    run = Path(plan["jobs"][0]["manifest"]).parent.parent
    r = json.loads((run / "outputs/l4/receipt.json").read_text())
    faults = [
        json.loads(line).get("fault")
        for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()
    ]
    stalls = [f for f in faults if f and f["fault"] == "timeout"]
    assert len(stalls) == 2
    assert [f["virtual_latency_seconds"] for f in stalls] == [30, 30]
    assert r["stop_verified"] is True and r["cleanup_error"] is None
    assert rc == 0
    measurement = json.loads((run / "outputs/l4/upload-measurement.json").read_text())
    assert 0 < measurement["elapsed_seconds"] <= 300
    assert 0 < measurement["transfer_seconds"] <= 150
    operations = r["control_operations"]
    assert {"mkdir", "verify", "PUT", "recovery-idle"} <= {o["operation"] for o in operations}
    for op in operations:
        assert op["ended_monotonic"] >= op["started_monotonic"]
        assert op["elapsed_seconds"] == op["ended_monotonic"] - op["started_monotonic"]
    assert all(
        c["timeout_seconds"] <= 30
        for c in r["cli_calls"]
        if c["command"] == "exec" and c["status"] == "failed"
    )


@pytest.mark.parametrize("remaining", [0, 7, 59])
def test_staging_controls_refuse_expiry_and_clip_all_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remaining: int,
) -> None:
    now = [100.0]
    calls = []
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")

    def run(argv: list[str], **kwargs: Any) -> None:
        timeout = kwargs["timeout"]
        calls.append(timeout)
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))
    monkeypatch.setattr(supervisor.subprocess, "run", run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *a, **k: True)
    with pytest.raises((supervisor.Refused, subprocess.TimeoutExpired)):
        with cli.staging_controls(now[0] + remaining):
            cli.poll_rpc("result=True")
    assert now[0] <= 100 + remaining
    assert all(0 < t <= policy.STAGING_CONTROL_SECONDS for t in calls)
    if remaining == 0:
        assert calls == [] and not cli.exec_uncertain
    else:
        assert cli.exec_uncertain
    assert cli.deadline is None


def test_staging_foreground_does_not_take_heartbeat_or_stop_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    cli.deadline = 1000
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    allowances = []
    with cli.staging_controls(101):
        assert cli.remaining_seconds(60) == 1
        worker = threading.Thread(target=lambda: allowances.append(cli.remaining_seconds(60)))
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert allowances == [60]
    assert cli.remaining_seconds(60) == 60


@pytest.mark.parametrize("failure", ["corrupt", "missing", "source-mutated", "lease-loss"])
def test_upload_fault_never_atomically_publishes_unverified_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Execute the production assembly code against a fake remote filesystem."""
    import re

    from tests.test_colab_cli_transfers import FakeCLI

    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 32)
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable" * 16)

    class FaultCLI(FakeCLI):
        def call(self, *args: str, timeout: float = 45) -> str:
            result = super().call(*args, timeout=timeout)
            if args[0] == "upload" and args[-1].endswith("00000000.part"):
                if failure == "corrupt":
                    self._path(args[-1]).write_bytes(b"corrupt")
                elif failure == "missing":
                    self._path(args[-1]).unlink()
                elif failure == "source-mutated":
                    source.write_bytes(b"different" * 16)
            return result

        def rpc(self, code: str, timeout: float = 20) -> Any:
            self.rpc_codes.append(code)
            match = re.search(r"spec = (\{[^\n]+\})", code)
            spec = json.loads(match[1])
            for key in ("parts", "temporary", "target", "source", "path"):
                if key in spec:
                    spec[key] = str(self._path(spec[key]))
            code = code[: match.start(1)] + repr(spec) + code[match.end(1) :]
            namespace: dict[str, Any] = {}
            exec(compile(code, "<fake-assembly>", "exec"), namespace)
            return namespace["result"]

    cli = FaultCLI(tmp_path / "remote")

    def renew() -> None:
        if failure == "lease-loss":
            raise supervisor.Refused("generation fence mismatch; lease was not renewed")

    with pytest.raises(transfers.TransferError):
        transfers.upload_file(cli, source, "/content/target.bin", renew)
    assert not cli._path("/content/target.bin").exists()
    assert not list((tmp_path / "remote").rglob("*.part"))
    assert not list((tmp_path / "remote").rglob("*.assembling"))
