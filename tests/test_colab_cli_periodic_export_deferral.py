"""Hermetic tests for bounded mid-run periodic export deferral (2026-11-12)."""

from __future__ import annotations

import errno
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import cli_guard, policy, supervisor, transfers
from tests.test_colab_cli_supervisor import (
    _fake_env,
    _isolate_account,
    _manifest,
    _patch_virtual_clock,
)
from tests.test_colab_cli_transport_fix_1112 import (
    _exec_wrapped_snippet,
    _owned_cli,
    _sessions_stdout,
)

__all__ = ["_isolate_account"]


def _write_remote_export(
    remote_output: Path, rel: str, payload: bytes, *, sidecar_digest: str | None = None
) -> dict[str, Any]:
    """Write a complete remote export triple and return its inventory entry."""
    target = remote_output / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    target.write_bytes(payload)
    Path(str(target) + ".sha256").write_text(f"{sidecar_digest or digest}  {Path(rel).name}\n")
    Path(str(target) + ".complete").write_text("complete\n")
    return {"path": rel, "size": len(payload), "sha256": digest}


def _deferred_warnings(stderr_text: str) -> list[dict[str, Any]]:
    """Extract parsed one-line PERIODIC_EXPORT_DEFERRED JSON warnings from stderr."""
    warnings: list[dict[str, Any]] = []
    for line in stderr_text.splitlines():
        line_stripped = line.strip()
        if not line_stripped.startswith("{"):
            continue
        parsed = json.loads(line_stripped)
        if parsed.get("WARNING") == "PERIODIC_EXPORT_DEFERRED":
            assert "\n" not in line
            warnings.append(parsed)
    return warnings


def _assert_no_temp_or_partial(output: Path) -> None:
    """Ensure no temp directories or partial files exist under output."""
    if not output.exists():
        return
    leftover_temps = [
        p
        for p in output.rglob("*")
        if p.name.startswith((".transfer-", ".batch-download-", ".batch-export-"))
    ]
    assert leftover_temps == []


def test_broker_periodic_export_timeout_defers_then_collects_and_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reproduce v3-ppo-train-a100-rwamc-d-002: cycle 1 times out, cycle 2 collects and passes."""
    binary = _fake_env(monkeypatch, tmp_path)
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 30.0
    manifest["budget"]["unit_cap"] = 10.0

    meta_rel = "checkpoints/light_u0002004.meta.json"
    meta_bytes = b'{"updates_completed": 2004}\n'
    ckpt_rel = "checkpoints/light_u0002004.pt"
    ckpt_bytes = b"W" * (transfers.DOWNLOAD_CHUNK_BYTES + 4096)

    ready_seen = [False]
    poll_count = [0]
    split_timeouts = [0]
    chunk_timed_out = [False]
    cycle1_snapshot: dict[str, Any] = {}
    inventory: list[dict[str, Any]] = []

    original_run = supervisor.subprocess.run
    original_poll = supervisor.CLI.poll_rpc

    def controlled_poll_rpc(
        self: supervisor.CLI, code: str, timeout: float | None = None, *, mutating: bool = False
    ) -> Any:
        result = original_poll(self, code, timeout=timeout, mutating=mutating)
        if ready_seen[0] and "read_state" in code:
            poll_count[0] += 1
            if poll_count[0] == 2:
                # Inspect state right after poll 1's deferred periodic collection + 15s sleep.
                exports_dir = Path(str(manifest["output_dir"])) / "exports"
                cycle1_snapshot["ckpt_exists"] = (exports_dir / ckpt_rel).exists()
                cycle1_snapshot["ckpt_sha_exists"] = (exports_dir / (ckpt_rel + ".sha256")).exists()
                cycle1_snapshot["ckpt_complete_exists"] = (
                    exports_dir / (ckpt_rel + ".complete")
                ).exists()
                cycle1_snapshot["meta_exists"] = (exports_dir / meta_rel).is_file()
                cycle1_snapshot["elapsed_after_cycle1"] = now[0]
                _assert_no_temp_or_partial(Path(str(manifest["output_dir"])))
            if poll_count[0] in {1, 2}:
                return {
                    "state": {"status": "running"},
                    "terminal": None,
                    "exports": [dict(item) for item in inventory],
                    "job_log_tail": f"step={poll_count[0]}\n",
                }
        return result

    def fault_injected_subprocess(
        argv: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if ready_seen[0] and poll_count[0] == 1 and len(argv) > 5:
            cmd = argv[5]
            timeout = float(kwargs.get("timeout") or 0.0)
            if cmd == "exec" and "-f" in argv:
                script = Path(argv[argv.index("-f") + 1]).read_text()
                if "SNAKE_TRANSFER_DOWNLOAD_SPLIT" in script and split_timeouts[0] < 2:
                    split_timeouts[0] += 1
                    now[0] += timeout
                    raise subprocess.TimeoutExpired(argv, timeout)
            if (
                cmd == "download"
                and split_timeouts[0] == 2
                and not chunk_timed_out[0]
                and ".part" in argv[-2]
            ):
                # Consume the remaining per-file download cap on chunk 0 so retry hits cap <= 0.
                chunk_timed_out[0] = True
                now[0] += timeout
                raise subprocess.TimeoutExpired(argv, timeout)
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", controlled_poll_rpc)
    monkeypatch.setattr(supervisor.subprocess, "run", fault_injected_subprocess)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_kw: True)

    def observer(event: str, context: dict[str, Any]) -> None:
        if event != "ready":
            return
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        session = context["cli"].session
        remote = state["runtime"][session]
        remote_output = tmp_path / "remote" / session / remote["root"].lstrip("/") / "output"
        inventory.extend(
            [
                _write_remote_export(remote_output, meta_rel, meta_bytes),
                _write_remote_export(remote_output, ckpt_rel, ckpt_bytes),
            ]
        )
        remote["terminal"]["exports"] = [dict(item) for item in inventory]
        state_path.write_text(json.dumps(state))
        ready_seen[0] = True

    output_dir = Path(str(manifest["output_dir"]))
    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    captured = capsys.readouterr()
    deferred = _deferred_warnings(captured.err)
    assert len(deferred) == 1
    assert deferred[0]["path"] == ckpt_rel
    assert deferred[0]["size"] == len(ckpt_bytes)
    assert deferred[0]["attempt"] == 1
    assert deferred[0]["max_attempts"] == policy.MAX_PERIODIC_EXPORT_ATTEMPTS
    assert "total download time cap reached" in deferred[0]["reason"]

    # Cycle 1 left no partial checkpoint, published meta, and charged ~300s to wall clock.
    assert cycle1_snapshot["meta_exists"] is True
    assert cycle1_snapshot["ckpt_exists"] is False
    assert cycle1_snapshot["ckpt_sha_exists"] is False
    assert cycle1_snapshot["ckpt_complete_exists"] is False
    assert cycle1_snapshot["elapsed_after_cycle1"] >= 300.0

    # Cycle 2 collected the checkpoint with valid sidecars; poll 3 completed with PASS.
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    assert receipt["elapsed_seconds"] >= 315.0
    assert {row["path"] for row in receipt["exports"]} == {meta_rel, ckpt_rel}

    published_ckpt = output_dir / "exports" / ckpt_rel
    assert published_ckpt.read_bytes() == ckpt_bytes
    expected_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    assert (
        Path(str(published_ckpt) + ".sha256").read_text(encoding="utf-8").split()[0] == expected_sha
    )
    assert Path(str(published_ckpt) + ".complete").is_file()
    _assert_no_temp_or_partial(output_dir)


@pytest.mark.parametrize(
    "violation",
    [
        "payload_sha_mismatch",
        "sidecar_sha_mismatch",
        "prefetched_local_mutated",
        "remote_meta_changed",
    ],
)
def test_periodic_integrity_violations_refuse_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    violation: str,
) -> None:
    """Digest/sidecar mismatches, prefetched tampering, and changed metadata refuse at once."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-integrity"
    remote_out = remote / root.lstrip("/") / "output"
    rel = "checkpoints/light_u0002004.pt"
    payload = b"immutable-checkpoint-bytes"
    bad_sidecar = "f" * 64 if violation == "sidecar_sha_mismatch" else None
    entry = _write_remote_export(remote_out, rel, payload, sidecar_digest=bad_sidecar)

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            data = src.read_bytes()
            if violation == "payload_sha_mismatch" and src.name == Path(rel).name:
                data = b"corrupted-" + data[10:]
            Path(argv[-1]).write_bytes(data)
            return subprocess.CompletedProcess(argv, 0, "", "")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    if violation == "remote_meta_changed":
        supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
        published = out_dir / "exports" / rel
        assert published.read_bytes() == payload
        with pytest.raises(supervisor.Refused, match="immutable snapshot metadata changed"):
            supervisor.collect_exports(
                cli,
                root,
                [{**entry, "sha256": "0" * 64}],
                out_dir,
                limit=1024 * 1024,
                periodic=True,
            )
    elif violation == "prefetched_local_mutated":
        prefetched = out_dir / "exports" / rel
        prefetched.parent.mkdir(parents=True, exist_ok=True)
        prefetched.write_bytes(b"locally-tampered-bytes")
        Path(str(prefetched) + ".sha256").write_text(f"{entry['sha256']}  {Path(rel).name}\n")
        Path(str(prefetched) + ".complete").write_text("complete\n")
        with pytest.raises(supervisor.Refused, match="immutable export changed"):
            supervisor.collect_exports(
                cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True
            )
    else:
        expected_match = (
            "download hash mismatch or transfer failure"
            if violation == "payload_sha_mismatch"
            else "download digest sidecar mismatch"
        )
        with pytest.raises(supervisor.Refused, match=expected_match):
            supervisor.collect_exports(
                cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True
            )
        assert not (out_dir / "exports" / rel).exists()
        assert not (out_dir / "exports" / (rel + ".complete")).exists()
        _assert_no_temp_or_partial(out_dir)

    assert _deferred_warnings(capsys.readouterr().err) == []


def test_broker_corrupt_midrun_download_refuses_immediately_with_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In Broker.run, a mid-run sha256 mismatch refuses on cycle 1 without deferring."""
    binary = _fake_env(monkeypatch, tmp_path, mode="corrupt_download")
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    manifest = _manifest(tmp_path)
    ready_seen = [False]
    original_poll = supervisor.CLI.poll_rpc

    def midrun_poll(
        self: supervisor.CLI, code: str, timeout: float | None = None, *, mutating: bool = False
    ) -> Any:
        result = original_poll(self, code, timeout=timeout, mutating=mutating)
        if ready_seen[0] and "read_state" in code:
            return {
                "state": {"status": "running"},
                "terminal": None,
                "exports": result["terminal"]["exports"],
            }
        return result

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", midrun_poll)

    def observer(event: str, _context: dict[str, Any]) -> None:
        if event == "ready":
            ready_seen[0] = True

    with pytest.raises(supervisor.Refused, match="download hash mismatch"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    assert _deferred_warnings(capsys.readouterr().err) == []
    output_dir = Path(str(manifest["output_dir"]))
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"] is True
    assert receipt["exports"] == []
    _assert_no_temp_or_partial(output_dir)


def test_terminal_collection_timeout_still_produces_incomplete_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal export collection never defers: a failed download produces INCOMPLETE."""
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 30.0
    manifest["budget"]["unit_cap"] = 10.0

    ready_seen = [False]
    polls = [0]
    original_poll = supervisor.CLI.poll_rpc
    original_run = supervisor.subprocess.run

    def running_then_terminal(
        self: supervisor.CLI, code: str, timeout: float | None = None, *, mutating: bool = False
    ) -> Any:
        result = original_poll(self, code, timeout=timeout, mutating=mutating)
        if ready_seen[0] and "read_state" in code:
            polls[0] += 1
            if polls[0] == 1:
                return {
                    "state": {"status": "running"},
                    "terminal": None,
                    "exports": result["terminal"]["exports"],
                }
        return result

    def always_timeout_checkpoint_download(
        argv: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if (
            ready_seen[0]
            and len(argv) > 5
            and argv[5] == "download"
            and "checkpoint.bin" in argv[-2]
        ):
            timeout = float(kwargs.get("timeout") or 0.0)
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", running_then_terminal)
    monkeypatch.setattr(supervisor.subprocess, "run", always_timeout_checkpoint_download)

    def observer(event: str, _context: dict[str, Any]) -> None:
        if event == "ready":
            ready_seen[0] = True

    with pytest.raises(supervisor.Refused, match="INCOMPLETE"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    # Poll 1 (mid-run) deferred once; Poll 2 (terminal) failed immediately without deferring.
    deferred = _deferred_warnings(capsys.readouterr().err)
    assert len(deferred) == 1
    assert deferred[0]["attempt"] == 1

    output_dir = Path(str(manifest["output_dir"]))
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["failure_class"] == "SUPERVISOR"
    assert receipt["stop_verified"] is True
    assert receipt["exports"] == []
    assert not (output_dir / "exports/checkpoint.bin").exists()
    _assert_no_temp_or_partial(output_dir)


def test_broker_periodic_deferral_bound_ends_job_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failing MAX_PERIODIC_EXPORT_ATTEMPTS cycles in a row ends the job as INCOMPLETE."""
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    now = [0.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 60.0
    manifest["budget"]["unit_cap"] = 20.0

    ready_seen = [False]
    polls = [0]
    original_poll = supervisor.CLI.poll_rpc
    original_run = supervisor.subprocess.run

    def always_running(
        self: supervisor.CLI, code: str, timeout: float | None = None, *, mutating: bool = False
    ) -> Any:
        result = original_poll(self, code, timeout=timeout, mutating=mutating)
        if ready_seen[0] and "read_state" in code:
            polls[0] += 1
            return {
                "state": {"status": "running"},
                "terminal": None,
                "exports": result["terminal"]["exports"],
            }
        return result

    def hung_download(
        argv: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if (
            ready_seen[0]
            and len(argv) > 5
            and argv[5] == "download"
            and "checkpoint.bin" in argv[-2]
        ):
            timeout = float(kwargs.get("timeout") or 0.0)
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(supervisor.CLI, "poll_rpc", always_running)
    monkeypatch.setattr(supervisor.subprocess, "run", hung_download)

    def observer(event: str, _context: dict[str, Any]) -> None:
        if event == "ready":
            ready_seen[0] = True

    with pytest.raises(supervisor.Refused, match="periodic export deferral bound reached"):
        supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    deferred = _deferred_warnings(capsys.readouterr().err)
    assert [row["attempt"] for row in deferred] == [1, 2, 3]
    assert polls[0] == policy.MAX_PERIODIC_EXPORT_ATTEMPTS

    output_dir = Path(str(manifest["output_dir"]))
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["failure_class"] == "SUPERVISOR"
    assert receipt["stop_verified"] is True
    assert receipt["elapsed_seconds"] >= policy.MAX_PERIODIC_EXPORT_ATTEMPTS * 300.0
    _assert_no_temp_or_partial(output_dir)


def test_periodic_stalled_cycles_bound_triggers_on_growing_remote_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dead link with newly prepended files triggers stalled_cycles bound on cycle 4."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 7200.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-stall"
    remote_out = remote / root.lstrip("/") / "output"

    def hung_download(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = float(kwargs["timeout"])
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.subprocess, "run", hung_download)
    out_dir = tmp_path / "out"
    entries: list[dict[str, Any]] = []

    # Prepend lexicographically smaller singletons (> 4 MiB so each is a singleton batch)
    # each cycle so no single path reaches 4 attempts, testing stalled_cycles directly.
    payload = b"P" * (4 * 1024 * 1024 + 17)
    for cycle, name in enumerate(["d.bin", "c.bin", "b.bin", "a.bin"], start=1):
        entries.insert(0, _write_remote_export(remote_out, name, payload))
        if cycle < policy.MAX_PERIODIC_EXPORT_ATTEMPTS:
            supervisor.collect_exports(
                cli, root, entries, out_dir, limit=16 * 1024 * 1024, periodic=True
            )
        else:
            with pytest.raises(
                supervisor.Refused,
                match="periodic export deferral bound reached for a.bin.*stalled_cycles 4",
            ):
                supervisor.collect_exports(
                    cli, root, entries, out_dir, limit=16 * 1024 * 1024, periodic=True
                )

    deferred = _deferred_warnings(capsys.readouterr().err)
    assert [row["path"] for row in deferred] == ["d.bin", "c.bin", "b.bin"]
    assert [row["stalled_cycles"] for row in deferred] == [1, 2, 3]
    _assert_no_temp_or_partial(out_dir)


def test_failed_periodic_download_counts_against_session_wall_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed periodic downloads consume cli.deadline and never defer once session cap expires."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 450.0  # Enough for 1 failed 300s try, but expires during the 2nd try!
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-wall"
    entry = _write_remote_export(remote / root.lstrip("/") / "output", "ckpt.bin", b"data")

    def hung_download(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = float(kwargs["timeout"])
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.subprocess, "run", hung_download)
    out_dir = tmp_path / "out"

    # Attempt 1 consumes 300s (leaving 150s on cli.deadline) and defers cleanly.
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
    assert now[0] == pytest.approx(300.0, abs=5.0)
    assert len(_deferred_warnings(capsys.readouterr().err)) == 1

    # Attempt 2 hits cli.deadline at t=450s; session deadline expiration must NOT defer.
    with pytest.raises(supervisor.Refused):
        supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
    assert now[0] == pytest.approx(450.0, abs=1.0)
    assert _deferred_warnings(capsys.readouterr().err) == []


def test_periodic_split_rpc_timeout_reconciles_exec_uncertain_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full-window DOWNLOAD_SPLIT timeout sets cleanup_uncertain_reason and recovers next cycle."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-split"
    rel = "checkpoints/light_u0002004.pt"
    payload = b"S" * (transfers.DOWNLOAD_CHUNK_BYTES + 123)
    entry = _write_remote_export(remote / root.lstrip("/") / "output", rel, payload)

    cycle = [1]
    idle_state = [False]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint or "ep"), ""
            )
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        if cycle[0] == 1:
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_kw: idle_state[0])
    out_dir = tmp_path / "out"

    # Cycle 1: DOWNLOAD_SPLIT times out across 300s while kernel stays busy at deferral exit.
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=32 * 1024 * 1024, periodic=True)
    assert len(_deferred_warnings(capsys.readouterr().err)) == 1
    assert cli.exec_uncertain is True
    assert cli.cleanup_uncertain_reason is not None
    _assert_no_temp_or_partial(out_dir)

    # Cycle 2: kernel becomes idle; even a mutating recoverable_rpc (DOWNLOAD_SPLIT) recovers
    # via recover_idle() instead of failing with "previous exec uncertain/busy".
    cycle[0] = 2
    idle_state[0] = True
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=32 * 1024 * 1024, periodic=True)
    assert cli.exec_uncertain is False
    assert cli.cleanup_uncertain_reason is None
    assert (out_dir / "exports" / rel).read_bytes() == payload
    assert (out_dir / "exports" / (rel + ".complete")).is_file()
    _assert_no_temp_or_partial(out_dir)


def test_periodic_small_export_batch_timeout_defers_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Multi-file small export_batch timeout mid-run defers cleanly and collects on cycle 2."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-batch"
    remote_out = remote / root.lstrip("/") / "output"
    entries = [
        _write_remote_export(remote_out, "metrics/u0001.jsonl", b'{"u":1}\n'),
        _write_remote_export(remote_out, "checkpoints/light_u0001.meta.json", b'{"u":1}'),
    ]

    fail_archive_download = [True]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "download":
            if fail_archive_download[0]:
                now[0] += timeout
                raise subprocess.TimeoutExpired(argv, timeout)
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        script_path = Path(argv[argv.index("-f") + 1])
        wrapped = script_path.read_text()
        local_root = str(remote / root.lstrip("/"))
        wrapped = wrapped.replace(
            "'/content/.snake-rpc-", repr(str(remote / "content/.snake-rpc-"))[:-1]
        )
        (remote / "content").mkdir(parents=True, exist_ok=True)
        wrapped = wrapped.replace(root, local_root)
        wrapped = wrapped.replace(
            "'path': str(archive)",
            f"'path': str(archive).replace({local_root!r}, {root!r}, 1)",
        )
        script_path.write_text(wrapped)
        _, proc = _exec_wrapped_snippet(argv, remote_root=None)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    # Cycle 1: archive download times out -> both entries deferred, no temp dirs left.
    supervisor.collect_exports(cli, root, entries, out_dir, limit=1024 * 1024, periodic=True)
    deferred = _deferred_warnings(capsys.readouterr().err)
    assert [row["path"] for row in deferred] == [
        "metrics/u0001.jsonl",
        "checkpoints/light_u0001.meta.json",
    ]
    assert cli.verified_exports == {}
    _assert_no_temp_or_partial(out_dir)

    # Cycle 2: archive download succeeds -> both entries verified with sidecars,
    # and cycle 1's failed .snake-export-batch-*.zip was swept inline.
    fail_archive_download[0] = False
    supervisor.collect_exports(cli, root, entries, out_dir, limit=1024 * 1024, periodic=True)
    assert set(cli.verified_exports) == {
        "metrics/u0001.jsonl",
        "checkpoints/light_u0001.meta.json",
    }
    remaining_archives = list((remote / root.lstrip("/")).glob(".snake-export-batch-*.zip"))
    assert len(remaining_archives) == 1
    assert cli._deferred_remote_cleanups == {}
    assert (out_dir / "exports/metrics/u0001.jsonl").read_bytes() == b'{"u":1}\n'
    assert (out_dir / "exports/metrics/u0001.jsonl.complete").is_file()
    assert (out_dir / "exports/checkpoints/light_u0001.meta.json").read_bytes() == b'{"u":1}'
    assert (out_dir / "exports/checkpoints/light_u0001.meta.json.complete").is_file()
    _assert_no_temp_or_partial(out_dir)


@pytest.mark.parametrize(
    ("phase", "stderr_text"),
    [
        (
            "download",
            (
                "Traceback (most recent call last):\n"
                '  File "/opt/colab/contents.py", line 403, in get_file\n'
                "    resp = self.session.get(url, timeout=30)\n"
                "requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: "
                "https://colab.research.google.com/api/contents\n"
            ),
        ),
        (
            "download",
            (
                "Traceback (most recent call last):\n"
                '  File "/opt/colab/contents.py", line 188, in get\n'
                "    return client.get(url, timeout=30.0)\n"
                "httpx.HTTPStatusError: Client error '403 Forbidden' for url 'https://example'\n"
            ),
        ),
        (
            "exec",
            (
                "Traceback (most recent call last):\n"
                '  File "/opt/colab/kernel.py", line 212, in execute\n'
                "    resp = requests.get(url, timeout=30)\n"
                "RuntimeError: HTTP 403 Forbidden\n"
            ),
        ),
        (
            "download",
            (
                "Traceback (most recent call last):\n"
                "  resp = self.session.get(url, timeout=30)\n"
                "RuntimeError: 503 Service Unavailable\n"
            ),
        ),
    ],
)
def test_periodic_403_or_non_timeout_with_timeout_kwarg_in_traceback_refuses_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    stderr_text: str,
) -> None:
    """403 (and non-timeout HTTP errors) with get(url, timeout=30) in traceback refuse at once."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-403"
    rel = "checkpoints/light_u0002004.pt"
    payload = (
        b"X" * (transfers.DOWNLOAD_CHUNK_BYTES + 64) if phase == "exec" else b"small-checkpoint"
    )
    entry = _write_remote_export(remote / root.lstrip("/") / "output", rel, payload)
    calls: list[str] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        calls.append(cmd)
        if cmd == phase:
            return subprocess.CompletedProcess(argv, 1, "", stderr_text)
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_kw: True)
    out_dir = tmp_path / "out"

    with pytest.raises(supervisor.Refused):
        supervisor.collect_exports(
            cli, root, [entry], out_dir, limit=32 * 1024 * 1024, periodic=True
        )

    assert _deferred_warnings(capsys.readouterr().err) == []
    if "403" in stderr_text:
        assert calls.count(phase) == 1
    assert cli.verified_exports == {}
    _assert_no_temp_or_partial(out_dir)


@pytest.mark.parametrize(
    "local_error",
    [
        TimeoutError(errno.ETIMEDOUT, "local disk operation timed out"),
        OSError(errno.EIO, "local filesystem input/output error"),
        PermissionError(errno.EACCES, "local export permission denied"),
    ],
)
def test_periodic_local_oserror_and_disk_timeout_refuse_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    local_error: OSError,
) -> None:
    """Local OSError (including TimeoutError subclass) refuses on cycle 1 without deferring."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-oserror"
    entry = _write_remote_export(remote / root.lstrip("/") / "output", "ckpt.bin", b"data")
    attempts = [0]

    def failing_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[5] == "download":
            attempts[0] += 1
            raise local_error
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", failing_run)
    out_dir = tmp_path / "out"

    with pytest.raises(supervisor.Refused, match="download hash mismatch or transfer failure"):
        supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)

    assert attempts[0] == 1
    assert _deferred_warnings(capsys.readouterr().err) == []
    _assert_no_temp_or_partial(out_dir)


def test_periodic_cycle_does_not_read_bytes_of_already_verified_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Already-verified exports are neither re-read nor re-hashed on later periodic polls."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-norehash"
    remote_out = remote / root.lstrip("/") / "output"
    first_rel = "checkpoints/light_u0001000.pt"
    second_rel = "checkpoints/light_u0002000.pt"
    terminal_rel = "checkpoints/final.pt"
    entry1 = _write_remote_export(remote_out, first_rel, b"first-checkpoint-bytes")
    entry2 = _write_remote_export(remote_out, second_rel, b"second-checkpoint-bytes")
    entry3 = _write_remote_export(remote_out, terminal_rel, b"terminal-checkpoint-bytes")

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[5] == "download":
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    # Poll 1: collect first_rel and verify it locally.
    supervisor.collect_exports(cli, root, [entry1], out_dir, limit=1024 * 1024, periodic=True)
    published_first = out_dir / "exports" / first_rel
    assert published_first.read_bytes() == b"first-checkpoint-bytes"

    # Forbid any byte read or sha256 call on published_first (and its sidecars) during Poll 2.
    forbidden = {
        published_first.resolve(),
        Path(str(published_first) + ".sha256").resolve(),
        Path(str(published_first) + ".complete").resolve(),
    }
    real_open = Path.open
    real_sha256 = supervisor.sha256

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in str(mode) and self.exists() and self.resolve() in forbidden:
            raise AssertionError(f"periodic poll re-read already-verified export: {self}")
        return real_open(self, *args, **kwargs)

    def guarded_sha256(path: Path) -> str:
        if path.exists() and path.resolve() in forbidden:
            raise AssertionError(f"periodic poll re-hashed already-verified export: {path}")
        return real_sha256(path)

    with monkeypatch.context() as poll2_guard:
        poll2_guard.setattr(Path, "open", guarded_open)
        poll2_guard.setattr(supervisor, "sha256", guarded_sha256)
        supervisor.collect_exports(
            cli, root, [entry1, entry2], out_dir, limit=1024 * 1024, periodic=True
        )

    assert (out_dir / "exports" / second_rel).read_bytes() == b"second-checkpoint-bytes"

    # Touching/modifying an already-published local export mid-run must not abort the running job.
    published_first.write_bytes(b"touched-locally-after-publish")
    supervisor.collect_exports(
        cli, root, [entry1, entry2], out_dir, limit=1024 * 1024, periodic=True
    )

    # Terminal collection (periodic=False) still performs full verification on uncollected exports.
    supervisor.collect_exports(
        cli, root, [entry1, entry2, entry3], out_dir, limit=1024 * 1024, periodic=False
    )
    assert (out_dir / "exports" / terminal_rel).read_bytes() == b"terminal-checkpoint-bytes"
    assert _deferred_warnings(capsys.readouterr().err) == []


def test_deferred_multichunk_leftover_swept_on_next_try_preserving_job_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deferred multi-chunk download sweeps leftover .snake-transfer-* on retry safely."""
    monkeypatch.setattr(transfers, "DOWNLOAD_CHUNK_BYTES", 512)
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-sweep"
    remote_out = remote / root.lstrip("/") / "output"
    rel = "checkpoints/light_u0002004.pt"
    payload = b"M" * 1400  # 3 chunks at 512B
    entry = _write_remote_export(remote_out, rel, payload)
    # Sibling job output files that must NEVER be touched by cleanup sweeps.
    in_progress_job_file = remote_out / "checkpoints/in_progress.tmp"
    in_progress_job_file.write_bytes(b"live-job-writer-temp")
    metrics_job_file = remote_out / "metrics/train.jsonl"
    metrics_job_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_job_file.write_bytes(b'{"step": 100}\n')

    cycle = [1]
    rpc_events: list[tuple[int, str, str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "download":
            if cycle[0] == 1 and ".part" in argv[-2]:
                # Exhaust the per-file 300s deadline so finally:_cleanup skips REMOVE_TREE.
                now[0] += max(timeout, 305.0)
                raise subprocess.TimeoutExpired(argv, timeout)
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        script_text = Path(argv[argv.index("-f") + 1]).read_text()
        for kind in ("SNAKE_TRANSFER_REMOVE_TREE", "SNAKE_TRANSFER_DOWNLOAD_SPLIT"):
            if kind in script_text:
                rpc_events.append((cycle[0], kind, script_text))
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    # Cycle 1: DOWNLOAD_SPLIT creates .snake-transfer-<uuid1>, chunk download expires deadline,
    # and _cleanup returns early leaving .snake-transfer-<uuid1> behind on remote disk.
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
    assert len(_deferred_warnings(capsys.readouterr().err)) == 1
    leftover_dirs = list((remote_out / "checkpoints").glob(".snake-transfer-*"))
    assert len(leftover_dirs) == 1
    leftover_name = leftover_dirs[0].name

    # Cycle 2: sweeps .snake-transfer-<uuid1> BEFORE splitting .snake-transfer-<uuid2>,
    # then collects and cleans .snake-transfer-<uuid2>, leaving zero scratch dirs.
    cycle[0] = 2
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
    cycle2_kinds = [kind for cyc, kind, _ in rpc_events if cyc == 2]
    assert cycle2_kinds == [
        "SNAKE_TRANSFER_REMOVE_TREE",
        "SNAKE_TRANSFER_DOWNLOAD_SPLIT",
        "SNAKE_TRANSFER_REMOVE_TREE",
    ]
    first_cycle2_script = next(script for cyc, _, script in rpc_events if cyc == 2)
    assert leftover_name in first_cycle2_script
    assert list((remote_out / "checkpoints").glob(".snake-transfer-*")) == []
    assert cli._deferred_remote_cleanups == {}

    # Job's own output files and verified local export are completely intact.
    assert (remote_out / rel).read_bytes() == payload
    assert Path(str(remote_out / rel) + ".sha256").is_file()
    assert Path(str(remote_out / rel) + ".complete").is_file()
    assert in_progress_job_file.read_bytes() == b"live-job-writer-temp"
    assert metrics_job_file.read_bytes() == b'{"step": 100}\n'
    assert (out_dir / "exports" / rel).read_bytes() == payload


def test_deferred_batch_archive_swept_on_singleton_retry_with_bounded_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed batch archive is swept even when retried via singleton export_file in bounded time."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-batch-sweep"
    local_root = remote / root.lstrip("/")
    remote_out = local_root / "output"
    entry_a = _write_remote_export(remote_out, "metrics/u0001.jsonl", b'{"u":1}\n')
    entry_b = _write_remote_export(remote_out, "metrics/u0002.jsonl", b'{"u":2}\n')

    fail_archive_download = [True]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "download":
            if fail_archive_download[0]:
                now[0] += max(timeout, 305.0)
                raise subprocess.TimeoutExpired(argv, timeout)
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        script_path = Path(argv[argv.index("-f") + 1])
        wrapped = script_path.read_text()
        if "SNAKE_EXPORT_BATCH" in wrapped:
            wrapped = wrapped.replace(
                "'/content/.snake-rpc-", repr(str(remote / "content/.snake-rpc-"))[:-1]
            )
            (remote / "content").mkdir(parents=True, exist_ok=True)
            wrapped = wrapped.replace(root, str(local_root))
            wrapped = wrapped.replace(
                "'path': str(archive)",
                f"'path': str(archive).replace({str(local_root)!r}, {root!r}, 1)",
            )
            script_path.write_text(wrapped)
            _, proc = _exec_wrapped_snippet(argv, remote_root=None)
            return proc
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    # Cycle 1: export_batch creates .snake-export-batch-<uuid>.zip, then archive download fails.
    supervisor.collect_exports(
        cli, root, [entry_a, entry_b], out_dir, limit=1024 * 1024, periodic=True
    )
    assert len(_deferred_warnings(capsys.readouterr().err)) == 2
    assert len(list(local_root.glob(".snake-export-batch-*.zip"))) == 1

    # Cycle 2: retry entry_a via singleton export_file -> sweeps failed batch zip via REMOVE_TREE.
    fail_archive_download[0] = False
    supervisor.collect_exports(cli, root, [entry_a], out_dir, limit=1024 * 1024, periodic=True)
    assert list(local_root.glob(".snake-export-batch-*.zip")) == []
    assert cli._deferred_remote_cleanups == {}
    assert (out_dir / "exports/metrics/u0001.jsonl").read_bytes() == b'{"u":1}\n'
    assert (remote_out / "metrics/u0001.jsonl").read_bytes() == b'{"u":1}\n'
    assert (remote_out / "metrics/u0002.jsonl").read_bytes() == b'{"u":2}\n'


def test_leftover_sweep_is_time_bounded_and_idempotent_when_already_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hung leftover sweep consumes <= LEFTOVER_SWEEP_SECONDS and missing leftovers succeed."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 3600.0
    monkeypatch.setattr(cli, "_put_lease", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_a, **_kw: True)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen-bounded-sweep"
    remote_out = remote / root.lstrip("/") / "output"
    rel = "checkpoints/light_u0002004.pt"
    entry = _write_remote_export(remote_out, rel, b"checkpoint-bytes")
    remote_path = f"{root}/output/{rel}"
    missing_scratch = f"{root}/output/checkpoints/.snake-transfer-{'a' * 32}"

    hang_cleanup = [True]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint or "ep"), ""
            )
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        script_text = Path(argv[argv.index("-f") + 1]).read_text()
        if hang_cleanup[0] and "SNAKE_TRANSFER_REMOVE_TREE" in script_text:
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"

    # Try 1: hung leftover REMOVE_TREE is bounded to LEFTOVER_SWEEP_SECONDS (30s),
    # leaving 270s of the 300s file window so the download itself still succeeds!
    transfers.record_remote_leftover(cli, remote_path, missing_scratch)
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024, periodic=True)
    assert now[0] == pytest.approx(transfers.LEFTOVER_SWEEP_SECONDS, abs=5.0)
    assert (out_dir / "exports" / rel).read_bytes() == b"checkpoint-bytes"
    # Unacknowledged cleanup stays recorded for a later sweep.
    assert transfers.pending_remote_leftovers(cli, remote_path) == [missing_scratch]

    # Try 2: idempotent sweep succeeds even when missing_scratch does not exist remotely.
    hang_cleanup[0] = False
    transfers.sweep_remote_leftovers(cli, remote_path, cli.renew)
    assert transfers.pending_remote_leftovers(cli, remote_path) == []
