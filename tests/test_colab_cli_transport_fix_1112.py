"""Fault-injection and safety-property regressions for 2026-11-12 transport fixes (G1-G4 + 60s poll)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import (
    accounting,
    cli_guard,
    failures,
    policy,
    runtime,
    source,
    supervisor,
    transfers,
)

# Verbatim failure strings captured from /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/.
G2_CLEANUP_TIMEOUT_LOG = (
    "Command '['/Users/josenunez/.local/bin/colab', '--auth', 'oauth2', '--config', "
    "'/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/"
    "v3-ppo-train-a100-sc-c2-002/state/v3-ppo-train-a100-sc-c2-002/sessions.json', "
    "'exec', '-s', 'snake-v3-ppo-train-a100-sc-c2-002-6c414a31cdee', '-f', "
    "'/var/folders/h2/r_nk05n12vg4v5f_rxyf65v40000gn/T/tmp2i8axfbg.py', "
    "'--timeout', '9.473861542006489']' timed out after 9.473861125996336 seconds"
)
G2_FATAL_REFUSAL_LOG = "previous exec uncertain/busy; refusing to resubmit to kernel"

G4_WEBSOCKET_CLOSED_STDERR = (
    "    │\n"
    "│   191 │   │   Sends UTF-8 encoded text.                                  │\n"
    '│   192 │   │   """                                                        │\n'
    "│   193 │   │   if not self.sock or self.sock.send(text_data, ABNF.OPCODE_TEXT) == 0: │\n"
    '│ ❱ 194 │   │   │   raise WebSocketConnectionClosedException("Connection is already closed.") │\n'
    "│   195 │                                                                  │\n"
    "│   196 │   def send_bytes(self, data: Union[bytes, bytearray]) -> None:   │\n"
    '│   197 │   │   """                                                        │\n'
    "╰──────────────────────────────────────────────────────────────────────────╯\n"
    "WebSocketConnectionClosedException: Connection is already closed.\n"
)

G3_RENEWAL_TIMEOUT_LOG = (
    "Command '['/Users/josenunez/.local/bin/colab', '--auth', 'oauth2', '--config', "
    "'/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/"
    "v3-ppo-train-l4-sl4-01-002/state/v3-ppo-train-l4-sl4-01-002/sessions.json', "
    "'exec', '-s', 'snake-v3-ppo-train-l4-sl4-01-002-862896ffe75d', '-f', "
    "'/var/folders/h2/r_nk05n12vg4v5f_rxyf65v40000gn/T/tmpobwaackt.py', "
    "'--timeout', '120.0']' timed out after 119.99999920799746 seconds"
)
G3_OLD_MISLABELLED_FAILURE_LOG = (
    "download hash mismatch or transfer failure: download "
    "/content/snake-cli/862896ffe75d4f17a8412763b1ca21a8/output/checkpoints/"
    "full_u0000521.sim_shard_0000_of_0003.pt called at "
    "/Users/josenunez/Projects/ml/snake-dqn/research/colab_cli/supervisor.py:1034: "
    "total upload time cap reached at download "
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/"
    "v3-ppo-train-l4-sl4-01-002/output/exports/checkpoints/.transfer-82raow8z/.transfer-w4dkj6x3.22"
)

G1_SOURCE_FETCH_TIMEOUT_LOG = (
    "Command '['/Users/josenunez/.local/bin/colab', '--auth', 'oauth2', '--config', "
    "'/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/"
    "v3-ppo-train-l4-rwal-01-002/state/v3-ppo-train-l4-rwal-01-002/sessions.json', "
    "'exec', '-s', 'snake-v3-ppo-train-l4-rwal-01-002-047a941f9019', '-f', "
    "'/var/folders/h2/r_nk05n12vg4v5f_rxyf65v40000gn/T/tmppc1d39wr.py', "
    "'--timeout', '164.99999304200173']' timed out after 164.99999212499824 seconds"
)


def _owned_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    now: list[float],
    *,
    session: str = "owned",
    endpoint: str = "gpu-a100-s-owned",
) -> supervisor.CLI:
    """Build a CLI bound to a durable owned assignment record and fake clock."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config = state_dir / "sessions.json"
    config.write_text(json.dumps({session: {"endpoint": endpoint}}))
    cli = supervisor.CLI("colab", config, session)
    cli.poll_recovery_endpoint = endpoint
    cli.assignment_record.write_text(
        json.dumps(
            {
                "endpoint": endpoint,
                "notebook_hash": str(uuid.UUID(cli.notebook_hash)),
                "session": session,
            }
        )
    )
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + float(delay))
    )
    monkeypatch.setattr(transfers.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        transfers.time, "sleep", lambda delay: now.__setitem__(0, now[0] + float(delay))
    )
    return cli


def _sessions_stdout(session: str, endpoint: str, hardware: str = "A100") -> str:
    return f"[{session}] {endpoint} | Hardware: {hardware}\n"


def _exec_wrapped_snippet(
    argv: list[str], *, remote_root: Path | None = None
) -> tuple[str, subprocess.CompletedProcess[str]]:
    script_path = Path(argv[argv.index("-f") + 1])
    wrapped = script_path.read_text()
    if remote_root is not None:
        wrapped = wrapped.replace("'/content", repr(str(remote_root / "content"))[:-1])
        wrapped = wrapped.replace('"/content', '"' + str(remote_root / "content"))
    output = io.StringIO()
    scope: dict[str, Any] = {"__name__": "__main__"}
    with contextlib.redirect_stdout(output):
        exec(compile(wrapped, "<fake-kernel>", "exec"), scope)
    return wrapped, subprocess.CompletedProcess(argv, 0, output.getvalue(), "")


# ---------------------------------------------------------------------------
# Fix 1 (G2): Swallowed download cleanup timeout + pre-existing exec_uncertain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exhaust_cleanup_budget", [False, True])
def test_g2_swallowed_download_cleanup_timeout_old_fails_new_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exhaust_cleanup_budget: bool
) -> None:
    """Reproduce v3-ppo-train-a100-sc-c2-002.log: REMOVE_TREE timeout then renewal."""
    assert "9.473861" in G2_CLEANUP_TIMEOUT_LOG
    monkeypatch.setattr(transfers, "DOWNLOAD_CHUNK_BYTES", 512)
    now = [0.0]
    cli = _owned_cli(
        tmp_path,
        monkeypatch,
        now,
        session="snake-v3-ppo-train-a100-sc-c2-002-6c414a31cdee",
        endpoint="gpu-a100-s-6c414a31cdee",
    )
    remote = tmp_path / "remote"
    payload = b"checkpoint-shard-" * 96  # > 512B -> multi-chunk download + REMOVE_TREE cleanup
    digest = hashlib.sha256(payload).hexdigest()
    remote_file = remote / "content/snake-cli/gen1/output/checkpoints/shard.pt"
    remote_file.parent.mkdir(parents=True)
    remote_file.write_bytes(payload)

    # 1) Demonstrate OLD failure: when REMOVE_TREE times out and leaves exec_uncertain=True
    # without pre-poll idle recovery, the very next rpc refuses fatally.
    cli.exec_uncertain = True
    with pytest.raises(supervisor.Refused, match=re.escape(G2_FATAL_REFUSAL_LOG)):
        cli.rpc("result=True")
    cli.exec_uncertain = False

    # 2) Exercise NEW production download_file + next poll_rpc on the exact fault.
    cleanup_timeouts = 0
    idle_checks = 0

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal cleanup_timeouts
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint), ""
            )
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        if cmd == "upload":
            dst = remote / argv[-1].lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(argv[-2], dst)
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        code = Path(argv[argv.index("-f") + 1]).read_text()
        if "REMOVE_TREE" in code:
            cleanup_timeouts += 1
            if exhaust_cleanup_budget:
                now[0] += timeout
                raise subprocess.TimeoutExpired(argv, timeout)
            return subprocess.CompletedProcess(argv, 1, "", "unacknowledged cleanup failure\n")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    def fake_idle(*args: Any, **kwargs: Any) -> bool:
        nonlocal idle_checks
        idle_checks += 1
        return True

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", fake_idle)

    file_deadline = now[0] + (9.473861542006489 if exhaust_cleanup_budget else 120.0)
    out_file = tmp_path / "local.pt"
    with cli.download_controls(file_deadline):
        observed = transfers.download_file(
            cli,
            "/content/snake-cli/gen1/output/checkpoints/shard.pt",
            out_file,
            len(payload),
            expected_sha256=digest,
            deadline=file_deadline,
        )
    assert observed == digest
    assert out_file.read_bytes() == payload
    assert cleanup_timeouts == 1

    if exhaust_cleanup_budget:
        # Cleanup ran out of file_deadline, so uncertainty stays recorded with provenance
        # and is cleared by pre-poll recover_idle() on the next renewal/status poll.
        assert cli.exec_uncertain is True
        assert cli.cleanup_uncertain_reason is not None
    else:
        assert cli.exec_uncertain is False
        assert cli.cleanup_uncertain_reason is None

    assert cli.poll_rpc("result={'healthy': True}") == {"healthy": True}
    assert cli.exec_uncertain is False
    assert cli.cleanup_uncertain_reason is None
    assert idle_checks >= 1


# ---------------------------------------------------------------------------
# Fix 2 (G4): WebSocketConnectionClosedException transient on read-only polls
# ---------------------------------------------------------------------------


def test_g4_websocket_closed_old_fails_mutating_and_new_recovers_read_only_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce v3-ppo-train-a100-rwa-c3a-001.log WebSocket closed traceback."""
    now = [0.0]
    cli = _owned_cli(
        tmp_path,
        monkeypatch,
        now,
        session="snake-v3-ppo-train-a100-rwa-c3a-001-f561fb97f4c2",
        endpoint="gpu-a100-s-ti73oelao63x",
    )
    ws_refusal = supervisor.Refused(f"colab exec failed (1): {G4_WEBSOCKET_CLOSED_STDERR}")

    # OLD policy / mutating policy: WebSocket-closed is NOT in general transient_error.
    assert policy.transient_error(ws_refusal) is False
    assert policy.transient_poll_error(ws_refusal, mutating=True) is False
    assert policy.transient_poll_error(ws_refusal, mutating=False) is True

    # GuardError with WebSocket context lines in Rich output must remain non-transient.
    guard_with_ws_context = supervisor.Refused(
        f"colab exec failed (1): {G4_WEBSOCKET_CLOSED_STDERR}"
        "GuardError: kernel creation already attempted; refusing duplicate kernel creation\n"
    )
    assert policy.transient_poll_error(guard_with_ws_context, mutating=False) is False

    exec_attempts = 0
    idle_probes = 0

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal exec_attempts
        cmd = argv[5]
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint), ""
            )
        assert cmd == "exec"
        exec_attempts += 1
        if exec_attempts == 1:
            return subprocess.CompletedProcess(argv, 1, "", G4_WEBSOCKET_CLOSED_STDERR)
        _, proc = _exec_wrapped_snippet(argv)
        return proc

    def fake_idle(*args: Any, **kwargs: Any) -> bool:
        nonlocal idle_probes
        idle_probes += 1
        return True

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", fake_idle)

    # NEW read-only poll recovers cleanly on attempt 2 after proving kernel_idle.
    assert cli.poll_rpc("result={'step': 2063}") == {"step": 2063}
    assert exec_attempts == 2
    assert idle_probes == 1
    assert not cli.exec_uncertain

    # Mutating recoverable_rpc MUST NOT retry WebSocketConnectionClosedException.
    exec_attempts = 0
    with pytest.raises(supervisor.Refused, match="WebSocketConnectionClosedException"):
        cli.recoverable_rpc("result={'mutated': True}")
    assert exec_attempts == 1
    assert cli.exec_uncertain is True


# ---------------------------------------------------------------------------
# Fix 3 (G3): Contents lease PUTs during downloads + unified per-file cap
# ---------------------------------------------------------------------------


def test_g3_download_uses_contents_lease_put_and_avoids_kernel_exec_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce v3-ppo-train-l4-sl4-01-002.log: kernel hangs during multi-chunk export."""
    assert "119.999999" in G3_RENEWAL_TIMEOUT_LOG
    # OLD mislabelled failure classified as UPLOAD_CAP; fixed download message is SUPERVISOR.
    assert (
        failures.classify_exception(supervisor.Refused(G3_OLD_MISLABELLED_FAILURE_LOG))
        == failures.FailureClass.UPLOAD_CAP
    )
    fixed_download_cap = G3_OLD_MISLABELLED_FAILURE_LOG.replace(
        "total upload time cap reached", "total download time cap reached"
    )
    assert (
        failures.classify_exception(supervisor.Refused(fixed_download_cap))
        == failures.FailureClass.SUPERVISOR
    )

    monkeypatch.setattr(transfers, "DOWNLOAD_CHUNK_BYTES", 256)
    now = [0.0]
    cli = _owned_cli(
        tmp_path,
        monkeypatch,
        now,
        session="snake-v3-ppo-train-l4-sl4-01-002-862896ffe75d",
        endpoint="gpu-l4-s-862896ffe75d",
    )
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen1"
    rel = "checkpoints/full_u0000521.sim_shard_0000_of_0003.pt"
    payload = bytes(range(256)) * 6  # 6 chunks
    digest = hashlib.sha256(payload).hexdigest()
    remote_target = remote / root.lstrip("/") / "output" / rel
    remote_target.parent.mkdir(parents=True)
    remote_target.write_bytes(payload)
    Path(str(remote_target) + ".sha256").write_text(f"{digest}  {Path(rel).name}\n")
    Path(str(remote_target) + ".complete").write_text("ok\n")

    lease_puts: list[dict[str, Any]] = []
    renewal_exec_attempts = 0
    cli.renew = lambda force=False: cli.renew_runtime_lease(root, "gen1", force=force)

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal renewal_exec_attempts
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint, "L4"), ""
            )
        if cmd == "upload":
            assert argv[-1] == f"{root}/lease-heartbeat.json"
            assert timeout > 0
            body = json.loads(Path(argv[-2]).read_text())
            assert body == {"generation": "gen1", "renewal_clock": "contents-mtime"}
            lease_puts.append({"dest": argv[-1], "body": body, "at": now[0]})
            return subprocess.CompletedProcess(argv, 0, "", "")
        if cmd == "download":
            # Advance 35s per chunk so every chunk triggers a lease renewal; 6 * 35s = 210s,
            # which fits easily in the 300s file budget ONLY if renewals do not hang 120s on exec.
            now[0] += 35.0
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        code = Path(argv[argv.index("-f") + 1]).read_text()
        if "renew_lease(" in code or "read_state(" in code:
            renewal_exec_attempts += 1
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)

    # Demonstrate OLD failure if kernel-exec renewal were invoked 3 times during a 300s file cap.
    old_cli = _owned_cli(tmp_path / "old", monkeypatch, [0.0])
    with old_cli.staging_controls(100.0):
        with pytest.raises(supervisor.Refused, match="total upload time cap reached"):
            now_old = [100.0]
            monkeypatch.setattr(supervisor.time, "monotonic", lambda: now_old[0])
            old_cli.rpc("result=True")
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])

    # Exercise NEW export_file: zero renewal kernel execs, Contents PUTs renew the lease,
    # and size + SHA-256 + sidecar + completion marker are published atomically.
    out_dir = tmp_path / "job-out"
    supervisor.export_file(
        cli,
        root,
        {"path": rel, "size": len(payload), "sha256": digest},
        out_dir,
        limit=10 * 1024 * 1024,
    )
    exported = out_dir / "exports" / rel
    assert exported.read_bytes() == payload
    assert Path(str(exported) + ".sha256").read_text().split()[0] == digest
    assert Path(str(exported) + ".complete").is_file()
    assert renewal_exec_attempts == 0
    assert len(lease_puts) >= 6


def test_g3_per_file_download_budget_bounds_sidecars_and_reports_download_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine download stall fails at the single per-file cap with 'download time cap'."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen1"
    rel = "metrics.json"
    payload = b'{"ok":true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    remote_target = remote / root.lstrip("/") / "output" / rel
    remote_target.parent.mkdir(parents=True)
    remote_target.write_bytes(payload)
    Path(str(remote_target) + ".sha256").write_text(f"{digest}  {rel}\n")
    Path(str(remote_target) + ".complete").write_text("ok\n")
    monkeypatch.setattr(cli, "_put_lease", lambda *args, **kwargs: None)
    cli.renew = lambda force=False: cli.renew_runtime_lease(root, "gen1", force=force)

    def stalling_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "download":
            if argv[-2].endswith(".sha256"):
                # Payload consumed almost the whole file_deadline; sidecar hits the SAME cap.
                now[0] += timeout
                raise subprocess.TimeoutExpired(argv, timeout)
            now[0] += 295.0
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", stalling_run)
    with pytest.raises(supervisor.Refused, match="total download time cap reached") as exc_info:
        supervisor.export_file(
            cli,
            root,
            {"path": rel, "size": len(payload), "sha256": digest},
            tmp_path / "out",
            limit=1024 * 1024,
        )
    assert "upload time cap" not in str(exc_info.value)
    assert not (tmp_path / "out/exports" / rel).exists()
    assert now[0] == pytest.approx(transfers.transfer_timeout(len(payload)))


# ---------------------------------------------------------------------------
# Fix 4 (G1): Source-fetch recovery window >= 3 attempts + create-or-verify
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _tiny_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repo = tmp_path / "origin-src"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    files = {
        "src/pkg/main.py": "print('v3')\n",
        "research/study/run.sh": "#!/bin/sh\necho ok\n",
        "research/colab_cli/source.py": "# source\n",
        "research/colab_cli/runtime.py": "# runtime\n",
        "research/colab_cli/bootstrap.sh": "#!/bin/sh\n",
        "research/colab_cli/release.py": "# release\n",
        "requirements.txt": "pytest\n",
        "pyproject.toml": "[project]\nname='tiny'\n",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (repo / "research/study/run.sh").chmod(0o755)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pinned")
    bare = tmp_path / "origin.git"
    _git(tmp_path, "clone", "--bare", "-q", str(repo), str(bare))
    manifest = {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "script": "research/study/run.sh",
        "env": {},
        "budget": {"rate_units_per_hour": 1.0, "unit_cap": 1.0, "reserve_units": 0.0},
    }
    plan = source.prepare(repo, manifest, tmp_path / "delivery")
    return repo, bare, plan


def test_g1_source_fetch_old_165s_window_fails_and_new_540s_recovers_after_two_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce v3-ppo-train-l4-rwal-01-002.log: 165s source-fetch hang at startup."""
    assert "164.99999304200173" in G1_SOURCE_FETCH_TIMEOUT_LOG
    _, bare, plan = _tiny_repo(tmp_path)
    monkeypatch.setattr(source, "GITHUB_URL", str(bare))
    real_run = subprocess.run

    # 1) Demonstrate OLD failure: when recovery window == attempt timeout (165s),
    # one 165s hang hits remaining_seconds == 0 and raises immediately with 0 retries.
    old_now = [0.0]
    old_cli = _owned_cli(tmp_path / "old-g1", monkeypatch, old_now)
    old_cli_attempts = 0

    def old_hang(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal old_cli_attempts
        old_cli_attempts += 1
        timeout = float(kwargs["timeout"])
        old_now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.subprocess, "run", old_hang)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)
    monkeypatch.setattr(policy, "SOURCE_RECOVERY_SECONDS", policy.SOURCE_CONTROL_SECONDS)
    with pytest.raises(supervisor.Refused, match="deadline reached during telemetry recovery"):
        with (
            old_cli.staging_controls(old_now[0] + 300.0, source_delivery=True),
            old_cli.operation("source-fetch"),
        ):
            old_cli.recoverable_rpc("# SNAKE_SOURCE_DELIVERY\nresult=True")
    assert old_cli_attempts == 1
    assert old_now[0] == pytest.approx(policy.SOURCE_CONTROL_SECONDS)
    monkeypatch.setattr(policy, "SOURCE_RECOVERY_SECONDS", 540.0)

    # 2) Exercise NEW staging_controls(..., source_delivery=True) + SOURCE_RECOVERY_SECONDS (540s):
    # survive TWO consecutive 165s hangs (including an interrupted 'started' record and a
    # partial checkout on disk) and succeed on attempt 3 with exact commit/tree verification.
    now = [0.0]
    cli = _owned_cli(
        tmp_path / "new-g1",
        monkeypatch,
        now,
        session="snake-v3-ppo-train-l4-rwal-01-002-047a941f9019",
        endpoint="gpu-l4-s-047a941f9019",
    )
    remote = tmp_path / "vm"
    (remote / "content/snake-cli/gen1").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "research/colab_cli/source.py",
        remote / "content/snake-cli/gen1/source.py",
    )

    attempts = 0
    observed_timeouts: list[float] = []

    def fake_source_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        if argv and argv[0] == "git":
            return real_run(argv, **kwargs)
        cmd = argv[5]
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint, "L4"), ""
            )
        assert cmd == "exec"
        timeout = float(kwargs["timeout"])
        observed_timeouts.append(timeout)
        attempts += 1
        if attempts == 1:
            # Simulate hang after recoverable_rpc wrote 'started' and partial repo dir appeared.
            wrapped = Path(argv[argv.index("-f") + 1]).read_text()
            match = re.search(r"_snake_record=_SnakePath\('([^']+)'\)", wrapped)
            assert match is not None
            record_file = remote / match.group(1).lstrip("/")
            record_file.parent.mkdir(parents=True, exist_ok=True)
            record_file.write_text(json.dumps({"status": "started"}))
            partial_repo = remote / "content/snake-cli/gen1/repo"
            partial_repo.mkdir(parents=True, exist_ok=True)
            (partial_repo / "UNTRACKED_PARTIAL").write_text("interrupted")
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        if attempts == 2:
            # Simulate second hang AFTER source.fetch completed checkout, before stdout nonce.
            _, _ = _exec_wrapped_snippet(argv, remote_root=remote)
            # Revert record to 'started' to force create-or-verify path on attempt 3!
            for rec in (remote / "content").glob(".snake-rpc-*.json"):
                rec.write_text(json.dumps({"status": "started"}))
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_source_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)

    source_deadline = now[0] + policy.SOURCE_RECOVERY_SECONDS
    operation = (
        "# SNAKE_SOURCE_DELIVERY\n"
        "import importlib.util\nfrom pathlib import Path\n"
        "root=Path('/content/snake-cli/gen1')\n"
        f"plan={plan!r}\n"
        "s=importlib.util.spec_from_file_location('snake_source',root/'source.py')\n"
        "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
        f"mod.GITHUB_URL={str(bare)!r}\n"
        "result=mod.fetch(root,plan)"
    )
    with (
        cli.staging_controls(source_deadline, source_delivery=True),
        cli.operation("source-fetch"),
    ):
        delivered = cli.recoverable_rpc(operation)

    assert attempts == 3
    assert observed_timeouts[:2] == [policy.SOURCE_CONTROL_SECONDS, policy.SOURCE_CONTROL_SECONDS]
    assert delivered == {
        "transport": "github",
        "commit": plan["commit"],
        "tree": plan["tree"],
    }
    assert (remote / "content/snake-cli/gen1/repo/src/pkg/main.py").read_text() == "print('v3')\n"
    assert not (remote / "content/snake-cli/gen1/repo/UNTRACKED_PARTIAL").exists()

    # Verify pinned-commit/tree checks still refuse a tampered tree hash even on an existing repo.
    bad_plan = {**plan, "tree": "f" * 40}
    with pytest.raises(
        ValueError, match="GitHub source identity does not match pinned commit/tree"
    ):
        source.fetch(remote / "content/snake-cli/gen1", bad_plan)


# ---------------------------------------------------------------------------
# Fix 5: 60s read-only status poll timeout + near-wall-cap export survival
# ---------------------------------------------------------------------------


def test_fix5_60s_poll_timeout_saves_near_wall_cap_final_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With 102s left before cli.deadline, old 120s poll hang loses job; new 60s poll exports."""
    assert policy.POLL_RPC_SECONDS == 60.0
    assert policy.RPC_SECONDS == 120.0
    assert policy.EXEC_POLL_RECOVERY_SECONDS == 1200.0

    # OLD 120s poll behaviour: with 102s left before cli.deadline, one hung poll hits deadline.
    old_now = [0.0]
    old_cli = _owned_cli(tmp_path / "old-cap", monkeypatch, old_now)
    old_cli.deadline = 102.0
    monkeypatch.setattr(policy, "POLL_RPC_SECONDS", 120.0)
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda argv, **kw: (
            old_now.__setitem__(0, old_now[0] + float(kw["timeout"]))
            or (_ for _ in ()).throw(subprocess.TimeoutExpired(argv, float(kw["timeout"])))
        ),
    )
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)
    with pytest.raises(supervisor.Refused, match="deadline reached during telemetry recovery"):
        old_cli.poll_rpc("result={'terminal': True}")
    monkeypatch.setattr(policy, "POLL_RPC_SECONDS", 60.0)

    # NEW 60s poll + Contents download renewal: survives one 60s poll hang at t=0 (leaves ~37s)
    # and exports the final artifact cleanly before cli.deadline = 102s!
    now = [0.0]
    cli = _owned_cli(tmp_path / "new-cap", monkeypatch, now)
    cli.deadline = 102.0
    remote = tmp_path / "remote-cap"
    root = "/content/snake-cli/gen1"
    rel = "results.json"
    payload = b'{"final_return": 13644.6}\n'
    digest = hashlib.sha256(payload).hexdigest()
    remote_target = remote / root.lstrip("/") / "output" / rel
    remote_target.parent.mkdir(parents=True)
    remote_target.write_bytes(payload)
    Path(str(remote_target) + ".sha256").write_text(f"{digest}  {rel}\n")
    Path(str(remote_target) + ".complete").write_text("ok\n")
    monkeypatch.setattr(cli, "_put_lease", lambda *args, **kwargs: None)
    cli.renew = lambda force=False: cli.renew_runtime_lease(root, "gen1", force=force)

    poll_timeouts: list[float] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint), ""
            )
        if cmd == "download":
            now[0] += 1.0
            src = remote / argv[-2].lstrip("/")
            Path(argv[-1]).write_bytes(src.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert cmd == "exec"
        code = Path(argv[argv.index("-f") + 1]).read_text()
        if "terminal" in code and not poll_timeouts:
            poll_timeouts.append(timeout)
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        poll_timeouts.append(timeout)
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)

    state = cli.poll_rpc("result={'terminal': True}")
    assert state == {"terminal": True}
    assert poll_timeouts[0] == 60.0
    supervisor.export_file(
        cli,
        root,
        {"path": rel, "size": len(payload), "sha256": digest},
        tmp_path / "final-out",
        limit=1024 * 1024,
    )
    assert (tmp_path / "final-out/exports" / rel).read_bytes() == payload
    assert now[0] < cli.deadline


# ---------------------------------------------------------------------------
# Standing Safety-Property Regressions
# ---------------------------------------------------------------------------


def test_safety_uncertain_mutating_exec_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncertain mutating execs are never re-dispatched or auto-cleared."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    exec_calls = 0

    def hanging_exec(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal exec_calls
        exec_calls += 1
        timeout = float(kwargs["timeout"])
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.subprocess, "run", hanging_exec)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)

    # 1) Direct mutating cli.rpc gets exactly 1 dispatch (120s) and sets exec_uncertain=True.
    with pytest.raises(subprocess.TimeoutExpired):
        cli.rpc("result={'launched': True}")
    assert exec_calls == 1
    assert cli.exec_uncertain is True

    # 2) Neither cli.rpc nor cli.recoverable_rpc clears an unattributed uncertain flag!
    with pytest.raises(supervisor.Refused, match=re.escape(G2_FATAL_REFUSAL_LOG)):
        cli.rpc("result={'retry': True}")
    with pytest.raises(supervisor.Refused, match=re.escape(G2_FATAL_REFUSAL_LOG)):
        cli.recoverable_rpc("result={'retry': True}")
    assert exec_calls == 1


def test_safety_stop_is_still_verified_even_after_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop_verified retains cleanup budget past cli.deadline and refuses if still listed."""
    now = [500.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    cli.deadline = 100.0  # Already expired!
    endpoint = "gpu-a100-s-owned"
    calls: list[str] = []
    listed = [True, False]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        calls.append(cmd)
        if cmd == "stop":
            return subprocess.CompletedProcess(argv, 0, "Stopped\n", "")
        assert cmd == "sessions"
        is_listed = listed.pop(0) if listed else False
        out = (
            _sessions_stdout(cli.session, endpoint)
            if is_listed
            else "No active sessions found on server.\n"
        )
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    supervisor.stop_verified(cli, endpoint)
    assert calls == ["stop", "sessions", "sessions"]

    # If the assignment never disappears, stop_verified raises STOP_UNVERIFIED.
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, _sessions_stdout(cli.session, endpoint) if argv[5] == "sessions" else "", ""
        ),
    )
    monkeypatch.setattr(cli_guard, "unassign_endpoint", lambda *args, **kwargs: None)
    with pytest.raises(supervisor.Refused, match="STOP_UNVERIFIED"):
        supervisor.stop_verified(cli, endpoint)


@pytest.mark.parametrize("failure_mode", ["busy_kernel", "dead_assignment"])
def test_safety_dead_or_busy_kernel_fails_within_recovery_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    """A genuinely busy or vanished runtime never clears uncertainty and fails in-window."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        timeout = float(kwargs["timeout"])
        if cmd == "sessions":
            if failure_mode == "dead_assignment":
                return subprocess.CompletedProcess(
                    argv, 0, "No active sessions found on server.\n", ""
                )
            return subprocess.CompletedProcess(
                argv, 0, _sessions_stdout(cli.session, cli.poll_recovery_endpoint), ""
            )
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: False)

    expected = (
        "supervised assignment disappeared during telemetry recovery"
        if failure_mode == "dead_assignment"
        else "deadline reached during telemetry recovery"
    )
    with pytest.raises(supervisor.Refused, match=expected):
        cli.poll_rpc("result=True")
    assert cli.exec_uncertain is True
    assert now[0] <= policy.EXEC_POLL_RECOVERY_SECONDS


def test_safety_completed_exports_stay_immutable_and_corrupt_downloads_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed exports cannot be overwritten, and corrupt downloads never publish."""
    now = [0.0]
    cli = _owned_cli(tmp_path, monkeypatch, now)
    remote = tmp_path / "remote"
    root = "/content/snake-cli/gen1"
    rel = "unit_0001.bin"
    payload = b"immutable-unit-payload"
    digest = hashlib.sha256(payload).hexdigest()
    remote_file = remote / root.lstrip("/") / "output" / rel
    remote_file.parent.mkdir(parents=True)
    remote_file.write_bytes(payload)
    Path(str(remote_file) + ".sha256").write_text(f"{digest}  {rel}\n")
    Path(str(remote_file) + ".complete").write_text("ok\n")
    monkeypatch.setattr(cli, "_put_lease", lambda *args, **kwargs: None)

    corrupt = [True]

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = argv[5]
        if cmd == "download":
            src = remote / argv[-2].lstrip("/")
            data = src.read_bytes()
            if corrupt[0] and src.name == rel:
                data += b"corrupt"
            Path(argv[-1]).write_bytes(data)
            return subprocess.CompletedProcess(argv, 0, "", "")
        _, proc = _exec_wrapped_snippet(argv, remote_root=remote)
        return proc

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    out_dir = tmp_path / "out"
    entry = {"path": rel, "size": len(payload), "sha256": digest}

    # 1) Corrupt bytes rejected before atomic publication.
    with pytest.raises(supervisor.Refused, match="download hash mismatch or transfer failure"):
        supervisor.export_file(cli, root, entry, out_dir, limit=1024 * 1024)
    assert not (out_dir / "exports" / rel).exists()

    # 2) Valid export publishes once; tampering with local file or remote metadata is refused.
    corrupt[0] = False
    supervisor.collect_exports(cli, root, [entry], out_dir, limit=1024 * 1024)
    assert (out_dir / "exports" / rel).read_bytes() == payload

    mutated_entry = {
        "path": rel,
        "size": len(payload),
        "sha256": hashlib.sha256(b"different").hexdigest(),
    }
    with pytest.raises(supervisor.Refused, match="immutable snapshot metadata changed"):
        supervisor.collect_exports(cli, root, [mutated_entry], out_dir, limit=1024 * 1024)

    (out_dir / "exports" / rel).write_bytes(b"tampered")
    with pytest.raises(supervisor.Refused, match="immutable export changed"):
        supervisor.export_file(cli, root, entry, out_dir, limit=1024 * 1024)


def test_safety_accounting_caps_and_lease_generation_fencing_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Account caps and generation-fenced lease renewals remain intact."""
    assert accounting.ACCOUNT_TYPE_CAPS == {
        "L4": 3,
        "T4": 3,
        "G4": 3,
        "A100": 3,
        "H100": 1,
        "CPU": 5,
    }
    assert accounting.ACCOUNT_ASSIGNMENT_CAP == 12

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "state.json").write_text(json.dumps({"generation": "gen-owned"}))
    (runtime_root / "lease.json").write_text(
        json.dumps({"generation": "gen-owned", "renewed_at": 1.0})
    )
    assert runtime.renew_lease(runtime_root, "gen-owned") is True
    with pytest.raises(RuntimeError, match="generation fence mismatch"):
        runtime.renew_lease(runtime_root, "gen-foreign")

    # Verify CLI._put_lease writes generation-fenced lease-heartbeat.json via Contents upload.
    now = [10.0]
    cli = _owned_cli(tmp_path / "lease-cli", monkeypatch, now)
    uploaded: dict[str, Any] = {}

    def capture_upload(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[5] == "upload"
        uploaded["dest"] = argv[-1]
        uploaded["body"] = json.loads(Path(argv[-2]).read_text())
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(supervisor.subprocess, "run", capture_upload)
    cli._put_lease("/content/snake-cli/gen-owned", "gen-owned", 30.0)
    assert uploaded["dest"] == "/content/snake-cli/gen-owned/lease-heartbeat.json"
    assert uploaded["body"] == {
        "generation": "gen-owned",
        "renewal_clock": "contents-mtime",
    }
