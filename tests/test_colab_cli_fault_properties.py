"""Seeded offline transport-fault properties for the acceptance broker.

The fixture's latency is virtual on purpose: a many-seed safety test must not
turn bounded production timeouts into a multi-hour unit suite.  Failure modes
still cross a real subprocess boundary and mutate the fixture's durable state.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, supervisor

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "tests" / "fixtures" / "fake_colab.py"
HELPER = REPO / "tests" / "fixtures" / "fake_colab_helper.py"


class _TransportClock:
    """Elapsed-time model with deterministic, concurrent lease scheduling.

    Each CLI still crosses a real subprocess boundary. Injected network waits
    advance this clock and dispatch scheduled Contents heartbeats while the
    foreground RPC waits. No production acceptance gate is replaced.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.beats: dict[supervisor.CLI, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.counter = 0

    def advance(self, seconds: float) -> None:
        end = self.now + max(0, seconds)
        while True:
            due = [(v["at"], cli, v) for cli, v in self.beats.items() if v["at"] <= end]
            if not due:
                break
            at, cli, value = min(due, key=lambda item: item[0])
            self.now = max(self.now, at)
            value["at"] = float("inf")
            try:
                cli._put_lease(
                    value["root"],
                    value["generation"],
                    cli.remaining_seconds(min(120, value["lease"] / 3)),
                )
            except Exception:
                pass  # Same fail-closed lease behavior as the production thread.
            if cli in self.beats:
                value["at"] = self.now + 30
            end = max(end, self.now)
        self.now = max(self.now, end)

    def start(self, cli: supervisor.CLI, root: str, generation: str, lease: float) -> None:
        self.beats[cli] = {"at": self.now, "root": root, "generation": generation, "lease": lease}
        self.advance(0)

    def stop(self, cli: supervisor.CLI, *, wait: bool = False) -> None:
        for owner in list(self.beats):
            if owner is cli or owner._lease_stop is cli._lease_stop:
                self.beats.pop(owner, None)


def _fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan: dict[str, Any]) -> str:
    state = tmp_path / "fake-state.json"
    state.write_text(json.dumps({"balance": 1178.3, "rate": 0.0, "sessions": {}, "exec_count": {}}))
    log = tmp_path / "fake-calls.jsonl"
    log.touch()
    remote = tmp_path / "remote"
    remote.mkdir()
    stub = tmp_path / "cli-stub" / "colab_cli"
    stub.mkdir(parents=True)
    stub.joinpath("__init__.py").write_text("")
    stub.joinpath("client.py").write_text(
        "class Client:\n"
        "    def assign(self,*a,**k): pass\n"
        "    def _issue_request(self,*a,**k): pass\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(stub.parent))
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state))
    monkeypatch.setenv("FAKE_COLAB_LOG", str(log))
    monkeypatch.setenv("FAKE_COLAB_REMOTE", str(remote))
    monkeypatch.setenv("FAKE_COLAB_MODE", "acceptance")
    monkeypatch.setenv("FAKE_COLAB_FAULT_PLAN", json.dumps(plan, sort_keys=True))
    monkeypatch.setenv("SNAKE_COLAB_OAUTH_PYTHON", str(HELPER))
    monkeypatch.setenv("FAKE_COLAB_OWN_RATE", "0.5")
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    return str(FAKE)


@pytest.fixture
def pinned_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Commit the exact acceptance source bytes used by the offline run.

    Acceptance intentionally refuses a dirty source tree.  The test therefore
    exercises that guard against a temporary committed checkout rather than
    weakening the source-binding check for the local fake provider.
    """
    repository = tmp_path / "pinned-repo"
    shutil.copytree(REPO / "research" / "colab_cli", repository / "research" / "colab_cli")
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        source = REPO / name
        if source.is_file():
            shutil.copy2(source, repository / name)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=acceptance-test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "pinned acceptance source",
        ],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(supervisor, "REPO", repository)
    monkeypatch.setenv("FAKE_COLAB_SOURCE_REPO", str(repository))
    return repository


def _pinned_plan(
    tmp_path: Path, pinned_repo: Path, *, profile: str = "final-only"
) -> dict[str, Any]:
    """Use a real committed source bundle, with a fresh acceptance output root."""
    return acceptance.build_plan(pinned_repo, tmp_path / "acceptance", None, profile=profile)[
        "plan"
    ]


def _assert_safe(tmp_path: Path, plan: dict[str, Any]) -> None:
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert state["sessions"] == {}, "an acceptance fault left a paid fake assignment"
    # Acceptance intentionally has CPU and L4 arms. A session name is the
    # mutation identity, so the property is one allocation per arm, not one
    # allocation for the whole two-arm qualification.
    allocation_names = state.get("allocation_names", [])
    assert len(allocation_names) == len(set(allocation_names))
    assert state.get("allocation_count", 0) <= 2, "uncertain allocation was submitted twice"
    assert all(
        count <= 1 for count in state.get("job_launches", {}).values()
    ), "uncertain job-launch exec was submitted twice"
    run_dir = Path(plan["jobs"][0]["manifest"]).parent.parent
    receipts = list(run_dir.glob("outputs/*/receipt.json"))
    assert receipts or state.get("allocation_count", 0) == 0
    allocated_receipts = set()
    assigned_arms = []
    for job in plan["jobs"]:
        manifest = json.loads(Path(job["manifest"]).read_text())
        matches = [
            name
            for name in allocation_names
            if name.startswith("snake-" + manifest["job_id"] + "-")
        ]
        assert len(matches) <= 1, "same acceptance arm allocated twice with different UUIDs"
        assigned_arms.extend(matches)
    assert set(assigned_arms) == set(allocation_names)
    assert set(state.get("job_launches", {})) <= set(allocation_names)
    for receipt in receipts:
        value = json.loads(receipt.read_text())
        if value["endpoint"]:
            allocated_receipts.add(value["session"])
            assert value["stop_verified"] is True
            assert value.get("cleanup_error") is None
        else:
            assert not value["stop_verified"]
            assert state.get("allocation_count", 0) == 0
    assert allocated_receipts == set(allocation_names), "allocated arm lacks its release receipt"


@pytest.fixture(autouse=True)
def _isolate_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _TransportClock:
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    clock = _TransportClock()
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if str(argv[0]) != str(FAKE):
            return real_run(argv, **kwargs)
        with clock.lock:
            clock.counter += 1
            call_id = str(clock.counter)
            budget = float(kwargs.get("timeout") or 120)
            kwargs["env"] = {
                **os.environ,
                **kwargs.get("env", {}),
                "FAKE_COLAB_NOW": str(clock.now),
                "FAKE_COLAB_TIMEOUT": str(budget),
                "FAKE_CALL_ID": call_id,
            }
            code = Path(argv[argv.index("-f") + 1]).read_text() if "-f" in argv else ""
            dispatched_at = clock.now
            result = real_run(argv, **kwargs)
            events = [
                json.loads(line)
                for line in Path(os.environ["FAKE_COLAB_LOG"]).read_text().splitlines()
            ]
            event = next(e for e in reversed(events) if e.get("call_id") == call_id)
            elapsed = event.get("fault", {}).get("virtual_latency_seconds", 0.25)
            if "time.sleep(65)" in code:
                elapsed += 65
            clock.advance(min(elapsed, budget))
            if clock.now - dispatched_at > budget or event.get("fault", {}).get("timed_out"):
                raise subprocess.TimeoutExpired(argv, budget)
            return result

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(supervisor.time, "sleep", clock.advance)
    monkeypatch.setattr(
        supervisor.CLI,
        "start_lease",
        lambda cli, root, generation, lease: clock.start(cli, root, generation, lease),
    )
    monkeypatch.setattr(
        supervisor.CLI, "stop_lease", lambda cli, wait=False: clock.stop(cli, wait=wait)
    )
    return clock


@pytest.mark.parametrize(
    "call,fault,phase",
    [
        (call, fault, "post" if call in {"new", "exec"} else "pre")
        for call in ("version", "sessions", "usage", "upload", "download", "new", "exec", "stop")
        for fault in ("latency", "timeout", "reset", "http_5xx")
    ],
)
def test_acceptance_fault_classes_never_orphan_or_repeat_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_repo: Path,
    call: str,
    fault: str,
    phase: str,
) -> None:
    """Every call class has a deterministic pre/post fault regression case."""
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 41,
            "max_latency_seconds": 180,
            "schedule": [{"call": call, "fault": fault, "phase": phase}],
        },
    )
    plan = _pinned_plan(tmp_path, pinned_repo)
    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(HELPER),
    )
    assert rc in {0, 1}
    _assert_safe(tmp_path, plan)
    events = [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]
    assert any(
        e.get("fault", {}).get("call") == call and e["fault"]["fault"] == fault for e in events
    )
    ambiguous = fault != "latency" or any(e.get("fault", {}).get("timed_out") for e in events)
    if call in {"new", "exec"} and phase == "post" and ambiguous:
        state = json.loads((tmp_path / "fake-state.json").read_text())
        assert state["allocation_count"] == 1
        if call == "exec":
            assert sum(state["job_launches"].values()) == 1


@pytest.mark.parametrize("seed", range(24))
def test_seeded_random_fault_property_preserves_broker_safety(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned_repo: Path, seed: int
) -> None:
    """Randomized call classes cover latency and transient responses offline."""
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": seed,
            "random": True,
            "probability": 0.12,
            "max_latency_seconds": 180,
            "classes": {
                "read": ["latency", "timeout", "reset", "http_5xx"],
                "sessions": ["latency", "timeout", "reset", "http_5xx"],
                "usage": ["latency", "timeout", "reset", "http_5xx"],
                "upload": ["latency", "timeout", "reset", "http_5xx"],
                "download": ["latency", "timeout", "reset", "http_5xx"],
                "allocate": ["latency", "timeout", "reset", "http_5xx"],
                "kernel_exec": ["latency", "timeout", "reset", "http_5xx"],
                "stop": ["latency", "timeout", "reset", "http_5xx"],
            },
        },
    )
    plan = _pinned_plan(tmp_path, pinned_repo)
    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(HELPER),
    )
    assert rc in {0, 1}
    _assert_safe(tmp_path, plan)


@pytest.mark.parametrize("profile", ("final-only", "full-state"))
def test_acceptance_anchor_reaches_every_profile_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned_repo: Path, profile: str
) -> None:
    """A no-fault seed reaches PASS through both actual Broker arms per profile."""
    binary = _fake_env(monkeypatch, tmp_path, {"seed": 0, "max_latency_seconds": 0})
    plan = _pinned_plan(tmp_path, pinned_repo, profile=profile)
    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(HELPER),
    )
    assert rc == 0
    result = json.loads(
        (Path(plan["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json").read_text()
    )
    assert result["qualified"] is True
    assert result["profile"] == profile
    assert result["final_plan"]["profile"] == profile
    assert {gate["id"] for gate in result["final_plan"]["gates"]} == {
        gate["id"] for gate in acceptance.review_gates(profile)
    }
    _assert_safe(tmp_path, plan)
    state = json.loads((tmp_path / "fake-state.json").read_text())
    events = [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]
    operations = {event.get("transfer_operation") for event in events}
    gates = {gate["id"]: gate for gate in result["final_plan"]["gates"]}
    if profile == "final-only":
        assert "DOWNLOAD_SPLIT" in operations
        assert "UPLOAD_ASSEMBLE" not in operations
        for gate in {"busy-full-snapshot", "exact-resume", "checkpoint-overhead"}:
            assert gates[gate]["required"] is False
            assert gates[gate]["status"] == "NOT_REQUIRED"
        assert gates["final-artifact-download"]["required"] is True
        assert gates["final-artifact-download"]["status"] == "OBSERVED"
        l4_output = Path(plan["jobs"][1]["manifest"]).parent.parent / "outputs" / "l4"
        artifact = l4_output / "exports" / "final-artifact.bin"
        assert artifact.stat().st_size == 64 * 1024 * 1024
        evidence = gates["final-artifact-download"]["evidence_record"]
        assert evidence["exported_bytes"] == artifact.stat().st_size
        assert evidence["sha256"] == supervisor.sha256(artifact)
        assert evidence["bytes_per_second_lower_bound"] > 0
        assert (
            json.loads(Path(plan["jobs"][0]["manifest"]).read_text())["env"][
                "COLAB_ACCEPTANCE_PROFILE"
            ]
            == "final-only"
        )
    else:
        assert {"DOWNLOAD_SPLIT", "UPLOAD_ASSEMBLE"} <= operations
        assert gates["final-artifact-download"]["required"] is False
        assert gates["final-artifact-download"]["status"] == "NOT_REQUIRED"
    assert len(state["watchdog_released"]) == 1
    assert "l4" in state["watchdog_released"][0]
    assert [
        json.loads(Path(j["manifest"]).read_text())["timeout_minutes"] for j in plan["jobs"]
    ] == [20, 30]


def test_fake_fault_plan_records_virtual_latency_and_post_commit_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fixture contract: random latency is bounded and post failures commit once."""
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 7,
            "max_latency_seconds": 17,
            "schedule": [{"at": 1, "call": "new", "fault": "timeout", "phase": "post"}],
        },
    )
    command = [binary, "--auth", "oauth2", "--config", str(tmp_path / "config"), "new", "-s", "one"]
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(command, check=False, capture_output=True, text=True)
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert state["allocation_count"] == 1
    event = json.loads((tmp_path / "fake-calls.jsonl").read_text())
    assert event["fault"] == {
        "call": "new",
        "class": "allocate",
        "fault": "timeout",
        "phase": "post",
        "virtual_latency_seconds": 120.0,
        "timed_out": True,
    }
    cli = supervisor.CLI(binary, tmp_path / "config", "one")
    supervisor.stop_verified(cli, next(iter(state["sessions"])))
    assert json.loads((tmp_path / "fake-state.json").read_text())["sessions"] == {}


def test_injected_latency_consumes_the_cli_operation_clock(monkeypatch, tmp_path: Path) -> None:
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 1,
            "max_latency_seconds": 17,
            "schedule": [{"call": "version", "fault": "latency", "latency_seconds": 17}],
        },
    )
    cli = supervisor.CLI(binary, tmp_path / "config")
    assert "0.7.4" in cli.call("version")
    assert cli.call_records[-1]["elapsed_seconds"] == pytest.approx(17)


def test_injected_latency_exhausts_shared_timeout_instead_of_restarting(
    monkeypatch, tmp_path: Path
) -> None:
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 1,
            "max_latency_seconds": 20,
            "schedule": [{"call": "version", "fault": "latency", "latency_seconds": 20}],
        },
    )
    cli = supervisor.CLI(binary, tmp_path / "config")
    with pytest.raises(supervisor.Refused, match="budget exhausted"):
        cli.call("version", timeout=10)
    assert len(cli.call_records) == 1
    assert cli.call_records[0]["elapsed_seconds"] == pytest.approx(10)


def test_full_acceptance_tolerates_ninety_second_launcher_ack(
    monkeypatch, tmp_path: Path, pinned_repo: Path
) -> None:
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 1,
            "max_latency_seconds": 90,
            "schedule": [{"call": "exec", "fault": "latency", "latency_seconds": 90}],
        },
    )
    plan = _pinned_plan(tmp_path, pinned_repo)
    assert (
        acceptance.execute(
            plan,
            state_dir=tmp_path / "state",
            colab=binary,
            review_token=acceptance.REVIEW_TOKEN,
            oauth_python=str(HELPER),
        )
        == 0
    )
    _assert_safe(tmp_path, plan)


def test_slow_concurrent_heartbeat_cannot_hide_foreground_deadline(
    monkeypatch, tmp_path: Path, _isolate_account: _TransportClock
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, {})
    clock = _isolate_account
    heartbeat = supervisor.CLI(binary, tmp_path / "heartbeat")
    monkeypatch.setattr(heartbeat, "_put_lease", lambda *args: clock.advance(60))
    clock.beats[heartbeat] = {
        "at": 0.1,
        "root": "/content/snake-cli/g",
        "generation": "g",
        "lease": 180,
    }
    cli = supervisor.CLI(binary, tmp_path / "foreground")
    with pytest.raises(supervisor.Refused, match="budget exhausted"):
        cli.call("version", timeout=10)
    assert cli.call_records[-1]["status"] == "failed"


def test_fake_binary_can_inject_real_wall_clock_latency(monkeypatch, tmp_path: Path) -> None:
    binary = _fake_env(
        monkeypatch,
        tmp_path,
        {
            "latency_mode": "real",
            "max_latency_seconds": 1,
            "schedule": [{"call": "version", "fault": "latency", "latency_seconds": 1}],
        },
    )
    started = time.perf_counter()
    assert "0.7.4" in supervisor.CLI(binary, tmp_path / "config").call("version", timeout=10)
    assert time.perf_counter() - started >= 1
