"""Offline identity negatives do not observe or instrument a live supervisor."""

from __future__ import annotations

import copy
import hashlib

import pytest

from research.bl_served_search.supervision_identity import (
    POLICY,
    QUALIFIED_SOURCES,
    heartbeat_request,
    process_id,
    rpc_request,
    verify_identity,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(template: str = "read_state") -> tuple[dict, list, list]:
    repo = "/other/worktree"
    guard = repo + "/research/colab_cli/cli_guard.py"
    hook = "/state/job/cli-hook/sitecustomize.py"
    hook_text = (
        "import os,runpy\ntry:\n    runpy.run_path("
        + repr(guard)
        + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
    )
    sources = {repo + "/" + path: digest for path, digest in QUALIFIED_SOURCES.items()}
    sources[repo + "/tools/sandbox/supervisor.py"] = sha("adapter")
    root = {
        "id": "train",
        "uid": 501,
        "pid": 101,
        "birth_ns": 1234567890123456,
        "boot_id": "host-boot-1",
        "interpreter": "/python",
        "module": "tools.sandbox.supervisor",
        "repo": repo,
        "cwd": repo,
        "state_dir": "/state",
        "source_hashes": sources,
        "startup_hook_hash": sha("no-hook"),
        "manifest": "/manifests/train.json",
        "manifest_digest": sha("manifest"),
        "assignment": "endpoint-1",
        "session": "train-1",
        "generation": "generation-01",
        "colab": "/bin/colab",
        "config": "/state/job/sessions.json",
        "cli_interpreter": "/cli/python",
        "cli_startup_hook_hash": sha(hook_text),
        "cli_source_hashes": {
            "/bin/colab": sha("CLI"),
            hook: sha(hook_text),
            guard: sources[guard],
        },
        "operation_timeout_seconds": 120,
    }
    root["argv"] = [
        root["interpreter"],
        "-m",
        root["module"],
        "--state-dir",
        root["state_dir"],
        "--colab",
        root["colab"],
        "run",
        root["manifest"],
    ]
    registry = {
        "policy": POLICY,
        "ruling_hash": sha("ruling"),
        "policy_source_hash": sha("policy"),
        "boot_id": root["boot_id"],
        "capture_start_ns": 0,
        "capture_end_ns": 5_000_000_000,
        "roots": [root],
    }

    def provenance(callsite: str) -> dict:
        source = repo + "/research/colab_cli/supervisor.py"
        return {
            "source": source,
            "source_hash": sources[source],
            "callsite": callsite,
            "independent": True,
        }

    nonce = "SNAKE_RPC_" + "a" * 32
    request = (
        rpc_request(template, root["generation"], nonce)
        if template in {"read_state", "renew_lease"}
        else heartbeat_request(root["generation"]) if template == "heartbeat_upload" else ""
    )
    callsite = (
        "CLI.renew_runtime_lease"
        if template in {"read_state", "renew_lease"}
        else "CLI._put_lease" if template == "heartbeat_upload" else "CLI." + template
    )
    operation = {
        "kind": "operation_start",
        "operation_id": "poll-1",
        "root_id": root["id"],
        "template": template,
        "request_text": request,
        "request_bytes": len(request.encode()),
        "request_digest": sha(request),
        "timeout_seconds": 120.0,
        "rpc_timeout_seconds": 120.0,
        "nonce": nonce,
        "payload_path": "/tmp/control.py",
        "provenance": provenance(callsite),
    }
    argv = [root["cli_interpreter"], root["colab"], "--auth", "oauth2", "--config", root["config"]]
    if template in {"usage", "sessions"}:
        argv += [template]
    elif template == "heartbeat_upload":
        argv += [
            "upload",
            "-s",
            root["session"],
            "/tmp/control.py",
            "/content/snake-cli/" + root["generation"] + "/lease-heartbeat.json",
        ]
    else:
        argv += ["exec", "-s", root["session"], "-f", "/tmp/control.py", "--timeout", "120.0"]
    identity = {
        "uid": root["uid"],
        "pid": 102,
        "birth_ns": 1234567891123456,
        "interpreter": root["cli_interpreter"],
        "argv": argv,
        "cwd": repo,
        "source_hashes": root["cli_source_hashes"],
        "startup_hook_hash": root["cli_startup_hook_hash"],
    }
    events = [
        {
            "kind": "root_identity",
            "root_id": root["id"],
            "identity": copy.deepcopy(root),
            "policy_source_hash": registry["policy_source_hash"],
        },
        operation,
        {
            "kind": "launch",
            "identity": identity,
            "parent": process_id(root),
            "operation_id": "poll-1",
            "launch_start_ns": 2_000_000_000,
            "launch_end_ns": 2_000_000_000,
        },
        {"kind": "exit", "pid": identity["pid"], "birth_ns": identity["birth_ns"]},
        {
            "kind": "operation_end",
            "operation_id": "poll-1",
            "response_bytes": 2,
            "response_digest": sha("{}"),
            "request_capture": {
                "bytes": len(request.encode()),
                "digest": sha(request),
                "independent": True,
            },
            "response_capture": {"bytes": 2, "digest": sha("{}"), "independent": True},
        },
        {
            "kind": "root_phase",
            "root_id": root["id"],
            "phase": "bookkeeping",
            "start_ns": 0,
            "end_ns": 5_000_000_000,
            "provenance": provenance("Broker.execute:monitor-bookkeeping"),
        },
    ]
    for index, event in enumerate(events):
        event.update(seq=index + 1, mono_ns=index * 1_000_000_000)
    observations = [
        {
            "mono_ns": 5_000_000_000,
            "root_id": root["id"],
            "identity": copy.deepcopy(root),
            "policy_source_hash": registry["policy_source_hash"],
        }
    ]
    return registry, events, observations


@pytest.mark.parametrize(
    "template", ["usage", "sessions", "read_state", "renew_lease", "heartbeat_upload"]
)
def test_correct_root_child_and_fixed_qualified_template(template: str) -> None:
    registry, events, observations = fixture(template)
    result = verify_identity(registry, events, observations)
    assert result["valid"], result["errors"]
    helper = result["lifecycles"][process_id(events[2]["identity"])]
    assert helper["parent_id"] == process_id(registry["roots"][0])
    assert helper["launch_ns"] == 2_000_000_000
    assert helper["exit_ns"] == 3_000_000_000
    assert result["verified_operations"]["poll-1"]["end_ns"] == 4_000_000_000


@pytest.mark.parametrize(
    "field,value",
    [
        ("uid", 502),
        ("pid", 999),
        ("birth_ns", 1234567890123457),
        ("assignment", "other"),
        ("session", "foreign"),
        ("generation", "other"),
        ("cwd", "/foreign"),
        ("repo", "/foreign"),
        ("interpreter", "/foreign/python"),
        ("boot_id", "other-boot"),
        ("source_hashes", {"/foreign.py": "0" * 64}),
        ("startup_hook_hash", "f" * 64),
        ("manifest_digest", "f" * 64),
        ("state_dir", "/foreign/state"),
        ("argv", ["/python", "-m", "tools.sandbox.supervisor", "--state-dir", "/spoof"]),
    ],
)
def test_every_observation_revalidates_root(field: str, value: object) -> None:
    registry, events, observations = fixture()
    observations[0]["identity"][field] = value
    assert not verify_identity(registry, events, observations)["valid"]


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_python",
        "pytest",
        "training",
        "generic_cli",
        "bulk_download",
        "wrong_uid",
        "wrong_source",
        "wrong_parent",
        "unobserved",
        "exec",
    ],
)
def test_helper_negative_grid(mutation: str) -> None:
    registry, events, observations = fixture()
    launch = events[2]
    identity = launch["identity"]
    if mutation == "foreign_python":
        identity["interpreter"] = "/other/python"
    elif mutation in {"pytest", "training"}:
        identity["argv"] = ["/cli/python", "-m", "pytest" if mutation == "pytest" else "src.train"]
    elif mutation in {"generic_cli", "bulk_download"}:
        identity["argv"][-1] = "download" if mutation == "bulk_download" else "print(1)"
    elif mutation == "wrong_uid":
        identity["uid"] = 502
    elif mutation == "wrong_source":
        identity["source_hashes"] = {"/evil.py": sha("evil")}
    elif mutation == "wrong_parent":
        launch["parent"] = "999@1"
    elif mutation == "unobserved":
        launch["kind"] = "observation"
    else:
        launch["kind"] = "exec"
    assert not verify_identity(registry, events, observations)["valid"]


@pytest.mark.parametrize(
    "mutation",
    [
        "generation",
        "injection",
        "nonce",
        "heartbeat",
        "destination",
        "auth",
        "config",
        "session",
        "timeout",
        "arbitrary_tag",
    ],
)
def test_operation_semantics_cannot_be_registered_away(mutation: str) -> None:
    registry, events, observations = fixture(
        "heartbeat_upload" if mutation in {"heartbeat", "destination"} else "read_state"
    )
    operation, identity = events[1], events[2]["identity"]
    if mutation in {"generation", "injection", "heartbeat"}:
        text = operation["request_text"]
        text = (
            text.replace("generation-01", "foreign")
            if mutation == "generation"
            else text + "\nprint(1)"
        )
        operation.update(
            request_text=text, request_bytes=len(text.encode()), request_digest=sha(text)
        )
        events[4]["request_capture"] = {
            "bytes": len(text.encode()),
            "digest": sha(text),
            "independent": True,
        }
    elif mutation == "nonce":
        operation["nonce"] = "SNAKE_RPC_" + "a" * 32 + "';import os"
    elif mutation == "destination":
        identity["argv"][-1] = "/content/bulk.tar"
    elif mutation in {"auth", "config", "session"}:
        index = {"auth": 3, "config": 5, "session": 8}[mutation]
        identity["argv"][index] = "foreign"
    elif mutation == "timeout":
        operation["rpc_timeout_seconds"] = 121
    else:
        operation["template"] = "read_state;pytest"
    assert not verify_identity(registry, events, observations)["valid"]


@pytest.mark.parametrize("total,valid", [(65535, True), (65536, True), (65537, False)])
def test_request_plus_response_exact_64k_boundary(total: int, valid: bool) -> None:
    registry, events, observations = fixture()
    end = events[4]
    size = total - events[1]["request_bytes"]
    end.update(response_bytes=size, response_digest=sha("response of declared size"))
    end["response_capture"] = {"bytes": size, "digest": end["response_digest"], "independent": True}
    assert verify_identity(registry, events, observations)["valid"] is valid


@pytest.mark.parametrize(
    "mutation",
    [
        "no_response_capture",
        "wrong_response_digest",
        "no_exit",
        "journal_gap",
        "root_restart",
        "missing_phase",
        "phase_tag",
        "untrusted_callsite",
        "identity_gap",
        "qualified_source",
        "policy",
    ],
)
def test_incomplete_or_untrusted_evidence_is_invalid(mutation: str) -> None:
    registry, events, observations = fixture()
    if mutation == "no_response_capture":
        events[4].pop("response_capture")
    elif mutation == "wrong_response_digest":
        events[4]["response_digest"] = sha("wrong")
    elif mutation == "no_exit":
        events[3]["kind"] = "unknown_exit"
    elif mutation == "journal_gap":
        events[3]["seq"] = 7
    elif mutation == "root_restart":
        events[3].update(pid=101, birth_ns=registry["roots"][0]["birth_ns"])
    elif mutation == "missing_phase":
        events.pop()
    elif mutation == "phase_tag":
        events[5]["phase"] = "pytest"
    elif mutation == "untrusted_callsite":
        events[5]["provenance"]["callsite"] = "Supervisor.execute:credential-refresh"
    elif mutation == "identity_gap":
        registry["capture_end_ns"] = observations[0]["mono_ns"] = 6_000_000_000
    elif mutation == "qualified_source":
        source = "/other/worktree/research/colab_cli/supervisor.py"
        registry["roots"][0]["source_hashes"][source] = sha("unqualified replacement")
    else:
        registry["policy"] = "classified-v1"
    assert not verify_identity(registry, events, observations)["valid"]


def test_reparented_helper_and_pid_reuse_do_not_acquire_identity() -> None:
    registry, events, observations = fixture()
    second = copy.deepcopy(events[2])
    second.update(seq=6, mono_ns=4_500_000_000)
    second["identity"]["birth_ns"] += 1
    events.insert(5, second)
    events[-1]["seq"] = 7
    assert not verify_identity(registry, events, observations)["valid"]


def test_response_digest_is_dynamic_and_not_registered() -> None:
    registry, events, observations = fixture("usage")
    assert "response_digest" not in str(registry)
    for response in ("usage changed", "another account read"):
        size, digest = len(response.encode()), sha(response)
        events[4].update(
            response_bytes=size,
            response_digest=digest,
            response_capture={"bytes": size, "digest": digest, "independent": True},
        )
        assert verify_identity(registry, events, observations)["valid"]


def qualified_fixture(seconds: int = 65, template: str = "read_state") -> tuple[dict, list, list]:
    """Return source-bound SYNTHETIC full-window evidence for composite replay tests."""
    registry, events, observations = fixture(template)
    end_ns = seconds * 1_000_000_000
    registry["capture_end_ns"] = end_ns
    events[-1].update(mono_ns=end_ns, end_ns=end_ns)
    final = observations[0]
    observations = [
        dict(copy.deepcopy(final), mono_ns=time_ns)
        for time_ns in range(5_000_000_000, end_ns + 1, 5_000_000_000)
    ]
    if not observations or observations[-1]["mono_ns"] != end_ns:
        observations.append(dict(copy.deepcopy(final), mono_ns=end_ns))
    return registry, events, observations


def test_full_admission_fixture_retains_every_root_phase_and_observation() -> None:
    registry, events, observations = qualified_fixture()
    result = verify_identity(registry, events, observations)
    assert result["valid"], result["errors"]
    assert len(observations) == 13
    assert result["verified_phases"]["train"][0]["end_ns"] == 65_000_000_000


@pytest.mark.parametrize(
    "malformed",
    [None, [], {"policy": POLICY, "roots": [None]}, {"policy": POLICY, "roots": [{"pid": []}]}],
)
def test_malformed_identity_evidence_fails_closed(malformed: object) -> None:
    assert not verify_identity(malformed, [], [])["valid"]


def test_launch_cpu_support_starts_at_earliest_possible_birth() -> None:
    registry, events, observations = fixture()
    events[2].update(launch_start_ns=1_500_000_000, launch_end_ns=1_900_000_000)
    result = verify_identity(registry, events, observations)
    assert result["valid"], result["errors"]
    child = result["lifecycles"][process_id(events[2]["identity"])]
    assert child["launch_ns"] == 1_500_000_000
    assert child["launch_observed_ns"] == 2_000_000_000


@pytest.mark.parametrize(
    "bracket",
    [
        (None, None),
        (3_000_000_000, 2_000_000_000),
        (0, 1_500_000_000),
        (2_000_000_000, 3_000_000_000),
    ],
)
def test_unknown_or_impossible_launch_brackets_are_refused(bracket: tuple) -> None:
    registry, events, observations = fixture()
    events[2].update(launch_start_ns=bracket[0], launch_end_ns=bracket[1])
    assert not verify_identity(registry, events, observations)["valid"]


def test_operation_completion_deadline_is_enforced() -> None:
    registry, events, observations = fixture()
    events[1].update(timeout_seconds=2.0, rpc_timeout_seconds=2.0)
    events[2]["identity"]["argv"][-1] = "2.0"
    result = verify_identity(registry, events, observations)
    assert not result["valid"]
    assert "operation final evidence exceeds qualified deadline" in result["errors"]


@pytest.mark.parametrize("force,template", [(False, "renew_lease"), (True, "read_state")])
def test_request_matches_actual_qualified_generator(
    tmp_path, monkeypatch, force: bool, template: str
) -> None:
    """Exercise qualified code offline, intercepting call BEFORE any CLI launch."""
    import re
    from pathlib import Path

    from research.colab_cli.supervisor import CLI

    cli = CLI("/never/launch", tmp_path / "never-read-config.json", "session")
    captured = []

    def capture(*args, **kwargs):
        assert args[:4] == ("exec", "-s", "session", "-f")
        script = Path(args[4]).read_text()
        nonce = re.search(r"SNAKE_RPC_[0-9a-f]{32}", script).group()
        captured.append((script, nonce))
        assert kwargs["timeout"] <= 120
        return nonce + '{"ok":true,"value":{}}'

    monkeypatch.setattr(cli, "call", capture)
    assert (
        cli.renew_runtime_lease("/content/snake-cli/generation-01", "generation-01", force=force)
        == {}
    )
    script, nonce = captured[0]
    assert script == rpc_request(template, "generation-01", nonce)


def test_heartbeat_matches_actual_qualified_generator(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from research.colab_cli.supervisor import CLI

    cli = CLI("/never/launch", tmp_path / "never-read-config.json", "session")
    captured = []

    def capture(*args, **kwargs):
        assert args[:3] == ("upload", "-s", "session")
        assert args[4] == "/content/snake-cli/generation-01/lease-heartbeat.json"
        assert kwargs["timeout"] == 120
        captured.append(Path(args[3]).read_text())
        return "{}"

    monkeypatch.setattr(cli, "call", capture)
    cli._put_lease("/content/snake-cli/generation-01", "generation-01", 120)
    assert captured == [heartbeat_request("generation-01")]


@pytest.mark.parametrize("template", ["usage", "sessions"])
def test_account_read_matches_actual_qualified_call(tmp_path, monkeypatch, template: str) -> None:
    from research.colab_cli.supervisor import CLI

    cli = CLI("/never/launch", tmp_path / "never-read-config.json", "session")
    captured = []

    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        return (
            "Current balance: 10\nUsage rate: 1\nActive assignments: 1\n"
            if template == "usage"
            else "No active sessions found on server."
        )

    monkeypatch.setattr(cli, "call", capture)
    getattr(cli, template)()
    assert captured == [((template,), {"timeout": 120.0})]


@pytest.mark.parametrize("complete_capture", [False, True])
def test_every_transport_attempt_spends_payload_allowance(complete_capture: bool) -> None:
    registry, events, observations = fixture()
    launch = copy.deepcopy(events[2])
    launch["identity"]["pid"] = 103
    launch["identity"]["birth_ns"] += 1
    launch.update(mono_ns=2_500_000_000, launch_start_ns=2_250_000_000, launch_end_ns=2_500_000_000)
    exit_record = {
        "kind": "exit",
        "pid": 103,
        "birth_ns": launch["identity"]["birth_ns"],
        "mono_ns": 3_500_000_000,
    }
    events[3:3] = [launch]
    events.insert(5, exit_record)
    for index, event in enumerate(events, 1):
        event["seq"] = index
    if complete_capture:
        request = events[1]["request_text"] * 2
        end = next(row for row in events if row["kind"] == "operation_end")
        end["request_capture"] = {
            "bytes": len(request.encode()),
            "digest": sha(request),
            "independent": True,
        }
    result = verify_identity(registry, events, observations)
    assert result["valid"] is complete_capture, result["errors"]
    if complete_capture:
        assert (
            result["verified_operations"]["poll-1"]["accounted_request_bytes"]
            == events[1]["request_bytes"] * 2
        )
