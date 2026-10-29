#!/usr/bin/env python3
"""Small subprocess-backed stand-in for the official Colab CLI.

The fake keeps account state on disk so every invocation is a real transport
boundary.  Tests use it to prove that the broker passes the auth/config flags,
does not retry an uncertain kernel call, and independently verifies releases.
It intentionally implements only the commands used by ``research.colab_cli``.
"""

from __future__ import annotations

import ast
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing {name}")
    return Path(value)


STATE = _env_path("FAKE_COLAB_STATE")
LOG = _env_path("FAKE_COLAB_LOG")
REMOTE = _env_path("FAKE_COLAB_REMOTE")
MODE = os.environ.get("FAKE_COLAB_MODE", "normal")
ACCEPTANCE_MODE = MODE == "acceptance"


def _acceptance_profile(code: str) -> str:
    """Read the profile from the immutable runtime spec embedded by the broker.

    The fake must exercise the same profile selection that reaches the remote
    launcher; a process environment value here would only test the fixture.
    """
    match = re.search(r"['\"]COLAB_ACCEPTANCE_PROFILE['\"]:\s*['\"]([^'\"]+)", code)
    if match is None:
        # The production default is final-only. Keeping that default here also
        # prevents an omitted manifest environment binding from silently
        # exercising the old full-state fixture.
        return "final-only"
    return match.group(1)


def _load() -> dict[str, Any]:
    if not STATE.exists():
        return {"balance": 1178.30, "rate": 6.84, "sessions": {}, "exec_count": {}}
    return json.loads(STATE.read_text())


def _save(value: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, STATE)


def _log(
    argv: list[str], auth: str | None, config: str | None, *, fault: dict[str, Any] | None = None
) -> None:
    entry = {
        "argv": argv,
        "call_id": os.environ.get("FAKE_CALL_ID"),
        "auth": auth,
        "config": config,
        "assign_retries": os.environ.get("COLAB_CLI_ASSIGN_RETRIES"),
        "request_timeout": os.environ.get("COLAB_CLI_REQUEST_TIMEOUT"),
    }
    if len(argv) > 4 and argv[4] == "upload":
        entry["upload_bytes"] = Path(argv[-2]).stat().st_size
    if len(argv) > 4 and argv[4] == "exec" and "-f" in argv:
        code = Path(argv[argv.index("-f") + 1]).read_text()
        operation = re.search(r"SNAKE_TRANSFER_([A-Z_]+)", code)
        if operation:
            entry["transfer_operation"] = operation.group(1)
    if fault is not None:
        # Latency is virtual: property tests exercise deadline/error handling
        # without making a many-seed suite sleep for minutes.
        entry["fault"] = fault
    with LOG.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _call_class(op: str) -> str:
    """Return the transport class used by the offline fault plan."""
    return {
        "new": "allocate",
        "exec": "kernel_exec",
        "stop": "stop",
        "sessions": "sessions",
        "usage": "usage",
        "upload": "upload",
        "download": "download",
    }.get(op, "read")


def _fault_for(
    state: dict[str, Any], op: str, *, session: str = "", command: list[str] | None = None
) -> dict[str, Any] | None:
    """Pick one reproducible fake transport fault without waiting in real time.

    ``FAKE_COLAB_FAULT_PLAN`` supports an exact ``schedule`` for regression
    cases and a seeded random mode for property tests.  A scheduled entry may
    target either an operation (``call``) or one of the classes above.  The
    phase is ``pre`` by default; ``post`` makes mutating operations commit
    before their ambiguous client-visible failure.
    """
    raw = os.environ.get("FAKE_COLAB_FAULT_PLAN")
    if not raw:
        return None
    plan = json.loads(raw)
    number = int(state.get("fault_call_number", 0)) + 1
    state["fault_call_number"] = number
    call_class = _call_class(op)
    selected: dict[str, Any] | None = None
    used = state.setdefault("fault_schedule_used", [])
    command_text = " ".join(command or ())
    if op == "exec" and command and "-f" in command:
        try:
            command_text += "\n" + Path(command[command.index("-f") + 1]).read_text()
        except OSError:
            # The real CLI will report a missing script; a fault selector must
            # not obscure that product error.
            pass
    for index, entry in enumerate(plan.get("schedule", [])):
        if index in used:
            continue
        if (
            ("at" not in entry or int(entry["at"]) == number)
            and entry.get("call", entry.get("class"))
            in {
                op,
                call_class,
                "any",
            }
            and (not entry.get("session_contains") or str(entry["session_contains"]) in session)
            and (not entry.get("contains") or str(entry["contains"]) in command_text)
        ):
            selected = dict(entry)
            used.append(index)
            break
    if selected is None and plan.get("random"):
        probability = float(plan.get("probability", 0.0))
        allowed = plan.get("classes", {}).get(call_class, plan.get("faults", []))
        if allowed:
            rng = random.Random(f"{plan.get('seed', 0)}:{number}:{op}")
            if rng.random() < probability:
                selected = {
                    "fault": rng.choice(allowed),
                    "phase": "post" if op == "new" else rng.choice(["pre", "post"]),
                }
    if selected is None:
        return None
    fault = str(selected.get("fault", "latency"))
    if fault not in {"latency", "timeout", "reset", "http_5xx"}:
        raise SystemExit(f"unsupported fake fault: {fault}")
    result = {
        "call": op,
        "class": call_class,
        "fault": fault,
        "phase": str(selected.get("phase", "pre")),
        "virtual_latency_seconds": 0.0,
    }
    if fault == "latency":
        maximum = float(plan.get("max_latency_seconds", 0.0))
        rng = random.Random(f"{plan.get('seed', 0)}:{number}:{op}:latency")
        delay = float(selected.get("latency_seconds", rng.uniform(0.0, maximum)))
        if not 0 <= delay <= maximum:
            raise ValueError("scheduled latency exceeds configured maximum")
        result["virtual_latency_seconds"] = round(delay, 6)
    return result


def _fault_exit(fault: dict[str, Any]) -> int:
    kind = fault["fault"]
    if kind == "timeout" or fault.get("timed_out") is True:
        print("TimeoutError: fake request deadline exceeded", file=sys.stderr)
    elif kind == "reset":
        print("ConnectionError: fake connection reset", file=sys.stderr)
    elif kind == "http_5xx":
        print("HTTP 503: fake transient provider failure", file=sys.stderr)
    else:
        return 0
    return 1


def _remote(session: str, remote_path: str) -> Path:
    if not remote_path.startswith("/"):
        raise SystemExit(f"remote path must be absolute: {remote_path}")
    path = REMOTE / session / remote_path.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_recoverable(record: Path | None, result: Any) -> None:
    """Finish the fake-side durable record for shortcut-only kernel paths."""
    if record is None:
        return
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": "completed", "result": result}, sort_keys=True))
    os.replace(temporary, record)


def _nonce(code: str) -> str:
    match = re.search(r"SNAKE_RPC_[0-9a-f]+", code)
    if not match:
        raise SystemExit("missing RPC nonce")
    return match.group(0)


def _reply(code: str, value: Any = True, *, ok: bool = True, error: str = "") -> None:
    response = {"ok": ok}
    if ok:
        response["value"] = value
    else:
        response["error"] = error
    print(_nonce(code) + json.dumps(response, sort_keys=True))


def _root_from_code(code: str) -> str:
    root_pattern = r"/content/snake-cli/[^/'\"]+"
    match = re.search(rf"(?:root|p)=Path\((['\"])({root_pattern})\1\)", code)
    if match is None:
        match = re.search(rf"Path\((['\"])({root_pattern})[^'\"]*\1\)", code)
    if match is None:
        match = re.search(rf"(['\"])({root_pattern})", code)
    if not match:
        raise SystemExit("RPC did not include a snake-cli root")
    return match.group(2)


def _new_session(state: dict[str, Any], name: str, gpu: str) -> None:
    endpoint = f"gpu-{gpu.lower()}-fake-{len(state['sessions']) + 1:04d}"
    state["sessions"][endpoint] = {
        "name": name,
        "gpu": gpu,
        "notebook_hash": os.environ.get("SNAKE_COLAB_NOTEBOOK_HASH"),
    }
    state["exec_count"][name] = 0
    state["allocation_count"] = int(state.get("allocation_count", 0)) + 1
    state.setdefault("allocation_names", []).append(name)
    state["allocated_usage_reads"] = 0
    state["rate"] = float(state["rate"]) + float(os.environ.get("FAKE_COLAB_OWN_RATE", "6.84"))
    config = os.environ.get("FAKE_COLAB_ACTIVE_CONFIG")
    if config:
        config_path = Path(config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {}
        existing[name] = {
            "name": name,
            "endpoint": endpoint,
            "url": "https://fake-runtime.invalid/",
            "token": "fake-proxy-token",
            "kernel_id": None,
        }
        config_path.write_text(json.dumps(existing, sort_keys=True))
    assignment_record = os.environ.get("SNAKE_COLAB_ASSIGNMENT_RECORD")
    if assignment_record:
        Path(assignment_record).write_text(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "notebook_hash": str(uuid.UUID(os.environ["SNAKE_COLAB_NOTEBOOK_HASH"])),
                }
            )
        )
    print(f"Created session {name}")


def _tick(state: dict[str, Any], now: float) -> None:
    for session, runtime in state.get("runtime", {}).items():
        expiry = runtime.get("renewed_at", now) + runtime.get("lease_seconds", 180)
        if now >= expiry:
            root = runtime["root"]
            checkpoint = _remote(session, root + "/output/lease-checkpoint.pkl")
            payload = b"fixture-lease-checkpoint"
            if not checkpoint.exists():
                checkpoint.write_bytes(payload)
                checkpoint.with_name(checkpoint.name + ".sha256").write_text(
                    hashlib.sha256(payload).hexdigest()
                )
            if now >= expiry + 15:
                for endpoint, record in list(state["sessions"].items()):
                    if record["name"] == session:
                        del state["sessions"][endpoint]
                        state["rate"] = max(
                            0, state["rate"] - float(os.environ.get("FAKE_COLAB_OWN_RATE", "0.5"))
                        )
                        state.setdefault("watchdog_released", []).append(session)
    _save(state)


def _handle_exec(state: dict[str, Any], session: str, code: str) -> None:
    # Most fixture paths execute the broker's wrapper verbatim.  A few
    # acceptance shortcuts model a detached runtime instead; give those paths
    # the same durable acknowledgement behavior without ever writing /content
    # on the Mac running the test.
    record_match = re.search(
        r"_snake_record=_SnakePath\((['\"])(/content/\.snake-rpc-[0-9a-f]+\.json)\1\)", code
    )
    recoverable_record: Path | None = None
    if record_match:
        remote_record = _remote(session, record_match.group(2))
        recoverable_record = remote_record
        if remote_record.exists():
            saved = json.loads(remote_record.read_text())
            if saved.get("status") == "completed":
                _reply(code, saved["result"])
                _save(state)
                return
        # Transfer and source-delivery paths run this wrapper themselves.  Map
        # only its record so their ``open('x')`` and atomic replace remain real
        # filesystem operations in the fake endpoint namespace.
        code = code.replace(record_match.group(2), str(remote_record))
    count = int(state.setdefault("exec_count", {}).get(session, 0))
    state["exec_count"][session] = count + 1
    if MODE in {"busy_exec", "poll_hiccup"} and count == 0:
        # A zero exit with no nonce is the ambiguous/busy-kernel case.
        print("kernel is busy; command accepted")
        _save(state)
        return
    if MODE == "kernel_error":
        _reply(code, ok=False, error="RuntimeError('fake kernel failure')")
        _save(state)
        return
    if "TRANSPORT_OUTAGE_READ" in code:
        _reply(code, True)
        _save(state)
        return
    if "SNAKE_RUNTIME_READINESS" in code:
        # Model the official CLI callback after acknowledged kernel startup.
        # The separate dispatch regression forces loss before this callback.
        config_path = Path(os.environ["FAKE_COLAB_ACTIVE_CONFIG"])
        sessions = json.loads(config_path.read_text())
        sessions[session]["kernel_id"] = sessions[session].get("kernel_id") or "fake-kernel-id"
        config_path.write_text(json.dumps(sessions, sort_keys=True))
        _reply(code, {"ready": True})
        _save(state)
        return

    # The acceptance observer's deliberately killed client is a plain CLI
    # exec, not an RPC wrapper. Model its durable completion markers without
    # sleeping for the production 120 seconds.
    if "client-start.json" in code and "client-finished.json" in code:
        root = _root_from_code(code)
        local_root = _remote(session, root)
        request = re.search(r"request=([^\n]+)", code)
        value = ast.literal_eval(request.group(1)) if request else "fixture-request"
        (local_root / "client-start.json").write_text(
            json.dumps({"request": value, "started_at": 0})
        )
        state.setdefault("pending_client_completion", {})[session] = {
            "root": root,
            "request": value,
        }
        _save(state)
        return

    if "after_silence" in code:
        _reply(code, {"after_silence": True})
        _save(state)
        return

    if "SNAKE_TRANSFER_" in code:
        match = re.search(r"spec = (\{[^\n]+\})", code)
        spec = json.loads(match.group(1))
        for key in ("parts", "temporary", "target", "source", "path"):
            if key in spec:
                spec[key] = str(_remote(session, spec[key]))
        code = code[: match.start(1)] + repr(spec) + code[match.end(1) :]
        if MODE == "corrupt_download" and "DOWNLOAD_SPLIT" in code:
            pass
        exec(compile(code, "<fake-transfer>", "exec"), {})
        _save(state)
        return

    if "acceptance-corrupt.bin" in code:
        root = _root_from_code(code)
        target = _remote(session, root) / "output" / "acceptance-corrupt.bin"
        if "write_bytes" in code:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"known-good")
            target.with_name(target.name + ".sha256").write_text("0" * 64 + "\n")
            target.with_name(target.name + ".complete").write_text("complete\n")
        else:
            for item in (
                target,
                target.with_name(target.name + ".sha256"),
                target.with_name(target.name + ".complete"),
            ):
                item.unlink(missing_ok=True)
        _reply(code, True)
        _save(state)
        return

    if "--resume-input" in code and "resume.json" in code:
        root = _root_from_code(code)
        local_root = _remote(session, root)
        checkpoint = local_root / "resume-test" / "input.pkl"
        payload = {
            "resumed_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "steps": 9,
        }
        target = local_root / "resume-test" / "output" / "resume.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        target.with_name(target.name + ".sha256").write_text(digest + "\n")
        target.with_name(target.name + ".complete").write_text("complete\n")
        result = {"path": "resume.json", "size": target.stat().st_size, "sha256": digest}
        _reply(code, result)
        _save(state)
        return

    if MODE == "poll_hiccup":
        _reply(code, True)
        _save(state)
        return

    root = _root_from_code(code)
    local_root = _remote(session, root)
    local_root.mkdir(parents=True, exist_ok=True)
    if "SNAKE_EXPORT_BATCH" in code:
        # Exercise the real bounded archive construction and durable operation
        # fence, replacing only the fake VM's filesystem prefix.
        code = code.replace(root, str(local_root))
        code = code.replace(
            "'path': str(archive)",
            f"'path': str(archive).replace({str(local_root)!r}, {root!r}, 1)",
        )
        exec(compile(code, "<fake-export-batch>", "exec"), {})
        _save(state)
        return
    if "SNAKE_SOURCE_DELIVERY" in code:
        # Execute the actual uploaded source helper. Only the network/git
        # boundary is faked; identity checks and fallback extraction are real.
        plan = ast.literal_eval(re.search(r"plan=(\{[^\n]+\})", code).group(1))
        repository = Path(
            os.environ.get("FAKE_COLAB_SOURCE_REPO", Path(__file__).resolve().parents[2])
        )

        def fake_source_run(argv: list[str], *, cwd: Path, timeout: float) -> str:
            if "fetch" in argv:
                state.setdefault("source_fetches", []).append(argv)
                if MODE == "github_unreachable":
                    raise subprocess.CalledProcessError(128, argv, "Could not resolve host")
                return ""
            if "rev-parse" in argv:
                expression = plan["commit"]
                if argv[-1].endswith("^{tree}"):
                    expression += "^{tree}"
                if MODE == "github_identity_mismatch":
                    return "0" * 40
                return subprocess.check_output(
                    ["git", "rev-parse", expression], cwd=repository, text=True
                ).strip()
            if "checkout" in argv:
                (local_root / "repo" / "fetched-source").write_text(plan["commit"])
            return ""

        code = code.replace(f"root=Path({root!r})", f"root=Path({str(local_root)!r})")
        code = code.replace(
            "s.loader.exec_module(mod)",
            "s.loader.exec_module(mod); mod._run = fake_source_run",
        )
        exec(compile(code, "<fake-source-delivery>", "exec"), {"fake_source_run": fake_source_run})
        _save(state)
        return
    if "spec.json" in code and "runtime.py" in code and "subprocess.Popen" in code:
        launches = state.setdefault("job_launches", {})
        launches[session] = int(launches.get(session, 0)) + 1
        generation_match = re.search(r"'generation': '([^']+)'", code)
        generation = generation_match.group(1) if generation_match else "fake-generation"
        output = local_root / "output"
        output.mkdir(parents=True, exist_ok=True)
        # Run the pinned production HTTP/size/hash/atomic-install helper, faking
        # only urlopen. The outer broker and its release receipt remain real.
        launch_spec = next(
            (
                ast.literal_eval(node)
                for node in ast.walk(ast.parse(code))
                if isinstance(node, ast.Dict)
                and any(
                    isinstance(key, ast.Constant) and key.value == "public_inputs"
                    for key in node.keys
                )
            ),
            {},
        )
        public_rows = launch_spec.get("public_inputs", [])
        public_failure = None
        if public_rows:
            encoded = re.search(r"base64.b64decode\('([A-Za-z0-9+/=]+)'\)", code).group(1)
            runtime_path = local_root / "runtime.py"
            runtime_path.write_bytes(base64.b64decode(encoded))
            module_spec = importlib.util.spec_from_file_location(
                "fake_pinned_runtime", runtime_path
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            payload_path = os.environ.get("FAKE_COLAB_PUBLIC_INPUT_PATH")
            payload = (
                Path(payload_path).read_bytes()
                if payload_path
                else base64.b64decode(os.environ["FAKE_COLAB_PUBLIC_INPUT_BYTES"])
            )
            fault = os.environ.get("FAKE_COLAB_PUBLIC_INPUT_FAULT", "")

            def public_reply(*_args: Any, **_kwargs: Any) -> io.BytesIO:
                time.sleep(0.02)  # Delayed completion must precede any trainer marker.
                if fault == "timeout":
                    raise URLError("fixture network timeout")
                return io.BytesIO(b"x" * len(payload) if fault == "mismatch" else payload)

            module.urlopen = public_reply
            try:
                verified = module._fetch_public_inputs(local_root, public_rows)
                (local_root / "public-inputs-verified.json").write_text(json.dumps(verified))
                state.setdefault("public_inputs_verified", {})[session] = verified
            except RuntimeError as exc:
                public_failure = str(exc)
                state.setdefault("public_input_failures", {})[session] = public_failure
        profile = _acceptance_profile(code)
        if profile not in {"final-only", "full-state"}:
            raise SystemExit(f"unsupported acceptance profile: {profile}")
        is_l4 = "-l4" in session
        if profile == "final-only":
            # This is deliberately the real S1 admission payload size.  The
            # supervisor's transfer and local SHA verification remain real;
            # only the provider boundary is faked.
            artifact_name = "final-artifact.bin" if is_l4 else "client-job.bin"
            artifact_size = 64 * 1024 * 1024 if is_l4 else 4096
            fixtures = {
                artifact_name: b"f" * artifact_size,
                "intentional-child-failure.json": b'{"exit_code":23}',
            }
        else:
            checkpoint = b"fixture-toy-checkpoint"
            checkpoint_digest = hashlib.sha256(checkpoint).hexdigest()
            toy_resume = {
                "exact": True,
                "scope": "fixture only; not evidence for a trainer",
                "uninterrupted_sha256": checkpoint_digest,
                "steps": 9,
            }
            fixtures = {
                "busy-snapshot.bin": (
                    b"s" * (9 * 1024 * 1024 + 17) if ACCEPTANCE_MODE else b"fixture-snapshot"
                ),
                "toy-checkpoint.pkl": checkpoint,
                "toy-resume.json": json.dumps(toy_resume).encode(),
                "checkpoint-overhead.json": b'{"overhead_percent":0.0}',
                "intentional-child-failure.json": b'{"exit_code":23}',
            }
        if public_failure:
            fixtures = {}
        exports: list[dict[str, Any]] = []
        for name, payload in fixtures.items():
            target = output / name
            target.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            target.with_name(name + ".sha256").write_text(digest + "\n")
            target.with_name(name + ".complete").write_text("complete\n")
            exports.append({"path": name, "size": len(payload), "sha256": digest})
        if not ACCEPTANCE_MODE:
            exports = []
        terminal: dict[str, Any] = {
            "job_id": "fake-job",
            "generation": generation,
            "status": "success",
            "returncode": 0,
            "reason": None,
            "exports": exports,
        }
        if public_failure:
            terminal.update(status="failed", returncode=74, reason="public_input_fetch_failed")
        if MODE in {"corrupt_download", "export_success"}:
            payload = b"GOOD"
            checkpoint = output / "checkpoint.bin"
            checkpoint.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (output / "checkpoint.bin.sha256").write_text(digest + "\n")
            (output / "checkpoint.bin.complete").write_text("complete\n")
            terminal["exports"] = [
                {"path": "checkpoint.bin", "size": len(payload), "sha256": digest}
            ]
        state.setdefault("runtime", {})[session] = {
            "root": root,
            "generation": generation,
            "renewed_at": float(os.environ.get("FAKE_COLAB_NOW", "0")),
            "lease_seconds": float(re.search(r"'lease_seconds': ([0-9.]+)", code).group(1)),
            "state": {
                "job_id": "fake-job",
                "generation": generation,
                "status": "succeeded",
                "max_export_bytes": 1024 * 1024,
            },
            "terminal": terminal,
            "probe": (
                {
                    "phase": "transfer_ready",
                    "pid": 4242,
                    "execution_id": "fixture-execution",
                    "snapshot": {
                        "path": (
                            "final-artifact.bin"
                            if profile == "final-only" and is_l4
                            else (
                                "client-job.bin" if profile == "final-only" else "busy-snapshot.bin"
                            )
                        ),
                        "bytes": len(
                            fixtures[
                                (
                                    "final-artifact.bin"
                                    if profile == "final-only" and is_l4
                                    else (
                                        "client-job.bin"
                                        if profile == "final-only"
                                        else "busy-snapshot.bin"
                                    )
                                )
                            ]
                        ),
                        "sha256": hashlib.sha256(
                            fixtures[
                                (
                                    "final-artifact.bin"
                                    if profile == "final-only" and is_l4
                                    else (
                                        "client-job.bin"
                                        if profile == "final-only"
                                        else "busy-snapshot.bin"
                                    )
                                )
                            ]
                        ).hexdigest(),
                    },
                    "measurement": {"overhead_percent": 0.0},
                    "profile": profile,
                }
                if ACCEPTANCE_MODE
                else {}
            ),
        }
        _cache_recoverable(recoverable_record, {"pid": 4242})
        _reply(code, {"pid": 4242})
        _save(state)
        return

    if "checks=" in code and "ready.json" in code:
        if session in state.get("public_inputs_verified", {}):
            assert (local_root / "public-inputs-verified.json").is_file()
            state.setdefault("trainer_starts", []).append(session)

    if "renew_lease" in code:
        runtime = state.setdefault("runtime", {}).get(session)
        if runtime is None:
            _reply(code, ok=False, error="RuntimeError('runtime has not started')")
        else:
            runtime["renewed_at"] = float(os.environ.get("FAKE_COLAB_NOW", "0"))
            _reply(code, True)
        _save(state)
        return

    if "read_state" in code:
        runtime = state.setdefault("runtime", {}).get(session)
        if runtime is None:
            _reply(code, ok=False, error="RuntimeError('runtime has not started')")
        else:
            if "renew=True" in code:
                runtime["renewed_at"] = float(os.environ.get("FAKE_COLAB_NOW", "0"))
            if "disable_contents=True" in code:
                runtime["contents_disabled"] = True
            runtime_value = {
                "state": runtime["state"],
                "terminal": runtime["terminal"],
                "exports": runtime["terminal"].get("exports", []),
            }
            # Normal broker polling imports runtime.read_state directly. The
            # acceptance observer wraps that value with its own probe JSON.
            value: dict[str, Any] = runtime_value
            if "acceptance-state.json" in code:
                value = {"runtime": runtime_value, "probe": runtime.get("probe", {})}
            _reply(
                code,
                value,
            )
        _save(state)
        return

    _cache_recoverable(recoverable_record, True)
    _reply(code, True)
    _save(state)


def _main(argv: list[str]) -> int:
    auth = None
    config = None
    args = list(argv)
    if len(args) < 4 or args[0] != "--auth" or args[2] != "--config":
        print("expected --auth oauth2 --config PATH", file=sys.stderr)
        return 2
    auth, config = args[1], args[3]
    command = args[4:]
    session = ""
    for index, value in enumerate(command):
        if value == "-s" and index + 1 < len(command):
            session = command[index + 1]
            break
    if not command:
        return 2
    op = command[0]
    os.environ["FAKE_COLAB_ACTIVE_CONFIG"] = config
    state = _load()
    now = float(os.environ.get("FAKE_COLAB_NOW", "0"))
    if ACCEPTANCE_MODE:
        _tick(state, now)
    fault = _fault_for(state, op, session=session, command=command)
    budget = float(os.environ.get("FAKE_COLAB_TIMEOUT", "120"))
    if fault and (fault["fault"] == "timeout" or fault["virtual_latency_seconds"] > budget):
        fault["virtual_latency_seconds"] = budget
        fault["timed_out"] = True
    _log(argv, auth, config, fault=fault)
    real_latency = (
        json.loads(os.environ.get("FAKE_COLAB_FAULT_PLAN", "{}")).get("latency_mode") == "real"
    )
    if fault and real_latency and fault["phase"] == "pre":
        _save(state)
        time.sleep(fault["virtual_latency_seconds"])
    # A latency-only event is deliberately a virtual elapsed interval. The
    # caller can assert it was injected without slowing the safety suite.
    if (
        fault is not None
        and (fault["fault"] != "latency" or fault.get("timed_out"))
        and fault["phase"] == "pre"
    ):
        _save(state)
        return _fault_exit(fault)
    if MODE == "error_url" and op == "version":
        print(
            "request failed: https://colab.test/api?access_token=secret-token&"
            "refresh_token=refresh-secret",
            file=sys.stderr,
        )
        return 1
    if op == "version":
        print("Colab CLI 0.7.4")
    elif op in {"url", "whoami"}:
        print("fixture")
    elif op == "sessions":
        if not state["sessions"]:
            print("No active sessions found on server.")
        for endpoint, record in state["sessions"].items():
            print(
                f"[{record['name']}] {endpoint} | Hardware: {record['gpu']} "
                "| Shape: Standard | Variant: GPU"
            )
    elif op == "usage":
        rate = float(state["rate"])
        rates = json.loads(os.environ.get("FAKE_COLAB_USAGE_RATES", "[]"))
        assignments = json.loads(os.environ.get("FAKE_COLAB_USAGE_ASSIGNMENTS", "[]"))
        if rates and "allocated_usage_reads" in state and state["sessions"]:
            index = state["allocated_usage_reads"]
            rate = float(rates[min(index, len(rates) - 1)])
            state["allocated_usage_reads"] = index + 1
            _save(state)
        elif assignments and "allocated_usage_reads" in state and state["sessions"]:
            index = state["allocated_usage_reads"]
            state["allocated_usage_reads"] = index + 1
            _save(state)
        else:
            index = 0
        print(f"Current balance: {float(state['balance']):.2f} compute units")
        print(f"Usage rate: {rate:.2f}/hr")
        count = (
            assignments[min(index, len(assignments) - 1)]
            if assignments and "allocated_usage_reads" in state
            else len(state["sessions"])
        )
        print(f"Active assignments: {count}")
    elif op == "new":
        name = session
        gpu = "CPU"
        if "--gpu" in command:
            gpu = command[command.index("--gpu") + 1]
        _new_session(state, name, gpu)
        _save(state)
    elif op == "stop":
        attempts = state.get("stop_attempts", 0) + 1
        state["stop_attempts"] = attempts
        if MODE == "stop_transient" and attempts < 3:
            _save(state)
            print("[colab] Error: No active sessions found. Create one with 'colab new'.")
            return 1
        if MODE != "stop_lie":
            for endpoint, record in list(state["sessions"].items()):
                if record["name"] == session:
                    del state["sessions"][endpoint]
                    state["rate"] = max(
                        0,
                        float(state["rate"]) - float(os.environ.get("FAKE_COLAB_OWN_RATE", "6.84")),
                    )
        _save(state)
        print(f"Stopped {session}")
    elif op == "upload":
        if ACCEPTANCE_MODE and not any(r["name"] == session for r in state["sessions"].values()):
            print("No active session found", file=sys.stderr)
            return 1
        source, destination = command[-2:]
        if MODE == "upload_transient" and not state.get("upload_failed_once"):
            state["upload_failed_once"] = True
            _remote(session, destination).write_bytes(Path(source).read_bytes()[:3])
            _save(state)
            print("ConnectionError: connection reset after partial PUT", file=sys.stderr)
            return 1
        if ACCEPTANCE_MODE and destination.endswith("/lease-heartbeat.json"):
            root_path = REMOTE / session / str(Path(destination).parent).lstrip("/")
            if not root_path.exists():
                print("HTTP 404: runtime root not created", file=sys.stderr)
                _save(state)
                return 1
        shutil.copyfile(source, _remote(session, destination))
        runtime = state.get("runtime", {}).get(session)
        if (
            runtime
            and destination.endswith("/lease-heartbeat.json")
            and not runtime.get("contents_disabled")
        ):
            runtime["renewed_at"] = now
        if destination.endswith("/client-release.json"):
            pending = state.get("pending_client_completion", {}).pop(session, None)
            if pending:
                completed = _remote(session, pending["root"] + "/client-finished.json")
                completed.write_text(json.dumps({"request": pending["request"], "finished_at": 1}))
                runtime = state.get("runtime", {}).get(session)
                if runtime:
                    runtime["probe"]["phase"] = "transfer_finished"
                _save(state)
        print(f"Uploaded {destination}")
    elif op == "download":
        if ACCEPTANCE_MODE and not any(r["name"] == session for r in state["sessions"].values()):
            print("No active session found", file=sys.stderr)
            return 1
        source, destination = command[-2:]
        source_path = _remote(session, source)
        if source.endswith("/client-finished.json") and not source_path.exists():
            print("HTTP 404: fixture completion is not yet visible", file=sys.stderr)
            return 1
        if not source_path.exists():
            print("HTTP 404: file not found", file=sys.stderr)
            _save(state)
            return 1
        payload = source_path.read_bytes()
        if MODE == "corrupt_download" and not source.endswith((".sha256", ".complete")):
            payload = b"CORRUPT"
        Path(destination).write_bytes(payload)
        print(f"Downloaded {source}")
    elif op == "exec":
        if ACCEPTANCE_MODE and not any(r["name"] == session for r in state["sessions"].values()):
            print("No active session found", file=sys.stderr)
            return 1
        code_path = command[command.index("-f") + 1]
        _handle_exec(state, session, Path(code_path).read_text())
    else:
        print(f"unsupported fake command: {op}", file=sys.stderr)
        return 2
    _save(state)
    if (
        fault is not None
        and (fault["fault"] != "latency" or fault.get("timed_out"))
        and fault["phase"] == "post"
    ):
        # ``new``, ``exec`` and ``stop`` reach this path after their state
        # transition. This models an ambiguous client timeout/reset after the
        # provider accepted a mutation, which is the dangerous retry case.
        return _fault_exit(fault)
    return 0


def main(argv: list[str]) -> int:
    """Serialize fake-provider state transitions across lease-heartbeat calls."""
    lock = STATE.with_suffix(STATE.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            result = _main(argv)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if result == 0 and "exec" in argv and "-f" in argv:
        code = Path(argv[argv.index("-f") + 1]).read_text()
        if "client-start.json" in code and "client-finished.json" in code:
            time.sleep(300)  # Kernel state remains durable after this CLI dies.
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
