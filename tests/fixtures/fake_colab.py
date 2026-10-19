#!/usr/bin/env python3
"""Small subprocess-backed stand-in for the official Colab CLI.

The fake keeps account state on disk so every invocation is a real transport
boundary.  Tests use it to prove that the broker passes the auth/config flags,
does not retry an uncertain kernel call, and independently verifies releases.
It intentionally implements only the commands used by ``research.colab_cli``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing {name}")
    return Path(value)


STATE = _env_path("FAKE_COLAB_STATE")
LOG = _env_path("FAKE_COLAB_LOG")
REMOTE = _env_path("FAKE_COLAB_REMOTE")
MODE = os.environ.get("FAKE_COLAB_MODE", "normal")


def _load() -> dict[str, Any]:
    if not STATE.exists():
        return {"balance": 1178.30, "rate": 6.84, "sessions": {}, "exec_count": {}}
    return json.loads(STATE.read_text())


def _save(value: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, STATE)


def _log(argv: list[str], auth: str | None, config: str | None) -> None:
    entry = {
        "argv": argv,
        "auth": auth,
        "config": config,
        "assign_retries": os.environ.get("COLAB_CLI_ASSIGN_RETRIES"),
        "request_timeout": os.environ.get("COLAB_CLI_REQUEST_TIMEOUT"),
    }
    if len(argv) > 4 and argv[4] == "upload":
        entry["upload_bytes"] = Path(argv[-2]).stat().st_size
    with LOG.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _remote(session: str, remote_path: str) -> Path:
    if not remote_path.startswith("/"):
        raise SystemExit(f"remote path must be absolute: {remote_path}")
    path = REMOTE / session / remote_path.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    state["rate"] = float(state["rate"]) + float(os.environ.get("FAKE_COLAB_OWN_RATE", "6.84"))
    config = os.environ.get("FAKE_COLAB_ACTIVE_CONFIG")
    if config:
        config_path = Path(config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {}
        existing[name] = {"endpoint": endpoint}
        config_path.write_text(json.dumps(existing, sort_keys=True))
    print(f"Created session {name}")


def _handle_exec(state: dict[str, Any], session: str, code: str) -> None:
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

    if MODE == "poll_hiccup":
        _reply(code, True)
        _save(state)
        return

    root = _root_from_code(code)
    local_root = _remote(session, root)
    local_root.mkdir(parents=True, exist_ok=True)
    if "spec.json" in code and "runtime.py" in code and "subprocess.Popen" in code:
        generation_match = re.search(r"'generation': '([^']+)'", code)
        generation = generation_match.group(1) if generation_match else "fake-generation"
        output = local_root / "output"
        output.mkdir(parents=True, exist_ok=True)
        terminal: dict[str, Any] = {
            "job_id": "fake-job",
            "generation": generation,
            "status": "success",
            "returncode": 0,
            "reason": None,
            "exports": [],
        }
        if MODE == "corrupt_download":
            payload = b"GOOD"
            checkpoint = output / "checkpoint.bin"
            checkpoint.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (output / "checkpoint.bin.sha256").write_text(digest + "\n")
            (output / "checkpoint.bin.complete").write_text("complete\n")
            terminal["exports"] = [
                {"path": "output/checkpoint.bin", "size": len(payload), "sha256": digest}
            ]
        state.setdefault("runtime", {})[session] = {
            "root": root,
            "generation": generation,
            "state": {
                "job_id": "fake-job",
                "generation": generation,
                "status": "succeeded",
                "max_export_bytes": 1024 * 1024,
            },
            "terminal": terminal,
        }
        _reply(code, {"pid": 4242})
        _save(state)
        return

    if "read_state" in code:
        runtime = state.setdefault("runtime", {}).get(session)
        if runtime is None:
            _reply(code, ok=False, error="RuntimeError('runtime has not started')")
        else:
            _reply(
                code,
                {
                    "state": runtime["state"],
                    "terminal": runtime["terminal"],
                    "exports": runtime["terminal"].get("exports", []),
                },
            )
        _save(state)
        return

    _reply(code, True)
    _save(state)


def main(argv: list[str]) -> int:
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
    _log(argv, auth, config)
    os.environ["FAKE_COLAB_ACTIVE_CONFIG"] = config
    state = _load()

    if not command:
        return 2
    op = command[0]
    if MODE == "error_url" and op == "version":
        print(
            "request failed: https://colab.test/api?access_token=secret-token&"
            "refresh_token=refresh-secret",
            file=sys.stderr,
        )
        return 1
    if op == "version":
        print("Colab CLI 0.7.4")
    elif op == "sessions":
        if not state["sessions"]:
            print("No active sessions found on server.")
        for endpoint, record in state["sessions"].items():
            print(
                f"[{record['name']}] {endpoint} | Hardware: {record['gpu']} "
                "| Shape: Standard | Variant: GPU"
            )
    elif op == "usage":
        print(f"Current balance: {float(state['balance']):.2f} compute units")
        print(f"Usage rate: {float(state['rate']):.2f}/hr")
        print(f"Active assignments: {len(state['sessions'])}")
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
        source, destination = command[-2:]
        shutil.copyfile(source, _remote(session, destination))
        print(f"Uploaded {destination}")
    elif op == "download":
        source, destination = command[-2:]
        source_path = _remote(session, source)
        payload = source_path.read_bytes()
        if MODE == "corrupt_download" and not source.endswith((".sha256", ".complete")):
            payload = b"CORRUPT"
        Path(destination).write_bytes(payload)
        print(f"Downloaded {source}")
    elif op == "exec":
        code_path = command[command.index("-f") + 1]
        _handle_exec(state, session, Path(code_path).read_text())
    else:
        print(f"unsupported fake command: {op}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
