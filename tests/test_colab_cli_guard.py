"""Offline contracts for the Colab CLI safety interposition."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
import types
import uuid
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "research" / "colab_cli" / "cli_guard.py"


def _load(name: str = "colab_cli_guard_test") -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sitecustomize_hook_forces_hash_writes_nonsecret_durable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = uuid.uuid4()
    forced = uuid.uuid4()
    record = tmp_path / "nested" / "assignment.json"

    class FakeResponse:
        endpoint = "endpoint-123"

    class FakeClient:
        def _issue_request(self, *_args: object, **kwargs: object) -> object:
            return kwargs

        def assign(
            self, notebook_hash: uuid.UUID, *_args: object, **_kwargs: object
        ) -> FakeResponse:
            self.seen = notebook_hash
            return FakeResponse()

    client_module = types.ModuleType("colab_cli.client")
    client_module.Client = FakeClient
    package = types.ModuleType("colab_cli")

    class FakeSession:
        def request(self, *_args: object, **kwargs: object) -> object:
            return kwargs

    requests_module = types.ModuleType("requests")
    requests_module.sessions = types.SimpleNamespace(Session=FakeSession)
    monkeypatch.setitem(sys.modules, "colab_cli", package)
    monkeypatch.setitem(sys.modules, "colab_cli.client", client_module)
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setenv("SNAKE_COLAB_NOTEBOOK_HASH", str(forced))
    monkeypatch.setenv("SNAKE_COLAB_ASSIGNMENT_RECORD", str(record))
    _load("guard_hook_test")
    client = FakeClient()
    assert client.assign(requested).endpoint == "endpoint-123"
    assert client.seen == forced
    assert json.loads(record.read_text()) == {
        "endpoint": "endpoint-123",
        "notebook_hash": str(forced),
    }
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert client._issue_request("unused") == {"timeout": (10, 45)}
    assert FakeSession().request("GET", "https://example.invalid") == {"timeout": (10, 45)}


def test_helpers_use_bounded_private_python_and_validate_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    guard = _load("guard_helper_test")
    seen: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(args)
        if "--recover-assignment" in args:
            return subprocess.CompletedProcess(args, 0, '{"endpoint": "ep"}', "")
        if "--kernel-idle" in args:
            return subprocess.CompletedProcess(args, 0, '{"idle": false}', "")
        return subprocess.CompletedProcess(args, 0, '{"unassigned": true}', "")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    assert guard.recover_assignment("/private/python", str(uuid.uuid4())) == "ep"
    assert guard.kernel_idle("/private/python", tmp_path / "sessions.json", "lane") is False
    guard.unassign_endpoint("/private/python", "ep")
    assert seen[0][1:] == [str(MODULE_PATH.resolve()), "--recover-assignment", seen[0][-1]]
    assert "--kernel-idle" in seen[1]
    assert "--unassign-endpoint" in seen[2]


def test_recovery_get_is_read_only_and_timeout_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _load("guard_recovery_test")
    notebook_hash = str(uuid.uuid4())
    calls: list[tuple[str, object]] = []

    class Assignment:
        endpoint = "found-endpoint"

    class Client:
        def _get_assignment(self, value: uuid.UUID) -> Assignment:
            calls.append(("get", value))
            return Assignment()

        def _post_assignment(self, *_args: object) -> None:  # pragma: no cover - must not run
            raise AssertionError("recovery must not allocate")

    monkeypatch.setattr(guard, "_noninteractive_client", lambda: Client())
    assert guard._recover_assignment_once(notebook_hash) == {"endpoint": "found-endpoint"}
    assert calls == [("get", uuid.UUID(notebook_hash))]


def test_kernel_idle_accepts_only_known_idle_or_busy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    guard = _load("guard_idle_test")
    calls: list[dict[str, object]] = []

    class State:
        kernel_id = "kid"
        url = "https://runtime.example/"
        token = "proxy-secret"

    class Store:
        def __init__(self, _path: str) -> None:
            pass

        def get(self, _session: str) -> State:
            return State()

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"execution_state": "idle"}

    state_module = types.ModuleType("colab_cli.state")
    state_module.StateStore = Store
    package = types.ModuleType("colab_cli")
    monkeypatch.setitem(sys.modules, "colab_cli", package)
    monkeypatch.setitem(sys.modules, "colab_cli.state", state_module)

    requests_module = types.ModuleType("requests")

    def get(url: str, **kwargs: object) -> Response:
        calls.append({"url": url, **kwargs})
        return Response()

    requests_module.get = get
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    assert guard._kernel_idle_once(str(tmp_path / "sessions.json"), "lane") == {"idle": True}
    assert calls[0]["timeout"] == (10, 45)
    assert calls[0]["url"] == "https://runtime.example/api/kernels/kid"
