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
    assert client._issue_request("unused") == {"timeout": (30, 120)}
    assert FakeSession().request("GET", "https://example.invalid") == {"timeout": (30, 120)}
    assert client._issue_request("unused", timeout=None) == {"timeout": (30, 120)}


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


def test_idempotent_read_helpers_retry_transient_failure_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load("guard_read_retry_test")
    attempts: list[float] = []

    def fake_run_helper(*_args: object, timeout: float, **_kwargs: object) -> dict[str, object]:
        attempts.append(timeout)
        if len(attempts) < 3:
            raise guard.GuardError("guard helper failed: TimeoutExpired")
        return {"endpoint": "recovered"}

    monkeypatch.setattr(guard, "_run_helper", fake_run_helper)
    monkeypatch.setattr(guard.time, "sleep", lambda _delay: None)
    assert guard.recover_assignment("/private/python", str(uuid.uuid4())) == "recovered"
    assert len(attempts) == guard.READ_HELPER_ATTEMPTS
    assert all(0 < timeout <= guard.HELPER_TIMEOUT_SECONDS for timeout in attempts)


def test_mutating_unassign_helper_never_gets_automatic_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load("guard_unassign_no_retry_test")
    calls = 0

    def fake_run_helper(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise guard.GuardError("guard helper failed: TimeoutExpired")

    monkeypatch.setattr(guard, "_run_helper", fake_run_helper)
    with pytest.raises(guard.GuardError, match="TimeoutExpired"):
        guard.unassign_endpoint("/private/python", "ep")
    assert calls == 1


def test_idempotent_read_helper_refuses_when_its_shared_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load("guard_read_budget_test")
    calls = 0

    def fake_run_helper(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"endpoint": "unexpected"}

    monkeypatch.setattr(guard, "_run_helper", fake_run_helper)
    with pytest.raises(guard.GuardError, match="operation budget exhausted"):
        guard._run_read_helper("/private/python", "--recover-assignment", timeout=0)
    assert calls == 0


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
    assert calls[0]["timeout"] == (30, 120)
    assert calls[0]["url"] == "https://runtime.example/api/kernels/kid"


@pytest.mark.parametrize("outcome", ["ready", "404", "403", "malformed"])
def test_contents_readiness_is_metadata_only_and_does_not_require_a_kernel(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    guard = _load("guard_readiness_test")
    calls = []
    state = types.SimpleNamespace(url="https://runtime.example/", token="private", kernel_id=None)

    class Store:
        def __init__(self, _path: str) -> None:
            pass

        def get(self, _session: str) -> object:
            return state

    class HTTPError(Exception):
        def __init__(self, status: int) -> None:
            self.response = types.SimpleNamespace(status_code=status)

    class Session:
        def request(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))

    original = Session.request

    class Contents:
        def __init__(self, saved: object) -> None:
            assert saved is state

        def _request(self, method: str, path: str, params: object) -> object:
            assert (method, path, params) == ("GET", "/content", {"content": "0"})
            Session().request(method, path)
            if outcome == "404":
                raise FileNotFoundError(path)
            if outcome == "403":
                raise HTTPError(403)
            return {"type": "directory" if outcome == "ready" else "file"}

    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(sys.modules, "colab_cli.state", types.SimpleNamespace(StateStore=Store))
    monkeypatch.setitem(
        sys.modules, "colab_cli.contents", types.SimpleNamespace(ContentsClient=Contents)
    )
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(sessions=types.SimpleNamespace(Session=Session), HTTPError=HTTPError),
    )
    if outcome == "403":
        with pytest.raises(HTTPError):
            guard._runtime_ready_once("session.json", "new")
    elif outcome == "malformed":
        with pytest.raises(guard.GuardError, match="unknown state"):
            guard._runtime_ready_once("session.json", "new")
    else:
        assert guard._runtime_ready_once("session.json", "new") == {"ready": outcome == "ready"}
    assert calls[0][1]["timeout"] == (30, 120)
    assert Session.request is original


def test_runtime_readiness_helper_uses_single_bounded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load("guard_ready_helper_test")
    calls = []

    def run(*args: object, **kwargs: object) -> dict[str, bool]:
        calls.append((args, kwargs))
        return {"ready": False}

    monkeypatch.setattr(guard, "_run_helper", run)
    assert not guard.runtime_ready("/private/python", "config.json", "new", timeout=17)
    assert calls == [
        (("/private/python", "--runtime-ready", "config.json", "new"), {"timeout": 17})
    ]
