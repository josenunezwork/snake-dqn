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


@pytest.mark.parametrize(
    ("inventory", "expected", "error"),
    [
        ([], False, None),
        ([{"id": "cold/id", "execution_state": "idle"}], True, None),
        ([{"id": "cold/id", "execution_state": "busy"}], False, None),
        ([{"id": "cold/id", "execution_state": "starting"}], False, None),
        ([{"id": "cold/id", "execution_state": "unknown"}], None, "execution_state"),
        ([{"id": "", "execution_state": "idle"}], None, "malformed kernel identity"),
        ([{"id": 123, "execution_state": "idle"}], None, "malformed kernel identity"),
        ([{"execution_state": "idle"}], None, "malformed kernel identity"),
        ([None], None, "malformed kernel identity"),
        ({"execution_state": "idle"}, None, "kernel inventory"),
        (
            [{"id": "a", "execution_state": "idle"}, {"id": "b", "execution_state": "idle"}],
            None,
            "ambiguous",
        ),
    ],
)
def test_cold_kernel_discovery_saves_only_unique_explicit_idle_identity(
    monkeypatch: pytest.MonkeyPatch, inventory: object, expected: bool | None, error: str | None
) -> None:
    guard = _load("guard_cold_kernel_test")
    state = types.SimpleNamespace(
        kernel_id=None, url="https://tracked.example/", token="private", endpoint="owned-endpoint"
    )
    saved = []
    calls = []

    class Store:
        def __init__(self, path: str) -> None:
            assert path == "sessions.json"

        def get(self, session: str) -> object:
            assert session == "cold"
            return state

        def add(self, value: object) -> None:
            saved.append(vars(value).copy())

    def get(url: str, **kwargs: object) -> object:
        calls.append((url, kwargs))
        body = {"execution_state": "idle"} if len(calls) > 1 else inventory
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(sys.modules, "colab_cli.state", types.SimpleNamespace(StateStore=Store))
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    if error:
        with pytest.raises(guard.GuardError, match=error):
            guard._kernel_idle_once("sessions.json", "cold")
    else:
        assert guard._kernel_idle_once("sessions.json", "cold") == {"idle": expected}
    assert calls[0][0] == "https://tracked.example/api/kernels"
    assert calls[0][1]["timeout"] == (30, 120)
    assert saved == (
        [dict(kernel_id="cold/id", url=state.url, token=state.token, endpoint=state.endpoint)]
        if expected is True
        else []
    )
    # A recovered ID is subsequently addressed exactly, including URL quoting.
    if expected is True:
        assert guard._kernel_idle_once("sessions.json", "cold") == {"idle": True}
        assert calls[1][0] == "https://tracked.example/api/kernels/cold%2Fid"


@pytest.mark.parametrize("missing", ["session", "url", "token"])
def test_kernel_discovery_refuses_missing_runtime_identity_without_network(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    guard = _load("guard_missing_runtime_test")
    state = types.SimpleNamespace(kernel_id=None, url="https://tracked.example", token="secret")
    if missing != "session":
        setattr(state, missing, "")
    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(
        sys.modules,
        "colab_cli.state",
        types.SimpleNamespace(
            StateStore=lambda _config: types.SimpleNamespace(
                get=lambda _session: None if missing == "session" else state
            )
        ),
    )
    # No get() exists: accidental discovery would fail this test.
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    with pytest.raises(guard.GuardError, match="lacks kernel probe metadata"):
        guard._kernel_idle_once("config.json", "cold")


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


@pytest.fixture
def creation_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Exercise the installed sitecustomize hook at its real requests boundary."""
    notebook_hash = str(uuid.uuid4())
    record = tmp_path / "assignment.json"
    record.write_text(json.dumps({"endpoint": "owned", "notebook_hash": notebook_hash}))
    state = types.SimpleNamespace(
        kernel_id=None, url="https://owned.example/", token="private", endpoint="owned"
    )
    saved = tmp_path / "saved.json"
    calls = []
    outcome = {"body": {"id": "created-by-us"}, "error": None}

    class Store:
        def __init__(self, config: str) -> None:
            assert config == "owned-config"

        def get(self, session: str) -> object:
            assert session == "owned-session"
            return state

        def add(self, value: object) -> None:
            saved.write_text(json.dumps(vars(value)))

    class Client:
        def assign(self, *_args: object, **_kwargs: object) -> None:
            pass

        def _issue_request(self, *_args: object, **_kwargs: object) -> None:
            pass

    class Session:
        def request(self, method: str, url: str, **kwargs: object) -> object:
            calls.append((method, url, kwargs))
            if outcome["error"]:
                raise outcome["error"]
            return types.SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: outcome["body"]
            )

    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(sys.modules, "colab_cli.client", types.SimpleNamespace(Client=Client))
    monkeypatch.setitem(sys.modules, "colab_cli.state", types.SimpleNamespace(StateStore=Store))
    requests = types.SimpleNamespace(sessions=types.SimpleNamespace(Session=Session))
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setenv("SNAKE_COLAB_NOTEBOOK_HASH", notebook_hash)
    monkeypatch.setenv("SNAKE_COLAB_ASSIGNMENT_RECORD", str(record))
    monkeypatch.setenv("SNAKE_COLAB_CONFIG", "owned-config")
    monkeypatch.setenv("SNAKE_COLAB_SESSION", "owned-session")
    guard = _load("guard_creation_hook")
    return types.SimpleNamespace(
        guard=guard,
        state=state,
        saved=saved,
        calls=calls,
        outcome=outcome,
        requests=requests,
        session=Session(),
        record=record,
        url="https://owned.example/api/kernels",
        headers={"X-Colab-Runtime-Proxy-Token": "private"},
    )


def test_creation_id_survives_websocket_timeout_and_ignores_unowned_kernels(
    creation_hook: object,
) -> None:
    fixture = creation_hook
    fixture.session.request("POST", fixture.url, headers=fixture.headers)
    # This point precedes websocket start_channels(), which may now time out.
    assert json.loads(fixture.saved.read_text())["kernel_id"] == "created-by-us"
    # Model the upstream runtime retry: it must not create a second kernel.
    with pytest.raises(fixture.guard.GuardError, match="already exists"):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    assert len(fixture.calls) == 1
    gets = []

    def get(url: str, **_kwargs: object) -> object:
        gets.append(url)
        # The tracked runtime also has another kernel. Inventory discovery is
        # ambiguous; the saved creation identity addresses only our own kernel.
        body = (
            [
                {"id": "other", "execution_state": "idle"},
                {"id": "created-by-us", "execution_state": "busy"},
            ]
            if url.endswith("/api/kernels")
            else {"id": "created-by-us", "execution_state": "busy"}
        )
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    fixture.requests.get = get
    assert fixture.guard._kernel_idle_once("owned-config", "owned-session") == {"idle": False}
    assert gets == [fixture.url + "/created-by-us"]
    fixture.requests.get = lambda *_args, **_kwargs: types.SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"id": "created-by-us", "execution_state": "idle"},
    )
    assert fixture.guard._kernel_idle_once("owned-config", "owned-session") == {"idle": True}


@pytest.mark.parametrize("mismatch", ["url", "token", "endpoint", "notebook", "config", "session"])
def test_creation_hook_refuses_foreign_or_missing_identity_before_post(
    creation_hook: object, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    fixture = creation_hook
    if mismatch == "url":
        fixture.url = "https://foreign.example/api/kernels"
    elif mismatch == "token":
        fixture.headers = {"X-Colab-Runtime-Proxy-Token": "foreign"}
    elif mismatch == "endpoint":
        fixture.state.endpoint = "foreign"
    elif mismatch == "notebook":
        fixture.record.write_text(json.dumps({"endpoint": "owned", "notebook_hash": "foreign"}))
    else:
        monkeypatch.delenv("SNAKE_COLAB_" + mismatch.upper())
    with pytest.raises(fixture.guard.GuardError, match="identity"):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    assert not fixture.calls
    assert not fixture.saved.exists()


@pytest.mark.parametrize("body", [{}, {"id": ""}, {"id": 3}, [], {"id": "  "}])
def test_creation_hook_refuses_malformed_creation_response(
    creation_hook: object, body: object
) -> None:
    fixture = creation_hook
    fixture.outcome["body"] = body
    with pytest.raises(fixture.guard.GuardError, match="malformed kernel identity"):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    assert not fixture.saved.exists()
    with pytest.raises(fixture.guard.GuardError, match="uncertain retry"):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    assert len(fixture.calls) == 1


def test_creation_hook_never_repeats_uncertain_post(creation_hook: object) -> None:
    fixture = creation_hook
    fixture.outcome["error"] = TimeoutError("creation response lost")
    with pytest.raises(TimeoutError):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    fixture.outcome["error"] = None
    with pytest.raises(fixture.guard.GuardError, match="uncertain retry"):
        fixture.session.request("POST", fixture.url, headers=fixture.headers)
    assert len(fixture.calls) == 1
    assert fixture.state.kernel_id is None
    assert not fixture.saved.exists()


def test_internal_startup_retry_recovers_only_after_exact_saved_kernel_idle(
    creation_hook: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import cli_guard, policy, supervisor

    fixture = creation_hook
    cli = supervisor.CLI("fake", tmp_path / "config", "owned-session")
    cli.oauth_python = "unused"
    cli.config = "owned-config"
    events = []

    def call(*_args: str, **_kwargs: object) -> str:
        if not events:
            fixture.session.request("POST", fixture.url, headers=fixture.headers)
            events.append("created-before-websocket-timeout")
            # Upstream retries startup using its still-empty in-memory ID.
            try:
                fixture.session.request("POST", fixture.url, headers=fixture.headers)
            except fixture.guard.GuardError as exc:
                raise supervisor.Refused(str(exc)) from exc
        assert events[-1] == "idle-proved"
        wrapped = Path(_args[_args.index("-f") + 1]).read_text()
        nonce = "SNAKE_RPC_" + wrapped.split("SNAKE_RPC_")[1][:32]
        events.append("reconnected")
        return nonce + '{"ok":true,"value":{"ready":true}}'

    def get(url: str, **_kwargs: object) -> object:
        assert cli.exec_uncertain
        assert url == fixture.url + "/created-by-us"
        events.append("idle-proved")
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": "created-by-us", "execution_state": "idle"},
        )

    fixture.requests.get = get
    monkeypatch.setattr(cli, "call", call)
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        cli_guard,
        "kernel_idle",
        lambda _python, config, session, **_kwargs: fixture.guard._kernel_idle_once(
            config, session
        )["idle"],
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _delay: None)
    cli.wait_runtime_ready(tmp_path)
    assert events == ["created-before-websocket-timeout", "idle-proved", "reconnected"]
    assert len(fixture.calls) == 1
    assert not cli.exec_uncertain
    assert not policy.transient_error(
        fixture.guard.GuardError("kernel creation already attempted; refusing uncertain retry")
    )
