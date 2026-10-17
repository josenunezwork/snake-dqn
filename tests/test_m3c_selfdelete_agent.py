"""Owner-key handoff tests for the M3-C pod agent.

These start only a loopback HTTP server and replace the RunPod transport; they never consult
the macOS Keychain or create a pod.
"""

import hashlib
import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1] / "research/redesign_m3c_20261008/gpu_agent_v2.py"
FAKE_KEY = "rp_owner_restricted_fake_key_never_persist"


def _server(tmp_path, monkeypatch, **env):
    token = "agent-test-token"
    monkeypatch.setenv("M3_ROOT", str(tmp_path / "r"))
    monkeypatch.setenv("M3_TOKEN_SHA256", hashlib.sha256(token.encode()).hexdigest())
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    spec = importlib.util.spec_from_file_location("m3c_selfdelete_agent", AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return mod, server, f"http://127.0.0.1:{server.server_port}", token


def _request(base, token, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    return _raw_request(base, token, path, data)


def _raw_request(base, token, path, data):
    req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
    req.add_header("X-M3-Token", token)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_owner_key_endpoint_is_authenticated_memory_only_and_selects_probe_delete(
    tmp_path, monkeypatch
):
    mod, server, base, token = _server(
        tmp_path, monkeypatch, RUNPOD_POD_ID="pod-test", RUNPOD_API_KEY="injected-key"
    )
    calls = []

    def fake_runpod(method, url, body=None, key=None):
        calls.append((method, url, body, key))
        return 200 if method == "GET" else 204

    monkeypatch.setattr(mod, "_runpod", fake_runpod)
    try:
        status, body = _request(base, "wrong", "/self-delete-key", {"key": FAKE_KEY})
        assert status == 403 and FAKE_KEY not in body.decode()

        status, body = _request(base, token, "/self-delete-key", {"key": FAKE_KEY})
        assert status == 200
        assert json.loads(body) == {"self_delete_armed": True, "self_delete_probe": 200}
        assert calls == [("GET", "https://rest.runpod.io/v1/pods/pod-test", None, FAKE_KEY)]
        assert mod.SELF_DELETE_KEY == [FAKE_KEY]

        mod.self_delete()
        assert calls[-1] == ("DELETE", "https://rest.runpod.io/v1/pods/pod-test", None, FAKE_KEY)
        health = json.loads(_request(base, token, "/health")[1])
        assert health["self_delete_armed"] is True and health["self_delete_probe"] == 200

        # The endpoint body and every agent-visible response/log/artifact remain secret-free.
        (tmp_path / "r/out/result.json").write_text(json.dumps({"ok": True}))
        _, pulled = _request(base, token, "/out/result.json")
        assert FAKE_KEY not in pulled.decode()
        assert all(
            FAKE_KEY.encode() not in path.read_bytes()
            for path in (tmp_path / "r").rglob("*")
            if path.is_file()
        )
    finally:
        server.shutdown()


def test_owner_key_invalid_body_is_generic_and_exec_cannot_receive_key(tmp_path, monkeypatch):
    mod, server, base, token = _server(tmp_path, monkeypatch, RUNPOD_POD_ID="pod-test")
    monkeypatch.setattr(mod, "_runpod", lambda *args, **kwargs: 200)
    try:
        status, body = _request(base, token, "/self-delete-key", {"key": ""})
        assert status == 400 and body == b'{"error": "invalid self-delete key"}'
        status, body = _raw_request(
            base, token, "/self-delete-key", b'{"key": "' + FAKE_KEY.encode()
        )
        assert status == 400 and FAKE_KEY not in body.decode()
        status, body = _request(base, token, "/self-delete-key", [FAKE_KEY])
        assert status == 400 and FAKE_KEY not in body.decode()
        status, body = _request(
            base, token, "/self-delete-key", {"key": FAKE_KEY, "extra": FAKE_KEY}
        )
        assert status == 400 and FAKE_KEY not in body.decode()
        status, body = _request(base, token, "/self-delete-key", {"key": "x" * 8193})
        assert status == 400 and FAKE_KEY not in body.decode()
        status, body = _request(base, token, "/self-delete-key", {"key": FAKE_KEY})
        assert status == 200 and FAKE_KEY not in body.decode()

        marker = tmp_path / "child-env.txt"
        payload = {
            "name": "env-check",
            "argv": [sys.executable, "-c", "import os; print(os.getenv('RUNPOD_API_KEY', ''))"],
            "cwd": str(tmp_path),
            "env": {"RUNPOD_API_KEY": FAKE_KEY},
        }
        assert _request(base, token, "/exec", payload)[0] == 200
        log_path = tmp_path / "r/logs/env-check.log"
        for _ in range(40):
            if log_path.exists() and mod.PROCS["env-check"].poll() is not None:
                break
            time.sleep(0.05)
        assert mod.PROCS["env-check"].poll() == 0
        assert FAKE_KEY.encode() not in log_path.read_bytes()
        assert marker.exists() is False
    finally:
        server.shutdown()


def test_injected_key_fallback_remains_when_no_owner_key_is_installed(tmp_path, monkeypatch):
    mod, server, _base, _token = _server(
        tmp_path, monkeypatch, RUNPOD_POD_ID="pod-test", RUNPOD_API_KEY="legacy-injected"
    )
    calls = []
    monkeypatch.setattr(
        mod,
        "_runpod",
        lambda method, url, body=None, key=None: calls.append((method, key)) or 403,
    )
    try:
        assert mod.probe_self_delete() == 403
        mod.self_delete()
        assert calls[0] == ("GET", "legacy-injected")
        assert calls[1] == ("DELETE", "legacy-injected")
    finally:
        server.shutdown()


def test_agent_log_redacts_owner_key_and_runpod_refuses_redirects(tmp_path, monkeypatch):
    mod, server, _base, _token = _server(tmp_path, monkeypatch)
    hits = []

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/target")
                self.end_headers()
            else:
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=redirect.serve_forever, daemon=True).start()
    try:
        mod.SELF_DELETE_KEY[0] = FAKE_KEY
        mod.log("owner key was " + FAKE_KEY)
        assert FAKE_KEY not in (tmp_path / "r/logs/agent.log").read_text()
        assert "[redacted]" in (tmp_path / "r/logs/agent.log").read_text()
        assert mod._runpod("GET", f"http://127.0.0.1:{redirect.server_port}/start") == 302
        assert hits == []
    finally:
        redirect.shutdown()
        server.shutdown()


def test_self_delete_epoch_and_deadman_use_owner_key(tmp_path, monkeypatch):
    mod, server, _base, _token = _server(tmp_path, monkeypatch, RUNPOD_POD_ID="pod-test")
    calls = []
    monkeypatch.setattr(
        mod,
        "_runpod",
        lambda method, url, body=None, key=None: calls.append((method, key)) or 204,
    )
    original_sleep = mod.time.sleep

    def one_cycle(_seconds):
        if getattr(one_cycle, "called", False):
            raise RuntimeError("stop reaper")
        one_cycle.called = True

    try:
        mod.SELF_DELETE_KEY[0] = FAKE_KEY
        monkeypatch.setattr(mod.time, "time", lambda: 1000.0)
        monkeypatch.setattr(mod.time, "sleep", one_cycle)
        mod.SELF_DELETE, mod.DEADMAN, mod.DEADLINE = 999.0, 0.0, 0.0
        try:
            mod.reaper()
        except RuntimeError as exc:
            assert str(exc) == "stop reaper"
        assert calls == [("DELETE", FAKE_KEY)]

        calls.clear()
        one_cycle.called = False
        mod.SELF_DELETE, mod.DEADMAN, mod.DEADLINE = 0.0, 1.0, 0.0
        mod.LAST_AUTH[0] = 0.0
        try:
            mod.reaper()
        except RuntimeError as exc:
            assert str(exc) == "stop reaper"
        assert calls == [("DELETE", FAKE_KEY)]
    finally:
        mod.time.sleep = original_sleep
        server.shutdown()


def test_real_runpod_request_uses_owner_bearer_for_probe_and_delete(tmp_path, monkeypatch):
    from types import SimpleNamespace

    mod, server, _base, _token = _server(
        tmp_path, monkeypatch, RUNPOD_POD_ID="pod-test", RUNPOD_API_KEY="legacy-injected"
    )
    requests = []

    class Response:
        def __init__(self, status):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_request(req, **kwargs):
        requests.append(req)
        return Response(200 if req.method == "GET" else 204)

    monkeypatch.setattr(
        mod.urllib.request, "build_opener", lambda *handlers: SimpleNamespace(open=open_request)
    )
    try:
        assert mod.install_self_delete_key(FAKE_KEY)["self_delete_probe"] == 200
        mod.self_delete()
        assert [req.method for req in requests] == ["GET", "DELETE"]
        for req in requests:
            assert req.full_url == "https://rest.runpod.io/v1/pods/pod-test"
            assert req.get_header("Authorization") == "Bearer " + FAKE_KEY
            assert req.data is None
        assert FAKE_KEY not in (tmp_path / "r/logs/agent.log").read_text()
    finally:
        server.shutdown()
