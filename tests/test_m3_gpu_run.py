"""Pure parts of the M3-B GPU runner (no RunPod call is ever made under pytest)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from research.redesign_m3_20261008 import gpu_run


def test_pod_body_is_money_safe():
    body = gpu_run.pod_body("m3b-gpu-x", "a" * 64, 1000.0, 2000.0)
    assert body["cloudType"] == "SECURE" and body["gpuCount"] == 1
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert body["env"]["M3_DEADLINE_EPOCH"] == "1000"
    assert body["env"]["M3_SELF_DELETE_EPOCH"] == "2000"
    assert "RUNPOD_API_KEY" not in body["env"] and body["volumeInGb"] == 0
    assert body["minRAMPerGPU"] >= 72 and body["minVCPUPerGPU"] >= 10
    assert gpu_run.MAX_HOURLY <= 1.0 and gpu_run.G2_RATE >= 4 * gpu_run.MAC_RATE


def _g2(rate, left, dead=(), seeds=None):
    seeds = gpu_run.SEEDS if seeds is None else seeds
    rates = {s: rate for s in seeds}
    done = {s: rate * 1800 for s in seeds}
    return gpu_run.g2_decision(rates, done, left, dead)


def test_g2_decision():
    ok = _g2(1500.0, 6.5 * 3600)
    assert ok["pass"] and ok["combined_rate"] == 7500.0
    slow = _g2(1000.0, 6.5 * 3600)
    assert not slow["pass"] and slow["reason"] == "rate"
    late = _g2(1200.0, 3 * 3600)
    assert not late["pass"] and late["reason"] == "lifetime"
    assert not _g2(9000.0, 6 * 3600, seeds=(0,))["pass"]
    dead = _g2(2000.0, 6.5 * 3600, dead=(2,))
    assert not dead["pass"] and dead["reason"] == "rate" and dead["combined_rate"] == 8000.0


def test_spend_cap():
    assert not gpu_run.spend_exceeded(66.0, 60.0, 8.0)
    assert gpu_run.spend_exceeded(66.0, 58.4, 8.0)


def test_commands():
    argv = gpu_run.train_argv(3)
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--transitions") + 1] == str(20_000_000)
    setup = gpu_run.setup_argv("f" * 64)[2]
    assert "sha256sum -c" in setup and "numba==0.68.0" in setup and "SETUP_OK" in setup
    assert "torch.cuda.is_available()" in setup and "_context(1)" in setup and "tqdm" in setup


class Clock:
    """Patches gpu_run.now / time.sleep: sleeping advances a fake clock."""

    def __init__(self, monkeypatch, t0=1_000_000.0):
        self.t = t0
        monkeypatch.setattr(gpu_run, "now", lambda: self.t)
        monkeypatch.setattr(gpu_run.time, "sleep", self.sleep)

    def sleep(self, s):
        self.t += s


class FakeRp:
    def __init__(self, pods=None, sticky=0, create=None):
        self.pods = list(pods or [])
        self.sticky = sticky  # deletes that silently fail before one sticks
        self.deleted = []
        self.create = create
        self.bal = 66.0

    def list_pods(self):
        return [dict(p) for p in self.pods]

    def delete_pod(self, pod_id, confirm):
        assert confirm
        if self.sticky:
            self.sticky -= 1
            return
        self.deleted.append(pod_id)
        self.pods = [p for p in self.pods if p["id"] != pod_id]

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm and max_hourly <= 1.0
        name = json.loads(open(body_path).read())["name"]
        self.pods.append({"id": "p1", "name": name})
        if self.create == "raise":
            raise RuntimeError("timeout after POST")
        return {"id": "p1", "costPerHr": 0.74}

    def balance(self):
        return self.bal


def test_delete_and_verify_retries_and_only_our_prefix(tmp_path, monkeypatch):
    Clock(monkeypatch)
    rp = FakeRp(
        pods=[{"id": "p1", "name": "m3b-gpu-1"}, {"id": "other", "name": "s2-phase-r"}], sticky=2
    )
    state = {"pod_id": "p1"}
    assert gpu_run.delete_and_verify(rp, tmp_path, state, "test")
    assert rp.deleted == ["p1"] and [p["id"] for p in rp.pods] == ["other"]


def test_delete_and_verify_waits_for_a_late_listing(tmp_path, monkeypatch):
    """An interrupted POST: the pod shows up in GET /pods only after 2 minutes."""
    clock = Clock(monkeypatch)
    rp = FakeRp()
    appear = clock.t + 120
    real_list = rp.list_pods

    def late_list():
        if clock.t >= appear and "p9" not in rp.deleted:
            rp.pods = [{"id": "p9", "name": "m3b-gpu-late"}]
        return real_list()

    rp.list_pods = late_list
    state = {"create_attempted_at": clock.t}
    assert gpu_run.delete_and_verify(rp, tmp_path, state, "interrupted create")
    assert rp.deleted == ["p9"] and clock.t - state["create_attempted_at"] >= 300


class FakeLedger:
    calls = []

    def __init__(self, policy):
        pass

    def register_run(self, *a):
        self.calls.append(("register",) + a[:2])

    def reserve_pod(self, run_id, key, rate, until, balance):
        self.calls.append(("reserve", key, rate))
        return None

    def bind_pod(self, run_id, key, pod_id, rate):
        self.calls.append(("bind", key, pod_id))

    def release_pod(self, run_id, key, cost):
        self.calls.append(("release", key))

    def finish_run(self, run_id, success, complete):
        self.calls.append(("finish", success))


def _patch_launch(monkeypatch, rp, drive_result):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod
    import research.runpod_fanout.tls as tls_mod

    FakeLedger.calls = []
    monkeypatch.setattr(ledger_mod, "SharedLedger", FakeLedger)
    monkeypatch.setattr(rp_mod, "RpClient", lambda: rp)
    monkeypatch.setattr(tls_mod, "preflight", lambda: [])
    Clock(monkeypatch)
    monkeypatch.setattr(gpu_run.signal, "signal", lambda *a: None)
    monkeypatch.setattr(gpu_run, "spawn_watchdog", lambda run_dir, until, name: 0)
    monkeypatch.setattr(gpu_run.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(
        gpu_run,
        "build_inputs",
        lambda work: {"commit": "c" * 40, "files": [], "demo_sha256": "d", "bytes": 0},
    )

    def fake_drive(agent, rp_, run_dir, state, inputs, args):
        if isinstance(drive_result, BaseException):
            raise drive_result
        return drive_result

    monkeypatch.setattr(gpu_run, "drive", fake_drive)


def _args(tmp_path):
    return SimpleNamespace(
        run_dir=str(tmp_path / "run"), lifetime_hours=7.0, spend_cap=8.0, rate=0.74
    )


def test_launch_deletes_on_success_and_on_interrupt(tmp_path, monkeypatch):
    for result, code in (("complete", 0), (KeyboardInterrupt("signal 15"), 1)):
        rp = FakeRp()
        _patch_launch(monkeypatch, rp, result)
        args = _args(tmp_path / str(code))
        assert gpu_run.launch(args) == code
        assert rp.pods == [] and rp.deleted == ["p1"]
        state = json.loads((tmp_path / str(code) / "run" / "state.json").read_text())
        assert state["pods_deleted_verified"] and state["pod_id"] == "p1"
        assert [c[0] for c in FakeLedger.calls] == [
            "register",
            "reserve",
            "bind",
            "release",
            "finish",
        ]


def test_launch_adopts_pod_created_despite_error(tmp_path, monkeypatch):
    rp = FakeRp(create="raise")
    _patch_launch(monkeypatch, rp, "complete")
    assert gpu_run.launch(_args(tmp_path)) == 0
    assert rp.deleted == ["p1"] and rp.pods == []


def test_download_confines_paths_and_checks_sha(tmp_path):
    good = b"weights"
    files = {
        "/ls/out": json.dumps(
            [
                {"path": "m3b_seed0/a.pth", "sha256": hashlib.sha256(good).hexdigest()},
                {"path": "../../escape", "sha256": "x"},
                {"path": "m3b_seed0/bad", "sha256": "0" * 64},
            ]
        ).encode(),
        "/out/m3b_seed0/a.pth": good,
        "/out/m3b_seed0/bad": b"corrupt",
    }

    class A:
        def get(self, path, timeout=30):
            return files.get(path, b"log")

    gpu_run.time.sleep, real = (lambda s: None), gpu_run.time.sleep
    try:
        out = gpu_run.download(A(), tmp_path)
    finally:
        gpu_run.time.sleep = real
    assert (tmp_path / "pod_logs/seed4.log").read_bytes() == b"log"
    assert out["files"] == 1 and set(out["failed"]) == {"../../escape", "m3b_seed0/bad"}
    assert (tmp_path / "out/m3b_seed0/a.pth").read_bytes() == good
    assert not (tmp_path.parent / "escape").exists()


def test_drive_refuses_unarmed_or_small_pod(tmp_path):
    class A:
        def __init__(self, h):
            self.h = h

        def health(self):
            return self.h

    rp = FakeRp()
    state = {"deadline": gpu_run.now() + 3600, "balance0": 66.0}
    args = SimpleNamespace(spend_cap=8.0)
    inputs = {"files": [], "bytes": 0, "demo_sha256": "d", "commit": "c"}
    big = {"cgroup_memory_gb": 100, "mem_total_gb": 200, "cpus_allowed": 16}
    h = dict(big, self_delete_armed=False, self_delete_probe=None)
    assert "not proven" in gpu_run.drive(A(h), rp, tmp_path, state, inputs, args)
    h = dict(big, self_delete_armed=True, self_delete_probe=401)
    assert "not proven" in gpu_run.drive(A(h), rp, tmp_path, state, inputs, args)
    ok = dict(self_delete_armed=True, self_delete_probe=200)
    for small in (
        dict(big, cgroup_memory_gb=48),
        dict(big, cgroup_memory_gb=None, mem_total_gb=None),
        dict(big, cgroup_cpus=6.0),
    ):
        assert "too small" in gpu_run.drive(A(dict(small, **ok)), rp, tmp_path, state, inputs, args)


def test_guard_stops_at_deadline_and_spend_cap():
    rp = FakeRp()
    args = SimpleNamespace(spend_cap=8.0)
    assert gpu_run.guard(rp, {"deadline": gpu_run.now() - 1, "balance0": 66.0}, args) == "lifetime"
    rp.bal = 58.0
    assert (
        gpu_run.guard(rp, {"deadline": gpu_run.now() + 99, "balance0": 66.0}, args) == "spend cap"
    )


def test_download_all_retries_then_reports_incomplete(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    calls = []

    def fake_download(agent, run_dir, budget_s=900):
        calls.append(budget_s)
        return {"files": 1, "failed": [] if len(calls) >= 3 else ["x"]}

    monkeypatch.setattr(gpu_run, "download", fake_download)
    assert gpu_run.download_all(None, tmp_path, {"until": clock.t + 3600})["failed"] == []
    calls.clear()
    out = gpu_run.download_all(None, tmp_path, {"until": clock.t + 200})
    assert out["failed"] == ["x"] and len(calls) == 1


def _agent(tmp_path, monkeypatch, **env):
    import hashlib as _h
    import importlib.util
    import threading
    from http.server import ThreadingHTTPServer

    token = "tok"
    monkeypatch.setenv("M3_ROOT", str(tmp_path / "r"))
    monkeypatch.setenv("M3_TOKEN_SHA256", _h.sha256(token.encode()).hexdigest())
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("m3_agent_test", gpu_run.AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return mod, srv, f"http://127.0.0.1:{srv.server_port}", token


def _call(base, path, token, method="GET", data=None, headers=None):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("X-M3-Token", token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_agent_auth_upload_confinement_and_deadline(tmp_path, monkeypatch):
    mod, srv, base, tok = _agent(tmp_path, monkeypatch, M3_DEADLINE_EPOCH=1)
    try:
        assert _call(base, "/health", "wrong")[0] == 403
        code, body = _call(base, "/health", tok)
        h = json.loads(body)
        assert code == 200 and h["self_delete_armed"] is False and h["self_delete_probe"] is None
        data = b"abc"
        good = {"X-Sha256": hashlib.sha256(data).hexdigest()}
        assert _call(base, "/in/x/a.bin", tok, "PUT", data, good)[0] == 200
        assert (tmp_path / "r/in/x/a.bin").read_bytes() == data
        assert _call(base, "/in/b.bin", tok, "PUT", data, {"X-Sha256": "0" * 64})[0] == 400
        assert not (tmp_path / "r/in/b.bin").exists()
        assert _call(base, "/in/../../evil", tok, "PUT", data, good)[0] in (400, 404)
        assert not (tmp_path / "evil").exists()
        assert _call(base, "/out/../in/x/a.bin", tok)[0] == 404
        req = json.dumps({"name": "j", "argv": ["true"]}).encode()
        assert _call(base, "/exec", tok, "POST", req)[0] == 410  # past its deadline
    finally:
        srv.shutdown()


def test_agent_exec_env_and_dead_man(tmp_path, monkeypatch):
    mod, srv, base, tok = _agent(
        tmp_path, monkeypatch, RUNPOD_API_KEY="secret", M3_DEADMAN_SECONDS=1
    )
    try:
        req = json.dumps(
            {"name": "j", "argv": ["sh", "-c", "env; sleep 30"], "cwd": str(tmp_path)}
        ).encode()
        assert _call(base, "/exec", tok, "POST", req)[0] == 200
        assert _call(base, "/exec", tok, "POST", req)[0] == 409  # already running
        deleted = []
        monkeypatch.setattr(mod, "self_delete", lambda: deleted.append(1))
        mod.LAST_AUTH[0] -= 10  # the runner went silent
        import threading
        import time as _t

        threading.Thread(target=mod.reaper, daemon=True).start()
        for _ in range(40):
            if deleted and mod.PROCS["j"].poll() is not None:
                break
            _t.sleep(0.25)
        assert deleted and mod.PROCS["j"].poll() is not None
        assert b"secret" not in (tmp_path / "r/logs/j.log").read_bytes()
    finally:
        srv.shutdown()


def test_watchdog_compiles_starts_and_arms(tmp_path, monkeypatch):
    import sys

    code = gpu_run.watchdog_code(tmp_path, gpu_run.now() + 7 * 3600, "m3b-gpu-x")
    compile(code, "<watchdog>", "exec")
    assert "\\n" in code and '"armed"' in code
    monkeypatch.setattr(sys, "platform", "linux")  # no caffeinate wrapper: the pid is python's
    pid = gpu_run.spawn_watchdog(tmp_path, gpu_run.now() + 7 * 3600, "m3b-gpu-x", wait_s=60)
    try:
        row = json.loads((tmp_path / "watchdog.log").read_text().splitlines()[0])
        assert row["watchdog"] == "armed" and row["pid"] == pid
        assert (tmp_path / "watchdog.err").read_text() == ""
        assert "== 'm3b-gpu-x'" in code  # exact name, never the prefix
    finally:
        gpu_run.stop_watchdog(None, tmp_path)  # the stop file alone, no signal
    import time as _t

    for _ in range(100):
        if '"stopped"' in (tmp_path / "watchdog.log").read_text():
            break
        _t.sleep(0.1)
    else:
        raise AssertionError("the watchdog did not stop on its stop file")


def test_spawn_watchdog_refuses_a_broken_script(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(gpu_run, "WATCHDOG", 'print("x\n")')
    with pytest.raises(SyntaxError):
        gpu_run.spawn_watchdog(tmp_path, gpu_run.now() + 3600, "m3b-gpu-x")


class RefusingRp(FakeRp):
    def __init__(self, refusals, definite=True):
        super().__init__()
        self.refusals = refusals
        self.definite = definite
        self.bodies = []

    def _call(self, *args):
        return {
            "dataCenters": [
                {
                    "id": "EU-CZ-1",
                    "gpuAvailability": [{"gpuTypeId": gpu_run.GPU, "stockStatus": "Low"}],
                },
                {
                    "id": "US-TX-3",
                    "gpuAvailability": [{"gpuTypeId": gpu_run.GPU, "stockStatus": None}],
                },
                {
                    "id": "EUR-IS-2",
                    "gpuAvailability": [{"gpuTypeId": gpu_run.GPU, "stockStatus": "Medium"}],
                },
            ]
        }

    def create_pod(self, body_path, max_hourly, confirm):
        from research.runpod_fanout.rp_client import RunPodError

        body = json.loads(open(body_path).read())
        self.bodies.append(body)
        if self.refusals:
            self.refusals -= 1
            payload = {"http_status": 500, "error": "no instances available"}
            raise RunPodError(
                "rp.py ('rest', 'POST', '/pods') exit 1: ", payload if self.definite else None
            )
        self.pods.append({"id": "p1", "name": body["name"]})
        return {"id": "p1", "costPerHr": 0.74}


def test_create_retries_definite_refusals_across_stocked_dcs(tmp_path, monkeypatch):
    Clock(monkeypatch)
    rp = RefusingRp(refusals=2)
    body = gpu_run.pod_body("m3b-gpu-t", "a" * 64, 1.0, 2.0)
    state = {}
    out, err = gpu_run.create_pod_retrying(rp, body, tmp_path, state)
    assert out["id"] == "p1" and err == ""
    assert [b.get("dataCenterIds") for b in rp.bodies] == [None, ["EUR-IS-2"], ["EU-CZ-1"]]
    assert "no instances available" in state["create_attempts"][0]["error"]


def test_create_never_reposts_after_an_unknown_outcome(tmp_path, monkeypatch):
    Clock(monkeypatch)
    rp = RefusingRp(refusals=5, definite=False)
    body = gpu_run.pod_body("m3b-gpu-t", "a" * 64, 1.0, 2.0)
    out, err = gpu_run.create_pod_retrying(rp, body, tmp_path, {})
    assert out is None and len(rp.bodies) == 1 and "exit 1" in err


def test_definite_refusal_only_for_4xx_or_runpod_capacity():
    from research.runpod_fanout.rp_client import RunPodError

    def e(status, body):
        return RunPodError("x", {"http_status": status, "error": body})

    assert gpu_run.definite_refusal(e(400, {"error": "bad"}))
    assert not gpu_run.definite_refusal(e(524, "<html>timeout</html>"))
    assert not gpu_run.definite_refusal(e(502, {"error": "bad gateway"}))
    msg = {"error": "There are no longer any instances available with the requested specs"}
    assert gpu_run.definite_refusal(e(500, msg))
    assert not gpu_run.definite_refusal(RunPodError("timed out"))
