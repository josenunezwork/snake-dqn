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


def test_g2_decision():
    ok = gpu_run.g2_decision({s: 1500.0 for s in gpu_run.SEEDS}, 1800, 6.5 * 3600)
    assert ok["pass"] and ok["combined_rate"] == 7500.0
    slow = gpu_run.g2_decision({s: 1000.0 for s in gpu_run.SEEDS}, 1800, 6.5 * 3600)
    assert not slow["pass"] and slow["reason"] == "rate"
    late = gpu_run.g2_decision({s: 1200.0 for s in gpu_run.SEEDS}, 1800, 3 * 3600)
    assert not late["pass"] and late["reason"] == "lifetime"
    missing = gpu_run.g2_decision({0: 9000.0}, 1800, 6 * 3600)
    assert not missing["pass"]


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
    monkeypatch.setattr(gpu_run.time, "sleep", lambda s: None)
    rp = FakeRp(
        pods=[{"id": "p1", "name": "m3b-gpu-1"}, {"id": "other", "name": "s2-phase-r"}], sticky=2
    )
    state = {"pod_id": "p1"}
    assert gpu_run.delete_and_verify(rp, tmp_path, state, "test")
    assert rp.deleted == ["p1"] and [p["id"] for p in rp.pods] == ["other"]


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
    monkeypatch.setattr(gpu_run.time, "sleep", lambda s: None)
    monkeypatch.setattr(gpu_run, "spawn_watchdog", lambda run_dir, until: 0)
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
        def get(self, path):
            return files[path]

    gpu_run.time.sleep, real = (lambda s: None), gpu_run.time.sleep
    try:
        out = gpu_run.download(A(), tmp_path)
    finally:
        gpu_run.time.sleep = real
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
    h = {"self_delete_armed": False, "cgroup_memory_gb": 100, "cpus_allowed": 16}
    assert "not armed" in gpu_run.drive(A(h), rp, tmp_path, state, inputs, args)
    h = {"self_delete_armed": True, "cgroup_memory_gb": 48, "cpus_allowed": 16}
    assert "too small" in gpu_run.drive(A(h), rp, tmp_path, state, inputs, args)


def test_guard_stops_at_deadline_and_spend_cap():
    rp = FakeRp()
    args = SimpleNamespace(spend_cap=8.0)
    assert gpu_run.guard(rp, {"deadline": gpu_run.now() - 1, "balance0": 66.0}, args) == "lifetime"
    rp.bal = 58.0
    assert (
        gpu_run.guard(rp, {"deadline": gpu_run.now() + 99, "balance0": 66.0}, args) == "spend cap"
    )
