"""GPU runner v2 (M3-C) and its pod agent: pure parts and simulated drives.

No RunPod call is ever made under pytest (the client, ledger and agent are fakes).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from research.redesign_m3c_20261008 import gpu_run_v2 as gr
from research.redesign_m3c_20261008 import probe_watch as pw


def _plan(n=4, **kw):
    plan = {
        "schema": gr.PLAN_SCHEMA,
        "transitions": 1000,
        "mac_rate": 10.0,
        "g2_factor": 4,
        "ram_per_run_gb": 13,
        "vcpu_per_run": 1.5,
        "max_concurrent_cap": 3,
        "speed_floor": 100.0,
        "min_ram_gb": 96,
        "min_vcpu": 16,
        "max_hourly": 0.95,
        "vram_per_run_gb": 2.0,
        "disk_per_run_gb": 12.0,
        "ring_gb_per_run": 9.3,
        "fill_beyond_guaranteed_vcpu": True,
        "runs": [{"name": f"B_s{i}", "seed": i, "extra": ["--reward", "mass"]} for i in range(n)],
    }
    plan.update(kw)
    return plan


def test_pod_body_is_money_safe():
    body = gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 96, 16)
    assert body["cloudType"] == "SECURE" and body["gpuCount"] == 1
    assert (
        body["env"]["M3_DEADLINE_EPOCH"] == "1000" and body["env"]["M3_SELF_DELETE_EPOCH"] == "2000"
    )
    assert "RUNPOD_API_KEY" not in body["env"] and body["volumeInGb"] == 0
    assert body["minRAMPerGPU"] == 96 and body["minVCPUPerGPU"] == 16
    assert gr.MAX_HOURLY <= 1.0 and gr.GPU_MAX_HOURLY <= 0.80 and gr.POD_NAME == "m3c-gpu-"


def test_plan_validation():
    assert gr.validate_plan(_plan()) == []
    assert "g2_factor < 4 (the per-step 4x rule)" in gr.validate_plan(_plan(g2_factor=3))
    bad = _plan()
    bad["runs"][0]["extra"] = ["--device", "cpu"]
    assert any("may not set --device" in p for p in gr.validate_plan(bad))
    dup = _plan()
    dup["runs"][1]["name"] = dup["runs"][0]["name"]
    assert gr.validate_plan(dup)
    assert gr.validate_plan(_plan(runs=[{"name": "../x", "seed": 0}]))
    assert any("max_hourly" in p for p in gr.validate_plan(_plan(max_hourly=1.5)))


def test_train_argv():
    plan = _plan(actors=2, ring_dir="/r/rings")
    argv = gr.train_argv(plan, plan["runs"][1])
    assert argv[argv.index("--ring-dir") + 1] == "/r/rings"
    assert argv[argv.index("--seed") + 1] == "1" and argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--out") + 1] == "/r/out/B_s1"
    assert argv[argv.index("--actors") + 1] == "2" and argv[-2:] == ["--reward", "mass"]
    assert "--actors" not in gr.train_argv(_plan(), _plan()["runs"][0])


def test_initial_concurrency_uses_guaranteed_vcpus_and_ram():
    plan = _plan(n=20, max_concurrent_cap=12, vcpu_per_run=1.0)
    # 256 visible CPUs but only 16 guaranteed: CPU bound 16; RAM (143 - 8) // 13 = 10
    assert gr.initial_concurrency(plan, 143.0, 256.0, 16.0) == 10
    assert gr.initial_concurrency(plan, 143.0, 256.0, 6.0) == 6
    assert gr.initial_concurrency(_plan(n=2), 400.0, 64.0, 64.0) == 2
    assert gr.initial_concurrency(plan, 15.0, 64.0, 64.0) == 0


def test_fill_decision():
    plan = _plan(n=10, max_concurrent_cap=8)
    tel = {
        "gpu": [{"util_pct": 40.0, "mem_total_mib": 24576, "mem_used_mib": 8192}],
        "cgroup_memory_gb": 143.0,
        "cgroup_memory_used_gb": 100.0,
        "mem_available_gb": 400.0,
        "host_cpu_pct": 20.0,
        "cpus": 32,
        "disk_free_gb": 50.0,
    }

    def fd(t=tel, running=5, queued=3, last=0.0, now=10_000.0, **kw):
        kw.setdefault("per_run_rate", 1000.0)
        return gr.fill_decision(
            plan if "p" not in kw else kw.pop("p"), t, running, queued, last, now, **kw
        )

    assert fd()
    assert fd(last=9_800.0) is None  # too soon
    assert fd(running=8) is None  # at the cap
    assert fd(queued=0) is None  # nothing queued
    assert fd(per_run_rate=None) is None  # no rate known yet
    bad = {
        "busy gpu": dict(tel, gpu=[dict(tel["gpu"][0], util_pct=90.0)]),
        # the cgroup is what counts, not the host's MemAvailable
        "cgroup full": dict(tel, cgroup_memory_used_gb=125.0),
        "vram full": dict(tel, gpu=[dict(tel["gpu"][0], mem_used_mib=23000)]),
        "disk full": dict(tel, disk_free_gb=8.0),
        "cpu busy": dict(tel, host_cpu_pct=95.0),
        "no gpu": dict(tel, gpu=[]),
    }
    for why, t in bad.items():
        assert fd(t=t) is None, why
    # the last fill lowered the per-run rate by > 15%, or did not add >= 3% combined
    assert fd(per_run_rate=800.0, per_run_rate_at_last_fill=1000.0) is None
    assert fd(per_run_rate=900.0, per_run_rate_at_last_fill=1000.0)
    assert fd(combined_rate=5100.0, combined_at_last_fill=5000.0) is None
    assert fd(combined_rate=5200.0, combined_at_last_fill=5000.0)
    # young runs rings are reserved: 43 GB headroom - 3 x 9.3 < 13 + 8
    assert fd(young_runs=3) is None and fd(young_runs=2)
    # a cgroup CPU quota caps concurrency and idle CPU comes from the jobs' measured use
    q = dict(tel, cgroup_cpus=12.0, jobs={"a": {"cpu_pct": 300.0}, "b": {"cpu_pct": 300.0}})
    assert fd(t=q, running=2)  # 3 x 1.5 <= 12, idle 6 >= 2.5
    assert fd(t=q, running=8) is None
    busyq = dict(q, jobs={"a": {"cpu_pct": 1100.0}})
    assert fd(t=busyq, running=2) is None  # idle 1 < 2.5
    # within the guaranteed vCPUs only, unless the plan opts in
    strict = dict(plan, fill_beyond_guaranteed_vcpu=False)  # 16 / 1.5 -> at most 10 runs
    assert fd(p=strict)
    assert fd(p=dict(strict, min_vcpu=8)) is None
    host_only = {k: v for k, v in tel.items() if not k.startswith("cgroup")}
    assert gr.mem_headroom_gb(host_only) is None and gr.mem_headroom_gb(tel) == 43.0
    assert gr.mem_headroom_gb(dict(tel, mem_total_gb=143.0, cgroup_memory_gb=9.2e9)) is None
    assert fd(t=host_only) is None


def test_g2_decision():
    ok = gr.g2_decision({"a": 30.0, "b": 30.0}, {"a": 600, "b": 600, "c": 1000}, 10.0, 4, 5000)
    assert ok["pass"] and ok["combined_rate"] == 60.0 and ok["projected_seconds_left"] == 2200 / 60
    slow = gr.g2_decision({"a": 15.0, "b": 15.0}, {"a": 1}, 10.0, 4, 5000)
    assert not slow["pass"] and slow["reason"] == "rate"
    late = gr.g2_decision({"a": 50.0}, {"a": 10_000_000}, 10.0, 4, 3600)
    assert not late["pass"] and late["reason"] == "lifetime"
    dead = gr.g2_decision({"a": 50.0}, {"a": 1}, 10.0, 4, 1e9, dead=["b"])
    assert not dead["pass"] and dead["reason"] == "rate"
    # wall clock: Mac needs 100_000 / 10 = 10_000 s; the pod must finish within 2_500 s
    # including the time already spent (upload, setup, the first 30 min)
    fast = gr.g2_decision({"a": 50.0}, {"a": 50_000}, 10.0, 4, 1e9, elapsed=1000, total=100_000)
    assert fast["pass"] and fast["mac_wall"] == 10_000
    over = gr.g2_decision({"a": 50.0}, {"a": 80_000}, 10.0, 4, 1e9, elapsed=1000, total=100_000)
    assert not over["pass"] and over["reason"] == "wall"


def test_run_rate_and_partial_lines():
    rows = gr.parse_jsonl(
        b'{"transitions": 100, "wall": 10}\n{"transitions": 300, "wall": 20}\n{"transi'
    )
    assert len(rows) == 2
    r = gr.run_rate(rows)
    assert r["rate"] == 15.0 and r["recent_rate"] == 20.0 and r["transitions"] == 300
    assert gr.run_rate([]) is None


class Clock:
    def __init__(self, monkeypatch, t0=1_000_000.0):
        self.t = t0
        monkeypatch.setattr(gr, "now", lambda: self.t)
        monkeypatch.setattr(gr.time, "sleep", self.sleep)

    def sleep(self, s):
        self.t += s


def test_puller_confines_checks_sha_and_is_incremental(tmp_path, monkeypatch):
    Clock(monkeypatch)
    good = b"weights"
    files = {
        "/out/B_s0/manifest.jsonl": (
            json.dumps(
                {
                    "file": "ckpt_500k.pth",
                    "sha256": hashlib.sha256(good).hexdigest(),
                    "transitions": 500,
                }
            )
            + "\n"
            + json.dumps({"file": "../evil.pth", "sha256": "x"})
            + "\n"
            + json.dumps({"file": "final.pth", "sha256": "0" * 64})
            + "\n"
        ).encode(),
        "/out/B_s0/ckpt_500k.pth": good,
        "/out/B_s0/final.pth": b"corrupt",
    }
    gets = []

    class A:
        def get(self, path, timeout=30):
            gets.append(path)
            if path not in files:
                raise OSError(404)
            return files[path]

    state = {}
    p = gr.Puller(A(), tmp_path, state)
    new = p.checkpoints("B_s0")
    assert [r["file"] for r in new] == ["ckpt_500k.pth"]
    assert (tmp_path / "out/B_s0/ckpt_500k.pth").read_bytes() == good
    assert not (tmp_path / "out/B_s0/final.pth").exists()
    landed = [json.loads(x) for x in (tmp_path / "landed.jsonl").read_text().splitlines()]
    assert landed[0]["run"] == "B_s0" and landed[0]["transitions"] == 500
    n = len(gets)
    assert p.checkpoints("B_s0") == []  # already pulled: only the manifest and the bad row
    assert "/out/B_s0/ckpt_500k.pth" not in gets[n:]
    with pytest.raises(ValueError):
        p.checkpoints("../x")


def test_delete_and_verify_only_our_prefix(tmp_path, monkeypatch):
    Clock(monkeypatch)

    class Rp:
        def __init__(self):
            self.pods = [{"id": "p1", "name": "m3c-gpu-1"}, {"id": "o", "name": "m3b-gpu-1"}]
            self.deleted = []

        def list_pods(self):
            return list(self.pods)

        def delete_pod(self, pid, confirm):
            assert confirm
            self.deleted.append(pid)
            self.pods = [p for p in self.pods if p["id"] != pid]

    rp = Rp()
    assert gr.delete_and_verify(rp, tmp_path, {"pod_id": "p1"}, "t")
    assert rp.deleted == ["p1"] and [p["id"] for p in rp.pods] == ["o"]


def test_watchdog_code_compiles_with_exact_name(tmp_path):
    code = gr.watchdog_code(tmp_path, 1e9, "m3c-gpu-x")
    compile(code, "<watchdog>", "exec")
    assert "== 'm3c-gpu-x'" in code
    with pytest.raises(AssertionError):
        gr.watchdog_code(tmp_path, 1e9, "m3b-gpu-x")


# ----------------------------------------------------------------------------- simulated drive
class FakeAgent:
    """A pod whose runs each finish after ``polls`` health polls, logging and checkpointing."""

    def __init__(
        self,
        polls=40,
        rate=50.0,
        fail=(),
        probe_rate=500.0,
        gpu_util=30.0,
        mem=30.0,
        hung=(),
        slow_after=None,
    ):
        self.slow_after = slow_after
        self.mem = mem
        self.hung = set(hung)
        self.jobs = {}
        self.polls = polls
        self.rate = rate
        self.fail = set(fail)
        self.probe_rate = probe_rate
        self.gpu_util = gpu_util
        self.max_running = 0
        self.uploads = []

    def health(self):
        procs = {}
        running = 0
        for name, j in self.jobs.items():
            j["n"] += 1
            done = j["n"] >= (2 if name in ("setup", "probe") else self.polls)
            rc = (1 if name in self.fail else 0) if done else None
            running += rc is None and name not in ("setup", "probe")
            procs[name] = {"pid": 1, "returncode": rc}
        self.max_running = max(self.max_running, running)
        return {
            "self_delete_armed": True,
            "self_delete_probe": 403,
            "cgroup_memory_gb": 143.0,
            "mem_total_gb": 143.0,
            "cpus": 256,
            "cpus_allowed": 256,
            "uptime": 100.0 + sum(j["n"] for j in self.jobs.values()),
            "procs": procs,
        }

    def telemetry(self):
        return {
            "gpu": [{"util_pct": self.gpu_util, "mem_total_mib": 49140, "mem_used_mib": 4000}],
            "mem_available_gb": self.mem,
            "mem_total_gb": 143.0,
            "cgroup_memory_gb": 143.0,
            "cgroup_memory_used_gb": 143.0 - self.mem,
            "host_cpu_pct": 10.0,
            "cpus": 256,
            "disk_free_gb": 500.0,
            "jobs": {},
        }

    def put(self, dst, src, sha):
        self.uploads.append(dst)

    def post_kill(self, name):
        self.killed = getattr(self, "killed", []) + [name]
        self.jobs[name]["n"] = 10**9  # its process group exits

    def exec(self, name, argv, cwd, env):
        assert name == "setup" or env["OMP_NUM_THREADS"] == "1"
        self.jobs[name] = {"n": 0, "argv": argv}

    def get(self, path, timeout=30):
        if path == "/log/setup":
            return b"... SETUP_OK\n"
        if path == "/out/_probe/actor.jsonl":
            return json.dumps({"rate": self.probe_rate}).encode()
        if path == "/ls/out":
            return b"[]"
        parts = path.split("/")
        run = parts[2] if len(parts) > 2 else ""
        j = self.jobs.get(run)
        if j is None:
            raise OSError(404)
        t = min(j["n"], self.polls) * 60
        if run in self.hung:
            t = min(t, 120)
        if parts[-1] == "log.jsonl":
            if t == 0:
                raise OSError(404)
            work = self.rate * t
            if self.slow_after is not None and t > self.slow_after * 60:
                work = self.rate * (self.slow_after * 60 + (t - self.slow_after * 60) / 10)
            rows = [{"transitions": work * k / 6, "wall": t * k / 6} for k in range(1, 7)]
            return "\n".join(json.dumps(r) for r in rows).encode()
        if parts[-1] == "manifest.jsonl":
            if j["n"] < self.polls:
                return b""
            return json.dumps(
                {
                    "file": "final.pth",
                    "sha256": hashlib.sha256(run.encode()).hexdigest(),
                    "transitions": 1,
                }
            ).encode()
        if parts[-1] == "final.pth":
            return run.encode()
        raise OSError(404)


def _drive(tmp_path, monkeypatch, agent, plan, waiver="user decision 2026-10-08 in chat"):
    clock = Clock(monkeypatch)
    tmp_path.mkdir(parents=True, exist_ok=True)

    class Rp:
        def balance(self):
            return 60.0

    state = {
        "deadline": clock.t + 8 * 3600,
        "until": clock.t + 8 * 3600 + 900,
        "balance0": 60.0,
        "created": clock.t - 1800,
    }
    args = SimpleNamespace(spend_cap=10.0, accept_no_self_delete=waiver)
    inputs = {
        "files": [{"dst": "repo.tar.gz", "src": "x", "sha256": "s"}],
        "bytes": 1,
        "demo_sha256": "d",
        "commit": "c",
    }
    out = gr.drive(agent, Rp(), tmp_path, state, inputs, args, plan)
    return out, state


def test_drive_runs_the_queue_with_bounded_concurrency_and_pulls(tmp_path, monkeypatch):
    plan = _plan(n=5, max_concurrent_cap=2, mac_rate=10.0, transitions=2400 * 50)
    agent = FakeAgent(polls=40, rate=50.0, gpu_util=95.0)
    out, state = _drive(tmp_path, monkeypatch, agent, plan)
    assert out == "complete", out
    assert agent.max_running == 2 and set(state["runs"]) == {f"B_s{i}" for i in range(5)}
    assert state["g2"]["pass"] and state["speed_probe"]["rate"] == 500.0
    # G2 measured runner-side over the >= 10 min freeze window, not since-start averages
    assert 600 <= state["g2"]["window_seconds"] <= 900
    assert abs(state["g2"]["combined_rate"] - 2 * 50.0) < 15
    landed = [json.loads(x) for x in (tmp_path / "landed.jsonl").read_text().splitlines()]
    assert sorted(r["run"] for r in landed) == [f"B_s{i}" for i in range(5)]
    tel = [json.loads(x) for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert tel and all("gpu" in r and "rates" in r for r in tel)
    stages = [
        json.loads(x)["stage"] for x in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert stages.index("speed_probe") < stages.index("sizing") < stages.index("run_started")


def test_drive_adaptive_fill_when_gpu_is_idle(tmp_path, monkeypatch):
    plan = _plan(n=6, max_concurrent_cap=4, ram_per_run_gb=60, transitions=4000 * 50)
    agent = FakeAgent(polls=80, rate=50.0, gpu_util=20.0, mem=200.0)
    out, state = _drive(tmp_path, monkeypatch, agent, plan)
    assert out == "complete"
    # RAM sizing allows 2 at first ((143 - 8) // 60); the idle GPU + free RAM adds more later
    assert agent.max_running >= 3
    stages = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines()]
    fills = [r for r in stages if "adaptive fill" in (r.get("reason") or "")]
    assert fills
    g2_t = [r["t"] for r in stages if r["stage"] == "G2"][0]
    first = [r["t"] for r in stages if r["stage"] == "run_started"][0]
    # no fill inside the freeze window [G2_SECONDS - 600, G2 decision]
    assert not [f for f in fills if first + gr.G2_SECONDS - 600 <= f["t"] <= g2_t]


def test_drive_refuses_slow_host_unwaived_pod_and_slow_g2(tmp_path, monkeypatch):
    out, _ = _drive(tmp_path / "a", monkeypatch, FakeAgent(probe_rate=50.0), _plan())
    assert out.startswith("slow host")
    out, _ = _drive(tmp_path / "b", monkeypatch, FakeAgent(), _plan(), waiver=None)
    assert "not proven" in out
    slow = _plan(n=3, mac_rate=100.0, transitions=10**9)
    out, state = _drive(tmp_path / "c", monkeypatch, FakeAgent(polls=10_000), slow)
    assert out == "G2 fail (rate)" and state["g2"]["combined_rate"] < 400.0


def test_drive_reports_failed_runs(tmp_path, monkeypatch):
    agent = FakeAgent(polls=40, fail={"B_s1"})
    out, state = _drive(tmp_path, monkeypatch, agent, _plan(n=3, transitions=2400 * 50))
    assert out == "finished with failures" and state["runs"]["B_s1"]["returncode"] == 1


def test_launch_deletes_on_success_and_interrupt(tmp_path, monkeypatch):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod
    import research.runpod_fanout.tls as tls_mod

    calls = []

    class Ledger:
        def __init__(self, policy):
            pass

        def register_run(self, *a):
            calls.append("register")

        def reserve_pod(self, *a):
            calls.append("reserve")

        def bind_pod(self, *a):
            calls.append("bind")

        def release_pod(self, *a):
            calls.append("release")

        def finish_run(self, *a):
            calls.append("finish")

    class Rp:
        def __init__(self):
            self.pods, self.deleted = [], []

        def list_pods(self):
            return list(self.pods)

        def delete_pod(self, pid, confirm):
            self.deleted.append(pid)
            self.pods = [p for p in self.pods if p["id"] != pid]

        def create_pod(self, body_path, max_hourly, confirm):
            assert confirm and max_hourly == 0.95
            self.pods.append({"id": "p1", "name": json.loads(open(body_path).read())["name"]})
            return {"id": "p1", "costPerHr": 0.74}

        def balance(self):
            return 66.0

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()))
    for result, code in (("complete", 0), (KeyboardInterrupt("signal 15"), 1)):
        rp = Rp()
        calls.clear()
        monkeypatch.setattr(ledger_mod, "SharedLedger", Ledger)
        monkeypatch.setattr(rp_mod, "RpClient", lambda: rp)
        monkeypatch.setattr(tls_mod, "preflight", lambda: [])
        Clock(monkeypatch)
        monkeypatch.setattr(gr.signal, "signal", lambda *a: None)
        monkeypatch.setattr(gr, "spawn_watchdog", lambda run_dir, until, name: 0)
        monkeypatch.setattr(gr.subprocess, "Popen", lambda *a, **k: None)
        monkeypatch.setattr(
            gr,
            "build_inputs",
            lambda work, plan: {
                "commit": "c",
                "files": [],
                "demo_sha256": "d",
                "bytes": 0,
                "plan_sha256": plan["_sha256"],
            },
        )

        def fake_drive(*a, _r=result):
            if isinstance(_r, BaseException):
                raise _r
            return _r

        monkeypatch.setattr(gr, "drive", fake_drive)
        args = SimpleNamespace(
            run_dir=str(tmp_path / f"run{code}"),
            plan=str(plan_path),
            lifetime_hours=8.0,
            spend_cap=10.0,
            rate=0.89,
        )
        assert gr.launch(args) == code
        assert rp.pods == [] and rp.deleted == ["p1"]
        state = json.loads((tmp_path / f"run{code}/attempt00/state.json").read_text())
        assert state["pods_deleted_verified"] and state["plan_sha256"]
        assert calls == ["register", "reserve", "bind", "release", "finish"]


# ----------------------------------------------------------------------------- pod agent v2
def _agent(tmp_path, monkeypatch, **env):
    import importlib.util
    import threading
    from http.server import ThreadingHTTPServer

    token = "tok"
    monkeypatch.setenv("M3_ROOT", str(tmp_path / "r"))
    monkeypatch.setenv("M3_TOKEN_SHA256", hashlib.sha256(token.encode()).hexdigest())
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("m3c_agent_test", gr.AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return mod, srv, gr.Agent("x", token, base=f"http://127.0.0.1:{srv.server_port}")


def test_agent_v2_telemetry_ls_cache_and_auth(tmp_path, monkeypatch):
    mod, srv, agent = _agent(tmp_path, monkeypatch)
    try:
        bad = gr.Agent("x", "wrong", base=agent.base)
        with pytest.raises(Exception):
            bad.health()
        h = agent.health()
        assert h["agent"] == "m3c-gpu-agent/v2" and "cpu_model" in h
        tel = agent.telemetry()
        assert {"gpu", "host_cpu_pct", "mem_available_gb", "jobs", "loadavg"} <= set(tel)
        out = tmp_path / "r/out/B_s0"
        out.mkdir(parents=True)
        (out / "final.pth").write_bytes(b"abc")
        (out / "final.pth.tmp").write_bytes(b"partial")
        rows = json.loads(agent.get("/ls/out"))
        assert [r["path"] for r in rows] == ["B_s0/final.pth"]
        assert rows[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
        key = str(out / "final.pth")
        size, mtime, _ = mod.HASHES[key]
        mod.HASHES[key] = (size, mtime, "cached")
        assert json.loads(agent.get("/ls/out"))[0]["sha256"] == "cached"  # (size, mtime) hit
        agent.exec("j", ["sh", "-c", "sleep 2"], str(tmp_path), {})
        jobs = agent.telemetry()["jobs"]
        assert "j" in jobs
    finally:
        srv.shutdown()


def test_agent_v2_keeps_deadline_and_deadman(tmp_path, monkeypatch):
    import threading
    import time as _t

    mod, srv, agent = _agent(tmp_path, monkeypatch, M3_DEADMAN_SECONDS=1, RUNPOD_API_KEY="k")
    try:
        agent.exec("j", ["sh", "-c", "env; sleep 30"], str(tmp_path), {})
        deleted = []
        monkeypatch.setattr(mod, "self_delete", lambda: deleted.append(1))
        mod.LAST_AUTH[0] -= 10
        threading.Thread(target=mod.reaper, daemon=True).start()
        for _ in range(40):
            if deleted and mod.PROCS["j"].poll() is not None:
                break
            _t.sleep(0.25)
        assert deleted and mod.PROCS["j"].poll() is not None
        assert b"=k\n" not in (tmp_path / "r/logs/j.log").read_bytes()
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------------- probe watcher
def test_probe_watch_jobs_order_and_idempotence(tmp_path):
    rows = [
        {"run": "B_s0", "file": "ckpt_5000k.pth", "path": "/x/a.pth"},
        {"run": "B_s0", "file": "ckpt_2500k.pth", "path": "/x/b.pth"},
        {"run": "B_s0", "file": "final.pth", "path": "/x/c.pth"},
        {"run": "../evil", "file": "final.pth", "path": "/x/d.pth"},
    ]
    jobs = pw.pending_jobs(rows, tmp_path, ["5000k", "final"], ["none", "v8"], set())
    assert len(jobs) == 2 * 2 * 3
    assert [j[3] for j in jobs[:6]] == ["none"] * 6  # no-veto first
    assert {j[1] for j in jobs} == {"ckpt_5000k", "final"}
    (tmp_path / "B_s0__final__none.jsonl").write_text(json.dumps({"mix": "frozen"}) + "\n")
    jobs2 = pw.pending_jobs(rows, tmp_path, ["5000k", "final"], ["none", "v8"], set())
    assert len(jobs2) == 11 and ("B_s0", "final", "/x/c.pth", "none", "frozen") not in jobs2
    argv = pw.probe_argv(jobs2[0], tmp_path, 16, 52, slots=False)
    assert argv[:3] == ["nice", "-n", "10"] and "--round-index" in argv
    assert pw.probe_argv(jobs2[0], tmp_path, 16, 52, slots=True)[1].endswith("slotrun.py")


def test_probe_watch_reads_landed_and_manifests(tmp_path):
    landed = tmp_path / "landed.jsonl"
    landed.write_text(json.dumps({"run": "A_s1", "file": "final.pth", "path": "/p"}) + "\nbad\n")
    d = tmp_path / "runs" / "B_s2"
    d.mkdir(parents=True)
    (d / "manifest.jsonl").write_text(json.dumps({"file": "ckpt_5000k.pth"}) + "\n")
    rows = pw.landed_rows(landed, [tmp_path / "runs"])
    assert {(r["run"], r["file"]) for r in rows} == {
        ("A_s1", "final.pth"),
        ("B_s2", "ckpt_5000k.pth"),
    }


def test_draft_plan_validates_and_matches_the_preregistration():
    from pathlib import Path

    path = Path(gr.__file__).parent / "plans/m3c_pod_plan_draft.json"
    plan = gr.load_plan(path)
    assert plan["transitions"] == 20_000_000 and plan["actors"] == 2
    assert plan["g2_factor"] == 4 and plan["mac_rate"] >= 2900 and plan["max_hourly"] <= 1.0
    arms = sorted({r["name"].split("_")[0] for r in plan["runs"]})
    assert arms == ["A", "B", "D", "L"] and len(plan["runs"]) == 20
    for r in plan["runs"]:
        x = " ".join(r["extra"])
        short = "--anchor-hold 2000000 --anchor-floor-at 6000000" in x
        assert short == (not r["name"].startswith("L"))
        assert ("--reward mass" in x) == (not r["name"].startswith("A"))
        assert ("--mixed-frac" in x) == r["name"].startswith("D")
    assert plan["ram_per_run_gb"] + gr.RAM_RESERVE_GB <= plan["min_ram_gb"]


def test_drive_kills_a_stalled_run_and_reports_it(tmp_path, monkeypatch):
    agent = FakeAgent(polls=40, hung={"B_s1"})
    out, state = _drive(tmp_path, monkeypatch, agent, _plan(n=3, transitions=2400 * 50))
    assert agent.killed == ["B_s1"] and state["runs"]["B_s1"]["killed"].startswith("stalled")
    assert out in ("finished with failures", "complete")


def test_queue_is_seed_major_and_truncated_by_the_lifetime(tmp_path, monkeypatch):
    plan = _plan(n=2, max_concurrent_cap=1, transitions=2400 * 50, mac_rate=1.0)
    plan["runs"] = [
        {"name": f"{a}_s{s}", "seed": s, "extra": []} for a in ("A", "B") for s in range(2)
    ]
    agent = FakeAgent(polls=40)
    out, state = _drive(tmp_path, monkeypatch, agent, plan)
    starts = [
        json.loads(x)["run"]
        for x in (tmp_path / "progress.jsonl").read_text().splitlines()
        if json.loads(x)["stage"] == "run_started"
    ]
    assert starts[:2] == ["A_s0", "B_s0"]  # seed-major, not arm-major
    assert out == "complete"
    # fast enough for G2 at 30 min, then 10x slower: the re-projection stops new starts
    agent2 = FakeAgent(polls=200, rate=50.0, slow_after=35)
    out2, state2 = _drive(tmp_path / "t", monkeypatch, agent2, plan)
    assert out2 == "finished partially (queue truncated by the lifetime)", out2
    assert 1 <= len(state2["runs"]) < 4


def test_probe_watch_reads_every_attempt_of_a_run_root(tmp_path):
    for k, run in ((0, "A_s0"), (1, "B_s0")):
        d = tmp_path / f"attempt{k:02d}"
        d.mkdir()
        (d / "landed.jsonl").write_text(json.dumps({"run": run, "file": "final.pth", "path": "/p"}))
    rows = pw.landed_rows(tmp_path, [])
    assert sorted(r["run"] for r in rows) == ["A_s0", "B_s0"]
