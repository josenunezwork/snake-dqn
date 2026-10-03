"""Unit/integration tests for research/runpod_fanout (no real pods, no network).

Every test runs with the RunPod client and all non-loopback sockets made fatal; the
"pods" are local pod_agent.py processes on 127.0.0.1 started by a fake rp client.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.runpod_fanout import episode as episode_mod
from research.runpod_fanout import jobspec, runner, watchdog
from research.runpod_fanout.rp_client import RpClient, RunPodError
from research.runpod_fanout.wrappers import HERO_SHA256, SCRIPTED_SHA256

OPP = "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195"
GREEDY, RANDOM = sorted(SCRIPTED_SHA256, key=lambda s: SCRIPTED_SHA256[s])
STUB = r"""
import argparse, hashlib, json, os, sys, time
p = argparse.ArgumentParser()
for k in ("--spec", "--out-root", "--ckpt-dir", "--provider", "--job-id", "--commit"):
    p.add_argument(k)
a = p.parse_args()
spec = json.load(open(a.spec))
group = f"{spec['wrapper']}__h{spec['horizon']}__{spec['engine']}"
key = f"{group}/{spec['arm']}-{spec['mix']}-{spec['world_seed']}.json"
flag = os.path.join(os.environ["STUB_STATE"], key.replace("/", "_"))
if os.environ.get("STUB_FAIL_ONCE") == key and not os.path.exists(flag):
    open(flag, "w").close()
    sys.exit(7)
time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
mass = 1.0 + spec["world_seed"] % 7
if os.environ.get("STUB_DIVERGE") == key and os.environ.get("FANOUT_PROVIDER_TAG") == "b":
    mass += 1
entry = {k: spec[k] for k in ("arm", "mix", "world_seed", "world_index", "roster_member_sha256s")}
entry.update({"wall_seconds": time.time(), "record": {"mass_integral": mass,
              "probes": {"x": {"apply_seconds_total": time.time()}}},
              "platform": {"platform_id": "stub"},
              "fanout": {"job_id": a.job_id, "episode_key": key, "repo_commit": a.commit,
                         "spec_sha256": hashlib.sha256(json.dumps(
                             spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}})
dest = os.path.join(a.out_root, key)
os.makedirs(os.path.dirname(dest), exist_ok=True)
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(entry, sort_keys=True, indent=2) + "\n")
os.replace(tmp, dest)
print(json.dumps({"key": key, "status": "written"}))
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Real RunPod calls and non-loopback connections are fatal in this module."""

    def fatal_call(self, *args):
        raise AssertionError(f"real rp.py call attempted: {args[:3]}")

    monkeypatch.setattr(RpClient, "_call", fatal_call)
    real_connect = socket.socket.connect

    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network call to {address} attempted")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def policy(**over):
    p = jobspec.load_policy()
    p = dict(p, **over)
    return p


def ep(arm="A", mix="scripted", seed=11, index=0, roster=None, horizon=1000, engine="live"):
    return {
        "wrapper": "apex-veto-v8-screen-v1",
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
        "world_index": index,
        "horizon": horizon,
        "engine": engine,
        "roster_member_sha256s": roster or [HERO_SHA256, RANDOM, GREEDY, RANDOM, GREEDY],
    }


def job(episodes=None, **over):
    j = {
        "schema": jobspec.JOB_SCHEMA,
        "job_id": "unit-job",
        "tier": "tier1",
        "purpose": "unit test",
        "repo_commit": "0" * 40,
        "max_wall_minutes": 30,
        "checkpoints": [HERO_SHA256],
        "episodes": episodes or [ep("A"), ep("B")],
    }
    j.update(over)
    return j


# ---------------------------------------------------------------- job validation


def test_valid_job_and_keys():
    j = jobspec.validate_job(job(), policy(), jobspec.load_allowlist())
    keys = [jobspec.episode_key(e) for e in j["episodes"]]
    assert keys == [
        "apex-veto-v8-screen-v1__h1000__live/A-scripted-11.json",
        "apex-veto-v8-screen-v1__h1000__live/B-scripted-11.json",
    ]


@pytest.mark.parametrize(
    "over",
    [
        {"tier": "tier2"},
        {"tier": "strict"},
        {"purpose": "strict gate run"},
        {"purpose": "serving qualification"},
        {"job_id": "v8-strict-a"},
        {"job_id": "ab--cd"},
        {"job_id": "abc-"},
    ],
)
def test_strict_and_serving_refused(over):
    with pytest.raises(jobspec.JobError):
        jobspec.validate_job(job(**over), policy(), jobspec.load_allowlist())


def test_unregistered_wrapper_refused():
    e = ep()
    e["wrapper"] = "apex-veto-v8-strict-v1"
    with pytest.raises(jobspec.JobError, match="not a registered"):
        jobspec.validate_job(job([e]), policy(), jobspec.load_allowlist())


def test_checkpoint_allowlist_enforced():
    allow = {HERO_SHA256: jobspec.load_allowlist()[HERO_SHA256]}  # pre-approval allow-list
    frozen = ep(mix="frozen", roster=[HERO_SHA256, OPP, HERO_SHA256, OPP, HERO_SHA256])
    with pytest.raises(jobspec.JobError, match="allow-list"):
        jobspec.validate_job(job([frozen], checkpoints=[HERO_SHA256, OPP]), policy(), allow)
    # roster member not declared in checkpoints
    with pytest.raises(jobspec.JobError, match="not in the job's checkpoints"):
        jobspec.validate_job(job([frozen]), policy(), jobspec.load_allowlist())
    # with the current allow-list and declared checkpoints it passes
    jobspec.validate_job(
        job([frozen], checkpoints=[HERO_SHA256, OPP]), policy(), jobspec.load_allowlist()
    )


def test_unused_checkpoint_and_duplicates_refused():
    with pytest.raises(jobspec.JobError, match="not used"):
        jobspec.validate_job(
            job(checkpoints=[HERO_SHA256, OPP]), policy(), jobspec.load_allowlist()
        )
    with pytest.raises(jobspec.JobError, match="duplicate"):
        jobspec.validate_job(job([ep(), ep()]), policy(), jobspec.load_allowlist())
    with pytest.raises(jobspec.JobError, match="SIMD"):
        e = ep(engine="simd")
        e["wrapper"] = "apex-veto-v8-dev-v1"
        jobspec.validate_job(job([e]), policy(), jobspec.load_allowlist())


def test_allowlist_is_exactly_the_approved_four():
    allow = jobspec.load_allowlist()
    assert sorted(allow) == sorted(
        [
            HERO_SHA256,
            OPP,
            "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
            "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
        ]
    )
    from research.apex_safety_20260926 import dev_screen

    assert {sha for _, sha in dev_screen.POOL} == set(allow)


def test_scripted_and_hero_pins():
    from research.apex_safety_20260926 import dev_screen
    from src.evaluation.strict_promotion import scripted_agent

    for sha, name in SCRIPTED_SHA256.items():
        assert scripted_agent(name)["sha256"] == sha
    assert dev_screen.CHAMPION[1] == HERO_SHA256


def test_policy_is_cpu3c_secure():
    p = jobspec.load_policy()
    assert p["flavor"] == "cpu3c" and p["cloud"] == "SECURE"
    assert p["vcpu_sizes_desc"] == [32, 16, 8, 4, 2]
    assert p["global_floor_usd"] == 5.0 and p["project_cap_usd"] == 50.0


def test_deterministic_bytes_drop_timing_and_stamps():
    a = {
        "x": 1,
        "wall_seconds": 3.0,
        "platform": {"p": 1},
        "fanout": {"f": 1},
        "veto_diagnostics": {"apply_seconds_total": 9.0, "n": 2},
    }
    b = {
        "x": 1,
        "wall_seconds": 4.0,
        "platform": {"p": 2},
        "fanout": {"f": 2},
        "veto_diagnostics": {"apply_seconds_total": 1.0, "n": 2},
    }
    assert jobspec.deterministic_bytes(a) == jobspec.deterministic_bytes(b)
    b["veto_diagnostics"]["n"] = 3
    assert jobspec.deterministic_bytes(a) != jobspec.deterministic_bytes(b)


def test_episode_write_record_create_only(tmp_path):
    entry = {"record": {"m": 1}, "wall_seconds": 1.0}
    key = "g/A-scripted-1.json"
    assert episode_mod.write_record(tmp_path, key, entry) == "written"
    raw = (tmp_path / key).read_bytes()
    assert raw == (json.dumps(entry, sort_keys=True, indent=2) + "\n").encode()
    assert episode_mod.write_record(tmp_path, key, dict(entry, wall_seconds=2.0)) == (
        "duplicate-verified"
    )
    assert (tmp_path / key).read_bytes() == raw
    assert episode_mod.write_record(tmp_path, key, {"record": {"m": 2}}) == "DIVERGENT"
    assert not list(tmp_path.rglob("*.tmp"))


def test_agent_start_command_roundtrip():
    cmd = runner.agent_start_command()
    blob = cmd[2].split("echo ", 1)[1].split(" |", 1)[0]
    assert gzip.decompress(base64.b64decode(blob)) == runner.AGENT_SOURCE.read_bytes()


def test_pod_body_has_no_secret_and_is_cpu3c():
    body = runner.pod_body(policy(), "rpf-unit-job--r-1", 8, "EU-RO-1", "ab" * 32, 123, "unit")
    assert body["cpuFlavorIds"] == ["cpu3c"] and body["cloudType"] == "SECURE"
    assert body["vcpuCount"] == 8 and body["ports"] == ["8000/http"]
    env_text = json.dumps(body["env"])
    assert "FANOUT_TOKEN_SHA256" in body["env"] and 'TOKEN":' not in env_text.replace("SHA256", "")
    assert "RUNPOD" not in env_text.upper().replace("RUNPOD:CPU3C", "")


def test_size_cap_and_workers():
    p = policy()
    assert runner.workers_for(p, 32) == 16
    assert runner.size_cap_for(p, 1) == 2
    assert runner.size_cap_for(p, 3) == 8
    assert runner.size_cap_for(p, 100) == 32


# ---------------------------------------------------------------- fake RunPod + local agents


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeRp:
    """Fake rp client: create_pod starts a local pod_agent; delete kills it."""

    def __init__(self, tmp, stub_env=None, stock=None, refuse=()):
        self.tmp = Path(tmp)
        self.procs = {}
        self.ports = {}
        self.names = {}
        self.created = []
        self.deleted = []
        self.balance_value = 40.0
        self.stub_env = stub_env or {}
        self.stock_rows = stock or {(8, "EU-RO-1"): ("High", 0.24), (2, "US-NC-1"): ("High", 0.06)}
        self.refuse = set(refuse)
        self.n = 0
        self.extra_pods = [{"id": "other1", "name": "someone-else"}]

    def balance(self):
        return self.balance_value

    def datacenters(self):
        return sorted({dc for _, dc in self.stock_rows})

    def stock(self, flavor, vcpu, ram, dc):
        assert flavor == "cpu3c" and ram == 2 * vcpu
        st = self.stock_rows.get((vcpu, dc))
        return {"stockStatus": st[0], "securePrice": st[1]} if st else {"stockStatus": None}

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm is True
        body = json.loads(Path(body_path).read_text())
        key = (body["vcpuCount"], body["dataCenterIds"][0])
        if key in self.refuse:
            raise RunPodError("no instances", {"http_status": 400})
        self.n += 1
        pod_id = f"pod{self.n}"
        port = free_port()
        root = self.tmp / pod_id
        env = dict(os.environ, **body["env"], **self.stub_env)
        env.update(
            {
                "FANOUT_ROOT": str(root),
                "FANOUT_PORT": str(port),
                "FANOUT_PIP_JSON": "[]",
                "FANOUT_WORKERS": str(body["vcpuCount"] // 2),
                "FANOUT_EXEC_JSON": json.dumps([sys.executable, str(self.tmp / "stub.py")]),
                "FANOUT_PROVIDER_TAG": "b" if self.n > 1 else "a",
            }
        )
        proc = subprocess.Popen(
            [sys.executable, str(runner.AGENT_SOURCE)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.procs[pod_id], self.ports[pod_id], self.names[pod_id] = proc, port, body["name"]
        self.created.append(body)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        return {"id": pod_id, "costPerHr": 0.03 * body["vcpuCount"], "desiredStatus": "RUNNING"}

    def delete_pod(self, pod_id, confirm):
        assert confirm is True
        if pod_id in self.procs:
            self.procs.pop(pod_id).kill()
            self.deleted.append(pod_id)
            self.names.pop(pod_id, None)
        elif pod_id.startswith("other"):
            raise AssertionError("deleted a pod the runner does not own")
        return {}

    def get_pod(self, pod_id):
        return {"id": pod_id} if pod_id in self.procs else None

    def list_pods(self):
        return self.extra_pods + [{"id": i, "name": n} for i, n in self.names.items()]

    def kill(self, pod_id):
        self.procs[pod_id].kill()

    def close(self):
        for p in self.procs.values():
            p.kill()


def make_repo(tmp):
    repo = Path(tmp) / "repo"
    (repo / "research/runpod_fanout").mkdir(parents=True)
    (repo / "research/apex_veto_v8_screen_20261002").mkdir(parents=True)
    for rel in jobspec.REQUIRED_REPO_FILES + ("research/apex_veto_v8_screen_20261002/screen.py",):
        (repo / rel).write_text("# stub\n")
    for rel in jobspec.PINNED_RUNNER_FILES:  # the guard requires byte-equal runner files
        shutil.copyfile(jobspec.HERE / Path(rel).name, repo / rel)
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "x"], check=True, env=env)
    (repo / "untracked_secret.txt").write_text("must never be uploaded")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, commit


class Clock:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        time.sleep(0.15)  # let the local agents work


def make_runner(tmp, episodes, fake, budget=1.0, pol=None, **over):
    repo, commit = make_repo(tmp)
    ck = Path(tmp) / "ck"
    ck.mkdir()
    data = b"fake hero weights"
    hero = hashlib.sha256(data).hexdigest()
    (ck / "hero.pth").write_bytes(data)
    allow = {hero: {"path": "hero.pth", "sha256": hero}}
    j = job(episodes, repo_commit=commit, checkpoints=[hero], **over)
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [hero, RANDOM, GREEDY, RANDOM, GREEDY]
    (Path(tmp) / "stub.py").write_text(STUB)
    p = pol or policy(
        checkpoint_root=str(ck),
        heartbeat_seconds=1,
        pull_every_seconds=3,
        artifacts_root=str(Path(tmp) / "artifacts"),
    )
    clock = Clock()
    return runner.Runner(
        j,
        p,
        allow,
        Path(tmp) / "runs" / "r1",
        rp=fake,
        budget=budget,
        confirm=True,
        repo=repo,
        clock=clock,
        sleep=clock.sleep,
        agent_factory=lambda pod_id, token: runner.AgentClient(
            f"http://127.0.0.1:{fake.ports[pod_id]}", token, timeout=5
        ),
        spawn_watchdog=lambda rr: None,
        preflight_fn=lambda: [],
        probe_workers=2,
    )


@pytest.fixture
def fake(tmp_path):
    state = tmp_path / "stubstate"
    state.mkdir()
    f = FakeRp(tmp_path, stub_env={"STUB_STATE": str(state)})
    yield f
    f.close()


def test_end_to_end_with_local_agents(tmp_path, fake):
    eps = [ep("A", seed=s, index=i) for i, s in enumerate((11, 12, 13))] + [
        ep("B", seed=s, index=i) for i, s in enumerate((11, 12, 13))
    ]
    r = make_runner(tmp_path, eps, fake)
    assert r.run() == 0
    recs = sorted(p.name for p in (r.run_dir / "records").rglob("*.json"))
    assert len(recs) == 6
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["final_get_pods_runner_owned"] == []
    assert receipt["completed"] == 6
    assert fake.created[0]["vcpuCount"] == 8  # biggest stocked size, capped at the need
    assert set(fake.deleted) == {"pod1"} and fake.procs == {}
    # the upload was a git archive: untracked files never reach the pod
    root = tmp_path / "pod1" / "repo"
    assert (root / "research/runpod_fanout/episode.py").exists()
    assert not (root / "untracked_secret.txt").exists()
    # write-once local records carry the stub platform/fanout stamps
    one = json.loads(next((r.run_dir / "records").rglob("A-scripted-11.json")).read_text())
    assert one["fanout"]["job_id"] == "unit-job"


def test_failed_episode_is_redispatched(tmp_path, fake):
    key = "apex-veto-v8-screen-v1__h1000__live/A-scripted-11.json"
    fake.stub_env["STUB_FAIL_ONCE"] = key
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    assert r.run() == 0
    assert r.attempts[key] == 1
    assert list((r.run_dir / "failures").glob("*.json"))


def test_lost_pod_requeues_and_cleans_up(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=s, index=i) for i, s in enumerate((1, 2))], fake)
    orig_tick = r.tick_pod
    killed = {}

    def tick(pod):
        if pod.id == "pod1" and pod.state == "ready" and pod.inflight and not killed:
            fake.kill("pod1")  # the pod vanishes mid-episode
            killed["t"] = r.clock()
        return orig_tick(pod)

    r.tick_pod = tick
    r.policy = dict(r.policy, heartbeat_lost_seconds=5, max_attempts_per_episode=3)
    code = r.run()
    assert code == 0, json.loads((r.run_dir / "receipt.json").read_text())
    assert "pod1" in fake.deleted and fake.procs == {}
    assert len(fake.created) >= 2


def test_divergent_duplicate_aborts_and_deletes(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    key = "apex-veto-v8-screen-v1__h1000__live/A-scripted-11.json"
    dest = r.run_dir / "records" / key
    orig = r.accept_record

    def accept(k, data, source):
        orig(k, data, source)
        entry = json.loads(data)
        entry["record"]["mass_integral"] += 5
        orig(k, json.dumps(entry).encode(), "pod-x")  # a differing duplicate arrives

    r.accept_record = accept
    assert r.run() in (3, 5)
    assert "DIVERGENT" in (r.stop_reason or "")
    assert fake.procs == {}
    assert dest.exists()


def test_budget_and_floor_block_launch(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake, budget=0.01)
    assert r.run() == 4  # nothing affordable: even 2 vCPU to the hard end exceeds $0.01
    assert fake.created == []
    r2 = runner.Runner(r.job, r.policy, r.allowlist, tmp_path / "x", rp=fake, budget=1.0)
    r2.hard_end = time.time() + 3600
    r2.balance_start = 5.5
    assert "floor" in r2.launch_allowed(0.96)
    r2.balance_start = 40
    assert r2.launch_allowed(0.96) is None
    assert "budget" in r2.launch_allowed(2.0) or "max_hourly" in r2.launch_allowed(2.0)


def test_create_refusal_falls_back_to_smaller(tmp_path, fake):
    fake.refuse = {(8, "EU-RO-1")}
    r = make_runner(tmp_path, [ep("A", seed=11), ep("B", seed=11)], fake)
    assert r.run() == 0
    assert [b["vcpuCount"] for b in fake.created] == [2]


def test_capacity_recheck_upgrades_and_retires_smaller(tmp_path, fake):
    fake.stock_rows = {(2, "US-NC-1"): ("High", 0.06)}
    fake.stub_env["STUB_SLEEP"] = "0.4"
    eps = [ep("A", seed=s, index=i) for i, s in enumerate(range(20, 30))]
    r = make_runner(tmp_path, eps, fake)
    r.policy = dict(r.policy, capacity_recheck_seconds=4)
    orig = r.tick_pod

    def tick(pod):
        if pod.state == "ready":  # a bigger size comes online once the first pod works
            fake.stock_rows[(8, "EU-RO-1")] = ("High", 0.24)
        return orig(pod)

    r.tick_pod = tick
    assert r.run() == 0
    assert [b["vcpuCount"] for b in fake.created][:2] == [2, 8]
    events = (r.run_dir / "events.jsonl").read_text()
    assert "pod_draining" in events and "retired (drained)" in events
    assert fake.procs == {}


def test_run_requires_confirm(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    r.confirm = False
    with pytest.raises(jobspec.JobError):
        r.run()


def test_watchdog_deletes_only_job_pods(tmp_path):
    class Rp:
        def __init__(self):
            self.pods = [
                {"id": "a", "name": "rpf-unit-job--r-1"},
                {"id": "b", "name": "rpf-unit-job-2--r-1"},
                {"id": "c", "name": "other"},
            ]
            self.deleted = []

        def list_pods(self):
            return list(self.pods)

        def delete_pod(self, pid, confirm):
            assert confirm
            self.deleted.append(pid)
            self.pods = [p for p in self.pods if p["id"] != pid]

    rp = Rp()
    t = {"now": 100.0}
    reason = watchdog.watch(
        "unit-job",
        150.0,
        tmp_path,
        1,
        rp=rp,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
        alive=lambda pid: True,
        poll=30,
    )
    assert reason == "hard deadline" and rp.deleted == ["a"]
    rp2 = Rp()
    (tmp_path / "n").mkdir()
    reason = watchdog.watch(
        "unit-job",
        1e12,
        tmp_path / "n",
        1,
        rp=rp2,
        clock=lambda: t["now"],
        sleep=lambda s: t.__setitem__("now", t["now"] + s),
        alive=lambda pid: False,
        poll=30,
    )
    assert reason == "runner dead" and rp2.deleted == ["a"]


def test_merge_refuses_mixed_platforms_and_compare(tmp_path):
    def write(run, key, entry):
        p = tmp_path / run / "records" / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(entry))

    write(
        "mac", "g/A-scripted-1.json", {"r": 1, "wall_seconds": 1, "platform": {"platform_id": "m"}}
    )
    write(
        "pod", "g/A-scripted-1.json", {"r": 1, "wall_seconds": 2, "platform": {"platform_id": "x"}}
    )
    write(
        "pod", "g/B-scripted-1.json", {"r": 2, "wall_seconds": 2, "platform": {"platform_id": "x"}}
    )
    with pytest.raises(runner.Abort, match="mixed"):
        runner.merge_runs([tmp_path / "mac", tmp_path / "pod"], tmp_path / "m")
    out = runner.compare_runs(tmp_path / "mac", tmp_path / "pod")
    assert out["compared"] == 1 and out["identical"] == 1 and out["only_b"]
    merged = runner.merge_runs([tmp_path / "pod"], tmp_path / "m2")
    assert merged["records"] == 2 and merged["platform_id"] == "x"


def test_cli_run_refuses_without_budget(tmp_path, monkeypatch, capsys):
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(job()))
    assert runner.main(["run", str(jf), "--confirm"]) == 2
    assert runner.main(["run", str(jf), "--budget", "0.5"]) == 2
    assert runner.main(["run", str(jf), "--budget", "60", "--confirm"]) == 2


def test_agent_rejects_bad_token_and_unlisted_uploads(tmp_path):
    port = free_port()
    token = "tok"
    env = dict(
        os.environ,
        FANOUT_ROOT=str(tmp_path / "r"),
        FANOUT_PORT=str(port),
        FANOUT_TOKEN_SHA256=hashlib.sha256(token.encode()).hexdigest(),
        FANOUT_PIP_JSON="[]",
    )
    proc = subprocess.Popen(
        [sys.executable, str(runner.AGENT_SOURCE)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        good = runner.AgentClient(f"http://127.0.0.1:{port}", token)
        bad = runner.AgentClient(f"http://127.0.0.1:{port}", "nope")
        assert good.get_json("/health")["pip"] == "ok"
        import urllib.error

        with pytest.raises(urllib.error.HTTPError):
            bad.get_json("/health")
        f = tmp_path / "x.pth"
        f.write_bytes(b"abc")
        with pytest.raises(urllib.error.HTTPError):
            good.put_file("/in/other.bin", f)
        with pytest.raises(urllib.error.HTTPError):  # name sha != content sha
            good.put_file("/in/ckpt/" + "0" * 64 + ".pth", f)
        sha = hashlib.sha256(b"abc").hexdigest()
        assert good.put_file(f"/in/ckpt/{sha}.pth", f)["sha256"] == sha
        with pytest.raises(urllib.error.HTTPError):
            good.get_bytes("/record/../agent.log")
    finally:
        proc.kill()
        shutil.rmtree(tmp_path / "r", ignore_errors=True)


def test_commit_with_different_runner_files_refused(tmp_path):
    repo, commit = make_repo(tmp_path)
    j = job(repo_commit=commit)
    assert jobspec.check_commit(repo, j) == []
    (repo / "research/runpod_fanout/wrappers.py").write_text("WRAPPERS = {}\n")
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "evil"], check=True, env=env)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    problems = jobspec.check_commit(repo, job(repo_commit=head))
    assert any("wrappers.py" in p for p in problems)


def test_create_timeout_that_made_a_pod_is_found_and_deleted(tmp_path, fake):
    real_create = fake.create_pod
    calls = {"n": 0}

    def flaky(body_path, max_hourly, confirm):
        calls["n"] += 1
        out = real_create(body_path, max_hourly, confirm)
        if calls["n"] == 1:  # the server made the pod but the client timed out
            raise RunPodError("rp.py timed out")
        return out

    fake.create_pod = flaky
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    assert r.run() == 0
    assert "pod1" in fake.deleted and fake.procs == {}
    events = (r.run_dir / "events.jsonl").read_text()
    assert "orphan_create_found" in events


def test_untracked_prefixed_pod_is_swept(tmp_path, fake):
    fake.extra_pods.append({"id": "pod-stray", "name": "rpf-unit-job--r1-99", "costPerHr": 0.06})
    deleted = []
    real_delete = fake.delete_pod

    def delete(pod_id, confirm):
        if pod_id == "pod-stray":
            deleted.append(pod_id)
            fake.extra_pods = [p for p in fake.extra_pods if p["id"] != pod_id]
            return {}
        return real_delete(pod_id, confirm)

    fake.delete_pod = delete
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    assert r.run() == 0
    assert deleted == ["pod-stray"]


def test_rp_client_refuses_under_pytest(monkeypatch):
    monkeypatch.undo()  # drop the fatal stub: the client's own guard must refuse
    with pytest.raises(RuntimeError, match="pytest"):
        RpClient().list_pods()
    with pytest.raises(RuntimeError, match="pytest"):
        runner.spawn_watchdog_process(None)


def test_ledger_counts_unsettled_budgets(tmp_path):
    p = policy(artifacts_root=str(tmp_path), prior_spend_usd=0.11)
    assert runner.project_ledger(p)["total_usd"] == 0.11
    runner.reserve_ledger(p, tmp_path / "a", "j", 2.0)
    assert runner.project_ledger(p)["total_usd"] == 2.11  # crashed run counts at full budget
    runner.settle_ledger(p, tmp_path / "a", 0.25, True)
    assert runner.project_ledger(p)["total_usd"] == 0.36


def test_project_cap_blocks_run(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake, budget=1.0)
    runner.reserve_ledger(r.policy, tmp_path / "other", "j", 49.5)
    assert r.run() == 3 and "project cap" in r.stop_reason
    assert fake.created == []


def test_deferred_signal_during_create_records_pod_first(tmp_path, fake):
    real_create = fake.create_pod

    def create(body_path, max_hourly, confirm):
        out = real_create(body_path, max_hourly, confirm)
        runner._raise_interrupt(2, None)  # Ctrl-C arrives while the POST is in flight
        return out

    fake.create_pod = create
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    assert r.run() == 130
    assert "pod1" in r.pods and "pod1" in fake.deleted and fake.procs == {}


# ---------------------------------------------------------------- TLS + watchdog survival


def test_ssl_context_verifies_with_certifi():
    import ssl

    from research.runpod_fanout import tls

    ctx = tls.ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert tls.ca_file() is None or Path(tls.ca_file()).is_file()


def test_tls_preflight_reports_cert_failure(monkeypatch):
    import ssl
    import urllib.error

    from research.runpod_fanout import tls

    def boom(req, timeout, context):
        assert context.verify_mode == ssl.CERT_REQUIRED
        raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer"))

    monkeypatch.setattr(tls.urllib.request, "urlopen", boom)
    problems = tls.preflight(["https://example.invalid/"])
    assert problems and "local issuer" in problems[0]


def test_run_aborts_before_any_pod_when_tls_preflight_fails(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    r.preflight_fn = lambda: ["https://rest.runpod.io: CERTIFICATE_VERIFY_FAILED"]
    assert r.run() == 3
    assert "TLS preflight" in r.stop_reason and fake.created == []


def test_runner_delete_retries_with_backoff(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    pod = runner.Pod(
        id="p1", name="rpf-unit-job--r1-1", vcpu=2, dc="X", rate=0.06, token="", created=r.clock()
    )
    calls = {"n": 0}

    class FlakyRp:
        def delete_pod(self, pod_id, confirm):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RunPodError("rp.py exit 1: no Keychain access")
            return {}

        def get_pod(self, pod_id):
            return {"id": pod_id}

    r.rp = FlakyRp()
    r.delete_pod(pod, "test")
    assert calls["n"] == 3 and pod.deleted is not None


def test_delete_everything_retries_tracked_pods_when_listing_fails(tmp_path, fake):
    r = make_runner(tmp_path, [ep("A", seed=11)], fake)
    pod = runner.Pod(
        id="p1", name="rpf-unit-job--r1-1", vcpu=2, dc="X", rate=0.06, token="", created=r.clock()
    )
    r.pods[pod.id] = pod
    state = {"deletes": 0, "alive": True}

    class Rp:
        def delete_pod(self, pod_id, confirm):
            state["deletes"] += 1
            if state["deletes"] < 4:
                raise RunPodError("rp.py exit 1")
            state["alive"] = False
            return {}

        def get_pod(self, pod_id):
            return {"id": pod_id} if state["alive"] else None

        def list_pods(self):
            raise RunPodError("rp.py exit 1 (GET /pods)")

    r.rp = Rp()
    left = r.delete_everything()
    assert pod.deleted is not None and state["deletes"] >= 4
    assert left  # listing never confirmed: reported as unconfirmed, not as clean


def test_watchdog_deletes_registry_pods_even_if_listing_fails(tmp_path):
    reg = tmp_path / "pods"
    reg.mkdir()
    (reg / "x.response.json").write_text(json.dumps({"id": "p9", "name": "rpf-unit-job--r1-1"}))
    (reg / "y.response.json").write_text(json.dumps({"id": "p8", "name": "rpf-other--r1-1"}))
    state = tmp_path / "rp.json"
    state.write_text(
        json.dumps(
            {
                "pods": [
                    {"id": "p9", "name": "rpf-unit-job--r1-1"},
                    {"id": "p8", "name": "rpf-other--r1-1"},
                ],
                "list_fails": True,
            }
        )
    )
    rp = watchdog.FileFakeRp(str(state))
    t = {"now": 0.0}
    watchdog.sweep(
        rp,
        "rpf-unit-job--r1-",
        lambda s: t.__setitem__("now", t["now"] + s),
        tmp_path,
        clock=lambda: t["now"],
        give_up_seconds=100,
    )
    data = json.loads(state.read_text())
    assert data["deleted"] and set(data["deleted"]) == {"p9"}


def test_watchdog_survives_killed_parent_and_hangup(tmp_path):
    """A parent spawns the watchdog detached and is SIGKILLed; the watchdog gets SIGHUP
    (terminal gone) and still deletes the run's pods at its deadline."""
    import signal as sig

    run_dir = tmp_path / "run"
    (run_dir / "pods").mkdir(parents=True)
    (run_dir / "pods" / "a.response.json").write_text(
        json.dumps({"id": "pz", "name": "rpf-unit-job--run-1"})
    )
    state = tmp_path / "rp.json"
    state.write_text(
        json.dumps(
            {
                "pods": [
                    {"id": "pz", "name": "rpf-unit-job--run-1"},
                    {"id": "keep", "name": "someone-else"},
                ]
            }
        )
    )
    repo = Path(runner.__file__).resolve().parents[2]
    child_argv = [
        sys.executable,
        "-m",
        "research.runpod_fanout.watchdog",
        "--job-id",
        "unit-job",
        "--fire-epoch",
        str(time.time() + 3),
        "--run-dir",
        str(run_dir),
        "--runner-pid",
        "0",
        "--prefix",
        "rpf-unit-job--run-",
        "--resweep-seconds",
        "0.5",
        "--poll-seconds",
        "0.5",
    ]
    parent_code = (
        "import subprocess,sys,os,time,json\n"
        f"p=subprocess.Popen({child_argv!r},cwd={str(repo)!r},stdin=subprocess.DEVNULL,"
        f"stdout=open({str(run_dir / 'watchdog.log')!r},'ab'),stderr=subprocess.STDOUT,"
        "start_new_session=True)\n"
        "print(p.pid,flush=True)\ntime.sleep(60)\n"
    )
    env = dict(os.environ, **{watchdog.FAKE_RP_ENV: str(state)})
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, env=env, text=True
    )
    wd_pid = int(parent.stdout.readline())
    parent.kill()  # the runner dies (kill -9)
    parent.wait()
    for _ in range(100):  # let the watchdog install its handlers (it writes its pid file)
        if (run_dir / "watchdog.pid").exists():
            break
        time.sleep(0.1)
    os.kill(wd_pid, sig.SIGHUP)  # and its terminal goes away
    deadline = time.time() + 60
    while time.time() < deadline and not (run_dir / "watchdog.done").exists():
        time.sleep(0.2)
    data = json.loads(state.read_text())
    assert data.get("deleted") == ["pz"]
    assert [p["id"] for p in data["pods"]] == ["keep"]
    assert json.loads((run_dir / "watchdog_result.json").read_text())["leftovers"] == []
