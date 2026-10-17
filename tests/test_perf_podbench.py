"""CUDA pod bench session (research/perf_podbench_20261013/pod_bench.py).

No RunPod call is ever made: the RunPod client, ledger, TLS preflight, watchdog and pod agent
are fakes (RpClient also refuses real rp.py calls under pytest).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.perf_podbench_20261013 import pod_bench as pb
from research.redesign_m3c_20261008 import gpu_run_v2 as gr
from research.runpod_fanout.rp_client import RunPodError

REPO = Path(__file__).resolve().parents[1]


def _caps(learner_configs=("baseline", "t1", "t1_fp16"), actor_flags=(), learner_flags=()):
    lflags = {"--devices", "--mode", "--configs", "--updates", "--rounds", "--warm", "--out"}
    lflags |= {"--drift", "--drift-updates", "--init", "--note", *learner_flags}
    aflags = {"--configs", "--worlds", "--devices", "--frames", "--warm", "--init", "--out"}
    aflags |= {"--grow-to", "--trainer-args", *actor_flags}
    return {
        pb.LEARNER: {"flags": lflags, "configs": set(learner_configs)},
        pb.ACTOR: {"flags": aflags, "configs": {"baseline", "vectorized", "numba"}},
    }


def _cmd(name, *args, script=pb.LEARNER, timeout=60):
    argv = ["python", script, *args, "--out", f"{{out}}/{name}.json"]
    return {"name": name, "argv": argv, "timeout_s": timeout}


# ----------------------------------------------------------------------------- commands
def test_default_commands_run_on_the_committed_bench_scripts():
    """Every default command parses with the CURRENT bench scripts (no skip); the compile
    commands appear exactly when the committed benches carry perf-compile's flags (they do
    once perf-compile is merged)."""
    caps = {p: pb.script_caps((REPO / p).read_text()) for p in (pb.LEARNER, pb.ACTOR)}
    runnable, skipped = pb.split_runnable(pb.parse_commands(pb.default_commands(caps)), caps)
    assert skipped == []
    names = [c["name"] for c in runnable]
    compiled = "--compile" in caps[pb.LEARNER]["flags"]
    assert names[0] == "learner_bv8"
    assert names[-1] == ("learner_drift_compile" if compiled else "learner_drift")
    assert "learner_drift" in names
    assert {"actor_b", "actor_b_long", "actor_b8", "actor_b8_long"} <= set(names)
    if compiled:
        assert {"learner_bv8_compile", "learner_bv8_autotune"} <= set(names)
    assert ("actor_compile" in names) == ("--compile-actor" in caps[pb.ACTOR]["flags"])
    for c in runnable:
        argv = pb.resolve_argv(c["argv"])
        assert argv[argv.index("--devices") + 1] == "cuda"
        assert argv[argv.index("--out") + 1].startswith("/r/out/")
        assert not any("{" in a for a in argv)
    actor = pb.resolve_argv(runnable[names.index("actor_b_long")]["argv"])
    assert actor[actor.index("--configs") + 1] == "numba" and "--grow-to" in actor
    assert actor[actor.index("--worlds") + 1] == "64,128,256"
    assert actor[actor.index("--init") + 1] == pb.PLACEHOLDERS["student"]
    learner = runnable[0]["argv"]
    cfgs = learner[learner.index("--configs") + 1].split(",")
    assert {"baseline", "t1", "t1_fp16", "t1_bf16", "t1_k2", "t1_k4"} <= set(cfgs)


def test_compile_modes_are_included_only_when_the_bench_has_them():
    plain = pb.default_commands(_caps())
    assert not [c for c in plain["commands"] if "compile" in c["name"]]
    # default learner configs the committed bench lacks are left out, not failed
    argv = plain["commands"][0]["argv"]
    assert argv[argv.index("--configs") + 1] == "baseline,t1,t1_fp16"
    caps = _caps(learner_configs=("baseline", "t1", "t1_compile", "t1_compile_max"))
    caps[pb.ACTOR]["configs"].add("numba_compile")
    doc = pb.default_commands(caps)
    by = {c["name"]: c["argv"] for c in doc["commands"]}
    lc = by["learner_bv8_compile"]
    assert lc[lc.index("--configs") + 1] == "t1,t1_compile,t1_compile_max"
    ac = by["actor_compile"]
    assert ac[ac.index("--configs") + 1] == "numba,numba_compile"
    runnable, skipped = pb.split_runnable(pb.parse_commands(doc), caps)
    assert not skipped and {"learner_bv8_compile", "actor_compile"} <= {c["name"] for c in runnable}


def test_perf_compile_flag_interface_gets_compile_commands():
    """perf-compile's benches: learner ``--compile MODES`` (twins of the listed configs,
    ``t1_c*`` CONFIGS), actor ``--compile-actor MODES``; no --init / --trainer-args."""
    caps = _caps(
        learner_configs=("baseline", "t1", "t1_bf16", "t1_k2", "t1_cl", "t1_c", "t1_bf16_k2_c"),
        learner_flags=("--compile", "--compile-scope"),
        actor_flags=("--compile-actor", "--note"),
    )
    caps[pb.LEARNER]["flags"].discard("--init")
    caps[pb.ACTOR]["flags"].discard("--trainer-args")
    assert pb.compile_configs(caps[pb.LEARNER]["configs"]) == ["t1_bf16_k2_c", "t1_c"]
    doc = pb.default_commands(caps)
    names = [c["name"] for c in doc["commands"]]
    assert names == [
        "learner_bv8",
        "learner_bv8_compile",
        "actor_b",
        "actor_b_long",
        "learner_bv8_autotune",
        "actor_compile",
        "learner_drift",
        "learner_drift_compile",
    ]
    by = {c["name"]: c["argv"] for c in doc["commands"]}

    def val(name, flag):
        return by[name][by[name].index(flag) + 1]

    assert val("learner_bv8", "--configs") == "baseline,t1,t1_bf16,t1_k2"  # no compile twins
    assert val("learner_bv8_compile", "--configs") == "t1,t1_bf16,t1_k2"
    assert val("learner_bv8_compile", "--compile") == "default,reduce-overhead"
    assert val("learner_bv8_autotune", "--compile") == "max-autotune-no-cudagraphs"
    assert val("actor_compile", "--compile-actor") == "off,default,reduce-overhead"
    assert val("actor_compile", "--configs") == "numba" and val("actor_b", "--note") == pb.POD_NOTE
    assert val("learner_drift_compile", "--compile") == "default,reduce-overhead"
    assert "--compile" not in by["learner_drift"] and "--init" not in by["learner_bv8"]
    runnable, skipped = pb.split_runnable(pb.parse_commands(doc), caps)
    assert skipped == [] and [c["name"] for c in runnable] == names
    # the same list on benches WITHOUT the compile flags: exactly those commands are skipped
    for script, flag in ((pb.LEARNER, "--compile"), (pb.ACTOR, "--compile-actor")):
        caps[script]["flags"].discard(flag)
    runnable, skipped = pb.split_runnable(pb.parse_commands(doc), caps)
    assert {s["name"] for s in skipped} == {
        "learner_bv8_compile",
        "learner_bv8_autotune",
        "actor_compile",
        "learner_drift_compile",
    }
    assert all("--compile" in s["reason"] for s in skipped)


def test_missing_compile_flags_and_configs_skip_gracefully():
    caps = _caps()
    doc = {
        "schema": pb.COMMANDS_SCHEMA,
        "commands": [
            _cmd("ok", "--devices", "cuda", "--configs", "baseline,t1"),
            _cmd("flag", "--devices", "cuda", "--compile", "max-autotune"),
            _cmd("cfg", "--devices", "cuda", "--configs=t1,t1_compile"),
            _cmd("script", "--devices", "cuda", script="research/nope/bench.py"),
            _cmd("value", "--trainer-args", "--obs-version bv8", script=pb.ACTOR),
            _cmd("eqvalue", "--trainer-args=--obs-version bv8 --reward mass", script=pb.ACTOR),
            _cmd("eqflag", "--trainer-args=--obs-version bv8 --reward mass"),  # learner: no flag
        ],
    }
    runnable, skipped = pb.split_runnable(pb.parse_commands(doc), caps)
    assert [c["name"] for c in runnable] == ["ok", "value", "eqvalue"]
    why = {s["name"]: s["reason"] for s in skipped}
    assert "--compile" in why["flag"] and "t1_compile" in why["cfg"]
    assert "not in the committed tree" in why["script"]
    assert "['--trainer-args']" in why["eqflag"]


def test_command_schema_problems_are_refused():
    ok = {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("a", "--devices", "cuda")]}
    assert pb.parse_commands(ok)[0]["timeout_s"] == 60.0
    bad_cases = [
        {"schema": "x", "commands": [_cmd("a")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": []},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("a"), _cmd("a")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("../a")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("setup")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [dict(_cmd("a"), argv=["bash", "-c", "x"])]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("a", script="research/../../etc.py")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [_cmd("a", "{secret}")]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [dict(_cmd("a"), timeout_s=10**6)]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [dict(_cmd("a"), env={"RUNPOD_API_KEY": "x"})]},
        {"schema": pb.COMMANDS_SCHEMA, "commands": [dict(_cmd("a"), env={"M3_TOKEN": "x"})]},
    ]
    for doc in bad_cases:
        with pytest.raises(SystemExit):
            pb.parse_commands(doc)


def test_script_caps_reads_argparse_flags_and_configs():
    src = (
        "import argparse\nCONFIGS = {'a': {}, 'b_compile': {}}\n"
        "def main():\n    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--x', type=int)\n    ap.add_argument('-y', '--why')\n"
    )
    caps = pb.script_caps(src)
    assert {"--x", "-y", "--why", "--help"} <= caps["flags"]
    assert caps["configs"] == {"a", "b_compile"}
    assert pb.script_caps("x = 1\n")["configs"] is None


def test_setup_script_is_valid_bash_and_fails_fast():
    argv = pb.bench_setup_argv()
    assert argv[:2] == ["bash", "-c"] and argv[2].startswith("set -euo pipefail\n")
    assert subprocess.run(["bash", "-n", "-c", argv[2]]).returncode == 0
    assert argv[2].rstrip().endswith("echo SETUP_OK") and "torch.cuda.is_available()" in argv[2]
    assert "demo" not in argv[2]  # benches need no demonstration data


def test_pod_body_and_cost_bound():
    args = _args(Path("/nonexistent"))
    body = pb.bench_pod_body("perfbench-x", "a" * 64, 1000.0, 1600.0, args)
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"] and body["gpuCount"] == 1
    assert body["cloudType"] == "SECURE" and body["volumeInGb"] == 0
    assert body["containerDiskInGb"] == pb.CONTAINER_DISK_GB and body["imageName"] == gr.IMAGE
    assert body["env"]["M3_DEADLINE_EPOCH"] == "1000" and "RUNPOD_API_KEY" not in body["env"]
    cost = pb.cost_bound(args)
    assert cost["worst_case_usd"] == 0.95 and cost["spend_cap_usd"] == 1.5
    assert cost["worst_case_if_runner_dies_usd"] > cost["worst_case_usd"]
    assert pb.check_args(args) is None
    assert "spend cap" in pb.check_args(SimpleNamespace(**{**vars(args), "lifetime_hours": 2.0}))
    assert pb.check_args(SimpleNamespace(**{**vars(args), "lifetime_hours": 5.0}))
    assert pb.check_args(SimpleNamespace(**{**vars(args), "pod_prefix": "m3c-gpu-"}))
    assert pb.check_args(SimpleNamespace(**{**vars(args), "job_id": "redesign-m3c-gpu"}))
    assert pb.check_args(SimpleNamespace(**{**vars(args), "max_hourly": 1.5}))
    assert pb.check_args(SimpleNamespace(**{**vars(args), "accept_no_self_delete": "ok"}))


def test_cli_defaults_are_the_approved_bounds():
    a = pb.build_parser().parse_args(["bench", "--dry-run"])
    assert pb.check_args(a) is None
    assert a.lifetime_hours == 1.0 and a.max_hourly == 0.95 and a.spend_cap == 1.5
    assert a.pod_prefix == "perfbench-" and a.job_id == "perf-cuda-bench" and not a.confirm
    assert a.min_ram_gb == 96 and a.min_vcpu == 16 and a.image == gr.IMAGE


# ----------------------------------------------------------------------------- dirty tree
def _git(repo, *args):
    cfg = ["-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *cfg, *args], cwd=repo, check=True, capture_output=True)


def test_dirty_tree_is_refused_before_any_runpod_call(tmp_path, monkeypatch):
    import research.runpod_fanout.rp_client as rp_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "c")
    commit = pb.clean_commit(repo)
    assert len(commit) == 40
    tar = pb.build_bench_inputs(tmp_path / "in", commit, repo=repo, ckpts=[])
    assert [f["dst"] for f in tar["files"]] == ["repo.tar.gz"]
    for dirty in ("modified", "untracked"):
        if dirty == "modified":
            (repo / "a.txt").write_text("2")
        else:
            _git(repo, "checkout", "-q", "--", "a.txt")
            (repo / "new.txt").write_text("x")
        with pytest.raises(SystemExit, match="dirty"):
            pb.clean_commit(repo)

        def no_rp(*a, **k):
            raise AssertionError("RunPod touched on a dirty tree")

        monkeypatch.setattr(rp_mod, "RpClient", no_rp)
        args = _args(tmp_path / f"out-{dirty}", confirm=True)
        assert pb.bench_cmd(args, repo=repo) == 2
        assert not (tmp_path / f"out-{dirty}").exists()


def test_bad_checkpoint_sha_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "c")
    ck = tmp_path / "c.pth"
    ck.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    out = pb.build_bench_inputs(
        tmp_path / "in", pb.clean_commit(repo), repo=repo, ckpts=[(ck, "ckpt/c.pth", good)]
    )
    assert out["files"][1] == {"src": str(ck), "dst": "ckpt/c.pth", "sha256": good}
    with pytest.raises(SystemExit, match="pinned sha"):
        pb.build_bench_inputs(tmp_path / "in2", out["commit"], repo=repo, ckpts=[(ck, "x", "0")])


# ----------------------------------------------------------------------------- dry run
def test_dry_run_prints_body_commands_and_cost_without_runpod(tmp_path, monkeypatch, capsys):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod

    def forbidden(*a, **k):
        raise AssertionError("the dry run must not touch RunPod or the ledger")

    monkeypatch.setattr(rp_mod, "RpClient", forbidden)
    monkeypatch.setattr(ledger_mod, "SharedLedger", forbidden)
    monkeypatch.setattr(gr, "POD_NAME", gr.POD_NAME)
    monkeypatch.setattr(pb, "clean_commit", lambda repo=None: "c0ffee")
    monkeypatch.setattr(
        pb,
        "build_bench_inputs",
        lambda work, commit, **kwargs: {
            "commit": commit,
            "files": [{"dst": "repo.tar.gz", "src": "x", "sha256": "s"}],
            "bytes": 123,
        },
    )
    caps = _caps(learner_configs=("baseline", "t1", "t1_fp16", "t1_bf16", "t1_k2", "t1_k4"))
    caps[pb.LEARNER]["configs"].add("t1_bf16_k2")
    args = _args(None, dry_run=True)
    assert pb.bench_cmd(args, repo=tmp_path, caps_loader=lambda r, c, s: caps) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and out["commit"] == "c0ffee" and out["pod_prefix"] == "perfbench-"
    body = out["pod_body"]
    assert body["gpuTypeIds"] == [gr.GPU] and body["cloudType"] == "SECURE"
    assert body["gpuCount"] == 1 and body["name"].startswith("perfbench-")
    assert body["dockerStartCmd"].startswith("<bash -c")  # the agent blob is not printed
    assert out["cost_bound"]["worst_case_usd"] == 0.95
    assert [c["name"] for c in out["commands"]][0] == "learner_bv8"
    assert out["account"] is None and "4x" in out["no_g2_gate"]
    assert gr.POD_NAME == "perfbench-"
    # without --dry-run or --confirm nothing happens
    assert pb.bench_cmd(_args(tmp_path / "o"), repo=tmp_path) == 2


# ----------------------------------------------------------------------------- simulated session
class Clock:
    def __init__(self, monkeypatch, t0=2_000_000.0):
        self.t = t0
        monkeypatch.setattr(gr, "now", lambda: self.t)
        monkeypatch.setattr(gr.time, "sleep", self.sleep)

    def sleep(self, s):
        self.t += s


class FakeRp:
    def __init__(self, create="ok"):
        self.pods, self.deleted, self.created = [], [], []
        self.create_mode = create
        self.lists, self.visible_from = 0, 0

    def list_pods(self):
        self.lists += 1
        return list(self.pods) if self.lists >= self.visible_from else []

    def balance(self):
        return 50.0

    def get_pod(self, pid):
        return next((p for p in self.pods if p["id"] == pid), None)

    def delete_pod(self, pid, confirm):
        assert confirm
        self.deleted.append(pid)
        self.pods = [p for p in self.pods if p["id"] != pid]

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm
        body = json.loads(Path(body_path).read_text())
        self.created.append((body, max_hourly))
        if self.create_mode == "refused":
            raise RunPodError("rp.py exit 1", {"http_status": 400, "error": "bad request"})
        self.pods.append({"id": "pod1", "name": body["name"]})
        if self.create_mode == "late":  # created, answer lost, listed only after adoption
            self.visible_from = self.lists + 11  # gave up (10 adoption polls)
        if self.create_mode in ("unknown", "late"):  # created, but the answer was lost
            raise RunPodError("rp.py timed out")
        return {"id": "pod1", "costPerHr": 0.69, "machine": {"gpuTypeId": gr.GPU}}


class FakeAgent:
    """Each job exits after two health polls; a successful bench writes its --out file."""

    def __init__(self, fail=(), hang=(), interrupt_on=None, setup_ok=True):
        self.jobs, self.order, self.files, self.uploads, self.killed = {}, [], {}, [], []
        self.fail, self.hang, self.interrupt_on = set(fail), set(hang), interrupt_on
        self.setup_ok = setup_ok

    def health(self):
        procs = {}
        for name, j in self.jobs.items():
            j["n"] += 1
            rc = None
            if j.get("killed"):
                rc = -9
            elif j["n"] >= 2 and name not in self.hang:
                rc = 1 if name in self.fail or (name == "setup" and not self.setup_ok) else 0
                if rc == 0 and "--out" in j["argv"]:
                    rel = j["argv"][j["argv"].index("--out") + 1][len("/r/out/") :]
                    self.files[rel] = json.dumps({"rows": [{"name": name}]}).encode()
            procs[name] = {"pid": 1, "returncode": rc}
        return {
            "self_delete_armed": True,
            "self_delete_probe": 403,
            "gpu": "NVIDIA GeForce RTX 4090, 1 MiB, 24564 MiB, 0",
            "cpu_model": "AMD EPYC 7B13",
            "cpus": 32,
            "procs": procs,
        }

    def put(self, dst, src, sha):
        self.uploads.append(dst)

    def exec(self, name, argv, cwd, env):
        if name == self.interrupt_on:
            raise KeyboardInterrupt("signal 15")
        if name != "setup":
            assert cwd == "/r/repo" and env["SNAKE_CKPT_DIR"] == "/r/in/ckpt"
            assert env["OMP_NUM_THREADS"] == "1" and env["M3_COMMIT"] == "c0ffee"
        self.order.append(name)
        self.jobs[name] = {"n": 0, "argv": argv}

    def post_kill(self, name):
        self.killed.append(name)
        self.jobs[name]["killed"] = True

    def get(self, path, timeout=30):
        if path == "/log/setup":
            return b"torch 2.5.1 ...\nSETUP_OK\n" if self.setup_ok else b"pip failed\n"
        if path.startswith("/log/"):
            return f"log of {path[5:]}\n".encode()
        if path == "/ls/out":
            rows = [
                {"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                for p, b in self.files.items()
            ]
            return json.dumps(rows).encode()
        if path.startswith("/out/") and path[5:] in self.files:
            return self.files[path[5:]]
        raise OSError(404)


class FakeLedger:
    calls: list = []

    def __init__(self, policy):
        assert policy["max_hourly_per_pod_usd"] >= 0.95

    def register_run(self, run_id, job_id, run_dir, budget, pid):
        FakeLedger.calls.append(("register", job_id, budget))

    def reserve_pod(self, run_id, key, rate, until, balance):
        FakeLedger.calls.append(("reserve", key, rate))

    def bind_pod(self, run_id, key, pod_id, rate):
        FakeLedger.calls.append(("bind", pod_id))

    def release_pod(self, run_id, key, cost):
        FakeLedger.calls.append(("release", round(cost, 4)))

    def finish_run(self, run_id, success, complete):
        FakeLedger.calls.append(("finish", success, complete))


def _args(out_dir, **kw):
    base = dict(
        out_dir=out_dir,
        dry_run=False,
        confirm=False,
        commands=None,
        emit_commands=None,
        lifetime_hours=1.0,
        max_hourly=0.95,
        spend_cap=1.5,
        min_ram_gb=96,
        min_vcpu=12,
        image=gr.IMAGE,
        pod_prefix="perfbench-",
        job_id="perf-cuda-bench",
        accept_no_self_delete="user approved in chat (test)",
        with_account=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


COMMANDS = [
    pb.parse_commands({"schema": pb.COMMANDS_SCHEMA, "commands": [c]})[0]
    for c in (
        _cmd("learner_bv8", "--devices", "cuda", timeout=600),
        _cmd("actor_b", "--devices", "cuda", script=pb.ACTOR, timeout=600),
        _cmd("learner_drift", "--drift", "--devices", "cuda", timeout=600),
    )
]


def _session(tmp_path, monkeypatch, rp, agent, commands=COMMANDS, **options):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod
    import research.runpod_fanout.tls as tls_mod

    clock = Clock(monkeypatch)
    FakeLedger.calls = []
    monkeypatch.setattr(gr, "POD_NAME", gr.POD_NAME)  # restored after the test
    monkeypatch.setattr(ledger_mod, "SharedLedger", FakeLedger)
    monkeypatch.setattr(rp_mod, "RpClient", lambda: rp)
    monkeypatch.setattr(tls_mod, "preflight", lambda: [])
    monkeypatch.setattr(gr.signal, "signal", lambda *a: None)
    monkeypatch.setattr(pb, "keep_awake", lambda: None)
    spawned = []
    monkeypatch.setattr(
        gr, "spawn_watchdog", lambda run_dir, until, name: spawned.append((until, name)) or 0
    )
    monkeypatch.setattr(pb, "make_agent", lambda pod_id, token: agent)
    monkeypatch.setattr(
        pb,
        "build_bench_inputs",
        lambda work, commit, **kwargs: {
            "commit": commit,
            "files": [{"dst": "repo.tar.gz", "src": "x", "sha256": "s"}],
            "bytes": 1,
        },
    )
    out_dir = tmp_path / "session"
    t0 = clock.t
    code = pb.launch(_args(out_dir, confirm=True, **options), "c0ffee", commands, [])
    state = json.loads((out_dir / "state.json").read_text())
    summary = json.loads((out_dir / "bench_session.json").read_text())
    return code, state, summary, out_dir, spawned, t0


def test_session_happy_path_pulls_results_and_deletes_the_pod(tmp_path, monkeypatch):
    rp, agent = FakeRp(), FakeAgent()
    code, state, summary, out_dir, spawned, t0 = _session(tmp_path, monkeypatch, rp, agent)
    assert code == 0 and summary["outcome"] == "complete", summary["outcome"]
    body, max_hourly = rp.created[0]
    assert max_hourly == 0.95 and body["gpuTypeIds"] == [gr.GPU] and body["gpuCount"] == 1
    assert body["cloudType"] == "SECURE" and body["name"].startswith("perfbench-")
    assert int(body["env"]["M3_SELF_DELETE_EPOCH"]) - int(t0) == 3600  # 1.0 h hard lifetime
    assert int(body["env"]["M3_SELF_DELETE_EPOCH"]) - int(body["env"]["M3_DEADLINE_EPOCH"]) == 600
    assert spawned and spawned[0][1] == body["name"]  # the watchdog deletes exactly this pod
    assert agent.order == ["setup", "learner_bv8", "actor_b", "learner_drift"]
    assert agent.uploads == ["repo.tar.gz"]
    for name in ("learner_bv8", "actor_b", "learner_drift"):
        assert json.loads((out_dir / "out" / f"{name}.json").read_text())["rows"][0]["name"] == name
        assert (out_dir / "logs" / f"{name}.log").exists()
        assert summary["results"][f"{name}.json"] == {
            "items": 1,
            "errors": 0,
            "skipped_devices": None,
        }
    assert rp.pods == [] and rp.deleted == ["pod1"]
    assert state["pods_deleted_verified"] and state["pod_gone_by_id"] and state["watchdog_stopped"]
    assert (out_dir / "watchdog.stop").exists()
    assert FakeLedger.calls[0] == ("register", "perf-cuda-bench", 1.5)
    assert FakeLedger.calls[1][0] == "reserve" and FakeLedger.calls[1][2] == 0.95
    assert [c[0] for c in FakeLedger.calls] == ["register", "reserve", "bind", "release", "finish"]
    assert FakeLedger.calls[-1] == ("finish", True, True)
    assert summary["commit"] == "c0ffee" and summary["host"]["cpu_model"] == "AMD EPYC 7B13"
    assert [c["returncode"] for c in summary["commands"]] == [0, 0, 0]
    assert gr.POD_NAME == "perfbench-"


def test_create_failure_still_deletes_and_settles(tmp_path, monkeypatch):
    rp, agent = FakeRp(create="refused"), FakeAgent()
    code, state, summary, _, _, _ = _session(tmp_path, monkeypatch, rp, agent)
    assert code == 1 and summary["outcome"].startswith("create refused")
    assert rp.pods == [] and agent.order == [] and state["pods_deleted_verified"]
    assert [c[0] for c in FakeLedger.calls] == ["register", "reserve", "release", "finish"]
    assert FakeLedger.calls[-1] == ("finish", False, False) and FakeLedger.calls[-2][1] == 0.0
    # an unknown create outcome: the pod exists after all -> adopted by name, then deleted
    rp2, agent2 = FakeRp(create="unknown"), FakeAgent(setup_ok=False)
    code2, state2, summary2, _, _, _ = _session(tmp_path / "u", monkeypatch, rp2, agent2)
    assert code2 == 1 and summary2["outcome"] == "setup failed" and state2["pod_id"] == "pod1"
    assert rp2.pods == [] and rp2.deleted == ["pod1"] and state2["pods_deleted_verified"]
    # unknown outcome and the pod is listed only after adoption gave up: no pod id, but the
    # prefix sweep still finds and deletes it, and the elapsed time is charged to the ledger
    rp3, agent3 = FakeRp(create="late"), FakeAgent()
    code3, state3, summary3, _, _, _ = _session(tmp_path / "late", monkeypatch, rp3, agent3)
    assert code3 == 1 and summary3["outcome"].startswith("create error")
    assert "pod_id" not in state3 and agent3.order == []
    assert rp3.pods == [] and rp3.deleted == ["pod1"] and state3["pods_deleted_verified"]
    assert [c[0] for c in FakeLedger.calls] == ["register", "reserve", "release", "finish"]
    assert FakeLedger.calls[-2][1] > 0 and FakeLedger.calls[-1] == ("finish", False, False)


def test_bench_command_failure_and_timeout_continue_and_delete(tmp_path, monkeypatch):
    rp, agent = FakeRp(), FakeAgent(fail={"learner_bv8"}, hang={"actor_b"})
    code, state, summary, out_dir, _, _ = _session(tmp_path, monkeypatch, rp, agent)
    assert code == 1
    assert summary["outcome"] == "complete with failures: learner_bv8,actor_b"
    assert agent.order == ["setup", "learner_bv8", "actor_b", "learner_drift"]  # went on
    assert agent.killed == ["actor_b"] and state["commands"]["actor_b"]["timeout"]
    assert state["commands"]["learner_bv8"]["returncode"] == 1
    assert (out_dir / "out" / "learner_drift.json").exists()
    assert not (out_dir / "out" / "learner_bv8.json").exists()
    assert rp.pods == [] and rp.deleted == ["pod1"] and state["pods_deleted_verified"]
    assert FakeLedger.calls[-1] == ("finish", False, False)


def test_interrupt_deletes_the_pod(tmp_path, monkeypatch):
    rp, agent = FakeRp(), FakeAgent(interrupt_on="actor_b")
    code, state, summary, out_dir, _, _ = _session(tmp_path, monkeypatch, rp, agent)
    assert code == 1 and summary["outcome"].startswith("error: KeyboardInterrupt")
    assert agent.order == ["setup", "learner_bv8"]
    assert (out_dir / "out" / "learner_bv8.json").exists()  # pulled before the interrupt
    assert rp.pods == [] and rp.deleted == ["pod1"] and state["pods_deleted_verified"]
    assert [c[0] for c in FakeLedger.calls][-2:] == ["release", "finish"]
    assert FakeLedger.calls[-1] == ("finish", False, False)


def test_spend_cap_stops_the_session_and_deletes(tmp_path, monkeypatch):
    class Draining(FakeRp):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def balance(self):
            self.calls += 1
            return 50.0 if self.calls < 6 else 48.0  # a $2 drop >= cap 1.5 - 0.5

    rp, agent = Draining(), FakeAgent()
    code, state, summary, _, _, _ = _session(tmp_path, monkeypatch, rp, agent)
    assert code == 1 and summary["outcome"] == "spend cap"
    assert rp.pods == [] and state["pods_deleted_verified"]


def test_existing_pod_with_our_prefix_refuses_before_creating(tmp_path, monkeypatch):
    rp = FakeRp()
    rp.pods.append({"id": "old", "name": "perfbench-1013000000"})
    with pytest.raises(SystemExit, match="already exists"):
        _session(tmp_path, monkeypatch, rp, FakeAgent())
    assert rp.created == [] and FakeLedger.calls == []


# ----------------------------------------------------------------------------- bench scripts
def test_bench_scripts_accept_cuda_and_skip_it_without_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA host: the skip path is not reachable")
    from research.perf_learner_20261013 import bench_learner as bl

    out = tmp_path / "l.json"
    args = ["--devices", "cuda", "--configs", "baseline", "--ring-rows", "64", "--out", str(out)]
    args += ["--note", "pod note"]
    assert bl.main(args) == 0
    res = json.loads(out.read_text())
    assert res["skipped_devices"] == ["cuda"] and res["cells"] == [] and res["note"] == "pod note"
    with pytest.raises(SystemExit):
        bl.init_path(str(tmp_path / "missing.pth"))

    from research.perf_actor_20261013 import bench_actor as ba

    out2 = tmp_path / "a.json"
    argv = ["bench_actor.py", "--devices", "cuda", "--worlds", "8", "--init", "unused"]
    argv += ["--trainer-args=--obs-version bv8", "--out", str(out2)]
    monkeypatch.setattr(ba.sys, "argv", argv)
    assert ba.main() == 0
    res2 = json.loads(out2.read_text())
    assert res2["skipped_devices"] == ["cuda"] and res2["rows"] == []
    assert res2["trainer_args"] == "--obs-version bv8"


@pytest.mark.parametrize("cloud", ["SECURE", "COMMUNITY"])
def test_h100_cli_body_and_scoped_policy(cloud):
    args = pb.build_parser().parse_args(
        [
            "bench",
            "--dry-run",
            "--gpu-type",
            pb.H100_GPU,
            "--cloud",
            cloud.lower(),
            "--lifetime-hours",
            "1.5",
            "--spend-cap",
            "8",
            "--t16-bundle",
            "/tmp/t16.tar.gz",
        ]
    )
    assert pb.check_args(args) is None
    assert args.max_hourly == 4 and args.min_vcpu == 16 and args.image == gr.IMAGE_DEVEL
    body = pb.bench_pod_body("perfbench-t18-test", "a" * 64, 4800, 5400, args)
    assert body["gpuTypeIds"] == [pb.H100_GPU] and body["cloudType"] == cloud
    assert body["minVCPUPerGPU"] == 16 and body["volumeInGb"] == 0
    assert body["env"]["M3_DEADLINE_EPOCH"] == "4800"
    assert body["env"]["M3_SELF_DELETE_EPOCH"] == "5400"
    assert "RUNPOD_API_KEY" not in body["env"]
    cost = pb.cost_bound(args)
    assert cost["worst_case_usd"] == 6.0
    assert cost["ledger_reservation_usd"] == 6.46
    assert cost["spend_cap_usd"] == 8.0
    original = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    scoped = pb.bench_policy(args)
    assert scoped == {**original, "max_hourly_per_pod_usd": 4.0}
    assert json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text()) == original
    setup = pb.bench_setup_argv(args)
    assert subprocess.run(["bash", "-n", "-c", setup[2]]).returncode == 0
    assert "--managed-python" in setup[2] and "torch==2.9.1" in setup[2]
    assert "prepare.py" in setup[2] and "command -v gcc" in setup[2]


@pytest.mark.parametrize(
    "changes",
    [
        {"max_hourly": 4.01},
        {"spend_cap": 8.01},
        {"lifetime_hours": 1.5001},
        {"spend_cap": float("nan")},
        {"spend_cap": float("inf")},
        {"lifetime_hours": float("nan")},
        {"max_hourly": float("nan")},
        {"spend_cap": 6},
        {"min_vcpu": 0},
        {"cloud": "ALL"},
    ],
)
def test_h100_refuses_unsafe_bounds(changes):
    args = _args(None, gpu_type=pb.H100_GPU, lifetime_hours=1.5, max_hourly=4, spend_cap=8)
    for key, value in changes.items():
        setattr(args, key, value)
    assert pb.check_args(args)


def test_t16_bundle_is_pinned_and_replaces_default_uploads(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("test")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "test")
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"pinned fixture")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="pinned sha"):
        pb.build_bench_inputs(tmp_path / "bad", pb.clean_commit(repo), repo, t16_bundle=archive)
    monkeypatch.setattr(pb, "T16_BUNDLE_SHA", sha)
    monkeypatch.setattr(pb, "bench_ckpts", lambda: pytest.fail("unrelated checkpoint upload"))
    result = pb.build_bench_inputs(tmp_path / "in", pb.clean_commit(repo), repo, t16_bundle=archive)
    assert [f["dst"] for f in result["files"]] == ["repo.tar.gz", "t16_colab_inputs.tar.gz"]
    assert result["files"][1]["sha256"] == sha


def test_t16_profile_is_runnable_and_dry_run_has_no_api(tmp_path, monkeypatch):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod

    monkeypatch.setattr(rp_mod, "RpClient", lambda: pytest.fail("dry run touched RunPod"))
    monkeypatch.setattr(
        ledger_mod, "SharedLedger", lambda *a: pytest.fail("dry run touched ledger")
    )
    monkeypatch.setattr(gr, "POD_NAME", gr.POD_NAME)
    profile = REPO / "research/t18_h100pod_20261017/commands.json"
    commands = pb.parse_commands(json.loads(profile.read_text()))
    caps = {c["argv"][1]: pb.script_caps((REPO / c["argv"][1]).read_text()) for c in commands}
    commands, skipped = pb.split_runnable(commands, caps)
    assert len(commands) == 5 and not skipped
    commit = "a" * 40
    monkeypatch.setattr(
        pb,
        "session_inputs",
        lambda *a: {
            "commit": commit,
            "files": [{"dst": "t16_colab_inputs.tar.gz", "sha256": pb.T16_BUNDLE_SHA}],
            "bytes": 123,
        },
    )
    args = _args(
        None,
        dry_run=True,
        gpu_type=pb.H100_GPU,
        cloud="COMMUNITY",
        max_hourly=4,
        lifetime_hours=1.5,
        spend_cap=8,
        t16_bundle=tmp_path / "bundle",
    )
    plan = pb.dry_run(args, commit, commands, skipped)
    assert plan["account"] is None and plan["pod_body"]["gpuTypeIds"] == [pb.H100_GPU]
    assert plan["pod_body"]["cloudType"] == "COMMUNITY"
    for command in plan["commands"]:
        assert command["argv"][0] == pb.T16_PYTHON
        assert not any("{" in arg for arg in command["argv"])
    smoke = plan["commands"][-1]["argv"]
    assert smoke[smoke.index("--commit") + 1] == commit
    assert smoke[smoke.index("--seconds") + 1] == "300"


@pytest.mark.parametrize(
    "failure",
    [
        "success",
        "command",
        "setup",
        "interrupt",
        "create",
        "spend",
        "pull",
        "nan",
        "inf",
        "overrate",
    ],
)
def test_h100_session_preserves_cleanup_and_ledger(tmp_path, monkeypatch, failure):
    class H100Rp(FakeRp):
        def __init__(self):
            super().__init__(create="refused" if failure == "create" else "ok")
            self.balance_calls = 0

        def balance(self):
            self.balance_calls += 1
            return 42 if failure == "spend" and self.balance_calls > 6 else 50

        def create_pod(self, body_path, max_hourly, confirm):
            value = super().create_pod(body_path, max_hourly, confirm)
            value["costPerHr"] = {"nan": float("nan"), "inf": float("inf"), "overrate": 4.5}.get(
                failure, 3.5
            )
            return value

    class H100Agent(FakeAgent):
        def health(self):
            result = super().health()
            result.update(gpu="NVIDIA H100 80GB HBM3", self_delete_probe=200)
            return result

        def get(self, path, timeout=30):
            if failure == "pull" and path == "/ls/out":
                raise OSError("transfer failed")
            return super().get(path, timeout)

    rp = H100Rp()
    agent = H100Agent(
        fail={"actor_b"} if failure == "command" else (),
        setup_ok=failure != "setup",
        interrupt_on="actor_b" if failure == "interrupt" else None,
    )
    code, state, summary, _, spawned, t0 = _session(
        tmp_path,
        monkeypatch,
        rp,
        agent,
        gpu_type=pb.H100_GPU,
        cloud="COMMUNITY",
        max_hourly=4,
        lifetime_hours=1.5,
        spend_cap=8,
        accept_no_self_delete=None,
    )
    assert code == (0 if failure == "success" else 1)
    body, rate = rp.created[0]
    assert body["gpuTypeIds"] == [pb.H100_GPU] and body["cloudType"] == "COMMUNITY"
    assert rate == 4 and spawned[0][0] == t0 + 5400
    assert int(body["env"]["M3_SELF_DELETE_EPOCH"]) == int(t0 + 5400)
    assert int(body["env"]["M3_DEADLINE_EPOCH"]) == int(t0 + 4800)
    assert rp.pods == [] and state["pods_deleted_verified"]
    assert FakeLedger.calls[0] == ("register", "perf-cuda-bench", 8)
    assert FakeLedger.calls[1][0] == "reserve" and FakeLedger.calls[1][2] == 4
    assert FakeLedger.calls[-2][0] == "release" and FakeLedger.calls[-1][0] == "finish"
    assert summary["requested_gpu"] == pb.H100_GPU and summary["cloud"] == "COMMUNITY"
    assert math.isfinite(state["cost_estimate"])
    if failure in ("nan", "inf"):
        assert not any(call[0] == "bind" for call in FakeLedger.calls)
    if failure not in ("create", "nan", "inf", "overrate"):
        assert summary["host"]["gpu"] == "NVIDIA H100 80GB HBM3"
        assert summary["host"]["cpu_model"] == "AMD EPYC 7B13"
        assert summary["host"]["cpus"] == 32


def test_t16_session_uses_isolated_runtime_and_verified_bundle_env(tmp_path, monkeypatch):
    profile = REPO / "research/t18_h100pod_20261017/commands.json"
    commands = pb.parse_commands(json.loads(profile.read_text()))

    class T16Agent(FakeAgent):
        def exec(self, name, argv, cwd, env):
            if name == "setup":
                assert "prepare.py" in argv[2] and "/r/venv" in argv[2]
                assert "t16_colab_inputs.tar.gz" in argv[2]
            else:
                assert argv[0] == "/r/venv/bin/python"
                assert env["SNAKE_CKPT_DIR"] == "/r/in/t16-colab-inputs/checkpoints"
                assert env["NUMBA_CACHE_DIR"] == "/r/scratch/t16/numba-cache"
                assert env["TORCHINDUCTOR_CACHE_DIR"] == "/r/scratch/t16/inductor-cache"
                assert not any("{" in arg for arg in argv)
                if name == "smoke03":
                    assert argv[argv.index("--commit") + 1] == "c0ffee"
                    assert argv[argv.index("--source-archive") + 1] == "/r/in/repo.tar.gz"
            # The base fake's legacy checkpoint assertion is intentionally adapted only
            # after verifying the actual environment delivered by the T16 driver.
            super().exec(name, argv, cwd, {**env, "SNAKE_CKPT_DIR": "/r/in/ckpt"})

    rp, agent = FakeRp(), T16Agent()
    code, state, summary, out_dir, _, _ = _session(
        tmp_path,
        monkeypatch,
        rp,
        agent,
        commands,
        gpu_type=pb.H100_GPU,
        max_hourly=4,
        lifetime_hours=1.5,
        spend_cap=8,
        t16_bundle=tmp_path / "bundle.tar.gz",
    )
    assert code == 0 and rp.pods == [] and state["pods_deleted_verified"]
    assert agent.order == ["setup", *[c["name"] for c in commands]]
    assert all((out_dir / "out" / f"{c['name']}.json").exists() for c in commands[:4])
    assert all(c["argv"][0] == pb.T16_PYTHON for c in summary["commands"])
