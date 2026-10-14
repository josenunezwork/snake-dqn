"""GPU runner v2: the opt-in M6 plan keys (image, extra_inputs, per-run init) and sizing checks.

"Off == old": a plan without the new keys must give the same pod body, trainer argv and
upload list as before (the golden values / the pre-M6 ``train_argv`` copy below fail on a
deliberate break). No RunPod call is ever made: the client, ledger and agent are fakes (the
real pod agent runs on 127.0.0.1 only).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.redesign_m3c_20261008 import gpu_run_v2 as gr
from tests.test_m3c_gpu_run_v2 import Clock, FakeAgent, _agent, _plan

M6_PLAN = gr.REPO / "research/redesign_m6_20261014/plans/m6_pod_plan.json"
B8_ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013")
M6_PENDING_FLAGS = {"--tight-cost", "--tight-min-len", "--death-mass-mult"}  # branch m6-reward
M6_COMMON = (
    "--reward mass --mass-kappa 1.0 --kill-scale 0 --gamma-end 0.998 --gamma-hold 0 "
    "--gamma-anneal-end 1 --mixed-frac 0.25 --anchor-mode v8margin --anchor-hold 0 "
    "--anchor-floor-at 1 --anchor-floor 0.1 --onpolicy-labels --obs-version bv8 "
    "--label-frac 1.0 --v61-backend numba --learner-fast --batch-scale 2 --obs-pack "
    "--worlds 128 --checkpoint-every 2000000"
).split()
M6_ARMS = {
    "CTRL": [],
    "TIGHT": ["--tight-cost", "2", "--tight-min-len", "600"],
    "DEATHW": ["--death-mass-mult", "2"],
}
# the plan's launch amendment (2026-10-14, plan notes): every arm then adds the fine-tune
# schedule flags; CTRL also adds --tight-telemetry (same-seed tight_frac baseline, same reward)
M6_SCHEDULE = ["--gamma-start", "0.998", "--eps-end-at", "1"]
M6_TAIL = {"CTRL": ["--tight-telemetry"], "TIGHT": [], "DEATHW": []}
SHA_A = "a" * 64


def _old_train_argv(plan, run):
    """gpu_run_v2.train_argv as it was before the opt-in keys (commit 836646c)."""
    argv = [
        "python",
        "research/redesign_m3c_20261008/train_m3c.py",
        "--seed",
        str(int(run["seed"])),
        "--init",
        "/r/in/ckpt/student_m2b_final.pth",
        "--demo-data",
        "/r/demo",
        "--transitions",
        str(int(plan["transitions"])),
        "--device",
        "cuda",
        "--out",
        f"/r/out/{run['name']}",
    ]
    if int(plan.get("actors", 0)):
        argv += ["--actors", str(int(plan["actors"]))]
        if plan.get("ring_dir"):
            argv += ["--ring-dir", str(plan["ring_dir"])]
    return argv + [str(a) for a in run.get("extra", [])]


def _committed_plans():
    return sorted(gr.REPO.glob("research/*/plans/*.json"))


def _opt_in_plan(tmp_path, n=3, **kw):
    """_plan() plus two pinned extra inputs; run 0 starts from the first one."""
    srcs = []
    for k in range(2):
        src = tmp_path / f"B8_s{k}.pth"
        src.write_bytes(f"weights {k}".encode())
        srcs.append(src)
    plan = _plan(n=n, **kw)
    plan["extra_inputs"] = [
        {"src": str(s), "dst": f"ckpt/B8_s{k}.pth", "sha256": gr.sha_file(s)}
        for k, s in enumerate(srcs)
    ]
    plan["runs"][0]["init"] = "/r/in/ckpt/B8_s0.pth"
    return plan


# ----------------------------------------------------------------------------- off == old
def test_pod_body_default_is_the_old_body():
    assert gr.IMAGE == "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    body = gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 96, 16)
    start = body.pop("dockerStartCmd")
    assert body == {
        "name": "m3c-gpu-x",
        "computeType": "GPU",
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "imageName": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        "containerDiskInGb": 120,
        "volumeInGb": 0,
        "minRAMPerGPU": 96,
        "minVCPUPerGPU": 16,
        "ports": ["8000/http"],
        "env": {
            "M3_TOKEN_SHA256": "a" * 64,
            "M3_DEADLINE_EPOCH": "1000",
            "M3_SELF_DELETE_EPOCH": "2000",
        },
    }
    blob = start[2].split("echo ", 1)[1].split(" |", 1)[0]
    assert gzip.decompress(base64.b64decode(blob)) == gr.AGENT.read_bytes()
    assert gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 96, 16, image=gr.IMAGE) == dict(
        body, dockerStartCmd=start
    )


def test_pod_body_image_is_opt_in_and_changes_nothing_else():
    old = gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 125, 36)
    new = gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 125, 36, image=gr.IMAGE_DEVEL)
    assert new["imageName"] == "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
    assert {k for k in old if old[k] != new[k]} == {"imageName"}
    two = gr.pod_body("m3c-gpu-x", "a" * 64, 1000.0, 2000.0, 125, 36, gpu_count=2)
    assert {k for k in old if old[k] != two[k]} == {"gpuCount", "minRAMPerGPU", "minVCPUPerGPU"}
    assert (two["gpuCount"], two["minRAMPerGPU"], two["minVCPUPerGPU"]) == (2, 63, 18)


@pytest.mark.parametrize("path", _committed_plans(), ids=lambda p: p.name)
def test_train_argv_of_every_plan_without_init_is_the_old_argv(path):
    plan = gr.load_plan(path)
    runs = [r for r in plan["runs"] if "init" not in r]
    if path != M6_PLAN:
        assert len(runs) == len(plan["runs"])  # the pre-M6 plans carry no opt-in key
    for run in runs:
        assert gr.train_argv(plan, run) == _old_train_argv(plan, run)


def test_train_argv_defaults_match_the_old_argv():
    for plan in (_plan(), _plan(actors=2), _plan(actors=3, ring_dir="/r/rings")):
        for run in plan["runs"]:
            assert gr.train_argv(plan, run) == _old_train_argv(plan, run)


@pytest.mark.parametrize("path", _committed_plans(), ids=lambda p: p.name)
def test_committed_plans_validate(path):
    assert gr.validate_plan(json.loads(path.read_text())) == []


@pytest.mark.parametrize(
    "path", [p for p in _committed_plans() if p != M6_PLAN], ids=lambda p: p.name
)
def test_pre_m6_plans_are_sized_consistently_and_use_known_flags(path):
    plan = gr.load_plan(path)
    # M4-A2 attempts 4-6 (branch m4b) size vcpu_per_run from pod-1 telemetry (2.3 = the
    # measured 215-236% CPU per run, plan notes), below the actors + 1 rule of thumb
    expected = (
        ["vcpu_per_run 2.3 < actors + 1 = 3: the learner and each actor keep about one core busy"]
        if path.name == "m4a2_pod_plan.json"
        else []
    )
    assert gr.plan_warnings(plan) == expected
    flags = gr.argparse_flags((gr.REPO / gr.TRAINER).read_text())
    assert gr.trainer_flag_problems(plan, flags) == []


# ----------------------------------------------------------------------------- validation
def test_validate_opt_in_keys(tmp_path):
    good = _opt_in_plan(tmp_path, image=gr.IMAGE_DEVEL, gpu_count=1)
    assert gr.validate_plan(good) == []

    def problems(**kw):
        plan = json.loads(json.dumps(good))
        plan.update(kw)
        return gr.validate_plan(plan)

    for image in ("", "pytorch/pytorch", "x:1; rm -rf /", "Pytorch/x:1", 7):
        assert any(p.startswith("image") for p in problems(image=image)), image
    for gpus in (0, 3, "2", True):
        assert any("gpu_count" in p for p in problems(gpu_count=gpus)), gpus
    # two GPUs (m4b): max_hourly may scale with the GPU count; RAM/vCPU minimums are totals
    assert problems(gpu_count=2, max_hourly=2 * gr.MAX_HOURLY) == []
    assert any("max_hourly" in p for p in problems(gpu_count=1, max_hourly=2 * gr.MAX_HOURLY))
    x = good["extra_inputs"][0]
    for dst in ("/abs.pth", "../x.pth", "ckpt/../x.pth", "ckpt/./x.pth", "a b.pth", ""):
        assert any(
            "must be a relative path" in p for p in problems(extra_inputs=[{**x, "dst": dst}])
        )
    for dst in ("repo.tar.gz", "ckpt/student_m2b_final.pth", "demo/part-000"):
        assert any("runner's own" in p for p in problems(extra_inputs=[{**x, "dst": dst}]))
    assert any("duplicate dst" in p for p in problems(extra_inputs=[x, dict(x)]))
    assert any("sha256" in p for p in problems(extra_inputs=[{**x, "sha256": "ABC"}]))
    assert any("exactly" in p for p in problems(extra_inputs=[{**x, "note": "n"}]))
    assert any("exactly" in p for p in problems(extra_inputs=[{"src": "s", "dst": "d"}]))
    assert "extra_inputs must be a list" in problems(extra_inputs={"src": "s"})
    # init must name an uploaded input; the student path is always uploaded
    bad = json.loads(json.dumps(good))
    bad["runs"][1]["init"] = "/r/in/ckpt/B8_s9.pth"
    assert any("B_s1: init" in p for p in gr.validate_plan(bad))
    ok = json.loads(json.dumps(good))
    ok["runs"][1]["init"] = gr.STUDENT_INIT
    assert gr.validate_plan(ok) == []
    # the runner's own flags stay out of extra, in either spelling
    for tok in (["--init", "/r/in/ckpt/B8_s0.pth"], ["--init=/r/in/ckpt/B8_s0.pth"], ["--seed=3"]):
        bad = json.loads(json.dumps(good))
        bad["runs"][0]["extra"] = tok
        assert any("extra may not set" in p for p in gr.validate_plan(bad)), tok


def test_per_run_init_replaces_the_student_once(tmp_path):
    plan = _opt_in_plan(tmp_path, actors=3, ring_dir="/r/rings")
    a0, a1 = (gr.train_argv(plan, r) for r in plan["runs"][:2])
    assert a0.count("--init") == 1 and a0[a0.index("--init") + 1] == "/r/in/ckpt/B8_s0.pth"
    assert a1.count("--init") == 1 and a1[a1.index("--init") + 1] == gr.STUDENT_INIT
    # everything but the init value is the old argv
    old = _old_train_argv(plan, plan["runs"][0])
    assert [t for t in a0 if t != "/r/in/ckpt/B8_s0.pth"] == [
        t for t in old if t != gr.STUDENT_INIT
    ]


def test_new_trainer_flags_pass_through_verbatim():
    extra = M6_COMMON + M6_ARMS["TIGHT"] + M6_ARMS["DEATHW"] + ["--death-value=-1.5"]
    plan = _plan(n=1, actors=3, ring_dir="/r/rings")
    plan["runs"][0]["extra"] = extra
    assert gr.validate_plan(plan) == []
    argv = gr.train_argv(plan, plan["runs"][0])
    assert argv[-len(extra) :] == extra  # not filtered, not reordered
    flags = gr.argparse_flags((gr.REPO / gr.TRAINER).read_text())
    problems = gr.trainer_flag_problems(plan, flags)
    pending = {p.split()[0] for p in problems}
    assert pending <= M6_PENDING_FLAGS  # the perf flags are all defined; values are skipped
    assert gr.trainer_flag_problems(plan, None) == [f"{gr.TRAINER} is not in the committed tree"]


# ----------------------------------------------------------------------------- warnings
def test_plan_warnings_ram_vs_obs_pack_and_other_sizing():
    def warns(extra=("--reward", "mass"), **kw):
        plan = _plan(n=4, **kw)
        for r in plan["runs"]:
            r["extra"] = list(extra)
        return gr.plan_warnings(plan)

    packed = ("--obs-pack",)
    assert warns(ram_per_run_gb=19) == []  # unpacked, sized like the pre-M6 plans
    assert any("< 16" in w for w in warns(ram_per_run_gb=8))
    assert any(
        "ring_gb_per_run 2.6 < 9" in w for w in warns(ram_per_run_gb=19, ring_gb_per_run=2.6)
    )
    assert warns(packed, ram_per_run_gb=8, ring_gb_per_run=2.6) == []
    assert any("< 6" in w for w in warns(packed, ram_per_run_gb=4, ring_gb_per_run=2.6))
    assert any("unpacked size" in w for w in warns(packed, ram_per_run_gb=19, ring_gb_per_run=2.6))
    assert any("ring_gb_per_run 9.3" in w for w in warns(packed, ram_per_run_gb=8))  # default
    mixed = _plan(n=2, ram_per_run_gb=19)
    mixed["runs"][0]["extra"] = ["--obs-pack"]
    assert any("1 of 2 runs" in w for w in gr.plan_warnings(mixed))
    assert any("actors + 1" in w for w in warns(ram_per_run_gb=19, actors=3, vcpu_per_run=3.2))
    assert warns(ram_per_run_gb=19, actors=3, vcpu_per_run=4.3) == []
    # torch.compile needs the devel image
    comp = ("--compile-learner", "reduce-overhead")
    assert any("devel" in w for w in warns(comp, ram_per_run_gb=19))
    assert any("devel" in w for w in warns(("--compile-actor=default",), ram_per_run_gb=19))
    assert warns(comp, ram_per_run_gb=19, image=gr.IMAGE_DEVEL) == []
    assert warns(("--compile-learner", "off"), ram_per_run_gb=19) == []
    # the initial wave on the smallest admitted host: no fit / VRAM
    assert any("no run fits" in w for w in warns(ram_per_run_gb=19, min_ram_gb=20))
    big = dict(ram_per_run_gb=19, max_concurrent_cap=10, min_vcpu=64, min_ram_gb=400)
    assert gr.initial_wave(_plan(n=12, **big)) == 10
    vram = gr.plan_warnings(_plan(n=12, vram_per_run_gb=2.5, **big))
    assert any("initial wave 10 runs" in w for w in vram)
    assert gr.plan_warnings(_plan(n=12, vram_per_run_gb=2.0, **big)) == []


# ----------------------------------------------------------------------------- inputs
def test_extra_input_files_are_pinned(tmp_path):
    plan = _opt_in_plan(tmp_path)
    rows = gr.extra_input_files(plan)
    assert rows == [{k: x[k] for k in ("src", "dst", "sha256")} for x in plan["extra_inputs"]]
    assert gr.extra_input_files(_plan()) == []
    bad = json.loads(json.dumps(plan))
    bad["extra_inputs"][1]["sha256"] = SHA_A
    with pytest.raises(SystemExit, match="B8_s1.pth hashes to"):
        gr.extra_input_files(bad)
    gone = json.loads(json.dumps(plan))
    gone["extra_inputs"][0]["src"] = str(tmp_path / "missing.pth")
    with pytest.raises(SystemExit, match="unreadable"):
        gr.extra_input_files(gone)


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """build_inputs against small stand-ins: git, the checkpoint pool, the student, demos."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008 import ni_spec

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    pins = []
    for name in ("pool_a.pth", "frp3.pth", "student.pth"):
        (ckpt / name).write_bytes(name.encode())
        pins.append((name, hashlib.sha256(name.encode()).hexdigest()))
    monkeypatch.setattr(dev_screen, "POOL", [pins[0]])
    monkeypatch.setattr(ni_spec, "FRP3_S12", pins[1])
    monkeypatch.setattr(gr, "CKPT_DIR", ckpt)
    monkeypatch.setattr(gr, "STUDENT", ckpt / "student.pth")
    monkeypatch.setattr(gr, "STUDENT_SHA", pins[2][1])
    demo = tmp_path / "demo"
    for r in gr.DEMO_ROUNDS:
        (demo / f"round-{r}").mkdir(parents=True)
        (demo / f"round-{r}" / "shard.npz").write_bytes(b"x" * 10)
    (demo / "margin.json").write_text("{}")
    monkeypatch.setattr(gr, "DEMO", demo)
    trainer_src = (gr.REPO / gr.TRAINER).read_text()
    real_run = subprocess.run

    def fake_run(argv, *a, **k):
        if argv[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="", returncode=0)
        if argv[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="c0ffee\n", returncode=0)
        if argv[:2] == ["git", "archive"]:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"tar")
            return SimpleNamespace(stdout="", returncode=0)
        if argv[:2] == ["git", "show"]:
            assert argv[2] == f"c0ffee:{gr.TRAINER}"
            return SimpleNamespace(stdout=trainer_src, returncode=0)
        return real_run(argv, *a, **k)

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    return tmp_path


def test_build_inputs_appends_pinned_extra_inputs_only_when_asked(fake_tree):
    old = _plan(n=2)
    old["_sha256"] = "p"
    base = gr.build_inputs(fake_tree / "w0", old)
    dsts = [f["dst"] for f in base["files"]]
    assert dsts == [
        "repo.tar.gz",
        "ckpt/pool_a.pth",
        "ckpt/frp3.pth",
        "ckpt/student_m2b_final.pth",
        "demo/part-000",
    ]
    assert base["trainer_flag_problems"] == []
    new = _opt_in_plan(fake_tree, n=2)
    new["_sha256"] = "p"
    got = gr.build_inputs(fake_tree / "w1", new)
    extra = [f for f in got["files"] if f["dst"].startswith("ckpt/B8_")]
    assert [f["dst"] for f in got["files"]] == dsts[:4] + ["ckpt/B8_s0.pth", "ckpt/B8_s1.pth"] + [
        "demo/part-000"
    ]
    assert extra == gr.extra_input_files(new)

    def pinned(files):
        return [(f["dst"], f["sha256"]) for f in files]

    assert pinned(f for f in got["files"] if f not in extra) == pinned(base["files"])  # as old
    assert got["bytes"] == base["bytes"] + sum(Path(f["src"]).stat().st_size for f in extra)


def test_build_inputs_refuses_a_bad_pin_a_clash_and_reports_unknown_flags(fake_tree):
    bad = _opt_in_plan(fake_tree, n=2)
    bad["_sha256"] = "p"
    bad["extra_inputs"][0]["sha256"] = SHA_A
    with pytest.raises(SystemExit, match="hashes to"):
        gr.build_inputs(fake_tree / "w2", bad)
    assert not (fake_tree / "w2" / "demo.tar").exists()  # refused before the demo tarball
    clash = _opt_in_plan(fake_tree, n=2)
    clash["_sha256"] = "p"
    clash["extra_inputs"][0]["dst"] = "ckpt/pool_a.pth"  # a pool checkpoint's slot
    clash["runs"][0]["init"] = "/r/in/ckpt/pool_a.pth"
    with pytest.raises(SystemExit, match="would replace"):
        gr.build_inputs(fake_tree / "w3", clash)
    m6 = _plan(n=2)
    m6["_sha256"] = "p"
    m6["runs"][1]["extra"] = M6_COMMON + M6_ARMS["TIGHT"]
    problems = gr.build_inputs(fake_tree / "w4", m6)["trainer_flag_problems"]
    assert {p.split()[0] for p in problems} <= M6_PENDING_FLAGS
    if "--tight-cost" not in gr.argparse_flags((gr.REPO / gr.TRAINER).read_text()):
        assert problems and "B_s1" in problems[0]


def test_agent_refuses_an_extra_input_whose_body_does_not_match(tmp_path, monkeypatch):
    mod, srv, agent = _agent(tmp_path, monkeypatch)
    try:
        src = tmp_path / "B8_s0.pth"
        src.write_bytes(b"B8 weights")
        with pytest.raises(urllib.error.HTTPError) as err:
            agent.put("ckpt/B8_s0.pth", str(src), SHA_A)
        assert err.value.code == 400  # a client error: drive() reports "upload failed"
        assert not (tmp_path / "r/in/ckpt/B8_s0.pth").exists()
        agent.put("ckpt/B8_s0.pth", str(src), gr.sha_file(src))
        assert (tmp_path / "r/in/ckpt/B8_s0.pth").read_bytes() == b"B8 weights"
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------------- drive / launch
def _drive_with(tmp_path, monkeypatch, agent, plan, inputs):
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
    args = SimpleNamespace(spend_cap=10.0, accept_no_self_delete="user decision in chat 2026-10")
    return gr.drive(agent, Rp(), tmp_path, state, inputs, args, plan), state


def test_drive_uploads_extra_inputs_and_starts_each_run_with_its_init(tmp_path, monkeypatch):
    plan = _opt_in_plan(tmp_path, n=3, transitions=2400 * 50, actors=3, ring_dir="/r/rings")
    files = [{"dst": "repo.tar.gz", "src": "x", "sha256": "s"}] + gr.extra_input_files(plan)
    inputs = {"files": files, "bytes": 1, "demo_sha256": "d", "commit": "c"}
    agent = FakeAgent(polls=40)
    out, _ = _drive_with(tmp_path / "run", monkeypatch, agent, plan, inputs)
    assert out == "complete", out
    assert agent.uploads == ["repo.tar.gz", "ckpt/B8_s0.pth", "ckpt/B8_s1.pth"]
    for run in plan["runs"]:
        argv = agent.jobs[run["name"]]["argv"]
        assert argv == gr.train_argv(plan, run) and argv.count("--init") == 1
    first = agent.jobs["B_s0"]["argv"]
    assert first[first.index("--init") + 1] == "/r/in/ckpt/B8_s0.pth"
    second = agent.jobs["B_s1"]["argv"]
    assert second[second.index("--init") + 1] == gr.STUDENT_INIT


class _Ledger:
    def __init__(self, policy):
        pass

    def register_run(self, *a):
        pass

    def reserve_pod(self, *a):
        return None

    def bind_pod(self, *a):
        pass

    def release_pod(self, *a):
        pass

    def finish_run(self, *a):
        pass


class _Rp:
    def __init__(self):
        self.pods, self.deleted, self.bodies = [], [], []

    def list_pods(self):
        return list(self.pods)

    def delete_pod(self, pid, confirm):
        self.deleted.append(pid)
        self.pods = [p for p in self.pods if p["id"] != pid]

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm
        body = json.loads(Path(body_path).read_text())
        self.bodies.append(body)
        self.pods.append({"id": "p1", "name": body["name"]})
        return {"id": "p1", "costPerHr": 0.74}

    def balance(self):
        return 66.0


def _launch(tmp_path, monkeypatch, plan, problems=(), rps=None):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod
    import research.runpod_fanout.tls as tls_mod

    rps = [] if rps is None else rps
    tmp_path.mkdir(parents=True, exist_ok=True)

    def make_rp():
        rps.append(_Rp())
        return rps[-1]

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(ledger_mod, "SharedLedger", _Ledger)
    monkeypatch.setattr(rp_mod, "RpClient", make_rp)
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
            "trainer_flag_problems": list(problems),
        },
    )
    monkeypatch.setattr(gr, "drive", lambda *a: "complete")
    args = SimpleNamespace(
        run_dir=str(tmp_path / "run"),
        plan=str(plan_path),
        lifetime_hours=3.0,
        spend_cap=4.0,
        rate=0.89,
    )
    return gr.launch(args), rps, tmp_path / "run"


def test_launch_creates_the_pod_with_the_plan_image_or_the_old_default(tmp_path, monkeypatch):
    code, rps, run = _launch(tmp_path / "old", monkeypatch, _plan())
    assert code == 0 and rps[0].bodies[0]["imageName"] == gr.IMAGE  # off == old
    state = json.loads((run / "attempt00/state.json").read_text())
    assert state["image"] == gr.IMAGE and state["pods_deleted_verified"]
    code, rps, run = _launch(tmp_path / "new", monkeypatch, _plan(image=gr.IMAGE_DEVEL))
    assert code == 0 and rps[0].bodies[0]["imageName"] == gr.IMAGE_DEVEL
    assert rps[0].deleted == ["p1"]
    assert json.loads((run / "attempt00/state.json").read_text())["image"] == gr.IMAGE_DEVEL


def test_launch_refuses_unknown_trainer_flags_before_any_runpod_call(tmp_path, monkeypatch):
    rps = []
    with pytest.raises(SystemExit, match="refusing: .*--tight-cost"):
        _launch(tmp_path, monkeypatch, _plan(), ["--tight-cost (1 runs) is not ..."], rps)
    # RpClient is never even constructed: no balance read, no create, nothing to delete
    assert rps == [] and not (tmp_path / "run" / "attempt00").exists()


def test_plan_cmd_reports_the_opt_in_keys_without_spending(tmp_path, monkeypatch, capsys):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod

    plan = _opt_in_plan(tmp_path, image=gr.IMAGE_DEVEL, actors=3, vcpu_per_run=4.3)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    calls = []

    class Rp:
        def create_pod(self, body_path, max_hourly, confirm):
            calls.append(("create", confirm, json.loads(Path(body_path).read_text())))
            return {"dry_run": True}

        def balance(self):
            return 66.0

    class Ledger:
        def __init__(self, policy):
            pass

        def peek(self):
            return {}

    monkeypatch.setattr(rp_mod, "RpClient", Rp)
    monkeypatch.setattr(ledger_mod, "SharedLedger", Ledger)
    monkeypatch.setattr(
        gr,
        "build_inputs",
        lambda work, plan: (
            work.mkdir(parents=True),
            {
                "commit": "c",
                "files": [1] * 7,
                "bytes": 9,
                "trainer_flag_problems": ["--tight-cost (1 runs) ..."],
            },
        )[1],
    )
    args = SimpleNamespace(
        plan=str(plan_path), run_dir=str(tmp_path / "m6"), lifetime_hours=3.0, spend_cap=4.0
    )
    assert gr.plan_cmd(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert [c[:2] for c in calls] == [("create", False)]  # rp.py's dry run only
    assert calls[0][2]["imageName"] == gr.IMAGE_DEVEL == out["image"]
    assert out["inits"]["B_s0"] == "/r/in/ckpt/B8_s0.pth"
    assert out["inits"]["B_s1"] == gr.STUDENT_INIT
    assert [x["dst"] for x in out["extra_inputs"]] == ["ckpt/B8_s0.pth", "ckpt/B8_s1.pth"]
    assert out["trainer_flag_problems_launch_refuses"] == ["--tight-cost (1 runs) ..."]
    assert out["plan_warnings"] == gr.plan_warnings(gr.load_plan(plan_path))
    assert out["g2_pod_wall_budget_s"] == pytest.approx(3 * 1000 / 10.0 / 4)


# ----------------------------------------------------------------------------- the M6 plan
def test_m6_plan_matches_its_spec():
    plan = gr.load_plan(M6_PLAN)
    want = {
        "schema": gr.PLAN_SCHEMA,
        "job_id": "redesign-m6-gpu",
        "gpu_count": 1,
        "image": gr.IMAGE_DEVEL,
        "min_ram_gb": 125,
        "min_vcpu": 32,
        "vcpu_per_run": 4.3,
        "ram_per_run_gb": 8,
        "actors": 3,
        "transitions": 8_000_000,
        "mac_rate": 5400.0,
        "g2_factor": 4,
        "max_hourly": 0.95,
        "ring_dir": "/r/rings",
        "max_concurrent_cap": 8,
        "vram_per_run_gb": 2.7,
        "preflight_transitions": 30_000,
    }
    assert {k: plan[k] for k in want} == want
    assert [x["dst"] for x in plan["extra_inputs"]] == [f"ckpt/B8_s{k}.pth" for k in range(5)]
    for k, x in enumerate(plan["extra_inputs"]):
        assert x["src"] == str(B8_ART / f"B8_s{k}" / "final.pth")
    names = [r["name"] for r in plan["runs"]]
    assert len(names) == 15 and len(set(names)) == 15
    by = {r["name"]: r for r in plan["runs"]}
    for k in range(5):
        for arm, flags in M6_ARMS.items():
            r = by[f"{arm}_s{10 + k}"]
            assert r["seed"] == 10 + k and r["init"] == f"/r/in/ckpt/B8_s{k}.pth"
            # pairs differ only by the arm's flags
            assert r["extra"] == M6_COMMON + flags + M6_SCHEDULE + M6_TAIL[arm]
    # no sizing warning: the initial wave on the 125 GB / 32 vCPU minimum is min(RAM 14,
    # CPU 7, cap 8) = 7 runs x 2.7 GB VRAM = 18.9 GB (< 90% of 24 GB); no RAM / pack mismatch
    assert gr.plan_warnings(plan) == []
    assert gr.initial_wave(plan) == 7


def test_m6_plan_flags_parse_with_the_trainer_except_the_pending_m6_ones():
    from research.redesign_m3c_20261008 import train_m3c

    plan = gr.load_plan(M6_PLAN)
    parser = train_m3c.build_parser()
    known = gr.argparse_flags((gr.REPO / gr.TRAINER).read_text())
    pending = M6_PENDING_FLAGS - known
    for run in plan["runs"]:
        argv = gr.train_argv(plan, run)[2:]
        keep, skip = [], False
        for tok in argv:
            if skip:
                skip = False
                continue
            if tok in pending:
                skip = True  # each pending flag takes one value
                continue
            keep.append(tok)
        args = parser.parse_args(keep)  # types / choices of every defined flag and value
        # attempt 4: an eager learner (--compile-learner's CUDA-graph pools ran out of VRAM)
        assert args.compile_learner == "off" and args.batch_scale == 2
        assert args.obs_pack and args.learner_fast and args.v61_backend == "numba"
        assert args.actors == 3 and args.worlds == 128 and args.init == run["init"]
        assert args.gamma_start == 0.998 and args.eps_end_at == 1
        assert args.tight_telemetry == run["name"].startswith("CTRL_")


@pytest.mark.skipif(not B8_ART.exists(), reason="the B8 finals live on the Mac only")
def test_m6_plan_pins_the_b8_finals_on_disk():
    plan = gr.load_plan(M6_PLAN)
    for x in plan["extra_inputs"]:
        assert gr.sha_file(Path(x["src"])) == x["sha256"]
        rows = [json.loads(r) for r in (Path(x["src"]).parent / "manifest.jsonl").open() if r]
        assert [r["sha256"] for r in rows if r["file"] == "final.pth"] == [x["sha256"]]
