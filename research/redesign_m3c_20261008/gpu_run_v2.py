#!/usr/bin/env python3
"""GPU runner v2 (M3-C): many training runs on ONE RunPod GPU pod, money-safe, observable.

v2 = the M3-B runner (``research/redesign_m3_20261008/gpu_run.py``) with every money-safety
path kept (ledger reservation, balance-drop hard cap, pod lifetime, detached Mac watchdog,
pod-side job kill at the deadline + dead-man switch, delete-and-verify on every exit, explicit
``--accept-no-self-delete`` waiver, definite-refusal-only create retries, opt-in stock polling
and GPU fallbacks), plus:

* **Run queue** from a sha-pinned plan JSON (``--plan``): N training runs (arm x seed), at most
  ``max_concurrent`` at once. The initial concurrency comes from the measured host (RAM, vCPU)
  and the plan's per-run footprint; an adaptive fill starts one more queued run (at most one
  per ``FILL_EVERY_S``) while the GPU is below ``FILL_GPU_UTIL`` %, RAM and CPUs have room.
  Sync runs do not depend on concurrency; multi-actor runs do (policy lag follows the
  learner's speed), so the queue is seed-major and the policy lag is logged per run.
* **Setup speed probe** (before any training): ``bench_m3c.py actor`` on the pod; the run is
  refused if the single-process actor rate is below ``plan.speed_floor`` (slow host).
* **Live telemetry** every poll: GPU util / VRAM / power, host CPU %, load, RAM, per-job CPU %
  and RSS (agent ``/telemetry``), per-run transitions and rate -> ``<run-dir>/telemetry.jsonl``.
* **Incremental pulls**: each run's ``manifest.jsonl`` is polled; every newly completed
  checkpoint is downloaded, sha-checked against the manifest and appended to
  ``<run-dir>/landed.jsonl`` -- ``probe_watch.py`` starts Mac-side probes from it. Logs and
  telemetry are pulled every ``LOG_PULL_S``.
* **Throughput gate G2** (the per-step 4x rule): ~30 min after the first run starts, the
  combined rate must be >= ``plan.g2_factor`` x ``plan.mac_rate`` (the measured Mac rate of
  this exact job) and the projected finish of the whole queue + 45 min must fit the remaining
  lifetime; otherwise the pod is deleted (Mac fallback).

Subcommands: ``plan`` (dry run: inputs, pod body, rp.py dry-run request; spends nothing),
``launch --confirm`` (spends money: only on the user's direct approval), ``cleanup --confirm``.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

AGENT = Path(__file__).resolve().parent / "gpu_agent_v2.py"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
CKPT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
STUDENT = ARTIFACTS / "redesign-m2b-20261008" / "student_m2b_final.pth"
STUDENT_SHA = "36a92948ff82618b1944431675414301ef27f22bc15c7e12329bf75fc7cac48f"
DEMO = ARTIFACTS / "redesign-m2b-20261008" / "data"
DEMO_ROUNDS = (10, 11, 12)
GPU = "NVIDIA GeForce RTX 4090"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
PIP = [
    "numpy==1.26.4",
    "numba==0.68.0",
    "llvmlite==0.50.0",
    "pydantic==2.12.5",
    "PyYAML==6.0.3",
    "psutil==5.9.8",
    "openskill==6.2.0",
    "tqdm",
]
TRAINER = "research/redesign_m3c_20261008/train_m3c.py"
BENCH = "research/redesign_m3c_20261008/bench_m3c.py"
G2_SECONDS = 1800
G2_ROWS_GRACE_S = 600
CHUNK = 16 << 20  # the RunPod proxy cuts requests at ~100 s
POD_NAME = "m3c-gpu-"
MAX_HOURLY = 1.0
PORT = 8000
UPLOAD_BUDGET_S = 3600
SETUP_BUDGET_S = 1800
PROBE_BUDGET_S = 600
STALL_S = 900
NET_FAIL_BUDGET_S = 900
CREATE_TRIES = 12
GPU_FALLBACKS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
)
NO_CAPACITY = re.compile(r"no (longer any |more )?instances? (currently |are )?available", re.I)
GPU_MAX_HOURLY = 0.80
STOCK_POLL_S = 300
STOCK_WAIT_S = 3 * 3600
CREATE_SETTLE_S = 300
POLL_S = 60
LOG_PULL_S = 300
PULL_BUDGET_S = 45
FILL_EVERY_S = 300  # each fill is judged on >= 5 min of the per-run rate
FILL_GPU_UTIL = 85.0
RAM_RESERVE_GB = 8.0
PLAN_SCHEMA = "m3c-gpu-plan/v1"


# ----------------------------------------------------------------------------- plan
def load_plan(path: Path) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    problems = validate_plan(plan)
    if problems:
        raise SystemExit(f"bad plan {path}: {problems}")
    plan["_sha256"] = hashlib.sha256(raw).hexdigest()
    return plan


def validate_plan(plan: Dict[str, Any]) -> List[str]:
    p: List[str] = []
    if plan.get("schema") != PLAN_SCHEMA:
        p.append("schema")
    runs = plan.get("runs") or []
    names = [r.get("name") for r in runs]
    if not runs or len(set(names)) != len(names):
        p.append("runs must be non-empty with unique names")
    for r in runs:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", str(r.get("name", ""))):
            p.append(f"bad run name {r.get('name')!r}")
        if not isinstance(r.get("seed"), int):
            p.append(f"{r.get('name')}: seed")
        for a in r.get("extra", []):
            if a in ("--out", "--device", "--init", "--demo-data", "--transitions", "--seed"):
                p.append(f"{r.get('name')}: extra may not set {a}")
    for k in ("transitions", "mac_rate", "g2_factor", "ram_per_run_gb", "vcpu_per_run"):
        if not isinstance(plan.get(k), (int, float)) or plan[k] <= 0:
            p.append(k)
    if plan.get("g2_factor", 0) < 4:
        p.append("g2_factor < 4 (the per-step 4x rule)")
    mh = plan.get("max_hourly")
    if not isinstance(mh, (int, float)) or not 0 < mh <= MAX_HOURLY:
        p.append(f"max_hourly must be in (0, {MAX_HOURLY}]")
    if int(plan.get("max_concurrent_cap", 0)) < 1:
        p.append("max_concurrent_cap")
    return p


# ----------------------------------------------------------------------------- pure parts
def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pod_body(
    name: str,
    token_sha: str,
    deadline: float,
    until: float,
    min_ram_gb: int = 96,
    min_vcpu: int = 16,
) -> Dict[str, Any]:
    blob = base64.b64encode(gzip.compress(AGENT.read_bytes(), mtime=0)).decode()
    start = (
        "mkdir -p /r && echo "
        + blob
        + " | base64 -d | gunzip > /r/agent.py && exec python /r/agent.py"
    )
    return {
        "name": name,
        "computeType": "GPU",
        "gpuTypeIds": [GPU],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "imageName": IMAGE,
        "containerDiskInGb": 120,  # code, demos (2.2 GB), file-backed rings (~7.6 GB per run)
        "volumeInGb": 0,
        "minRAMPerGPU": int(min_ram_gb),
        "minVCPUPerGPU": int(min_vcpu),
        "ports": [f"{PORT}/http"],
        "env": {
            "M3_TOKEN_SHA256": token_sha,
            "M3_DEADLINE_EPOCH": str(int(deadline)),
            "M3_SELF_DELETE_EPOCH": str(int(until)),
        },
        "dockerStartCmd": ["bash", "-c", start],
    }


def host_capacity(h: Dict[str, Any]) -> Tuple[float, float]:
    """(usable RAM GB, usable vCPUs) from an agent /health answer (the most restrictive)."""
    mems = [m for m in (h.get("cgroup_memory_gb"), h.get("mem_total_gb")) if m is not None]
    mem = min(mems) if mems else 0.0
    cpus = min(
        c
        for c in (h.get("cgroup_cpus"), h.get("cpus_allowed"), h.get("cpus"), 1e9)
        if c is not None
    )
    return float(mem), float(cpus)


def initial_concurrency(plan: Dict[str, Any], mem_gb: float, vcpus: float, guaranteed_vcpu: float):
    """Runs to start at once: bounded by RAM, by the GUARANTEED vCPUs (the pod's request, not
    the host's visible count) and by the plan's cap; at least 1 if one run fits at all."""
    by_ram = int((mem_gb - RAM_RESERVE_GB) // float(plan["ram_per_run_gb"]))
    by_cpu = int(min(vcpus, guaranteed_vcpu) // float(plan["vcpu_per_run"]))
    return max(0, min(by_ram, by_cpu, int(plan["max_concurrent_cap"]), len(plan["runs"])))


def mem_headroom_gb(tel: Dict[str, Any]) -> Optional[float]:
    """Container memory headroom: cgroup limit - cgroup usage (incl. page cache, so the
    file-backed rings count). None (= no adaptive fill) unless the cgroup reports both and
    the limit is real (not larger than the machine's memory)."""
    lim, used = tel.get("cgroup_memory_gb"), tel.get("cgroup_memory_used_gb")
    total = tel.get("mem_total_gb")
    if lim is None or used is None:
        return None
    if total is not None and float(lim) > float(total) * 1.01:
        return None
    return float(lim) - float(used)


def fill_decision(
    plan: Dict[str, Any],
    tel: Dict[str, Any],
    running: int,
    queued: int,
    last_fill: float,
    t: float,
    per_run_rate: Optional[float] = None,
    per_run_rate_at_last_fill: Optional[float] = None,
    combined_rate: Optional[float] = None,
    combined_at_last_fill: Optional[float] = None,
    young_runs: int = 0,
) -> Optional[str]:
    """Start one more queued run? Returns the reason it is OK, or None.

    All of: something queued; below the plan's hard cap; >= FILL_EVERY_S since the last fill;
    GPU < FILL_GPU_UTIL %; container memory headroom >= one run + reserve; VRAM headroom >=
    ``vram_per_run_gb``; disk >= ``disk_per_run_gb``; idle CPUs for one run; within the
    GUARANTEED vCPUs unless ``fill_beyond_guaranteed_vcpu``; and the last fill did not lower
    the per-run rate by more than 15% and raised the combined rate by >= 3% (adding runs must
    still add throughput). A known rate is required. Runs younger than 10 min still fill their
    observation rings: their ring bytes are reserved out of the memory headroom. With a cgroup
    CPU quota, concurrency is capped at quota / vcpu_per_run and idle CPU is the quota minus
    the jobs' measured CPU.
    """
    if queued <= 0 or running >= int(plan["max_concurrent_cap"]) or t - last_fill < FILL_EVERY_S:
        return None
    if per_run_rate is None:
        return None
    if per_run_rate_at_last_fill and per_run_rate < 0.85 * per_run_rate_at_last_fill:
        return None
    if combined_at_last_fill and (combined_rate or 0.0) < 1.03 * combined_at_last_fill:
        return None
    vcpu = float(plan["vcpu_per_run"])
    if not plan.get("fill_beyond_guaranteed_vcpu"):
        if (running + 1) * vcpu > float(plan.get("min_vcpu", 0)):
            return None
    quota = tel.get("cgroup_cpus")
    if quota:
        # the jobs' measured CPU (median, cores) if known; the plan's constant only before that
        used = sorted(
            j["cpu_pct"] / 100.0
            for j in (tel.get("jobs") or {}).values()
            if j.get("cpu_pct") is not None and j.get("processes")
        )
        per_job = used[len(used) // 2] if used else vcpu
        if (running + 1) * per_job > float(quota):
            return None
    gpus = [g for g in tel.get("gpu") or [] if g.get("util_pct") is not None]
    util = max((g["util_pct"] for g in gpus), default=None)
    vram = min(
        (
            (g["mem_total_mib"] - g["mem_used_mib"]) / 1024.0
            for g in gpus
            if g.get("mem_total_mib") is not None and g.get("mem_used_mib") is not None
        ),
        default=None,
    )
    mem = mem_headroom_gb(tel)
    if mem is not None:
        mem -= young_runs * float(plan.get("ring_gb_per_run", 9.3))
    cpu = tel.get("host_cpu_pct")
    ncpu = tel.get("cpus") or 0
    disk = tel.get("disk_free_gb")
    if None in (util, vram, mem, cpu, disk):
        return None
    if quota:
        busy = sum((j.get("cpu_pct") or 0.0) for j in (tel.get("jobs") or {}).values()) / 100.0
        idle_cpus = float(quota) - busy
        need_cpu = per_job + 0.5  # the jobs' measured CPU (A1: was the plan's vcpu + 1)
    else:
        idle_cpus = (1.0 - cpu / 100.0) * ncpu
        need_cpu = vcpu + 1.0
    if (
        util < FILL_GPU_UTIL
        and mem >= float(plan["ram_per_run_gb"]) + RAM_RESERVE_GB
        and vram >= float(plan.get("vram_per_run_gb", 2.0))
        and disk >= float(plan.get("disk_per_run_gb", 12.0))
        and idle_cpus >= need_cpu
    ):
        return (
            f"gpu {util:.0f}% < {FILL_GPU_UTIL:.0f}%, mem headroom {mem:.0f} GB, "
            f"vram {vram:.1f} GB, disk {disk:.0f} GB, {idle_cpus:.0f} idle cpus"
        )
    return None


def g2_decision(
    rates: Dict[str, float],
    left: Dict[str, float],
    mac_rate: float,
    factor: float,
    remaining_lifetime: float,
    dead: Sequence[str] = (),
    elapsed: float = 0.0,
    total: Optional[float] = None,
) -> Dict[str, Any]:
    """G2 ~30 min after the first start (the per-step 4x rule, on rate AND on wall clock):

    * the combined rate of the running runs is >= factor x the Mac rate of this job;
    * the whole queue (``left`` = transitions still to do, incl. queued runs) finishes, at
      that rate, 45 min before the lifetime ends;
    * the pod's whole wall time for the step (``elapsed`` since create, incl. upload / setup,
      + the projected rest) is <= the Mac's time for the step (``total`` / mac_rate) / factor.
    """
    live = {r: v for r, v in rates.items() if r not in dead}
    combined = float(sum(live.values()))
    need = float(sum(left.values())) / combined if combined > 0 else float("inf")
    ok_rate = not dead and combined >= factor * mac_rate
    ok_time = need + 45 * 60 <= remaining_lifetime
    mac_wall = float(total) / mac_rate if total else None
    ok_wall = mac_wall is None or elapsed + need <= mac_wall / factor
    reason = None
    if not ok_rate:
        reason = "rate"
    elif not ok_wall:
        reason = "wall"
    elif not ok_time:
        reason = "lifetime"
    return {
        "combined_rate": combined,
        "per_run": rates,
        "dead_runs": list(dead),
        "ratio_vs_mac": combined / mac_rate,
        "projected_seconds_left": need,
        "pod_wall_projected": elapsed + need,
        "mac_wall": mac_wall,
        "remaining_lifetime": remaining_lifetime,
        "pass": bool(ok_rate and ok_time and ok_wall),
        "reason": reason,
    }


def spend_exceeded(balance0: float, balance: float, cap: float, margin: float = 0.5) -> bool:
    return balance0 - balance >= cap - margin


def run_rate(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Rate of one run from its log rows: the last row's transitions / wall (as M3-B), plus
    the recent rate over the last <= 5 rows."""
    if not rows:
        return None
    last = rows[-1]
    out = {
        "transitions": float(last["transitions"]),
        "wall": float(last["wall"]),
        "rate": last["transitions"] / last["wall"],
    }
    if len(rows) >= 2:
        first = rows[max(0, len(rows) - 6)]
        dt = last["wall"] - first["wall"]
        if dt > 0:
            out["recent_rate"] = (last["transitions"] - first["transitions"]) / dt
    return out


def parse_jsonl(data: bytes) -> List[Dict[str, Any]]:
    rows = []
    for line in data.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a partially written last line
    return rows


# ----------------------------------------------------------------------------- inputs
def build_inputs(work: Path, plan: Dict[str, Any]) -> Dict[str, Any]:
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008 import ni_spec

    work.mkdir(parents=True, exist_ok=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if status:
        raise SystemExit("refusing: the worktree is dirty (the pod runs a committed tree)")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    repo_tar = work / "repo.tar.gz"
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(repo_tar), commit], cwd=REPO, check=True
    )
    files: List[Dict[str, Any]] = [
        {"src": str(repo_tar), "dst": "repo.tar.gz", "sha256": sha_file(repo_tar)}
    ]
    ckpts = list(dev_screen.POOL) + [ni_spec.FRP3_S12]
    for name, sha in ckpts:
        src = CKPT_DIR / name
        if sha_file(src) != sha:
            raise SystemExit(f"{src} does not hash to {sha}")
        files.append({"src": str(src), "dst": f"ckpt/{name}", "sha256": sha})
    if sha_file(STUDENT) != STUDENT_SHA:
        raise SystemExit("the M2b student does not hash to its pinned sha")
    files.append({"src": str(STUDENT), "dst": "ckpt/student_m2b_final.pth", "sha256": STUDENT_SHA})
    demo_tar = work / "demo.tar"
    if not demo_tar.exists():
        with tarfile.open(demo_tar, "w") as tar:
            tar.add(DEMO / "margin.json", arcname="margin.json")
            for r in DEMO_ROUNDS:
                tar.add(DEMO / f"round-{r}", arcname=f"round-{r}")
    demo_sha = sha_file(demo_tar)
    parts = []
    with demo_tar.open("rb") as fh:
        i = 0
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            p = work / f"demo-part-{i:03d}"
            p.write_bytes(block)
            parts.append(
                {
                    "src": str(p),
                    "dst": f"demo/part-{i:03d}",
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
            i += 1
    files += parts
    return {
        "commit": commit,
        "files": files,
        "demo_sha256": demo_sha,
        "bytes": sum(Path(f["src"]).stat().st_size for f in files),
        "plan_sha256": plan["_sha256"],
    }


def setup_argv(demo_sha: str, cuda_mps: bool = False) -> List[str]:
    steps = [
        "set -e",
        "mkdir -p /r/repo /r/demo /r/rings",
        "df -h /dev/shm /r || true",
        "tar -xzf /r/in/repo.tar.gz -C /r/repo",
        "cat /r/in/demo/part-* > /r/demo.tar",
        f"echo '{demo_sha}  /r/demo.tar' | sha256sum -c -",
        "tar -xf /r/demo.tar -C /r/demo",
        "rm /r/demo.tar",
        "pip install -q --no-cache-dir " + " ".join(PIP),
        "python -c 'import torch, numba, numpy; assert torch.cuda.is_available(); "
        "print(torch.__version__, numba.__version__, numpy.__version__)'",
        "cd /r/repo && SNAKE_CKPT_DIR=/r/in/ckpt python -c "
        '\'import sys; sys.path.insert(0, "."); '
        "from research.redesign_scope_20261007 import grid_h5000_identity as g; g._context(1); "
        "import research.redesign_m3c_20261008.train_m3c; "
        "import research.redesign_m3c_20261008.bench_m3c; "
        "import research.redesign_m2_20261008.gen_data; "
        "import src.simd_env.grid_sim, src.simd_env.fast_anchors; "
        "import src.simd_env.vector61_policy, src.simd_env.ego2s_policy; "
        "import src.simd_env.ego_raster, src.simd_env.ego_raster_b; "
        "import src.simd_env.ego_raster_nb, src.simd_env.grid_sim_nb; "
        "import src.simd_env.eval_engine, src.model.ego2s_network, src.core.world_runtime'",
    ]
    if cuda_mps:  # opt-in: concurrent kernels from many processes (no time-slicing)
        steps.append(
            "(nvidia-cuda-mps-control -d && echo CUDA_MPS_ON) || echo CUDA_MPS_UNAVAILABLE"
        )
    steps.append("echo SETUP_OK")
    return ["bash", "-c", " && ".join(steps)]


def probe_argv() -> List[str]:
    """The setup speed probe: one actor process, 64 worlds, CUDA acting, 200 frames."""
    return [
        "python",
        BENCH,
        "actor",
        "--device",
        "cuda",
        "--worlds",
        "64",
        "--frames",
        "200",
        "--student",
        "/r/in/ckpt/student_m2b_final.pth",
        "--out",
        "/r/out/_probe/actor.jsonl",
    ]


def train_argv(plan: Dict[str, Any], run: Dict[str, Any]) -> List[str]:
    argv = [
        "python",
        TRAINER,
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


# ----------------------------------------------------------------------------- agent client
_SSL: List[Any] = []


def _ssl():
    if not _SSL:
        from research.runpod_fanout.tls import ssl_context

        _SSL.append(ssl_context())
    return _SSL[0]


class Agent:
    def __init__(self, pod_id: str, token: str, base: Optional[str] = None) -> None:
        self.base = base or f"https://{pod_id}-{PORT}.proxy.runpod.net"
        self.token = token

    def _req(self, method: str, path: str, data: Optional[bytes] = None, headers=None, timeout=60):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("X-M3-Token", self.token)
        req.add_header("User-Agent", "m3c-gpu-runner/2.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        ctx = _ssl() if self.base.startswith("https") else None
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()

    def health(self) -> Dict[str, Any]:
        return json.loads(self._req("GET", "/health", timeout=20))

    def telemetry(self) -> Dict[str, Any]:
        return json.loads(self._req("GET", "/telemetry", timeout=30))

    def put(self, dst: str, src: str, sha: str) -> None:
        data = Path(src).read_bytes()
        out = json.loads(self._req("PUT", "/in/" + dst, data, {"X-Sha256": sha}, timeout=900))
        if out.get("sha256") != sha:
            raise RuntimeError(f"upload of {dst} not confirmed: {out}")

    def exec(self, name: str, argv: List[str], cwd: str, env: Dict[str, str]) -> None:
        body = json.dumps({"name": name, "argv": argv, "cwd": cwd, "env": env}).encode()
        for attempt in range(4):
            try:
                self._req("POST", "/exec", body, {"Content-Type": "application/json"})
                return
            except urllib.error.HTTPError as exc:
                if exc.code == 409:  # "already running": an earlier attempt got through
                    return
                if exc.code < 500 or attempt == 3:
                    raise  # 410 = past the deadline, 4xx = bad request: never retried
                time.sleep(10)  # proxy 502/503/524: a duplicate would get 409
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(10)

    def get(self, path: str, timeout: float = 30) -> bytes:
        return self._req("GET", path, timeout=timeout)

    def post_kill(self, name: str) -> None:
        body = json.dumps({"name": name}).encode()
        self._req("POST", "/kill", body, {"Content-Type": "application/json"})


# ----------------------------------------------------------------------------- state / logging
def now() -> float:
    return time.time()


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    tmp = state_path(run_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
    tmp.replace(state_path(run_dir))


def our_pods(rp) -> List[Dict[str, Any]]:
    return [p for p in rp.list_pods() if str(p.get("name", "")).startswith(POD_NAME)]


def say(msg: str, err: bool = False) -> None:
    """print that never raises (a closed tmux pty makes every write fail with EIO)."""
    try:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


def alert(msg: str) -> None:
    say("ALERT: " + msg, err=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# ----------------------------------------------------------------------------- delete / watchdog
def delete_and_verify(rp, run_dir: Path, state: Dict[str, Any], reason: str) -> bool:
    """Delete our pod(s) with backoff until GET /pods shows none (pod self-delete is not
    available, so this never gives up early: alerts every 30 min, stops trying after 24 h)."""
    deadline = now() + 24 * 3600
    next_alert = now() + 1800
    wait = 10.0
    clean_since: Optional[float] = None
    deleted_known = False
    all_ids: set = set()
    while True:
        try:
            ids = {p.get("id") for p in our_pods(rp)}
            if not deleted_known:  # the bound pod once, even if GET /pods lags behind
                ids.add(state.get("pod_id"))
            ids.discard(None)
            for pid in ids:
                try:
                    rp.delete_pod(pid, confirm=True)
                    if pid == state.get("pod_id"):
                        deleted_known = True
                except Exception as exc:  # noqa: BLE001
                    say(f"delete {pid} failed: {exc}")
                    if "404" in str(exc) and pid == state.get("pod_id"):
                        deleted_known = True
            if not state.get("pod_id"):
                deleted_known = True
            all_ids |= ids
            left = our_pods(rp)
            settled = now() - float(state.get("create_attempted_at") or 0) >= CREATE_SETTLE_S
            if left:
                clean_since = None
            elif clean_since is None:
                clean_since = now()
            if not left and settled and now() - clean_since >= 60:
                state.update(deleted=now(), delete_reason=reason, pods_left=[])
                save_state(run_dir, state)
                say(json.dumps({"deleted": sorted(all_ids), "reason": reason, "our_pods_left": []}))
                return True
            state["pods_left"] = [p.get("id") for p in left]
        except Exception as exc:  # noqa: BLE001
            say(f"delete/verify error: {exc}")
        if now() > next_alert:
            next_alert = now() + 1800
            alert(f"pod(s) still listed: {state.get('pods_left')}; still deleting (until 24 h)")
        if now() > deadline:
            state.update(delete_reason=reason, delete_failed=True)
            save_state(run_dir, state)
            alert(f"pod(s) may still be running: {state.get('pods_left')}; run `cleanup --confirm`")
            return False
        time.sleep(15 if (not state.get("pods_left") and clean_since) else wait)
        wait = min(600.0, wait * 2)


WATCHDOG = r"""
import json, os, sys, time
sys.path.insert(0, {repo!r})
from research.runpod_fanout.rp_client import RpClient
log = open({log!r}, "a")
t = {fire!r}
log.write(json.dumps({{"watchdog": "armed", "pid": os.getpid(), "fire": t}}) + "\n")
log.flush()
stop = {stop!r}
while time.time() < t:  # short sleeps: a long sleep does not count macOS sleep time
    if os.path.exists(stop):  # the runner verified its pod is gone
        log.write(json.dumps({{"watchdog": "stopped", "t": time.time()}}) + "\n")
        log.flush()
        sys.exit(0)
    time.sleep(2)
rp = RpClient()
end = time.time() + 24 * 3600  # no pod self-delete: keep trying until a clean listing
wait = 60
while time.time() < end:
    try:
        pods = [p for p in rp.list_pods() if p.get("name") == {name!r}]
        if not pods:
            log.write(json.dumps({{"watchdog": "clean", "t": time.time()}}) + "\n")
            break
        for p in pods:
            rp.delete_pod(p["id"], confirm=True)
            log.write(json.dumps({{"watchdog": "deleted", "id": p["id"], "t": time.time()}}) + "\n")
    except Exception as exc:
        log.write(json.dumps({{"watchdog": "error", "error": repr(exc), "t": time.time()}}) + "\n")
    log.flush()
    time.sleep(wait)
    wait = min(600, wait * 2)
"""


def watchdog_code(run_dir: Path, until: float, name: str) -> str:
    """The watchdog deletes only the pod of this exact name (never a later relaunch's)."""
    assert name.startswith(POD_NAME) and len(name) > len(POD_NAME)
    return WATCHDOG.format(
        repo=str(REPO),
        log=str(run_dir / "watchdog.log"),
        stop=str(run_dir / "watchdog.stop"),
        fire=until + 300,
        name=name,
    )


def stop_watchdog(pid: Optional[int], run_dir: Optional[Path] = None) -> None:
    if run_dir is not None:
        (run_dir / "watchdog.stop").write_text(str(now()))
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass


def spawn_watchdog(run_dir: Path, until: float, name: str, wait_s: float = 60.0) -> int:
    """Start the detached watchdog; return its pid only once it is alive and logged "armed"."""
    code = watchdog_code(run_dir, until, name)
    compile(code, "<watchdog>", "exec")  # SyntaxError here, not later
    caff = ["caffeinate", "-i"] if sys.platform == "darwin" else []
    proc = subprocess.Popen(
        caff + [sys.executable, "-c", code],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(run_dir / "watchdog.err", "a"),
    )
    log = run_dir / "watchdog.log"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise SystemExit(f"refusing: the watchdog exited ({proc.returncode}); see watchdog.err")
        if log.exists() and '"armed"' in log.read_text():
            return proc.pid
        time.sleep(0.5)
    stop_watchdog(proc.pid, run_dir)
    raise SystemExit("refusing: the watchdog did not log 'armed'")


# ----------------------------------------------------------------------------- create
def stock_by_gpu(rp) -> Dict[str, List[str]]:
    q = "query { dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"
    try:
        out = rp._call("gql", q)
    except Exception:  # noqa: BLE001
        return {}
    rank = {"High": 0, "Medium": 1, "Low": 2}
    hits: Dict[str, List[tuple]] = {}
    for dc in out.get("dataCenters") or []:
        for g in dc.get("gpuAvailability") or []:
            if g.get("stockStatus") in rank:
                hits.setdefault(g.get("gpuTypeId"), []).append((rank[g["stockStatus"]], dc["id"]))
    return {gpu: [d for _, d in sorted(v)] for gpu, v in hits.items()}


def error_detail(exc: BaseException) -> str:
    payload = getattr(exc, "payload", None)
    text = json.dumps(payload, default=str) if payload is not None else ""
    return f"{exc} {text}".strip()[:1500]


def definite_refusal(exc: BaseException) -> bool:
    """True when RunPod answered with an HTTP error (the pod was NOT created)."""
    payload = getattr(exc, "payload", None)
    status = payload.get("http_status") if isinstance(payload, dict) else None
    if not isinstance(status, int):
        return False
    if 400 <= status < 500:
        return True
    body = payload.get("error")
    text = json.dumps(body).lower() if isinstance(body, (dict, list, str)) else ""
    return 500 <= status < 600 and bool(NO_CAPACITY.search(text))


def create_plans(rp, gpus=(GPU,)) -> List[tuple]:
    stock = stock_by_gpu(rp)
    plans: List[tuple] = [(gpus[0], None)]
    for gpu in gpus:
        plans += [(gpu, [d]) for d in stock.get(gpu, [])]
    return plans[:CREATE_TRIES]


def create_pod_retrying(
    rp,
    body: Dict[str, Any],
    run_dir: Path,
    state: Dict[str, Any],
    gpus=(GPU,),
    max_hourly: float = GPU_MAX_HOURLY,
):
    attempts = state.setdefault("create_attempts", [])
    err = ""
    for i, (gpu, plan_dcs) in enumerate(create_plans(rp, gpus)):
        if i:
            time.sleep(20)
            if [p for p in our_pods(rp) if p.get("name") == body["name"]]:
                return None, err, False  # the caller adopts it by name
        b = dict(body, gpuTypeIds=[gpu])
        if plan_dcs:
            b["dataCenterIds"] = plan_dcs
        path = run_dir / f"pod_body_try{i}.json"
        path.write_text(json.dumps(b))
        state["create_attempted_at"] = now()
        save_state(run_dir, state)
        try:
            out = rp.create_pod(path, max_hourly, confirm=True)
            attempts.append({"try": i, "gpu": gpu, "dcs": plan_dcs, "ok": True})
            state["gpu"] = gpu
            save_state(run_dir, state)
            return out, "", False
        except Exception as exc:  # noqa: BLE001
            err = error_detail(exc)
            attempts.append({"try": i, "gpu": gpu, "dcs": plan_dcs, "error": err})
            save_state(run_dir, state)
            say(f"create try {i} ({gpu}, {plan_dcs or 'any DC'}): {err}")
            if not definite_refusal(exc):
                return None, err, False  # unknown outcome: adopt-or-give-up, never re-POST
    return None, err, True


# ----------------------------------------------------------------------------- pulls
class PullBudget:
    """One bounded artifact-transfer allowance for a control-loop poll.

    Control observations happen before this budget is created.  A slow checkpoint must
    therefore never postpone the next health/progress observation long enough to look
    like a trainer stall.  The HTTP client cannot interrupt DNS/body reads already in
    progress, but it never starts another request once this allowance is exhausted.
    """

    def __init__(self, deadline: float, seconds: float = PULL_BUDGET_S) -> None:
        self.ends_at = min(now() + seconds, deadline)

    def timeout(self, cap: float) -> Optional[float]:
        remaining = self.ends_at - now()
        return min(cap, remaining) if remaining > 0 else None


class Puller:
    """Incremental, sha-checked pulls of completed checkpoints (manifest rows) and logs."""

    def __init__(self, agent: Agent, run_dir: Path, state: Dict[str, Any]) -> None:
        self.agent = agent
        self.dest = (run_dir / "out").resolve()
        self.landed = run_dir / "landed.jsonl"
        self.state = state
        self.pulled = state.setdefault("pulled", {})  # run -> [file, ...]
        self.last_logs = 0.0

    def _failure(self, path: str, started: float, exc: Exception) -> None:
        """Keep bounded, non-secret diagnostics for deferred artifact pulls."""
        failures = self.state.setdefault("pull_failures", [])
        failures.append(
            {"t": now(), "path": path, "seconds": now() - started, "error": type(exc).__name__}
        )
        del failures[:-20]

    def rotate(self, runs: Sequence[str], cursor: str) -> List[str]:
        """Give every run first pull position on successive polls."""
        ordered = list(runs)
        if not ordered:
            return ordered
        key = f"pull_cursor_{cursor}"
        start = int(self.state.get(key, 0)) % len(ordered)
        self.state[key] = start + 1
        return ordered[start:] + ordered[:start]

    def _confined(self, rel: str) -> Path:
        path = (self.dest / rel).resolve()
        if self.dest not in path.parents:
            raise ValueError(f"path escapes the output dir: {rel}")
        return path

    def checkpoints(self, run: str, budget: Optional[PullBudget] = None) -> List[Dict[str, Any]]:
        """Pull every manifest row of ``run`` not yet pulled; returns the new landed rows."""
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", run):
            raise ValueError(run)
        budget = budget or PullBudget(float("inf"))
        started = now()
        try:
            timeout = budget.timeout(30.0)
            if timeout is None:
                return []
            rows = parse_jsonl(self.agent.get(f"/out/{run}/manifest.jsonl", timeout=timeout))
        except Exception as exc:  # noqa: BLE001  (no manifest yet / transient)
            self._failure(f"/out/{run}/manifest.jsonl", started, exc)
            return []
        done = self.pulled.setdefault(run, [])
        new = []
        for row in rows:
            name = str(row.get("file", ""))
            if name in done or not re.fullmatch(r"[A-Za-z0-9_.\-]{1,80}\.pth", name):
                continue
            path = self._confined(f"{run}/{name}")
            timeout = budget.timeout(300.0)
            if timeout is None:
                break
            started = now()
            try:
                data = self.agent.get(f"/out/{run}/{name}", timeout=timeout)
                if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                    raise ValueError("sha mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(path)
            except Exception as exc:  # noqa: BLE001
                self._failure(f"/out/{run}/{name}", started, exc)
                continue  # retried on the next poll
            done.append(name)
            landed = {
                "run": run,
                "file": name,
                "path": str(path),
                "sha256": row["sha256"],
                "transitions": row.get("transitions"),
                "t": now(),
            }
            append_jsonl(self.landed, landed)
            new.append(landed)
        return new

    def logs(
        self, runs: Sequence[str], budget: Optional[PullBudget] = None, force: bool = False
    ) -> None:
        budget = budget or PullBudget(float("inf"))
        if not force and now() - self.last_logs < LOG_PULL_S:
            return
        # Do not advance the fairness cursor until a request can actually start.  Otherwise
        # checkpoint-first polls can phase-lock log-first polls onto only half the run list.
        if budget.timeout(120.0) is None:
            return
        attempted = False
        for run in self.rotate(runs, "logs"):
            for name in ("log.jsonl", "telemetry.jsonl", "summary.json"):
                timeout = budget.timeout(120.0)
                if timeout is None:
                    if attempted:
                        self.last_logs = now()
                    return
                attempted = True
                started = now()
                try:
                    data = self.agent.get(f"/out/{run}/{name}", timeout=timeout)
                except Exception as exc:  # noqa: BLE001
                    self._failure(f"/out/{run}/{name}", started, exc)
                    continue
                path = self._confined(f"{run}/{name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        if attempted:
            self.last_logs = now()


def download_rest(agent: Agent, run_dir: Path, budget_s: float = 900) -> Dict[str, Any]:
    """Final sweep: every file in /ls/out not yet present locally with the same sha."""
    dest = (run_dir / "out").resolve()
    t0, got, failed = now(), 0, []
    try:
        listing = json.loads(agent.get("/ls/out", timeout=max(10.0, min(300.0, budget_s))))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ls failed: {exc}"}
    for row in listing:
        if now() - t0 > budget_s:
            failed.append("<budget>")
            break
        path = (dest / row["path"]).resolve()
        if dest not in path.parents:
            failed.append(row["path"])
            continue
        if path.exists() and sha_file(path) == row.get("sha256"):
            continue
        ok = False
        for _ in range(3):
            if now() - t0 > budget_s:
                break
            try:
                timeout = max(10.0, min(300.0, budget_s - (now() - t0)))
                data = agent.get("/out/" + row["path"], timeout=timeout)
                if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                    raise ValueError("sha mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                got += 1
                ok = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(3)
        if not ok:
            failed.append(row["path"])
    return {"files": got, "failed": failed}


# ----------------------------------------------------------------------------- launch
def _raise_interrupt(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt(f"signal {signum}")


class PreCreateTransient(RuntimeError):
    """A read failed before anything was reserved or created (safe to retry next poll)."""


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)
    signal.signal(signal.SIGINT, signal.default_int_handler)


def launch(args) -> int:
    from research.runpod_fanout.rp_client import RpClient
    from research.runpod_fanout.tls import preflight

    if sys.platform == "darwin":  # keep the Mac awake while this process lives
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], start_new_session=True)
    plan = load_plan(Path(args.plan))
    root = Path(args.run_dir)
    if root.exists():
        print(f"refusing: {root} exists", file=sys.stderr)
        return 2
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({k: v for k, v in plan.items() if k != "_sha256"}))
    problems = preflight()
    if problems:
        raise SystemExit(f"TLS preflight failed: {problems}")
    inputs = build_inputs(root / "inputs", plan)
    rp = RpClient()
    t0 = now()
    k = 0
    while True:
        start = now()
        sub = root / f"attempt{k:02d}"
        sub.mkdir()
        (root / "CURRENT_ATTEMPT").write_text(sub.name)
        try:
            code, no_capacity = launch_attempt(args, rp, sub, inputs, plan)
        except PreCreateTransient as exc:
            say(f"{sub.name}: transient before create: {exc}")
            code, no_capacity = 1, True
        if not no_capacity:
            return code
        wait_s = min(STOCK_WAIT_S, 3600.0 * float(getattr(args, "stock_wait_hours", 0.0) or 0.0))
        if now() - t0 + STOCK_POLL_S > wait_s:
            alert(f"no capacity after {(now() - t0) / 3600:.2f} h; giving up at $0 (Mac fallback)")
            return code
        k += 1
        _install_signal_handlers()
        time.sleep(max(0.0, STOCK_POLL_S - (now() - start)))


def launch_attempt(args, rp, run_dir: Path, inputs: Dict[str, Any], plan: Dict[str, Any]):
    from research.runpod_fanout.ledger import SharedLedger

    _install_signal_handlers()
    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    try:
        mine = our_pods(rp)
        balance0 = rp.balance()
    except Exception as exc:  # noqa: BLE001
        raise PreCreateTransient(repr(exc)) from exc
    if mine:
        raise SystemExit("refusing: a pod of ours already exists")
    token = secrets.token_urlsafe(32)
    created = now()
    until = created + args.lifetime_hours * 3600
    deadline = until - 15 * 60
    name = POD_NAME + time.strftime("%m%d%H%M%S")
    body = pod_body(
        name,
        hashlib.sha256(token.encode()).hexdigest(),
        deadline,
        until,
        int(plan.get("min_ram_gb", 96)),
        int(plan.get("min_vcpu", 16)),
    )
    (run_dir / "pod_body.json").write_text(json.dumps(body))
    watchdog_pid = spawn_watchdog(run_dir, until, name)
    ledger = SharedLedger(policy)
    run_id = f"m3c-gpu-{int(created)}"
    try:
        ledger.register_run(run_id, "redesign-m3c-gpu", str(run_dir), args.spend_cap, os.getpid())
        refusal = ledger.reserve_pod(run_id, name, float(plan["max_hourly"]), until, balance0)
    except BaseException:
        stop_watchdog(watchdog_pid, run_dir)
        raise
    if refusal:
        ledger.finish_run(run_id, False, False)
        stop_watchdog(watchdog_pid, run_dir)
        raise SystemExit(f"ledger refused: {refusal}")
    state: Dict[str, Any] = {
        "accept_no_self_delete": getattr(args, "accept_no_self_delete", None),
        "run_id": run_id,
        "name": name,
        "balance0": balance0,
        "created": created,
        "until": until,
        "deadline": deadline,
        "commit": inputs["commit"],
        "plan_sha256": inputs["plan_sha256"],
        "spend_cap": args.spend_cap,
        "token_sha256": body["env"]["M3_TOKEN_SHA256"],
        "watchdog_pid": watchdog_pid,
    }
    save_state(run_dir, state)
    outcome, rate, no_capacity = "unknown", float(plan["max_hourly"]), False
    ok = False
    try:
        gpus = GPU_FALLBACKS if getattr(args, "gpu_fallbacks", False) else (GPU,)
        state["gpus_allowed"] = list(gpus)
        out, err, no_capacity = create_pod_retrying(
            rp, body, run_dir, state, gpus, float(plan["max_hourly"])
        )
        if err:
            outcome = f"create error: {err}"
        pod_id = out.get("id") if isinstance(out, dict) else None
        if not pod_id and not no_capacity:
            for _ in range(10):
                hits = [p for p in our_pods(rp) if p.get("name") == name]
                if hits:
                    pod_id = hits[0]["id"]
                    break
                time.sleep(30)
        if not pod_id:
            outcome = outcome if outcome != "unknown" else f"create failed: {out}"
        else:
            rate = float((out or {}).get("costPerHr") or args.rate)
            state.update(pod_id=pod_id, rate=rate, machine=(out or {}).get("machine"))
            save_state(run_dir, state)
            ledger.bind_pod(run_id, name, pod_id, rate)
            say(
                json.dumps(
                    {"pod_id": pod_id, "gpu": state.get("gpu"), "rate": rate, "until": until}
                )
            )
            outcome = drive(Agent(pod_id, token), rp, run_dir, state, inputs, args, plan)
    except BaseException as exc:  # noqa: BLE001  (incl. KeyboardInterrupt / SIGTERM / SIGHUP)
        outcome = f"error: {type(exc).__name__}: {exc}"
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the delete
        state["outcome"] = outcome
        save_state(run_dir, state)
        ok = delete_and_verify(rp, run_dir, state, outcome)
        try:
            cost = rate * (now() - created) / 3600.0 if state.get("pod_id") else 0.0
            ledger.release_pod(run_id, name, cost)
            ledger.finish_run(run_id, outcome == "complete", outcome == "complete")
            state.update(cost_estimate=cost, balance_end=rp.balance())
        except Exception as exc:  # noqa: BLE001
            alert(f"ledger/balance bookkeeping failed: {exc}")
        state["pods_deleted_verified"] = ok
        if ok:
            stop_watchdog(watchdog_pid, run_dir)
            state["watchdog_stopped"] = True
        save_state(run_dir, state)
    code = 0 if outcome == "complete" and state.get("pods_deleted_verified") else 1
    return code, bool(no_capacity and not state.get("pod_id") and ok)


def guard(rp, state, args) -> Optional[str]:
    if now() >= state["deadline"]:  # the agent kills jobs here
        return "lifetime"
    try:
        bal = rp.balance()
    except Exception:  # noqa: BLE001  (the lifetime already bounds the spend)
        return None
    state["balance_last"] = bal
    if spend_exceeded(state["balance0"], bal, args.spend_cap):
        return "spend cap"
    return None


class NetBudget:
    """Consecutive network failures are tolerated up to NET_FAIL_BUDGET_S."""

    def __init__(self) -> None:
        self.first_fail: Optional[float] = None

    def ok(self) -> None:
        self.first_fail = None

    def fail(self) -> bool:
        self.first_fail = self.first_fail or now()
        return now() - self.first_fail > NET_FAIL_BUDGET_S


def _wait_job(agent, rp, state, args, net, name: str, budget_s: float) -> Tuple[Optional[int], str]:
    """Poll /health until job ``name`` exits; (returncode, stop-reason-or-'')."""
    t0 = now()
    while True:
        stop = guard(rp, state, args)
        if stop:
            return None, stop
        if now() - t0 > budget_s:
            return None, f"{name} timeout"
        time.sleep(20)
        try:
            procs = agent.health()["procs"]
            net.ok()
        except Exception:  # noqa: BLE001
            if net.fail():
                return None, f"lost the agent during {name}"
            continue
        rc = procs.get(name, {}).get("returncode")
        if rc is not None:
            return rc, ""


def drive(agent, rp, run_dir, state, inputs, args, plan) -> str:
    log = run_dir / "progress.jsonl"
    tel_log = run_dir / "telemetry.jsonl"
    net = NetBudget()

    def note(**row):
        row["t"] = now()
        append_jsonl(log, row)
        say(json.dumps(row, default=str))

    # 1. agent up + host checks
    t0 = now()
    while True:
        stop = guard(rp, state, args)
        if stop:
            return stop
        try:
            h = agent.health()
            break
        except Exception:  # noqa: BLE001
            if now() - t0 > 1200:
                return "agent never came up"
            time.sleep(15)
    keys = ("gpu", "cpus", "cpus_allowed", "cgroup_cpus", "cgroup_memory_gb", "cpu_model")
    note(
        stage="agent_up",
        self_delete_armed=h.get("self_delete_armed"),
        **{k: h.get(k) for k in keys},
    )
    if not h.get("self_delete_armed") or h.get("self_delete_probe") != 200:
        why = f"pod self-delete not proven (probe {h.get('self_delete_probe')})"
        waiver = getattr(args, "accept_no_self_delete", None)
        if not waiver:
            return why
        state["self_delete_waived"] = {"reason": why, "note": waiver, "t": now()}
        save_state(run_dir, state)
        note(stage="self_delete_waived", reason=why, waiver=waiver)
    mem, cpus = host_capacity(h)
    min_ram, min_vcpu = float(plan.get("min_ram_gb", 96)), float(plan.get("min_vcpu", 16))
    if mem < min_ram - 2 or cpus < min_vcpu:
        return f"host too small (memory {mem} GB, cpus {cpus})"
    # 2. uploads (sha-verified), spend guard between files, time budget
    t0 = now()
    for f in inputs["files"]:
        stop = guard(rp, state, args)
        if stop:
            return stop
        if now() - t0 > UPLOAD_BUDGET_S:
            return "upload budget exceeded"
        while True:
            try:
                agent.put(f["dst"], f["src"], f["sha256"])
                net.ok()
                break
            except Exception as exc:  # noqa: BLE001
                client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < 500
                if client_error or net.fail() or now() - t0 > UPLOAD_BUDGET_S:
                    return f"upload failed: {f['dst']}: {exc}"
                time.sleep(15)
    note(stage="uploaded", files=len(inputs["files"]), bytes=inputs["bytes"], seconds=now() - t0)
    # 3. setup (bounded)
    agent.exec("setup", setup_argv(inputs["demo_sha256"], bool(plan.get("cuda_mps"))), "/r", {})
    rc, stop = _wait_job(agent, rp, state, args, net, "setup", SETUP_BUDGET_S)
    if stop:
        return stop
    tail = agent.get("/log/setup").decode(errors="replace")[-1500:]
    note(stage="setup_done", returncode=rc, tail=tail)
    if rc != 0 or "SETUP_OK" not in tail:
        return "setup failed"
    env = {
        "SNAKE_CKPT_DIR": "/r/in/ckpt",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "M3_COMMIT": inputs["commit"],
    }
    # 4. speed probe (refuse slow hosts before any training)
    agent.exec("probe", probe_argv(), "/r/repo", env)
    rc, stop = _wait_job(agent, rp, state, args, net, "probe", PROBE_BUDGET_S)
    if stop:
        return stop
    try:
        probe = parse_jsonl(agent.get("/out/_probe/actor.jsonl"))[-1]
    except Exception as exc:  # noqa: BLE001
        return f"speed probe failed ({rc}): {exc}"
    state["speed_probe"] = probe
    save_state(run_dir, state)
    note(stage="speed_probe", returncode=rc, rate=probe.get("rate"), floor=plan.get("speed_floor"))
    if float(probe.get("rate") or 0.0) < float(plan.get("speed_floor", 0.0)):
        return f"slow host: actor probe {probe.get('rate'):.0f}/s < floor {plan['speed_floor']}"
    # 5. the run queue: seed-major (A_s0, B_s0, D_s0, L_s0, A_s1, ...) so every arm trains
    # under the same concurrency profile and a truncated queue never drops a whole arm.
    order = {r["name"]: i for i, r in enumerate(plan["runs"])}
    queue_ = sorted(plan["runs"], key=lambda r: (int(r["seed"]), order[r["name"]]))
    status: Dict[str, Dict[str, Any]] = state.setdefault("runs", {})
    conc = initial_concurrency(plan, mem, cpus, float(plan.get("min_vcpu", 16)))
    if conc < 1:
        return f"no run fits the host (memory {mem} GB, cpus {cpus})"
    state["max_concurrent"] = conc
    note(stage="sizing", max_concurrent=conc, mem_gb=mem, cpus=cpus, runs=len(queue_))
    puller = Puller(agent, run_dir, state)
    total = float(plan["transitions"]) * len(queue_)

    def start_next(reason: str) -> None:
        run = queue_.pop(0)
        agent.exec(run["name"], train_argv(plan, run), "/r/repo", env)
        status[run["name"]] = {"started": now(), "reason": reason}
        run_progress[run["name"]] = now()
        save_state(run_dir, state)
        note(stage="run_started", run=run["name"], reason=reason)

    run_progress: Dict[str, float] = {}
    # A missing log is an observation failure, not evidence that a live trainer stalled.
    # Sustained loss of every affected observation gets its own bounded failure path below.
    log_read_failed: Dict[str, float] = {}

    def dl_budget() -> float:
        """Never download past the job-kill deadline + 10 min (the pod is deleted by `until`)."""
        return max(60.0, min(900.0, state["deadline"] + 600 - now()))

    for _ in range(conc):
        start_next("initial wave")
    first_start = now()
    g2, last_progress, best, uptime = None, now(), {}, 0.0
    last_fill, rate_at_fill, no_new_starts = now(), None, False
    combined_at_fill: Optional[float] = None
    last_per_run: Optional[float] = None
    # (t, {run: (transitions, wall) of its first log row inside the freeze window})
    freeze: Optional[Tuple[float, Dict[str, Tuple[float, float]]]] = None
    while True:
        stop = guard(rp, state, args)
        if stop:
            note(stage="stopping", reason=stop, download=download_rest(agent, run_dir, dl_budget()))
            return stop
        time.sleep(POLL_S)
        try:
            h = agent.health()
            tel = agent.telemetry()
            net.ok()
        except Exception:  # noqa: BLE001
            if net.fail():
                return "lost the agent"
            continue
        if h.get("uptime", 0) < uptime:
            return "pod container restarted"
        uptime = h.get("uptime", 0)
        procs = h.get("procs", {})
        started = list(status)
        if any(r not in procs for r in started):
            return "trainer process missing"
        rcs = {r: procs[r].get("returncode") for r in started}
        running = [r for r, rc in rcs.items() if rc is None]
        rates: Dict[str, Dict[str, float]] = {}
        observed_logs = set()
        for r in running:
            started_read = now()
            try:
                remaining = state["deadline"] - now()
                if remaining <= 0:
                    return "lifetime"
                rr = run_rate(
                    parse_jsonl(agent.get(f"/out/{r}/log.jsonl", timeout=min(30.0, remaining)))
                )
                observed_logs.add(r)
                log_read_failed.pop(r, None)
            except Exception:  # noqa: BLE001
                rr = None
                log_read_failed.setdefault(r, started_read)
            if rr:
                rates[r] = rr
                if rr["transitions"] > best.get(r, -1):
                    best[r] = rr["transitions"]
                    last_progress = run_progress[r] = now()
        for r in started:
            if r in running:
                continue
            if "finished" not in status[r]:
                status[r].update(finished=now(), returncode=rcs[r])
                note(stage="run_finished", run=r, returncode=rcs[r])
                last_progress = now()
        unreadable = [r for r in running if r not in observed_logs]
        if unreadable and any(now() - log_read_failed[r] > NET_FAIL_BUDGET_S for r in unreadable):
            note(stage="progress_observation_lost", runs=unreadable)
            return "lost progress observations"
        # per-run stall: a run without progress for STALL_S is killed (its slot is freed)
        for r in running:
            if (
                r in observed_logs
                and now() - run_progress.get(r, now()) > STALL_S
                and not status[r].get("killed")
            ):
                try:
                    agent.post_kill(r)
                    status[r]["killed"] = f"stalled {STALL_S}s"
                    note(stage="run_killed", run=r, reason="stalled")
                except Exception as exc:  # noqa: BLE001
                    note(stage="run_kill_failed", run=r, error=str(exc))
        # Evaluate stalls from this fresh control observation before any bounded artifact work.
        # A slow or failing pull is deferred to a later poll and cannot age a healthy run.
        if now() - last_progress > STALL_S and running and not unreadable:
            note(stage="stall", download=download_rest(agent, run_dir, dl_budget()))
            return "stalled"
        pull_budget = PullBudget(state["deadline"])
        # Alternate first access between logs and manifests.  Together with the rotating
        # manifest order this prevents one persistently slow run from starving every later
        # run's checkpoints or the diagnostic logs.
        logs_first = bool(state.get("pull_logs_first", False))
        state["pull_logs_first"] = not logs_first
        if logs_first:
            puller.logs(started, pull_budget)
        for r in puller.rotate(started, "checkpoints"):
            puller.checkpoints(r, pull_budget)
        if not logs_first:
            puller.logs(started, pull_budget)
        recent = [v.get("recent_rate", v["rate"]) for v in rates.values()]
        per_run = sum(recent) / len(recent) if recent else None
        if per_run:
            last_per_run = per_run  # sticky: a poll without rows does not unknow the rate
        combined = sum(recent) if recent else 0.0
        # Fills freeze from G2_SECONDS - 600 s until G2 is decided: G2 then measures a fixed
        # layout over >= 10 min (runner-side progress, not the runs' since-start averages).
        frozen = g2 is None and now() - first_start >= G2_SECONDS - 600
        if frozen and freeze is None:
            freeze = (now(), {})
        if frozen:  # each run's first row inside the window is its G2 baseline
            for r, v in rates.items():
                freeze[1].setdefault(r, (v["transitions"], v["wall"]))
        left = {r["name"]: float(plan["transitions"]) for r in queue_}
        for r in running:
            left[r] = max(0.0, float(plan["transitions"]) - best.get(r, 0.0))
        projected = sum(left.values()) / combined if combined > 0 else None
        append_jsonl(
            tel_log,
            {
                "t": now(),
                "gpu": tel.get("gpu"),
                "host_cpu_pct": tel.get("host_cpu_pct"),
                "loadavg": tel.get("loadavg"),
                "mem_available_gb": tel.get("mem_available_gb"),
                "mem_headroom_gb": mem_headroom_gb(tel),
                "disk_free_gb": tel.get("disk_free_gb"),
                "jobs": tel.get("jobs"),
                "rates": {r: v.get("recent_rate", v["rate"]) for r, v in rates.items()},
                "combined_rate": combined,
                "projected_seconds_left": projected,
                "transitions": best,
                "running": len(running),
                "queued": len(queue_),
                "balance": state.get("balance_last"),
            },
        )
        # Per-run start decision (every poll, after G2): start a queued run only if it alone can
        # finish (at 1.1 x the current per-run time) 45 min before the deadline. Once none can,
        # no new run starts; running runs continue to the deadline.
        hold = False
        if g2 is not None and queue_ and not no_new_starts:
            if last_per_run is None:
                hold = True  # no rate known yet: start nothing this poll
            else:
                need_one = float(plan["transitions"]) / last_per_run * 1.1
                if now() + need_one + 45 * 60 > state["deadline"]:
                    no_new_starts = True
                    note(stage="no_new_starts", per_run_seconds=need_one, queued=len(queue_))
        started_now = False
        if not no_new_starts and not hold:
            while queue_ and len(running) < state["max_concurrent"]:
                started_now = True
                start_next("slot freed")
                running.append("<new>")
            young = sum(1 for r in running if r == "<new>" or now() - status[r]["started"] < 600)
            why = None
            if not frozen and not started_now:
                why = fill_decision(
                    plan,
                    tel,
                    len(running),
                    len(queue_),
                    last_fill,
                    now(),
                    per_run,
                    rate_at_fill,
                    combined,
                    combined_at_fill,
                    young,
                )
            # The fill also runs before G2 (until the freeze): G2 then measures the layout the
            # guards allow (the guaranteed-vCPU start alone cannot reach 4x the Mac).
            if why:
                state["max_concurrent"] = len(running) + 1
                last_fill, rate_at_fill, combined_at_fill = now(), per_run, combined
                start_next(f"adaptive fill: {why}")
        done = set(rcs) == set(status) and all(rc is not None for rc in rcs.values())
        if done and (not queue_ or no_new_starts):
            dl = download_rest(agent, run_dir, dl_budget())
            puller.logs(started, PullBudget(state["deadline"] + 600), force=True)
            note(
                stage="finished",
                returncodes=rcs,
                download=dl,
                not_started=[r["name"] for r in queue_],
            )
            if queue_:
                return "finished partially (queue truncated by the lifetime)"
            if not all(rc == 0 for rc in rcs.values()):
                return "finished with failures"
            return (
                "complete" if not (dl.get("failed") or dl.get("error")) else "download incomplete"
            )
        if g2 is None and now() - first_start >= G2_SECONDS:
            dead = [r for r in started if rcs[r] not in (None, 0)]
            if (
                not dead
                and len(rates) < len(running)
                and now() - first_start < G2_SECONDS + G2_ROWS_GRACE_S
            ):
                continue
            # exact per-run rates: latest row vs the run's first row inside the window (runs
            # without two rows in the window are excluded; the window is >= 10 min)
            t0, b0 = freeze if freeze else (first_start, {})
            window = max(1.0, now() - t0)
            g2_rates = {}
            for r, v in rates.items():
                if r in b0 and v["wall"] > b0[r][1]:
                    g2_rates[r] = (v["transitions"] - b0[r][0]) / (v["wall"] - b0[r][1])
            g2 = g2_decision(
                g2_rates,
                left,
                float(plan["mac_rate"]),
                float(plan["g2_factor"]),
                state["until"] - now(),
                dead,
                elapsed=now() - float(state["created"]),
                total=total,
            )
            g2["window_seconds"] = window
            state["g2"] = g2
            save_state(run_dir, state)
            note(stage="G2", **{k: v for k, v in g2.items() if k != "per_run"})
            if not g2["pass"]:
                note(stage="g2_fail", download=download_rest(agent, run_dir, dl_budget()))
                return f"G2 fail ({g2['reason']})"


def cleanup(args) -> int:
    from research.runpod_fanout.rp_client import RpClient

    run_dir = Path(args.run_dir)
    sp = state_path(run_dir)
    state = json.loads(sp.read_text()) if sp.exists() else {}
    rp = RpClient()
    if not args.confirm:
        print(
            json.dumps({"would_delete": [p.get("id") for p in our_pods(rp)] or state.get("pod_id")})
        )
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    return 0 if delete_and_verify(rp, run_dir, state, "manual cleanup") else 1


def plan_cmd(args) -> int:
    from research.runpod_fanout.ledger import SharedLedger
    from research.runpod_fanout.rp_client import RpClient

    plan = load_plan(Path(args.plan))
    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    work = Path(str(args.run_dir) + "-plan")
    inputs = build_inputs(work / "inputs", plan)
    until = now() + args.lifetime_hours * 3600
    body = pod_body(
        POD_NAME + "plan",
        "0" * 64,
        until - 900,
        until,
        int(plan.get("min_ram_gb", 96)),
        int(plan.get("min_vcpu", 16)),
    )
    (work / "pod_body.json").write_text(json.dumps(body))
    rp = RpClient()
    dry = rp.create_pod(work / "pod_body.json", float(plan["max_hourly"]), confirm=False)
    out = {
        "commit": inputs["commit"],
        "plan_sha256": plan["_sha256"],
        "runs": [r["name"] for r in plan["runs"]],
        "transitions_per_run": plan["transitions"],
        "upload_files": len(inputs["files"]),
        "upload_bytes": inputs["bytes"],
        "balance": rp.balance(),
        "ledger": SharedLedger(policy).peek(),
        "project_cap_usd": policy["project_cap_usd"],
        "worst_case_usd": float(plan["max_hourly"]) * args.lifetime_hours,
        "max_hourly": plan["max_hourly"],
        "spend_cap_usd": args.spend_cap,
        "g2_rate": plan["g2_factor"] * plan["mac_rate"],
        "dry_run": dry,
        "body_without_cmd": {k: v for k, v in body.items() if k != "dockerStartCmd"},
    }
    print(json.dumps(out, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("plan", "launch", "cleanup"))
    ap.add_argument("--plan", default=None, help="the run plan JSON (schema m3c-gpu-plan/v1)")
    ap.add_argument("--run-dir", default=str(ARTIFACTS / "redesign-m3c-gpu"))
    ap.add_argument("--lifetime-hours", type=float, default=8.0)
    ap.add_argument("--spend-cap", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=0.89, help="fallback $/h if the create omits it")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--accept-no-self-delete", metavar="NOTE", default=None)
    ap.add_argument("--gpu-fallbacks", action="store_true")
    ap.add_argument("--stock-wait-hours", type=float, default=0.0)
    args = ap.parse_args()
    if args.accept_no_self_delete is not None and len(args.accept_no_self_delete.strip()) < 10:
        print("--accept-no-self-delete needs a real note (who decided, when)", file=sys.stderr)
        return 2
    if args.cmd == "cleanup":
        return cleanup(args)
    if not args.plan:
        print("--plan is required", file=sys.stderr)
        return 2
    plan = load_plan(Path(args.plan))
    if args.lifetime_hours * float(plan["max_hourly"]) > args.spend_cap:
        print("refusing: lifetime x the plan's max_hourly exceeds the spend cap", file=sys.stderr)
        return 2
    if args.cmd == "plan":
        return plan_cmd(args)
    if not args.confirm:
        print(
            "launch spends money: re-run with --confirm after the plan is approved", file=sys.stderr
        )
        return 2
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
