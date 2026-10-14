#!/usr/bin/env python3
"""GPU runner v2 (M3-C): many training runs on ONE RunPod GPU pod, money-safe, observable.

v2 = the M3-B runner (``research/redesign_m3_20261008/gpu_run.py``) with every money-safety
path kept (ledger reservation, balance-drop hard cap, pod lifetime, detached Mac watchdog,
pod-side job kill at the deadline + dead-man switch, delete-and-verify on every exit, explicit
``--accept-no-self-delete`` waiver, definite-refusal-only create retries, opt-in stock polling
and GPU fallbacks), plus:

* **Stock wait** (opt-in ``--stock-wait-hours``, at most 3 h): when every create try is a
  "no instances available" refusal ($0), another round follows every ``--stock-poll-s``
  (default 60 s, start to start); a throttled round (HTTP 429) backs off exponentially (up to
  ``STOCK_THROTTLE_MAX_S``) within the same bound. Each round is self-contained (no pod, a
  finished ledger run); its watchdog stays armed until the round's late-listing window
  (``CREATE_SETTLE_S``) has passed with a clean listing. A round is one row in
  ``<run-dir>/stock_wait.jsonl`` and one printed line; only the latest refused round's
  attempt dir (and those whose watchdog is still armed) is kept. Any other create error ends
  the wait at once (``StockWait``); a session deletes a stray late-listed pod of an earlier
  round (``sweep_strays``).

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

Opt-in plan keys (M6; a plan without them runs exactly as before, tested):

* ``"image"``: the pod's container image (default ``IMAGE``, the runtime image). torch.compile
  (``--compile-learner``) needs a C compiler, which only ``IMAGE_DEVEL`` has: the runtime
  image silently falls back to eager (research/perf_quiet_20261013 RESULTS).
* ``"extra_inputs"``: ``[{"src": local path, "dst": path under the pod's /r/in, "sha256"}]``
  uploaded with the sha-checked input upload (refused locally before anything is created if
  a file does not hash to its pin; the pod agent refuses a body that does not either).
* per-run ``"init"``: a pod-side path (``/r/in/<an extra_inputs dst>``) that REPLACES the
  default M2b-student ``--init`` in that run's argv (never a duplicate flag).

``plan_warnings`` flags sizing that looks inconsistent with the runs' flags (``--obs-pack``
vs ``ram_per_run_gb`` / ``ring_gb_per_run``, actors vs ``vcpu_per_run``, the initial wave's
VRAM, ``--compile-learner`` without the devel image). Every run's argv is checked against
the COMMITTED trainer's argparse: ``plan`` reports flags it does not define, ``launch``
refuses them before any RunPod call (a run would die in argparse on a paid pod).

Subcommands: ``plan`` (dry run: inputs, pod body, rp.py dry-run request; spends nothing),
``launch --confirm`` (spends money: only on the user's direct approval), ``cleanup --confirm``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
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
#: opt-in (plan ``"image"``): the same torch / CUDA with gcc, for torch.compile on the pod
IMAGE_DEVEL = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
IMAGE_RE = re.compile(r"[a-z0-9]+([._/-][a-z0-9]+)*:[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
POD_IN = "/r/in/"
STUDENT_DST = "ckpt/student_m2b_final.pth"
STUDENT_INIT = POD_IN + STUDENT_DST
INPUT_DST_RE = re.compile(r"[A-Za-z0-9_.\-]{1,80}(/[A-Za-z0-9_.\-]{1,80}){0,4}")
RESERVED_DSTS = ("repo.tar.gz", STUDENT_DST)
RESERVED_DST_PREFIXES = ("demo/",)
RUNNER_FLAGS = ("--out", "--device", "--init", "--demo-data", "--transitions", "--seed")
# Sanity bounds for plan sizing (warnings only), from research/perf_ram_20261013 (per run on
# the RTX 4090 pods: 16.2 GB default / 6.6 GB --obs-pack; bv8 12.4 / 5.5 GB; plans used 19
# GB unpacked) and the pod telemetry (2.0 GB VRAM per run with a CUDA learner + 2 CUDA actors).
OBS_PACK_RAM_GB = 6.0
NO_PACK_RAM_GB = 16.0
UNPACKED_RING_GB = 9.0  # rings per run: 9.3-9.5 GB raw, 2.4-2.6 GB --obs-pack (bench_ram)
GPU_VRAM_GB = 24.0  # the RTX 4090, in nvidia-smi GiB (the fill's unit)
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
PREFLIGHT_BUDGET_S = 1200  # opt-in trainer preflight (plan "preflight_transitions")
FAILED_LOG_TAIL = 4000  # bytes of a failed run's stdout+stderr kept locally
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
#: request-rate refusal text (with HTTP 429 itself: ``is_throttled``); the stock wait backs off
THROTTLED = re.compile(r"too many requests|rate[ _-]?limit|throttl", re.I)
GPU_MAX_HOURLY = 0.80
#: opt-in stock wait (``--stock-wait-hours``): one round of create tries every STOCK_POLL_S
#: (start to start; ``--stock-poll-s``) for up to STOCK_WAIT_S. A refused create costs $0.
STOCK_POLL_S = 60
STOCK_POLL_MIN_S = 30
STOCK_POLL_MAX_S = 3600
STOCK_WAIT_S = 3 * 3600
#: a throttled stock-wait round (HTTP 429) is followed min(STOCK_THROTTLE_MAX_S, poll x 2^n)
#: later (start to start; n = consecutive throttled rounds), never sooner than the poll
STOCK_THROTTLE_MAX_S = 300
#: create-try kinds the stock wait retries (no pod was made): "no instances available", a
#: request-rate refusal, a failed pod listing between two refused tries (the round stops there)
STOCK_RETRYABLE = ("no_capacity", "throttled", "transient")
CREATE_SETTLE_S = 300
STRAY_SWEEP_S = 300  # a session deletes a late-listed pod of an earlier refused round this often
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
            flag = str(a).split("=", 1)[0]  # "--init=x" sets --init too
            if flag in RUNNER_FLAGS:
                p.append(f"{r.get('name')}: extra may not set {flag}")
    p += _validate_opt_in(plan, runs)
    for k in ("transitions", "mac_rate", "g2_factor", "ram_per_run_gb", "vcpu_per_run"):
        if not isinstance(plan.get(k), (int, float)) or plan[k] <= 0:
            p.append(k)
    if plan.get("g2_factor", 0) < 4:
        p.append("g2_factor < 4 (the per-step 4x rule)")
    mh = plan.get("max_hourly")
    gpus = plan.get("gpu_count", 1)  # RAM/vCPU minimums are pod totals (pod_body)
    if isinstance(gpus, bool) or gpus not in (1, 2):
        p.append("gpu_count must be 1 or 2")
        gpus = 1
    if not isinstance(mh, (int, float)) or not 0 < mh <= MAX_HOURLY * max(1, gpus):
        p.append(f"max_hourly must be in (0, {MAX_HOURLY} x gpu_count]")
    if int(plan.get("max_concurrent_cap", 0)) < 1:
        p.append("max_concurrent_cap")
    return p


def _validate_opt_in(plan: Dict[str, Any], runs: List[Dict[str, Any]]) -> List[str]:
    """Problems with the opt-in keys ``image``, ``extra_inputs`` and per-run ``init``
    (``gpu_count`` is checked by ``validate_plan``)."""
    p: List[str] = []
    if "image" in plan:
        image = plan["image"]
        if not isinstance(image, str) or not IMAGE_RE.fullmatch(image):
            p.append(f"image must be a tagged image reference (e.g. {IMAGE_DEVEL}), got {image!r}")
    xs = plan.get("extra_inputs", [])
    dsts: List[str] = []
    if not isinstance(xs, list):
        p.append("extra_inputs must be a list")
        xs = []
    for i, x in enumerate(xs):
        if (
            not isinstance(x, dict)
            or set(x) != {"src", "dst", "sha256"}
            or not all(isinstance(x[k], str) for k in x)
        ):
            p.append(f"extra_inputs[{i}]: needs exactly the strings src, dst, sha256")
            continue
        dst = x["dst"]
        parts = dst.split("/")
        if not INPUT_DST_RE.fullmatch(dst) or any(c in (".", "..") for c in parts):
            p.append(f"extra_inputs[{i}]: dst {dst!r} must be a relative path under {POD_IN}")
        elif dst in RESERVED_DSTS or dst.startswith(RESERVED_DST_PREFIXES):
            p.append(f"extra_inputs[{i}]: dst {dst!r} is one of the runner's own inputs")
        if dst in dsts:
            p.append(f"extra_inputs[{i}]: duplicate dst {dst!r}")
        dsts.append(dst)
        if not re.fullmatch(r"[0-9a-f]{64}", x["sha256"]):
            p.append(f"extra_inputs[{i}]: sha256 must be 64 lowercase hex digits")
    inits = {STUDENT_INIT} | {POD_IN + d for d in dsts}
    for r in runs:
        if "init" in r and r["init"] not in inits:
            p.append(
                f"{r.get('name')}: init {r['init']!r} is not an uploaded input "
                f"({POD_IN}<an extra_inputs dst> or {STUDENT_INIT})"
            )
    return p


def run_flags(run: Dict[str, Any]) -> List[str]:
    return [str(a) for a in run.get("extra", [])]


def flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    """The value argparse would give ``flag`` (the last occurrence wins); None if absent."""
    out = None
    for i, tok in enumerate(argv):
        if tok == flag:
            out = argv[i + 1] if i + 1 < len(argv) else ""
        elif tok.startswith(flag + "="):
            out = tok.split("=", 1)[1]
    return out


def initial_wave(plan: Dict[str, Any]) -> int:
    """The runs started at once on the smallest host the plan admits (min_ram_gb, min_vcpu)."""
    mem, vcpu = float(plan.get("min_ram_gb", 96)), float(plan.get("min_vcpu", 16))
    return initial_concurrency(plan, mem, vcpu, vcpu)


def plan_warnings(plan: Dict[str, Any]) -> List[str]:
    """Sizing that looks inconsistent with the runs' flags (printed, never refused).

    G2 and the sizing use the plan's own ``ram_per_run_gb`` / ``vcpu_per_run``; these checks
    compare them with what was measured for the flags the runs carry.
    """
    w: List[str] = []
    runs = plan["runs"]
    packed = [r["name"] for r in runs if "--obs-pack" in run_flags(r)]
    unpacked = [r["name"] for r in runs if "--obs-pack" not in run_flags(r)]
    ram = float(plan["ram_per_run_gb"])
    ring = float(plan.get("ring_gb_per_run", 9.3))
    if packed and unpacked:
        w.append(
            f"--obs-pack in {len(packed)} of {len(runs)} runs: ram_per_run_gb ({ram:g}) and "
            "ring_gb_per_run are one number for every run; size them for the unpacked runs"
        )
    if unpacked and ram < NO_PACK_RAM_GB:
        w.append(
            f"ram_per_run_gb {ram:g} < {NO_PACK_RAM_GB:g}: runs without --obs-pack measured "
            "~16-19 GB each on the pods (perf-ram): the initial wave may run out of memory"
        )
    if unpacked and ring < UNPACKED_RING_GB:
        w.append(
            f"ring_gb_per_run {ring:g} < {UNPACKED_RING_GB:g}: unpacked rings are ~9.3 GB per "
            "run (the adaptive fill reserves this for young runs)"
        )
    if packed and not unpacked:
        if ram < OBS_PACK_RAM_GB:
            w.append(
                f"ram_per_run_gb {ram:g} < {OBS_PACK_RAM_GB:g}: --obs-pack runs measured "
                "~6-7 GB each on the pods (perf-ram): the initial wave may run out of memory"
            )
        elif ram >= NO_PACK_RAM_GB:
            w.append(
                f"ram_per_run_gb {ram:g} is the unpacked size, but every run has --obs-pack "
                "(~6-7 GB per run): the RAM bound starts fewer runs than fit"
            )
        if ring >= UNPACKED_RING_GB:
            w.append(
                f"ring_gb_per_run {ring:g} (default 9.3) is the unpacked ring size; --obs-pack "
                "rings are ~2.4-2.6 GB per run: the adaptive fill under-fills"
            )
    actors = int(plan.get("actors", 0))
    if actors and float(plan["vcpu_per_run"]) < actors + 1:
        w.append(
            f"vcpu_per_run {plan['vcpu_per_run']} < actors + 1 = {actors + 1}: the learner and "
            "each actor keep about one core busy"
        )
    wave = initial_wave(plan)
    if wave < 1:
        w.append("no run fits the smallest admitted host (min_ram_gb / min_vcpu): launch fails")
    vram = float(plan.get("vram_per_run_gb", 2.0))
    if wave * vram > 0.9 * GPU_VRAM_GB:
        w.append(
            f"initial wave {wave} runs x vram_per_run_gb {vram:g} = {wave * vram:.1f} GB > 90% "
            f"of the RTX 4090's {GPU_VRAM_GB:g} GB: the initial wave is not VRAM-checked (only "
            "the adaptive fill is); lower max_concurrent_cap if a run can OOM"
        )
    compiled = [
        r["name"]
        for r in runs
        if any(
            flag_value(run_flags(r), f) not in (None, "off")
            for f in ("--compile-learner", "--compile-actor")
        )
    ]
    if compiled and plan.get("image", IMAGE) == IMAGE:
        w.append(
            f"{len(compiled)} runs use torch.compile but the image is the runtime image (no C "
            f'compiler: compile falls back to eager); set "image": "{IMAGE_DEVEL}"'
        )
    return w


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
    image: str = IMAGE,
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """``min_ram_gb`` / ``min_vcpu`` are pod totals (split per GPU for the request)."""
    gpu_count = max(1, int(gpu_count))
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
        "gpuCount": gpu_count,
        "cloudType": "SECURE",
        "imageName": image,
        "containerDiskInGb": 120,  # code, demos (2.2 GB), file-backed rings (~7.6 GB per run)
        "volumeInGb": 0,
        "minRAMPerGPU": -(-int(min_ram_gb) // gpu_count),
        "minVCPUPerGPU": -(-int(min_vcpu) // gpu_count),
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
    files.append({"src": str(STUDENT), "dst": STUDENT_DST, "sha256": STUDENT_SHA})
    taken = {f["dst"] for f in files}
    extra = extra_input_files(plan)
    clash = sorted(taken & {f["dst"] for f in extra})
    if clash:
        raise SystemExit(f"refusing: extra_inputs dst {clash} would replace the runner's inputs")
    files += extra
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
        "trainer_flag_problems": trainer_flag_problems(plan, committed_trainer_flags(commit)),
    }


def extra_input_files(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The plan's opt-in ``extra_inputs`` as upload rows, each re-hashed against its pin."""
    out = []
    for x in plan.get("extra_inputs", []):
        src = Path(x["src"])
        try:
            got = sha_file(src)
        except OSError as exc:
            raise SystemExit(f"refusing: extra input {src} is unreadable: {exc}") from exc
        if got != x["sha256"]:
            raise SystemExit(f"refusing: extra input {src} hashes to {got}, not {x['sha256']}")
        out.append({"src": str(src), "dst": x["dst"], "sha256": x["sha256"]})
    return out


def argparse_flags(source: str) -> set:
    """Every option string a script's ``add_argument`` calls define (plus -h / --help)."""
    flags = {"-h", "--help"}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    flags.add(str(arg.value))
    return flags


def committed_trainer_flags(commit: str) -> Optional[set]:
    """The flags of the trainer AS COMMITTED (what the pod runs); None if it is missing."""
    done = subprocess.run(
        ["git", "show", f"{commit}:{TRAINER}"], cwd=REPO, capture_output=True, text=True
    )
    return argparse_flags(done.stdout) if done.returncode == 0 else None


def trainer_flag_problems(plan: Dict[str, Any], flags: Optional[set]) -> List[str]:
    """Option strings in the runs' argv that the committed trainer does not define."""
    if flags is None:
        return [f"{TRAINER} is not in the committed tree"]
    bad: Dict[str, List[str]] = {}
    for run in plan["runs"]:
        for tok in train_argv(plan, run)[2:]:
            flag = tok.split("=", 1)[0]
            if tok.startswith("--") and flag not in flags:  # values never start with "--"
                bad.setdefault(flag, []).append(run["name"])
    return [
        f"{flag} ({len(names)} runs, e.g. {names[0]}) is not defined by the committed {TRAINER}"
        for flag, names in sorted(bad.items())
    ]


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
    """The run's trainer argv. A per-run ``init`` (opt-in) replaces the student path; the
    plan's ``extra`` flags are appended verbatim (validation keeps the runner's own out)."""
    argv = [
        "python",
        TRAINER,
        "--seed",
        str(int(run["seed"])),
        "--init",
        str(run.get("init", STUDENT_INIT)),
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
def delete_and_verify(
    rp,
    run_dir: Path,
    state: Dict[str, Any],
    reason: str,
    settle_s: float = CREATE_SETTLE_S,
    clean_s: float = 60.0,
    quiet: bool = False,
) -> bool:
    """Delete our pod(s) with backoff until GET /pods shows none (pod self-delete is not
    available, so this never gives up early: alerts every 30 min, stops trying after 24 h).

    Verified = no pod of ours listed for ``clean_s`` and ``settle_s`` since the last create
    try (a fresh pod can be missing from GET /pods for minutes). Only the stock wait passes
    ``settle_s=0, clean_s=0`` (one clean listing), for a round whose every create try was a
    capacity refusal; its loop then owns the late-listing settle (``StockWait``)."""
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
            settled = now() - float(state.get("create_attempted_at") or 0) >= settle_s
            if left:
                clean_since = None
            elif clean_since is None:
                clean_since = now()
            if not left and settled and now() - clean_since >= clean_s:
                state.update(deleted=now(), delete_reason=reason, pods_left=[])
                save_state(run_dir, state)
                if not quiet or all_ids:
                    out = {"deleted": sorted(all_ids), "reason": reason, "our_pods_left": []}
                    say(json.dumps(out))
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
    # its process group only: it guards no pod yet (spawned before the create), and a stop
    # file would also stop an armed watchdog sharing this dir (pod_bench's stock-wait rounds)
    stop_watchdog(proc.pid)
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


def _http(exc: BaseException) -> Tuple[Optional[int], str]:
    """(HTTP status, error body text) of an rp.py error; (None, "") when RunPod gave none."""
    payload = getattr(exc, "payload", None)
    status = payload.get("http_status") if isinstance(payload, dict) else None
    if not isinstance(status, int):
        return None, ""
    body = payload.get("error")
    return status, json.dumps(body).lower() if isinstance(body, (dict, list, str)) else ""


def definite_refusal(exc: BaseException) -> bool:
    """True when RunPod answered with an HTTP error (the pod was NOT created)."""
    status, text = _http(exc)
    if status is None:
        return False
    if 400 <= status < 500:
        return True
    return 500 <= status < 600 and bool(NO_CAPACITY.search(text))


def is_throttled(exc: BaseException) -> bool:
    """A request-rate refusal: HTTP 429, or a 4xx/5xx body saying so. For a create it counts
    only as a definite refusal (``refusal_kind``: a 5xx stays an unknown outcome)."""
    status, text = _http(exc)
    if status is None:
        return False
    return status == 429 or (400 <= status < 600 and bool(THROTTLED.search(text)))


def refusal_kind(exc: BaseException) -> str:
    """A failed create try: "no_capacity" / "throttled" (the stock wait retries both),
    "refused" (any other definite refusal: a bad request, auth, not found ...) or "unknown"
    (no definite answer, e.g. a 5xx: the pod may exist; adopt it by name, never re-POST)."""
    if not definite_refusal(exc):
        return "unknown"
    if NO_CAPACITY.search(error_detail(exc)):
        return "no_capacity"
    return "throttled" if is_throttled(exc) else "refused"


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
    stock_wait: bool = False,
):
    """One round of create tries (``create_plans``: at most one per stocked DC, 20 s apart).
    Returns (create answer, last error, every try was a definite refusal). Each failed try is
    recorded in ``create_attempts`` with its ``kind`` (``refusal_kind``). ``stock_wait``: no
    line per capacity refusal or throttle (the stock wait prints one line per round), and a
    throttle or a failed pod listing between two tries ends the round (the next round, backed
    off, tries again; more POSTs would only be throttled too)."""
    attempts = state.setdefault("create_attempts", [])
    err = ""
    for i, (gpu, plan_dcs) in enumerate(create_plans(rp, gpus)):
        if i:
            time.sleep(20)
            try:
                hits = [p for p in our_pods(rp) if p.get("name") == body["name"]]
            except Exception as exc:  # noqa: BLE001
                if not stock_wait:
                    raise
                # every earlier try of this round was a definite refusal: no pod to adopt
                kind = "throttled" if is_throttled(exc) else "transient"
                err = f"pod listing before try {i} failed: {error_detail(exc)}"[:1500]
                row = {"try": i, "gpu": gpu, "dcs": plan_dcs, "error": err, "kind": kind}
                attempts.append(dict(row, skipped=True))  # no create was sent
                save_state(run_dir, state)
                return None, err, True
            if hits:
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
            kind = refusal_kind(exc)
            attempts.append({"try": i, "gpu": gpu, "dcs": plan_dcs, "error": err, "kind": kind})
            save_state(run_dir, state)
            if not (stock_wait and kind in STOCK_RETRYABLE):
                say(f"create try {i} ({gpu}, {plan_dcs or 'any DC'}): {err}")
            if kind == "unknown":
                return None, err, False  # unknown outcome: adopt-or-give-up, never re-POST
            if stock_wait and kind == "throttled":
                return None, err, True  # no more POSTs this round: the stock wait backs off
    return None, err, True


# ----------------------------------------------------------------------------- stock wait
def stock_poll_problem(poll_s: Any) -> Optional[str]:
    """Why ``--stock-poll-s`` is unusable (None: it is fine)."""
    try:
        s = float(poll_s)
    except (TypeError, ValueError):
        return "--stock-poll-s must be a number of seconds"
    if not STOCK_POLL_MIN_S <= s <= STOCK_POLL_MAX_S:  # NaN fails too
        return f"--stock-poll-s must be in [{STOCK_POLL_MIN_S}, {STOCK_POLL_MAX_S}] seconds"
    return None


def try_kind(t: Dict[str, Any]) -> str:
    """A recorded create try's kind: "ok" or its ``refusal_kind`` (a row saved before kinds
    were recorded is classified by its error text)."""
    if t.get("ok"):
        return "ok"
    if t.get("kind"):
        return str(t["kind"])
    return "no_capacity" if NO_CAPACITY.search(str(t.get("error") or "")) else "refused"


def stock_refusals_only(state: Dict[str, Any]) -> bool:
    """Every create try of a round was a refusal the stock wait retries (``STOCK_RETRYABLE``;
    no pod was created), not a bad request, an auth error or an unknown outcome."""
    tries = state.get("create_attempts") or []
    return bool(tries) and all(try_kind(t) in STOCK_RETRYABLE for t in tries)


def throttled_round(state: Dict[str, Any]) -> bool:
    return any(try_kind(t) == "throttled" for t in state.get("create_attempts") or [])


def tries_text(tries: Sequence[Dict[str, Any]]) -> str:
    """A round's create tries, compact: ``GPU: any DC, EU-RO-1; GPU2: EU-SE-1``."""
    by: Dict[str, List[str]] = {}
    for t in tries:
        by.setdefault(str(t.get("gpu")), []).append(", ".join(t.get("dcs") or []) or "any DC")
    return "; ".join(f"{g}: {', '.join(d)}" for g, d in by.items()) or "no create try"


def short_error(err: str) -> str:
    m = NO_CAPACITY.search(err or "")
    return f"no capacity ({m.group(0)})" if m else (err or "")[:200]


def read_state(run_dir: Path) -> Dict[str, Any]:
    """A round's saved state ({} when the round wrote none, e.g. a transient before create)."""
    try:
        return json.loads(state_path(run_dir).read_text())
    except (OSError, ValueError):
        return {}


class StockWait:
    """Opt-in capacity polling for scarce GPUs (``--stock-wait-hours``, ``--stock-poll-s``).

    A round is one self-contained attempt (own pod name, watchdog, ledger run, create tries,
    delete-and-verify). Rounds start every ``poll_s`` (start to start; a longer round is
    followed at once) for up to ``wait_s``, while every create try is a refusal the wait
    retries (``STOCK_RETRYABLE``); any other create error (a bad request, auth, not found
    ...: fast), an unknown create outcome (adopt-by-name, as before), an interrupted round or
    a created pod ends the wait. A throttled round (HTTP 429) backs off: the next round
    starts min(STOCK_THROTTLE_MAX_S, poll_s x 2^n) after it (n = consecutive throttled
    rounds), still within ``wait_s``. Nothing is reserved between rounds: a refused round
    finishes its ledger run.

    Late-listing settle: a refused round's delete-and-verify does not wait CREATE_SETTLE_S
    (that alone would space the rounds >= 5 min apart); one clean listing ends it. The settle
    is owned here instead, and the round's detached watchdog stays armed as its backstop
    (``pending``) until its own window has passed with a clean listing: ``reap`` (between
    rounds: CREATE_SETTLE_S after that round's create, no pod of its name listed), a later
    round's full settle (a clean listing of our whole prefix CREATE_SETTLE_S after a later
    create covers every earlier window), or ``settle`` (on every other exit: give-up,
    interrupt, error). A watchdog whose window never verified clean stays armed (a pod may
    exist; it deletes that pod at its lifetime + 5 min). Every round's pre-create check
    lists our pods (one found stops the wait; ``settle`` deletes it), and a session sweeps
    our prefix every STRAY_SWEEP_S (``sweep_strays``) for a pod of an earlier refused round
    that is listed only after the session's create.

    Each round appends one row to ``log`` and prints one compact line. ``stop_files``: stop a
    watchdog with its ``watchdog.stop`` file too (each round has its own dir); pod_bench's
    rounds share one dir, so they are stopped by process group only.
    """

    def __init__(self, wait_s: float, poll_s: float, log: Path, stop_files: bool = True) -> None:
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self.log = log
        self.stop_files = stop_files
        self.t0 = now()
        self.deferred_at: Optional[float] = None  # the last refused create we owe a settle
        self.deferred_dir: Optional[Path] = None  # where that round's state.json lives
        #: refused rounds whose watchdog is still armed: {"name", "pid", "at", "dir"}
        self.pending: List[Dict[str, Any]] = []
        #: watchdogs left armed for good (a pod may exist): their files are kept too
        self.left_armed: List[Dict[str, Any]] = []
        self.throttled = 0  # consecutive throttled rounds (the backoff exponent)
        self.gap_s = self.poll_s  # start to start, the latest round -> the next

    @property
    def active(self) -> bool:
        return self.wait_s > 0

    def holds(self, path: Path) -> bool:
        """``path`` holds a refused round whose watchdog is still armed (keep its files)."""
        return self.owner(path) is not None

    def owner(self, path: Path) -> Optional[Dict[str, Any]]:
        return next((e for e in self.pending + self.left_armed if e["dir"] == path), None)

    def moved(self, src: Path, dst: Path) -> None:
        """A round's files moved from ``src`` to ``dst`` (pod_bench sets refused rounds
        aside): its settle record and watchdog bookkeeping follow them."""
        for e in self.pending + self.left_armed:
            if e["dir"] == src:
                e["dir"] = dst
        if self.deferred_dir == src:
            self.deferred_dir = dst

    def _note(self, e: Dict[str, Any], **kw: Any) -> None:
        """Record a watchdog decision in its round's state.json (when it is still there)."""
        st = read_state(e["dir"])
        if st and st.get("name") == e["name"]:
            st.update(kw)
            save_state(e["dir"], st)

    def _stop(self, entries: Sequence[Dict[str, Any]], why: str) -> None:
        for e in list(entries):
            try:
                stop_watchdog(e["pid"], e["dir"] if self.stop_files else None)
            except OSError:  # its dir is gone: the process group alone
                stop_watchdog(e["pid"])
            self._note(e, watchdog_stopped=True, watchdog_stop={"why": why, "t": now()})
            self.pending.remove(e)

    def _leave_armed(self, why: str) -> None:
        """A pod may exist: every armed watchdog stays armed (it deletes the pod of its name
        at its lifetime + 5 min); the loop owes no further settle."""
        for e in self.pending:
            self._note(e, watchdog_left_armed=why)
        if self.pending:
            names = [e["name"] for e in self.pending]
            alert(f"watchdog(s) left armed for {names} ({why}); run `cleanup --confirm`")
        self.left_armed += self.pending
        self.pending = []
        self.deferred_at = self.deferred_dir = None

    def track(self, run_dir: Path, st: Dict[str, Any]) -> None:
        """Note what a finished round left to this loop: a deferred settle and an armed
        watchdog (a refused round), or a full settle that covers every earlier round."""
        if st.get("settle_deferred"):
            self.deferred_at = float(st["create_attempted_at"])
            self.deferred_dir = run_dir
            if st.get("watchdog_pid") and not st.get("watchdog_stopped"):
                entry = {"name": st.get("name"), "pid": int(st["watchdog_pid"]), "dir": run_dir}
                self.pending.append(dict(entry, at=float(st["create_attempted_at"])))
            if not st.get("pods_deleted_verified"):
                self._leave_armed("a refused round's pod listing never came back clean")
        elif st.get("create_attempted_at"):  # a later create try, settled in full
            if st.get("pods_deleted_verified"):
                self._stop(self.pending, "covered by a later round's full late-listing settle")
            else:
                self._leave_armed("a later round's delete-and-verify did not verify")
            self.deferred_at = self.deferred_dir = None

    def reap(self, rp) -> List[Dict[str, Any]]:
        """Between rounds: stop each armed watchdog whose round's window (CREATE_SETTLE_S
        after its create) has passed, once a listing shows no pod of its name. A failed
        listing or a listed pod keeps it armed (the pre-create check then stops the wait on
        that pod and ``settle`` deletes it)."""
        due = [e for e in self.pending if now() - e["at"] >= CREATE_SETTLE_S]
        if not due:
            return []
        try:
            listed = {str(p.get("name")) for p in our_pods(rp)}
        except Exception:  # noqa: BLE001  (armed until a later round's listing)
            return []
        done = [e for e in due if e["name"] not in listed]
        self._stop(done, f"clean listing >= {CREATE_SETTLE_S} s after its create")
        return done

    def sleep_s(self, start: float) -> float:
        return max(0.0, self.gap_s - (now() - start))

    def another_round(
        self,
        k: int,
        label: str,
        start: float,
        st: Dict[str, Any],
        transient: Optional[str] = None,
        throttled: bool = False,
    ) -> bool:
        """After a round that created nothing: poll again? Logs the round when active.
        ``transient`` (a read failed before the create; ``throttled``: a request-rate one)."""
        failed = str(st.get("outcome", "")).startswith("error:")
        retry = transient is not None or (stock_refusals_only(st) and not failed)
        throttle = throttled if transient is not None else throttled_round(st)
        self.throttled = self.throttled + 1 if throttle else 0
        self.gap_s = self.poll_s
        if self.throttled:  # exponential backoff, capped (never sooner than the poll)
            backoff = min(STOCK_THROTTLE_MAX_S, self.poll_s * 2.0**self.throttled)
            self.gap_s = max(self.poll_s, backoff)
        more = now() - self.t0 + self.sleep_s(start) <= self.wait_s
        if self.active:
            attempts = st.get("create_attempts") or []
            sent = [a for a in attempts if not a.get("skipped")]  # POSTs (not a failed listing)
            tries = [{"gpu": a.get("gpu"), "dcs": a.get("dcs")} for a in sent]
            last = attempts[-1] if attempts else {}
            if transient is not None:
                err = f"transient before create: {transient}"[:200]
            elif failed or not last.get("error"):
                err = str(st.get("outcome"))[:200]  # an interrupted / failed round
            elif try_kind(last) == "no_capacity":
                err = short_error(str(last["error"]))
            else:
                err = f"{try_kind(last)}: {last['error']}"[:200]
            row = {
                "round": k,
                "attempt": label,
                "t": now(),
                "elapsed_s": round(now() - self.t0, 1),
                "round_s": round(now() - start, 1),
                "tries": tries,
                "error": err,
                "throttled": throttle,
                "run_id": st.get("run_id"),
                "cost_usd": st.get("cost_estimate"),
                "pods_deleted_verified": st.get("pods_deleted_verified"),
                "settle_deferred": bool(st.get("settle_deferred")),
                "watchdogs_armed": len(self.pending),
            }
            append_jsonl(self.log, row)
            nxt = "stop"
            if retry and more:
                nxt = f"next round in {self.sleep_s(start):.0f} s"
                nxt += " (throttled: backing off)" if throttle else ""
            say(
                f"stock wait round {k} ({label}): {err} [{tries_text(tries)}]; "
                f"{(now() - self.t0) / 3600:.2f} h of {self.wait_s / 3600:.2f} h; {nxt}"
            )
            if not retry:
                alert(
                    f"{label}: {err}; not polling again (only capacity refusals and "
                    "throttles are retried)"
                )
                return False
        if not more:
            hours = (now() - self.t0) / 3600
            alert(f"no capacity after {hours:.2f} h; giving up at $0 (Mac fallback)")
            return False
        return True

    def settle(self, rp, run_dir: Optional[Path] = None) -> Optional[bool]:
        """Before the loop exits: wait out CREATE_SETTLE_S after the last refused create whose
        settle a round deferred, deleting any pod of ours listed meanwhile, then stop every
        watchdog still armed (None: nothing was deferred). A settle that does not verify
        leaves them armed (a pod may exist). The result lands in the latest deferred round's
        state.json (``late_listing_settle``)."""
        if self.deferred_at is None and not self.pending:
            return None
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the delete
        need = max([self.deferred_at or 0.0] + [e["at"] for e in self.pending])
        where = run_dir or self.deferred_dir or self.pending[-1]["dir"]
        where.mkdir(parents=True, exist_ok=True)
        st = read_state(where)
        st["create_attempted_at"] = max(float(st.get("create_attempted_at") or 0), need)
        left = CREATE_SETTLE_S - (now() - need)
        if left > 0:
            say(f"stock wait: watching {left:.0f} s for a late-listed pod (last refused create)")
        ok = delete_and_verify(
            rp,
            where,
            st,
            "stock wait: late-listing settle after the last refused create",
            quiet=True,
        )
        armed = [e["name"] for e in self.pending]
        st["late_listing_settle"] = {"pods_deleted_verified": ok, "t": now(), "watchdogs": armed}
        st["pods_deleted_verified"] = bool(st.get("pods_deleted_verified", True)) and ok
        save_state(where, st)
        self.deferred_at = self.deferred_dir = None
        if ok:
            self._stop(self.pending, "late-listing settle verified")
        else:
            self._leave_armed("the late-listing settle did not verify")
            alert(f"late-listing settle failed; run `cleanup --confirm` ({where})")
        return ok


def stray_pod(state: Dict[str, Any], pod: Dict[str, Any]) -> bool:
    """A pod a session deletes: named exactly like one of this runner's rounds (our prefix +
    the 10-digit timestamp; never another prefix, even one that extends ours) but not the
    session's own pod (by name and id)."""
    name, own = str(pod.get("name") or ""), state.get("name")
    if not own or not pod.get("id") or not re.fullmatch(re.escape(POD_NAME) + r"\d{10}", name):
        return False
    return name != own and pod.get("id") != state.get("pod_id")


def sweep_strays(rp, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """During a session (``guard``), every STRAY_SWEEP_S: delete any pod of ours that is not
    the session's own (a pod an earlier stock-wait round's refused create made after all,
    listed only after this session's create) and record it in ``state["strays"]``. Best
    effort: a failed listing or delete is retried at the next sweep, and the session's
    delete-and-verify sweeps our whole prefix at its end anyway."""
    if not state.get("name") or now() - float(state.get("stray_sweep_at") or 0) < STRAY_SWEEP_S:
        return []
    state["stray_sweep_at"] = now()
    try:
        pods = [p for p in rp.list_pods() if stray_pod(state, p)]
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for p in pods:
        row = {"id": p["id"], "name": p.get("name"), "t": now(), "deleted": False}
        try:
            rp.delete_pod(p["id"], confirm=True)
            row["deleted"] = True
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        state.setdefault("strays", []).append(row)
        verb = "deleted" if row["deleted"] else "could not delete (retried next sweep)"
        alert(f"stray pod of ours {row['name']} ({row['id']}), not this session's: {verb}")
        rows.append(row)
    return rows


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
    """A read failed before anything was reserved or created (safe to retry next poll);
    ``throttled``: a request-rate refusal (the stock wait backs off)."""

    def __init__(self, msg: str, throttled: bool = False) -> None:
        super().__init__(msg)
        self.throttled = throttled

    @classmethod
    def of(cls, exc: BaseException) -> "PreCreateTransient":
        return cls(repr(exc), throttled=is_throttled(exc))


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)
    signal.signal(signal.SIGINT, signal.default_int_handler)


def launch(args) -> int:
    from research.runpod_fanout.rp_client import RpClient
    from research.runpod_fanout.tls import preflight

    poll_s = getattr(args, "stock_poll_s", None)
    poll_s = STOCK_POLL_S if poll_s is None else poll_s
    problem = stock_poll_problem(poll_s)
    if problem:
        raise SystemExit(problem)
    if sys.platform == "darwin":  # keep the Mac awake while this process lives
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], start_new_session=True)
    plan = load_plan(Path(args.plan))
    root = Path(args.run_dir)
    if root.exists():
        print(f"refusing: {root} exists", file=sys.stderr)
        return 2
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({k: v for k, v in plan.items() if k != "_sha256"}))
    for w in plan_warnings(plan):
        say(f"plan warning: {w}", err=True)
    problems = preflight()
    if problems:
        raise SystemExit(f"TLS preflight failed: {problems}")
    inputs = build_inputs(root / "inputs", plan)
    if inputs.get("trainer_flag_problems"):  # before any RunPod call: nothing to clean up
        raise SystemExit(f"refusing: {inputs['trainer_flag_problems']}")
    rp = RpClient()
    wait_s = min(STOCK_WAIT_S, 3600.0 * float(getattr(args, "stock_wait_hours", 0.0) or 0.0))
    wait = StockWait(wait_s, float(poll_s), root / "stock_wait.jsonl")
    k = 0
    refused: List[Path] = []  # refused rounds' attempt dirs (pruned: see _prune_rounds)
    try:
        while True:
            start = now()
            if k:  # stop the watchdogs whose late-listing window has passed clean
                wait.reap(rp)
                _prune_rounds(refused, wait)
            sub = root / f"attempt{k:02d}"
            sub.mkdir()
            (root / "CURRENT_ATTEMPT").write_text(sub.name)
            transient, throttled = None, False
            try:
                code, no_capacity = launch_attempt(
                    args, rp, sub, inputs, plan, defer_settle=wait.active
                )
            except PreCreateTransient as exc:
                say(f"{sub.name}: transient before create: {exc}")
                code, no_capacity, transient, throttled = 1, True, str(exc), exc.throttled
            st = read_state(sub)
            wait.track(sub, st)
            if not no_capacity:
                return code
            if not wait.another_round(k, sub.name, start, st, transient, throttled):
                return code
            # bounded attempt dirs: a refused round is one row in stock_wait.jsonl (a
            # transient round's dir is empty)
            if not st:
                shutil.rmtree(sub, ignore_errors=True)
            else:
                refused.append(sub)
            _prune_rounds(refused, wait)
            k += 1
            _install_signal_handlers()
            time.sleep(wait.sleep_s(start))
    finally:
        wait.settle(rp)
        _prune_rounds(refused, wait)


def _prune_rounds(refused: List[Path], wait: StockWait) -> None:
    """Keep the latest refused round's attempt dir, the one a deferred settle writes to and
    every dir whose watchdog is still armed (its stop file and log live there)."""
    for d in list(refused[:-1]):
        if d != wait.deferred_dir and not wait.holds(d):
            shutil.rmtree(d, ignore_errors=True)
            refused.remove(d)


def launch_attempt(
    args,
    rp,
    run_dir: Path,
    inputs: Dict[str, Any],
    plan: Dict[str, Any],
    defer_settle: bool = False,
):
    """One self-contained attempt. Returns (exit code, it created nothing and every create try
    was a definite refusal). ``defer_settle`` (the stock wait): a round whose every try was a
    refusal the wait retries skips the CREATE_SETTLE_S late-listing settle and leaves its
    watchdog armed (``StockWait`` owns both)."""
    from research.runpod_fanout.ledger import SharedLedger

    _install_signal_handlers()
    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    try:
        mine = our_pods(rp)
        balance0 = rp.balance()
    except Exception as exc:  # noqa: BLE001
        raise PreCreateTransient.of(exc) from exc
    if mine:
        raise SystemExit("refusing: a pod of ours already exists")
    token = secrets.token_urlsafe(32)
    created = now()
    until = created + args.lifetime_hours * 3600
    deadline = until - 15 * 60
    name = POD_NAME + time.strftime("%m%d%H%M%S", time.localtime(created))
    body = pod_body(
        name,
        hashlib.sha256(token.encode()).hexdigest(),
        deadline,
        until,
        int(plan.get("min_ram_gb", 96)),
        int(plan.get("min_vcpu", 16)),
        image=plan.get("image", IMAGE),
        gpu_count=int(plan.get("gpu_count", 1)),
    )
    (run_dir / "pod_body.json").write_text(json.dumps(body))
    watchdog_pid = spawn_watchdog(run_dir, until, name)
    ledger = SharedLedger(policy)
    run_id = f"m3c-gpu-{int(created)}"
    try:
        ledger.register_run(
            run_id,
            plan.get("job_id", "redesign-m3c-gpu"),
            str(run_dir),
            args.spend_cap,
            os.getpid(),
        )
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
        "image": body["imageName"],
    }
    save_state(run_dir, state)
    outcome, rate, no_capacity = "unknown", float(plan["max_hourly"]), False
    ok = False
    try:
        gpus = GPU_FALLBACKS if getattr(args, "gpu_fallbacks", False) else (GPU,)
        state["gpus_allowed"] = list(gpus)
        out, err, no_capacity = create_pod_retrying(
            rp, body, run_dir, state, gpus, float(plan["max_hourly"]), stock_wait=defer_settle
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
        quick = {}
        if defer_settle and no_capacity and not state.get("pod_id"):
            if stock_refusals_only(state):  # no pod was made: one clean listing for now
                quick = {"settle_s": 0.0, "clean_s": 0.0, "quiet": True}
                state["settle_deferred"] = True
        save_state(run_dir, state)
        ok = delete_and_verify(rp, run_dir, state, outcome, **quick)
        try:
            cost = rate * (now() - created) / 3600.0 if state.get("pod_id") else 0.0
            ledger.release_pod(run_id, name, cost)
            ledger.finish_run(run_id, outcome == "complete", outcome == "complete")
            state.update(cost_estimate=cost, balance_end=rp.balance())
        except Exception as exc:  # noqa: BLE001
            alert(f"ledger/balance bookkeeping failed: {exc}")
        state["pods_deleted_verified"] = ok
        if ok and not quick:
            stop_watchdog(watchdog_pid, run_dir)
            state["watchdog_stopped"] = True
        elif ok:  # the late-listing window is still open: armed until StockWait stops it
            state["watchdog_stopped"] = False
        save_state(run_dir, state)
    code = 0 if outcome == "complete" and state.get("pods_deleted_verified") else 1
    return code, bool(no_capacity and not state.get("pod_id") and ok)


def guard(rp, state, args) -> Optional[str]:
    """A session's spend / lifetime check (each poll); also sweeps our prefix for a stray pod
    of an earlier stock-wait round every STRAY_SWEEP_S (``sweep_strays``)."""
    if now() >= state["deadline"]:  # the agent kills jobs here
        return "lifetime"
    sweep_strays(rp, state)
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


def preflight_argv(plan: Dict[str, Any], run: Dict[str, Any], transitions: int) -> List[str]:
    """``train_argv`` of ``run`` with a short ``--transitions`` and its own ``--out``."""
    argv = list(train_argv(plan, run))
    for flag, value in (("--transitions", str(int(transitions))), ("--out", "/r/out/_preflight")):
        argv[argv.index(flag) + 1] = value
    return argv


def fetch_log_tail(agent, name: str, run_dir: Path, limit: int = FAILED_LOG_TAIL) -> str:
    """The tail of a pod process's stdout+stderr, also saved to ``run_dir/failed_logs``;
    best effort (an unreachable agent returns a marker, never raises)."""
    try:
        tail = agent.get(f"/log/{name}").decode(errors="replace")[-limit:]
    except Exception as exc:  # noqa: BLE001
        return f"<log unavailable: {type(exc).__name__}: {exc}>"
    try:
        out = run_dir / "failed_logs"
        out.mkdir(exist_ok=True)
        (out / f"{name}.log").write_text(tail)
    except OSError:
        pass
    return tail


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
    # 4b. opt-in trainer preflight: the first run's exact command for a few transitions, so a
    # trainer that cannot start on this host fails once (with its log) instead of N times.
    if int(plan.get("preflight_transitions", 0)):
        argv = preflight_argv(plan, plan["runs"][0], int(plan["preflight_transitions"]))
        agent.exec("preflight", argv, "/r/repo", env)
        rc, stop = _wait_job(agent, rp, state, args, net, "preflight", PREFLIGHT_BUDGET_S)
        tail = fetch_log_tail(agent, "preflight", run_dir)
        note(stage="preflight", returncode=rc, stop=stop or None, tail=tail[-1500:])
        if stop:
            return f"trainer preflight: {stop}"
        if rc != 0:
            return f"trainer preflight failed (rc {rc}); log in failed_logs/preflight.log"
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
                if rcs[r] != 0:
                    tail = fetch_log_tail(agent, r, run_dir)
                    note(stage="run_failed_log", run=r, tail=tail[-1500:])
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
        image=plan.get("image", IMAGE),
        gpu_count=int(plan.get("gpu_count", 1)),
    )
    (work / "pod_body.json").write_text(json.dumps(body))
    rp = RpClient()
    dry = rp.create_pod(work / "pod_body.json", float(plan["max_hourly"]), confirm=False)
    total = float(plan["transitions"]) * len(plan["runs"])
    out = {
        "commit": inputs["commit"],
        "plan_sha256": plan["_sha256"],
        "runs": [r["name"] for r in plan["runs"]],
        "transitions_per_run": plan["transitions"],
        "image": body["imageName"],
        "extra_inputs": [
            {k: x[k] for k in ("dst", "sha256")} for x in plan.get("extra_inputs", [])
        ],
        "inits": {r["name"]: r.get("init", STUDENT_INIT) for r in plan["runs"]},
        "initial_wave_on_min_host": initial_wave(plan),
        "plan_warnings": plan_warnings(plan),
        "trainer_flag_problems_launch_refuses": inputs.get("trainer_flag_problems"),
        "g2_pod_wall_budget_s": total / float(plan["mac_rate"]) / float(plan["g2_factor"]),
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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    ap.add_argument(
        "--stock-wait-hours",
        type=float,
        default=0.0,
        help="opt-in: keep polling for capacity for up to this long (at most 3 h; default 0 = "
        "a single round of create tries)",
    )
    ap.add_argument(
        "--stock-poll-s",
        type=float,
        default=STOCK_POLL_S,
        help=f"seconds between stock-wait rounds, start to start (default {STOCK_POLL_S}; "
        f"{STOCK_POLL_MIN_S}-{STOCK_POLL_MAX_S}); a refused create costs $0",
    )
    ap.add_argument(
        "--pod-prefix",
        default=POD_NAME,
        help="pod name prefix: this runner only sees, guards and deletes pods with it "
        "(distinct prefixes let independent runs share the account concurrently)",
    )
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*-", args.pod_prefix):
        print("--pod-prefix must look like 'name-' (lowercase, ends with '-')", file=sys.stderr)
        return 2
    globals()["POD_NAME"] = args.pod_prefix
    if args.accept_no_self_delete is not None and len(args.accept_no_self_delete.strip()) < 10:
        print("--accept-no-self-delete needs a real note (who decided, when)", file=sys.stderr)
        return 2
    problem = stock_poll_problem(args.stock_poll_s)
    if problem:
        print(problem, file=sys.stderr)
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
