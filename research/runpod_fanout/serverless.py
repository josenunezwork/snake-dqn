#!/usr/bin/env python3
"""Serverless CPU backend for the fan-out runner (the default backend for Tier-1 CPU jobs).

Code and checkpoints reach workers ONLY through a private RunPod network volume (no image
or registry holds our code). The worker image is a public base image pinned by digest; its
start command runs ``sls_handler.py`` from the volume. Lifecycle (all spending steps are
dry runs unless ``--confirm``)::

    serverless.py volume-create   [--confirm]               # one-time, ~$0.35/month (5 GB)
    serverless.py seed --commit C [--confirm]                # short cpu5c-2 pod: archive +
                                                             #   4 checkpoints + venvs
    serverless.py template-create [--confirm]                # serverless template (free)
    runner.py plan JOB.json --budget USD                     # sizing + speed/cost table
    runner.py run  JOB.json --budget USD --confirm           # per-run endpoint, then delete
    serverless.py endpoint-cleanup --job-id ID [--confirm]   # manual teardown
    serverless.py volume-cleanup  [--confirm]                # delete the volume
    serverless.py settle                                     # ledger: upper bound -> billing
    serverless.py status

Per-run endpoints (``rpf-sls-<job>--<run>-<n>``), not one shared endpoint per commit: each
run owns its queue (cancel/purge/scale-to-0/delete never touch another run's jobs), its
ledger reservation is exactly its own ``workersMax x vCPU`` worst case, and its watchdog may
delete it without coordination. The cost is one cold start per run (~2-3 min).

Determinism (per WORLD, see platform_rule.py): one serverless job = one world unit (every
arm of one wrapper/mix/world_seed) on ONE worker, so paired comparisons never cross CPU
models; different worlds may land on different models (the endpoint rents cpu5c, then
cpu3c). A probe job verifies the runtime first. Workers run with fixed threads and
MKL_CBWR=COMPATIBLE; duplicates must be byte-identical (deterministic) or the run aborts.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec, platform_rule  # noqa: E402
from research.runpod_fanout import runner as rmod  # noqa: E402
from research.runpod_fanout.ledger import SLACK_SECONDS, DuplicateRun, SharedLedger  # noqa: E402
from research.runpod_fanout.rp_client import RpClient, RunPodError  # noqa: E402

HERE = Path(__file__).resolve().parent
SLS_POLICY_PATH = HERE / "serverless_policy.json"
HANDLER_SOURCE = HERE / "sls_handler.py"
SEED_AGENT_SOURCE = HERE / "seed_agent.py"
REGISTRY_SCHEMA = "runpod-fanout-serverless-registry/v1"
LEDGER_SAFETY = 1.02
MAX_UNIT_EPISODES = 64  # sls_handler accepts at most 64 episodes per job
Abort = rmod.Abort


# ---------------------------------------------------------------- policy / runtime


def load_sls_policy(path: Path = SLS_POLICY_PATH) -> Dict[str, Any]:
    pol = json.loads(Path(path).read_text(encoding="utf-8"))
    if pol.get("schema") != "runpod-fanout-serverless-policy/v1":
        raise jobspec.JobError("unknown serverless policy schema")
    if not set(pol["flavors_pref"]) <= {"cpu5c", "cpu3c"}:
        raise jobspec.JobError("serverless flavors must stay cpu5c/cpu3c (user-approved)")
    if "@sha256:" not in pol["image"]:
        raise jobspec.JobError("serverless image must be pinned by digest")
    return pol


def volume_root(sp: Mapping[str, Any]) -> str:
    return f"{sp['mount']}/{sp['root_dir']}"


def runtime_spec(fp: Mapping[str, Any], sp: Mapping[str, Any]) -> Dict[str, Any]:
    """What the seeder builds: the pod pins (same order as a pod) + the handler venv."""
    return {
        "evenv_pip": [
            ["install", "-q", "--no-cache-dir", *fp["pip_pins"]],
            [
                "install",
                "-q",
                "--no-cache-dir",
                fp["torch_pin"],
                "--index-url",
                fp["torch_index_url"],
            ],
        ],
        "hvenv_pip": [list(x) for x in sp["handler_pip"]],
        "hvenv_import": sp["handler_import"],
        "expect": {
            "torch": fp["torch_pin"].split("==")[1] + "+cpu",
            "numpy": next(p for p in fp["pip_pins"] if p.startswith("numpy=="))[7:],
        },
    }


def handler_sha256() -> str:
    return hashlib.sha256(HANDLER_SOURCE.read_bytes()).hexdigest()


def runtime_id(fp: Mapping[str, Any], sp: Mapping[str, Any]) -> str:
    blob = json.dumps(
        {
            "handler": handler_sha256(),
            "image": sp["image"],
            "root": volume_root(sp),
            "spec": runtime_spec(fp, sp),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def runtime_dir(sp: Mapping[str, Any], rid: str) -> str:
    return f"{volume_root(sp)}/runtime/{rid}"


# ---------------------------------------------------------------- local registry


class Registry:
    """What has been seeded/created (written only after RunPod confirmed it). fcntl-locked."""

    def __init__(self, fp: Mapping[str, Any]):
        self.root = Path(fp["artifacts_root"]) / "runpod-fanout" / "serverless"
        self.path = self.root / "registry.json"

    @contextmanager
    def locked(self, write: bool = True) -> Iterator[Dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "registry.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                state = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {
                        "schema": REGISTRY_SCHEMA,
                        "volume": None,
                        "commits": {},
                        "ckpts": {},
                        "runtimes": {},
                    }
                )
                yield state
                if write:
                    tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
                    os.replace(tmp, self.path)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> Dict[str, Any]:
        with self.locked(write=False) as state:
            return json.loads(json.dumps(state))


def seed_problems(
    reg: Mapping[str, Any], commit: str, repo_sha: str, ckpts: Sequence[str], rid: str
) -> List[str]:
    out = []
    if not reg.get("volume"):
        out.append("no network volume registered (serverless.py volume-create)")
    if (reg.get("commits") or {}).get(commit, {}).get("repo_sha256") != repo_sha:
        out.append(
            f"commit {commit[:12]} not seeded with archive {repo_sha[:12]} "
            f"(serverless.py seed --commit {commit})"
        )
    missing = [s[:12] for s in ckpts if s not in (reg.get("ckpts") or {})]
    if missing:
        out.append(f"checkpoints not seeded: {missing}")
    rt = (reg.get("runtimes") or {}).get(rid) or {}
    if not rt.get("ready"):
        out.append(f"runtime {rid} not built on the volume (serverless.py seed)")
    elif not rt.get("template_id"):
        out.append(f"no serverless template for runtime {rid} (serverless.py template-create)")
    return out


# ---------------------------------------------------------------- bodies


def template_body(sp: Mapping[str, Any], rid: str) -> Dict[str, Any]:
    rdir = runtime_dir(sp, rid)
    return {
        "name": f"{sp['template_prefix']}{rid}",
        "imageName": sp["image"],
        "isServerless": True,
        "isPublic": False,
        "category": "CPU",
        "containerDiskInGb": int(sp["container_disk_gb"]),
        "volumeInGb": 0,
        "dockerStartCmd": ["bash", "-c", f"exec {rdir}/hvenv/bin/python -u {rdir}/sls_handler.py"],
        "env": {"RPF_VOLUME_ROOT": volume_root(sp), "PYTHONUNBUFFERED": "1"},
        "readme": "snake-dqn fan-out serverless worker; code lives on a private network volume",
    }


def endpoint_body(
    sp: Mapping[str, Any],
    name: str,
    template_id: str,
    volume: Mapping[str, Any],
    flavors: Sequence[str],
    workers: int,
    vcpu: int,
    execution_timeout_s: float,
) -> Dict[str, Any]:
    return {
        "name": name,
        "templateId": template_id,
        "computeType": "CPU",
        # rent order = preference (cpu5c first); mixed models are fine: one unit per worker
        "cpuFlavorIds": list(flavors),
        "vcpuCount": int(vcpu),
        "dataCenterIds": [volume["dataCenterId"]],
        "networkVolumeId": volume["id"],
        "workersMin": 0,
        "workersMax": int(workers),
        "idleTimeout": int(sp["idle_timeout_seconds"]),
        "scalerType": sp["scaler_type"],
        "scalerValue": int(sp["scaler_value"]),
        "executionTimeoutMs": int(execution_timeout_s * 1000),
        "flashboot": False,
    }


def start_command(source: Path, dest: str) -> List[str]:
    blob = base64.b64encode(gzip.compress(Path(source).read_bytes(), mtime=0)).decode()
    return [
        "bash",
        "-c",
        f"mkdir -p /r && echo {blob} | base64 -d | gunzip > {dest} && " f"exec python {dest}",
    ]


def seeder_pod_body(
    fp: Mapping[str, Any],
    sp: Mapping[str, Any],
    name: str,
    volume: Mapping[str, Any],
    token_sha: str,
    until_epoch: float,
    job_id: str,
    flavor: Optional[str] = None,
    vcpu: Optional[int] = None,
) -> Dict[str, Any]:
    s = sp["seeder"]
    return {
        "name": name,
        "computeType": "CPU",
        "cpuFlavorIds": [flavor or s["flavors"][0]],
        "vcpuCount": int(vcpu or s["vcpu"][0]),
        "cloudType": "SECURE",
        "dataCenterIds": [volume["dataCenterId"]],
        "imageName": sp["image"],  # the same image the workers run (venv base python)
        "containerDiskInGb": int(sp["container_disk_gb"]),
        "volumeInGb": 0,
        "networkVolumeId": volume["id"],
        "volumeMountPath": sp["mount"],  # same absolute path as serverless workers
        "ports": [f"{fp['agent_port']}/http"],
        "env": {
            "FANOUT_TOKEN_SHA256": token_sha,
            "FANOUT_SELF_DELETE_EPOCH": str(int(until_epoch)),
            "SEED_ROOT": volume_root(sp),
            "SEED_RUNTIME_JSON": json.dumps(runtime_spec(fp, sp), sort_keys=True),
            "FANOUT_JOB": job_id,
        },
        "dockerStartCmd": start_command(SEED_AGENT_SOURCE, "/r/seed_agent.py"),
    }


# ---------------------------------------------------------------- sizing


def unit_times(
    fp: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]], slots: int
) -> List[float]:
    """Per world unit: wall on ONE worker with ``slots`` parallel episodes."""
    by: Dict[str, List[float]] = {}
    for e in episodes:
        by.setdefault(platform_rule.unit_of_episode(e), []).append(rmod.episode_seconds(fp, e))
    return [math.ceil(len(v) / max(1, slots)) * max(v) for v in by.values()]


def sizing_plan(
    fp: Mapping[str, Any],
    sp: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    budget: float,
    max_wall_minutes: float,
    flavors: Sequence[str],
    headroom: Optional[float] = None,
    target_minutes: Optional[float] = None,
    workers: Optional[int] = None,
    vcpu: Optional[int] = None,
) -> Dict[str, Any]:
    """Pick workersMax x vCPU/worker: the cheapest config finishing within the target wall
    (default 30 min, else <= 60 min, else the fastest that fits). One serverless job = one
    world unit (all its arms on one worker). Every option's worst case (= what the ledger
    reserves) must fit the budget, the account headroom and the hourly cap."""
    secs = [rmod.episode_seconds(fp, e) for e in episodes]
    price = max(float(sp["usd_per_vcpu_hr"][f]) for f in flavors)
    margin = float(sp["price_margin"])
    threads = int(fp["threads_per_episode"])
    horizon_s = 60 * max_wall_minutes + float(fp["watchdog_grace_seconds"]) + 300 + SLACK_SECONDS
    cold, idle = float(sp["cold_start_seconds_est"]), float(sp["idle_timeout_seconds"])
    target = float(target_minutes or sp["target_wall_minutes"])
    rows = []
    n_units = len({platform_rule.unit_of_episode(e) for e in episodes})
    for w in sp["vcpu_sizes"]:
        slots = max(1, int(w) // threads)
        times = unit_times(fp, episodes, slots)
        mean_t, max_t = sum(times) / len(times), max(times)
        for n in range(1, int(sp["max_workers"]) + 1):
            wall = cold + max(max_t, math.ceil(len(times) / n) * mean_t)
            rate = n * int(w) * price
            worst = LEDGER_SAFETY * margin * rate * horizon_s / 3600
            why = []
            if worst > budget + 1e-9:
                why.append("budget")
            if headroom is not None and worst > headroom:
                why.append("account headroom")
            if rate * margin > float(sp["max_endpoint_hourly_usd"]):
                why.append("hourly cap")
            if wall > 60 * max_wall_minutes:
                why.append("max wall")
            rows.append(
                {
                    "workers": n,
                    "vcpu_per_worker": int(w),
                    "slots_per_worker": slots,
                    "parallel_units": n,
                    "vcpu_total": n * int(w),
                    "wall_minutes": round(wall / 60, 1),
                    "expected_usd": round(rate * (wall + idle) / 3600, 3),
                    "worst_case_usd": round(worst, 3),
                    "usd_per_hr": round(rate, 3),
                    "refused": why,
                }
            )
    ok = [r for r in rows if not r["refused"]]
    if workers or vcpu:
        pick = [
            r
            for r in rows
            if (not workers or r["workers"] == workers)
            and (not vcpu or r["vcpu_per_worker"] == vcpu)
        ]
        pick.sort(key=lambda r: (r["refused"] != [], r["wall_minutes"], r["expected_usd"]))
        choice = pick[0] if pick else None
    else:

        def best(limit):
            c = [r for r in ok if r["wall_minutes"] <= limit]
            return min(
                c,
                key=lambda r: (r["expected_usd"], r["workers"], -r["vcpu_per_worker"]),
                default=None,
            )

        choice = (
            best(target)
            or best(float(sp["max_target_wall_minutes"]))
            or min(ok, key=lambda r: (r["wall_minutes"], r["expected_usd"]), default=None)
        )
    front: Dict[float, Dict[str, Any]] = {}
    for r in sorted(ok, key=lambda r: (r["wall_minutes"], r["expected_usd"])):
        if all(r["expected_usd"] < f["expected_usd"] for f in front.values()):
            front[r["wall_minutes"]] = r
    return {
        "flavors": list(flavors),
        "usd_per_vcpu_hr_reserved": price,
        "price_margin_reserved": margin,
        "episodes": len(secs),
        "world_units": n_units,
        "episode_seconds_total_est": round(sum(secs), 1),
        "longest_episode_seconds_est": round(max(secs), 1),
        "target_wall_minutes": target,
        "reservation_horizon_minutes": round(horizon_s / 60, 1),
        "choice": choice,
        "choice_ok": bool(choice and not choice["refused"]),
        "speed_cost_tradeoff": list(front.values())[:10],
        "note": "one job = one world unit; expected = all workers billed for the whole wall "
        "(+idle timeout); worst = workersMax x vCPU x max flavor price x margin to the run's "
        "hard end (what the ledger reserves)",
    }


# ---------------------------------------------------------------- endpoint teardown


def owned_endpoint(prefix: str, name: Any) -> bool:
    """``prefix`` + counter (RunPod may append a suffix after a space, e.g. ' -fb'). A
    prefix ending in ``--`` (a job) or equal to the bare policy prefix matches any run."""
    head = str(name or "").split(" ")[0]
    if prefix.endswith("--") or "--" not in prefix:
        return head.startswith(prefix)
    return head.startswith(prefix) and head[len(prefix) :].isdigit()


def _status(exc: RunPodError) -> Optional[int]:
    return exc.payload.get("http_status") if isinstance(exc.payload, dict) else None


def _gone(rp: Any, endpoint_id: str) -> bool:
    try:
        return rp.get_endpoint(endpoint_id) is None
    except RunPodError:
        return False


def teardown_endpoint(
    rp: Any,
    endpoint_id: str,
    job_ids: Sequence[str] = (),
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    wait_workers_seconds: float = 180.0,
) -> bool:
    """Scale to 0, purge the queue, cancel ``job_ids``, wait for workers to leave, DELETE.
    True only when the endpoint is confirmed gone."""
    try:
        rp.update_endpoint(endpoint_id, {"workersMin": 0, "workersMax": 0}, confirm=True)
    except RunPodError as exc:
        if _status(exc) == 404 or _gone(rp, endpoint_id):
            return True
        log(f"scale-to-0 {endpoint_id} failed: {str(exc)[:160]}")
    for op, jid in [("purge-queue", None)] + [("cancel", j) for j in job_ids]:
        try:
            rp.sls(endpoint_id, op, jid, confirm=True)
        except RunPodError as exc:
            log(f"{op} {endpoint_id} {jid or ''} failed: {str(exc)[:120]}")
    t0 = clock()
    while clock() - t0 < wait_workers_seconds:
        try:
            info = rp.get_endpoint(endpoint_id, workers=True)
        except RunPodError:
            break
        if info is None:
            return True
        if not info.get("workers"):
            break
        sleep(15)
    for attempt in range(4):
        try:
            rp.delete_endpoint(endpoint_id, confirm=True)
        except RunPodError as exc:
            if _status(exc) == 404 or _gone(rp, endpoint_id):
                return True
            log(f"delete {endpoint_id} failed (attempt {attempt + 1}): {str(exc)[:160]}")
            sleep(10 * (attempt + 1))
            continue
        for _ in range(3):
            if _gone(rp, endpoint_id):
                return True
            sleep(10)
    return False


def endpoint_registry(run_dir: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
    """endpoint id -> registry row from ``<run>/endpoints/*.response.json`` (owned names)."""
    out = {}
    for path in sorted((Path(run_dir) / "endpoints").glob("*.response.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if row.get("id") and owned_endpoint(prefix, row.get("name")):
            out[str(row["id"])] = row
    return out


def open_jobs(run_dir: Optional[Path]) -> Dict[str, List[str]]:
    """endpoint id -> job ids the runner had in flight (``<run>/endpoints/jobs.json``)."""
    if run_dir is None:
        return {}
    try:
        data = json.loads((Path(run_dir) / "endpoints" / "jobs.json").read_text())
        return {str(k): [str(j) for j in v] for k, v in data.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def sweep_endpoints(
    rp: Any,
    prefix: str,
    run_dir: Optional[Path],
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    give_up_seconds: float = 3600.0,
    only_overdue_before: Optional[float] = None,
) -> List[str]:
    """Tear down this prefix's endpoints (registry ids even if listing fails), cancelling
    the jobs the runner persisted; returns leftovers ([] = confirmed none left)."""
    started, gone = clock(), set()
    delay, left = 15.0, ["unconfirmed"]
    while clock() - started < give_up_seconds:
        reg = endpoint_registry(run_dir, prefix) if run_dir else {}
        if only_overdue_before is not None:
            reg = {
                i: r
                for i, r in reg.items()
                if float(r.get("rpf_until_epoch") or 0) + 120 < only_overdue_before
            }
        wanted = set(reg) - gone
        listed, list_ok = [], True
        try:
            listed = [
                str(e["id"]) for e in rp.list_endpoints() if owned_endpoint(prefix, e.get("name"))
            ]
        except RunPodError as exc:
            list_ok = False
            log(f"list endpoints failed: {str(exc)[:120]}")
        if only_overdue_before is not None:
            listed = [i for i in listed if i in reg]
        targets = sorted(set(listed) | wanted)
        if list_ok:
            gone |= wanted - set(listed)
            if not listed:
                return []
        jobs = open_jobs(run_dir)
        for eid in targets:
            if teardown_endpoint(rp, eid, jobs.get(eid, ()), log, sleep, clock):
                gone.add(eid)
                log(f"endpoint {eid} deleted")
        left = [t for t in targets if t not in gone]
        if not left and list_ok:
            sleep(10)  # then confirm with one more listing
            continue
        sleep(delay)
        delay = min(120.0, delay * 1.5)
    return left


# ---------------------------------------------------------------- serverless run


@dataclass
class SlsJob:
    id: str
    endpoint: str  # endpoint name
    op: str
    unit: Optional[str]
    keys: List[str]
    submitted: float
    timeout_s: float
    status: str = "IN_QUEUE"
    started: Optional[float] = None
    errors: int = 0


@dataclass
class Endpoint:
    name: str
    key: str
    flavors: List[str]
    workers: int
    vcpu: int
    rate: float  # reserved $/hr for all workers (max flavor price x margin)
    price_rate: float  # list $/hr for all workers (max flavor price, no margin)
    created: float
    until: float
    id: Optional[str] = None
    deleted: Optional[float] = None
    exec_seconds: float = 0.0
    workers_seen: set = field(default_factory=set)
    billing_usd: Optional[float] = None
    released: bool = False

    def estimate(self, sp: Mapping[str, Any]) -> float:
        """Best guess (job execution + one cold start/idle per worker seen); NOT a bound."""
        boot = len(self.workers_seen) * (
            float(sp["cold_start_seconds_est"]) + float(sp["idle_timeout_seconds"])
        )
        return self.price_rate / max(1, self.workers) * (self.exec_seconds + boot) / 3600

    def upper_bound(self, now: float) -> float:
        """Every worker billed for the endpoint's whole life at the reserved rate (+60 s)."""
        end = self.deleted if self.deleted is not None else now
        return self.rate * (max(0.0, end - self.created) + 60) / 3600

    def worst(self, now: float) -> float:
        end = self.deleted if self.deleted is not None else self.until
        return LEDGER_SAFETY * self.rate * (max(0.0, end - self.created) + 60) / 3600


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ServerlessRunner(rmod.Runner):
    """Same job files, records, ledger, receipts and watchdog as pods; work goes to a
    per-run serverless endpoint, one world unit per job (see the module docstring)."""

    def __init__(
        self,
        job: Mapping[str, Any],
        policy: Mapping[str, Any],
        allowlist: Mapping[str, Mapping[str, str]],
        run_dir: Path,
        *,
        sls_policy: Mapping[str, Any],
        sizing: Mapping[str, Any],
        flavors: Sequence[str],
        registry: Optional[Registry] = None,
        **kw: Any,
    ):
        super().__init__(job, policy, allowlist, run_dir, **kw)
        self.sp = sls_policy
        self.workers = int(sizing["workers"])
        self.vcpu = int(sizing["vcpu_per_worker"])
        self.slots = max(1, self.vcpu // int(policy["threads_per_episode"]))
        self.flavors = list(flavors)
        self.registry = registry or Registry(policy)
        self.rid = runtime_id(policy, sls_policy)
        self.endpoints: Dict[str, Endpoint] = {}
        self.jobs: Dict[str, SlsJob] = {}
        self.ep_counter = 0
        self.volume: Dict[str, Any] = {}
        self.template_id: Optional[str] = None
        self.probe_ok = False
        self.probe_job: Optional[str] = None
        self.probe_worker: Dict[str, Any] = {}
        self.last_billing_check = 0.0
        big = [u for u, keys in self.units.items() if len(keys) > MAX_UNIT_EPISODES]
        if big:
            raise jobspec.JobError(
                f"{len(big)} world units exceed {MAX_UNIT_EPISODES} episodes (one serverless "
                f"job each), e.g. {big[0]}"
            )

    # ------------------------------------------------------------ naming
    def endpoint_prefix(self) -> str:
        return f"{self.sp['endpoint_prefix']}{self.job['job_id']}--{self.run_dir.name}-"

    # ------------------------------------------------------------ money
    def estimated_cost(self) -> float:
        return sum(e.estimate(self.sp) for e in self.endpoints.values())

    def committed_worst(self) -> float:
        now = self.clock()
        return sum(e.worst(now) for e in self.endpoints.values())

    # ------------------------------------------------------------ preparation
    def prepare_uploads(self, workdir: Path) -> Dict[str, Any]:
        """Nothing is uploaded by a run: verify the commit/checkpoints/runtime are seeded."""
        problems = jobspec.check_commit(self.repo, self.job)
        if problems:
            raise jobspec.JobError("; ".join(problems))
        archive = Path(workdir) / "repo.tar.gz"
        self.repo_sha = jobspec.git_archive(self.repo, self.job["repo_commit"], archive)
        archive.unlink()
        self.ckpt_paths = jobspec.resolve_checkpoints(
            self.job["checkpoints"], self.policy, self.allowlist
        )
        reg = self.registry.read()
        problems = seed_problems(
            reg, self.job["repo_commit"], self.repo_sha, self.job["checkpoints"], self.rid
        )
        if problems:
            raise jobspec.JobError("serverless volume not ready: " + "; ".join(problems))
        self.volume = dict(reg["volume"])
        self.template_id = reg["runtimes"][self.rid]["template_id"]
        return {
            "uploads": "none (code and checkpoints are already on the private volume)",
            "volume": self.volume,
            "runtime_id": self.rid,
            "template_id": self.template_id,
            "repo_sha256": self.repo_sha,
            "checkpoints": sorted(self.job["checkpoints"]),
        }

    # ------------------------------------------------------------ endpoint
    def exec_timeout(self, keys: Sequence[str]) -> float:
        return 4 * self.unit_seconds(keys, self.slots) + 300

    def max_exec_timeout(self) -> float:
        return max(self.exec_timeout(keys) for keys in self.units.values())

    def open_endpoint(self) -> Endpoint:
        now = self.clock()
        until = self.hard_end
        price = max(float(self.sp["usd_per_vcpu_hr"][f]) for f in self.flavors)
        margin = float(self.sp["price_margin"])
        self.ep_counter += 1
        name = f"{self.endpoint_prefix()}{self.ep_counter}"
        key = f"sls:{name}"
        try:
            balance = self.account_balance()
        except RunPodError as exc:
            raise Abort(f"balance unavailable: {exc}") from exc
        why: Optional[str] = "no worker count fits"
        workers = self.workers
        while workers >= 1:
            rate = workers * self.vcpu * price * margin
            worst = LEDGER_SAFETY * rate * (max(0.0, until - now) + 60) / 3600
            if worst > self.budget - self.committed_worst() + 1e-9:
                why = f"job budget: worst {worst:.3f} > remaining budget"
            else:
                why = self.ledger.reserve_pod(
                    self.run_id, key, rate, until, balance, kind="serverless"
                )
                if why is None:
                    break
            workers -= 1
        if workers < 1:
            raise Abort(f"no serverless reservation fits: {why}")
        if workers != self.workers:
            self.log("workers_reduced", wanted=self.workers, got=workers, reason=why)
        ep = Endpoint(
            name,
            key,
            list(self.flavors),
            workers,
            self.vcpu,
            workers * self.vcpu * price * margin,
            workers * self.vcpu * price,
            now,
            until,
        )
        self.endpoints[name] = ep
        body = endpoint_body(
            self.sp,
            name,
            self.template_id,
            self.volume,
            self.flavors,
            workers,
            self.vcpu,
            min(self.max_exec_timeout(), until - now),
        )
        path = self.run_dir / "endpoints" / f"{name}.request.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(body, indent=1))
        rmod._DEFER["active"] = True
        try:
            out = self.rp.create_endpoint(path, confirm=self.confirm)
        except RunPodError as exc:
            rmod._DEFER["active"] = False
            status = _status(exc)
            self.log("endpoint_create_refused", name=name, detail=str(exc.payload)[:300])
            if isinstance(status, int) and 400 <= status < 500:
                ep.deleted = self.clock()
                self.ledger_call("release_pod", self.run_id, key, 0.0)
                ep.released = True
                self.raise_deferred()
                raise Abort(f"endpoint create refused ({status}): {str(exc.payload)[:200]}")
            out = self.find_endpoint_by_name(name)
            if out is None:
                ep.deleted = self.clock()  # proven absent (two empty listings >= 60 s apart)
                self.ledger_call("release_pod", self.run_id, key, 0.0)
                ep.released = True
                self.raise_deferred()
                raise Abort("endpoint create failed (no endpoint exists)")
        except BaseException:
            rmod._DEFER["active"] = False
            raise
        rmod._DEFER["active"] = False
        if not isinstance(out, dict) or not out.get("id"):
            out = self.find_endpoint_by_name(name) or {}
            if not out.get("id"):
                ep.deleted = self.clock()  # proven absent (two empty listings >= 60 s apart)
                self.ledger_call("release_pod", self.run_id, key, 0.0)
                ep.released = True
                self.raise_deferred()
                raise Abort("endpoint create returned no id and no endpoint exists")
        ep.id = str(out["id"])
        (self.run_dir / "endpoints" / f"{name}.response.json").write_text(
            json.dumps(
                {
                    "id": ep.id,
                    "name": name,
                    "rpf_until_epoch": until,
                    "flavors": self.flavors,
                    "workersMax": workers,
                    "vcpuCount": self.vcpu,
                },
                indent=1,
            )
        )
        self.ledger_call("annotate", self.run_id, key, endpoint_id=ep.id)
        self.log(
            "endpoint_created",
            endpoint=ep.id,
            name=name,
            flavors=self.flavors,
            workers=workers,
            vcpu=self.vcpu,
            reserved_usd_per_hr=round(ep.rate, 4),
            committed_worst=round(self.committed_worst(), 4),
        )
        self.raise_deferred()
        return ep

    def find_endpoint_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """A create with an unknown outcome: found, or absent in two listings >= 60 s apart."""
        empty_since: Optional[float] = None
        for _ in range(30):
            try:
                rows = self.rp.list_endpoints()
            except RunPodError:
                self.sleep(10)
                continue
            for e in rows:
                if str(e.get("name", "")).split(" ")[0] == name:
                    return e
            now = self.clock()
            if empty_since is not None and now - empty_since >= 60:
                return None
            empty_since = empty_since if empty_since is not None else now
            self.sleep(30)
        raise Abort(f"cannot tell whether endpoint {name} exists (listing keeps failing)")

    def persist_jobs(self) -> None:
        """Open job ids per endpoint, so the watchdog can cancel them if this process dies."""
        out: Dict[str, List[str]] = {}
        for j in self.jobs.values():
            ep = self.endpoints[j.endpoint]
            if ep.id:
                out.setdefault(ep.id, []).append(j.id)
        d = self.run_dir / "endpoints"
        d.mkdir(exist_ok=True)
        tmp = d / ".jobs.json.tmp"
        tmp.write_text(json.dumps(out, sort_keys=True))
        os.replace(tmp, d / "jobs.json")

    def close_endpoint(self, ep: Endpoint, why: str) -> bool:
        if ep.deleted is not None:
            return True
        open_ids = [j.id for j in self.jobs.values() if j.endpoint == ep.name]
        ok = True
        if ep.id:
            ok = teardown_endpoint(
                self.rp,
                ep.id,
                open_ids,
                lambda m: self.log("teardown", detail=m),
                self.sleep,
                self.clock,
            )
        if ok:
            ep.deleted = self.clock()
            self.log(
                "endpoint_deleted", endpoint=ep.id, why=why, est_cost=round(ep.estimate(self.sp), 5)
            )
            for jid in open_ids:
                self.jobs.pop(jid, None)
            self.persist_jobs()
        else:
            print(
                f"!!! ENDPOINT {ep.id} NOT CONFIRMED DELETED; the watchdog keeps trying; "
                f"manual: serverless.py endpoint-cleanup --job-id {self.job['job_id']} "
                "--confirm",
                file=sys.stderr,
                flush=True,
            )
            self.log("endpoint_delete_unconfirmed", endpoint=ep.id)
        return ok

    # ------------------------------------------------------------ jobs
    def base_input(self, op: str) -> Dict[str, Any]:
        return {
            "op": op,
            "runtime_id": self.rid,
            "commit": self.job["repo_commit"],
            "repo_sha256": self.repo_sha,
            "checkpoints": sorted(self.job["checkpoints"]),
        }

    def submit(self, ep: Endpoint, op: str, unit: Optional[str] = None) -> bool:
        now = self.clock()
        keys = self.unit_keys(unit) if unit else []
        timeout = self.exec_timeout(keys) if keys else 600.0
        timeout = min(timeout, self.hard_end - now - 60)
        if timeout < 30 or (unit and not keys):
            return False
        inp = self.base_input(op)
        if op == "episodes":
            inp.update(
                {
                    "job_id": self.job["job_id"],
                    "provider": f"runpod-serverless:{'/'.join(ep.flavors)}",
                    "slots": self.slots,
                    "timeout_seconds": max(30.0, timeout - 60),  # inside RunPod's limit
                    "deadline_epoch": self.deadline,
                    "require_cpu_model": None,  # per-world: any model, one worker per unit
                    "numerics_env": dict(self.numerics),
                    "episodes": [{"key": k, "spec": self.episodes[k]} for k in keys],
                }
            )
        body = {
            "input": inp,
            "policy": {
                "executionTimeout": int(max(5.0, timeout) * 1000),
                "ttl": int(max(10.0, self.deadline - now) * 1000),
            },
        }
        try:
            out = self.rp.sls(ep.id, "run", body=body, confirm=True)
        except RunPodError as exc:
            self.log("submit_failed", endpoint=ep.id, detail=str(exc)[:200])
            return False
        if not isinstance(out, dict) or not out.get("id"):
            self.log("submit_no_id", detail=str(out)[:200])
            return False
        jid = str(out["id"])
        if unit:
            self.take_unit(unit, f"sls:{jid}")
        self.jobs[jid] = SlsJob(jid, ep.name, op, unit, list(keys), now, timeout)
        if op == "probe":
            self.probe_job = jid
        self.persist_jobs()
        self.log("submitted", job=jid, op=op, unit=unit, keys=len(keys))
        return True

    def save_failure(self, job: SlsJob, info: Any) -> None:
        d = self.run_dir / "failures"
        d.mkdir(exist_ok=True)
        (d / f"{job.id}.json").write_text(json.dumps(info, indent=1, default=str)[:200000])

    def poll(self) -> None:
        if not self.jobs:
            return
        jobs = list(self.jobs.values())

        def one(j: SlsJob):
            ep = self.endpoints[j.endpoint]
            try:
                return j, self.rp.sls(ep.id, "status", j.id), None
            except RunPodError as exc:
                return j, None, exc

        with concurrent.futures.ThreadPoolExecutor(max(1, self.probe_workers)) as pool:
            results = list(pool.map(one, jobs))
        for j, st, exc in results:
            if j.id not in self.jobs:
                continue
            if exc is not None:
                j.errors += 1
                if _status(exc) == 404 or j.errors >= 20:
                    self.cancel_quietly(j)
                    self.drop_job(j)
                    self.save_failure(j, {"status_error": str(exc)[:500]})
                    self.on_lost(j, "status unavailable")
                continue
            j.errors = 0
            self.on_status(j, st if isinstance(st, dict) else {})
        self.persist_jobs()

    def drop_job(self, j: SlsJob) -> None:
        self.jobs.pop(j.id, None)
        if self.probe_job == j.id:
            self.probe_job = None

    def on_lost(self, j: SlsJob, why: str) -> None:
        if j.unit:
            self.requeue_unit(j.unit, why)  # the whole world restarts on one worker

    def on_status(self, j: SlsJob, st: Mapping[str, Any]) -> None:
        status = str(st.get("status") or "")
        j.status = status
        now = self.clock()
        if status == "IN_PROGRESS" and j.started is None:
            j.started = now
        if status in ("IN_QUEUE", "IN_PROGRESS", ""):
            if j.started and now - j.started > j.timeout_s + 600:  # should have timed out
                self.drop_job(j)
                self.cancel_quietly(j)
                self.on_lost(j, "job overran its execution timeout")
            return
        self.drop_job(j)
        ep = self.endpoints[j.endpoint]
        ep.exec_seconds += float(st.get("executionTime") or 0) / 1000.0
        if st.get("workerId"):
            ep.workers_seen.add(str(st["workerId"]))
        out = st.get("output")
        if status != "COMPLETED" or not isinstance(out, dict):
            self.save_failure(j, dict(st))
            if j.op == "probe":
                raise Abort(f"probe job {status}: {str(st.get('error'))[:300]}")
            self.on_lost(j, f"job {status or 'without output'}")
            return
        worker = dict(out.get("worker") or {})
        if out.get("refused"):
            self.save_failure(j, dict(st))
            if j.op == "probe":
                raise Abort(f"probe refused: {out['refused']}")
            self.on_lost(j, f"refused: {str(out['refused'])[:80]}")
            return
        if j.op == "probe":
            self.on_probe(worker)
            return
        winfo = {
            "isa_flags": list(worker.get("isa_flags") or []),
            "worker_id": worker.get("worker_id") or st.get("workerId"),
            "backend": "serverless",
        }
        results = {r.get("key"): r for r in out.get("results") or [] if isinstance(r, dict)}
        unknown = set(results) - set(j.keys)
        if unknown:
            raise Abort(f"job {j.id} returned unknown records {sorted(unknown)[:3]}")
        bad = []
        for key in j.keys:
            r = results.get(key)
            if not r or not r.get("ok"):
                bad.append(key)
                continue
            data = str(r["record"]).encode("utf-8")
            if hashlib.sha256(data).hexdigest() != r.get("sha256"):
                bad.append(key)
                continue
            self.accept_record(key, data, source=f"sls:{j.id}", worker=winfo)
        if bad:
            self.save_failure(j, {"bad": bad, "results": [results.get(k) for k in bad]})
            self.requeue_unit(j.unit, "episode error on worker")
        self.log(
            "collected",
            job=j.id,
            unit=j.unit,
            ok=len(j.keys) - len(bad),
            bad=len(bad),
            completed=len(self.completed),
            of=len(self.order),
        )

    def on_probe(self, worker: Mapping[str, Any]) -> None:
        if worker.get("runtime_id") != self.rid:
            raise Abort(f"worker runtime {worker.get('runtime_id')} != {self.rid}")
        if worker.get("handler_sha256") != handler_sha256():
            raise Abort("worker handler bytes differ from this runner's sls_handler.py")
        self.probe_ok = True
        self.probe_worker = dict(worker)
        self.log("probe_ok", worker=worker)

    def cancel_quietly(self, j: SlsJob) -> None:
        try:
            self.rp.sls(self.endpoints[j.endpoint].id, "cancel", j.id, confirm=True)
        except RunPodError:
            pass

    def fill(self, ep: Endpoint) -> None:
        cap = max(1, math.ceil(ep.workers * float(self.sp["inflight_jobs_per_worker"])))
        for unit in self.pending_units():
            if len([j for j in self.jobs.values() if j.op == "episodes"]) >= cap:
                return
            if not self.unit_keys(unit):
                continue
            if not self.submit(ep, "episodes", unit):
                return

    def billing_guard(self, ep: Endpoint) -> None:
        """Every 10 min: what RunPod billed this endpoint must stay inside its reserved rate
        (the policy prices are not readable from the API)."""
        now = self.clock()
        if now - self.last_billing_check < 600 or not ep.id:
            return
        self.last_billing_check = now
        try:
            billed = self.rp.endpoint_billing(ep.id, iso(ep.created - 3600), iso(now + 3600))
        except (RunPodError, ValueError, TypeError) as exc:
            self.log("billing_check_failed", detail=str(exc)[:120])
            return
        allowed = ep.rate * (now - ep.created + 120) / 3600
        self.log("billing_check", billed=billed, allowed=round(allowed, 4))
        if billed > allowed + 0.01:
            raise Abort(
                f"billing {billed:.4f} exceeds the reserved rate ({allowed:.4f}): the serverless "
                "prices in serverless_policy.json are too low"
            )

    # ------------------------------------------------------------ main loop
    def loop(self) -> None:
        if not self.order:
            self.stop_reason = "nothing to run (every unit already complete)"
            return
        ep = self.open_endpoint()
        poll_s = float(self.sp["poll_seconds"])
        while True:
            now = self.clock()
            if len(self.completed) + len(self.failed) == len(self.order):
                self.stop_reason = "all episodes finished"
                return
            if now >= self.deadline:
                self.stop_reason = "max wall time reached"
                return
            self.check_watchdog()
            self.try_legacy_lock()
            if now - self.last_reconcile >= 300:
                self.last_reconcile = now
                try:
                    out = self.ledger.reconcile(self.rp)
                    if out["released"] or out["deleted_overdue_dead_run_pods"]:
                        self.log("ledger_reconciled", **out)
                except (RunPodError, OSError, ValueError, KeyError) as exc:
                    self.log("ledger_reconcile_failed", detail=str(exc)[:160])
            self.poll()
            self.billing_guard(ep)
            if not self.probe_ok:
                probe = self.jobs.get(self.probe_job) if self.probe_job else None
                if probe is None:
                    self.submit(ep, "probe")
                elif probe.status == "IN_QUEUE" and now - ep.created > float(
                    self.sp["startup_deadline_seconds"]
                ):
                    self.stop_reason = "no serverless worker started (capacity)"
                    self.log("no_worker_started", flavors=ep.flavors)
                    return
            else:
                self.fill(ep)
            if self.committed_worst() > self.budget + 1e-6:
                raise Abort("committed worst case exceeds the budget")
            self.write_state()
            self.sleep(poll_s)

    def release(self, ep: Endpoint, billing: Optional[float] = None) -> None:
        """Ledger cost = an upper bound (every worker for the endpoint's whole life at the
        reserved rate); ``SharedLedger.settle_serverless`` lowers it from final billing."""
        if ep.released or ep.deleted is None:
            return  # an unconfirmed endpoint keeps its full reservation until its horizon
        upper = max(ep.upper_bound(ep.deleted), ep.estimate(self.sp))
        self.ledger_call("release_pod", self.run_id, ep.key, upper)
        self.ledger_call(
            "annotate",
            self.run_id,
            ep.key,
            settle={
                "endpoint_id": ep.id,
                "start": ep.created,
                "end": ep.deleted,
                "estimate": round(ep.estimate(self.sp), 6),
                "upper": upper,
                "billing_at_finish": billing,
            },
        )
        ep.released = True

    def teardown_all(self) -> List[Dict[str, Any]]:
        for ep in list(self.endpoints.values()):
            if ep.deleted is None and ep.id:
                self.close_endpoint(ep, "run exit")
        # a create with an unknown outcome or a failed DELETE: sweep this run's prefix
        try:
            left = sweep_endpoints(
                self.rp,
                self.endpoint_prefix(),
                self.run_dir,
                lambda m: self.log("sweep", detail=m),
                self.sleep,
                self.clock,
                give_up_seconds=600,
            )
        except Exception as exc:  # noqa: BLE001 - the watchdog retries
            left = [f"sweep error: {exc!r}"[:200]]
        if not left:  # confirmed: no endpoint of this run exists any more
            for ep in self.endpoints.values():
                if ep.deleted is None:
                    ep.deleted = self.clock()
            self.jobs.clear()
            self.persist_jobs()
        out = [{"id": i} for i in left]
        out += [
            {"id": e.id, "unconfirmed": True}
            for e in self.endpoints.values()
            if e.deleted is None and e.id not in left
        ]
        if out:
            self.log("cleanup_incomplete", leftovers=out)
        return out

    def write_state(self) -> None:
        state = {
            "backend": "serverless",
            "job_id": self.job["job_id"],
            "updated_utc": rmod.utc_now(),
            "deadline_epoch": self.deadline,
            "episodes": len(self.order),
            "units": len(self.units),
            "completed": len(self.completed),
            "pending": len(self.pending),
            "failed": self.failed,
            "jobs_open": len(self.jobs),
            "endpoints": [self.ep_public(e) for e in self.endpoints.values()],
            "estimated_cost_usd": round(self.estimated_cost(), 4),
            "committed_worst_usd": round(self.committed_worst(), 4),
            "budget_usd": self.budget,
            "stop_reason": self.stop_reason,
            "runner_pid": os.getpid(),
        }
        tmp = self.run_dir / ".state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(tmp, self.run_dir / "state.json")

    def ep_public(self, e: Endpoint) -> Dict[str, Any]:
        return {
            "id": e.id,
            "name": e.name,
            "flavors": e.flavors,
            "workers_max": e.workers,
            "vcpu_per_worker": e.vcpu,
            "reserved_usd_per_hr": round(e.rate, 4),
            "created": e.created,
            "deleted": e.deleted,
            "exec_seconds": round(e.exec_seconds, 1),
            "workers_seen": len(e.workers_seen),
            "est_cost_usd": round(e.estimate(self.sp), 5),
            "upper_bound_usd": round(e.upper_bound(e.deleted or self.clock()), 5),
            "billing_usd": e.billing_usd,
        }

    def run(self) -> int:
        if not self.confirm:
            raise jobspec.JobError("run needs --confirm")
        self.start = self.clock()
        self.deadline = self.start + 60.0 * float(self.job["max_wall_minutes"])
        self.hard_end = self.deadline + float(self.policy["watchdog_grace_seconds"]) + 300.0
        if self.run_dir.exists():
            raise jobspec.JobError(f"run dir {self.run_dir} already exists")
        self.try_legacy_lock()
        try:
            totals = self.ledger.register_run(
                self.run_id, self.job["job_id"], self.run_id, self.budget, os.getpid()
            )
        except DuplicateRun as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            self.stop_reason = f"duplicate: {exc}"
            return rmod.EXIT_DUPLICATE
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "job.json").write_text(json.dumps(self.job, indent=1, sort_keys=True))
        self.events = (self.run_dir / "events.jsonl").open("x")
        self.log("account_ledger", **totals)
        workdir = Path(tempfile.mkdtemp(prefix="rpf-sls-"))
        leftovers: List[Dict[str, Any]] = [{"unknown": True}]
        old_handlers = rmod.install_signal_handlers()
        code, awake = 1, None
        try:
            self.tls_preflight()
            manifest = self.prepare_uploads(workdir)
            self.balance_start = self.rp.balance()
            if self.balance_start - float(self.policy["global_floor_usd"]) <= 0:
                raise Abort(f"balance {self.balance_start:.2f} is at or below the floor")
            plan = {
                "backend": "serverless",
                "seeded": manifest,
                "workers": self.workers,
                "vcpu_per_worker": self.vcpu,
                "flavors": self.flavors,
                "balance_start": self.balance_start,
                "budget_usd_hard_cap": self.budget,
            }
            (self.run_dir / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
            self.watchdog = self.spawn_watchdog(self)
            self.check_watchdog(startup=True)
            self.log(
                "watchdog_armed",
                fires_at_epoch=self.deadline + float(self.policy["watchdog_grace_seconds"]),
            )
            awake = rmod.keep_awake(getattr(self.watchdog, "pid", None))
            self.loop()
            code = 0 if len(self.completed) == len(self.order) else 4
        except Abort as exc:
            self.stop_reason = f"abort: {exc}"
            self.log("abort", reason=str(exc))
            code = 3
        except KeyboardInterrupt:
            self.stop_reason = "interrupted (signal)"
            self.log("interrupted")
            code = 130
        finally:
            rmod.ignore_signals()
            try:
                leftovers = self.teardown_all() if self.endpoints else []
                shutil.rmtree(workdir, ignore_errors=True)
                self.finish(leftovers, code)
                self.ledger_call(
                    "finish_run",
                    self.run_id,
                    success=(code == 0 and not leftovers),
                    episodes_complete=len(self.completed) == len(self.order),
                )
                if self.legacy_lock is not None:
                    self.legacy_lock.close()
            finally:
                if awake is not None:
                    awake.terminate()
                rmod.restore_signal_handlers(old_handlers)
        return code if not leftovers else 5

    def finish(self, leftovers: List[Dict[str, Any]], code: int) -> None:
        balance_end = None
        self.sleep(60)  # let per-second billing settle
        try:
            balance_end = self.rp.balance()
        except RunPodError:
            pass
        end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for ep in self.endpoints.values():
            if ep.id:
                start_iso = datetime.fromtimestamp(ep.created - 3600, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                try:
                    ep.billing_usd = self.rp.endpoint_billing(ep.id, start_iso, end_iso)
                except (RunPodError, ValueError, TypeError):
                    ep.billing_usd = None
            self.release(ep, ep.billing_usd)
        receipt = {
            "schema": rmod.RECEIPT_SCHEMA,
            "backend": "serverless",
            "job_id": self.job["job_id"],
            "run_dir": str(self.run_dir),
            "exit_code": code,
            "stop_reason": self.stop_reason,
            "episodes": len(self.order),
            "completed": len(self.completed),
            "failed": self.failed,
            **self.unit_summary(),
            "probe_worker": self.probe_worker,
            "runtime_id": self.rid,
            "endpoints": [self.ep_public(e) for e in self.endpoints.values()],
            "estimated_cost_usd": round(self.estimated_cost(), 5),
            "billing_usd": round(sum(e.billing_usd or 0 for e in self.endpoints.values()), 5),
            "ledger_cost_usd_upper_bound": round(
                sum(e.upper_bound(e.deleted or self.clock()) for e in self.endpoints.values()), 5
            ),
            "balance_start": self.balance_start,
            "balance_end": balance_end,
            "actual_cost_usd_balance_delta": (
                None
                if balance_end is None or self.balance_start is None
                else round(self.balance_start - balance_end, 5)
            ),
            "cost_note": "the ledger records the upper bound (all workers x endpoint life x "
            "reserved rate) and lowers it from /billing/endpoints once hour buckets are final "
            "(SharedLedger.settle_serverless); the balance delta is account-wide",
            "final_endpoints_runner_owned": leftovers,
            "final_get_pods_runner_owned": [],
            "finished_utc": rmod.utc_now(),
        }
        (self.run_dir / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
        self.log(
            "actual_spend",
            estimated=receipt["estimated_cost_usd"],
            billing=receipt["billing_usd"],
            balance_delta=receipt["actual_cost_usd_balance_delta"],
            leftover_endpoints=len(leftovers),
        )
        self.write_state()
        if self.events is not None:
            self.events.close()
            self.events = None


# ---------------------------------------------------------------- seeder


class SeedRunner(rmod.Runner):
    """One short-lived cpu5c-2 pod with the volume mounted: upload the commit archive, the
    allow-listed checkpoints and the handler, build the venvs, verify, delete the pod."""

    def __init__(
        self,
        fp,
        sp,
        allowlist,
        run_dir,
        *,
        commit: str,
        volume: Mapping[str, Any],
        ckpts: Sequence[str],
        registry: Optional[Registry] = None,
        **kw,
    ):
        self.sp = sp
        self.rid = runtime_id(fp, sp)
        s = sp["seeder"]
        seed_policy = dict(
            fp,
            flavor=s["flavors"][0],
            vcpu_sizes_desc=sorted((int(v) for v in s["vcpu"]), reverse=True),
            max_hourly_per_pod_usd=float(s["max_hourly_usd"]),
            pod_max_lifetime_seconds=60 * float(s["max_wall_minutes"]) + 600,
        )
        ident = json.dumps([commit, self.rid, str(volume.get("id")), sorted(ckpts)])
        job = {
            # one id per (commit, runtime, volume, checkpoint set): a re-created volume or a
            # new allow-listed checkpoint gets a new seed, an identical one is idempotent
            "job_id": "seed-" + hashlib.sha256(ident.encode()).hexdigest()[:16],
            "repo_commit": commit,
            "episodes": [],
            "checkpoints": list(ckpts),
            "max_wall_minutes": float(s["max_wall_minutes"]),
        }
        super().__init__(job, seed_policy, allowlist, run_dir, **kw)
        self.volume = dict(volume)
        self.registry = registry or Registry(fp)
        self.fp = fp
        self.stage = "boot"
        self.seed_done = False

    def pod_body_for(self, name, row, token_sha, now, until):  # hook used by create_pod
        return seeder_pod_body(
            self.fp,
            self.sp,
            name,
            self.volume,
            token_sha,
            until,
            self.job["job_id"],
            flavor=row.get("flavor"),
            vcpu=row.get("vcpu"),
        )

    def probe(self, sizes=None):
        """Only the volume's data center can mount the volume; cheapest stocked first (the
        seeder's CPU does not matter: it only copies files and installs wheels)."""
        dc, rows = self.volume["dataCenterId"], []
        for flavor in self.sp["seeder"]["flavors"]:
            for v in sorted(int(x) for x in self.sp["seeder"]["vcpu"]):
                s = self.rp.stock(flavor, v, rmod.ram_gb(self.policy, v), dc)
                if s.get("stockStatus") in rmod.STOCKED and s.get("securePrice"):
                    rows.append(
                        {
                            "flavor": flavor,
                            "vcpu": v,
                            "dc": dc,
                            "stock": s["stockStatus"],
                            "usd_per_hr": float(s["securePrice"]),
                        }
                    )
        return sorted(rows, key=lambda r: r["usd_per_hr"])

    def prepare_uploads(self, workdir: Path) -> Dict[str, Any]:
        commit = self.job["repo_commit"]
        problems = jobspec.check_commit(self.repo, self.job)
        if problems:
            raise jobspec.JobError("; ".join(problems))
        self.repo_archive = Path(workdir) / "repo.tar.gz"
        self.repo_sha = jobspec.git_archive(self.repo, commit, self.repo_archive)
        self.ckpt_paths = jobspec.resolve_checkpoints(
            self.job["checkpoints"], self.policy, self.allowlist
        )
        return {
            "volume": self.volume,
            "repo": {
                "commit": commit,
                "kind": "git archive (tracked files only)",
                "sha256": self.repo_sha,
                "bytes": self.repo_archive.stat().st_size,
            },
            "checkpoints": [
                {"sha256": sha, "path": self.allowlist[sha]["path"], "bytes": p.stat().st_size}
                for sha, p in sorted(self.ckpt_paths.items())
            ],
            "handler": {"runtime_id": self.rid, "sha256": handler_sha256()},
            "venvs": runtime_spec(self.fp, self.sp),
            "nothing_else": "no untracked files, artifacts, scores.db, keys or venv",
        }

    def tick_seed(self, pod, agent) -> None:
        h = agent.get_json("/health")
        pod.cpu_model = h.get("cpu_model")
        if pod.state == "booting":
            pod.state = "ready"
            self.log(
                "seeder_ready",
                pod=pod.id,
                health={
                    k: h.get(k)
                    for k in (
                        "python",
                        "cpu_model",
                        "root_exists",
                        "free_bytes",
                        "self_delete_capable",
                    )
                },
            )
        inv = h.get("inventory") or {}
        commit = self.job["repo_commit"]
        if self.stage == "boot":
            if (inv.get("commits") or {}).get(commit) != self.repo_sha:
                got = agent.put_file(f"/in/commit/{commit}.tar.gz", self.repo_archive)
                if got.get("sha256") != self.repo_sha:
                    raise Abort("archive upload sha mismatch")
                out = agent.post_json("/commit", {"commit": commit, "repo_sha256": self.repo_sha})
                self.log("commit_published", out=out)
            for sha, path in sorted(self.ckpt_paths.items()):
                if sha not in (inv.get("ckpts") or []):
                    got = agent.put_file(f"/in/ckpt/{sha}.pth", path)
                    if got.get("sha256") != sha:
                        raise Abort("checkpoint upload sha mismatch")
            rt = (inv.get("runtimes") or {}).get(self.rid)
            if not rt:
                got = agent.put_file(f"/in/handler/{self.rid}", HANDLER_SOURCE)
                if got.get("sha256") != handler_sha256():
                    raise Abort("handler upload sha mismatch")
            out = agent.post_json(
                "/runtime", {"runtime_id": self.rid, "handler_sha256": handler_sha256()}
            )
            self.log("runtime_build", out=out)
            self.stage = "building"
            return
        if self.stage == "building":
            state = (h.get("build") or {}).get(self.rid)
            if state and state.startswith("failed"):
                raise Abort(f"runtime build failed: {state}")
            if state != "ready":
                return
            rep = agent.post_json("/verify", {})
            inv = rep.get("inventory") or {}
            ready = (inv.get("runtimes") or {}).get(self.rid) or {}
            spec = runtime_spec(self.fp, self.sp)
            problems = list(rep.get("bad") or [])
            if (inv.get("commits") or {}).get(commit) != self.repo_sha:
                problems.append("commit missing after seeding")
            problems += [
                f"ckpt {s[:12]} missing"
                for s in self.ckpt_paths
                if s not in (inv.get("ckpts") or [])
            ]
            if ready.get("handler_sha256") != handler_sha256():
                problems.append("runtime handler sha differs")
            for k, want in spec["expect"].items():
                if ready.get(k) != want:
                    problems.append(f"runtime {k} {ready.get(k)} != {want}")
            if problems:
                raise Abort("volume verification failed: " + "; ".join(problems))
            utc = rmod.utc_now()
            with self.registry.locked() as reg:
                if reg.get("volume") and reg["volume"].get("id") != self.volume["id"]:
                    raise Abort("seeded a volume other than the registered one")
                reg["volume"] = reg.get("volume") or dict(self.volume)
                reg["commits"][commit] = {"repo_sha256": self.repo_sha, "seeded_utc": utc}
                for sha in self.ckpt_paths:
                    reg["ckpts"].setdefault(sha, utc)
                old = reg["runtimes"].get(self.rid) or {}
                reg["runtimes"][self.rid] = {**old, "ready": ready, "image": self.sp["image"]}
            self.log("seeded", verify_checked=rep.get("checked"), runtime=ready)
            self.stage = "done"
            self.seed_done = True
            self.completed["seed"] = self.repo_sha

    def loop(self) -> None:
        startup = float(self.policy["startup_deadline_seconds"])
        build_deadline = float(self.policy["ready_deadline_seconds"]) + 1800
        while not self.seed_done:
            now = self.clock()
            if now >= self.deadline:
                raise Abort("seeding did not finish before its max wall time")
            self.check_watchdog()
            self.check_pending_creates()
            live = [p for p in self.pods.values() if p.deleted is None]
            if not live:
                if now - self.last_capacity_check >= 120:
                    self.last_capacity_check = now
                    rows = self.probe()
                    self.log("capacity_probe", stocked=rows)
                    for row in rows:
                        if self.launch_allowed(row["usd_per_hr"]) is None and self.create_pod(row):
                            break
            for pod in live:
                try:
                    self.tick_seed(pod, self.agents[pod.id])
                    pod.last_ok = now
                except rmod.NET_ERRORS as exc:
                    code = getattr(exc, "code", None)
                    if isinstance(code, int) and 400 <= code < 500 and code != 404:
                        raise Abort(f"seeder refused a step ({code}): {str(exc)[:160]}")
                    if pod.state == "booting" and now - pod.created > startup:
                        raise Abort(
                            f"seeder pod {pod.id} unreachable before the startup "
                            f"deadline: {str(exc)[:120]}"
                        )
                    if pod.state == "ready" and now - pod.last_ok > 300:
                        raise Abort(f"seeder pod {pod.id} lost: {str(exc)[:120]}")
                if self.stage == "building" and now - pod.created > build_deadline:
                    raise Abort("runtime build too slow")
            self.write_state()
            if not self.seed_done:
                self.sleep(10)

    def unit_summary(self) -> Dict[str, Any]:
        return {"seed": {"runtime_id": self.rid, "stage": self.stage}}

    def cost_plan(self, sizes_rows=None) -> Dict[str, Any]:
        return {
            "cost": {
                "budget_usd_hard_cap": self.budget,
                "seeder": dict(self.sp["seeder"]),
                "note": "one seeder pod; worst = its rate to its reservation horizon",
            }
        }

    def run(self) -> int:
        """Runner.run's skeleton: register, watchdog, loop, always delete the pod. The one
        "episode" is the seeding itself, so an aborted seed is not 'complete' in the ledger.
        One seeder at a time (a local lock): two would race on a shared runtime build."""
        self.episodes, self.order, self.pending, self.units = {}, ["seed"], [], {}
        self.registry.root.mkdir(parents=True, exist_ok=True)
        with (self.registry.root / "seed.lock").open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("refusing: another seeder is running", file=sys.stderr)
                return 2
            try:
                return super().run()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- planning / CLI


def account_headroom(rp: Any, fp: Mapping[str, Any]) -> Dict[str, Any]:
    balance = float(rp.balance())
    peek = SharedLedger(fp).peek()
    head = min(
        balance - float(fp["global_floor_usd"]) - peek["reserved_usd"],
        float(fp["project_cap_usd"]) - peek["spent_usd"] - peek["reserved_usd"],
    )
    return {"balance_usd": balance, "ledger": peek, "headroom_usd": round(head, 3)}


def plan_serverless(
    job: Mapping[str, Any],
    fp: Mapping[str, Any],
    sp: Mapping[str, Any],
    allow: Mapping[str, Any],
    budget: float,
    rp: Any = None,
    flavor: Optional[str] = None,
    target_minutes: Optional[float] = None,
    workers: Optional[int] = None,
    vcpu: Optional[int] = None,
    repo: Path = REPO,
) -> Dict[str, Any]:
    rp = rp or RpClient()
    problems = jobspec.check_commit(repo, job)
    work = Path(tempfile.mkdtemp(prefix="rpf-sls-plan-"))
    try:
        repo_sha = jobspec.git_archive(repo, job["repo_commit"], work / "repo.tar.gz")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    jobspec.resolve_checkpoints(job["checkpoints"], fp, allow)
    rid = runtime_id(fp, sp)
    reg = Registry(fp).read()
    seeded = seed_problems(reg, job["repo_commit"], repo_sha, job["checkpoints"], rid)
    acct = account_headroom(rp, fp)
    flavors = [flavor] if flavor else list(sp["flavors_pref"])
    sizing = sizing_plan(
        fp,
        sp,
        job["episodes"],
        budget,
        float(job["max_wall_minutes"]),
        flavors,
        acct["headroom_usd"],
        target_minutes,
        workers,
        vcpu,
    )
    choice = sizing["choice"] or {"workers": 1, "vcpu_per_worker": 2}
    vol = reg.get("volume") or {"id": "<volume id>", "dataCenterId": sp["data_center"]}
    tmpl = ((reg.get("runtimes") or {}).get(rid) or {}).get("template_id") or "<template id>"
    slots = max(1, int(choice["vcpu_per_worker"]) // int(fp["threads_per_episode"]))
    body = endpoint_body(
        sp,
        f"{sp['endpoint_prefix']}{job['job_id']}--sls-<stamp>-1",
        tmpl,
        vol,
        flavors,
        choice["workers"],
        choice["vcpu_per_worker"],
        4 * max(unit_times(fp, job["episodes"], slots)) + 300,
    )
    gb = int((reg.get("volume") or {}).get("size") or sp["volume_size_gb"])
    endpoints = []
    try:
        endpoints = [
            {"id": e.get("id"), "name": e.get("name"), "workersMax": e.get("workersMax")}
            for e in rp.list_endpoints()
            if str(e.get("name", "")).startswith(sp["endpoint_prefix"])
        ]
    except RunPodError as exc:
        endpoints = [{"error": str(exc)[:120]}]
    return {
        "dry_run": True,
        "backend": "serverless",
        "job": jobspec.summarize_job(job),
        "commit_problems": problems,
        "seeding": {
            "runtime_id": rid,
            "repo_sha256": repo_sha,
            "ready": not seeded,
            "problems": seeded,
        },
        "flavors_in_order": flavors,
        "sizing": sizing,
        "endpoint_body_dry_run": body,
        "account": acct,
        "storage": {"volume_gb": gb, "usd_per_month": round(gb * sp["volume_usd_per_gb_month"], 3)},
        "existing_runner_endpoints": endpoints,
        "budget_usd": budget,
    }


def run_serverless(a, job, fp, allow, run_dir: Path) -> int:
    sp = load_sls_policy()
    if a.flavor and a.flavor not in sp["flavors_pref"]:
        print(f"--flavor must be one of {sp['flavors_pref']}", file=sys.stderr)
        return 2
    rp = RpClient()
    acct = account_headroom(rp, fp)
    flavors = [a.flavor] if a.flavor else list(sp["flavors_pref"])
    episodes = job["episodes"]
    if a.resume_from:  # size for the units still to run, not the whole job
        probe = rmod.Runner(job, fp, allow, run_dir, rp=rmod._NoRunPod(), resume_from=a.resume_from)
        episodes = [probe.episodes[k] for k in probe.order] or episodes
    plan = sizing_plan(
        fp,
        sp,
        episodes,
        a.budget,
        float(job["max_wall_minutes"]),
        flavors,
        acct["headroom_usd"],
        a.target_minutes,
        a.workers,
        a.vcpu_per_worker,
    )
    if not plan["choice_ok"]:
        print(
            json.dumps({"refused": "no serverless size fits", "choice": plan["choice"]}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"serverless_choice": plan["choice"], "flavors": flavors}), flush=True)
    r = ServerlessRunner(
        job,
        fp,
        allow,
        run_dir,
        sls_policy=sp,
        sizing=plan["choice"],
        flavors=flavors,
        rp=rp,
        budget=a.budget,
        confirm=True,
        preflight_fn=lambda: rmod.preflight(SLS_PREFLIGHT_URLS),
        resume_from=a.resume_from,
    )
    return r.run()


SLS_PREFLIGHT_URLS = (
    "https://rest.runpod.io/v1/endpoints",
    "https://api.runpod.io/graphql",
    "https://api.runpod.ai/v2/",
)


def _volume_from_api(rp: Any, volume_id: str, sp: Mapping[str, Any]) -> Dict[str, Any]:
    vol = rp.get_volume(volume_id)
    if not vol:
        raise SystemExit(f"network volume {volume_id} not found")
    if vol.get("dataCenterId") != sp["data_center"]:
        raise SystemExit(
            f"volume is in {vol.get('dataCenterId')}, policy wants {sp['data_center']}"
        )
    return {k: vol.get(k) for k in ("id", "name", "size", "dataCenterId")}


def cmd_volume_create(a, fp, sp) -> int:
    rp, reg = RpClient(), Registry(fp)
    if reg.read().get("volume"):
        print("refusing: a volume is already registered", file=sys.stderr)
        return 2
    body = {
        "name": sp["volume_name"],
        "size": int(sp["volume_size_gb"]),
        "dataCenterId": sp["data_center"],
    }
    print(
        json.dumps(
            {
                "cost_usd_per_month": round(body["size"] * sp["volume_usd_per_gb_month"], 3),
                "body": body,
            }
        )
    )
    out = rp.create_volume(body, confirm=a.confirm)
    print(json.dumps(out, indent=1))
    if a.confirm and isinstance(out, dict) and out.get("id"):
        with reg.locked() as state:
            state["volume"] = {
                "id": out["id"],
                "name": out.get("name"),
                "size": out.get("size"),
                "dataCenterId": out.get("dataCenterId"),
            }
    return 0


def cmd_seed(a, fp, sp) -> int:
    allow = jobspec.load_allowlist()
    rp = RpClient()
    reg = Registry(fp).read()
    registered = (reg.get("volume") or {}).get("id")
    if a.volume_id and registered and a.volume_id != registered:
        print(f"refusing: volume {registered} is registered, not {a.volume_id}", file=sys.stderr)
        return 2
    vol_id = a.volume_id or registered
    if not vol_id and a.confirm:
        print("no volume (serverless.py volume-create first, or --volume-id)", file=sys.stderr)
        return 2
    volume = (
        _volume_from_api(rp, vol_id, sp)
        if vol_id
        else {"id": "<volume id>", "dataCenterId": sp["data_center"]}
    )
    commit = jobspec.git(REPO, "rev-parse", a.commit).stdout.strip()
    ckpts = sorted(allow)  # the 4 allow-listed checkpoints, nothing else
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    probe = SeedRunner(
        fp,
        sp,
        allow,
        Path(tempfile.gettempdir()) / "rpf-seed-plan-unused",
        commit=commit,
        volume=volume,
        ckpts=ckpts,
        rp=rp,
    )
    base = Path(fp["artifacts_root"]) / "runpod-fanout" / probe.job["job_id"]
    if not a.confirm:
        work = Path(tempfile.mkdtemp(prefix="rpf-seed-plan-"))
        try:
            manifest = probe.prepare_uploads(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rows = probe.probe()
        rate = rows[0]["usd_per_hr"] if rows else None
        life_h = (60 * float(sp["seeder"]["max_wall_minutes"]) + 900) / 3600
        body = seeder_pod_body(
            fp,
            sp,
            f"rpf-{probe.job['job_id']}--seed-{stamp}-1",
            volume,
            "<sha256 of a fresh token>",
            time.time() + life_h * 3600,
            probe.job["job_id"],
            flavor=rows[0]["flavor"] if rows else None,
            vcpu=rows[0]["vcpu"] if rows else None,
        )
        body["dockerStartCmd"] = [
            "bash",
            "-c",
            "<seed_agent.py, gzip+base64, "
            f"{len(start_command(SEED_AGENT_SOURCE, '/r/s.py')[2])} chars>",
        ]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "job_id": probe.job["job_id"],
                    "stock": rows,
                    "pod_body": body,
                    "uploads": manifest,
                    "cost": {
                        "usd_per_hr": rate,
                        "expected_usd": None if rate is None else round(rate * 0.33, 4),
                        "worst_case_usd": None if rate is None else round(1.02 * rate * life_h, 4),
                        "worst_case_cap_usd": round(
                            1.02 * float(sp["seeder"]["max_hourly_usd"]) * life_h, 4
                        ),
                        "note": "expected ~20 min (venv build); worst = pod lifetime cap",
                    },
                    "budget_needed": (
                        None if rate is None else round(1.02 * rate * life_h + 0.01, 3)
                    ),
                },
                indent=1,
            )
        )
        return 0
    budget = float(a.budget or 0.2)
    r = SeedRunner(
        fp,
        sp,
        allow,
        base / f"seed-{stamp}",
        commit=commit,
        volume=volume,
        ckpts=ckpts,
        rp=rp,
        budget=budget,
        confirm=True,
    )
    return r.run()


def cmd_template_create(a, fp, sp) -> int:
    rp, reg = RpClient(), Registry(fp)
    rid = runtime_id(fp, sp)
    rt = (reg.read().get("runtimes") or {}).get(rid) or {}
    if not rt.get("ready"):
        print(
            f"runtime {rid} is not seeded yet (serverless.py seed): --confirm is refused "
            "until then; the body would be:",
            file=sys.stderr,
        )
        print(json.dumps(template_body(sp, rid), indent=1))
        return 2 if a.confirm else 0
    if rt.get("template_id"):
        print(f"template already exists: {rt['template_id']}")
        return 0
    body = template_body(sp, rid)
    path = reg.root / f"template-{rid}.request.json"
    reg.root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    print(json.dumps({"cost": "templates are free; only endpoints bill", "body": body}, indent=1))
    out = rp.create_template(path, confirm=a.confirm)
    print(json.dumps(out, indent=1))
    if a.confirm and isinstance(out, dict) and out.get("id"):
        with reg.locked() as state:
            state["runtimes"][rid]["template_id"] = out["id"]
    return 0


def cmd_endpoint_cleanup(a, fp, sp) -> int:
    rp = RpClient()
    if not a.job_id and not a.all_runner_endpoints:
        print("need --job-id ID or --all-runner-endpoints", file=sys.stderr)
        return 2
    prefix = f"{sp['endpoint_prefix']}{a.job_id}--" if a.job_id else sp["endpoint_prefix"]
    peek = SharedLedger(fp).peek()
    live = [j for j in peek["live_jobs"] if not a.job_id or j == a.job_id]
    if live and a.confirm and not a.force:
        print(f"refusing: live runs could lose endpoints: {live} (use --force)", file=sys.stderr)
        return 2
    eps = [e for e in rp.list_endpoints() if owned_endpoint(prefix, e.get("name"))]
    print(
        json.dumps(
            {
                "prefix": prefix,
                "confirm": a.confirm,
                "endpoints": [
                    {"id": e["id"], "name": e["name"], "workersMax": e.get("workersMax")}
                    for e in eps
                ],
            }
        )
    )
    if not a.confirm:
        return 0
    left = [e["id"] for e in eps if not teardown_endpoint(rp, e["id"])]
    print(json.dumps({"left": left}))
    return 0 if not left else 5


def cmd_volume_cleanup(a, fp, sp) -> int:
    rp, reg = RpClient(), Registry(fp)
    vol = (reg.read().get("volume") or {}).get("id") or a.volume_id
    if not vol:
        print("no registered volume", file=sys.stderr)
        return 2
    attached = [
        e.get("id")
        for e in rp.list_endpoints()
        if e.get("networkVolumeId") == vol or vol in (e.get("networkVolumeIds") or [])
    ]
    pods = [p.get("id") for p in rp.list_pods() if p.get("networkVolumeId") == vol]
    print(
        json.dumps(
            {
                "volume": vol,
                "attached_endpoints": attached,
                "attached_pods": pods,
                "confirm": a.confirm,
                "warning": "permanent: deletes every seeded "
                "commit, checkpoint and runtime on it",
            }
        )
    )
    if attached or pods:
        print("refusing: endpoints/pods still use the volume", file=sys.stderr)
        return 2
    out = rp.delete_volume(vol, confirm=a.confirm)
    print(json.dumps(out))
    if a.confirm:
        with reg.locked() as state:
            state.update(
                {
                    "volume": None,
                    "commits": {},
                    "ckpts": {},
                    "runtimes": {k: {**v, "ready": None} for k, v in state["runtimes"].items()},
                }
            )
    return 0


def cmd_settle(a, fp, sp) -> int:
    """Lower released serverless ledger rows to final billing (reads billing, writes only
    the local ledger; runs also do this in their periodic reconcile)."""
    done = SharedLedger(fp).settle_serverless(RpClient(), min_age_seconds=a.min_age_seconds)
    print(json.dumps({"settled": done}))
    return 0


def cmd_status(a, fp, sp) -> int:
    rp = RpClient()
    out = {"registry": Registry(fp).read(), "runtime_id_current": runtime_id(fp, sp)}
    for name, fn in (
        ("endpoints", rp.list_endpoints),
        ("volumes", rp.list_volumes),
        ("templates", rp.list_templates),
    ):
        try:
            rows = fn()
            out[name] = [
                {
                    k: r.get(k)
                    for k in ("id", "name", "workersMax", "dataCenterId", "size", "networkVolumeId")
                }
                for r in rows
                if str(r.get("name", "")).startswith("rpf-")
            ]
        except RunPodError as exc:
            out[name] = str(exc)[:160]
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    vc = sub.add_parser("volume-create")
    vc.add_argument("--confirm", action="store_true")
    sd = sub.add_parser("seed")
    sd.add_argument("--commit", default="HEAD")
    sd.add_argument("--volume-id")
    sd.add_argument("--budget", type=float, default=0.2)
    sd.add_argument("--confirm", action="store_true")
    tc = sub.add_parser("template-create")
    tc.add_argument("--confirm", action="store_true")
    ec = sub.add_parser("endpoint-cleanup")
    ec.add_argument("--job-id")
    ec.add_argument("--all-runner-endpoints", action="store_true")
    ec.add_argument("--confirm", action="store_true")
    ec.add_argument("--force", action="store_true")
    vd = sub.add_parser("volume-cleanup")
    vd.add_argument("--volume-id")
    vd.add_argument("--confirm", action="store_true")
    sub.add_parser("status")
    st = sub.add_parser("settle")
    st.add_argument("--min-age-seconds", type=float, default=7200.0)
    a = p.parse_args(argv)
    fp, sp = jobspec.load_policy(), load_sls_policy()
    return {
        "volume-create": cmd_volume_create,
        "seed": cmd_seed,
        "template-create": cmd_template_create,
        "endpoint-cleanup": cmd_endpoint_cleanup,
        "volume-cleanup": cmd_volume_cleanup,
        "settle": cmd_settle,
        "status": cmd_status,
    }[a.cmd](a, fp, sp)


if __name__ == "__main__":
    raise SystemExit(main())
