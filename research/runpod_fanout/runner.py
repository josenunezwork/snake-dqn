#!/usr/bin/env python3
"""RunPod fan-out runner for Tier-1 episodes (screens, sweeps, censuses). NOT for gates.

Commands (run from the repo with the project venv)::

    runner.py plan    JOB.json [--budget USD]            # dry run (serverless default; read-only):
                                                         #   sizing / speed-cost table, seed state
    runner.py run     JOB.json --budget USD --confirm    # real RunPod run (spends money); default
                                                         #   backend serverless (serverless.py)
    runner.py run     JOB.json --backend pods ...        # CPU pods (explicit fallback)
    runner.py run     JOB.json --backend local           # same episodes on this Mac, 1 slot
    runner.py status  [RUN_DIR]                          # run state + live runner pods
    runner.py cleanup --job-id ID [--confirm]            # delete ONLY rpf-<ID>-- pods
    runner.py compare RUN_DIR_A RUN_DIR_B                # per-episode deterministic equality
    runner.py merge   RUN_DIR... --out DIR               # one records tree; mixed platforms refused

A run writes create-only to ``<artifacts>/runpod-fanout/<job_id>/<run-name>/``:
``job.json``, ``plan.json``, ``events.jsonl``, ``state.json`` (rewritten), ``receipt.json``
and ``records/<wrapper>__h<H>__<engine>/<arm>-<mix>-<seed>.json`` (screen entry format plus
``platform`` and ``fanout`` stamps). See docs/research/runpod_fanout_runner_2026-10-03.md.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import gzip
import hashlib
import http.client
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec, platform_rule  # noqa: E402
from research.runpod_fanout.rp_client import RpClient, RunPodError  # noqa: E402

HERE = Path(__file__).resolve().parent
AGENT_SOURCE = HERE / "pod_agent.py"
from research.runpod_fanout.ledger import DuplicateRun, SharedLedger  # noqa: E402
from research.runpod_fanout.tls import USER_AGENT, preflight, ssl_context  # noqa: E402

STOCKED = ("High", "Medium", "Low")
RECEIPT_SCHEMA = "runpod-fanout-receipt/v1"
BACKENDS = ("serverless", "pods", "local", "runpod")  # runpod = pods (old name)
EXIT_DUPLICATE = 6  # job already has a live or successful run: refused, nothing spent
NET_ERRORS = (urllib.error.URLError, OSError, ValueError, http.client.HTTPException)


class Abort(RuntimeError):
    """Stop the run now (divergent duplicate, budget breach, ...); pods are deleted."""


def spec_sha256(ep: Mapping[str, Any]) -> str:
    """The episode spec hash stamped by episode.py (``fanout.spec_sha256``)."""
    return hashlib.sha256(
        json.dumps(ep, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parallelism_plan(
    policy: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    budget: float,
    max_wall_minutes: float,
    stock_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    target_wall_minutes: float = 120.0,
) -> Dict[str, Any]:
    """Effective parallelism the budget/stock/caps allow, expected wall time and cost.

    Model: V vCPUs run continuously (V/2 workers); wall = setup x (1 + replacements) +
    max(longest episode, total episode-seconds / workers). The budget must cover what is
    spent by the end plus the last reservations: 1.02 x rate x V x (wall + horizon), with
    horizon = min(max wall, pod max lifetime + grace). Stock: one pod per stocked
    (size, data center) row, at most max_pods (an estimate; stock changes by the minute).
    """
    secs = [episode_seconds(policy, ep) for ep in episodes]
    total, longest = sum(secs), max(secs)
    per_vcpu = 0.03
    if stock_rows:
        per_vcpu = max(float(r["usd_per_hr"]) / int(r["vcpu"]) for r in stock_rows)
    life = float(policy["pod_max_lifetime_seconds"])
    horizon_h = min(
        max_wall_minutes / 60.0, (life + float(policy["pod_delete_grace_seconds"])) / 3600
    )
    setup = float(policy["pod_setup_seconds"])
    cap_v = int(policy["max_pods"]) * max(policy["vcpu_sizes_desc"])
    cap_v = min(cap_v, int(policy["max_live_pods"]) * max(policy["vcpu_sizes_desc"]))
    stock_v = None
    if stock_rows is not None:
        stock_v = sum(int(r["vcpu"]) for r in list(stock_rows)[: int(policy["max_pods"])])

    def wall_s(v: int) -> float:
        work = max(longest, total / max(1, v // int(policy["threads_per_episode"])))
        return setup * (1 + math.floor(work / life)) + work

    def worst(v: int) -> float:
        return 1.02 * per_vcpu * v * (wall_s(v) / 3600 + horizon_h)

    def expected(v: int) -> float:
        return per_vcpu * v * wall_s(v) / 3600

    vs = list(range(2, cap_v + 1, 2))
    v_budget = max([v for v in vs if worst(v) <= budget], default=0)
    v_eff = min(x for x in (v_budget, stock_v if stock_v is not None else cap_v, cap_v))
    v_eff -= v_eff % 2

    def need_for(minutes: float) -> Optional[Dict[str, Any]]:
        ok = [v for v in vs if wall_s(v) <= minutes * 60]
        if not ok:
            return None
        v = ok[0]
        return {
            "vcpu": v,
            "budget_usd": round(worst(v), 2),
            "expected_usd": round(expected(v), 2),
            "wall_minutes": round(wall_s(v) / 60, 1),
        }

    return {
        "pod_episode_seconds_total_est": round(total, 1),
        "longest_episode_seconds_est": round(longest, 1),
        "usd_per_vcpu_hr": per_vcpu,
        "reservation_horizon_hours": round(horizon_h, 3),
        "parallelism": {
            "vcpu_allowed_by_budget": v_budget,
            "vcpu_allowed_by_stock_est": stock_v,
            "vcpu_allowed_by_caps": cap_v,
            "effective_vcpu": v_eff,
            "effective_workers": v_eff // int(policy["threads_per_episode"]),
            "expected_wall_minutes": round(wall_s(v_eff) / 60, 1) if v_eff else None,
            "expected_cost_usd": round(expected(v_eff), 3) if v_eff else None,
            "fits_max_wall": bool(v_eff) and wall_s(v_eff) <= max_wall_minutes * 60,
        },
        "budget_for_max_wall": need_for(max_wall_minutes),
        f"budget_for_{int(target_wall_minutes)}_min": need_for(target_wall_minutes),
    }


def owned_name(prefix: str, name: Any) -> bool:
    """``name`` is exactly ``prefix`` + a pod counter (so run ``a`` never matches ``a-2``)."""
    name = str(name or "")
    return name.startswith(prefix) and name[len(prefix) :].isdigit()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pod_prefix(policy: Mapping[str, Any], job_id: str) -> str:
    return f"{policy['pod_name_prefix']}{job_id}--"


def episode_seconds(policy: Mapping[str, Any], ep: Mapping[str, Any]) -> float:
    t = policy["pod_seconds_per_episode"]
    eng = ep["engine"]
    return float(t[f"{eng}_fixed"]) + float(t[f"{eng}_per_frame"]) * int(ep["horizon"])


def numerics_env(policy: Mapping[str, Any]) -> Dict[str, str]:
    """Worker numerics for every backend: fixed OMP/MKL threads plus the policy's
    ``numerics_env`` (MKL_CBWR=COMPATIBLE: the same MKL code path on AVX2 and AVX-512
    hosts); an ``isa_cap`` overrides MKL_CBWR and caps oneDNN as before."""
    threads = str(policy["threads_per_episode"])
    env = {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
    env.update({str(k): str(v) for k, v in (policy.get("numerics_env") or {}).items()})
    isa = policy.get("isa_cap")
    if isa:
        env.update({"ONEDNN_MAX_CPU_ISA": isa, "MKL_ENABLE_INSTRUCTIONS": isa, "MKL_CBWR": isa})
    return env


def ram_gb(policy: Mapping[str, Any], vcpu: int) -> int:
    return int(vcpu) * int(policy["ram_gb_per_vcpu"])


def workers_for(policy: Mapping[str, Any], vcpu: int) -> int:
    return max(1, int(vcpu) // int(policy["threads_per_episode"]))


def size_cap_for(policy: Mapping[str, Any], outstanding: int) -> int:
    """Smallest allowed size whose workers cover ``outstanding`` episodes (else biggest)."""
    for size in sorted(policy["vcpu_sizes_desc"]):
        if workers_for(policy, size) >= outstanding:
            return size
    return max(policy["vcpu_sizes_desc"])


# ---------------------------------------------------------------- agent channel


class AgentClient:
    """HTTPS to the pod agent through RunPod's proxy (token in a header, never in URLs)."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    def _req(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        ctype: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("X-Fanout-Token", self._token)
        if ctype:
            req.add_header("Content-Type", ctype)
        with urllib.request.urlopen(
            req, timeout=timeout or self.timeout, context=ssl_context()
        ) as resp:
            return resp.read()

    def get_json(self, path: str) -> Any:
        return json.loads(self._req("GET", path))

    def get_bytes(self, path: str) -> bytes:
        return self._req("GET", path, timeout=120)

    def post_json(self, path: str, body: Mapping[str, Any]) -> Any:
        return json.loads(self._req("POST", path, json.dumps(body).encode(), "application/json"))

    def put_file(self, path: str, file_path: Path) -> Any:
        data = Path(file_path).read_bytes()
        return json.loads(self._req("PUT", path, data, "application/octet-stream", timeout=900))


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


# ---------------------------------------------------------------- pod body


def agent_start_command() -> List[str]:
    blob = base64.b64encode(gzip.compress(AGENT_SOURCE.read_bytes(), mtime=0)).decode()
    script = (
        "mkdir -p /r && echo " + blob + " | base64 -d | gunzip > /r/agent.py && "
        "exec python /r/agent.py"
    )
    return ["bash", "-c", script]


def pod_body(
    policy: Mapping[str, Any],
    name: str,
    vcpu: int,
    dc: str,
    token_sha: str,
    deadline_epoch: float,
    job_id: str,
    until_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """``deadline_epoch``: the agent stops work; ``until_epoch``: the pod deletes itself
    (pod-scoped RunPod credentials) if the runner and watchdog have not by then."""
    until_epoch = until_epoch or (deadline_epoch + float(policy["watchdog_grace_seconds"]) + 240)
    pip = [
        ["pip", "install", "-q", "--no-cache-dir", *policy["pip_pins"]],
        [
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            policy["torch_pin"],
            "--index-url",
            policy["torch_index_url"],
        ],
    ]
    return {
        "name": name,
        "computeType": "CPU",
        "cpuFlavorIds": [policy["flavor"]],
        "vcpuCount": int(vcpu),
        "cloudType": policy["cloud"],
        "dataCenterIds": [dc],
        "imageName": policy["image"],
        "containerDiskInGb": int(policy["container_disk_gb"]),
        "volumeInGb": 0,
        "ports": [f"{policy['agent_port']}/http"],
        "env": {
            "FANOUT_TOKEN_SHA256": token_sha,
            "FANOUT_DEADLINE_EPOCH": str(int(deadline_epoch)),
            "FANOUT_PROVIDER": f"runpod:{policy['flavor']}",
            "FANOUT_PIP_JSON": json.dumps(pip),
            "FANOUT_THREADS": str(policy["threads_per_episode"]),
            "FANOUT_WORKERS": str(workers_for(policy, vcpu)),
            "FANOUT_ISA_CAP": policy["isa_cap"] or "",
            "FANOUT_SELF_DELETE_EPOCH": str(int(until_epoch)),
            "FANOUT_JOB": job_id,
            "FANOUT_NUMERICS_JSON": json.dumps(numerics_env(policy), sort_keys=True),
        },
        "dockerStartCmd": agent_start_command(),
    }


# ---------------------------------------------------------------- run state


@dataclass
class Pod:
    id: str
    name: str
    vcpu: int
    dc: str
    rate: float
    token: str
    created: float
    state: str = "booting"  # booting -> ready -> draining -> deleted | lost
    slots: int = 0
    boot_id: Optional[str] = None
    last_ok: float = 0.0
    last_pull: float = 0.0
    inflight: set = field(default_factory=set)  # assigned, not yet done on the pod
    unpulled: set = field(default_factory=set)  # done on the pod, record not yet pulled
    deleted: Optional[float] = None
    retire_pending: bool = False
    boot_from: float = 0.0  # start of the current startup window (create or agent restart)
    first_fail: Optional[float] = None
    cpu_model: Optional[str] = None
    until: float = 0.0  # reservation horizon: created + max lifetime + delete grace (<= hard end)
    note: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vcpu": self.vcpu,
            "dc": self.dc,
            "usd_per_hr": self.rate,
            "created": self.created,
            "state": self.state,
            "slots": self.slots,
            "boot_id": self.boot_id,
            "inflight": sorted(self.inflight),
            "unpulled": sorted(self.unpulled),
            "deleted": self.deleted,
            "note": self.note,
        }

    def cost_until(self, t: float) -> float:
        end = self.deleted if self.deleted is not None else t
        return self.rate * max(0.0, end - self.created) / 3600.0


class Runner:
    def __init__(
        self,
        job: Mapping[str, Any],
        policy: Mapping[str, Any],
        allowlist: Mapping[str, Mapping[str, str]],
        run_dir: Path,
        *,
        rp: Any = None,
        budget: float = 0.0,
        confirm: bool = False,
        repo: Path = REPO,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        agent_factory: Optional[Callable[[str, str], Any]] = None,
        spawn_watchdog: Optional[Callable[["Runner"], Any]] = None,
        probe_workers: int = 8,
        preflight_fn: Optional[Callable[[], List[str]]] = None,
        ledger: Optional[SharedLedger] = None,
        resume_from: Sequence[Path] = (),
    ):
        self.job = job
        self.policy = policy
        self.allowlist = allowlist
        self.run_dir = Path(run_dir)
        self.rp = rp if rp is not None else RpClient()
        self.budget = float(budget)
        self.confirm = bool(confirm)
        self.repo = Path(repo)
        self.clock = clock
        self.sleep = sleep
        port = int(policy["agent_port"])
        self.agent_factory = agent_factory or (
            lambda pod_id, token: AgentClient(proxy_url(pod_id, port), token)
        )
        self.spawn_watchdog = spawn_watchdog or spawn_watchdog_process
        self.probe_workers = probe_workers
        self.preflight_fn = preflight_fn or preflight
        self.episodes = {jobspec.episode_key(ep): ep for ep in job["episodes"]}
        self.order = list(self.episodes)
        self.pending: List[str] = list(self.order)
        self.attempts: Dict[str, int] = {k: 0 for k in self.order}
        self.completed: Dict[str, str] = {}  # key -> sha256 of the local record bytes
        self.failed: Dict[str, Any] = {}
        self.pods: Dict[str, Pod] = {}
        self.agents: Dict[str, Any] = {}
        self.bad_slots: Dict[tuple, float] = {}  # (vcpu, dc) -> retry-after epoch
        self.events = None
        self.start = 0.0
        self.deadline = 0.0
        self.hard_end = 0.0
        self.balance_start: Optional[float] = None
        self.last_capacity_check = 0.0
        self.repo_archive: Optional[Path] = None
        self.repo_sha: Optional[str] = None
        self.ckpt_paths: Dict[str, Path] = {}
        self.pod_counter = 0
        self.stop_reason: Optional[str] = None
        self.verified: set = set()  # (key, sha256 on a pod) already accepted
        self.ledger = ledger or SharedLedger(policy, clock=clock)
        self.run_id = f"{self.run_dir.resolve()}#{secrets.token_hex(4)}"
        self.held: Dict[str, tuple] = {}  # failed-create reservations: name -> (rate, until, t0)
        self.pending_creates: Dict[str, Dict[str, Any]] = {}  # name -> check state
        self.free_redispatch: Dict[str, int] = {}
        self.legacy_lock: Any = None
        self.account_refusals = 0
        self.last_reconcile = 0.0
        self._balance: Optional[tuple] = None
        self.platform_ids: set = set()
        self.watchdog: Any = None
        self.cpu_model_pin: Optional[str] = None  # serverless probe only (not a run pin)
        # Per-world units: every arm of one (wrapper, mix, world_seed) runs on ONE worker.
        self.units: Dict[str, List[str]] = platform_rule.group_units(self.order)
        self.unit_src: Dict[str, Optional[str]] = {}  # unit -> source of its current attempt
        self.staged: Dict[str, Dict[str, Dict[str, Any]]] = {}  # unit -> key -> entry
        self.seen: Dict[str, Dict[str, Any]] = {}  # key -> first entry seen (duplicate checks)
        self.record_cache: Dict[tuple, bytes] = {}  # (key, sha) of unpublished units
        self.unit_attempts: Dict[str, int] = {}
        self.unit_free: Dict[str, int] = {}
        self.workers_info: Dict[str, Dict[str, Any]] = {}  # source -> isa flags / id
        self.numerics = numerics_env(policy)
        self.resume_info: Optional[Dict[str, Any]] = None
        if resume_from:
            self.apply_resume([Path(d) for d in resume_from])

    # ------------------------------------------------------------ logging
    def log(self, event: str, **fields: Any) -> None:
        row = {"utc": utc_now(), "event": event, **fields}
        line = json.dumps(row, sort_keys=True, default=str)
        print(line, flush=True)
        if self.events is not None:
            self.events.write(line + "\n")
            self.events.flush()

    def write_state(self) -> None:
        state = {
            "job_id": self.job["job_id"],
            "updated_utc": utc_now(),
            "deadline_epoch": self.deadline,
            "episodes": len(self.order),
            "completed": len(self.completed),
            "pending": len(self.pending),
            "failed": self.failed,
            "pods": [p.public() for p in self.pods.values()],
            "estimated_cost_usd": round(self.estimated_cost(), 4),
            "committed_worst_usd": round(self.committed_worst(), 4),
            "budget_usd": self.budget,
            "stop_reason": self.stop_reason,
            "runner_pid": os.getpid(),
        }
        tmp = self.run_dir / ".state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(tmp, self.run_dir / "state.json")

    # ------------------------------------------------------------ money
    def estimated_cost(self) -> float:
        now = self.clock()
        return sum(p.cost_until(now) for p in self.pods.values())

    def committed_worst(self) -> float:
        """This run's spend if every live pod ran to its reservation horizon, plus slack.

        A pod's horizon is ``min(hard end, created + max lifetime + delete grace)``; the
        runner retires pods at their lifetime and the watchdog/pod self-delete enforce it.
        Slack: 2% and one extra minute per pod (billing starts before our timestamp).
        """
        total = 0.0
        for p in self.pods.values():
            end = p.deleted if p.deleted is not None else (p.until or self.hard_end)
            total += p.rate * (max(0.0, end - p.created) + 60.0) / 3600.0
        for rate, until, t0 in self.held.values():
            total += rate * (max(0.0, until - t0) + 60.0) / 3600.0
        return total * 1.02

    def pod_until(self, now: float) -> float:
        life = float(self.policy["pod_max_lifetime_seconds"])
        grace = float(self.policy["pod_delete_grace_seconds"])
        return min(self.hard_end, now + life + grace)

    def launch_allowed(self, rate: float) -> Optional[str]:
        """None if a new pod at ``rate`` fits the job budget and the global floor."""
        now = self.clock()
        extra = 1.02 * rate * (max(0.0, self.pod_until(now) - now) + 60.0) / 3600.0
        worst = self.committed_worst() + extra
        if worst > self.budget + 1e-9:
            return f"job budget: worst case {worst:.4f} > budget {self.budget:.4f}"
        if rate > float(self.policy["max_hourly_per_pod_usd"]):
            return f"rate {rate} above policy max_hourly_per_pod_usd"
        if len([p for p in self.pods.values() if p.deleted is None]) >= int(
            self.policy["max_pods"]
        ):
            return "max_pods reached"
        if self.hard_end - now < 900:
            return "less than 15 min left before the hard end"
        return None

    # ------------------------------------------------------------ capacity
    def probe(self, sizes: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
        """Read-only stock probe: rows (vcpu, dc, stock, price) for stocked cpu3c sizes."""
        sizes = list(sizes or self.policy["vcpu_sizes_desc"])
        dcs = self.rp.datacenters()
        tasks = [(v, dc) for v in sizes for dc in dcs]
        flavor = self.policy["flavor"]

        def one(task):
            v, dc = task
            try:
                s = self.rp.stock(flavor, v, ram_gb(self.policy, v), dc)
            except RunPodError:
                return None
            if s.get("stockStatus") in STOCKED and s.get("securePrice"):
                return {
                    "vcpu": v,
                    "dc": dc,
                    "stock": s["stockStatus"],
                    "usd_per_hr": float(s["securePrice"]),
                }
            return None

        with concurrent.futures.ThreadPoolExecutor(self.probe_workers) as pool:
            rows = [r for r in pool.map(one, tasks) if r]
        rank = {"High": 0, "Medium": 1, "Low": 2}
        rows.sort(key=lambda r: (-r["vcpu"], rank[r["stock"]], r["dc"]))
        return rows

    def outstanding(self) -> int:
        return len(self.order) - len(self.completed) - len(self.failed)

    def live_slots(self) -> int:
        return sum(
            p.slots or workers_for(self.policy, p.vcpu)
            for p in self.pods.values()
            if p.state in ("booting", "ready")
        )

    def maybe_launch(self, force_probe: bool = False) -> None:
        """Launch the biggest stocked size (capped at what the work needs) if it fits."""
        need = self.outstanding() - self.live_slots()
        if need <= 0 or self.stop_reason:
            return
        self.sweep_orphans()
        cap = size_cap_for(self.policy, need)
        sizes = [v for v in self.policy["vcpu_sizes_desc"] if v <= cap]
        rows = self.probe(sizes)
        self.last_capacity_check = self.clock()
        self.log("capacity_probe", need_workers=need, size_cap=cap, stocked=rows[:12])
        now = self.clock()
        budget_refusals = 0
        for row in rows:
            if self.bad_slots.get((row["vcpu"], row["dc"]), 0) > now:
                continue
            why = self.launch_allowed(row["usd_per_hr"])
            if why:
                self.log("launch_refused_budget", row=row, reason=why)
                budget_refusals += 1
                continue
            if self.create_pod(row):
                return
        no_pods = not any(p.deleted is None for p in self.pods.values())
        if rows and no_pods and budget_refusals == len(rows):
            self.stop_reason = "budget: no stocked size is affordable within the budget/floor"
            self.log("stop", reason=self.stop_reason)

    def account_balance(self) -> float:
        now = self.clock()
        if self._balance is None or now - self._balance[0] > 60:
            self._balance = (now, float(self.rp.balance()))
        return self._balance[1]

    def create_pod(self, row: Mapping[str, Any]) -> bool:
        now = self.clock()
        until = self.pod_until(now)
        self.pod_counter += 1
        name = (
            f"{pod_prefix(self.policy, self.job['job_id'])}{self.run_dir.name}-{self.pod_counter}"
        )
        try:
            balance = self.account_balance()
        except RunPodError as exc:
            self.log("launch_refused_account", row=row, reason=f"balance unavailable: {exc}")
            return False
        try:
            why = self.ledger.reserve_pod(
                self.run_id, name, float(row["usd_per_hr"]), until, balance
            )
        except (OSError, ValueError, KeyError) as exc:
            self.log("launch_refused_account", row=row, reason=f"ledger error: {exc}")
            return False
        if why:
            self.log("launch_refused_account", row=row, reason=why)
            self.account_refusals += 1
            if self.account_refusals % 20 == 1:
                print(
                    f"note: account-level refusal ({why}); waiting for headroom",
                    file=sys.stderr,
                    flush=True,
                )
            return False
        token = secrets.token_urlsafe(32)
        body = self.pod_body_for(name, row, hashlib.sha256(token.encode()).hexdigest(), now, until)
        body_path = self.run_dir / "pods" / f"{name}.request.json"
        body_path.parent.mkdir(exist_ok=True)
        body_path.write_text(json.dumps(body, indent=1))
        _DEFER["active"] = True
        try:
            out = self.rp.create_pod(
                body_path, max_hourly=row["usd_per_hr"] * 1.01, confirm=self.confirm
            )
        except RunPodError as exc:
            _DEFER["active"] = False
            self.bad_slots[(row["vcpu"], row["dc"])] = self.clock() + 600
            self.log(
                "create_refused", vcpu=row["vcpu"], dc=row["dc"], detail=str(exc.payload)[:300]
            )
            status = exc.payload.get("http_status") if isinstance(exc.payload, dict) else None
            if isinstance(status, int) and 400 <= status < 500:
                self.ledger_call("release_pod", self.run_id, name, 0.0)  # definite refusal
            else:  # timeout / 5xx: a pod may exist; keep the reservation until proven absent
                self.hold_unknown_create(name, row, until, now)
            self.raise_deferred()
            return False
        except BaseException:
            _DEFER["active"] = False
            raise
        _DEFER["active"] = False
        if not isinstance(out, dict) or not out.get("id"):
            self.log("create_no_pod", detail=str(out)[:300])
            self.hold_unknown_create(name, row, until, now)
            self.raise_deferred()
            return False
        rate = float(out.get("costPerHr") or row["usd_per_hr"])
        pod = Pod(
            id=out["id"],
            name=name,
            vcpu=int(row["vcpu"]),
            dc=row["dc"],
            rate=rate,
            token=token,
            created=now,
            until=until,
        )
        pod.boot_from = pod.created
        self.pods[pod.id] = pod
        self.agents[pod.id] = self.agent_factory(pod.id, token)
        self.ledger_call("bind_pod", self.run_id, name, pod.id, rate)
        registry = {k: v for k, v in out.items() if k != "env"}
        registry.update({"name": name, "rpf_until_epoch": until})
        (self.run_dir / "pods" / f"{name}.response.json").write_text(json.dumps(registry, indent=1))
        self.log(
            "pod_created",
            pod=pod.id,
            name=name,
            vcpu=pod.vcpu,
            dc=pod.dc,
            usd_per_hr=rate,
            committed_worst=round(self.committed_worst(), 4),
        )
        self.raise_deferred()
        if self.committed_worst() > self.budget + 1e-6:
            raise Abort("committed worst case exceeds the budget after create")
        return True

    def pod_body_for(
        self, name: str, row: Mapping[str, Any], token_sha: str, now: float, until: float
    ) -> Dict[str, Any]:
        """The create body (subclasses, e.g. the serverless seeder, override this)."""
        return pod_body(
            self.policy,
            name,
            row["vcpu"],
            row["dc"],
            token_sha,
            min(self.deadline, now + float(self.policy["pod_max_lifetime_seconds"])),
            self.job["job_id"],
            until_epoch=until,
        )

    def ledger_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Ledger bookkeeping must never stop cleanup: errors are logged (unreleased
        reservations only make the account check more conservative)."""
        try:
            return getattr(self.ledger, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.log("ledger_error", method=method, detail=repr(exc)[:200])
            return None

    def hold_unknown_create(
        self, name: str, row: Mapping[str, Any], until: float, now: float
    ) -> None:
        """A create that may have made a pod: keep its reservation (also in this run's
        committed worst case) until two empty listings >= 60 s apart prove no pod exists;
        a pod that does surface is adopted and deleted (see :meth:`check_pending_creates`)."""
        self.held[name] = (float(row["usd_per_hr"]), until, now)
        self.pending_creates[name] = {"row": dict(row), "empty_since": None, "since": now}

    def check_pending_creates(self) -> None:
        if not self.pending_creates:
            return
        try:
            listed = {str(p.get("name")): p for p in self.rp.list_pods()}
        except RunPodError as exc:
            self.log("list_failed", detail=str(exc)[:120])
            return
        now = self.clock()
        for name, st in list(self.pending_creates.items()):
            raw = listed.get(name)
            if raw is not None:
                rate = float(raw.get("costPerHr") or st["row"]["usd_per_hr"])
                pod = Pod(
                    id=str(raw["id"]),
                    name=name,
                    vcpu=int(st["row"]["vcpu"]),
                    dc=st["row"]["dc"],
                    rate=rate,
                    token="",
                    created=st["since"],
                    until=self.held[name][1],
                    state="lost",
                    note="orphan create",
                )
                self.pods[pod.id] = pod
                self.ledger_call("bind_pod", self.run_id, name, pod.id, rate)
                self.held.pop(name, None)
                self.pending_creates.pop(name)
                self.log("orphan_create_found", pod=pod.id, name=name)
                self.delete_pod(pod, "orphan of a failed create")
                continue
            if st["empty_since"] is None:
                st["empty_since"] = now
            elif now - st["empty_since"] >= 60:
                self.pending_creates.pop(name)
                self.held.pop(name, None)
                self.ledger_call("release_pod", self.run_id, name, 0.0)
                self.log("failed_create_confirmed_absent", name=name)

    @staticmethod
    def raise_deferred() -> None:
        signum, _DEFER["pending"] = _DEFER["pending"], None
        if signum is not None:
            raise KeyboardInterrupt(f"signal {signum} (deferred until the create was recorded)")

    def reap_by_name(self, name: str, row: Mapping[str, Any]) -> None:
        """A failed/timed-out create may still have made a pod: find it by name, delete it."""
        for attempt in range(3):
            try:
                found = [p for p in self.rp.list_pods() if p.get("name") == name]
            except RunPodError:
                found = None
            if found:
                for raw in found:
                    pod = Pod(
                        id=raw["id"],
                        name=name,
                        vcpu=int(row["vcpu"]),
                        dc=row["dc"],
                        rate=float(raw.get("costPerHr") or row["usd_per_hr"]),
                        token="",
                        created=self.clock() - 120,
                        state="lost",
                        note="orphan create",
                    )
                    self.pods[pod.id] = pod
                    self.log("orphan_create_found", pod=pod.id, name=name)
                    self.delete_pod(pod, "orphan of a failed create")
                return
            if found == [] and attempt >= 1:
                return
            self.sleep(10)

    def sweep_orphans(self) -> None:
        """Delete pods with this job's prefix that this runner does not track."""
        try:
            listed = self.owned_live_pods()
        except RunPodError:
            return
        for raw in listed:
            if raw["id"] not in self.pods:
                pod = Pod(
                    id=raw["id"],
                    name=str(raw.get("name")),
                    vcpu=0,
                    dc="?",
                    rate=float(raw.get("costPerHr") or 0.0),
                    token="",
                    created=self.clock() - 120,
                    state="lost",
                    note="untracked orphan",
                )
                self.pods[pod.id] = pod
                self.log("orphan_found", pod=pod.id, name=pod.name)
                self.delete_pod(pod, "untracked pod with this job's prefix")

    # ------------------------------------------------------------ pod lifecycle
    def delete_pod(self, pod: Pod, why: str, tries: int = 3) -> None:
        """DELETE with backoff; a failure is logged loudly (stderr) and retried by callers."""
        if pod.deleted is not None:
            return
        for attempt in range(tries):
            try:
                self.rp.delete_pod(pod.id, confirm=True)
                break
            except RunPodError as exc:
                try:
                    still = self.rp.get_pod(pod.id)
                except RunPodError:
                    still = True
                if not still:
                    break
                self.log("delete_failed", pod=pod.id, attempt=attempt + 1, detail=str(exc)[:300])
                print(
                    f"!!! DELETE OF POD {pod.id} FAILED (attempt {attempt + 1}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt + 1 < tries:
                    self.sleep(5 * 2**attempt)
        else:
            return
        pod.deleted = self.clock()
        self.ledger_call("release_pod", self.run_id, pod.name, pod.cost_until(pod.deleted))
        if pod.state != "lost":
            pod.state = "deleted"
        self.log("pod_deleted", pod=pod.id, why=why, est_cost=round(pod.cost_until(pod.deleted), 5))

    def requeue(self, pod: Pod, keys: Iterable[str], why: str, penalize: bool = True) -> None:
        """A worker lost/retired/restarted: every unit it held restarts IN FULL elsewhere."""
        units = set()
        for key in keys:
            pod.inflight.discard(key)
            pod.unpulled.discard(key)
            units.add(platform_rule.unit_of_key(key))
        for unit in sorted(units):
            if self.unit_src.get(unit) in (pod.id, None):  # never strand a unit's keys
                self.requeue_unit(unit, why, penalize)

    # ------------------------------------------------------------ units
    def unit_keys(self, unit: str) -> List[str]:
        return [
            k for k in self.units.get(unit, []) if k not in self.completed and k not in self.failed
        ]

    def unit_seconds(self, keys: Sequence[str], slots: int) -> float:
        """Wall estimate for a unit on one worker with ``slots`` parallel episodes."""
        longest = max(episode_seconds(self.policy, self.episodes[k]) for k in keys)
        return math.ceil(len(keys) / max(1, slots)) * longest

    def pending_units(self) -> List[str]:
        out: List[str] = []
        for key in self.pending:
            u = platform_rule.unit_of_key(key)
            if u not in out:
                out.append(u)
        return out

    def take_unit(self, unit: str, source: str) -> List[str]:
        """Dispatch ``unit`` (all its open keys) to ``source``: a fresh attempt."""
        keys = self.unit_keys(unit)
        for k in keys:
            if k in self.pending:
                self.pending.remove(k)
        self.unit_src[unit] = source
        self.staged[unit] = {}
        return keys

    def requeue_unit(self, unit: str, why: str, penalize: bool = True) -> None:
        keys = self.unit_keys(unit)
        for pod in self.pods.values():
            pod.inflight.difference_update(keys)
            pod.unpulled.difference_update(keys)
        self.staged.pop(unit, None)  # partial results never mix with another worker's
        self.unit_src[unit] = None
        if not keys:
            return
        if not penalize and self.unit_free.get(unit, 0) < 1:
            self.unit_free[unit] = self.unit_free.get(unit, 0) + 1
            self.pending[:0] = [k for k in keys if k not in self.pending]
            self.log("unit_redispatched", unit=unit, keys=len(keys), why=why)
            return
        self.unit_attempts[unit] = self.unit_attempts.get(unit, 0) + 1
        for k in keys:
            self.attempts[k] = self.attempts.get(k, 0) + 1
        if self.unit_attempts[unit] >= int(self.policy["max_attempts_per_episode"]):
            self.fail_unit(unit, why)
        else:
            self.pending[:0] = [k for k in keys if k not in self.pending]
            self.log("unit_requeued", unit=unit, keys=len(keys), why=why)

    def fail_unit(self, unit: str, why: str) -> None:
        for k in self.unit_keys(unit):
            if k in self.pending:
                self.pending.remove(k)
            self.failed[k] = {"why": why, "attempts": self.attempts.get(k, 0), "unit": unit}
        self.staged.pop(unit, None)
        self.unit_src[unit] = None
        self.log("unit_failed", unit=unit, why=why)

    def apply_resume(self, prior_dirs: Sequence[Path]) -> None:
        """Skip units a prior run of THIS job completed on one platform; prior partial
        records stay as duplicate references (a re-run must be byte-identical or abort)."""
        want = json.dumps(self.job, sort_keys=True)
        complete: Dict[str, str] = {}
        for d in prior_dirs:
            prior_job = json.loads((d / "job.json").read_text())
            if json.dumps(prior_job, sort_keys=True) != want:
                raise jobspec.JobError(f"{d} ran a different job file; resume refused")
            recs = load_records(d)
            unknown = sorted(set(recs) - set(self.episodes))
            if unknown:
                raise jobspec.JobError(f"{d} has records outside the job: {unknown[:3]}")
            for k, e in recs.items():
                prev = self.seen.get(k)
                if prev is not None and jobspec.deterministic_bytes(
                    prev
                ) != jobspec.deterministic_bytes(e):
                    raise jobspec.JobError(f"prior runs disagree on {k}; resume refused")
                self.seen.setdefault(k, e)
            for unit, keys in self.units.items():
                if unit in complete or not all(k in recs for k in keys):
                    continue
                if len({platform_rule.platform_signature(recs[k]) for k in keys}) == 1:
                    complete[unit] = str(d)
        skip = {k for u in complete for k in self.units[u]}
        self.order = [k for k in self.order if k not in skip]
        self.pending = list(self.order)
        self.units = platform_rule.group_units(self.order)
        self.resume_info = {
            "from": [str(d) for d in prior_dirs],
            "units_already_complete": len(complete),
            "keys_skipped": len(skip),
            "units_to_run": len(self.units),
            "keys_to_run": len(self.order),
        }

    def bring_up(self, pod: Pod, agent: Any, health: Mapping[str, Any]) -> None:
        """Upload the repo archive and checkpoints, then setup (idempotent)."""
        if health.get("setup") is None:
            got = agent.put_file("/in/repo.tar.gz", self.repo_archive)
            if got.get("sha256") != self.repo_sha:
                raise Abort(f"repo upload sha mismatch on {pod.id}")
            for sha, path in sorted(self.ckpt_paths.items()):
                got = agent.put_file(f"/in/ckpt/{sha}.pth", path)
                if got.get("sha256") != sha:
                    raise Abort(f"checkpoint upload sha mismatch on {pod.id}")
            out = agent.post_json(
                "/setup",
                {
                    "repo_sha256": self.repo_sha,
                    "commit": self.job["repo_commit"],
                    "job_id": self.job["job_id"],
                },
            )
            setup = out.get("setup") or {}
            if sorted(setup.get("ckpts") or []) != sorted(self.ckpt_paths):
                raise Abort(f"pod {pod.id} checkpoint set differs after setup")
            self.log("pod_setup", pod=pod.id, setup=setup)

    def tick_pod(self, pod: Pod) -> None:
        now = self.clock()
        agent = self.agents[pod.id]
        try:
            health = agent.get_json("/health")
        except NET_ERRORS as exc:
            if pod.state == "booting":
                if now - pod.boot_from > float(self.policy["startup_deadline_seconds"]):
                    self.log("startup_deadline", pod=pod.id, detail=str(exc)[:120])
                    self.bad_slots[(pod.vcpu, pod.dc)] = now + 1800
                    pod.state = "lost"
                    self.delete_pod(pod, "no channel before the startup deadline")
                    self.requeue(pod, pod.inflight | pod.unpulled, "pod lost")
                return
            pod.first_fail = pod.first_fail or now
            if now - pod.first_fail >= float(self.policy["heartbeat_lost_seconds"]):
                self.log("pod_lost", pod=pod.id, detail=str(exc)[:120])
                pod.state = "lost"
                self.delete_pod(pod, "heartbeat lost")
                self.requeue(pod, pod.inflight | pod.unpulled, "pod lost")
            return
        pod.last_ok = now
        pod.first_fail = None
        if pod.boot_id and health.get("boot_id") != pod.boot_id:
            self.log("pod_rebooted", pod=pod.id)
            self.pull(pod)
            self.requeue(pod, set(pod.inflight), "agent restarted")
            if health.get("setup") is None and pod.state in ("ready", "draining"):
                pod.state = "booting"  # container restarted: set up again
                pod.boot_from = now
        pod.boot_id = health.get("boot_id")
        pod.slots = min(int(health.get("slots") or 1), workers_for(self.policy, pod.vcpu))
        if pod.state == "booting":
            self.tick_booting(pod, agent, health, now)
            return
        if self.retire_if_old(pod, now):
            return
        retried = set()
        for key, info in (health.get("failed") or {}).items():
            if key in pod.inflight:
                self.log("episode_error", pod=pod.id, key=key, info=info)
                (self.run_dir / "failures").mkdir(exist_ok=True)
                fname = key.replace("/", "__") + f".{pod.id}.{self.attempts[key]}.json"
                (self.run_dir / "failures" / fname).write_text(json.dumps(info, indent=1))
                self.retry_on_pod(pod, key, info)
                retried.add(key)
        done = set(health.get("done") or {}) & pod.inflight
        pod.inflight -= done
        pod.unpulled |= done
        known = set(health.get("queued") or []) | set(health.get("running") or []) | retried
        orphans = pod.inflight - known
        if orphans:  # the agent no longer knows this work (should not happen)
            self.requeue(pod, orphans, "agent lost the episode")
        due = now - pod.last_pull >= float(self.policy["pull_every_seconds"])
        all_remaining_here = self.outstanding() <= len(pod.unpulled)
        if pod.unpulled and (due or pod.state == "draining" or all_remaining_here):
            self.pull(pod)
        if pod.state == "draining" and not pod.inflight and not pod.unpulled:
            self.delete_pod(pod, "retired (drained)")

    def retry_on_pod(self, pod: Pod, key: str, info: Mapping[str, Any]) -> None:
        """An episode failed on a healthy pod: retry it on the SAME pod (same CPU model, the
        unit stays whole); out of attempts or unreachable -> the unit restarts elsewhere."""
        unit = platform_rule.unit_of_key(key)
        if info.get("rc") == "deadline":
            self.requeue(pod, [key], "pod deadline", penalize=False)
            return
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if self.attempts[key] >= int(self.policy["max_attempts_per_episode"]):
            self.fail_unit(unit, f"episode error rc={info.get('rc')}")
            for k in self.units.get(unit, []):
                pod.inflight.discard(k)
            return
        try:
            out = self.agents[pod.id].post_json(
                "/assign",
                {
                    "episodes": [{"key": key, "spec": self.episodes[key]}],
                    "timeout_seconds": 4 * episode_seconds(self.policy, self.episodes[key]) + 300,
                },
            )
        except NET_ERRORS as exc:
            out = {"error": str(exc)[:120]}
        if key in (out.get("accepted") or []):
            self.log("episode_retry_same_pod", pod=pod.id, key=key)
        else:
            self.requeue(pod, [key], f"episode error rc={info.get('rc')} (retry refused)")

    def life_left(self, pod: Pod, now: float) -> float:
        return pod.created + float(self.policy["pod_max_lifetime_seconds"]) - now

    def retire_if_old(self, pod: Pod, now: float) -> bool:
        """Max pod lifetime: pull, re-dispatch unfinished work (no penalty), delete."""
        left = self.life_left(pod, now)
        idle = not pod.inflight and not pod.unpulled
        slots = pod.slots or workers_for(self.policy, pod.vcpu)
        shortest = min(
            (self.unit_seconds(self.unit_keys(u), slots) for u in self.pending_units()),
            default=0,
        )
        too_short = left < 1.5 * shortest + 60 if self.pending else False
        if left > 0 and not (idle and too_short and pod.state == "ready"):
            return False
        self.pull(pod)
        self.requeue(pod, pod.inflight | pod.unpulled, "pod max lifetime", penalize=False)
        self.delete_pod(pod, "max pod lifetime reached" if left <= 0 else "retired near lifetime")
        return True

    def tick_booting(self, pod: Pod, agent: Any, health: Mapping[str, Any], now: float) -> None:
        if health.get("pip") == "failed":
            pod.state = "lost"
            self.bad_slots[(pod.vcpu, pod.dc)] = now + 1800
            self.delete_pod(pod, "pip install failed")
            return
        pod.cpu_model = health.get("cpu_model")
        # No run-level CPU pin: units (worlds) are pinned to one pod, so any model is usable.
        self.workers_info[pod.id] = {
            "isa_flags": list(health.get("isa_flags") or []),
            "worker_id": pod.id,
            "backend": "pods",
        }
        try:
            self.bring_up(pod, agent, health)
            if health.get("setup") is None:
                health = agent.get_json("/health")
        except NET_ERRORS as exc:
            self.log("upload_retry", pod=pod.id, detail=str(exc)[:160])
        if health.get("pip") == "ok" and health.get("setup") is not None:
            pod.state = "ready"
            self.log(
                "pod_ready",
                pod=pod.id,
                slots=pod.slots,
                cores=health.get("cores"),
                cpu_model=health.get("cpu_model"),
                isa_flags=health.get("isa_flags"),
                self_delete_capable=health.get("self_delete_capable"),
                seconds_to_ready=round(now - pod.created, 1),
            )
        elif now - pod.boot_from > float(self.policy["ready_deadline_seconds"]):
            pod.state = "lost"
            self.bad_slots[(pod.vcpu, pod.dc)] = now + 1800
            self.delete_pod(pod, "not ready before the ready deadline")
            self.requeue(pod, pod.inflight | pod.unpulled, "pod lost")

    def pull(self, pod: Pod) -> None:
        agent = self.agents.get(pod.id)
        if agent is None:  # an orphan we only delete
            return
        try:
            listing = agent.get_json("/records")
        except NET_ERRORS as exc:
            self.log("pull_failed", pod=pod.id, detail=str(exc)[:120])
            return
        pod.last_pull = self.clock()
        got = 0
        for row in listing:
            key = row["key"]
            if key not in self.episodes:
                raise Abort(f"pod {pod.id} produced an unknown record {key}")
            ck = (key, row["sha256"])
            unit = platform_rule.unit_of_key(key)
            mine = self.unit_src.get(unit) == pod.id
            if ck in self.verified and (key in self.completed or not mine):
                pod.unpulled.discard(key)
                if not mine:
                    pod.inflight.discard(key)
                continue
            if mine and key in self.staged.get(unit, {}):
                pod.unpulled.discard(key)
                pod.inflight.discard(key)
                continue
            data = self.record_cache.get(ck)
            if data is None:
                try:
                    data = agent.get_bytes("/record/" + key)
                except NET_ERRORS as exc:
                    self.log("pull_record_failed", pod=pod.id, key=key, detail=str(exc)[:120])
                    continue
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    self.log("pull_sha_mismatch", pod=pod.id, key=key)
                    continue
            if key not in self.completed:
                self.record_cache[ck] = data
            self.accept_record(key, data, source=pod.id)
            self.verified.add(ck)
            pod.unpulled.discard(key)
            pod.inflight.discard(key)
            got += 1
        if got:
            self.log(
                "pulled", pod=pod.id, new=got, completed=len(self.completed), of=len(self.order)
            )
            self.write_state()

    def accept_record(
        self, key: str, data: bytes, source: str, worker: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Validate a record; duplicates must match deterministically (any source, any
        platform) or the run aborts; a record counts only for its unit's current attempt,
        and a unit is published (write-once) only when complete on that one worker."""
        entry = json.loads(data)
        stamp = entry.get("fanout") or {}
        ep = self.episodes[key]
        if stamp.get("episode_key") != key or stamp.get("job_id") != self.job["job_id"]:
            raise Abort(f"record {key} from {source} has a wrong fanout stamp")
        if stamp.get("repo_commit") != self.job["repo_commit"]:
            raise Abort(f"record {key} from {source} ran a different commit")
        for name in ("arm", "mix", "world_seed", "world_index", "roster_member_sha256s"):
            if entry.get(name) != ep[name]:
                raise Abort(f"record {key} field {name} differs from the job")
        if stamp.get("spec_sha256") != spec_sha256(ep):
            raise Abort(f"record {key} from {source} was produced from a different spec")
        platform_id = (entry.get("platform") or {}).get("platform_id")
        if not platform_id:
            raise Abort(f"record {key} from {source} has no platform stamp")
        platform_rule.stamp_worker(entry, worker or self.workers_info.get(source))
        self.platform_ids.add(platform_id)
        prior = self.seen.get(key)
        dest = self.run_dir / "records" / key
        if prior is None and dest.exists():
            prior = json.loads(dest.read_bytes())
        if prior is not None and jobspec.deterministic_bytes(prior) != jobspec.deterministic_bytes(
            entry
        ):
            dup = self.run_dir / "divergent" / (key.replace("/", "__") + f".{source}.json")
            dup.parent.mkdir(exist_ok=True)
            dup.write_bytes(data)
            raise Abort(f"DIVERGENT duplicate for {key} from {source}")
        self.seen.setdefault(key, entry)
        if key in self.completed or dest.exists():
            self.log("duplicate_verified", key=key, source=source)
            return
        unit = platform_rule.unit_of_key(key)
        if self.unit_src.get(unit) != source:
            self.log("stale_record", key=key, source=source, unit_source=self.unit_src.get(unit))
            return
        self.staged.setdefault(unit, {})[key] = entry
        if all(k in self.staged[unit] for k in self.unit_keys(unit)):
            self.publish_unit(unit, source)

    def publish_unit(self, unit: str, source: str) -> None:
        entries = self.staged.pop(unit)
        sigs = {platform_rule.platform_signature(e) for e in entries.values()}
        if len(sigs) > 1:
            raise Abort(f"unit {unit} from {source} spans platforms {sorted(sigs)}")
        for key, entry in sorted(entries.items()):
            dest = self.run_dir / "records" / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = (json.dumps(entry, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            tmp = dest.with_name(f".{dest.name}.tmp")
            tmp.write_bytes(data)
            os.link(tmp, dest)  # write-once
            tmp.unlink()
            self.completed[key] = hashlib.sha256(data).hexdigest()
            if key in self.pending:
                self.pending.remove(key)
        for ck in [ck for ck in self.record_cache if ck[0] in entries]:
            self.record_cache.pop(ck, None)
        self.log("unit_published", unit=unit, source=source, records=len(entries))

    def assign(self) -> None:
        """Whole units (all arms of one world) to one pod each, while it has free slots."""
        for pod in self.pods.values():
            if pod.state != "ready" or not self.pending:
                continue
            free = pod.slots - len(pod.inflight)
            if free <= 0:
                continue
            left = self.life_left(pod, self.clock())
            chosen = []
            for unit in self.pending_units():
                keys = self.unit_keys(unit)
                if not keys or 1.5 * self.unit_seconds(keys, pod.slots) + 60 > left:
                    continue
                chosen.append((unit, keys))
                free -= len(keys)
                if free <= 0:
                    break
            if not chosen:
                continue
            batch = [k for _, keys in chosen for k in keys]
            longest = max(episode_seconds(self.policy, self.episodes[k]) for k in batch)
            body = {
                "episodes": [{"key": k, "spec": self.episodes[k]} for k in batch],
                "timeout_seconds": 4 * longest + 300,
            }
            try:
                out = self.agents[pod.id].post_json("/assign", body)
            except NET_ERRORS as exc:
                self.log("assign_failed", pod=pod.id, detail=str(exc)[:120])
                continue
            for unit, keys in chosen:
                self.take_unit(unit, pod.id)
                pod.inflight.update(keys)  # accepted now, or already queued/done there
            self.log(
                "assigned",
                pod=pod.id,
                units=[u for u, _ in chosen],
                keys=out.get("accepted", []),
            )

    def retire_smaller(self) -> None:
        """Every capacity_recheck_seconds: launch a bigger size, drain the smallest pods."""
        now = self.clock()
        if now - self.last_capacity_check < float(self.policy["capacity_recheck_seconds"]):
            return
        live = [p for p in self.pods.values() if p.state in ("booting", "ready")]
        self.last_capacity_check = now
        if not live or self.outstanding() <= 0:
            return
        smallest = min(live, key=lambda p: p.vcpu)
        cap = size_cap_for(self.policy, self.outstanding())
        bigger = [v for v in self.policy["vcpu_sizes_desc"] if smallest.vcpu < v <= cap]
        if not bigger:
            return
        rows = self.probe(bigger)
        self.log("capacity_recheck", smallest=smallest.vcpu, stocked=rows[:8])
        before = set(self.pods)
        for row in rows:
            if self.launch_allowed(row["usd_per_hr"]) is None and self.create_pod(row):
                break
        if set(self.pods) != before:
            smallest.retire_pending = True
            self.log("retire_planned", pod=smallest.id)

    def drain_retirees(self) -> None:
        """Drain pods marked for retirement once a newer, bigger pod is ready."""
        for pod in self.pods.values():
            if pod.retire_pending and pod.state == "ready":
                bigger_ready = any(
                    q.state == "ready" and q.vcpu > pod.vcpu for q in self.pods.values()
                )
                if bigger_ready:
                    try:
                        self.agents[pod.id].post_json("/drain", {})
                    except NET_ERRORS:
                        pass
                    pod.state = "draining"
                    pod.retire_pending = False
                    self.log("pod_draining", pod=pod.id)

    # ------------------------------------------------------------ cleanup
    def run_prefix(self) -> str:
        """Pods of THIS run only (job-wide prefix + run dir name); cleanup CLI is job-wide."""
        return f"{pod_prefix(self.policy, self.job['job_id'])}{self.run_dir.name}-"

    def owned_live_pods(self) -> List[Dict[str, Any]]:
        """This run's pods only: names are exactly ``<run prefix><counter>``."""
        return [p for p in self.rp.list_pods() if owned_name(self.run_prefix(), p.get("name"))]

    def delete_everything(self, attempts: int = 10) -> List[Dict[str, Any]]:
        """Delete every runner-owned pod of this job; return GET /pods leftovers (want [])."""
        for pod in self.pods.values():
            if pod.deleted is None:
                self.delete_pod(pod, "run exit")
        left: List[Dict[str, Any]] = []
        empty_since: Optional[float] = None
        for i in range(attempts + 8):
            for pod in self.pods.values():  # works even when GET /pods is failing
                if pod.deleted is None:
                    self.delete_pod(pod, "run exit (retry)", tries=1)
            try:
                left = self.owned_live_pods()
            except RunPodError as exc:
                self.log("list_failed", detail=str(exc)[:120])
                left = [{"unknown": "list failed"}]
                self.sleep(10)
                continue
            if not left:
                # A pod from a create that was in flight can surface late: require two empty
                # listings at least 60 s apart.
                if empty_since is not None and self.clock() - empty_since >= 60:
                    return []
                empty_since = empty_since if empty_since is not None else self.clock()
                self.sleep(30)
                continue
            empty_since = None
            for row in left:
                try:
                    self.rp.delete_pod(row["id"], confirm=True)
                    self.log("pod_deleted_by_sweep", pod=row["id"])
                    self.ledger_call("release_pod", self.run_id, str(row.get("name")), 0.0)
                except RunPodError as exc:
                    self.log("delete_failed", pod=row["id"], detail=str(exc)[:300])
            self.sleep(min(60, 10 * (i + 1)))
        undeleted = [p.id for p in self.pods.values() if p.deleted is None]
        if left or undeleted:
            msg = (
                f"!!! PODS MAY STILL BE BILLING: tracked-undeleted={undeleted} listed={left}. "
                "The watchdog keeps retrying; manual: runner.py cleanup --job-id "
                f"{self.job['job_id']} --confirm"
            )
            print(msg, file=sys.stderr, flush=True)
            self.log("cleanup_incomplete", undeleted=undeleted, listed=left)
            return left or [{"id": i, "unconfirmed": True} for i in undeleted]
        return left

    # ------------------------------------------------------------ main loop
    def prepare_uploads(self, workdir: Path) -> Dict[str, Any]:
        problems = jobspec.check_commit(self.repo, self.job)
        if problems:
            raise jobspec.JobError("; ".join(problems))
        self.repo_archive = Path(workdir) / "repo.tar.gz"
        self.repo_sha = jobspec.git_archive(self.repo, self.job["repo_commit"], self.repo_archive)
        self.ckpt_paths = jobspec.resolve_checkpoints(
            self.job["checkpoints"], self.policy, self.allowlist
        )
        return {
            "repo": {
                "commit": self.job["repo_commit"],
                "kind": "git archive (tracked files only)",
                "sha256": self.repo_sha,
                "bytes": self.repo_archive.stat().st_size,
            },
            "checkpoints": [
                {"sha256": sha, "path": self.allowlist[sha]["path"], "bytes": p.stat().st_size}
                for sha, p in sorted(self.ckpt_paths.items())
            ],
            "nothing_else": "no untracked files, artifacts, scores.db, keys or venv",
        }

    def run(self) -> int:
        if not self.confirm:
            raise jobspec.JobError("run needs --confirm")
        self.start = self.clock()
        self.deadline = self.start + 60.0 * float(self.job["max_wall_minutes"])
        self.hard_end = self.deadline + float(self.policy["watchdog_grace_seconds"]) + 300.0
        if self.run_dir.exists():
            raise jobspec.JobError(f"run dir {self.run_dir} already exists")
        life = float(self.policy["pod_max_lifetime_seconds"])
        big = workers_for(self.policy, max(self.policy["vcpu_sizes_desc"]))
        too_long = [
            u for u, keys in self.units.items() if 1.5 * self.unit_seconds(keys, big) + 60 >= life
        ]
        if too_long:
            raise jobspec.JobError(
                f"{len(too_long)} world units cannot fit the {life:.0f} s pod lifetime "
                f"(e.g. {too_long[0]}); raise pod_max_lifetime_seconds"
            )
        self.try_legacy_lock()  # before registering: an old-format runner cannot start now
        try:  # idempotency + account-level registration (no pod, no spend on refusal)
            totals = self.ledger.register_run(
                self.run_id, self.job["job_id"], self.run_id, self.budget, os.getpid()
            )
        except DuplicateRun as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            self.stop_reason = f"duplicate: {exc}"
            return EXIT_DUPLICATE
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "job.json").write_text(json.dumps(self.job, indent=1, sort_keys=True))
        self.events = (self.run_dir / "events.jsonl").open("x")
        self.log("account_ledger", **totals)
        self.try_legacy_lock()
        workdir = Path(tempfile.mkdtemp(prefix="rpf-upload-"))
        leftovers: List[Dict[str, Any]] = [{"unknown": True}]
        old_handlers = install_signal_handlers()
        code = 1
        awake = None
        try:
            self.tls_preflight()
            manifest = self.prepare_uploads(workdir)
            self.balance_start = self.rp.balance()
            if self.balance_start - float(self.policy["global_floor_usd"]) <= 0:
                raise Abort(f"balance {self.balance_start:.2f} is at or below the floor")
            plan = self.cost_plan()
            plan.update({"uploads": manifest, "balance_start": self.balance_start})
            (self.run_dir / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
            self.log("planned_cost", **plan["cost"])
            self.watchdog = self.spawn_watchdog(self)
            self.check_watchdog(startup=True)
            self.log(
                "watchdog_armed",
                fires_at_epoch=self.deadline + float(self.policy["watchdog_grace_seconds"]),
            )
            awake = keep_awake(getattr(self.watchdog, "pid", None))
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
            ignore_signals()  # a second Ctrl-C must not cut the cleanup short
            try:
                leftovers = self.delete_everything() if self.pods else []
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
                restore_signal_handlers(old_handlers)
        return code if not leftovers else 5

    def unit_summary(self) -> Dict[str, Any]:
        recs = load_records(self.run_dir) if (self.run_dir / "records").is_dir() else {}
        sigs = platform_rule.unit_signatures(recs)
        return {
            "units": len(self.units),
            "units_published": len({platform_rule.unit_of_key(k) for k in self.completed}),
            "units_failed": len({platform_rule.unit_of_key(k) for k in self.failed}),
            "platforms_by_unit": {u: sorted(v) for u, v in sorted(sigs.items())},
            "per_world_problems": platform_rule.check_per_world(recs),
            "numerics_env": self.numerics,
            "resume": self.resume_info,
        }

    def settled_cost(self) -> float:
        """What this run spent: the larger of the balance delta and the pod estimate."""
        try:
            receipt = json.loads((self.run_dir / "receipt.json").read_text())
        except (OSError, ValueError):
            return self.committed_worst()
        delta = receipt.get("actual_cost_usd_balance_delta") or 0.0
        return max(float(delta), float(receipt.get("estimated_cost_usd") or 0.0))

    def tls_preflight(self) -> None:
        """Fail fast (before any pod exists) if verified HTTPS to RunPod/proxy does not work."""
        problems = self.preflight_fn()
        if problems:
            raise Abort("TLS preflight failed (no pod created): " + "; ".join(problems))
        self.log("tls_preflight_ok")

    def try_legacy_lock(self) -> None:
        """Hold a SHARED lock on the old single-run lock file while live, so an old-format
        runner (exclusive lock) cannot start beside v2 runs; if one is live, its ledger rows
        already count here and we retry each tick."""
        if self.legacy_lock is not None:
            return
        root = Path(self.policy["artifacts_root"]) / "runpod-fanout"
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / ".runpod-runner.lock").open("a")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            self.legacy_lock = handle
        except BlockingIOError:
            handle.close()

    def check_watchdog(self, startup: bool = False) -> None:
        """The detached watchdog must be alive while pods may exist."""
        wd = self.watchdog
        if wd is None:  # injected stub (tests)
            return
        if startup:
            for _ in range(60):
                log = self.run_dir / "watchdog.log"
                if log.exists() and b"armed" in log.read_bytes():
                    break
                self.sleep(0.5)
            else:
                raise Abort("watchdog did not arm")
        if wd.poll() is not None:
            raise Abort(f"watchdog exited early (rc {wd.returncode})")

    def loop(self) -> None:
        self.maybe_launch()
        hb = float(self.policy["heartbeat_seconds"])
        while True:
            now = self.clock()
            if self.stop_reason and not any(p.deleted is None for p in self.pods.values()):
                return
            if len(self.completed) + len(self.failed) == len(self.order):
                self.stop_reason = "all episodes finished"
                for pod in self.pods.values():
                    if pod.deleted is None and pod.state not in ("lost",):
                        self.pull(pod)
                return
            if now >= self.deadline:
                self.stop_reason = "max wall time reached"
                for pod in self.pods.values():
                    if pod.deleted is None:
                        self.pull(pod)
                return
            self.check_watchdog()
            self.check_pending_creates()
            self.try_legacy_lock()
            if now - self.last_reconcile >= 300:
                self.last_reconcile = now
                try:
                    out = self.ledger.reconcile(self.rp)
                    if out["released"] or out["deleted_overdue_dead_run_pods"]:
                        self.log("ledger_reconciled", **out)
                except (RunPodError, OSError, ValueError, KeyError) as exc:
                    self.log("ledger_reconcile_failed", detail=str(exc)[:160])
            for pod in list(self.pods.values()):
                if pod.deleted is None and pod.id not in self.agents:
                    self.delete_pod(pod, "retry delete of an orphan")
                elif pod.deleted is None:
                    self.tick_pod(pod)
            self.drain_retirees()
            self.assign()
            if not any(p.deleted is None for p in self.pods.values()):
                if now - self.last_capacity_check >= 120:
                    self.maybe_launch()
            else:
                if self.live_slots() < self.outstanding() and (
                    now - self.last_capacity_check >= 300
                ):
                    self.maybe_launch()
                self.retire_smaller()
            if self.committed_worst() > self.budget + 1e-6:
                raise Abort("committed worst case exceeds the budget")
            self.write_state()
            self.sleep(hb)

    def cost_plan(self, sizes_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        par = parallelism_plan(
            self.policy,
            list(self.episodes.values()),
            self.budget,
            float(self.job["max_wall_minutes"]),
            sizes_rows,
        )
        return {
            "cost": {
                "episodes": len(self.episodes),
                "budget_usd_hard_cap": self.budget,
                **par,
                "note": "the runner never lets this run's committed worst case (each live pod "
                "billed to its reservation horizon = min(hard end, created + max pod lifetime "
                "+ grace)) exceed the budget; account-wide, all live runs' reservations must "
                "fit balance - floor and the project cap",
            }
        }

    def finish(self, leftovers: List[Dict[str, Any]], code: int) -> None:
        balance_end = None
        try:
            self.sleep(60)  # let per-second billing settle
            balance_end = self.rp.balance()
        except RunPodError:
            pass
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "backend": "pods",
            "job_id": self.job["job_id"],
            "run_dir": str(self.run_dir),
            "exit_code": code,
            **self.unit_summary(),
            "stop_reason": self.stop_reason,
            "episodes": len(self.order),
            "completed": len(self.completed),
            "failed": self.failed,
            "pods": [p.public() for p in self.pods.values()],
            "estimated_cost_usd": round(self.estimated_cost(), 5),
            "balance_start": self.balance_start,
            "balance_end": balance_end,
            "actual_cost_usd_balance_delta": (
                None
                if balance_end is None or self.balance_start is None
                else round(self.balance_start - balance_end, 5)
            ),
            "cost_note": "balance delta also includes any other spend on the account in the "
            "same window; billing may lag a few seconds",
            "final_get_pods_runner_owned": leftovers,
            "cost_basis": "estimated_cost_usd (per-pod costPerHr x lifetime) is this run's "
            "cost; the balance delta is account-wide and includes concurrent runs",
            "finished_utc": utc_now(),
        }
        (self.run_dir / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
        self.log(
            "actual_spend",
            estimated=receipt["estimated_cost_usd"],
            balance_delta=receipt["actual_cost_usd_balance_delta"],
            leftover_pods=len(leftovers),
        )
        self.write_state()
        if self.events is not None:
            self.events.close()
            self.events = None


# ---------------------------------------------------------------- signals / watchdog


def keep_awake(pid: Optional[int]) -> Optional[subprocess.Popen]:
    """macOS: keep the Mac awake while the watchdog (or else this runner) lives."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(
        ["caffeinate", "-ims", "-w", str(pid or os.getpid())], start_new_session=True
    )


def ignore_signals() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except ValueError:
            pass


_DEFER: Dict[str, Any] = {"active": False, "pending": None}


def _raise_interrupt(signum, frame):  # noqa: ARG001
    if _DEFER["active"]:  # a create is in flight: finish and record it first
        _DEFER["pending"] = signum
        return
    raise KeyboardInterrupt(f"signal {signum}")


def install_signal_handlers() -> Dict[int, Any]:
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            old[sig] = signal.signal(sig, _raise_interrupt)
        except ValueError:  # not the main thread (tests)
            pass
    return old


def restore_signal_handlers(old: Mapping[int, Any]) -> None:
    for sig, handler in old.items():
        signal.signal(sig, handler)


class WatchdogHandle:
    """A watchdog started detached (launchd job or setsid child), tracked via its pid file."""

    def __init__(
        self, run_dir: Path, popen: Optional[subprocess.Popen] = None, label: Optional[str] = None
    ):
        self.run_dir = Path(run_dir)
        self.popen = popen
        self.label = label
        self.returncode: Optional[int] = None

    @property
    def pid(self) -> Optional[int]:
        try:
            return int((self.run_dir / "watchdog.pid").read_text())
        except (OSError, ValueError):
            return None

    def poll(self) -> Optional[int]:
        if self.popen is not None and self.popen.poll() is not None:
            self.returncode = self.popen.returncode
            return self.returncode
        pid = self.pid
        if pid is None:
            return None  # not started yet (check_watchdog waits for "armed")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.returncode = -1
            return -1
        except PermissionError:
            pass
        return None


def watchdog_argv(runner: Runner, label: Optional[str]) -> List[str]:
    argv = [
        sys.executable,
        "-m",
        "research.runpod_fanout.watchdog",
        "--job-id",
        runner.job["job_id"],
        "--fire-epoch",
        str(runner.deadline + float(runner.policy["watchdog_grace_seconds"])),
        "--run-dir",
        str(runner.run_dir),
        "--runner-pid",
        str(os.getpid()),
        "--prefix",
        runner.run_prefix(),
    ]
    if hasattr(runner, "endpoint_prefix"):  # serverless run: also tear down its endpoints
        argv += ["--endpoint-prefix", runner.endpoint_prefix()]
    return argv + (["--launchd-label", label] if label else [])


def spawn_watchdog_process(runner: Runner, use_launchd: Optional[bool] = None) -> WatchdogHandle:
    """Start the watchdog fully detached from this process and its terminal.

    macOS: a launchd job in the user's GUI domain (survives the runner, its terminal or
    tmux server, and keeps Keychain access for rp.py). Elsewhere: setsid child with stdin
    from /dev/null and output to ``watchdog.log``.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("refusing to spawn the real watchdog under pytest")
    if use_launchd is None:
        use_launchd = sys.platform == "darwin" and shutil.which("launchctl") is not None
    log_path = runner.run_dir / "watchdog.log"
    if use_launchd:
        import plistlib

        label = launchd_label_for(runner)
        plist = {
            "Label": label,
            "ProgramArguments": watchdog_argv(runner, label),
            "WorkingDirectory": str(REPO),
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(log_path),
            "StandardInPath": "/dev/null",
            "RunAtLoad": True,
            "KeepAlive": False,
            "AbandonProcessGroup": True,
            "ProcessType": "Background",
            "EnvironmentVariables": {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONUNBUFFERED": "1",
            },
        }
        path = runner.run_dir / "watchdog.plist"
        path.write_bytes(plistlib.dumps(plist))
        done = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise Abort(f"launchctl bootstrap failed: {done.stderr.strip()[:200]}")
        return WatchdogHandle(runner.run_dir, label=label)
    log = log_path.open("ab")
    popen = subprocess.Popen(
        watchdog_argv(runner, None),
        cwd=str(REPO),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return WatchdogHandle(runner.run_dir, popen=popen)


def launchd_label_for(runner: Runner) -> str:
    return f"com.snakedqn.rpf.{runner.job['job_id']}.{runner.run_dir.name}"


# ---------------------------------------------------------------- local backend


def run_local(
    job: Mapping[str, Any],
    policy: Mapping[str, Any],
    allowlist: Mapping[str, Mapping[str, str]],
    run_dir: Path,
    slot_timeout: float = 3600.0,
    resume_from: Sequence[Path] = (),
) -> int:
    """Play the job's episodes serially on this Mac in ONE shared slot (pool 3 order)."""
    from research.apex_safety_20260926 import dev_screen as ds

    if not ds.on_ac_power():
        print("refusing: on battery", file=sys.stderr)
        return 2
    problems = jobspec.check_commit(REPO, job)
    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 2
    ckpts = jobspec.resolve_checkpoints(job["checkpoints"], policy, allowlist)
    slots = ds.acquire_cpu_slots(Path(policy["slot_lock_root"]), 1, slot_timeout, pool=3)
    run_dir = Path(run_dir)
    try:
        run_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps(job, indent=1, sort_keys=True))
        work = Path(tempfile.mkdtemp(prefix="rpf-local-"))
        try:
            archive = work / "repo.tar.gz"
            repo_sha = jobspec.git_archive(REPO, job["repo_commit"], archive)
            src = work / "repo"
            src.mkdir()
            subprocess.run(["tar", "-xzf", str(archive), "-C", str(src)], check=True)
            cdir = work / "ckpt"
            cdir.mkdir()
            for sha, path in ckpts.items():
                shutil.copyfile(path, cdir / f"{sha}.pth")
            out_root = work / "out"
            runner = Runner(
                job,
                policy,
                allowlist,
                run_dir,
                rp=_NoRunPod(),
                confirm=False,
                resume_from=resume_from,
            )
            for unit in runner.units:  # one worker (this Mac) holds every unit
                runner.unit_src[unit] = "local"
            runner.workers_info["local"] = {
                "isa_flags": platform_rule.local_isa_flags(),
                "worker_id": "local",
                "backend": "local",
            }
            env = dict(os.environ, SNAKE_DQN_DEVICE="cpu", PYTHONHASHSEED="0")
            env.update(runner.numerics)
            events = (run_dir / "events.jsonl").open("x")
            meta = {
                "backend": "local",
                "slot": Path(slots[0].name).name,
                "repo_sha256": repo_sha,
                "started_utc": utc_now(),
            }
            events.write(json.dumps({"event": "start", **meta}) + "\n")
            for key in runner.order:
                spec_path = work / "spec.json"
                spec_path.write_text(json.dumps(runner.episodes[key]))
                t0 = time.time()
                done = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "research.runpod_fanout.episode",
                        "--spec",
                        str(spec_path),
                        "--out-root",
                        str(out_root),
                        "--ckpt-dir",
                        str(cdir),
                        "--provider",
                        "local:mac",
                        "--job-id",
                        job["job_id"],
                        "--commit",
                        job["repo_commit"],
                    ],
                    cwd=str(src),
                    env=env,
                    capture_output=True,
                    text=True,
                )
                row = {
                    "utc": utc_now(),
                    "key": key,
                    "rc": done.returncode,
                    "seconds": round(time.time() - t0, 1),
                }
                if done.returncode != 0:
                    row["stderr_tail"] = done.stderr[-1500:]
                else:
                    runner.accept_record(key, (out_root / key).read_bytes(), source="local")
                events.write(json.dumps(row) + "\n")
                events.flush()
                print(json.dumps(row), flush=True)
            events.close()
            ok = len(runner.completed) == len(runner.order)
            (run_dir / "receipt.json").write_text(
                json.dumps(
                    {
                        "schema": RECEIPT_SCHEMA,
                        "backend": "local",
                        "completed": len(runner.completed),
                        "episodes": len(runner.order),
                        "finished_utc": utc_now(),
                        **runner.unit_summary(),
                        **meta,
                    },
                    indent=1,
                )
            )
            return 0 if ok else 4
        finally:
            shutil.rmtree(work, ignore_errors=True)
    finally:
        ds.release_cpu_slots(slots)


class _NoRunPod:
    def __getattr__(self, name):
        raise RuntimeError("local backend must not call RunPod")


# ---------------------------------------------------------------- compare / merge


def load_records(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    root = Path(run_dir) / "records"
    return {
        str(p.relative_to(root)): json.loads(p.read_bytes())
        for p in sorted(root.rglob("*.json"))
        if not p.name.startswith(".")
    }


def compare_runs(a: Path, b: Path) -> Dict[str, Any]:
    ra, rb = load_records(a), load_records(b)
    common = sorted(set(ra) & set(rb))
    diffs = [
        k
        for k in common
        if jobspec.deterministic_bytes(ra[k]) != jobspec.deterministic_bytes(rb[k])
    ]
    return {
        "a": str(a),
        "b": str(b),
        "platforms_a": sorted(
            {(r.get("platform") or {}).get("platform_id", "-") for r in ra.values()}
        ),
        "platforms_b": sorted(
            {(r.get("platform") or {}).get("platform_id", "-") for r in rb.values()}
        ),
        "only_a": sorted(set(ra) - set(rb)),
        "only_b": sorted(set(rb) - set(ra)),
        "compared": len(common),
        "identical": len(common) - len(diffs),
        "different": diffs,
        "rule": "canonical JSON minus wall_seconds, any *seconds* key, platform and fanout",
    }


def platform_check(records: Mapping[str, Mapping[str, Any]]) -> List[str]:
    ids = sorted({r.get("platform", {}).get("platform_id", "MISSING") for r in records.values()})
    return ids


def merge_runs(run_dirs: Sequence[Path], out: Path) -> Dict[str, Any]:
    """One records tree from several runs under the PER-WORLD rule: each unit (all arms of
    one wrapper/mix/world) comes from one run on one platform; any key present in several
    runs must agree deterministically; different units may come from different platforms."""
    runs: List[tuple] = []
    for d in run_dirs:
        root = Path(d) / "records"
        recs = {}
        for p in sorted(root.rglob("*.json")):
            if p.name.startswith("."):
                continue
            data = p.read_bytes()
            recs[str(p.relative_to(root))] = (data, json.loads(data))
        runs.append((str(d), recs))
    first: Dict[str, Dict[str, Any]] = {}
    for label, recs in runs:
        for key, (_, entry) in recs.items():
            if key in first and jobspec.deterministic_bytes(
                first[key]
            ) != jobspec.deterministic_bytes(entry):
                raise Abort(f"merge: {key} differs between runs")
            first.setdefault(key, entry)
    source = platform_rule.choose_unit_sources(
        [(lb, {k: v[1] for k, v in r.items()}) for lb, r in runs]
    )
    merged: Dict[str, tuple] = {}
    for label, recs in runs:
        for key, val in recs.items():
            if source[platform_rule.unit_of_key(key)] == label:
                merged[key] = val
    for label, recs in runs:  # keys of a unit only another run has: same platform only
        for key, val in recs.items():
            if key in merged:
                continue
            unit = platform_rule.unit_of_key(key)
            sig = {
                platform_rule.platform_signature(v[1])
                for k, v in merged.items()
                if platform_rule.unit_of_key(k) == unit
            }
            if sig != {platform_rule.platform_signature(val[1])}:
                raise Abort(f"merge: unit {unit} would span platforms (re-run the whole unit)")
            merged[key] = val
    problems = platform_rule.check_per_world({k: v[1] for k, v in merged.items()})
    if problems:
        raise Abort("merge refuses: " + "; ".join(problems[:5]))
    Path(out).mkdir(parents=True)
    for key, (data, _) in merged.items():
        dest = Path(out) / "records" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as fh:
            fh.write(data)
    pids = sorted({(v[1].get("platform") or {}).get("platform_id") for v in merged.values()})
    return {
        "records": len(merged),
        "units": len({platform_rule.unit_of_key(k) for k in merged}),
        "platform_ids": pids,
        "rule": "per-world: one platform per (wrapper, mix, world_seed) unit",
        "out": str(out),
    }


# ---------------------------------------------------------------- CLI


def default_run_dir(policy: Mapping[str, Any], job: Mapping[str, Any], backend: str) -> Path:
    base = Path(policy["artifacts_root"]) / "runpod-fanout" / job["job_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{backend}-{stamp}"


def cmd_plan(a, job, policy, allow) -> int:
    runner = Runner(job, policy, allow, Path(tempfile.mkdtemp()) / "plan", budget=a.budget or 0)
    problems = preflight()
    if problems:
        print(
            "TLS preflight FAILED (verified HTTPS to RunPod/proxy does not work from this "
            "interpreter; no pod would get a channel):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 2
    work = Path(tempfile.mkdtemp(prefix="rpf-plan-"))
    try:
        manifest = runner.prepare_uploads(work)
        rows = runner.probe()
        balance = runner.rp.balance()
        plan = runner.cost_plan(rows)
        par = plan["cost"]["parallelism"]
        print(
            f"PLAN {job['job_id']}: {len(job['episodes'])} episodes; "
            f"budget ${a.budget or 0} allows "
            f"{par['vcpu_allowed_by_budget']} vCPU, "
            f"stock ~{par['vcpu_allowed_by_stock_est']}, "
            f"caps {par['vcpu_allowed_by_caps']} -> effective {par['effective_vcpu']} vCPU "
            f"({par['effective_workers']} workers), expected wall "
            f"{par['expected_wall_minutes']} min, expected cost ${par['expected_cost_usd']}",
            file=sys.stderr,
        )
        live = [
            p
            for p in runner.rp.list_pods()
            if str(p.get("name", "")).startswith(policy["pod_name_prefix"])
        ]
        out = {
            "dry_run": True,
            "job": jobspec.summarize_job(job),
            "capacity_cpu3c_secure_stocked": rows,
            "balance_usd": balance,
            "global_floor_usd": policy["global_floor_usd"],
            **plan,
            "upload_manifest": manifest,
            "existing_runner_pods": live,
            "account": {
                **SharedLedger(policy).peek(),
                "headroom_usd_balance_minus_floor_minus_reserved": round(
                    balance
                    - float(policy["global_floor_usd"])
                    - SharedLedger(policy).peek()["reserved_usd"],
                    3,
                ),
            },
        }
        if a.budget:
            par = plan["cost"]["parallelism"]
            out["budget_check"] = {
                "budget": a.budget,
                "effective_vcpu": par["effective_vcpu"],
                "expected_wall_minutes": par["expected_wall_minutes"],
                "fits_max_wall": par["fits_max_wall"],
            }
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_status(a, policy) -> int:
    out: Dict[str, Any] = {}
    if a.run_dir:
        state = Path(a.run_dir) / "state.json"
        out["state"] = json.loads(state.read_text()) if state.exists() else None
        receipt = Path(a.run_dir) / "receipt.json"
        out["receipt"] = json.loads(receipt.read_text()) if receipt.exists() else None
    out["live_runner_pods"] = [
        {k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr")}
        for p in RpClient().list_pods()
        if str(p.get("name", "")).startswith(policy["pod_name_prefix"])
    ]
    from research.runpod_fanout.serverless import load_sls_policy

    eprefix = load_sls_policy()["endpoint_prefix"]
    out["live_runner_endpoints"] = [
        {k: e.get(k) for k in ("id", "name", "workersMin", "workersMax")}
        for e in RpClient().list_endpoints()
        if str(e.get("name", "")).startswith(eprefix)
    ]
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


def cmd_cleanup(a, policy) -> int:
    rp = RpClient()
    prefix = pod_prefix(policy, a.job_id) if a.job_id else policy["pod_name_prefix"]
    if not a.job_id and not a.all_runner_pods:
        print("need --job-id ID or --all-runner-pods", file=sys.stderr)
        return 2
    peek = SharedLedger(policy).peek()
    live = peek["live_jobs"] if not a.job_id else [j for j in peek["live_jobs"] if j == a.job_id]
    if live and a.confirm and not a.force:
        print(f"refusing: live runs would lose pods: {live} (use --force)", file=sys.stderr)
        return 2
    pods = [p for p in rp.list_pods() if str(p.get("name", "")).startswith(prefix)]
    print(
        json.dumps(
            {
                "prefix": prefix,
                "pods": [{"id": p["id"], "name": p["name"]} for p in pods],
                "confirm": a.confirm,
            }
        )
    )
    for p in pods:
        rp.delete_pod(p["id"], confirm=a.confirm)
    # serverless runs of the same job(s): their endpoints too (scale to 0, then delete)
    from research.runpod_fanout import serverless

    eprefix = serverless.load_sls_policy()["endpoint_prefix"] + (
        f"{a.job_id}--" if a.job_id else ""
    )
    eps = [
        e for e in rp.list_endpoints() if str(e.get("name", "")).split(" ")[0].startswith(eprefix)
    ]
    print(json.dumps({"endpoint_prefix": eprefix, "endpoints": [e["id"] for e in eps]}))
    if a.confirm:
        bad = [e["id"] for e in eps if not serverless.teardown_endpoint(rp, e["id"])]
        left = [p for p in rp.list_pods() if str(p.get("name", "")).startswith(prefix)]
        print(json.dumps({"left": [p["id"] for p in left], "endpoints_left": bad}))
        return 0 if not left and not bad else 5
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("job", type=Path)
    pl.add_argument("--budget", type=float, default=None)
    pl.add_argument("--backend", choices=BACKENDS, default="serverless")
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    r.add_argument(
        "--backend",
        choices=BACKENDS,
        default="serverless",
        help="serverless (default for Tier-1 CPU jobs), pods (alias: runpod) or local",
    )
    r.add_argument("--budget", type=float, default=None)
    r.add_argument("--confirm", action="store_true")
    r.add_argument("--run-dir", type=Path, default=None)
    r.add_argument("--slot-timeout", type=float, default=3600.0)
    r.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        help="earlier run dir(s) of the SAME job: only units they did not complete run",
    )
    for sp_ in (pl, r):  # serverless sizing overrides (default: plan's recommendation)
        sp_.add_argument("--flavor", choices=("cpu5c", "cpu3c"), default=None)
        sp_.add_argument("--target-minutes", type=float, default=None)
        sp_.add_argument("--workers", type=int, default=None)
        sp_.add_argument("--vcpu-per-worker", type=int, default=None)
    st = sub.add_parser("status")
    st.add_argument("run_dir", nargs="?", type=Path)
    cl = sub.add_parser("cleanup")
    cl.add_argument("--job-id")
    cl.add_argument("--all-runner-pods", action="store_true")
    cl.add_argument("--confirm", action="store_true")
    cl.add_argument("--force", action="store_true", help="even while runs are live")
    cp = sub.add_parser("compare")
    cp.add_argument("a", type=Path)
    cp.add_argument("b", type=Path)
    mg = sub.add_parser("merge")
    mg.add_argument("runs", nargs="+", type=Path)
    mg.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    policy = jobspec.load_policy()
    if a.cmd == "status":
        return cmd_status(a, policy)
    if a.cmd == "cleanup":
        return cmd_cleanup(a, policy)
    if a.cmd == "compare":
        out = compare_runs(a.a, a.b)
        print(json.dumps(out, indent=1))
        return 0 if out["compared"] and not out["different"] else 1
    if a.cmd == "merge":
        print(json.dumps(merge_runs(a.runs, a.out), indent=1))
        return 0
    allow = jobspec.load_allowlist()
    job = jobspec.validate_job(json.loads(a.job.read_text()), policy, allow)
    backend = "pods" if a.backend == "runpod" else a.backend
    if a.cmd == "plan":
        if backend == "serverless":
            from research.runpod_fanout import serverless

            out = serverless.plan_serverless(
                job,
                policy,
                serverless.load_sls_policy(),
                allow,
                float(a.budget or 0),
                flavor=a.flavor,
                target_minutes=a.target_minutes,
                workers=a.workers,
                vcpu=a.vcpu_per_worker,
            )
            print(json.dumps(out, indent=1, sort_keys=True))
            return 0
        return cmd_plan(a, job, policy, allow)
    run_dir = a.run_dir or default_run_dir(
        policy, job, {"serverless": "sls", "pods": "runpod"}.get(backend, backend)
    )
    if backend == "local":
        return run_local(job, policy, allow, run_dir, a.slot_timeout, a.resume_from)
    if not a.confirm or a.budget is None:
        print(
            f"run --backend {backend} needs --budget USD and --confirm (dry run: use plan)",
            file=sys.stderr,
        )
        return 2
    if not 0 < a.budget <= float(policy["project_cap_usd"]):
        print("--budget must be in (0, project cap]", file=sys.stderr)
        return 2
    if backend == "serverless":
        from research.runpod_fanout import serverless

        return serverless.run_serverless(a, job, policy, allow, run_dir)
    runner = Runner(
        job, policy, allow, run_dir, budget=a.budget, confirm=True, resume_from=a.resume_from
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
