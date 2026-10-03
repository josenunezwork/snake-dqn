#!/usr/bin/env python3
"""RunPod fan-out runner for Tier-1 episodes (screens, sweeps, censuses). NOT for gates.

Commands (run from the repo with the project venv)::

    runner.py plan    JOB.json [--budget USD]            # dry run: capacity probe (read-only),
                                                         #   cost estimate, upload manifest
    runner.py run     JOB.json --budget USD --confirm    # real RunPod run (spends money)
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

from research.runpod_fanout import jobspec  # noqa: E402
from research.runpod_fanout.rp_client import RpClient, RunPodError  # noqa: E402

HERE = Path(__file__).resolve().parent
AGENT_SOURCE = HERE / "pod_agent.py"
from research.runpod_fanout.tls import USER_AGENT, preflight, ssl_context  # noqa: E402

STOCKED = ("High", "Medium", "Low")
RECEIPT_SCHEMA = "runpod-fanout-receipt/v1"
NET_ERRORS = (urllib.error.URLError, OSError, ValueError, http.client.HTTPException)


class Abort(RuntimeError):
    """Stop the run now (divergent duplicate, budget breach, ...); pods are deleted."""


def spec_sha256(ep: Mapping[str, Any]) -> str:
    """The episode spec hash stamped by episode.py (``fanout.spec_sha256``)."""
    return hashlib.sha256(
        json.dumps(ep, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pod_prefix(policy: Mapping[str, Any], job_id: str) -> str:
    return f"{policy['pod_name_prefix']}{job_id}--"


def episode_seconds(policy: Mapping[str, Any], ep: Mapping[str, Any]) -> float:
    t = policy["pod_seconds_per_episode"]
    eng = ep["engine"]
    return float(t[f"{eng}_fixed"]) + float(t[f"{eng}_per_frame"]) * int(ep["horizon"])


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
) -> Dict[str, Any]:
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
            "FANOUT_SELF_DELETE_EPOCH": str(
                int(deadline_epoch + float(policy["watchdog_grace_seconds"]) + 240)
            ),
            "FANOUT_JOB": job_id,
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
        self.platform_ids: set = set()
        self.watchdog: Any = None
        self.cpu_model_pin: Optional[str] = None

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
        """Spend if every live pod ran until the hard end (watchdog firing), plus slack.

        Slack: 2% and one extra minute per pod (billing starts before our timestamp).
        """
        total = 0.0
        for p in self.pods.values():
            end = p.deleted if p.deleted is not None else self.hard_end
            total += p.rate * (max(0.0, end - p.created) + 60.0) / 3600.0
        return total * 1.02

    def launch_allowed(self, rate: float) -> Optional[str]:
        """None if a new pod at ``rate`` fits the job budget and the global floor."""
        now = self.clock()
        extra = 1.02 * rate * (max(0.0, self.hard_end - now) + 60.0) / 3600.0
        worst = self.committed_worst() + extra
        if worst > self.budget + 1e-9:
            return f"job budget: worst case {worst:.4f} > budget {self.budget:.4f}"
        if self.balance_start is not None:
            floor = float(self.policy["global_floor_usd"])
            if self.balance_start - worst < floor:
                return f"global floor: balance {self.balance_start:.2f} - {worst:.4f} < {floor}"
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

    def create_pod(self, row: Mapping[str, Any]) -> bool:
        self.pod_counter += 1
        name = (
            f"{pod_prefix(self.policy, self.job['job_id'])}{self.run_dir.name}-{self.pod_counter}"
        )
        token = secrets.token_urlsafe(32)
        body = pod_body(
            self.policy,
            name,
            row["vcpu"],
            row["dc"],
            hashlib.sha256(token.encode()).hexdigest(),
            self.deadline,
            self.job["job_id"],
        )
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
            self.reap_by_name(name, row)
            self.raise_deferred()
            return False
        except BaseException:
            _DEFER["active"] = False
            raise
        _DEFER["active"] = False
        if not isinstance(out, dict) or not out.get("id"):
            self.log("create_no_pod", detail=str(out)[:300])
            self.reap_by_name(name, row)
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
            created=self.clock(),
        )
        pod.boot_from = pod.created
        self.pods[pod.id] = pod
        self.agents[pod.id] = self.agent_factory(pod.id, token)
        (self.run_dir / "pods" / f"{name}.response.json").write_text(
            json.dumps({k: v for k, v in out.items() if k != "env"}, indent=1)
        )
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
        if pod.state != "lost":
            pod.state = "deleted"
        self.log("pod_deleted", pod=pod.id, why=why, est_cost=round(pod.cost_until(pod.deleted), 5))

    def requeue(self, pod: Pod, keys: Iterable[str], why: str) -> None:
        for key in sorted(keys):
            pod.inflight.discard(key)
            pod.unpulled.discard(key)
            if key in self.completed or key in self.failed or key in self.pending:
                continue
            self.attempts[key] += 1
            if self.attempts[key] >= int(self.policy["max_attempts_per_episode"]):
                self.failed[key] = {"why": why, "attempts": self.attempts[key]}
                self.log("episode_failed", key=key, why=why)
            else:
                self.pending.insert(0, key)
                self.log("episode_requeued", key=key, why=why)

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
        for key, info in (health.get("failed") or {}).items():
            if key in pod.inflight:
                self.log("episode_error", pod=pod.id, key=key, info=info)
                (self.run_dir / "failures").mkdir(exist_ok=True)
                fname = key.replace("/", "__") + f".{pod.id}.{self.attempts[key]}.json"
                (self.run_dir / "failures" / fname).write_text(json.dumps(info, indent=1))
                self.requeue(pod, [key], f"episode error rc={info.get('rc')}")
        done = set(health.get("done") or {}) & pod.inflight
        pod.inflight -= done
        pod.unpulled |= done
        known = set(health.get("queued") or []) | set(health.get("running") or [])
        orphans = pod.inflight - known
        if orphans:  # the agent no longer knows this work (should not happen)
            self.requeue(pod, orphans, "agent lost the episode")
        due = now - pod.last_pull >= float(self.policy["pull_every_seconds"])
        all_remaining_here = self.outstanding() <= len(pod.unpulled)
        if pod.unpulled and (due or pod.state == "draining" or all_remaining_here):
            self.pull(pod)
        if pod.state == "draining" and not pod.inflight and not pod.unpulled:
            self.delete_pod(pod, "retired (drained)")

    def tick_booting(self, pod: Pod, agent: Any, health: Mapping[str, Any], now: float) -> None:
        if health.get("pip") == "failed":
            pod.state = "lost"
            self.bad_slots[(pod.vcpu, pod.dc)] = now + 1800
            self.delete_pod(pod, "pip install failed")
            return
        pod.cpu_model = health.get("cpu_model")
        pinned = self.pinned_cpu_model()
        if pinned and pod.cpu_model != pinned:
            # One run = one platform: refuse hosts with another CPU model (MKL code paths).
            pod.state = "lost"
            self.bad_slots[(pod.vcpu, pod.dc)] = now + 4 * 3600
            self.log("cpu_model_mismatch", pod=pod.id, got=pod.cpu_model, pinned=pinned)
            self.delete_pod(pod, "cpu model differs from the run's pinned model")
            return
        try:
            self.bring_up(pod, agent, health)
            if health.get("setup") is None:
                health = agent.get_json("/health")
        except NET_ERRORS as exc:
            self.log("upload_retry", pod=pod.id, detail=str(exc)[:160])
        if health.get("pip") == "ok" and health.get("setup") is not None:
            if not self.pinned_cpu_model():
                self.cpu_model_pin = pod.cpu_model
                self.log("cpu_model_pinned", cpu_model=pod.cpu_model)
            pod.state = "ready"
            self.log(
                "pod_ready",
                pod=pod.id,
                slots=pod.slots,
                cores=health.get("cores"),
                cpu_model=health.get("cpu_model"),
                self_delete_capable=health.get("self_delete_capable"),
                seconds_to_ready=round(now - pod.created, 1),
            )
        elif now - pod.boot_from > float(self.policy["ready_deadline_seconds"]):
            pod.state = "lost"
            self.bad_slots[(pod.vcpu, pod.dc)] = now + 1800
            self.delete_pod(pod, "not ready before the ready deadline")
            self.requeue(pod, pod.inflight | pod.unpulled, "pod lost")

    def pinned_cpu_model(self) -> Optional[str]:
        return getattr(self, "cpu_model_pin", None)

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
            if (key, row["sha256"]) in self.verified:
                pod.unpulled.discard(key)
                continue
            try:
                data = agent.get_bytes("/record/" + key)
            except NET_ERRORS as exc:
                self.log("pull_record_failed", pod=pod.id, key=key, detail=str(exc)[:120])
                continue
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                self.log("pull_sha_mismatch", pod=pod.id, key=key)
                continue
            self.accept_record(key, data, source=pod.id)
            self.verified.add((key, row["sha256"]))
            pod.unpulled.discard(key)
            pod.inflight.discard(key)
            got += 1
        if got:
            self.log(
                "pulled", pod=pod.id, new=got, completed=len(self.completed), of=len(self.order)
            )
            self.write_state()

    def accept_record(self, key: str, data: bytes, source: str) -> None:
        """Validate and write-once a record; a duplicate must match deterministically."""
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
        self.platform_ids.add(platform_id)
        if len(self.platform_ids) > 1:
            raise Abort(f"mixed platforms inside one run: {sorted(self.platform_ids)}")
        dest = self.run_dir / "records" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            old = json.loads(dest.read_bytes())
            if jobspec.deterministic_bytes(old) != jobspec.deterministic_bytes(entry):
                dup = self.run_dir / "divergent" / (key.replace("/", "__") + f".{source}.json")
                dup.parent.mkdir(exist_ok=True)
                dup.write_bytes(data)
                raise Abort(f"DIVERGENT duplicate for {key} from {source}")
            self.log("duplicate_verified", key=key, source=source)
        else:
            tmp = dest.with_name(f".{dest.name}.tmp")
            tmp.write_bytes(data)
            os.link(tmp, dest)
            tmp.unlink()
        self.completed[key] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if key in self.pending:
            self.pending.remove(key)

    def assign(self) -> None:
        for pod in self.pods.values():
            if pod.state != "ready" or not self.pending:
                continue
            free = pod.slots - len(pod.inflight)
            if free <= 0:
                continue
            batch = [k for k in self.pending if k not in self.completed][:free]
            if not batch:
                continue
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
            for key in out.get("accepted", []):
                if key in self.pending:
                    self.pending.remove(key)
                pod.inflight.add(key)
            self.log("assigned", pod=pod.id, keys=out.get("accepted", []))

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
        prefix = self.run_prefix()
        return [p for p in self.rp.list_pods() if str(p.get("name", "")).startswith(prefix)]

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
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "job.json").write_text(json.dumps(self.job, indent=1, sort_keys=True))
        self.events = (self.run_dir / "events.jsonl").open("x")
        workdir = Path(tempfile.mkdtemp(prefix="rpf-upload-"))
        leftovers: List[Dict[str, Any]] = [{"unknown": True}]
        old_handlers = install_signal_handlers()
        code = 1
        lock = None
        awake = None
        try:
            lock = acquire_global_lock(self.policy)
            ledger = project_ledger(self.policy)
            cap = float(self.policy["project_cap_usd"])
            if ledger["total_usd"] + self.budget > cap:
                raise Abort(
                    f"project cap: spent {ledger['total_usd']:.4f} + budget {self.budget} > {cap}"
                )
            reserve_ledger(self.policy, self.run_dir, self.job["job_id"], self.budget)
            self.log("project_ledger", **ledger)
            self.tls_preflight()
            manifest = self.prepare_uploads(workdir)
            self.balance_start = self.rp.balance()
            if self.balance_start - self.budget < float(self.policy["global_floor_usd"]):
                raise Abort(
                    f"balance {self.balance_start:.2f} - budget {self.budget:.2f} would go "
                    f"below the ${self.policy['global_floor_usd']} floor"
                )
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
                if lock is not None:
                    leftovers = self.delete_everything()
                else:  # never got the lock: nothing was created by this run
                    leftovers = []
                shutil.rmtree(workdir, ignore_errors=True)
                self.finish(leftovers, code)
                settle_ledger(self.policy, self.run_dir, self.settled_cost(), lock is not None)
            finally:
                if awake is not None:
                    awake.terminate()
                if lock is not None:
                    lock.close()
                restore_signal_handlers(old_handlers)
        return code if not leftovers else 5

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
        eps = list(self.episodes.values())
        total = sum(episode_seconds(self.policy, ep) for ep in eps)
        longest = max(episode_seconds(self.policy, ep) for ep in eps)
        size = size_cap_for(self.policy, len(eps))
        rate = 0.03 * size
        if sizes_rows:
            fits = [r for r in sizes_rows if r["vcpu"] <= size]
            if fits:
                size, rate = fits[0]["vcpu"], fits[0]["usd_per_hr"]
        workers = workers_for(self.policy, size)
        waves = math.ceil(len(eps) / workers)
        wall = float(self.policy["pod_setup_seconds"]) + max(longest, waves * total / len(eps))
        max_wall_h = float(self.job["max_wall_minutes"]) / 60.0
        hard_h = max_wall_h + (float(self.policy["watchdog_grace_seconds"]) + 300) / 3600
        return {
            "cost": {
                "episodes": len(eps),
                "pod_episode_seconds_total_est": round(total, 1),
                "first_pod": {"vcpu": size, "workers": workers, "usd_per_hr": rate},
                "expected_wall_minutes_one_pod": round(wall / 60, 1),
                "expected_cost_usd_one_pod": round(rate * wall / 3600, 4),
                "worst_case_first_pod_usd": round(rate * hard_h, 4),
                "budget_usd_hard_cap": self.budget,
                "note": "the runner never lets committed worst case (every live pod billed to "
                "the watchdog hard end) exceed the budget, nor the balance drop below the floor",
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
            "job_id": self.job["job_id"],
            "run_dir": str(self.run_dir),
            "exit_code": code,
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


def acquire_global_lock(policy: Mapping[str, Any]) -> Any:
    """One live RunPod fan-out run per machine (shared balance, floor and cap checks)."""
    root = Path(policy["artifacts_root"]) / "runpod-fanout"
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".runpod-runner.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise Abort("another RunPod fan-out run is live (global lock held)")
    return handle


def ledger_path(policy: Mapping[str, Any]) -> Path:
    return Path(policy["artifacts_root"]) / "runpod-fanout" / "ledger.jsonl"


def _ledger_append(policy: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    path = ledger_path(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"utc": utc_now(), **row}, sort_keys=True) + "\n")


def reserve_ledger(policy: Mapping[str, Any], run_dir: Path, job_id: str, budget: float) -> None:
    _ledger_append(
        policy,
        {"kind": "reserve", "run": str(run_dir), "job_id": job_id, "budget_usd": float(budget)},
    )


def settle_ledger(policy: Mapping[str, Any], run_dir: Path, cost: float, reserved: bool) -> None:
    if reserved:
        _ledger_append(policy, {"kind": "settle", "run": str(run_dir), "cost_usd": float(cost)})


def project_ledger(policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Project spend = prior spend + settled run costs + FULL budgets of unsettled runs.

    One central ``ledger.jsonl`` (any --run-dir); a run that died before settling keeps
    counting at its whole budget until a human settles it.
    """
    reserved: Dict[str, float] = {}
    settled: Dict[str, float] = {}
    path = ledger_path(policy)
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "reserve":
                reserved[row["run"]] = float(row["budget_usd"])
            elif row["kind"] == "settle":
                settled[row["run"]] = float(row["cost_usd"])
    open_runs = {r: b for r, b in reserved.items() if r not in settled}
    prior = float(policy.get("prior_spend_usd", 0.0))
    total = prior + sum(settled.values()) + sum(open_runs.values())
    return {
        "prior_usd": prior,
        "settled_runs": len(settled),
        "unsettled_runs": len(open_runs),
        "total_usd": round(total, 5),
    }


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
            runner = Runner(job, policy, allowlist, run_dir, rp=_NoRunPod(), confirm=False)
            env = dict(
                os.environ,
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                SNAKE_DQN_DEVICE="cpu",
                PYTHONHASHSEED="0",
            )
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
    merged: Dict[str, bytes] = {}
    platforms = set()
    for d in run_dirs:
        root = Path(d) / "records"
        for p in sorted(root.rglob("*.json")):
            if p.name.startswith("."):
                continue
            key = str(p.relative_to(root))
            data = p.read_bytes()
            entry = json.loads(data)
            platforms.add((entry.get("platform") or {}).get("platform_id", "MISSING"))
            if key in merged and jobspec.deterministic_bytes(
                json.loads(merged[key])
            ) != jobspec.deterministic_bytes(entry):
                raise Abort(f"merge: {key} differs between runs")
            merged.setdefault(key, data)
    if len(platforms) != 1 or "MISSING" in platforms:
        raise Abort(f"merge refuses mixed or missing platforms: {sorted(platforms)}")
    Path(out).mkdir(parents=True)
    for key, data in merged.items():
        dest = Path(out) / "records" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as fh:
            fh.write(data)
    return {"records": len(merged), "platform_id": platforms.pop(), "out": str(out)}


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
            "pod_request_example": {
                k: v
                for k, v in pod_body(
                    policy,
                    pod_prefix(policy, job["job_id"]) + "x-1",
                    plan["cost"]["first_pod"]["vcpu"],
                    "<dc>",
                    "<sha256 " "of an ephemeral token>",
                    0,
                    job["job_id"],
                ).items()
                if k != "dockerStartCmd"
            },
        }
        if a.budget:
            worst = plan["cost"]["worst_case_first_pod_usd"]
            out["budget_check"] = {
                "budget": a.budget,
                "first_pod_worst_fits": worst <= a.budget,
                "balance_minus_budget_above_floor": balance - a.budget
                >= policy["global_floor_usd"],
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
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


def cmd_cleanup(a, policy) -> int:
    rp = RpClient()
    prefix = pod_prefix(policy, a.job_id) if a.job_id else policy["pod_name_prefix"]
    if not a.job_id and not a.all_runner_pods:
        print("need --job-id ID or --all-runner-pods", file=sys.stderr)
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
    if a.confirm:
        left = [p for p in rp.list_pods() if str(p.get("name", "")).startswith(prefix)]
        print(json.dumps({"left": [p["id"] for p in left]}))
        return 0 if not left else 5
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("job", type=Path)
    pl.add_argument("--budget", type=float, default=None)
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    r.add_argument("--backend", choices=("runpod", "local"), default="runpod")
    r.add_argument("--budget", type=float, default=None)
    r.add_argument("--confirm", action="store_true")
    r.add_argument("--run-dir", type=Path, default=None)
    r.add_argument("--slot-timeout", type=float, default=3600.0)
    st = sub.add_parser("status")
    st.add_argument("run_dir", nargs="?", type=Path)
    cl = sub.add_parser("cleanup")
    cl.add_argument("--job-id")
    cl.add_argument("--all-runner-pods", action="store_true")
    cl.add_argument("--confirm", action="store_true")
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
    if a.cmd == "plan":
        return cmd_plan(a, job, policy, allow)
    run_dir = a.run_dir or default_run_dir(policy, job, a.backend)
    if a.backend == "local":
        return run_local(job, policy, allow, run_dir, a.slot_timeout)
    if not a.confirm or a.budget is None:
        print(
            "run --backend runpod needs --budget USD and --confirm (dry run: use plan)",
            file=sys.stderr,
        )
        return 2
    if not 0 < a.budget <= float(policy["project_cap_usd"]):
        print("--budget must be in (0, project cap]", file=sys.stderr)
        return 2
    runner = Runner(job, policy, allow, run_dir, budget=a.budget, confirm=True)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
