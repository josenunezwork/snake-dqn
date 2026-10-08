#!/usr/bin/env python3
"""M3-B on ONE RunPod RTX 4090 pod (secure): 5 seed trainers in parallel, money-safe.

Approved plan (relayed 2026-10-08; the spend itself is only ever sent with ``--confirm``):
one RTX 4090, secure cloud (~$0.74/h); upload the M3 inputs (sha-pinned); G2 gate = the
first 30 minutes of the real run measure >= 5.6k transitions/s combined (4 x the Mac's
1.39k) AND a projected finish inside the pod lifetime, else stop and fall back to the Mac.

Money safety:
* ledger: a run registered in the shared fan-out ledger (``research/runpod_fanout``) with a
  pod reservation for the worst case (rate x lifetime) inside the project cap;
* hard spend cap ``--spend-cap`` (default $8): the account balance is polled; a drop of more
  than the cap since launch stops the trainers and deletes the pod;
* pod lifetime ``--lifetime-hours`` (default 7): the agent kills jobs 15 min before it, the
  runner deletes the pod at it, a detached watchdog deletes it 5 min later if the runner is
  gone, and the pod deletes itself with its pod-scoped credentials;
* ``cleanup --confirm`` deletes the recorded pod; every exit path deletes the pod and
  verifies ``GET /pods`` holds none of ours.

Subcommands: ``plan`` (dry run: inputs, pod body, rp.py dry-run request; spends nothing),
``launch --confirm``, ``cleanup --confirm``.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

AGENT = Path(__file__).resolve().parent / "gpu_agent.py"
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
SEEDS = (0, 1, 2, 3, 4)
TRANSITIONS = 20_000_000
MAC_RATE = 1388.7248756266617  # M3-A measured end-to-end (results/m3a/summary.json)
G2_RATE = 5600.0  # >= 4 x MAC_RATE, rounded up (approved wording: ">= 5.6k combined")
G2_SECONDS = 1800
CHUNK = 16 << 20  # the RunPod proxy cuts requests at ~100 s
POD_NAME = "m3b-gpu-"
MAX_HOURLY = 1.0
PORT = 8000
MIN_RAM_GB = 72  # 5 trainers x ~12 GB RSS (+ CUDA / numba / page cache)
MIN_VCPU = 10  # 5 single-thread trainers + prefetch threads + the agent
UPLOAD_BUDGET_S = 3600
SETUP_BUDGET_S = 1800
STALL_S = 900
NET_FAIL_BUDGET_S = 900
CREATE_TRIES = 6
G2_ROWS_GRACE_S = 600  # G2 fails if some seed has still not logged a row by then
CREATE_SETTLE_S = 300  # a fresh pod can be missing from GET /pods for minutes


# ----------------------------------------------------------------------------- pure parts
def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pod_body(name: str, token_sha: str, deadline: float, until: float) -> Dict[str, Any]:
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
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "minRAMPerGPU": MIN_RAM_GB,
        "minVCPUPerGPU": MIN_VCPU,
        "ports": [f"{PORT}/http"],
        "env": {
            "M3_TOKEN_SHA256": token_sha,
            "M3_DEADLINE_EPOCH": str(int(deadline)),
            "M3_SELF_DELETE_EPOCH": str(int(until)),
        },
        "dockerStartCmd": ["bash", "-c", start],
    }


def g2_decision(
    rates: Dict[int, float],
    done: Dict[int, float],
    remaining_lifetime: float,
    dead: Sequence[int] = (),
) -> Dict[str, Any]:
    """G2 at ~30 min: all 5 seeds alive and logging, combined rate >= 5.6k/s, AND the
    slowest seed's projected finish + 45 min fits in the remaining lifetime."""
    live = {s: r for s, r in rates.items() if s not in dead}
    combined = float(sum(live.values()))
    need = 0.0
    for s in SEEDS:
        r = live.get(s, 0.0)
        left = max(0.0, TRANSITIONS - float(done.get(s, 0.0)))
        need = max(need, left / r if r > 0 else float("inf"))
    ok_rate = not dead and len(live) == len(SEEDS) and combined >= G2_RATE
    ok_time = need + 45 * 60 <= remaining_lifetime
    return {
        "combined_rate": combined,
        "per_seed": rates,
        "dead_seeds": list(dead),
        "ratio_vs_mac": combined / MAC_RATE,
        "projected_seconds_left": need,
        "remaining_lifetime": remaining_lifetime,
        "pass": bool(ok_rate and ok_time),
        "reason": None if ok_rate and ok_time else ("rate" if not ok_rate else "lifetime"),
    }


def spend_exceeded(balance0: float, balance: float, cap: float, margin: float = 0.5) -> bool:
    return balance0 - balance >= cap - margin


# ----------------------------------------------------------------------------- inputs
def build_inputs(work: Path) -> Dict[str, Any]:
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
    }


def setup_argv(demo_sha: str) -> List[str]:
    script = " && ".join(
        [
            "set -e",
            "mkdir -p /r/repo /r/demo",
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
            "import research.redesign_m3_20261008.train_m3; "
            "import research.redesign_m2_20261008.gen_data; "
            "import src.simd_env.grid_sim, src.simd_env.fast_anchors; "
            "import src.simd_env.vector61_policy, src.simd_env.ego2s_policy; "
            "import src.simd_env.ego_raster, src.simd_env.ego_raster_b; "
            "import src.simd_env.ego_raster_nb, src.simd_env.grid_sim_nb; "
            "import src.simd_env.eval_engine, src.model.ego2s_network, src.core.world_runtime'",
            "echo SETUP_OK",
        ]
    )
    return ["bash", "-c", script]


def train_argv(seed: int) -> List[str]:
    return [
        "python",
        "research/redesign_m3_20261008/train_m3.py",
        "--seed",
        str(seed),
        "--init",
        "/r/in/ckpt/student_m2b_final.pth",
        "--demo-data",
        "/r/demo",
        "--transitions",
        str(TRANSITIONS),
        "--device",
        "cuda",
        "--out",
        f"/r/out/m3b_seed{seed}",
    ]


# ----------------------------------------------------------------------------- agent client
_SSL = []


def _ssl():
    if not _SSL:
        from research.runpod_fanout.tls import ssl_context

        _SSL.append(ssl_context())
    return _SSL[0]


class Agent:
    def __init__(self, pod_id: str, token: str) -> None:
        self.base = f"https://{pod_id}-{PORT}.proxy.runpod.net"
        self.token = token

    def _req(self, method: str, path: str, data: Optional[bytes] = None, headers=None, timeout=60):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("X-M3-Token", self.token)
        req.add_header("User-Agent", "m3-gpu-runner/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as resp:
            return resp.read()

    def health(self) -> Dict[str, Any]:
        return json.loads(self._req("GET", "/health", timeout=20))

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


# ----------------------------------------------------------------------------- launch
def now() -> float:
    return time.time()


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    tmp = state_path(run_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
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


def delete_and_verify(rp, run_dir: Path, state: Dict[str, Any], reason: str) -> bool:
    """Delete our pod(s) with backoff for up to ~30 min; True iff GET /pods shows none."""
    deadline = now() + 1800
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
        if now() > deadline:
            state.update(delete_reason=reason, delete_failed=True)
            save_state(run_dir, state)
            alert(f"pod(s) may still be running: {state.get('pods_left')}; run `cleanup --confirm`")
            return False
        time.sleep(15 if (not state.get("pods_left") and clean_since) else wait)
        wait = min(120.0, wait * 2)


WATCHDOG = r"""
import json, os, sys, time
sys.path.insert(0, {repo!r})
from research.runpod_fanout.rp_client import RpClient
log = open({log!r}, "a")
t = {fire!r}
log.write(json.dumps({{"watchdog": "armed", "pid": os.getpid(), "fire": t}}) + "\n")
log.flush()
while time.time() < t:  # short sleeps: a long sleep does not count macOS sleep time
    time.sleep(60)
rp = RpClient()
end = time.time() + 1800
while time.time() < end:
    try:
        pods = [p for p in rp.list_pods() if str(p.get("name", "")).startswith({prefix!r})]
        if not pods:
            log.write(json.dumps({{"watchdog": "clean", "t": time.time()}}) + "\n")
            break
        for p in pods:
            rp.delete_pod(p["id"], confirm=True)
            log.write(json.dumps({{"watchdog": "deleted", "id": p["id"], "t": time.time()}}) + "\n")
    except Exception as exc:
        log.write(json.dumps({{"watchdog": "error", "error": repr(exc), "t": time.time()}}) + "\n")
    log.flush()
    time.sleep(60)
"""


def watchdog_code(run_dir: Path, until: float) -> str:
    return WATCHDOG.format(
        repo=str(REPO), log=str(run_dir / "watchdog.log"), fire=until + 300, prefix=POD_NAME
    )


def spawn_watchdog(run_dir: Path, until: float, wait_s: float = 60.0) -> int:
    """Start the detached watchdog; return its pid only once it is alive and logged "armed"."""
    compile(watchdog_code(run_dir, until), "<watchdog>", "exec")  # SyntaxError here, not later
    caff = ["caffeinate", "-i"] if sys.platform == "darwin" else []
    proc = subprocess.Popen(
        caff + [sys.executable, "-c", watchdog_code(run_dir, until)],
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
    raise SystemExit("refusing: the watchdog did not log 'armed' (it is left running harmlessly)")


def stocked_data_centers(rp) -> List[str]:
    """Secure data centers listing RTX 4090 stock right now (read-only GraphQL)."""
    q = "query { dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"
    try:
        out = rp._call("gql", q)
    except Exception:  # noqa: BLE001
        return []
    rank = {"High": 0, "Medium": 1, "Low": 2}
    hits = []
    for dc in out.get("dataCenters") or []:
        for g in dc.get("gpuAvailability") or []:
            if g.get("gpuTypeId") == GPU and g.get("stockStatus") in rank:
                hits.append((rank[g["stockStatus"]], dc["id"]))
    return [d for _, d in sorted(hits)]


def error_detail(exc: BaseException) -> str:
    """rp.py's own output for a failed call (RunPod's error body; rp.py never prints the key)."""
    payload = getattr(exc, "payload", None)
    text = json.dumps(payload, default=str) if payload is not None else ""
    return f"{exc} {text}".strip()[:1500]


def definite_refusal(exc: BaseException) -> bool:
    """True when RunPod answered with an HTTP error (the pod was NOT created), so a retry
    cannot duplicate a pod; a timeout or unparsable answer is never retried."""
    payload = getattr(exc, "payload", None)
    status = payload.get("http_status") if isinstance(payload, dict) else None
    return isinstance(status, int) and 400 <= status < 600


def create_pod_retrying(rp, body: Dict[str, Any], run_dir: Path, state: Dict[str, Any]):
    """POST /pods; on a definite refusal (e.g. no capacity) retry pinned to the data centers
    that list stock, adopting any pod of this name that shows up first. Returns (out, error)."""
    attempts = state.setdefault("create_attempts", [])
    dcs = stocked_data_centers(rp)
    plans: List[Optional[List[str]]] = [None] + [[d] for d in dcs] + ([dcs] if dcs else [])
    err = ""
    for i, plan_dcs in enumerate(plans[:CREATE_TRIES]):
        if i:
            time.sleep(20)
            if [p for p in our_pods(rp) if p.get("name") == body["name"]]:
                return None, err  # the caller adopts it by name
        b = dict(body)
        if plan_dcs:
            b["dataCenterIds"] = plan_dcs
        path = run_dir / f"pod_body_try{i}.json"
        path.write_text(json.dumps(b))
        state["create_attempted_at"] = now()
        save_state(run_dir, state)
        try:
            out = rp.create_pod(path, MAX_HOURLY, confirm=True)
            attempts.append({"try": i, "dcs": plan_dcs, "ok": True})
            save_state(run_dir, state)
            return out, ""
        except Exception as exc:  # noqa: BLE001
            err = error_detail(exc)
            attempts.append({"try": i, "dcs": plan_dcs, "error": err})
            save_state(run_dir, state)
            say(f"create try {i} ({plan_dcs or 'any DC'}): {err}")
            if not definite_refusal(exc):
                return None, err  # unknown outcome: adopt-or-give-up, never re-POST
    return None, err


def latest_rates(agent: Agent, tries: int = 3) -> Dict[int, Dict[str, float]]:
    rows = {}
    for s in SEEDS:
        for _ in range(tries):
            try:
                lines = agent.get(f"/out/m3b_seed{s}/log.jsonl").decode().strip().splitlines()
                row = json.loads(lines[-1])
                rows[s] = {
                    "rate": float(row["transitions"]) / float(row["wall"]),
                    "transitions": float(row["transitions"]),
                }
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
    return rows


def download(agent: Agent, run_dir: Path, budget_s: float = 900) -> Dict[str, Any]:
    """Best effort, time-bounded, per-file retries, paths confined, sha-checked."""
    dest = (run_dir / "out").resolve()
    t0, got, failed = now(), 0, []
    try:
        listing = json.loads(agent.get("/ls/out"))
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
        for _ in range(3):
            try:
                data = agent.get("/out/" + row["path"], timeout=300)
                if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                    raise ValueError("sha mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                got += 1
                break
            except Exception:  # noqa: BLE001
                time.sleep(3)
        else:
            failed.append(row["path"])
    logs = dest.parent / "pod_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in ["agent", "setup"] + [f"seed{s}" for s in SEEDS]:
        try:
            (logs / f"{name}.log").write_bytes(agent.get(f"/log/{name}"))
        except Exception:  # noqa: BLE001
            pass
    return {"files": got, "failed": failed}


def download_all(agent: Agent, run_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Retry failed files while the pod still has time (it self-deletes at ``until``)."""
    t0 = now()
    dl = download(agent, run_dir)
    rounds = 1
    while (dl.get("failed") or dl.get("error")) and rounds < 3:
        if now() > min(state["until"] - 300, t0 + 1800):
            break
        time.sleep(30)
        dl = download(
            agent, run_dir, budget_s=max(60.0, min(state["until"] - 300, t0 + 1800) - now())
        )
        rounds += 1
    return dl


def _raise_interrupt(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt(f"signal {signum}")


def launch(args) -> int:
    from research.runpod_fanout.ledger import SharedLedger
    from research.runpod_fanout.rp_client import RpClient
    from research.runpod_fanout.tls import preflight

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)
    if sys.platform == "darwin":  # keep the Mac awake while this process lives
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], start_new_session=True)
    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    rp = RpClient()
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        print(f"refusing: {run_dir} exists", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    problems = preflight()
    if problems:
        raise SystemExit(f"TLS preflight failed: {problems}")
    inputs = build_inputs(run_dir / "inputs")
    if our_pods(rp):
        raise SystemExit("refusing: a pod of ours already exists")
    balance0 = rp.balance()
    token = secrets.token_urlsafe(32)
    created = now()
    until = created + args.lifetime_hours * 3600
    deadline = until - 15 * 60
    name = POD_NAME + time.strftime("%m%d%H%M%S")
    body = pod_body(name, hashlib.sha256(token.encode()).hexdigest(), deadline, until)
    body_path = run_dir / "pod_body.json"
    body_path.write_text(json.dumps(body))
    # The watchdog must be alive and armed before anything can cost money.
    watchdog_pid = spawn_watchdog(run_dir, until)
    ledger = SharedLedger(policy)
    run_id = f"m3b-gpu-{int(created)}"
    ledger.register_run(run_id, "redesign-m3b-gpu", str(run_dir), args.spend_cap, os.getpid())
    refusal = ledger.reserve_pod(run_id, name, MAX_HOURLY, until, balance0)
    if refusal:
        ledger.finish_run(run_id, False, False)
        raise SystemExit(f"ledger refused: {refusal}")
    state: Dict[str, Any] = {
        "run_id": run_id,
        "name": name,
        "balance0": balance0,
        "created": created,
        "until": until,
        "deadline": deadline,
        "commit": inputs["commit"],
        "spend_cap": args.spend_cap,
        "token_sha256": body["env"]["M3_TOKEN_SHA256"],
    }
    save_state(run_dir, state)
    # From here on every exit path deletes whatever pod of ours exists (by name prefix).
    state["watchdog_pid"] = watchdog_pid
    save_state(run_dir, state)
    outcome, rate = "unknown", MAX_HOURLY
    try:
        out, err = create_pod_retrying(rp, body, run_dir, state)
        if err:
            outcome = f"create error: {err}"
        pod_id = out.get("id") if isinstance(out, dict) else None
        if not pod_id:
            for _ in range(10):  # adopt a pod created despite the error, by its unique name
                hits = [p for p in our_pods(rp) if p.get("name") == name]
                if hits:
                    pod_id = hits[0]["id"]
                    break
                time.sleep(30)
        if not pod_id:
            outcome = outcome if outcome != "unknown" else f"create failed: {out}"
            return 1
        rate = float((out or {}).get("costPerHr") or args.rate)
        state.update(pod_id=pod_id, rate=rate, machine=(out or {}).get("machine"))
        save_state(run_dir, state)
        ledger.bind_pod(run_id, name, pod_id, rate)
        say(json.dumps({"pod_id": pod_id, "rate": rate, "until": until}))
        outcome = drive(Agent(pod_id, token), rp, run_dir, state, inputs, args)
    except BaseException as exc:  # noqa: BLE001  (incl. KeyboardInterrupt / SIGTERM / SIGHUP)
        outcome = f"error: {type(exc).__name__}: {exc}"
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the delete
        state["outcome"] = outcome
        save_state(run_dir, state)
        ok = delete_and_verify(rp, run_dir, state, outcome)
        try:
            cost = rate * (now() - created) / 3600.0
            ledger.release_pod(run_id, name, cost)
            ledger.finish_run(run_id, outcome == "complete", outcome == "complete")
            state.update(cost_estimate=cost, balance_end=rp.balance())
        except Exception as exc:  # noqa: BLE001
            alert(f"ledger/balance bookkeeping failed: {exc}")
        state["pods_deleted_verified"] = ok
        save_state(run_dir, state)
    return 0 if outcome == "complete" and state.get("pods_deleted_verified") else 1


def guard(rp, state, args) -> Optional[str]:
    if now() >= state["deadline"]:  # the agent kills jobs here; download before self-delete
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


def drive(agent, rp, run_dir, state, inputs, args) -> str:
    log = run_dir / "progress.jsonl"
    net = NetBudget()

    def note(**row):
        row["t"] = now()
        with log.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
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
    note(
        stage="agent_up",
        **{
            k: h.get(k)
            for k in (
                "gpu",
                "cpus",
                "cpus_allowed",
                "cgroup_cpus",
                "cgroup_memory_gb",
                "self_delete_armed",
            )
        },
    )
    if not h.get("self_delete_armed") or h.get("self_delete_probe") != 200:
        return f"pod self-delete not proven (probe {h.get('self_delete_probe')})"
    mems = [m for m in (h.get("cgroup_memory_gb"), h.get("mem_total_gb")) if m is not None]
    mem = min(mems) if mems else 0.0
    cpus = min(
        c
        for c in (h.get("cgroup_cpus"), h.get("cpus_allowed"), h.get("cpus"), 1e9)
        if c is not None
    )
    if mem < MIN_RAM_GB - 2 or cpus < MIN_VCPU:
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
    agent.exec("setup", setup_argv(inputs["demo_sha256"]), "/r", {})
    t0 = now()
    while True:
        stop = guard(rp, state, args)
        if stop:
            return stop
        if now() - t0 > SETUP_BUDGET_S:
            return "setup timeout"
        time.sleep(20)
        try:
            procs = agent.health()["procs"]
            net.ok()
        except Exception:  # noqa: BLE001
            if net.fail():
                return "lost the agent during setup"
            continue
        rc = procs.get("setup", {}).get("returncode")
        if rc is not None:
            tail = agent.get("/log/setup").decode(errors="replace")[-1200:]
            note(stage="setup_done", returncode=rc, tail=tail)
            if rc != 0 or "SETUP_OK" not in tail:
                return "setup failed"
            break
    # 4. trainers; a failing seed does not stop the others
    env = {
        "SNAKE_CKPT_DIR": "/r/in/ckpt",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "M3_COMMIT": inputs["commit"],
    }
    for s in SEEDS:
        agent.exec(f"seed{s}", train_argv(s), "/r/repo", env)
    started = now()
    note(stage="trainers_started")
    g2, last_progress, best, uptime = None, now(), {}, 0.0
    while True:
        stop = guard(rp, state, args)
        if stop:
            note(stage="stopping", reason=stop, download=download(agent, run_dir))
            return stop
        time.sleep(60)
        try:
            h = agent.health()
            net.ok()
        except Exception:  # noqa: BLE001
            if net.fail():
                return "lost the agent"
            continue
        if h.get("uptime", 0) < uptime:  # container restarted: /r is gone
            return "pod container restarted"
        uptime = h.get("uptime", 0)
        procs = h.get("procs", {})
        if any(f"seed{s}" not in procs for s in SEEDS):
            return "trainer process missing"
        rcs = {s: procs[f"seed{s}"].get("returncode") for s in SEEDS}
        rows = latest_rates(agent)
        for s, r in rows.items():
            if r["transitions"] > best.get(s, -1):
                best[s] = r["transitions"]
                last_progress = now()
        note(
            stage="running",
            returncodes=rcs,
            rates={s: r["rate"] for s, r in rows.items()},
            transitions=best,
            balance=state.get("balance_last"),
        )
        if all(rc is not None for rc in rcs.values()):
            dl = download_all(agent, run_dir, state)
            note(stage="finished", returncodes=rcs, download=dl)
            if not all(rc == 0 for rc in rcs.values()):
                return "finished with failures"
            return (
                "complete" if not (dl.get("failed") or dl.get("error")) else "download incomplete"
            )
        if now() - last_progress > STALL_S and any(rc is None for rc in rcs.values()):
            note(stage="stall", download=download(agent, run_dir))
            return "stalled"
        if g2 is None and now() - started >= G2_SECONDS:
            dead = [s for s, rc in rcs.items() if rc is not None]
            if (
                not dead
                and len(rows) < len(SEEDS)
                and now() - started < G2_SECONDS + G2_ROWS_GRACE_S
            ):
                continue  # a transient GET miss: decide on the next poll
            g2 = g2_decision(
                {s: r["rate"] for s, r in rows.items()},
                {s: r["transitions"] for s, r in rows.items()},
                state["until"] - now(),  # §9: projected finish + 45 min <= lifetime left
                dead,
            )
            state["g2"] = g2
            save_state(run_dir, state)
            note(stage="G2", **{k: v for k, v in g2.items() if k != "per_seed"})
            if not g2["pass"]:
                note(stage="g2_fail", download=download(agent, run_dir))
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


def plan(args) -> int:
    from research.runpod_fanout.ledger import SharedLedger
    from research.runpod_fanout.rp_client import RpClient

    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    work = Path(str(args.run_dir) + "-plan")
    inputs = build_inputs(work / "inputs")
    until = now() + args.lifetime_hours * 3600
    body = pod_body(POD_NAME + "plan", "0" * 64, until - 900, until)
    (work / "pod_body.json").write_text(json.dumps(body))
    rp = RpClient()
    dry = rp.create_pod(work / "pod_body.json", MAX_HOURLY, confirm=False)
    totals = SharedLedger(policy).peek()
    out = {
        "commit": inputs["commit"],
        "upload_files": len(inputs["files"]),
        "upload_bytes": inputs["bytes"],
        "demo_sha256": inputs["demo_sha256"],
        "balance": rp.balance(),
        "ledger": totals,
        "project_cap_usd": policy["project_cap_usd"],
        "worst_case_usd": args.rate * args.lifetime_hours,
        "spend_cap_usd": args.spend_cap,
        "dry_run": dry,
        "body_without_cmd": {k: v for k, v in body.items() if k != "dockerStartCmd"},
    }
    print(json.dumps(out, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("plan", "launch", "cleanup"))
    ap.add_argument("--run-dir", default=str(ARTIFACTS / "redesign-m3b-gpu-20261008"))
    ap.add_argument("--lifetime-hours", type=float, default=7.0)
    ap.add_argument("--spend-cap", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=0.74)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if args.cmd == "plan":
        return plan(args)
    if args.cmd == "cleanup":
        return cleanup(args)
    if not args.confirm:
        print(
            "launch spends money: re-run with --confirm after the plan is approved", file=sys.stderr
        )
        return 2
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
