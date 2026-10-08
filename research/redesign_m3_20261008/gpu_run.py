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
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

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
]
SEEDS = (0, 1, 2, 3, 4)
TRANSITIONS = 20_000_000
MAC_RATE = 1388.7248756266617  # M3-A measured end-to-end (results/m3a/summary.json)
G2_RATE = 5600.0  # >= 4 x MAC_RATE, rounded up (approved wording: ">= 5.6k combined")
G2_SECONDS = 1800
CHUNK = 48 << 20
POD_NAME = "m3b-gpu-"
MAX_HOURLY = 1.0
PORT = 8000


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
        "minRAMPerGPU": 64,
        "minVCPUPerGPU": 8,
        "ports": [f"{PORT}/http"],
        "env": {
            "M3_TOKEN_SHA256": token_sha,
            "M3_DEADLINE_EPOCH": str(int(deadline)),
            "M3_SELF_DELETE_EPOCH": str(int(until)),
        },
        "dockerStartCmd": ["bash", "-c", start],
    }


def g2_decision(
    rates: Dict[int, float], elapsed: float, remaining_lifetime: float
) -> Dict[str, Any]:
    """G2 at ~30 min: combined rate >= 5.6k/s AND the projected finish fits the lifetime."""
    combined = float(sum(rates.values()))
    slowest = min(rates.values()) if rates else 0.0
    need = max(0.0, TRANSITIONS / slowest - elapsed) if slowest > 0 else float("inf")
    ok_rate = len(rates) == len(SEEDS) and combined >= G2_RATE
    ok_time = need + 45 * 60 <= remaining_lifetime
    return {
        "combined_rate": combined,
        "per_seed": rates,
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
            "python -c 'import torch, numba, numpy; print(torch.__version__, "
            "torch.cuda.is_available(), numba.__version__, numpy.__version__)'",
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
        self._req("POST", "/exec", body, {"Content-Type": "application/json"})

    def get(self, path: str) -> bytes:
        return self._req("GET", path, timeout=300)


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


def delete_and_verify(rp, run_dir: Path, state: Dict[str, Any], reason: str) -> None:
    pod_id = state.get("pod_id")
    if pod_id:
        for _ in range(5):
            try:
                rp.delete_pod(pod_id, confirm=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"delete failed ({exc}); retrying", flush=True)
                time.sleep(10)
    left = our_pods(rp)
    state.update(deleted=now(), delete_reason=reason, pods_left=[p.get("id") for p in left])
    save_state(run_dir, state)
    print(json.dumps({"deleted": pod_id, "reason": reason, "our_pods_left": state["pods_left"]}))


def spawn_watchdog(run_dir: Path, pod_id: str, until: float) -> int:
    code = (
        "import sys,time,json;sys.path.insert(0,%r);"
        "from research.runpod_fanout.rp_client import RpClient;"
        "time.sleep(max(0,%f-time.time()));rp=RpClient();"
        "ids=[p.get('id') for p in rp.list_pods()];"
        "rp.delete_pod(%r,confirm=True) if %r in ids else None;"
        "open(%r,'a').write(json.dumps({'watchdog':'fired','t':time.time()})+'\\n')"
    ) % (str(REPO), until + 300, pod_id, pod_id, str(run_dir / "watchdog.log"))
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid


def latest_rates(agent: Agent) -> Dict[int, float]:
    rates = {}
    for s in SEEDS:
        try:
            lines = agent.get(f"/out/m3b_seed{s}/log.jsonl").decode().strip().splitlines()
            row = json.loads(lines[-1])
            rates[s] = float(row["transitions"]) / float(row["wall"])
        except Exception:  # noqa: BLE001
            continue
    return rates


def launch(args) -> int:
    from research.runpod_fanout.ledger import SharedLedger
    from research.runpod_fanout.rp_client import RpClient

    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    rp = RpClient()
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        print(f"refusing: {run_dir} exists", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    inputs = build_inputs(run_dir / "inputs")
    if our_pods(rp):
        raise SystemExit("refusing: a pod of ours already exists")
    balance0 = rp.balance()
    token = secrets.token_urlsafe(32)
    created = now()
    until = created + args.lifetime_hours * 3600
    deadline = until - 15 * 60
    name = POD_NAME + time.strftime("%m%d%H%M")
    body = pod_body(name, hashlib.sha256(token.encode()).hexdigest(), deadline, until)
    body_path = run_dir / "pod_body.json"
    body_path.write_text(json.dumps(body))
    ledger = SharedLedger(policy)
    run_id = f"m3b-gpu-{int(created)}"
    ledger.register_run(run_id, "redesign-m3b-gpu", str(run_dir), args.spend_cap, os.getpid())
    refusal = ledger.reserve_pod(run_id, "gpu0", args.rate, until, balance0)
    if refusal:
        ledger.finish_run(run_id, False, False)
        raise SystemExit(f"ledger refused: {refusal}")
    state: Dict[str, Any] = {
        "run_id": run_id,
        "balance0": balance0,
        "created": created,
        "until": until,
        "deadline": deadline,
        "commit": inputs["commit"],
        "spend_cap": args.spend_cap,
        "token_sha256": body["env"]["M3_TOKEN_SHA256"],
    }
    save_state(run_dir, state)
    out = rp.create_pod(body_path, MAX_HOURLY, confirm=True)
    pod_id = out.get("id") if isinstance(out, dict) else None
    if not pod_id:
        ledger.finish_run(run_id, False, False)
        raise SystemExit(f"pod create failed: {out}")
    rate = float(out.get("costPerHr") or args.rate)
    ledger.bind_pod(run_id, "gpu0", pod_id, rate)
    state.update(pod_id=pod_id, rate=rate, machine=out.get("machine"))
    state["watchdog_pid"] = spawn_watchdog(run_dir, pod_id, until)
    save_state(run_dir, state)
    print(json.dumps({"pod_id": pod_id, "rate": rate, "until": until}), flush=True)
    agent = Agent(pod_id, token)
    outcome = "unknown"
    try:
        outcome = drive(agent, rp, run_dir, state, inputs, balance0, args)
    except BaseException as exc:  # noqa: BLE001  (incl. KeyboardInterrupt: always clean up)
        outcome = f"error: {type(exc).__name__}: {exc}"
        raise
    finally:
        state["outcome"] = outcome
        save_state(run_dir, state)
        delete_and_verify(rp, run_dir, state, outcome)
        cost = rate * (now() - created) / 3600.0
        ledger.release_pod(run_id, "gpu0", cost)
        ledger.finish_run(run_id, outcome == "complete", outcome == "complete")
        state.update(cost_estimate=cost, balance_end=rp.balance())
        save_state(run_dir, state)
    return 0 if outcome == "complete" else 1


def guard(rp, state, args) -> Optional[str]:
    if now() >= state["until"]:
        return "lifetime"
    try:
        bal = rp.balance()
    except Exception:  # noqa: BLE001
        return None
    state["balance_last"] = bal
    if spend_exceeded(state["balance0"], bal, args.spend_cap):
        return "spend cap"
    return None


def drive(agent, rp, run_dir, state, inputs, balance0, args) -> str:
    log = run_dir / "progress.jsonl"

    def note(**row):
        row["t"] = now()
        with log.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    # 1. agent up
    t0 = now()
    while True:
        stop = guard(rp, state, args)
        if stop:
            return stop
        try:
            h = agent.health()
            note(stage="agent_up", gpu=h.get("gpu"), cpus=h.get("cpus"), mem=h.get("meminfo"))
            break
        except Exception:  # noqa: BLE001
            if now() - t0 > 1200:
                return "agent never came up"
            time.sleep(15)
    # 2. uploads (sha-verified by the agent)
    for f in inputs["files"]:
        for attempt in range(3):
            try:
                agent.put(f["dst"], f["src"], f["sha256"])
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    return f"upload failed: {f['dst']}: {exc}"
                time.sleep(5)
    note(stage="uploaded", files=len(inputs["files"]), bytes=inputs["bytes"])
    # 3. setup
    agent.exec("setup", setup_argv(inputs["demo_sha256"]), "/r", {})
    while True:
        stop = guard(rp, state, args)
        if stop:
            return stop
        procs = agent.health()["procs"]
        rc = procs.get("setup", {}).get("returncode")
        if rc is not None:
            tail = agent.get("/log/setup").decode(errors="replace")[-800:]
            note(stage="setup_done", returncode=rc, tail=tail)
            if rc != 0 or "SETUP_OK" not in tail:
                return "setup failed"
            break
        time.sleep(20)
    # 4. trainers
    env = {"SNAKE_CKPT_DIR": "/r/in/ckpt", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for s in SEEDS:
        agent.exec(f"seed{s}", train_argv(s), "/r/repo", env)
    started = now()
    note(stage="trainers_started")
    g2 = None
    while True:
        stop = guard(rp, state, args)
        if stop:
            return stop
        time.sleep(60)
        h = agent.health()
        rcs = {s: h["procs"].get(f"seed{s}", {}).get("returncode") for s in SEEDS}
        rates = latest_rates(agent)
        note(stage="running", returncodes=rcs, rates=rates, balance=state.get("balance_last"))
        if any(rc not in (None, 0) for rc in rcs.values()):
            for s, rc in rcs.items():
                if rc not in (None, 0):
                    tail = agent.get(f"/log/seed{s}").decode(errors="replace")[-1500:]
                    note(stage="trainer_failed", seed=s, returncode=rc, tail=tail)
            return "trainer failed"
        if g2 is None and now() - started >= G2_SECONDS:
            g2 = g2_decision(rates, now() - started, state["until"] - now())
            state["g2"] = g2
            save_state(run_dir, state)
            note(stage="G2", **{k: v for k, v in g2.items() if k != "per_seed"})
            if not g2["pass"]:
                download(agent, run_dir)
                return f"G2 fail ({g2['reason']})"
        if all(rc == 0 for rc in rcs.values()):
            download(agent, run_dir)
            return "complete"


def download(agent: Agent, run_dir: Path) -> None:
    dest = run_dir / "out"
    for row in json.loads(agent.get("/ls/out")):
        path = dest / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(agent.get("/out/" + row["path"]))


def cleanup(args) -> int:
    from research.runpod_fanout.rp_client import RpClient

    run_dir = Path(args.run_dir)
    state = json.loads(state_path(run_dir).read_text())
    rp = RpClient()
    if not args.confirm:
        print(json.dumps({"would_delete": state.get("pod_id"), "our_pods": our_pods(rp)}))
        return 0
    delete_and_verify(rp, run_dir, state, "manual cleanup")
    return 0


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
