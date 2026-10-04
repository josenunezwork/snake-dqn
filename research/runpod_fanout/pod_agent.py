"""Pod-side agent for the RunPod fan-out runner (stdlib only; shipped in the start command).

Reached only through RunPod's HTTPS proxy (``https://<pod>-8000.proxy.runpod.net``).
Every request needs header ``X-Fanout-Token`` whose sha256 equals env
``FANOUT_TOKEN_SHA256`` (the pod env holds only the hash, never the token).

GET  /health            status: pip/setup flags, slots, queued/running/done/failed, boot id
PUT  /in/repo.tar.gz    git archive of the job commit (sha256 echoed)
PUT  /in/ckpt/<sha>.pth checkpoint; refused unless its bytes hash to <sha>
POST /setup             {"repo_sha256", "commit", "job_id"}: verify + extract the archive
POST /assign            {"episodes": [{"key", "spec"}], "timeout_seconds"}: queue work
POST /drain             accept no more work
GET  /records           [{"key", "sha256", "bytes"}] of finished records
GET  /record/<key>      one record's bytes
GET  /log               tail of the agent log

Hard stop: after ``FANOUT_DEADLINE_EPOCH`` running episodes are killed and nothing new
starts (the runner and its watchdog delete the pod; the pod cannot delete itself).
"""

import hashlib
import hmac
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("FANOUT_ROOT", "/r"))
IN, OUT, CKPT, REPO = ROOT / "in", ROOT / "out" / "records", ROOT / "ckpt", ROOT / "repo"
SPECS = ROOT / "specs"
TOKEN_SHA = os.environ.get("FANOUT_TOKEN_SHA256", "")
DEADLINE = float(os.environ.get("FANOUT_DEADLINE_EPOCH", "0"))
PROVIDER = os.environ.get("FANOUT_PROVIDER", "runpod")
PIP = json.loads(os.environ.get("FANOUT_PIP_JSON", "[]"))
EXEC = json.loads(os.environ.get("FANOUT_EXEC_JSON", "null"))  # tests: replace the episode cmd
THREADS = os.environ.get("FANOUT_THREADS", "2")
ISA = os.environ.get("FANOUT_ISA_CAP", "")
SELF_DELETE = float(os.environ.get("FANOUT_SELF_DELETE_EPOCH", "0"))
NUMERICS = json.loads(os.environ.get("FANOUT_NUMERICS_JSON", "{}"))  # MKL_CBWR, threads
ISA_FLAGS = (
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "avx512_fp16",
    "amx_tile",
    "amx_bf16",
    "amx_int8",
)  # same list as platform_rule.ISA_FLAGS
SETUP_LOCK = threading.Lock()
BOOT = uuid.uuid4().hex[:12]
LOCK = threading.Lock()
STATE = {
    "pip": "pending",
    "setup": None,
    "draining": False,
    "queued": [],
    "running": {},
    "done": {},
    "failed": {},
    "procs": {},
}
WORK: "queue.Queue" = queue.Queue()
LOG = ROOT / "agent.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n"
    with LOCK:
        with LOG.open("a") as fh:
            fh.write(line)


def slots():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 2
    n = int(os.environ.get("FANOUT_WORKERS", "0")) or max(1, cores // int(THREADS))
    return n, cores


def pip_install():
    if not PIP:
        STATE["pip"] = "ok"
        return
    for cmd in PIP:
        r = subprocess.run(cmd, capture_output=True, text=True)
        log(f"pip rc={r.returncode} {' '.join(cmd)[:120]} {r.stderr[-400:]}")
        if r.returncode != 0:
            STATE["pip"] = "failed"
            return
    STATE["pip"] = "ok"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def records():
    out = []
    if OUT.is_dir():
        for p in sorted(OUT.rglob("*.json")):
            if p.name.startswith("."):
                continue
            data = p.read_bytes()
            out.append(
                {
                    "key": str(p.relative_to(OUT)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    return out


def run_one(key, spec, timeout):
    setup = STATE["setup"]
    SPECS.mkdir(parents=True, exist_ok=True)
    spec_path = SPECS / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    spec_path.write_text(json.dumps(spec))
    cmd = EXEC or [sys.executable, "-m", "research.runpod_fanout.episode"]
    cmd = list(cmd) + [
        "--spec",
        str(spec_path),
        "--out-root",
        str(OUT),
        "--ckpt-dir",
        str(CKPT),
        "--provider",
        PROVIDER,
        "--job-id",
        setup["job_id"],
        "--commit",
        setup["commit"],
    ]
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("FANOUT_TOKEN"):
            env.pop(k)
    env.update(
        {
            "OMP_NUM_THREADS": THREADS,
            "MKL_NUM_THREADS": THREADS,
            "SNAKE_DQN_DEVICE": "cpu",
            "PYTHONHASHSEED": "0",
        }
    )
    env.update({str(k): str(v) for k, v in NUMERICS.items()})
    if ISA:  # optional ISA cap (off by default: the validated x86 check ran without it)
        env.update({"ONEDNN_MAX_CPU_ISA": ISA, "MKL_ENABLE_INSTRUCTIONS": ISA, "MKL_CBWR": ISA})
    for k in ("RUNPOD_API_KEY",):
        env.pop(k, None)
    left = DEADLINE - time.time() if DEADLINE else timeout
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    with LOCK:
        STATE["procs"][key] = proc
    try:
        out, err = proc.communicate(timeout=max(1.0, min(timeout, left)))
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        rc = "timeout"
    seconds = round(time.time() - started, 3)
    status = None
    for line in (out or "").splitlines()[::-1]:
        try:
            status = json.loads(line).get("status")
            break
        except (ValueError, AttributeError):
            continue
    with LOCK:
        STATE["procs"].pop(key, None)
        STATE["running"].pop(key, None)
        if rc == 0 and status in ("written", "duplicate-verified"):
            STATE["done"][key] = {"status": status, "seconds": seconds}
        else:
            STATE["failed"][key] = {
                "rc": rc,
                "status": status,
                "seconds": seconds,
                "stderr_tail": (err or "")[-1500:],
            }
    log(f"episode {key} rc={rc} status={status} {seconds}s")


def worker():
    while True:
        key, spec, timeout = WORK.get()
        with LOCK:
            if key in STATE["queued"]:
                STATE["queued"].remove(key)
            late = DEADLINE and time.time() >= DEADLINE
            if late:
                STATE["failed"][key] = {"rc": "deadline", "status": None, "seconds": 0}
            else:
                STATE["running"][key] = time.time()
        if not late:
            try:
                run_one(key, spec, timeout)
            except Exception as exc:  # noqa: BLE001
                with LOCK:
                    STATE["running"].pop(key, None)
                    STATE["failed"][key] = {"rc": "agent-error", "stderr_tail": repr(exc)}


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def isa_flags():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                have = set(line.split(":", 1)[1].split())
                return [f for f in ISA_FLAGS if f in have]
    except OSError:
        pass
    return []


def self_delete():
    """Last resort if the runner and its watchdog are gone: the pod removes itself with the
    pod-scoped credentials RunPod injects (RUNPOD_POD_ID / RUNPOD_API_KEY), if any."""
    import urllib.request

    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        log("self-delete: no pod-scoped credentials; waiting for the runner/watchdog")
        return
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", "rpf-agent/1.0")
    try:
        urllib.request.urlopen(req, timeout=30).read()
        log("self-delete requested")
    except Exception as exc:  # noqa: BLE001
        log(f"self-delete failed: {type(exc).__name__} {getattr(exc, 'code', '')}")


def reaper():
    tried = 0.0
    while True:
        time.sleep(5)
        if SELF_DELETE and time.time() >= SELF_DELETE and time.time() - tried > 600:
            tried = time.time()
            self_delete()
        if DEADLINE and time.time() >= DEADLINE:
            with LOCK:
                STATE["draining"] = True
                procs = list(STATE["procs"].values())
            for proc in procs:
                try:
                    proc.kill()
                except OSError:
                    pass


def health():
    n, cores = slots()
    with LOCK:
        return {
            "agent": "runpod-fanout-agent/v1",
            "boot_id": BOOT,
            "now": time.time(),
            "deadline_epoch": DEADLINE,
            "pip": STATE["pip"],
            "setup": STATE["setup"],
            "draining": STATE["draining"],
            "slots": n,
            "cores": cores,
            "cpu_model": cpu_model(),
            "isa_flags": isa_flags(),
            "numerics": NUMERICS,
            "self_delete_capable": bool(
                os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_API_KEY")
            ),
            "queued": list(STATE["queued"]),
            "running": sorted(STATE["running"]),
            "done": dict(STATE["done"]),
            "failed": dict(STATE["failed"]),
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        token = self.headers.get("X-Fanout-Token", "")
        digest = hashlib.sha256(token.encode()).hexdigest()
        return bool(TOKEN_SHA) and hmac.compare_digest(digest, TOKEN_SHA)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        if self.path == "/health":
            return self._send(200, health())
        if self.path == "/records":
            return self._send(200, records())
        if self.path == "/log":
            data = LOG.read_bytes()[-20000:] if LOG.exists() else b""
            return self._send(200, data, "text/plain")
        if self.path.startswith("/record/"):
            target = (OUT / self.path[len("/record/") :]).resolve()
            if str(target).startswith(str(OUT.resolve()) + "/") and target.is_file():
                return self._send(200, target.read_bytes(), "application/octet-stream")
        return self._send(404, {"error": "not found"})

    def do_PUT(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        name = self.path[len("/in/") :] if self.path.startswith("/in/") else ""
        if name == "repo.tar.gz":
            dest = IN / name
            want = None
        elif name.startswith("ckpt/") and name.endswith(".pth") and len(name) == 5 + 64 + 4:
            want = name[5:-4]
            if not all(c in "0123456789abcdef" for c in want):
                return self._send(400, {"error": "bad name"})
            dest = CKPT / f"{want}.pth"
        else:
            return self._send(403, {"error": "name not allowed"})
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.headers.get("Content-Length", "0"))
        h = hashlib.sha256()
        tmp = dest.with_name(dest.name + ".part")
        left = n
        with tmp.open("wb") as fh:
            while left:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                fh.write(chunk)
                h.update(chunk)
                left -= len(chunk)
        digest = h.hexdigest()
        if left or (want and digest != want):
            tmp.unlink(missing_ok=True)
            return self._send(400, {"error": "incomplete or sha256 mismatch", "sha256": digest})
        os.replace(tmp, dest)
        return self._send(200, {"name": name, "bytes": n, "sha256": digest})

    def do_POST(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        try:
            body = self._body()
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if self.path == "/setup":
            return self._setup(body)
        if self.path == "/drain":
            with LOCK:
                STATE["draining"] = True
            return self._send(200, {"draining": True})
        if self.path == "/assign":
            return self._assign(body)
        return self._send(404, {"error": "not found"})

    def _setup(self, body):
        repo = IN / "repo.tar.gz"
        if not repo.is_file() or sha_file(repo) != body.get("repo_sha256"):
            return self._send(400, {"error": "repo archive missing or sha256 differs"})
        if not SETUP_LOCK.acquire(blocking=False):
            return self._send(409, {"error": "setup in progress"})
        try:
            self._do_setup(body)
        finally:
            SETUP_LOCK.release()
        return self._send(200, {"setup": STATE["setup"]})

    def _do_setup(self, body):
        repo = IN / "repo.tar.gz"
        if STATE["setup"] is None:
            if REPO.exists():
                shutil.rmtree(REPO)
            REPO.mkdir(parents=True)
            with tarfile.open(repo) as tar:
                tar.extractall(REPO, filter="data")
            STATE["setup"] = {
                "repo_sha256": body["repo_sha256"],
                "commit": str(body.get("commit")),
                "job_id": str(body.get("job_id")),
                "ckpts": sorted(p.name[:-4] for p in CKPT.glob("*.pth")),
            }
            try:
                freeze = subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
                ).stdout
            except OSError:
                freeze = ""
            STATE["setup"]["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
            (ROOT / "pip_freeze.txt").write_text(freeze)
            log(f"setup {STATE['setup']}")

    def _assign(self, body):
        code, out = self._enqueue(body)
        return self._send(code, out)

    def _enqueue(self, body):
        with LOCK:
            if STATE["setup"] is None or STATE["pip"] != "ok":
                return 409, {"error": "not ready"}
            if STATE["draining"] or (DEADLINE and time.time() >= DEADLINE):
                return 409, {"error": "draining"}
            accepted = []
            busy = set(STATE["queued"]) | set(STATE["running"]) | set(STATE["done"])
            for item in body.get("episodes", []):
                key = item["key"]
                if key in busy:
                    continue
                STATE["failed"].pop(key, None)
                STATE["queued"].append(key)
                WORK.put((key, item["spec"], float(body.get("timeout_seconds", 3600))))
                accepted.append(key)
                busy.add(key)
        return 200, {"accepted": accepted}

    def log_message(self, *args):
        pass


def main():
    for d in (IN, OUT, CKPT):
        d.mkdir(parents=True, exist_ok=True)
    log(f"agent boot {BOOT} deadline={DEADLINE}")
    threading.Thread(target=pip_install, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()
    for _ in range(slots()[0]):
        threading.Thread(target=worker, daemon=True).start()
    port = int(os.environ.get("FANOUT_PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
