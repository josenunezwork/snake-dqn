"""Pod-side agent for the M3-B GPU run (stdlib only; shipped in the pod's start command).

Reached only through RunPod's HTTPS proxy (``https://<pod>-8000.proxy.runpod.net``). Every
request needs header ``X-M3-Token`` whose sha256 equals env ``M3_TOKEN_SHA256`` (the pod env
holds only the hash). Modelled on ``research/runpod_fanout/pod_agent.py``.

GET  /health                 uptime, deadline, disk, GPU, processes, inputs received
PUT  /in/<relpath>           a file; header X-Sha256 must equal the body's sha256
POST /exec                   {"name", "argv", "cwd", "env"}: start one named process
GET  /log/<name>             tail of that process's stdout+stderr
GET  /out/<relpath>          a file under /r/out (results)
GET  /ls/out                 [{"path", "bytes"}] under /r/out

Money safety: at ``M3_DEADLINE_EPOCH`` every process is killed and /exec is refused; at
``M3_SELF_DELETE_EPOCH`` the pod deletes itself with the pod-scoped RunPod credentials
(RUNPOD_POD_ID / RUNPOD_API_KEY) if the local runner and its watchdog have not.
"""

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("M3_ROOT", "/r"))
IN, OUT, LOGS = ROOT / "in", ROOT / "out", ROOT / "logs"
TOKEN_SHA = os.environ.get("M3_TOKEN_SHA256", "")
DEADLINE = float(os.environ.get("M3_DEADLINE_EPOCH", "0"))
SELF_DELETE = float(os.environ.get("M3_SELF_DELETE_EPOCH", "0"))
DEADMAN = float(os.environ.get("M3_DEADMAN_SECONDS", "1200"))
BOOT = time.time()
LAST_AUTH = [time.time()]
LOCK = threading.Lock()
PROCS = {}
for d in (IN, OUT, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    with (LOGS / "agent.log").open("a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def safe(base, rel):
    path = (base / rel).resolve()
    if base.resolve() not in path.parents and path != base.resolve():
        raise ValueError("path escapes its root")
    return path


def gpu():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({type(exc).__name__})"


def cgroup_memory_gb():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                return int(raw) / 1e9
        except (OSError, ValueError):
            continue
    return None


def cgroup_cpus():
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def health():
    with LOCK:
        procs = {n: {"pid": p.pid, "returncode": p.poll()} for n, p in PROCS.items()}
    du = shutil.disk_usage(str(ROOT))
    mem = ""
    try:
        mem = Path("/proc/meminfo").read_text().splitlines()[:3]
    except OSError:
        pass
    return {
        "agent": "m3-gpu-agent/v1",
        "now": time.time(),
        "uptime": time.time() - BOOT,
        "deadline_epoch": DEADLINE,
        "self_delete_epoch": SELF_DELETE,
        "disk_free_gb": du.free / 1e9,
        "meminfo": mem,
        "cpus": os.cpu_count(),
        "cpus_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_memory_gb": cgroup_memory_gb(),
        "cgroup_cpus": cgroup_cpus(),
        "self_delete_armed": bool(
            os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_API_KEY")
        ),
        "last_auth_age": time.time() - LAST_AUTH[0],
        "gpu": gpu(),
        "procs": procs,
        "inputs": sorted(str(p.relative_to(IN)) for p in IN.rglob("*") if p.is_file())[:200],
    }


def self_delete():
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        log("self-delete: no pod-scoped credentials")
        return
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", "m3-gpu-agent/1.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            log(f"self-delete requested: HTTP {resp.status}")
    except Exception as exc:  # noqa: BLE001
        log(f"self-delete failed: {type(exc).__name__} {getattr(exc, 'code', '')}")


def kill_all():
    with LOCK:
        for p in PROCS.values():
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass


def reaper():
    tried = 0.0
    while True:
        time.sleep(5)
        now = time.time()
        # Dead-man switch: the local runner polls every minute; if it is gone, stop paying.
        if DEADMAN and now - LAST_AUTH[0] > DEADMAN:
            kill_all()
            if now - tried > 600:
                tried = now
                log("dead-man: no runner contact; self-deleting")
                self_delete()
            continue
        if DEADLINE and now >= DEADLINE:
            with LOCK:
                for p in PROCS.values():
                    if p.poll() is None:
                        try:
                            p.kill()
                        except OSError:
                            pass
        if SELF_DELETE and now >= SELF_DELETE and now - tried > 600:
            tried = now
            self_delete()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _auth(self):
        token = self.headers.get("X-M3-Token", "")
        ok = TOKEN_SHA and hmac.compare_digest(
            hashlib.sha256(token.encode()).hexdigest(), TOKEN_SHA
        )
        if not ok:
            self._send(403, {"error": "forbidden"})
        else:
            LAST_AUTH[0] = time.time()
        return ok

    def _send(self, code, payload, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream" if raw else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if not self._auth():
            return
        try:
            if self.path == "/health":
                return self._send(200, health())
            if self.path == "/ls/out":
                rows = []
                for p in OUT.rglob("*"):
                    if p.is_file():
                        rows.append(
                            {
                                "path": str(p.relative_to(OUT)),
                                "bytes": p.stat().st_size,
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            }
                        )
                return self._send(200, rows)
            if self.path.startswith("/log/"):
                path = safe(LOGS, self.path[5:] + ".log")
                data = path.read_bytes()[-20000:] if path.exists() else b""
                return self._send(200, None, raw=data or b" ")
            if self.path.startswith("/out/"):
                return self._send(200, None, raw=safe(OUT, self.path[5:]).read_bytes())
        except (OSError, ValueError) as exc:
            return self._send(404, {"error": str(exc)})
        self._send(404, {"error": "unknown"})

    def do_PUT(self):  # noqa: N802
        if not self._auth():
            return
        if not self.path.startswith("/in/"):
            return self._send(404, {"error": "unknown"})
        try:
            path = safe(IN, self.path[4:])
            n = int(self.headers.get("Content-Length", "0"))
            want = self.headers.get("X-Sha256", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            h = hashlib.sha256()
            with tmp.open("wb") as fh:
                left = n
                while left:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    h.update(chunk)
                    fh.write(chunk)
                    left -= len(chunk)
            if left or h.hexdigest() != want:
                tmp.unlink(missing_ok=True)
                return self._send(400, {"error": "sha256 mismatch or short body"})
            tmp.replace(path)
            return self._send(200, {"sha256": h.hexdigest(), "bytes": n})
        except (OSError, ValueError) as exc:
            return self._send(400, {"error": str(exc)})

    def do_POST(self):  # noqa: N802
        if not self._auth():
            return
        if self.path != "/exec":
            return self._send(404, {"error": "unknown"})
        if DEADLINE and time.time() >= DEADLINE:
            return self._send(409, {"error": "past deadline"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            name = str(req["name"])
            with LOCK:
                if name in PROCS and PROCS[name].poll() is None:
                    return self._send(409, {"error": f"{name} is running"})
                out = safe(LOGS, name + ".log").open("ab")
                env = dict(os.environ)
                env.pop("RUNPOD_API_KEY", None)  # never hand the pod key to the jobs
                env.update({str(k): str(v) for k, v in req.get("env", {}).items()})
                PROCS[name] = subprocess.Popen(
                    [str(a) for a in req["argv"]],
                    cwd=str(req.get("cwd", ROOT)),
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            log(f"exec {name}")
            return self._send(200, {"name": name, "pid": PROCS[name].pid})
        except (OSError, ValueError, KeyError) as exc:
            return self._send(400, {"error": str(exc)})


def main():
    log("agent up")
    threading.Thread(target=reaper, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
