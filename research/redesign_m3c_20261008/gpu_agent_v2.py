"""Pod-side agent v2 for M3-C GPU runs (stdlib only; shipped in the pod's start command).

v2 = the M3-B agent (``research/redesign_m3_20261008/gpu_agent.py``, every money-safety path
unchanged) plus: ``GET /telemetry`` (GPU util / VRAM via nvidia-smi, host CPU % from
/proc/stat deltas, load, RAM, and per-job CPU % / RSS summed over each job's process group),
CPU model in ``/health``, and ``/ls/out`` hashes cached by (size, mtime) so a listing does
not re-read every checkpoint each minute.

Reached only through RunPod's HTTPS proxy (``https://<pod>-8000.proxy.runpod.net``). Every
request needs header ``X-M3-Token`` whose sha256 equals env ``M3_TOKEN_SHA256`` (the pod env
holds only the hash). Modelled on ``research/runpod_fanout/pod_agent.py``.

GET  /health                 uptime, deadline, disk, GPU, processes, inputs received
PUT  /in/<relpath>           a file; header X-Sha256 must equal the body's sha256
POST /exec                   {"name", "argv", "cwd", "env"}: start one named process
POST /kill                   {"name"}: SIGKILL that job's process group (a stalled run)
POST /self-delete-key        {"key"}: install an owner key in memory and probe it
GET  /log/<name>             tail of that process's stdout+stderr
GET  /out/<relpath>          a file under /r/out (results)
GET  /ls/out                 [{"path", "bytes", "sha256"}] under /r/out (hashes cached)
GET  /telemetry              GPU / CPU / RAM / per-job usage snapshot

Money safety: at ``M3_DEADLINE_EPOCH`` every process is killed and /exec is refused; at
``M3_SELF_DELETE_EPOCH`` the pod deletes itself using the runtime owner key, or the injected
``RUNPOD_API_KEY`` when no owner key was supplied. Both paths target ``RUNPOD_POD_ID`` if
the local runner and its watchdog have not already deleted the pod.
"""

import hashlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
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
PROBE = [None]  # HTTP status of an authenticated GET of this pod with the active key
# Never persist or log this key. The Mac runner delivers it through the authenticated proxy
# after boot, so the pod-start environment and every child process remain key-free.
SELF_DELETE_KEY = [None]
LOCK = threading.Lock()
CREDENTIAL_LOCK = threading.RLock()
PROCS = {}
HASHES = {}  # path -> (size, mtime_ns, sha256)
CPU_LAST = [None]  # (total, idle) jiffies of the previous /telemetry call
JOB_LAST = {}  # job -> (t, cpu seconds)
CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
for d in (IN, OUT, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    with (LOGS / "agent.log").open("a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {_redact(msg)}\n")


def _redact(value):
    """Replace any in-memory or legacy RunPod key before it reaches an agent log/JSON error."""
    text = str(value)
    for key in (SELF_DELETE_KEY[0], os.environ.get("RUNPOD_API_KEY")):
        if key:
            text = text.replace(key, "[redacted]")
    return text


def _redact_payload(value):
    if isinstance(value, dict):
        return {_redact(k): _redact_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_payload(v) for v in value]
    if isinstance(value, str):
        return _redact(value)
    return value


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


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def gpu_stats():
    """[{name, util_pct, mem_used_mib, mem_total_mib, power_w}] from nvidia-smi."""
    fields = "name,utilization.gpu,memory.used,memory.total,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return [{"error": type(exc).__name__}]
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 5:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None

        rows.append(
            {
                "name": parts[0],
                "util_pct": num(parts[1]),
                "mem_used_mib": num(parts[2]),
                "mem_total_mib": num(parts[3]),
                "power_w": num(parts[4]),
            }
        )
    return rows


def host_cpu_pct():
    """Busy % of all host CPUs since the previous call (None on the first call)."""
    try:
        vals = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    except (OSError, ValueError, IndexError):
        return None
    total, idle = sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)
    prev, CPU_LAST[0] = CPU_LAST[0], (total, idle)
    if prev is None or total <= prev[0]:
        return None
    return 100.0 * (1.0 - (idle - prev[1]) / (total - prev[0]))


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError):
        pass
    return None


def group_usage(pgid):
    """(cpu seconds, rss GB, processes) summed over every process in process group pgid."""
    cpu = rss = 0.0
    n = 0
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:  # no procfs (not Linux)
        return None, None, 0
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            fields = stat[stat.rindex(")") + 2 :].split()
            if int(fields[2]) != pgid:  # field 5 (pgrp) after pid and comm
                continue
            cpu += (int(fields[11]) + int(fields[12])) / CLK  # utime + stime
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024 / 1e9
            n += 1
        except (OSError, ValueError, IndexError):
            continue
    return cpu, rss, n


def telemetry():
    now = time.time()
    with LOCK:
        procs = {name: p for name, p in PROCS.items()}
    jobs = {}
    for name, p in procs.items():
        if p.poll() is not None:
            jobs[name] = {"returncode": p.returncode}
            continue
        cpu, rss, n = group_usage(p.pid)  # each job is its own session: pgid == pid
        prev = JOB_LAST.get(name)
        pct = None
        if cpu is None:
            jobs[name] = {"cpu_pct": None, "rss_gb": None, "processes": 0}
            continue
        JOB_LAST[name] = (now, cpu)
        if prev and now > prev[0]:
            pct = 100.0 * (cpu - prev[1]) / (now - prev[0])
        jobs[name] = {"cpu_pct": pct, "rss_gb": rss, "processes": n}
    try:
        load = [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        load = None
    return {
        "now": now,
        "gpu": gpu_stats(),
        "host_cpu_pct": host_cpu_pct(),
        "cpus": os.cpu_count(),
        "loadavg": load,
        "mem_available_gb": mem_available_gb(),
        "mem_total_gb": mem_total_gb(),
        "cgroup_memory_gb": cgroup_memory_gb(),
        "cgroup_memory_used_gb": cgroup_memory_used_gb(),
        "cgroup_cpus": cgroup_cpus(),
        "disk_free_gb": shutil.disk_usage(str(ROOT)).free / 1e9,
        "jobs": jobs,
    }


def cached_sha(p):
    st = p.stat()
    key = str(p)
    hit = HASHES.get(key)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    HASHES[key] = (st.st_size, st.st_mtime_ns, h.hexdigest())
    return HASHES[key][2]


def cgroup_memory_used_gb():
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(path).read_text().strip()) / 1e9
        except (OSError, ValueError):
            continue
    return None


def cgroup_memory_gb():
    """The container's memory limit; None if unlimited (incl. cgroup v1's huge sentinel)."""
    total = mem_total_gb()
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                gb = int(raw) / 1e9
                return None if total is not None and gb > total * 1.01 else gb
        except (OSError, ValueError):
            continue
    return None


def cgroup_cpus():
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def mem_total_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError):
        pass
    return None


def _active_self_delete_key():
    """The owner-delivered key when present, otherwise RunPod's legacy injected key."""
    with CREDENTIAL_LOCK:
        return SELF_DELETE_KEY[0] or os.environ.get("RUNPOD_API_KEY", "")


class _NoRunPodRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward an Authorization header to a redirect destination."""

    def _reject(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def _runpod(method, url, body=None, key=None):
    key = _active_self_delete_key() if key is None else key
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", "m3c-gpu-agent/2.0")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.build_opener(_NoRunPodRedirect()).open(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}"


def probe_self_delete():
    """Prove the active key can see this pod (the runner refuses to train otherwise)."""
    # Keep an install/probe and a deadline/dead-man delete ordered. A delete must never use a
    # half-installed key, and no caller ever receives the key or an exception containing it.
    with CREDENTIAL_LOCK:
        pod = os.environ.get("RUNPOD_POD_ID")
        key = _active_self_delete_key()
        if pod and key:
            PROBE[0] = _runpod("GET", f"https://rest.runpod.io/v1/pods/{pod}", key=key)
            log(f"self-delete probe: {PROBE[0]}")
        else:
            PROBE[0] = None
        return PROBE[0]


def install_self_delete_key(key):
    """Install an owner key only in memory, then return its redacted probe result."""
    if (
        not isinstance(key, str)
        or not key
        or len(key) > 4096
        or any(char.isspace() or ord(char) < 33 or ord(char) > 126 for char in key)
    ):
        raise ValueError("invalid self-delete key")
    with CREDENTIAL_LOCK:
        SELF_DELETE_KEY[0] = key
        status = probe_self_delete()
        return {
            "self_delete_armed": bool(os.environ.get("RUNPOD_POD_ID")),
            "self_delete_probe": status,
        }


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
        "agent": "m3c-gpu-agent/v2",
        "cpu_model": cpu_model(),
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
        "mem_total_gb": mem_total_gb(),
        "self_delete_probe": PROBE[0],
        "self_delete_armed": bool(os.environ.get("RUNPOD_POD_ID") and _active_self_delete_key()),
        "last_auth_age": time.time() - LAST_AUTH[0],
        "gpu": gpu(),
        "procs": procs,
        "inputs": sorted(str(p.relative_to(IN)) for p in IN.rglob("*") if p.is_file())[:200],
    }


def self_delete():
    with CREDENTIAL_LOCK:
        pod = os.environ.get("RUNPOD_POD_ID")
        key = _active_self_delete_key()
        if not pod or not key:
            log("self-delete: no pod-scoped credentials")
            return
        code = _runpod("DELETE", f"https://rest.runpod.io/v1/pods/{pod}", key=key)
        log(f"self-delete REST: {code}")
        if code not in (200, 202, 204):
            query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod
            body = json.dumps({"query": query}).encode()
            graphql = _runpod("POST", "https://api.runpod.io/graphql", body, key)
            log(f"self-delete GraphQL: {graphql}")


def kill_all():
    """SIGKILL every job's whole process group (each job is its own session), so no child
    keeps the GPU busy; the pod then idles until the Mac side deletes it."""
    with LOCK:
        for p in PROCS.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
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
            kill_all()
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
        body = raw if raw is not None else json.dumps(_redact_payload(payload)).encode()
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
            if self.path == "/telemetry":
                return self._send(200, telemetry())
            if self.path == "/ls/out":
                rows = []
                for p in OUT.rglob("*"):
                    if p.is_file() and not p.name.endswith(".tmp"):
                        rows.append(
                            {
                                "path": str(p.relative_to(OUT)),
                                "bytes": p.stat().st_size,
                                "sha256": cached_sha(p),
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
        if self.path == "/self-delete-key":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 8192:
                    raise ValueError("invalid self-delete key")
                req = json.loads(self.rfile.read(size))
                if not isinstance(req, dict) or set(req) != {"key"}:
                    raise ValueError("invalid self-delete key")
                result = install_self_delete_key(req["key"])
                return self._send(200, result)
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                # Do not reflect a malformed body: it may contain the key.
                return self._send(400, {"error": "invalid self-delete key"})
        if self.path == "/kill":  # stop one job (its whole process group); never the agent
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                name = str(req["name"])
                with LOCK:
                    proc = PROCS.get(name)
                if proc is None:
                    return self._send(404, {"error": f"no job {name}"})
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
                log(f"kill {name}")
                return self._send(200, {"name": name})
            except (OSError, ValueError, KeyError) as exc:
                return self._send(400, {"error": str(exc)})
        if self.path != "/exec":
            return self._send(404, {"error": "unknown"})
        if DEADLINE and time.time() >= DEADLINE:
            return self._send(410, {"error": "past deadline"})
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
                env.pop("RUNPOD_API_KEY", None)  # request bodies cannot reintroduce a key
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
    probe_self_delete()
    threading.Thread(target=reaper, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
