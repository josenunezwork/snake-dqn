"""Seeder-pod agent: populates the PRIVATE network volume for serverless workers (stdlib).

Runs on a short-lived CPU pod with the network volume mounted at ``/runpod-volume`` (the
same path serverless workers see), shipped in the start command like ``pod_agent.py`` and
reached only through RunPod's HTTPS proxy with header ``X-Fanout-Token`` (the pod env
holds only its sha256). Layout under ``SEED_ROOT`` (default ``/runpod-volume/rpf``)::

    commits/<commit>/repo.tar.gz + manifest.json     immutable, one per commit
    ckpt/<sha256>.pth                                content-addressed, immutable
    runtime/<runtime_id>/sls_handler.py, evenv/, hvenv/, READY.json   immutable once READY

GET  /health                     inventory, build state, cpu model, python
PUT  /in/commit/<commit>.tar.gz  staged; POST /commit {"commit", "repo_sha256"} publishes
PUT  /in/ckpt/<sha>.pth          refused unless the bytes hash to <sha>
PUT  /in/handler/<runtime_id>    sls_handler.py for that runtime (before it is READY)
POST /runtime {"runtime_id", "handler_sha256"}   build both venvs (background), then READY
POST /verify                     re-hash every archive/checkpoint/handler, report
GET  /log                        agent log tail

Pip commands come only from env ``SEED_RUNTIME_JSON`` (set by the runner at create time).
"""

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("SEED_ROOT", "/runpod-volume/rpf"))
STAGE = Path(os.environ.get("SEED_STAGE", "/r/stage"))
LOG = Path(os.environ.get("SEED_LOG", "/r/seed.log"))
TOKEN_SHA = os.environ.get("FANOUT_TOKEN_SHA256", "")
RUNTIME = json.loads(os.environ.get("SEED_RUNTIME_JSON", "{}"))
SELF_DELETE = float(os.environ.get("FANOUT_SELF_DELETE_EPOCH", "0"))
BOOT = uuid.uuid4().hex[:12]
LOCK = threading.Lock()
BUILD = {}  # runtime_id -> "building" | "ready" | "failed: ..."
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RID_RE = re.compile(r"^[0-9a-f]{16}$")


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        with LOG.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def inventory():
    commits, ckpts, runtimes = {}, [], {}
    for m in sorted((ROOT / "commits").glob("*/manifest.json")):
        try:
            commits[m.parent.name] = json.loads(m.read_text())["repo_sha256"]
        except (OSError, ValueError, KeyError):
            commits[m.parent.name] = None
    for p in sorted((ROOT / "ckpt").glob("*.pth")):
        ckpts.append(p.name[:-4])
    for d in sorted((ROOT / "runtime").glob("*")):
        if d.is_dir():
            try:
                runtimes[d.name] = json.loads((d / "READY.json").read_text())
            except (OSError, ValueError):
                runtimes[d.name] = None
    return {"commits": commits, "ckpts": ckpts, "runtimes": runtimes}


def publish_commit(commit, repo_sha):
    staged = STAGE / f"{commit}.tar.gz"
    dest_dir = ROOT / "commits" / commit
    manifest = dest_dir / "manifest.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text()).get("repo_sha256")
        if old == repo_sha and sha_file(dest_dir / "repo.tar.gz") == repo_sha:
            return 200, {"commit": commit, "status": "exists"}
        return 409, {"error": "commit already published with different bytes (immutable)"}
    if not staged.is_file() or sha_file(staged) != repo_sha:
        return 400, {"error": "staged archive missing or sha256 differs"}
    with tarfile.open(staged) as tar:
        names = set(tar.getnames())
    if "research/runpod_fanout/episode.py" not in names:
        return 400, {"error": "archive lacks research/runpod_fanout/episode.py"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".repo.{BOOT}.part"
    shutil.copyfile(staged, tmp)
    os.replace(tmp, dest_dir / "repo.tar.gz")
    body = {"commit": commit, "repo_sha256": repo_sha, "bytes": staged.stat().st_size}
    (dest_dir / ".manifest.part").write_text(json.dumps(body, sort_keys=True))
    os.replace(dest_dir / ".manifest.part", manifest)  # last: a manifest means complete
    log(f"published commit {commit} {repo_sha}")
    return 200, {"commit": commit, "status": "published"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"rc={r.returncode} {' '.join(cmd)[:160]} {r.stderr[-600:]}")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[:4]} rc={r.returncode}: {r.stderr[-300:]}")
    return r.stdout


def build_runtime(rid, handler_sha):
    d = ROOT / "runtime" / rid
    try:
        for name in ("evenv", "hvenv"):
            if (d / name).exists():
                shutil.rmtree(d / name)  # an earlier incomplete build (no READY.json)
            run([sys.executable, "-m", "venv", str(d / name)])
            py = str(d / name / "bin" / "python")
            for args in RUNTIME.get(f"{name}_pip", []):
                run([py, "-m", "pip", *args])
        epy, hpy = str(d / "evenv/bin/python"), str(d / "hvenv/bin/python")
        probe = RUNTIME.get("probe_code") or (
            "import json,sys,numpy,torch;print(json.dumps({'torch':torch.__version__,"
            "'numpy':numpy.__version__,'python':sys.version.split()[0]}))"
        )
        versions = json.loads(run([epy, "-c", probe]))
        for k, want in (RUNTIME.get("expect") or {}).items():
            if versions.get(k) != want:
                raise RuntimeError(f"{k} {versions.get(k)} != expected {want}")
        if RUNTIME.get("hvenv_import"):
            run([hpy, "-c", f"import {RUNTIME['hvenv_import']}"])
        efreeze = run([epy, "-m", "pip", "freeze"])
        hfreeze = run([hpy, "-m", "pip", "freeze"])
        (d / "evenv.freeze.txt").write_text(efreeze)
        (d / "hvenv.freeze.txt").write_text(hfreeze)
        if sha_file(d / "sls_handler.py") != handler_sha:
            raise RuntimeError("handler sha256 changed during the build")
        ready = {
            "runtime_id": rid,
            "handler_sha256": handler_sha,
            "evenv_freeze_sha256": hashlib.sha256(efreeze.encode()).hexdigest(),
            "hvenv_freeze_sha256": hashlib.sha256(hfreeze.encode()).hexdigest(),
            "runtime_spec_sha256": hashlib.sha256(
                json.dumps(RUNTIME, sort_keys=True).encode()
            ).hexdigest(),
            "built_cpu_model": cpu_model(),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **versions,
        }
        (d / ".READY.part").write_text(json.dumps(ready, sort_keys=True))
        os.replace(d / ".READY.part", d / "READY.json")
        BUILD[rid] = "ready"
        log(f"runtime {rid} READY {versions}")
    except Exception as exc:  # noqa: BLE001
        BUILD[rid] = f"failed: {exc}"[:400]
        log(f"runtime {rid} FAILED {exc}")


def verify():
    out = {"bad": [], "checked": 0}
    for m in sorted((ROOT / "commits").glob("*/manifest.json")):
        want = json.loads(m.read_text())["repo_sha256"]
        out["checked"] += 1
        if sha_file(m.parent / "repo.tar.gz") != want:
            out["bad"].append(str(m.parent.name))
    for p in sorted((ROOT / "ckpt").glob("*.pth")):
        out["checked"] += 1
        if sha_file(p) != p.name[:-4]:
            out["bad"].append(p.name)
    for r in sorted((ROOT / "runtime").glob("*/READY.json")):
        out["checked"] += 1
        if sha_file(r.parent / "sls_handler.py") != json.loads(r.read_text())["handler_sha256"]:
            out["bad"].append(r.parent.name)
    out["inventory"] = inventory()
    return out


def self_delete():
    import urllib.request

    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        log("self-delete: no pod-scoped credentials")
        return
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", "rpf-seed-agent/1.0")
    try:
        urllib.request.urlopen(req, timeout=30).read()
        log("self-delete requested")
    except Exception as exc:  # noqa: BLE001
        log(f"self-delete failed: {type(exc).__name__}")


def reaper():
    tried = 0.0
    while True:
        time.sleep(5)
        if SELF_DELETE and time.time() >= SELF_DELETE and time.time() - tried > 600:
            tried = time.time()
            self_delete()


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
        digest = hashlib.sha256(self.headers.get("X-Fanout-Token", "").encode()).hexdigest()
        return bool(TOKEN_SHA) and hmac.compare_digest(digest, TOKEN_SHA)

    def do_GET(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        if self.path == "/health":
            usage = shutil.disk_usage(ROOT) if ROOT.exists() else None
            return self._send(
                200,
                {
                    "agent": "rpf-seed-agent/v1",
                    "boot_id": BOOT,
                    "cpu_model": cpu_model(),
                    "python": sys.version.split()[0],
                    "root": str(ROOT),
                    "root_exists": ROOT.is_dir(),
                    "free_bytes": usage.free if usage else None,
                    "build": dict(BUILD),
                    "inventory": inventory() if ROOT.is_dir() else None,
                    "self_delete_capable": bool(
                        os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_API_KEY")
                    ),
                },
            )
        if self.path == "/log":
            return self._send(200, LOG.read_bytes()[-20000:] if LOG.exists() else b"", "text/plain")
        return self._send(404, {"error": "not found"})

    def _receive(self, dest, want=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.headers.get("Content-Length", "0"))
        h = hashlib.sha256()
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
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
            return 400, {"error": "incomplete or sha256 mismatch", "sha256": digest}
        os.replace(tmp, dest)
        return 200, {"bytes": n, "sha256": digest}

    def do_PUT(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        p = self.path
        m = re.match(r"^/in/commit/([0-9a-f]{40})\.tar\.gz$", p)
        if m:
            return self._send(*self._receive(STAGE / f"{m.group(1)}.tar.gz"))
        m = re.match(r"^/in/ckpt/([0-9a-f]{64})\.pth$", p)
        if m:
            sha = m.group(1)
            dest = ROOT / "ckpt" / f"{sha}.pth"
            if dest.is_file():
                if sha_file(dest) == sha:
                    n = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(n)
                    return self._send(200, {"sha256": sha, "status": "exists"})
                return self._send(409, {"error": "existing checkpoint is corrupt"})
            return self._send(*self._receive(dest, want=sha))
        m = re.match(r"^/in/handler/([0-9a-f]{16})$", p)
        if m:
            d = ROOT / "runtime" / m.group(1)
            if (d / "READY.json").is_file():
                n = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(n)
                return self._send(409, {"error": "runtime already READY (immutable)"})
            return self._send(*self._receive(d / "sls_handler.py"))
        return self._send(403, {"error": "name not allowed"})

    def do_POST(self):
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if self.path == "/commit":
            c, s = str(body.get("commit")), str(body.get("repo_sha256"))
            if not COMMIT_RE.match(c) or not SHA_RE.match(s):
                return self._send(400, {"error": "bad commit/sha"})
            return self._send(*publish_commit(c, s))
        if self.path == "/runtime":
            rid, hsha = str(body.get("runtime_id")), str(body.get("handler_sha256"))
            if not RID_RE.match(rid) or not SHA_RE.match(hsha):
                return self._send(400, {"error": "bad runtime id / sha"})
            d = ROOT / "runtime" / rid
            if (d / "READY.json").is_file():
                ready = json.loads((d / "READY.json").read_text())
                if ready.get("handler_sha256") != hsha:
                    return self._send(409, {"error": "READY runtime has another handler"})
                BUILD[rid] = "ready"
                return self._send(200, {"runtime_id": rid, "status": "ready"})
            if not (d / "sls_handler.py").is_file() or sha_file(d / "sls_handler.py") != hsha:
                return self._send(400, {"error": "handler missing or sha256 differs"})
            with LOCK:
                if BUILD.get(rid) == "building":
                    return self._send(200, {"runtime_id": rid, "status": "building"})
                BUILD[rid] = "building"
            threading.Thread(target=build_runtime, args=(rid, hsha), daemon=True).start()
            return self._send(200, {"runtime_id": rid, "status": "building"})
        if self.path == "/verify":
            return self._send(200, verify())
        return self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


def main():
    for d in (ROOT, STAGE):
        d.mkdir(parents=True, exist_ok=True)
    log(f"seed agent boot {BOOT} root={ROOT}")
    threading.Thread(target=reaper, daemon=True).start()
    port = int(os.environ.get("FANOUT_PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
