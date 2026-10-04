"""RunPod Serverless CPU worker handler for the fan-out runner (lives on the PRIVATE volume).

Never baked into an image: the seeder copies this file to
``<volume>/rpf/runtime/<runtime_id>/sls_handler.py`` next to two venvs it builds there:

* ``hvenv``: the ``runpod`` SDK only (runs this handler);
* ``evenv``: exactly the pod pins (numpy/pydantic/PyYAML/psutil/openskill + torch CPU), so an
  episode runs in the same environment as on a fan-out pod.

The template's start command is ``<runtime>/hvenv/bin/python -u <runtime>/sls_handler.py``.
One RunPod job = one small batch of episodes (``slots`` run in parallel, 2 threads each)::

    {"op": "probe", "runtime_id", "commit", "repo_sha256", "checkpoints": [sha...]}
    {"op": "episodes", "runtime_id", "commit", "repo_sha256", "checkpoints": [sha...],
     "job_id", "provider", "slots", "timeout_seconds", "deadline_epoch",
     "require_cpu_model", "episodes": [{"key", "spec"}, ...]}

Each episode runs the job commit's own ``research.runpod_fanout.episode`` (from the git
archive on the volume, sha256-verified, extracted to local disk) in ``evenv``; the result
carries the record bytes exactly as written (same format as pod records).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

HANDLER_SCHEMA = "rpf-sls-handler/v1"
ROOT = Path(os.environ.get("RPF_VOLUME_ROOT", "/runpod-volume/rpf"))
RUNTIME_DIR = Path(os.environ.get("RPF_RUNTIME_DIR", str(Path(__file__).resolve().parent)))
LOCAL = Path(os.environ.get("RPF_LOCAL_ROOT", "/tmp/rpf"))
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,23}$")
KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.json$")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "WEBHOOK", "AWS_")
# Numerics knobs a job may set for its episodes (nothing else from the input reaches env).
NUMERICS_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_CBWR",
    "MKL_ENABLE_INSTRUCTIONS",
    "ONEDNN_MAX_CPU_ISA",
    "ATEN_CPU_CAPABILITY",
)
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
_LOCK = threading.Lock()
_VERIFIED: Dict[str, str] = {}  # path -> sha256 already verified in this worker


class Refused(ValueError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def isa_flags() -> List[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                have = set(line.split(":", 1)[1].split())
                return [f for f in ISA_FLAGS if f in have]
    except OSError:
        pass
    return []


def runtime_manifest() -> Dict[str, Any]:
    try:
        return json.loads((RUNTIME_DIR / "READY.json").read_text())
    except (OSError, ValueError):
        return {}


def worker_info() -> Dict[str, Any]:
    ready = runtime_manifest()
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 0
    return {
        "handler": HANDLER_SCHEMA,
        "cpu_model": cpu_model(),
        "isa_flags": isa_flags(),
        "cores": cores,
        "runtime_id": ready.get("runtime_id"),
        "handler_sha256": sha_file(Path(__file__)),
        "evenv_freeze_sha256": ready.get("evenv_freeze_sha256"),
        "torch": ready.get("torch"),
        "numpy": ready.get("numpy"),
        "python": ready.get("python"),
        "worker_id": os.environ.get("RUNPOD_POD_ID"),
        "dc": os.environ.get("RUNPOD_DC_ID"),
    }


def _verified(path: Path, sha: str) -> None:
    if _VERIFIED.get(str(path)) == sha:
        return
    if not path.is_file():
        raise Refused(f"missing on the volume: {path.relative_to(ROOT)}")
    actual = sha_file(path)
    if actual != sha:
        raise Refused(f"{path.name} sha256 {actual[:12]} != expected {sha[:12]}")
    _VERIFIED[str(path)] = sha


def ensure_runtime(inp: Mapping[str, Any]) -> None:
    ready = runtime_manifest()
    if not ready or ready.get("runtime_id") != inp.get("runtime_id"):
        raise Refused(
            f"runtime {inp.get('runtime_id')} is not this worker's ({ready.get('runtime_id')})"
        )
    if ready.get("handler_sha256") != sha_file(Path(__file__)):
        raise Refused("handler bytes differ from the runtime manifest")


def ensure_repo(commit: str, repo_sha: str) -> Path:
    """The job commit's git archive (verified) extracted once per worker to local disk."""
    if not COMMIT_RE.match(commit) or not SHA_RE.match(repo_sha):
        raise Refused("bad commit / repo sha")
    dest = LOCAL / "repos" / commit
    with _LOCK:
        if (dest / ".rpf-ok").is_file() and (dest / ".rpf-ok").read_text() == repo_sha:
            return dest
        archive = ROOT / "commits" / commit / "repo.tar.gz"
        _verified(archive, repo_sha)
        try:
            manifest = json.loads((ROOT / "commits" / commit / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise Refused(f"commit {commit[:12]} has no manifest on the volume") from exc
        if manifest.get("repo_sha256") != repo_sha:
            raise Refused("volume manifest repo sha differs from the job's")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
        (dest / ".rpf-ok").write_text(repo_sha)
        return dest


def ensure_ckpts(shas: List[str]) -> Path:
    out = LOCAL / "ckpt"
    with _LOCK:
        out.mkdir(parents=True, exist_ok=True)
        for sha in shas:
            if not isinstance(sha, str) or not SHA_RE.match(sha):
                raise Refused("bad checkpoint sha")
            local = out / f"{sha}.pth"
            if _VERIFIED.get(str(local)) == sha:
                continue
            src = ROOT / "ckpt" / f"{sha}.pth"
            _verified(src, sha)
            tmp = out / f".{sha}.{uuid.uuid4().hex[:8]}"
            shutil.copyfile(src, tmp)
            os.replace(tmp, local)
            _verified(local, sha)
    return out


def episode_key(spec: Mapping[str, Any]) -> str:
    """Same formula as ``jobspec.episode_key`` (the pinned runner file)."""
    group = f"{spec['wrapper']}__h{int(spec['horizon'])}__{spec['engine']}"
    return f"{group}/{spec['arm']}-{spec['mix']}-{int(spec['world_seed'])}.json"


def clean_env(numerics: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(m in k.upper() for m in SECRET_MARKERS) and not k.startswith(("RPF_", "RUNPOD_"))
    }
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "SNAKE_DQN_DEVICE": "cpu",
            "PYTHONHASHSEED": "0",
        }
    )
    for k, v in (numerics or {}).items():
        if k not in NUMERICS_KEYS:
            raise Refused(f"numerics_env key {k!r} not allowed")
        env[k] = str(v)
    return env


def run_episode(
    item: Mapping[str, Any], inp: Mapping[str, Any], repo: Path, ckpt: Path, out_root: Path
) -> Dict[str, Any]:
    key, spec = item["key"], item["spec"]
    started = time.time()
    left = float(inp["deadline_epoch"]) - time.time()
    timeout = max(1.0, min(float(inp["timeout_seconds"]), left))
    spec_path = out_root.parent / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.spec.json"
    spec_path.write_text(json.dumps(spec))
    python = str(RUNTIME_DIR / "evenv" / "bin" / "python")
    head = [python, "-m", "research.runpod_fanout.episode"]
    if os.environ.get("RPF_TEST_EPISODE_EXEC_JSON"):  # unit tests only (local stub)
        head = json.loads(os.environ["RPF_TEST_EPISODE_EXEC_JSON"])
    cmd = head + [
        "--spec",
        str(spec_path),
        "--out-root",
        str(out_root),
        "--ckpt-dir",
        str(ckpt),
        "--provider",
        str(inp["provider"]),
        "--job-id",
        str(inp["job_id"]),
        "--commit",
        str(inp["commit"]),
    ]
    try:
        done = subprocess.run(
            cmd,
            cwd=str(repo),
            env=clean_env(inp.get("numerics_env")),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc: Any = done.returncode
        out, err = done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out, err = "timeout", "", str(exc.stderr or "")[-1500:]
    seconds = round(time.time() - started, 3)
    status = None
    for line in (out or "").splitlines()[::-1]:
        try:
            status = json.loads(line).get("status")
            break
        except (ValueError, AttributeError):
            continue
    path = out_root / key
    if rc == 0 and status == "written" and path.is_file():
        data = path.read_bytes()
        return {
            "key": key,
            "ok": True,
            "record": data.decode("utf-8"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "seconds": seconds,
        }
    return {
        "key": key,
        "ok": False,
        "rc": rc,
        "status": status,
        "seconds": seconds,
        "stderr_tail": (err or "")[-1500:],
    }


def check_common(inp: Mapping[str, Any]) -> None:
    ensure_runtime(inp)
    if not isinstance(inp.get("checkpoints"), list):
        raise Refused("checkpoints must be a list")


def handle(inp: Mapping[str, Any]) -> Dict[str, Any]:
    """The pure handler (``runpod`` SDK not needed; tests call this directly)."""
    if not isinstance(inp, Mapping):
        return {"refused": "input must be an object"}
    info = worker_info()
    op = inp.get("op")
    try:
        if op == "probe":
            check_common(inp)
            ensure_repo(str(inp.get("commit")), str(inp.get("repo_sha256")))
            ensure_ckpts(list(inp["checkpoints"]))
            return {"ok": True, "worker": info}
        if op != "episodes":
            return {"refused": f"unknown op {op!r}", "worker": info}
        check_common(inp)
        want = inp.get("require_cpu_model")
        if want and info["cpu_model"] != want:
            # Optional pin: leave without running; the worker is recycled.
            return {"refused": "cpu_model", "worker": info, "refresh_worker": True}
        clean_env(inp.get("numerics_env"))  # validate before running anything
        if time.time() >= float(inp["deadline_epoch"]):
            return {"refused": "deadline", "worker": info}
        if not JOB_ID_RE.match(str(inp.get("job_id"))):
            raise Refused("bad job id")
        slots = int(inp["slots"])
        items = list(inp["episodes"])
        if not 1 <= slots <= 16 or not 1 <= len(items) <= 64:
            raise Refused("slots/episodes out of range")
        for item in items:
            key = item.get("key")
            if not isinstance(key, str) or not KEY_RE.match(key) or ".." in key:
                raise Refused(f"bad episode key {key!r}")
            if episode_key(item["spec"]) != key:
                raise Refused(f"episode key {key} does not match its spec")
        repo = ensure_repo(str(inp["commit"]), str(inp["repo_sha256"]))
        ckpt = ensure_ckpts(list(inp["checkpoints"]))
        work = LOCAL / "jobs" / uuid.uuid4().hex
        out_root = work / "out"
        out_root.mkdir(parents=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(slots) as pool:
                results = list(
                    pool.map(lambda it: run_episode(it, inp, repo, ckpt, out_root), items)
                )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return {"ok": True, "worker": info, "results": results}
    except Refused as exc:
        return {"refused": str(exc), "worker": info}
    except (KeyError, TypeError, ValueError) as exc:
        return {"refused": f"bad input: {type(exc).__name__}: {exc}", "worker": info}


def handler(job: Mapping[str, Any]) -> Dict[str, Any]:
    return handle(job.get("input") or {})


if __name__ == "__main__":
    import runpod  # noqa: E402  (only in hvenv on the volume)

    runpod.serverless.start({"handler": handler})
