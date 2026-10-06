"""RunPod Serverless CPU worker handler for strict gates (sequential strict template v3).

Lives on the PRIVATE network volume, never in an image. ``serverless.py seed --handler strict``
copies this file to ``<volume>/rpf/runtime/<runtime_id>/sls_handler.py`` (its own runtime id:
the Tier-1 handler and runtime are untouched) beside the same two venvs the Tier-1 runtime
uses (``hvenv``: the ``runpod`` SDK; ``evenv``: the pinned numpy/torch CPU stack).

One RunPod job = a batch of whole world UNITS (every arm of one (phase, mix, world)), all run on
this one worker; ``slots`` episodes run at once, 2 threads each::

    {"op": "probe", "runtime_id", "commit", "repo_sha256", "checkpoints": [sha...]}
    {"op": "units", "runtime_id", "commit", "repo_sha256", "checkpoints": [sha...], "job_id",
     "slots", "timeout_seconds", "deadline_epoch", "numerics_env", "intent_text",
     "intent_sha256", "units": [{"unit", "row", "episodes": [episode, ...]}, ...]}

Every episode runs the frozen commit's own ``research.sequential_strict_template.remote_worker``
(from the commit's git archive on the volume, sha256-verified, extracted to local disk) in
``evenv``, one process per episode. The module is fixed here: a job cannot name other code.
Per episode the result says ``kind``: ``ok``, ``episode_error`` (the study's own code raised;
the gate stops), ``binding_error`` (frozen intent / source / spec mismatch on the worker; the
gate stops) or ``worker_failure`` (killed, timed out, no output; the orchestrator re-dispatches
the whole unit). Standard library only, plus the ``runpod`` SDK when run as ``__main__``.
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

HANDLER_SCHEMA = "rpf-sls-strict-handler/v1"
WORKER_MODULE = "research.sequential_strict_template.remote_worker"
EXIT_EPISODE_ERROR = 3
EXIT_BINDING_ERROR = 4
ROOT = Path(os.environ.get("RPF_VOLUME_ROOT", "/runpod-volume/rpf"))
RUNTIME_DIR = Path(os.environ.get("RPF_RUNTIME_DIR", str(Path(__file__).resolve().parent)))
LOCAL = Path(os.environ.get("RPF_LOCAL_ROOT", "/tmp/rpf"))
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,23}$")
EPISODE_ID_RE = re.compile(r"^(calibration|final)-(incumbent|candidate)-[A-Za-z0-9_.-]+-w[0-9]{5}$")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "WEBHOOK", "AWS_")
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
)  # same list as research/runpod_fanout/platform_rule.ISA_FLAGS
MAX_SLOTS = 16
MAX_EPISODES = 64
OUTPUT_TAIL = 1500
_LOCK = threading.Lock()
_VERIFIED: Dict[str, str] = {}


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
    return {
        "handler": HANDLER_SCHEMA,
        "cpu_model": cpu_model(),
        "isa_flags": isa_flags(),
        "cores": os.cpu_count() or 0,
        "runtime_id": ready.get("runtime_id"),
        "handler_sha256": sha_file(Path(__file__)),
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
        raise Refused(f"missing on the volume: {path.name}")
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
    """The frozen commit's git archive (verified) extracted once per worker to local disk."""
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


def worker_command() -> List[str]:
    if os.environ.get("RPF_TEST_STRICT_WORKER_EXEC_JSON"):  # unit tests only (local stub)
        return list(json.loads(os.environ["RPF_TEST_STRICT_WORKER_EXEC_JSON"]))
    return [str(RUNTIME_DIR / "evenv" / "bin" / "python"), "-m", WORKER_MODULE]


def classify(rc: Any, out_path: Path) -> str:
    if rc == 0 and out_path.is_file():
        return "ok"
    if rc == EXIT_EPISODE_ERROR:
        return "episode_error"
    if rc == EXIT_BINDING_ERROR:
        return "binding_error"
    return "worker_failure"


def run_episode(
    unit: str,
    episode: Mapping[str, Any],
    row: Mapping[str, Any],
    inp: Mapping[str, Any],
    repo: Path,
    ckpt: Path,
    work: Path,
) -> Dict[str, Any]:
    episode_id = str(episode["episode_id"])
    started = time.time()
    left = float(inp["deadline_epoch"]) - time.time()
    timeout = max(1.0, min(float(inp["timeout_seconds"]), left))
    payload = work / f"{episode_id}.payload.json"
    out_path = work / f"{episode_id}.out.json"
    payload.write_text(
        json.dumps(
            {
                "unit": unit,
                "episode": dict(episode),
                "row": dict(row),
                "intent_sha256": inp["intent_sha256"],
                "job_id": inp["job_id"],
            },
            sort_keys=True,
        )
    )
    cmd = worker_command() + [
        "--intent",
        str(work / "intent.json"),
        "--payload",
        str(payload),
        "--ckpt-dir",
        str(ckpt),
        "--out",
        str(out_path),
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
        err = done.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, err = "timeout", str(exc.stderr or "")
    kind = classify(rc, out_path)
    result: Dict[str, Any] = {
        "episode_id": episode_id,
        "kind": kind,
        "rc": rc,
        "seconds": round(time.time() - started, 3),
    }
    if kind == "ok":
        data = out_path.read_bytes()
        result.update(output=data.decode("utf-8"), sha256=hashlib.sha256(data).hexdigest())
    else:
        result["stderr_tail"] = err[-OUTPUT_TAIL:]
        if out_path.is_file():  # the worker's own error report (episode / binding error)
            result["error"] = out_path.read_text(encoding="utf-8")[-OUTPUT_TAIL:]
    return result


def check_units(inp: Mapping[str, Any]) -> List[Dict[str, Any]]:
    units = list(inp["units"])
    if not units:
        raise Refused("no units")
    seen, episodes = set(), 0
    for unit in units:
        key = str(unit.get("unit"))
        eps = list(unit.get("episodes") or [])
        if not eps:
            raise Refused(f"unit {key} has no episodes")
        world = {
            (e.get("phase"), e.get("mix"), e.get("world_index"), e.get("world_seed")) for e in eps
        }
        if len(world) != 1:
            raise Refused(f"unit {key} spans several worlds (a unit is one world, never split)")
        row = unit.get("row") or {}
        phase, mix, index, seed = next(iter(world))
        if (
            row.get("mix") != mix
            or row.get("world_index") != index
            or row.get("world_seed") != seed
        ):
            raise Refused(f"unit {key}: roster row is not the unit's world")
        for e in eps:
            eid = str(e.get("episode_id"))
            if not EPISODE_ID_RE.match(eid) or eid in seen:
                raise Refused(f"bad or duplicate episode id {eid!r}")
            seen.add(eid)
        episodes += len(eps)
    if episodes > MAX_EPISODES:
        raise Refused(f"{episodes} episodes > {MAX_EPISODES} per job")
    return units


def handle(inp: Mapping[str, Any]) -> Dict[str, Any]:
    """The pure handler (``runpod`` SDK not needed; tests call this directly)."""
    if not isinstance(inp, Mapping):
        return {"refused": "input must be an object"}
    info = worker_info()
    op = inp.get("op")
    try:
        if op == "probe":
            ensure_runtime(inp)
            ensure_repo(str(inp.get("commit")), str(inp.get("repo_sha256")))
            ensure_ckpts(list(inp["checkpoints"]))
            return {"ok": True, "worker": info}
        if op != "units":
            return {"refused": f"unknown op {op!r}", "worker": info}
        ensure_runtime(inp)
        clean_env(inp.get("numerics_env"))  # validate before running anything
        if time.time() >= float(inp["deadline_epoch"]):
            return {"refused": "deadline", "worker": info}
        if not JOB_ID_RE.match(str(inp.get("job_id"))):
            raise Refused("bad job id")
        slots = int(inp["slots"])
        if not 1 <= slots <= MAX_SLOTS:
            raise Refused("slots out of range")
        units = check_units(inp)
        text = str(inp["intent_text"])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != inp.get("intent_sha256"):
            raise Refused("intent text does not match its sha256")
        repo = ensure_repo(str(inp["commit"]), str(inp["repo_sha256"]))
        ckpt = ensure_ckpts(list(inp["checkpoints"]))
        work = LOCAL / "jobs" / uuid.uuid4().hex
        work.mkdir(parents=True)
        (work / "intent.json").write_text(text, encoding="utf-8")
        items = [(u["unit"], e, u["row"]) for u in units for e in u["episodes"]]
        try:
            with concurrent.futures.ThreadPoolExecutor(slots) as pool:
                results = list(pool.map(lambda it: run_episode(*it, inp, repo, ckpt, work), items))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        by_unit: Dict[str, List[Dict[str, Any]]] = {}
        for (unit, _episode, _row), result in zip(items, results):
            by_unit.setdefault(str(unit), []).append(result)
        return {
            "ok": True,
            "worker": info,
            "units": [{"unit": str(u["unit"]), "results": by_unit[str(u["unit"])]} for u in units],
        }
    except Refused as exc:
        return {"refused": str(exc), "worker": info}
    except (KeyError, TypeError, ValueError) as exc:
        return {"refused": f"bad input: {type(exc).__name__}: {exc}", "worker": info}


def handler(job: Mapping[str, Any]) -> Dict[str, Any]:
    return handle(job.get("input") or {})


if __name__ == "__main__":
    import runpod  # noqa: E402  (only in hvenv on the volume)

    runpod.serverless.start({"handler": handler})
