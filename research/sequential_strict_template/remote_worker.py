#!/usr/bin/env python3
"""Play ONE strict-gate episode on a remote worker (sequential strict template v3).

Run by ``strict_sls_handler.py`` inside the frozen commit's git archive, one process per
episode, with the pinned ``evenv``::

    python -m research.sequential_strict_template.remote_worker --intent INTENT.json \
        --payload EPISODE.json --ckpt-dir DIR --out OUT.json

Before playing it checks, fail closed, that it runs the code and the plan the intent froze:

* the intent bytes have the sha256 the orchestrator sent (``intent_sha256``);
* every source-closure file of the intent has its frozen sha256 in this checkout (paths are
  taken relative to the intent's repo), and the remapped closure digest equals the intent's;
* the study spec resolved from ``intent.spec_ref`` has exactly the frozen descriptor and
  provides ``remote_worker_setup`` (it maps checkpoint sha256s to files in ``--ckpt-dir``).

Exit codes: 0 played (``OUT.json`` holds the record, a platform stamp and the binding checks),
3 the study's own code raised while playing (``episode_error``: the gate stops, as a worker
crash does on the Mac), 4 a binding check or the remote setup failed (``binding_error``: the
gate stops). Anything else (killed, timed out) is a lost worker and the orchestrator
re-dispatches the whole unit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sequential_strict_template import sequential_runner as R  # noqa: E402

OUTPUT_SCHEMA = "sequential-strict-remote-episode/v1"
EXIT_OK = 0
EXIT_EPISODE_ERROR = 3
EXIT_BINDING_ERROR = 4


class BindingError(RuntimeError):
    """The worker would not run exactly the frozen intent, source or spec."""


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "unknown"


def isa_flags() -> list:
    try:
        from research.runpod_fanout.platform_rule import local_isa_flags

        return list(local_isa_flags())
    except ImportError:  # pragma: no cover - the archive always has it
        return []


def platform_stamp(backend: str) -> Dict[str, Any]:
    """What must be equal within a world unit (one worker): OS/arch, Python, torch, numpy, CPU
    model, ISA flags, threads and the numerics env (torch is not imported for the stamp)."""
    model = cpu_model()
    return {
        "backend": backend,
        "platform_id": "|".join(
            [
                f"{platform.system().lower()}-{platform.machine()}",
                f"py{platform.python_version()}",
                f"torch-{_version('torch')}",
                f"numpy-{_version('numpy')}",
                f"cpu-{model}",
            ]
        ),
        "cpu_model": model,
        "isa_flags": isa_flags(),
        "numerics_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR", "ATEN_CPU_CAPABILITY")
        },
    }


def closure_digest(intent: Mapping[str, Any], repo_root: Path) -> Tuple[str, list]:
    """(digest of the intent's closure recomputed from ``repo_root``, files that differ)."""
    closure = intent["source_closure"]
    base = Path(intent["repo"])
    files: Dict[str, str] = {}
    bad = []
    for key, expected in sorted(closure["files"].items()):
        rel = Path(key).relative_to(base)
        path = Path(repo_root) / rel
        actual = R.sha256_file(path) if path.is_file() else "missing"
        files[key] = actual
        if actual != expected:
            bad.append(str(rel))
    return R.canonical_sha(files), bad


def bind(
    intent_text: str, intent_sha: str, payload: Mapping[str, Any], repo_root: Path, spec: Any
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Check the bindings; returns (intent, spec, checks). Raises :class:`BindingError`."""
    if hashlib.sha256(intent_text.encode("utf-8")).hexdigest() != intent_sha:
        raise BindingError("intent bytes do not have the orchestrator's sha256")
    if payload.get("intent_sha256") != intent_sha:
        raise BindingError("payload is bound to another intent")
    intent = json.loads(intent_text)
    if intent.get("template_version") != R.TEMPLATE_VERSION_REMOTE or "execution" not in intent:
        raise BindingError("not a template v3 (remote execution) intent")
    digest, bad = closure_digest(intent, repo_root)
    if bad or digest != intent["source_closure"]["digest"]:
        raise BindingError(f"source closure differs on the worker: {bad[:5]}")
    if spec is None:
        spec = R.resolve_spec(intent["spec_ref"])
    if spec.descriptor() != intent["spec"]:
        raise BindingError("study spec descriptor differs from the intent's")
    if spec.remote_worker_setup is None:
        raise BindingError("the study spec has no remote_worker_setup")
    episode, row = payload["episode"], payload["row"]
    arm = episode["arm"]
    if arm not in intent["arms"]:
        raise BindingError(f"unknown arm {arm!r}")
    if int(row["world_seed"]) != int(episode["world_seed"]) or row["mix"] != episode["mix"]:
        raise BindingError("roster row is not the episode's world")
    checks = {
        "intent_sha256": intent_sha,
        "spec_descriptor_sha256": R.canonical_sha(intent["spec"]),
        "arm_identity_sha256": R.canonical_sha(intent["arms"][arm]),
        "row_sha256": R.canonical_sha(dict(row)),
        "closure_digest": digest,
    }
    return intent, spec, checks


def execute(
    intent_text: str,
    intent_sha: str,
    payload: Mapping[str, Any],
    ckpt_dir: Path,
    *,
    repo_root: Path = REPO,
    spec: Any = None,
    backend: str = "runpod-serverless",
) -> Tuple[int, Dict[str, Any]]:
    """Bind, set up and play one episode; returns (exit code, output document)."""
    episode = payload.get("episode") or {}
    out: Dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "episode_id": episode.get("episode_id"),
        "unit": payload.get("unit"),
    }
    try:
        intent, spec, checks = bind(intent_text, intent_sha, payload, repo_root, spec)
        context = spec.remote_worker_setup(intent, Path(ckpt_dir))
    except Exception as exc:  # noqa: BLE001 - every binding/setup failure is fatal
        out["error"] = f"binding: {type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-3000:]
        return EXIT_BINDING_ERROR, out
    began = time.monotonic()
    try:
        record = spec.episode_runner(dict(episode), dict(payload["row"]), context)
    except Exception as exc:  # noqa: BLE001 - the study's code raised: the gate stops
        out["error"] = f"episode: {type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-3000:]
        return EXIT_EPISODE_ERROR, out
    out.update(
        record=R.json_safe(dict(record)),
        wall_seconds=time.monotonic() - began,
        platform=platform_stamp(backend),
        checks=checks,
    )
    return EXIT_OK, out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    intent_text = args.intent.read_text(encoding="utf-8")
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    code, out = execute(
        intent_text, str(payload.get("intent_sha256")), payload, args.ckpt_dir, repo_root=Path.cwd()
    )
    data = json.dumps(out, sort_keys=True, allow_nan=False)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(data)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
