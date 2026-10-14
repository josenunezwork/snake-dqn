#!/usr/bin/env python3
"""Short-lived CUDA benchmark session on ONE RunPod RTX 4090 pod (perf pass, item 2).

Runs the perf benches (``research/perf_learner_20261013/bench_learner.py`` and
``research/perf_actor_20261013/bench_actor.py``) one after another on a 1x RTX 4090 SECURE
pod with a hard lifetime, pulls their result JSONs to a local directory and deletes the pod.
It reuses the M3-C GPU runner's money-safety pieces (``research/redesign_m3c_20261008/
gpu_run_v2.py``, unchanged): the pod body and pod agent v2 (jobs killed and ``/exec`` refused
at the deadline, dead-man switch), the detached Mac watchdog, the shared spend ledger
(reserve before create, bind, release, finish), definite-refusal-only create retries,
adopt-by-name after an unknown create outcome, the balance-drop spend cap, and
delete-and-verify on EVERY exit (success, command failure, any error, KeyboardInterrupt /
SIGTERM / SIGHUP). Pods are named ``perfbench-*`` and this tool only ever sees, guards or
deletes pods with that prefix.

No G2 / per-step 4x gate. G2 compares a pod's rate for a training step with the Mac's rate
for the same step and refuses pods that are not >= 4x faster. This session trains nothing:
it measures CUDA throughput (learner precision / batch-scale / compile, CUDA acting), which
the Mac cannot measure at all, so there is no Mac baseline for the rule to compare against
and the measurement itself is what a later G2 decision needs. The spend is bounded instead
by the hard lifetime x max_hourly (default 1.0 h x $0.95 = $0.95), the ledger reservation
and the balance-drop spend cap. The 4x rule still governs every TRAINING step that uses
these numbers (a pod plan's G2).

Launching spends money: only with ``--confirm`` AND the user's explicit approval in chat
(AGENTS.md / STATUS.md: approved spend and uploads only, always clean up).

Usage::

    # dry run: pod body, commands (checked against the committed bench scripts), uploads and
    # the cost bound. Spends nothing, calls no RunPod API (unless --with-account, read-only).
    ./venv/bin/python research/perf_podbench_20261013/pod_bench.py bench --dry-run

    # the session (clean committed tree only); results land in <out-dir>/out/
    ./venv/bin/python research/perf_podbench_20261013/pod_bench.py bench --confirm \\
        --out-dir ../snake-dqn-artifacts/perf-cuda-bench-20261014 \\
        --accept-no-self-delete "user approved the bench pod in chat on 2026-10-14"

    # edit the command list (e.g. compile modes from another branch), then run it
    ... pod_bench.py bench --emit-commands /tmp/cmds.json
    ... pod_bench.py bench --dry-run --commands /tmp/cmds.json

    # manual cleanup of any perfbench- pod
    ./venv/bin/python research/perf_podbench_20261013/pod_bench.py cleanup --out-dir DIR --confirm

Commands file (schema ``perf-podbench-commands/v1``)::

    {"schema": "perf-podbench-commands/v1",
     "commands": [{"name": "learner_bv8", "timeout_s": 1200,
                   "argv": ["python", "research/perf_learner_20261013/bench_learner.py",
                            "--devices", "cuda", "--out", "{out}/learner_bv8.json"],
                   "env": {"M3C_V8_BACKEND": "numba"}}]}

Placeholders in ``argv``: ``{out}`` (the pod's result dir; everything there is pulled),
``{student}`` (the M2b student checkpoint), ``{ckpt}`` (the vector61 checkpoint dir). Every
command is checked against the COMMITTED script before anything is created: a flag the
script's argparse does not define, or a ``--configs`` name missing from its ``CONFIGS``,
SKIPS that command with the reason (e.g. compile modes that live on another branch) instead
of spending pod time on an argparse error. A command that fails or times out on the pod is
recorded (exit code, log tail) and the session continues with the next one. The DEFAULT list
is built from the committed benches too: torch.compile commands appear only when they have
compile support (perf-compile's learner ``--compile`` / actor ``--compile-actor`` flags, or
compile-named ``CONFIGS``), and options a bench lacks (``--init``, ``--note``,
``--trainer-args``) are left out.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3c_20261008 import gpu_run_v2 as gr  # noqa: E402

PREFIX = "perfbench-"
PREFIX_RE = re.compile(r"perfbench(-[a-z0-9]+)*-")
JOB_ID = "perf-cuda-bench"
JOB_ID_RE = re.compile(r"perf-cuda-bench(-[a-z0-9]+)*")
COMMANDS_SCHEMA = "perf-podbench-commands/v1"
SESSION_SCHEMA = "perf-podbench-session/v1"
LIFETIME_H = 1.0
MAX_LIFETIME_H = 3.0
MAX_HOURLY = 0.95  # covers the RTX 4090 secure list price ($0.89/h on 2026-10-08)
SPEND_CAP = 1.50  # the balance-drop guard trips at cap - 0.50 = $1.00 (> the $0.95 bound)
DEADLINE_MARGIN_S = 600  # the agent kills jobs / refuses /exec 10 min before the lifetime
WATCHDOG_GRACE_S = 300  # gpu_run_v2's watchdog fires at the lifetime + 5 min
CONTAINER_DISK_GB = 40
MIN_RAM_GB = 96  # the production pod plans' host class (m3c / m4a2 plans), so the CPU-side
MIN_VCPU = 12  # actor numbers transfer to production pods
AGENT_UP_S = 1200
UPLOAD_BUDGET_S = 900
SETUP_BUDGET_S = 1200
PULL_MAX_S = 300
DEFAULT_TIMEOUT_S = 900
SETUP_ESTIMATE_S = 900  # boot + image pull + uploads + pip (for the dry run's budget note)
POD_ROOT = "/r"
POD_REPO = "/r/repo"
PLACEHOLDERS = {
    "out": "/r/out",
    "student": "/r/in/ckpt/student_m2b_final.pth",
    "ckpt": "/r/in/ckpt",
}
LEARNER = "research/perf_learner_20261013/bench_learner.py"
ACTOR = "research/perf_actor_20261013/bench_actor.py"
LEARNER_CONFIGS = ("baseline", "t1", "t1_fp16", "t1_bf16", "t1_k2", "t1_k4", "t1_bf16_k2")
# torch.compile benches (perf-compile branch: bench_learner ``--compile MODE[,MODE]`` adds a
# compiled twin of each listed fast config; bench_actor ``--compile-actor MODE[,MODE]``).
# max-autotune compiles slowly, so it gets its own lower-priority command.
COMPILE_LEARNER_CONFIGS = ("t1", "t1_bf16", "t1_k2")
COMPILE_LEARNER_MODES = "default,reduce-overhead"
COMPILE_LEARNER_AUTOTUNE = "max-autotune-no-cudagraphs"
COMPILE_ACTOR_MODES = "off,default,reduce-overhead"
COMPILE_DRIFT_MODES = "default,reduce-overhead"
# The B8 recipe's actor-side flags (research/redesign_m4a_20261012/plans/m4a2_pod_plan.json):
# bv8 observation (one compiled v8 evaluation per frame), on-policy v8 labels, mixed roster.
B8_ACTOR_ARGS = (
    "--obs-version bv8 --onpolicy-labels --label-frac 1.0 --anchor-mode v8margin "
    "--mixed-frac 0.25 --reward mass --mass-kappa 1.0 --kill-scale 0"
)
POD_NOTE = "RunPod RTX 4090 secure pod, perf-cuda-bench session (one host, single session)"
NAME_RE = re.compile(r"[A-Za-z0-9_\-]{1,40}")
SCRIPT_RE = re.compile(r"(research|src)/[A-Za-z0-9_\-/]+\.py")
ENV_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
ENV_DENY_PREFIXES = ("RUNPOD_", "M3_")
PH_RE = re.compile(r"\{([A-Za-z_]+)\}")
HOST_KEYS = (
    "gpu",
    "cpu_model",
    "cpus",
    "cpus_allowed",
    "cgroup_cpus",
    "cgroup_memory_gb",
    "mem_total_gb",
    "disk_free_gb",
)


# ----------------------------------------------------------------------------- the tree
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def clean_commit(repo: Path = REPO) -> str:
    """HEAD of a clean worktree; refuses a dirty one (the pod runs exactly a committed tree)."""
    if _git(repo, "status", "--porcelain").strip():
        raise SystemExit(
            "refusing: the worktree is dirty (the pod runs a committed tree: commit first)"
        )
    return _git(repo, "rev-parse", "HEAD").strip()


def committed_source(repo: Path, commit: str, path: str) -> Optional[str]:
    done = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, text=True
    )
    return done.stdout if done.returncode == 0 else None


def script_caps(source: str) -> Dict[str, Any]:
    """The flags a script's argparse defines and the keys of its top-level ``CONFIGS``."""
    tree = ast.parse(source)
    flags = {"-h", "--help"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    flags.add(str(arg.value))
    configs = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        named = any(isinstance(t, ast.Name) and t.id == "CONFIGS" for t in targets)
        if named and isinstance(node.value, ast.Dict):
            configs = {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    return {"flags": flags, "configs": configs}


def load_caps(
    repo: Path, commit: str, scripts: Sequence[str]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Capabilities of each script AS COMMITTED (None: not in the commit)."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for path in sorted(set(scripts)):
        src = committed_source(repo, commit, path)
        out[path] = script_caps(src) if src is not None else None
    return out


# ----------------------------------------------------------------------------- commands
def compile_configs(configs: Optional[Any]) -> List[str]:
    """Config names that are torch.compile variants: ``*compile*`` or a ``c`` name part
    (perf-compile's ``t1_c`` / ``t1_c_ro`` / ``t1_bf16_k2_c``; not ``t1_cl``)."""
    return sorted(n for n in (configs or ()) if "compile" in n or "c" in n.split("_"))


def default_commands(caps: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """The default bench list for the committed scripts' capabilities.

    Order = priority (a lifetime-truncated session keeps the first ones): learner configs
    (production flags); torch.compile (default / reduce-overhead) if the committed bench has
    it; the actor (b obs, as the perf-actor bench; then the B8 recipe's bv8 actor), each at
    64 / 128 / 256 worlds with the numba vector61 backend and with long bodies; the slow
    max-autotune learner compile and the compiled actor forward; the precision drift on CUDA
    (then its compiled variants). Compile support is detected from the COMMITTED scripts: a
    ``--compile`` flag (perf-compile's learner bench: compiled twins of the listed configs),
    compile-named ``CONFIGS`` entries, or the actor bench's ``--compile-actor`` flag; a bench
    without them simply gets no compile command.
    """
    lc = caps.get(LEARNER) or {"flags": set(), "configs": set()}
    ac = caps.get(ACTOR) or {"flags": set(), "configs": set()}
    # learner options this commit's bench may lack (then left out, not failed)
    lopts = ["--init", "{student}"] if "--init" in lc["flags"] else []
    if "--note" in lc["flags"]:  # the bench's default note says "noisy Mac"
        lopts += ["--note", POD_NOTE]
    learner = ["python", LEARNER, "--devices", "cuda", "--mode", "bv8"]
    timing = ["--updates", "50", "--rounds", "5", "--warm", "10"]
    known = lc["configs"]

    def lcmd(name: str, configs: Sequence[str], extra: Sequence[str], timeout: int):
        argv = learner + ["--configs", ",".join(configs)] + list(extra) + timing + lopts
        return {
            "name": name,
            "timeout_s": timeout,
            "argv": argv + ["--out", f"{{out}}/{name}.json"],
        }

    lcfg = [n for n in LEARNER_CONFIGS if known is None or n in known]
    cmds: List[Dict[str, Any]] = [lcmd("learner_bv8", lcfg, [], 1200)]
    late: List[Dict[str, Any]] = []  # slow compile benches: after the actor
    if "--compile" in lc["flags"]:
        base = [n for n in COMPILE_LEARNER_CONFIGS if known is None or n in known]
        cmds.append(lcmd("learner_bv8_compile", base, ["--compile", COMPILE_LEARNER_MODES], 1500))
        late.append(
            lcmd("learner_bv8_autotune", ["t1"], ["--compile", COMPILE_LEARNER_AUTOTUNE], 1500)
        )
    elif compile_configs(known):
        cmds.append(lcmd("learner_bv8_compile", ["t1"] + compile_configs(known), [], 1500))
    actor = ["python", ACTOR, "--configs", "numba", "--worlds", "64,128,256", "--devices", "cuda"]
    actor += ["--frames", "150", "--warm", "30", "--init", "{student}"]
    if "--note" in ac["flags"]:
        actor += ["--note", POD_NOTE]
    variants = [("actor_b", [])]
    if "--trainer-args" in ac["flags"]:
        variants.append(("actor_b8", ["--trainer-args=" + B8_ACTOR_ARGS]))
    for name, extra in variants:
        for suffix, grow in (("", []), ("_long", ["--grow-to", "1200"])):
            cmds.append(
                {
                    "name": name + suffix,
                    "timeout_s": 1200,
                    "argv": actor + extra + grow + ["--out", f"{{out}}/{name}{suffix}.json"],
                }
            )
    cmds += late
    if "--compile-actor" in ac["flags"]:
        argv = actor + ["--compile-actor", COMPILE_ACTOR_MODES, "--out", "{out}/actor_compile.json"]
        cmds.append({"name": "actor_compile", "timeout_s": 1500, "argv": argv})
    elif compile_configs(ac["configs"]):
        argv = actor + ["--out", "{out}/actor_compile.json"]
        argv[argv.index("--configs") + 1] = ",".join(["numba"] + compile_configs(ac["configs"]))
        cmds.append({"name": "actor_compile", "timeout_s": 1500, "argv": argv})
    drift = ["python", LEARNER, "--drift", "--devices", "cuda", "--mode", "bv8"]
    drift += ["--drift-updates", "200"] + lopts
    cmds.append(
        {
            "name": "learner_drift",
            "timeout_s": 900,
            "argv": drift + ["--out", "{out}/learner_drift.json"],
        }
    )
    if "--compile" in lc["flags"]:  # compiled drift vs fp32 (a separate, last command)
        argv = drift + [
            "--compile",
            COMPILE_DRIFT_MODES,
            "--out",
            "{out}/learner_drift_compile.json",
        ]
        cmds.append({"name": "learner_drift_compile", "timeout_s": 1200, "argv": argv})
    return {"schema": COMMANDS_SCHEMA, "commands": cmds}


def parse_commands(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Schema-check a commands document; raises SystemExit listing every problem."""
    problems: List[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != COMMANDS_SCHEMA:
        raise SystemExit(f"commands: schema must be {COMMANDS_SCHEMA!r}")
    raw = doc.get("commands")
    if not isinstance(raw, list) or not raw:
        raise SystemExit("commands: 'commands' must be a non-empty list")
    out, seen = [], set()
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            problems.append(f"#{i}: not an object")
            continue
        name = str(c.get("name", ""))
        if not NAME_RE.fullmatch(name) or name == "setup":
            problems.append(f"#{i}: bad name {name!r}")
        if name in seen:
            problems.append(f"#{i}: duplicate name {name!r}")
        seen.add(name)
        argv = c.get("argv")
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(a, str) for a in argv):
            problems.append(f"{name}: argv must be a list of >= 2 strings")
            continue
        if argv[0] != "python":
            problems.append(f"{name}: argv[0] must be 'python'")
        if not SCRIPT_RE.fullmatch(argv[1]) or ".." in argv[1].split("/"):
            problems.append(f"{name}: argv[1] must be a repo-relative research/ or src/ script")
        for a in argv:
            for key in PH_RE.findall(a):
                if key not in PLACEHOLDERS:
                    problems.append(f"{name}: unknown placeholder {{{key}}}")
        timeout = c.get("timeout_s", DEFAULT_TIMEOUT_S)
        if not isinstance(timeout, (int, float)) or not 30 <= timeout <= 3600:
            problems.append(f"{name}: timeout_s must be in [30, 3600]")
        env = c.get("env", {})
        if not isinstance(env, dict):
            problems.append(f"{name}: env must be an object")
            env = {}
        for k, v in env.items():
            if not ENV_RE.fullmatch(str(k)) or str(k).startswith(ENV_DENY_PREFIXES):
                problems.append(f"{name}: env key {k!r} not allowed")
            if not isinstance(v, str):
                problems.append(f"{name}: env {k} must be a string")
        out.append(
            {"name": name, "argv": list(argv), "timeout_s": float(timeout), "env": dict(env)}
        )
    if problems:
        raise SystemExit("bad commands: " + "; ".join(problems))
    return out


def unsupported(argv: Sequence[str], caps: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why the committed script cannot run ``argv`` (None: it can)."""
    if caps is None:
        return f"{argv[1]} is not in the committed tree"
    bad_flags = []
    bad_configs: List[str] = []
    for i, tok in enumerate(argv[2:], start=2):
        flag, eq, inline = tok.partition("=")
        if not tok.startswith("-") or tok == "-" or re.fullmatch(r"-\d.*", tok):
            continue  # a value
        if " " in flag:
            continue  # a value: argparse reads a dash-led token with a space as a value
            # (but ``--flag=value with spaces`` is the flag ``--flag``: checked below)
        if flag not in caps["flags"]:
            bad_flags.append(flag)
        elif flag == "--configs" and caps.get("configs") is not None:
            value = inline if eq else (argv[i + 1] if i + 1 < len(argv) else "")
            bad_configs += [n for n in value.split(",") if n and n not in caps["configs"]]
    why = []
    if bad_flags:
        why.append(f"flags not in the committed {argv[1]}: {sorted(set(bad_flags))}")
    if bad_configs:
        why.append(f"configs not in the committed {argv[1]} CONFIGS: {sorted(set(bad_configs))}")
    return "; ".join(why) or None


def split_runnable(
    commands: List[Dict[str, Any]], caps: Dict[str, Optional[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(runnable, skipped): a command the committed script cannot parse is skipped, not run."""
    runnable, skipped = [], []
    for c in commands:
        why = unsupported(c["argv"], caps.get(c["argv"][1]))
        if why:
            skipped.append({"name": c["name"], "reason": why, "argv": c["argv"]})
        else:
            runnable.append(c)
    return runnable, skipped


def resolve_argv(argv: Sequence[str]) -> List[str]:
    return [PH_RE.sub(lambda m: PLACEHOLDERS[m.group(1)], a) for a in argv]


# ----------------------------------------------------------------------------- pod pieces
def bind_prefix(prefix: str) -> None:
    """gpu_run_v2 reads its pod prefix from a module global (as its own --pod-prefix does):
    our_pods / delete_and_verify / the watchdog then only see ``perfbench-`` pods."""
    if not PREFIX_RE.fullmatch(prefix):
        raise SystemExit(f"pod prefix must look like 'perfbench-' or 'perfbench-x-', got {prefix}")
    gr.POD_NAME = prefix


def bench_pod_body(
    name: str, token_sha: str, deadline: float, until: float, args
) -> Dict[str, Any]:
    """gpu_run_v2's pod body (1x RTX 4090, SECURE, agent v2 in the start command, the
    deadline / self-delete epochs) with a smaller container disk and the chosen image."""
    body = gr.pod_body(name, token_sha, deadline, until, args.min_ram_gb, args.min_vcpu)
    body.update(containerDiskInGb=CONTAINER_DISK_GB, imageName=args.image)
    assert body["gpuTypeIds"] == [gr.GPU] and body["gpuCount"] == 1
    assert body["cloudType"] == "SECURE" and body["volumeInGb"] == 0
    return body


def bench_ckpts() -> List[Tuple[Path, str, str]]:
    """(local path, upload dst, pinned sha): the vector61 pool + frp3-s12 (the actor's
    opponents and v8 carrier, loaded via SNAKE_CKPT_DIR) and the M2b student (``{student}``)."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008 import ni_spec

    out = [
        (gr.CKPT_DIR / name, f"ckpt/{name}", sha)
        for name, sha in list(dev_screen.POOL) + [ni_spec.FRP3_S12]
    ]
    out.append((gr.STUDENT, "ckpt/student_m2b_final.pth", gr.STUDENT_SHA))
    return out


def build_bench_inputs(
    work: Path,
    commit: str,
    repo: Path = REPO,
    ckpts: Optional[List[Tuple[Path, str, str]]] = None,
) -> Dict[str, Any]:
    """git archive of ``commit`` + the sha-pinned checkpoints (no demo data: the benches
    build synthetic replay / use ``--demo-data .``)."""
    work.mkdir(parents=True, exist_ok=True)
    repo_tar = work / "repo.tar.gz"
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(repo_tar), commit], cwd=repo, check=True
    )
    files = [{"src": str(repo_tar), "dst": "repo.tar.gz", "sha256": gr.sha_file(repo_tar)}]
    for src, dst, sha in bench_ckpts() if ckpts is None else ckpts:
        if gr.sha_file(Path(src)) != sha:
            raise SystemExit(f"{src} does not hash to its pinned sha {sha}")
        files.append({"src": str(src), "dst": dst, "sha256": sha})
    return {
        "commit": commit,
        "files": files,
        "bytes": sum(Path(f["src"]).stat().st_size for f in files),
    }


def bench_setup_argv() -> List[str]:
    """Pod setup: the committed tree, the training setup's pinned pip deps, a CUDA check, the
    host's GPU / CPU, and an import smoke of both benches (+ the actor's checkpoint context).
    One command per line under ``set -euo pipefail`` (a failing step stops the setup)."""
    versions = (
        "import torch, numba, numpy; assert torch.cuda.is_available(); "
        'print("torch", torch.__version__, "cuda", torch.version.cuda, '
        'torch.cuda.get_device_name(0), "numba", numba.__version__, "numpy", numpy.__version__)'
    )
    smoke = (
        'import sys; sys.path.insert(0, "."); '
        "import research.perf_learner_20261013.bench_learner; "
        "import research.perf_actor_20261013.bench_actor; "
        "from research.redesign_scope_20261007 import grid_h5000_identity as g; g._context(1)"
    )
    steps = [
        "set -euo pipefail",
        "mkdir -p /r/repo /r/out",
        "tar -xzf /r/in/repo.tar.gz -C /r/repo",
        "pip install -q --no-cache-dir " + " ".join(gr.PIP),
        f"python -c '{versions}'",
        "nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit "
        "--format=csv || true",
        "(lscpu | head -20) || true",
        "cd /r/repo",
        f"SNAKE_CKPT_DIR=/r/in/ckpt python -c '{smoke}'",
        "echo SETUP_OK",
    ]
    return ["bash", "-c", "\n".join(steps)]


def bench_env(commit: str) -> Dict[str, str]:
    return {
        "SNAKE_CKPT_DIR": "/r/in/ckpt",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "M3_COMMIT": commit,
    }


def cost_bound(args) -> Dict[str, float]:
    life_s = float(args.lifetime_hours) * 3600.0
    rate = float(args.max_hourly)
    return {
        "max_hourly_usd": rate,
        "lifetime_hours": float(args.lifetime_hours),
        "worst_case_usd": round(rate * life_s / 3600.0, 4),
        "worst_case_if_runner_dies_usd": round(rate * (life_s + WATCHDOG_GRACE_S) / 3600.0, 4),
        "ledger_reservation_usd": round(1.02 * rate * (life_s + 300.0) / 3600.0, 4),
        "spend_cap_usd": float(args.spend_cap),
        "spend_guard_trips_at_balance_drop_usd": round(float(args.spend_cap) - 0.5, 4),
    }


def make_agent(pod_id: str, token: str):
    return gr.Agent(pod_id, token)


def keep_awake() -> None:
    if sys.platform == "darwin":  # keep the Mac awake while this process lives
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], start_new_session=True)


# ----------------------------------------------------------------------------- the drive
def _note(run_dir: Path, **row: Any) -> None:
    row["t"] = gr.now()
    gr.append_jsonl(run_dir / "progress.jsonl", row)
    gr.say(json.dumps(row, default=str))


def _upload(agent, rp, state, args, net, files) -> Optional[str]:
    t0 = gr.now()
    for f in files:
        stop = gr.guard(rp, state, args)
        if stop:
            return stop
        while True:
            if gr.now() - t0 > UPLOAD_BUDGET_S:
                return "upload budget exceeded"
            try:
                agent.put(f["dst"], f["src"], f["sha256"])
                net.ok()
                break
            except Exception as exc:  # noqa: BLE001
                client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < 500
                if client_error or net.fail():
                    return f"upload failed: {f['dst']}: {exc}"
                time.sleep(15)
    return None


def _fetch_log(agent, run_dir: Path, name: str) -> Optional[str]:
    try:
        data = agent.get(f"/log/{name}", timeout=60)
    except Exception:  # noqa: BLE001
        return None
    path = run_dir / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data.decode(errors="replace")[-1500:]


def _pull(agent, run_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Pull every new / changed file under the pod's /r/out (sha-checked) into run_dir/out."""
    budget = max(30.0, min(float(PULL_MAX_S), state["deadline"] + 600 - gr.now()))
    try:
        res = gr.download_rest(agent, run_dir, budget)
    except Exception as exc:  # noqa: BLE001
        res = {"error": f"{type(exc).__name__}: {exc}"}
    state.setdefault("pulls", []).append({"t": gr.now(), **res})
    return res


def drive_bench(agent, rp, run_dir: Path, state, inputs, args, commands) -> str:
    """Agent up -> uploads -> setup -> each command in order -> pulls. Returns the outcome."""
    net = gr.NetBudget()
    t0 = gr.now()
    while True:
        stop = gr.guard(rp, state, args)
        if stop:
            return stop
        try:
            h = agent.health()
            net.ok()
            break
        except Exception:  # noqa: BLE001
            if gr.now() - t0 > AGENT_UP_S:
                return "agent never came up"
            time.sleep(15)
    state["host"] = {k: h.get(k) for k in HOST_KEYS}
    gr.save_state(run_dir, state)
    _note(run_dir, stage="agent_up", self_delete_armed=h.get("self_delete_armed"), **state["host"])
    if not h.get("self_delete_armed") or h.get("self_delete_probe") != 200:
        why = f"pod self-delete not proven (probe {h.get('self_delete_probe')})"
        if not args.accept_no_self_delete:
            return why
        state["self_delete_waived"] = {"reason": why, "note": args.accept_no_self_delete}
        _note(run_dir, stage="self_delete_waived", reason=why, waiver=args.accept_no_self_delete)
    stop = _upload(agent, rp, state, args, net, inputs["files"])
    if stop:
        return stop
    _note(run_dir, stage="uploaded", files=len(inputs["files"]), bytes=inputs["bytes"])
    agent.exec("setup", bench_setup_argv(), POD_ROOT, {})
    rc, stop = gr._wait_job(agent, rp, state, args, net, "setup", SETUP_BUDGET_S)
    if stop:
        return stop
    tail = _fetch_log(agent, run_dir, "setup") or ""
    state["setup"] = {"returncode": rc, "tail": tail}
    _note(run_dir, stage="setup_done", returncode=rc, tail=tail)
    if rc != 0 or "SETUP_OK" not in tail:
        return "setup failed"
    env = bench_env(inputs["commit"])
    results: Dict[str, Dict[str, Any]] = state.setdefault("commands", {})
    for cmd in commands:
        name = cmd["name"]
        if gr.now() >= state["deadline"]:
            return "lifetime"
        started = gr.now()
        argv = resolve_argv(cmd["argv"])
        _note(run_dir, stage="command_started", name=name, argv=argv)
        try:
            agent.exec(name, argv, POD_REPO, {**env, **cmd.get("env", {})})
        except urllib.error.HTTPError as exc:
            if exc.code == 410:  # past the agent's deadline
                return "lifetime"
            results[name] = {"returncode": None, "error": f"exec refused: HTTP {exc.code}"}
            _note(run_dir, stage="command_refused", name=name, code=exc.code)
            continue
        rc, stop = gr._wait_job(agent, rp, state, args, net, name, float(cmd["timeout_s"]))
        row: Dict[str, Any] = {"returncode": rc, "seconds": round(gr.now() - started, 1)}
        if stop == f"{name} timeout":  # this command only: kill it and go on
            row["timeout"] = True
            try:
                agent.post_kill(name)
            except Exception as exc:  # noqa: BLE001
                row["kill_error"] = str(exc)
            stop = None
        row["log_tail"] = _fetch_log(agent, run_dir, name)
        if stop:
            row["stopped"] = stop
        results[name] = row
        gr.save_state(run_dir, state)
        _note(
            run_dir,
            stage="command_done",
            name=name,
            **{k: v for k, v in row.items() if k != "log_tail"},
        )
        _pull(agent, run_dir, state)  # incremental: an early stop keeps the earlier results
        if stop:
            return stop
    _pull(agent, run_dir, state)  # final sweep (sha-checked; skips files already here)
    failed = [n for n, r in results.items() if r.get("returncode") != 0]
    return "complete" if not failed else "complete with failures: " + ",".join(failed)


# ----------------------------------------------------------------------------- the session
def summarize_results(out: Path) -> Dict[str, Any]:
    """Per pulled JSON: how many rows / cells and how many carry an ``error``."""
    rows: Dict[str, Any] = {}
    for path in sorted(out.rglob("*.json")) if out.exists() else []:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            rows[str(path.relative_to(out))] = {"unreadable": True}
            continue
        items = data.get("rows") or data.get("cells") or data.get("drift") or []
        rows[str(path.relative_to(out))] = {
            "items": len(items),
            "errors": sum(1 for r in items if isinstance(r, dict) and r.get("error")),
            "skipped_devices": data.get("skipped_devices"),
        }
    return rows


def write_summary(run_dir: Path, state, commands, skipped, args, outcome: str) -> Path:
    path = run_dir / "bench_session.json"
    results = state.get("commands", {})
    doc = {
        "schema": SESSION_SCHEMA,
        "note": "CUDA numbers from ONE RTX 4090 secure pod (one host); single runs, not a "
        "distribution: re-run before treating small differences as real",
        "commit": state.get("commit"),
        "job_id": args.job_id,
        "run_id": state.get("run_id"),
        "pod_name": state.get("name"),
        "pod_id": state.get("pod_id"),
        "gpu": state.get("gpu"),
        "host": state.get("host"),
        "image": args.image,
        "outcome": outcome,
        "pods_deleted_verified": state.get("pods_deleted_verified"),
        "cost_estimate_usd": state.get("cost_estimate"),
        "cost_bound": cost_bound(args),
        "commands": [
            {"name": c["name"], "argv": resolve_argv(c["argv"]), **results.get(c["name"], {})}
            for c in commands
        ],
        "skipped": skipped,
        "results": summarize_results(run_dir / "out"),
    }
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return path


def launch_attempt(args, rp, run_dir: Path, inputs, commands, skipped) -> int:
    from research.runpod_fanout.ledger import SharedLedger

    gr._install_signal_handlers()
    policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
    if float(args.max_hourly) > float(policy["max_hourly_per_pod_usd"]):
        raise SystemExit("refusing: --max-hourly is above the policy's max_hourly_per_pod_usd")
    if gr.our_pods(rp):
        raise SystemExit(f"refusing: a {gr.POD_NAME} pod already exists (run cleanup first)")
    balance0 = rp.balance()
    token = secrets.token_urlsafe(32)
    created = gr.now()
    until = created + float(args.lifetime_hours) * 3600.0
    deadline = until - DEADLINE_MARGIN_S
    name = gr.POD_NAME + time.strftime("%m%d%H%M%S")
    body = bench_pod_body(name, hashlib.sha256(token.encode()).hexdigest(), deadline, until, args)
    (run_dir / "pod_body.json").write_text(json.dumps(body))
    watchdog_pid = gr.spawn_watchdog(run_dir, until, name)
    ledger = SharedLedger(policy)
    run_id = f"{args.job_id}-{int(created)}"
    try:
        ledger.register_run(run_id, args.job_id, str(run_dir), args.spend_cap, os.getpid())
        refusal = ledger.reserve_pod(run_id, name, float(args.max_hourly), until, balance0)
    except BaseException:
        gr.stop_watchdog(watchdog_pid, run_dir)
        raise
    if refusal:
        ledger.finish_run(run_id, False, False)
        gr.stop_watchdog(watchdog_pid, run_dir)
        raise SystemExit(f"ledger refused: {refusal}")
    state: Dict[str, Any] = {
        "run_id": run_id,
        "job_id": args.job_id,
        "name": name,
        "balance0": balance0,
        "created": created,
        "until": until,
        "deadline": deadline,
        "commit": inputs["commit"],
        "spend_cap": args.spend_cap,
        "token_sha256": body["env"]["M3_TOKEN_SHA256"],
        "watchdog_pid": watchdog_pid,
        "commands_planned": [c["name"] for c in commands],
        "commands_skipped": skipped,
    }
    gr.save_state(run_dir, state)
    outcome, rate = "unknown", float(args.max_hourly)
    ok, no_capacity = False, False
    try:
        out, err, no_capacity = gr.create_pod_retrying(
            rp, body, run_dir, state, (gr.GPU,), float(args.max_hourly)
        )
        if err:
            outcome = f"create error: {err}"
        pod_id = out.get("id") if isinstance(out, dict) else None
        if not pod_id and not no_capacity:  # unknown outcome: adopt our pod by its name
            for _ in range(10):
                hits = [p for p in gr.our_pods(rp) if p.get("name") == name]
                if hits:
                    pod_id = hits[0]["id"]
                    break
                time.sleep(30)
        if not pod_id:  # every try was refused (no capacity / 4xx): no pod, no spend
            outcome = f"create refused: {err}" if no_capacity else outcome
            if outcome == "unknown":
                outcome = f"create failed: {out}"
        else:
            rate = float((out or {}).get("costPerHr") or args.max_hourly)
            state.update(pod_id=pod_id, rate=rate, machine=(out or {}).get("machine"))
            gr.save_state(run_dir, state)
            ledger.bind_pod(run_id, name, pod_id, rate)
            gr.say(json.dumps({"pod_id": pod_id, "gpu": state.get("gpu"), "rate": rate}))
            outcome = drive_bench(
                make_agent(pod_id, token), rp, run_dir, state, inputs, args, commands
            )
    except BaseException as exc:  # noqa: BLE001  (incl. KeyboardInterrupt / SIGTERM / SIGHUP)
        outcome = f"error: {type(exc).__name__}: {exc}"
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the delete
        state["outcome"] = outcome
        gr.save_state(run_dir, state)
        ok = gr.delete_and_verify(rp, run_dir, state, outcome)
        if ok and state.get("pod_id") and hasattr(rp, "get_pod"):
            try:  # belt and braces: the pod itself answers 404 now
                state["pod_gone_by_id"] = rp.get_pod(state["pod_id"]) is None
            except Exception as exc:  # noqa: BLE001
                state["pod_gone_by_id"] = f"unknown: {exc}"
        # a pod may have existed without our knowing its id (a create with an unknown outcome
        # whose pod showed up only for delete_and_verify's prefix sweep): charge the elapsed
        # time then too; only a session whose every create try was refused costs nothing
        charged = bool(state.get("pod_id")) or (
            bool(state.get("create_attempted_at")) and not no_capacity
        )
        try:
            cost = rate * (gr.now() - created) / 3600.0 if charged else 0.0
            state["cost_estimate"] = round(cost, 4)
            ledger.release_pod(run_id, name, cost)
            ledger.finish_run(run_id, outcome == "complete", outcome == "complete")
            state["balance_end"] = rp.balance()
        except Exception as exc:  # noqa: BLE001
            gr.alert(f"ledger/balance bookkeeping failed: {exc}")
        state["pods_deleted_verified"] = ok
        if ok:
            gr.stop_watchdog(watchdog_pid, run_dir)
            state["watchdog_stopped"] = True
        gr.save_state(run_dir, state)
        summary = write_summary(run_dir, state, commands, skipped, args, outcome)
        gr.say(
            json.dumps(
                {
                    "outcome": outcome,
                    "pods_deleted_verified": ok,
                    "results": str(run_dir / "out"),
                    "summary": str(summary),
                }
            )
        )
    return 0 if outcome == "complete" and ok else 1


def launch(args, commit: str, commands, skipped) -> int:
    from research.runpod_fanout.rp_client import RpClient
    from research.runpod_fanout.tls import preflight

    bind_prefix(args.pod_prefix)
    run_dir = Path(args.out_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing: {run_dir} exists and is not empty")
    problems = preflight()
    if problems:
        raise SystemExit(f"TLS preflight failed: {problems}")
    run_dir.mkdir(parents=True, exist_ok=True)
    keep_awake()
    inputs = build_bench_inputs(run_dir / "inputs", commit)
    (run_dir / "commands.json").write_text(
        json.dumps({"runnable": commands, "skipped": skipped}, indent=1)
    )
    return launch_attempt(args, RpClient(), run_dir, inputs, commands, skipped)


def dry_run(args, commit: str, commands, skipped) -> Dict[str, Any]:
    """Everything a launch would do up to the create, without spending or (by default) any
    RunPod call; inputs are built in a temporary directory that is removed again."""
    bind_prefix(args.pod_prefix)
    until = gr.now() + float(args.lifetime_hours) * 3600.0
    body = bench_pod_body(gr.POD_NAME + "dryrun", "0" * 64, until - DEADLINE_MARGIN_S, until, args)
    start = body["dockerStartCmd"]
    shown = {k: v for k, v in body.items() if k != "dockerStartCmd"}
    shown["dockerStartCmd"] = f"<bash -c: agent v2 gzip+base64, {len(start[-1])} chars>"
    with tempfile.TemporaryDirectory(prefix="podbench-dry-") as tmp:
        inputs = build_bench_inputs(Path(tmp), commit)
        uploads = [{"dst": f["dst"], "sha256": f["sha256"]} for f in inputs["files"]]
        total = inputs["bytes"]
        account = None
        if args.with_account:  # read-only: balance, ledger peek, rp.py's dry-run request
            from research.runpod_fanout.ledger import SharedLedger
            from research.runpod_fanout.rp_client import RpClient

            rp = RpClient()
            policy = json.loads((REPO / "research/runpod_fanout/fanout_policy.json").read_text())
            body_path = Path(tmp) / "pod_body.json"
            body_path.write_text(json.dumps(body))
            account = {
                "balance": rp.balance(),
                "ledger": SharedLedger(policy).peek(),
                "rp_dry_run": rp.create_pod(body_path, float(args.max_hourly), confirm=False),
            }
    timeouts = sum(float(c["timeout_s"]) for c in commands)
    budget = float(args.lifetime_hours) * 3600.0 - DEADLINE_MARGIN_S - SETUP_ESTIMATE_S
    out = {
        "dry_run": True,
        "commit": commit,
        "pod_prefix": gr.POD_NAME,
        "job_id": args.job_id,
        "gpu": gr.GPU,
        "no_g2_gate": "benchmark-only session: measures CUDA throughput the Mac cannot "
        "measure (no Mac baseline for the per-step 4x rule); bounded by lifetime x max_hourly",
        "pod_body": shown,
        "setup_argv": bench_setup_argv(),
        "env": bench_env(commit),
        "commands": [
            {
                "name": c["name"],
                "argv": resolve_argv(c["argv"]),
                "timeout_s": c["timeout_s"],
                "env": c.get("env", {}),
            }
            for c in commands
        ],
        "skipped": skipped,
        "uploads": uploads,
        "upload_bytes": total,
        "cost_bound": cost_bound(args),
        "time_budget": {
            "command_seconds_after_setup_estimate": budget,
            "sum_of_command_timeouts": timeouts,
            "note": "timeouts are upper bounds; a session that runs past the deadline stops "
            "there (results so far are pulled), so the list is in priority order",
        },
        "account": account,
    }
    return out


def check_args(args) -> Optional[str]:
    if not JOB_ID_RE.fullmatch(args.job_id):
        return "--job-id must look like 'perf-cuda-bench' or 'perf-cuda-bench-<suffix>'"
    if not PREFIX_RE.fullmatch(args.pod_prefix):
        return "--pod-prefix must look like 'perfbench-' (lowercase, ends with '-')"
    if not 0 < float(args.lifetime_hours) <= MAX_LIFETIME_H:
        return f"--lifetime-hours must be in (0, {MAX_LIFETIME_H}]"
    if not 0 < float(args.max_hourly) <= gr.MAX_HOURLY:
        return f"--max-hourly must be in (0, {gr.MAX_HOURLY}]"
    if float(args.lifetime_hours) * float(args.max_hourly) > float(args.spend_cap):
        return "refusing: lifetime x max_hourly exceeds the spend cap"
    note = args.accept_no_self_delete
    if note is not None and len(note.strip()) < 10:
        return "--accept-no-self-delete needs a real note (who decided, when)"
    return None


def bench_cmd(args, repo: Path = REPO, caps_loader: Callable = load_caps) -> int:
    problem = check_args(args)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.emit_commands:
        commit = _git(repo, "rev-parse", "HEAD").strip()
        doc = default_commands(caps_loader(repo, commit, [LEARNER, ACTOR]))
        args.emit_commands.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.emit_commands} (default commands for {commit[:12]})")
        return 0
    if not args.dry_run and not args.confirm:
        print(
            "bench spends money: --dry-run first, then --confirm after the user approves",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and not args.out_dir:
        print("--out-dir is required for a session", file=sys.stderr)
        return 2
    try:
        commit = clean_commit(repo)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.commands:
        parsed = parse_commands(json.loads(Path(args.commands).read_text()))
    else:
        parsed = parse_commands(default_commands(caps_loader(repo, commit, [LEARNER, ACTOR])))
    caps = caps_loader(repo, commit, [c["argv"][1] for c in parsed])
    commands, skipped = split_runnable(parsed, caps)
    for s in skipped:
        gr.say(f"skipping {s['name']}: {s['reason']}", err=True)
    if not commands:
        print("refusing: no runnable command (all skipped)", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(dry_run(args, commit, commands, skipped), indent=1, default=str))
        return 0
    return launch(args, commit, commands, skipped)


def cleanup_cmd(args) -> int:
    bind_prefix(args.pod_prefix)
    ns = argparse.Namespace(run_dir=str(args.out_dir), confirm=args.confirm)
    return gr.cleanup(ns)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="one CUDA bench session (dry run unless --confirm)")
    b.add_argument("--dry-run", action="store_true", help="print the plan; spend nothing")
    b.add_argument("--confirm", action="store_true", help="spend money (user-approved only)")
    b.add_argument("--out-dir", type=Path, default=None, help="local session dir (results: out/)")
    b.add_argument("--commands", type=Path, default=None, help=f"JSON ({COMMANDS_SCHEMA})")
    b.add_argument("--emit-commands", type=Path, default=None, help="write the defaults; exit")
    b.add_argument("--lifetime-hours", type=float, default=LIFETIME_H)
    b.add_argument("--max-hourly", type=float, default=MAX_HOURLY)
    b.add_argument("--spend-cap", type=float, default=SPEND_CAP)
    b.add_argument("--min-ram-gb", type=int, default=MIN_RAM_GB)
    b.add_argument("--min-vcpu", type=int, default=MIN_VCPU)
    b.add_argument("--image", default=gr.IMAGE, help="default: the training pods' image")
    b.add_argument("--pod-prefix", default=PREFIX)
    b.add_argument("--job-id", default=JOB_ID, help="ledger job id (a new suffix per re-run)")
    b.add_argument("--accept-no-self-delete", metavar="NOTE", default=None)
    b.add_argument(
        "--with-account",
        action="store_true",
        help="dry run only: also read the balance, the ledger and rp.py's dry-run request",
    )
    c = sub.add_parser("cleanup", help="delete any pod with the bench prefix")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--pod-prefix", default=PREFIX)
    c.add_argument("--confirm", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "cleanup":
        return cleanup_cmd(args)
    return bench_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
