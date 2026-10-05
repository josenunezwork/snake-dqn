"""Opt-in remote execution backend of the sequential strict runner (template v3).

Implements ``docs/research/governance_amendment_strict_on_runpod_2026-10-05.md``: a strict gate
may play its episodes on RunPod serverless CPU workers while the Mac orchestrator keeps look
boundaries, receipts, the ledger, create-only records and the independent audit. The four
conditions map to code as follows.

1. **Platform named in the pre-registration.** :func:`execution_block` freezes the platform,
   the remote configuration document (sha256), the sizing, the cost caps, the identity sample
   and the fallback rule into ``intent.json``; the protocol must carry the line
   ``Execution platform: runpod-serverless``.
2. **Per-world single worker.** A *unit* is every arm of one ``(phase, mix, world)``; a job
   carries whole units only and a unit is never split (:func:`pack_units`). A lost unit is
   re-dispatched whole; any episode that comes back twice must have identical deterministic
   bytes or the run stops.
3. **Pre-gate identity check.** :func:`run_identity_check` plays a pre-registered sample of the
   candidate's own gate worlds (both arms, full horizon) on the Mac and on RunPod and keeps
   salted digests only. ``run`` reads :func:`identity_state`: PASSED runs remote, anything else
   (FAILED, or a check that started but never completed) runs on the Mac (2 slots) under the
   same intent; a check that was never run refuses the run.
4. **Serving qualification stays on the Mac** (nothing here serves or qualifies).

Money: the endpoint's worst case is reserved in the shared RunPod ledger (and must fit the
pre-registered cap), and the account balance is read every ``balance_guard_seconds``: this
run's spend reaching its cap tears the endpoint down and stops the run ``INVALID_STOP``.
Speed: :func:`plan_remote` projects the wall-clock against 2 Mac slots (cold starts, seeding,
the identity check and look barriers included) and ``prepare`` refuses below 5x unless the
remote configuration forces it (recorded).

Nothing in this module runs unless an intent opts in with ``prepare --remote-config``.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from research.runpod_fanout import jobspec
from research.runpod_fanout import runner as rmod
from research.runpod_fanout import serverless
from research.runpod_fanout.ledger import SAFETY, SLACK_SECONDS, DuplicateRun, SharedLedger
from research.runpod_fanout.rp_client import RpClient, RunPodError
from research.sequential_strict_template import remote_worker
from research.sequential_strict_template import sequential_runner as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REMOTE_POLICY_PATH = HERE / "remote_policy.json"
STRICT_HANDLER = HERE / "strict_sls_handler.py"
WORKER_MODULE = "research.sequential_strict_template.remote_worker"
PLATFORM = "runpod-serverless"
REMOTE_EXECUTOR = "runpod-serverless"
REMOTE_BACKEND = "runpod-serverless"
FALLBACK_BACKEND = "local-mac-fallback"
POLICY_SCHEMA = "sequential-strict-remote-policy/v1"
CONFIG_SCHEMA = "sequential-strict-remote-config/v1"
EXECUTION_SCHEMA = "sequential-strict-execution/v1"
IDENTITY_SCHEMA = "sequential-strict-identity-check/v1"
SEGMENT_SCHEMA = "sequential-strict-remote-segment/v1"
RECEIPT_SCHEMA = "sequential-strict-remote-receipt/v1"
IDENTITY_DIR = "identity_check"
IDENTITY_FILE = "identity_check.json"
REMOTE_DIR = "remote"
ORCHESTRATOR_LOCK = "orchestrator.lock"
IDENTITY_STATES = ("NOT_RUN", "IN_PROGRESS", "PASSED", "FAILED", "ABANDONED")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RATIFIED_RE = re.compile(r"^\s*-\s*Decision:\s*ratified\b", re.IGNORECASE | re.MULTILINE)
TRANSPORT_PRODUCTION = "rp.py"

CONFIG_REQUIRED = {
    "schema",
    "platform",
    "budget_usd",
    "identity_budget_usd",
    "remote_wall_minutes",
    "identity_wall_minutes",
    "mac_episode_seconds",
    "mac_episode_seconds_source",
    "engine",
    "horizon",
    "horizon_path",
}
CONFIG_OPTIONAL: Dict[str, Any] = {
    "cloud_episode_seconds": None,
    "record_pins": {},
    "workers": None,
    "vcpu_per_worker": None,
    "flavors": None,
    "identity_worlds_per_mix": None,
    "force_below_5x": False,
    "sizing_objective": "fastest",
    "note": "",
}


class RemoteAbort(R.StrictRunError):
    """A remote failure after admission: the run stops (INVALID_STOP), never relabelled."""


class SpendStop(RemoteAbort):
    """This run's spend reached its pre-registered cap (balance-delta hard stop)."""


class GuardStop(RemoteAbort):
    """AC power, the lid or the watchdog failed while remote work ran."""


class SessionRefused(R.StrictRunError):
    """Nothing usable was created (preflight): the run or check does not start."""


# ---------------------------------------------------------------- policies and config


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_remote_policy(path: Path = REMOTE_POLICY_PATH) -> Dict[str, Any]:
    pol = _read(path)
    R.require(pol.get("schema") == POLICY_SCHEMA, "unknown remote policy schema")
    R.require(pol.get("platform") == PLATFORM, f"remote policy platform must be {PLATFORM}")
    R.require(int(pol["mac_slots"]) == R.WORKERS, "the Mac baseline is the strict gate's 2 slots")
    R.require(float(pol["min_speedup"]) >= 5.0, "the per-step rule is >= 5x")
    return pol


def policies() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(remote policy, fan-out policy, serverless policy)."""
    return load_remote_policy(), jobspec.load_policy(), serverless.load_sls_policy()


def _positive(value: Any) -> bool:
    return R._real(value) and value > 0


def load_remote_config(
    path: Path, rpol: Mapping[str, Any], sp: Mapping[str, Any]
) -> Dict[str, Any]:
    """The study's remote configuration (a hash-bound pre-registration document)."""
    raw = _read(path)
    R.require(isinstance(raw, dict), "remote config must be a JSON object")
    unknown = set(raw) - CONFIG_REQUIRED - set(CONFIG_OPTIONAL)
    missing = CONFIG_REQUIRED - set(raw)
    R.require(
        not unknown and not missing, f"remote config keys: missing {missing}, unknown {unknown}"
    )
    cfg = {**CONFIG_OPTIONAL, **raw}
    R.require(cfg["schema"] == CONFIG_SCHEMA, f"remote config schema must be {CONFIG_SCHEMA}")
    R.require(cfg["platform"] == PLATFORM, f"remote config platform must be {PLATFORM}")
    cap = float(rpol["max_budget_usd"])
    R.require(
        _positive(cfg["budget_usd"]) and cfg["budget_usd"] <= cap,
        f"budget_usd must be in (0, {cap}]",
    )
    R.require(
        _positive(cfg["identity_budget_usd"]) and cfg["identity_budget_usd"] <= cfg["budget_usd"],
        "identity_budget_usd must be in (0, budget_usd]",
    )
    longest = float(rpol["max_remote_wall_minutes"])
    for key in ("remote_wall_minutes", "identity_wall_minutes"):
        R.require(
            _positive(cfg[key]) and 10 <= cfg[key] <= longest, f"{key} must be in [10, {longest}]"
        )
    R.require(_positive(cfg["mac_episode_seconds"]), "mac_episode_seconds must be > 0")
    R.require(
        isinstance(cfg["mac_episode_seconds_source"], str)
        and cfg["mac_episode_seconds_source"].strip(),
        "mac_episode_seconds_source must say where the Mac timing was measured",
    )
    R.require(cfg["engine"] in ("live", "simd"), "engine must be live or simd")
    R.require(
        isinstance(cfg["horizon"], int)
        and not isinstance(cfg["horizon"], bool)
        and 1 <= cfg["horizon"] <= jobspec.MAX_HORIZON,
        f"horizon must be an int in [1, {jobspec.MAX_HORIZON}]",
    )
    R.require(
        isinstance(cfg["horizon_path"], str) and cfg["horizon_path"].strip(),
        "horizon_path (dot path of the horizon inside a record) is required",
    )
    R.require(
        cfg["cloud_episode_seconds"] is None or _positive(cfg["cloud_episode_seconds"]),
        "cloud_episode_seconds must be > 0 or null",
    )
    R.require(isinstance(cfg["record_pins"], dict), "record_pins must be an object")
    quota = int(sp.get("account_worker_quota") or sp["max_workers"])
    R.require(
        cfg["workers"] is None
        or (isinstance(cfg["workers"], int) and 1 <= cfg["workers"] <= quota),
        f"workers must be null or an int in [1, {quota}]",
    )
    R.require(
        cfg["vcpu_per_worker"] is None or cfg["vcpu_per_worker"] in sp["vcpu_sizes"],
        f"vcpu_per_worker must be null or one of {sp['vcpu_sizes']}",
    )
    flavors = cfg["flavors"] if cfg["flavors"] is not None else list(sp["flavors_pref"])
    R.require(
        isinstance(flavors, list) and flavors and set(flavors) <= set(sp["flavors_pref"]),
        f"flavors must be a non-empty subset of {sp['flavors_pref']}",
    )
    cfg["flavors"] = list(flavors)
    k = cfg["identity_worlds_per_mix"]
    k = int(rpol["identity_worlds_per_mix_default"]) if k is None else k
    R.require(
        isinstance(k, int) and 1 <= k <= int(rpol["identity_worlds_per_mix_max"]),
        "identity_worlds_per_mix out of range",
    )
    cfg["identity_worlds_per_mix"] = k
    R.require(isinstance(cfg["force_below_5x"], bool), "force_below_5x must be a boolean")
    R.require(
        cfg["sizing_objective"] in ("fastest", "cheapest"),
        "sizing_objective must be fastest (default) or cheapest (cheapest at >= 5x)",
    )
    return cfg


def cloud_episode_seconds(
    cfg: Mapping[str, Any], fp: Mapping[str, Any], sp: Mapping[str, Any]
) -> float:
    """Per-episode wall on a serverless worker: the config's measured value, else the
    pod-calibrated estimate x the serverless ``episode_time_factor``."""
    if cfg.get("cloud_episode_seconds") is not None:
        return float(cfg["cloud_episode_seconds"])
    base = rmod.episode_seconds(fp, {"engine": cfg["engine"], "horizon": cfg["horizon"]})
    return base * float(sp.get("episode_time_factor") or 1.0)


def amendment_status(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else ""
    return {
        "path": str(path),
        "sha256": R.sha256_file(path) if Path(path).is_file() else None,
        "ratified": bool(RATIFIED_RE.search(text)),
    }


def protocol_names_platform(path: Path, line: str) -> bool:
    text = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else ""
    return any(row.strip() == line for row in text.splitlines())


def is_remote(intent: Mapping[str, Any]) -> bool:
    return intent.get("template_version") == R.TEMPLATE_VERSION_REMOTE


# ---------------------------------------------------------------- seeding (local registry only)


def seed_status(
    fp: Mapping[str, Any],
    sp: Mapping[str, Any],
    commit: str,
    checkpoints: Sequence[str],
    repo: Path = REPO,
    registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """What the private volume still needs for this commit (reads the LOCAL registry and the
    commit's git archive; no RunPod call)."""
    rid = serverless.runtime_id(fp, sp, STRICT_HANDLER)
    reg = registry if registry is not None else serverless.Registry(fp).read()
    work = Path(tempfile.mkdtemp(prefix="sst-seed-"))
    try:
        repo_sha = jobspec.git_archive(Path(repo), commit, work / "repo.tar.gz")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    problems = serverless.seed_problems(reg, commit, repo_sha, list(checkpoints), rid)
    runtime = (reg.get("runtimes") or {}).get(rid) or {}
    if not reg.get("volume") or not runtime.get("ready"):
        needs = "runtime"
    elif problems:
        needs = "commit"
    else:
        needs = None
    return {
        "runtime_id": rid,
        "repo_sha256": repo_sha,
        "template_id": runtime.get("template_id"),
        "ready": not problems,
        "needs": needs,
        "problems": problems,
    }


# ---------------------------------------------------------------- plan (speed-up vs the Mac)


def segment_shapes(mixes: Sequence[str], n_calibration: int, look_sizes: Sequence[int]) -> List:
    """``(name, units, episodes per unit)`` of the calibration segment and every final look."""
    m = len(mixes)
    out = [("calibration", int(n_calibration) * m, 1)]
    prev = 0
    for k, size in enumerate(look_sizes):
        out.append((f"final look {k}", (int(size) - prev) * m, len(R.ARMS)))
        prev = int(size)
    return out


def units_per_job(slots: int, episodes_per_unit: int) -> int:
    """One-wave jobs: as many whole units as fit the worker's slots (at least one unit)."""
    return max(1, int(slots) // max(1, int(episodes_per_unit)))


def segment_seconds(
    units: int, episodes_per_unit: int, slots: int, workers: int, episode_s: float, overhead: float
) -> float:
    per = units_per_job(slots, episodes_per_unit)
    jobs = math.ceil(int(units) / per)
    job_s = math.ceil(per * episodes_per_unit / max(1, int(slots))) * episode_s
    return math.ceil(jobs / max(1, int(workers))) * job_s + overhead


def plan_remote(
    *,
    mixes: Sequence[str],
    n_calibration: int,
    look_sizes: Sequence[int],
    cfg: Mapping[str, Any],
    rpol: Mapping[str, Any],
    fp: Mapping[str, Any],
    sp: Mapping[str, Any],
    seeding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Projected wall-clock of the whole gate on RunPod vs 2 Mac slots, and the sizing.

    Both sides plan the worst case (every look played). The remote side counts seeding (from
    the local registry), the identity check (its Mac half and its remote half, run one after
    the other), one cold start per endpoint, one-wave batches of whole units per worker and a
    barrier per segment. Every option must fit the gate and identity caps (worst-case ledger
    reservations), the hourly cap, the worker quota and ``max_wall_fill`` of each wall cap.
    ``sizing_objective`` "fastest" (default) picks the fastest such option (the budget caps
    the spend; ties go to the cheaper); "cheapest" picks the cheapest at >= ``min_speedup``.
    """
    mac_ep = float(cfg["mac_episode_seconds"])
    cloud_ep = cloud_episode_seconds(cfg, fp, sp)
    mac_slots = int(rpol["mac_slots"])
    shapes = segment_shapes(mixes, n_calibration, look_sizes)
    episodes = [units * epu for _name, units, epu in shapes]
    mac_by_stop = [
        sum(episodes[: k + 2]) * mac_ep / mac_slots for k in range(len(look_sizes))
    ]  # calibration + final looks 0..k
    k_id = int(cfg["identity_worlds_per_mix"])
    id_units = k_id * len(mixes)
    id_mac = math.ceil(id_units * len(R.ARMS) / mac_slots) * mac_ep
    cold = float(sp["cold_start_seconds_est"])
    overhead = float(rpol["barrier_overhead_seconds"])
    seed_s = 0.0
    needs = "runtime" if seeding is None else seeding.get("needs")
    if needs == "runtime":
        seed_s = 60.0 * float(rpol["seed_minutes_runtime"])
    elif needs in ("commit", "checkpoints"):
        seed_s = 60.0 * float(rpol["seed_minutes_commit"])
    threads = int(fp["threads_per_episode"])
    margin = float(sp["price_margin"])
    quota = int(sp.get("account_worker_quota") or sp["max_workers"])
    fill = float(rpol["max_wall_fill"])
    gate_wall_s = 60.0 * float(cfg["remote_wall_minutes"])
    id_wall_s = 60.0 * float(cfg["identity_wall_minutes"])
    grace = float(fp["watchdog_grace_seconds"]) + 300.0
    flavors = list(cfg["flavors"])
    vcpus = [cfg["vcpu_per_worker"]] if cfg["vcpu_per_worker"] else list(sp["vcpu_sizes"])
    workers_opts = [cfg["workers"]] if cfg["workers"] else list(range(1, quota + 1))
    rows = []
    for vcpu in vcpus:
        slots = max(1, min(serverless.MAX_JOB_SLOTS, int(vcpu) // threads))
        per_worker = serverless.worker_rate(sp, flavors, int(vcpu))
        for w in workers_opts:
            segs = [
                segment_seconds(units, epu, slots, w, cloud_ep, overhead)
                for _name, units, epu in shapes
            ]
            id_jobs = math.ceil(id_units / units_per_job(slots, len(R.ARMS)))
            id_workers = max(1, min(w, id_jobs))
            id_remote = cold + segment_seconds(
                id_units, len(R.ARMS), slots, id_workers, cloud_ep, overhead
            )
            gate_remote = cold + sum(segs)
            fixed = seed_s + id_mac + id_remote
            remote_by_stop = [fixed + cold + sum(segs[: k + 2]) for k in range(len(look_sizes))]
            total = remote_by_stop[-1]
            rate = w * per_worker
            id_rate = id_workers * per_worker
            worst_gate = SAFETY * rate * margin * (gate_wall_s + grace + SLACK_SECONDS) / 3600
            worst_id = SAFETY * id_rate * margin * (id_wall_s + grace + SLACK_SECONDS) / 3600
            why = []
            if worst_gate > float(cfg["budget_usd"]) + 1e-9:
                why.append("gate budget (worst-case reservation)")
            if worst_id > float(cfg["identity_budget_usd"]) + 1e-9:
                why.append("identity budget (worst-case reservation)")
            if rate * margin > float(sp["max_endpoint_hourly_usd"]):
                why.append("hourly cap")
            if w > quota:
                why.append("account worker quota")
            if gate_remote > fill * gate_wall_s:
                why.append("remote wall cap")
            if id_mac + id_remote > fill * id_wall_s:  # the endpoint idles during the Mac half
                why.append("identity wall cap")
            rows.append(
                {
                    "workers": w,
                    "vcpu_per_worker": int(vcpu),
                    "slots_per_worker": slots,
                    "units_per_job_final": units_per_job(slots, len(R.ARMS)),
                    "identity_workers": id_workers,
                    "remote_gate_minutes": round(gate_remote / 60, 1),
                    "remote_total_minutes": round(total / 60, 1),
                    "speedup": round(mac_by_stop[-1] / total, 2),
                    "speedup_by_stop_look": [
                        round(m / r, 2) for m, r in zip(mac_by_stop, remote_by_stop)
                    ],
                    "expected_usd": round((rate * gate_remote + id_rate * id_remote) / 3600, 2),
                    "worst_gate_usd": round(worst_gate, 2),
                    "worst_identity_usd": round(worst_id, 2),
                    "reserved_usd_per_hr": round(rate * margin, 3),
                    "refused": why,
                }
            )
    target = float(rpol["min_speedup"])
    ok = [r for r in rows if not r["refused"]]
    fast = [r for r in ok if r["speedup"] >= target]
    objective = cfg.get("sizing_objective") or "fastest"
    if fast and objective == "cheapest":
        choice = min(fast, key=lambda r: (r["expected_usd"], r["workers"], -r["vcpu_per_worker"]))
    else:  # fastest within every cap (the budget bounds the spend); ties go to the cheaper
        choice = max(ok, key=lambda r: (r["speedup"], -r["expected_usd"]), default=None)
    meets = bool(choice and choice["speedup"] >= target)
    return {
        "basis": "worst case: calibration + every final look, both sides",
        "mac_slots": mac_slots,
        "mac_episode_seconds": mac_ep,
        "mac_episode_seconds_source": cfg["mac_episode_seconds_source"],
        "cloud_episode_seconds": round(cloud_ep, 1),
        "segments": [{"name": n, "units": u, "episodes_per_unit": e} for n, u, e in shapes],
        "episodes_total": sum(episodes),
        "mac_minutes": round(mac_by_stop[-1] / 60, 1),
        "mac_minutes_by_stop_look": [round(m / 60, 1) for m in mac_by_stop],
        "identity_check": {
            "units": id_units,
            "episodes": id_units * len(R.ARMS),
            "mac_minutes": round(id_mac / 60, 1),
        },
        "seeding": {"needs": needs, "minutes": round(seed_s / 60, 1)},
        "cold_start_seconds": cold,
        "barrier_overhead_seconds": overhead,
        "min_speedup": target,
        "sizing_objective": objective,
        "choice": choice,
        "meets_min_speedup": meets,
        "forced_below_min": bool(cfg.get("force_below_5x")) and not meets,
        "options_ok": len(ok),
        "options_refused": len(rows) - len(ok),
        "fastest_ok": max(ok, key=lambda r: r["speedup"], default=None),
    }


# ---------------------------------------------------------------- intent execution block


def identity_sample(intent: Mapping[str, Any], worlds_per_mix: int) -> List[Dict[str, Any]]:
    """Pre-registered identity sample: ``worlds_per_mix`` worlds of the first look's prefix,
    ranked by ``sha256("<final namespace>|identity-sample|<i>")``, every mix, world-major."""
    namespace = intent["spec"]["namespaces"]["final"]
    first = int(intent["plan"]["look_sizes"][0])
    ranked = sorted(
        range(first),
        key=lambda i: hashlib.sha256(f"{namespace}|identity-sample|{i}".encode()).hexdigest(),
    )[: int(worlds_per_mix)]
    bank = intent["banks"]["final"]
    return [
        {"mix": mix, "world_index": i, "world_seed": int(bank[i])}
        for i in sorted(ranked)
        for mix in intent["spec"]["mixes"]
    ]


def _doc(path: Path) -> Dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": R.sha256_file(path)}


def execution_block(
    spec: R.StudySpec,
    intent: Mapping[str, Any],
    config_path: Path,
    *,
    repo: Path = REPO,
    seeding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything condition 1 freezes before anything is played (``intent["execution"]``)."""
    rpol, fp, sp = policies()
    config_path = Path(config_path).resolve()
    R.require(config_path.is_file(), f"remote config {config_path} does not exist")
    cfg = load_remote_config(config_path, rpol, sp)
    R.require(
        callable(spec.remote_worker_setup) and callable(spec.remote_checkpoints),
        "a remote (template v3) study spec needs remote_worker_setup and remote_checkpoints",
    )
    protocol = Path(intent["preregistration"]["protocol"]["path"])
    R.require(
        protocol_names_platform(protocol, rpol["protocol_platform_line"]),
        f"the protocol must name the platform in a line {rpol['protocol_platform_line']!r}",
    )
    amendment = amendment_status(Path(repo) / rpol["amendment_path"])
    R.require(amendment["sha256"] is not None, "the strict-on-RunPod amendment is missing")
    R.require(
        intent["dry_run"] or amendment["ratified"],
        "the strict-on-RunPod governance amendment is not ratified: production remote "
        "intents are refused until its ratification section records the decision",
    )
    checkpoints = sorted({str(s) for s in spec.remote_checkpoints()})
    allow = jobspec.load_allowlist()
    R.require(checkpoints and all(SHA_RE.match(s) for s in checkpoints), "remote checkpoints")
    missing = [s[:12] for s in checkpoints if s not in allow]
    R.require(
        not missing,
        f"checkpoints not on the RunPod upload allow-list (owner approval needed): {missing}",
    )
    closure = R.source_closure(rpol["closure_roots"], repo)
    R.require(
        intent["allow_dirty"] or not closure["dirty"],
        f"remote closure is dirty: {closure['dirty']}",
    )
    commit = intent["source_closure"]["git_commit"]
    if seeding is None:
        seeding = seed_status(fp, sp, commit, checkpoints, repo)
    plan = plan_remote(
        mixes=intent["spec"]["mixes"],
        n_calibration=len(intent["banks"]["calibration"]),
        look_sizes=intent["plan"]["look_sizes"],
        cfg=cfg,
        rpol=rpol,
        fp=fp,
        sp=sp,
        seeding=seeding,
    )
    R.require(plan["choice"] is not None, "no remote sizing fits the caps (see remote-plan)")
    R.require(
        plan["meets_min_speedup"] or cfg["force_below_5x"],
        f"projected speed-up {plan['choice']['speedup']}x < {plan['min_speedup']}x vs 2 Mac "
        "slots: the per-step 5x rule keeps this gate on the Mac (force_below_5x overrides, "
        "recorded)",
    )
    choice = plan["choice"]
    return {
        "schema": EXECUTION_SCHEMA,
        "backend": REMOTE_BACKEND,
        "platform": PLATFORM,
        "band_template_version": spec.template_version,
        "amendment": amendment,
        "remote_config": {**_doc(config_path), "values": cfg},
        "policies": {
            "remote": _doc(REMOTE_POLICY_PATH),
            "fanout": _doc(jobspec.POLICY_PATH),
            "serverless": _doc(serverless.SLS_POLICY_PATH),
            "allowlist": _doc(jobspec.ALLOWLIST_PATH),
        },
        "handler": {
            **_doc(STRICT_HANDLER),
            "runtime_id": serverless.runtime_id(fp, sp, STRICT_HANDLER),
            "worker_module": WORKER_MODULE,
        },
        "repo_commit": commit,
        "closure": closure,
        "checkpoints": checkpoints,
        "seeding_at_prepare": dict(seeding),
        "unit_rule": "one unit = every arm episode of one (phase, mix, world); dispatched whole "
        "to one worker in one job and never split; a lost unit is re-dispatched whole",
        "duplicate_rule": "an episode returned twice must have identical deterministic bytes "
        "(jobspec.deterministic_bytes: wall-clock, platform and dispatch fields dropped) or "
        "the run stops INVALID_STOP",
        "sizing": {
            "workers": choice["workers"],
            "vcpu_per_worker": choice["vcpu_per_worker"],
            "slots_per_worker": choice["slots_per_worker"],
            "identity_workers": choice["identity_workers"],
            "flavors": cfg["flavors"],
            "batching": "one-wave jobs of whole units (units per job = slots // arms)",
        },
        "plan": plan,
        "speedup": choice["speedup"],
        "min_speedup": plan["min_speedup"],
        "forced_below_min": plan["forced_below_min"],
        "cost_cap": {
            "gate_usd": float(cfg["budget_usd"]),
            "identity_usd": float(cfg["identity_budget_usd"]),
            "rule": "balance-delta hard stop: this run's spend (balance drop since its start, "
            "less other runners' ledger reservations over that time) reaching the cap tears "
            "the endpoint down and stops the run INVALID_STOP (never relabelled)",
        },
        "remote_wall_seconds": 60.0 * float(cfg["remote_wall_minutes"]),
        "identity_wall_seconds": 60.0 * float(cfg["identity_wall_minutes"]),
        "max_attempts_per_unit": int(rpol["max_attempts_per_unit"]),
        "record_checks": {
            "horizon": cfg["horizon"],
            "horizon_path": cfg["horizon_path"],
            "record_pins": dict(cfg["record_pins"]),
            "bindings": "intent sha256, spec descriptor, arm identity, roster row sha256 and "
            "source-closure digest recomputed on the worker; one platform stamp per unit",
        },
        "identity_check": {
            "sample": identity_sample(intent, cfg["identity_worlds_per_mix"]),
            "worlds_per_mix": cfg["identity_worlds_per_mix"],
            "arms": list(R.ARMS),
            "phase": "final (the candidate's own gate worlds, first look's prefix)",
            "comparison": "per episode: HMAC-SHA256 under a discarded random key of "
            "jobspec.deterministic_bytes({'record': record}); every sample episode must be "
            "present and equal on both sides",
            "storage": "salted digests only: no record and no metric is kept or shown",
            "runs": "once per intent, after prepare and before run; never re-run",
        },
        "fallback": {
            "backend": FALLBACK_BACKEND,
            "executor": R.PRODUCTION_EXECUTOR,
            "slots": R.WORKERS,
            "rule": "identity check FAILED (any mismatch or missing episode) or ABANDONED "
            "(started, never completed): the whole gate runs on the Mac (2 slots) under this "
            "intent; NOT_RUN: run refuses; decided before any gate world is played",
        },
        "orchestrator": "the Mac: look boundaries, receipts, ledger, create-only records, the "
        "independent audit; AC power and lid guards; holds an orchestrator lock, no CPU slot",
        "serving_qualification": "stays on the Mac (condition 4)",
    }


def validate_execution(intent: Mapping[str, Any], spec: R.StudySpec) -> None:
    """Drift and shape checks of a v3 intent's execution block (before any child starts)."""
    block = intent["execution"]
    R.require(
        block.get("schema") == EXECUTION_SCHEMA
        and block.get("backend") == REMOTE_BACKEND
        and block.get("platform") == PLATFORM,
        "execution block schema/backend/platform",
    )
    R.require(
        block["band_template_version"] == spec.template_version,
        "execution band template version differs from the spec's",
    )
    rpol, fp, sp = policies()
    for name, doc in [
        ("remote config", block["remote_config"]),
        ("strict handler", block["handler"]),
        ("amendment", block["amendment"]),
        *[(f"{k} policy", v) for k, v in block["policies"].items()],
    ]:
        R.require(
            Path(doc["path"]).is_file() and R.sha256_file(Path(doc["path"])) == doc["sha256"],
            f"{name} drift",
        )
    cfg = load_remote_config(Path(block["remote_config"]["path"]), rpol, sp)
    R.require(cfg == block["remote_config"]["values"], "remote config values drift")
    R.require(
        block["handler"]["runtime_id"] == serverless.runtime_id(fp, sp, STRICT_HANDLER),
        "strict runtime id drift",
    )
    R.require(block["repo_commit"] == intent["source_closure"]["git_commit"], "remote commit")
    drift = R.closure_drift(block["closure"])
    R.require(not drift, f"remote closure drift: {drift}")
    R.require(
        callable(spec.remote_worker_setup) and callable(spec.remote_checkpoints),
        "the spec lost its remote hooks",
    )
    R.require(
        sorted({str(s) for s in spec.remote_checkpoints()}) == block["checkpoints"],
        "remote checkpoints drift",
    )
    sample = identity_sample(intent, block["identity_check"]["worlds_per_mix"])
    R.require(sample == block["identity_check"]["sample"], "identity sample drift")
    R.require(
        intent["dry_run"] or block["amendment"]["ratified"] is True,
        "production remote intent without a ratified amendment",
    )
    R.require(
        protocol_names_platform(
            Path(intent["preregistration"]["protocol"]["path"]), rpol["protocol_platform_line"]
        ),
        "the protocol no longer names the platform",
    )
    R.require(
        block["plan"]["meets_min_speedup"] or block["remote_config"]["values"]["force_below_5x"],
        "plan below the 5x rule without a recorded force",
    )


def execution_drift(intent: Mapping[str, Any]) -> List[str]:
    """Files of the execution block whose bytes changed (checked before every segment)."""
    block = intent["execution"]
    docs = [block["remote_config"], block["handler"], block["amendment"]]
    docs += list(block["policies"].values())
    out = [
        doc["path"]
        for doc in docs
        if not Path(doc["path"]).is_file() or R.sha256_file(Path(doc["path"])) != doc["sha256"]
    ]
    return out + R.closure_drift(block["closure"])


# ---------------------------------------------------------------- identity state and binding


def _lock_held_elsewhere(path: Path) -> bool:
    if not Path(path).is_file():
        return False
    with Path(path).open("r") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def identity_state(root: Path) -> Dict[str, Any]:
    """The identity check's state for a run root (see the module docstring)."""
    root = Path(root)
    result = root / IDENTITY_FILE
    idir = root / IDENTITY_DIR
    if result.is_file():
        data = R.read_json(result)
        passes = data.get("passes") is True
        return {
            "state": "PASSED" if passes else "FAILED",
            "file": IDENTITY_FILE,
            "sha256": R.sha256_file(result),
            "reason": (
                "identity check passed"
                if passes
                else "identity check failed: " + "; ".join(data.get("reasons") or ["mismatch"])
            ),
        }
    if not idir.exists():
        return {"state": "NOT_RUN", "file": None, "sha256": None, "reason": "never run"}
    if _lock_held_elsewhere(idir / "running.lock"):
        return {"state": "IN_PROGRESS", "file": None, "sha256": None, "reason": "running"}
    started = idir / "started.json"
    return {
        "state": "ABANDONED",
        "file": f"{IDENTITY_DIR}/started.json" if started.is_file() else None,
        "sha256": R.sha256_file(started) if started.is_file() else None,
        "reason": "identity check started but never completed: treated as failed",
    }


def identity_gate(root: Path) -> Dict[str, Any]:
    """The identity binding every v3 segment marker carries."""
    state = identity_state(root)
    return {k: state[k] for k in ("state", "file", "sha256")}


# ---------------------------------------------------------------- digests (identity check)


def record_bytes(record: Mapping[str, Any]) -> bytes:
    """Deterministic bytes of one episode record (keys containing ``seconds`` dropped)."""
    return jobspec.deterministic_bytes({"record": R.json_safe(dict(record))})


def salted_digests(key: bytes, record: Mapping[str, Any]) -> Dict[str, Any]:
    """HMAC-SHA256 of the whole record and of each top-level field, under ``key`` (random and
    discarded after the check, so the stored digests reveal nothing about the values)."""
    safe = R.json_safe(dict(record))
    return {
        "digest": hmac.new(key, record_bytes(safe), hashlib.sha256).hexdigest(),
        "fields": {
            str(k): hmac.new(key, jobspec.deterministic_bytes({"v": v}), hashlib.sha256).hexdigest()
            for k, v in sorted(safe.items())
            if "seconds" not in str(k)
        },
    }


def judge_identity(
    expected: Sequence[str], mac: Mapping[str, Any], remote: Mapping[str, Any]
) -> Dict[str, Any]:
    """Every expected episode present on both sides with equal digests."""
    mac_d, rem_d = mac.get("digests") or {}, remote.get("digests") or {}
    missing_mac = sorted(set(expected) - set(mac_d))
    missing_remote = sorted(set(expected) - set(rem_d))
    mismatched = sorted(
        e
        for e in expected
        if e in mac_d and e in rem_d and mac_d[e]["digest"] != rem_d[e]["digest"]
    )
    fields = {
        e: sorted(
            k
            for k in set(mac_d[e]["fields"]) | set(rem_d[e]["fields"])
            if mac_d[e]["fields"].get(k) != rem_d[e]["fields"].get(k)
        )
        for e in mismatched
    }
    reasons = []
    if not mac.get("complete"):
        reasons.append(f"Mac side incomplete: {mac.get('error')}")
    if not remote.get("complete"):
        reasons.append(f"RunPod side incomplete: {remote.get('error')}")
    if missing_mac or missing_remote:
        reasons.append(f"missing episodes (Mac {len(missing_mac)}, RunPod {len(missing_remote)})")
    if mismatched:
        reasons.append(f"{len(mismatched)} episode(s) differ")
    return {
        "expected": len(expected),
        "identical": len([e for e in expected if e in mac_d and e in rem_d]) - len(mismatched),
        "mismatched": mismatched,
        "mismatched_fields": fields,
        "missing_mac": missing_mac,
        "missing_remote": missing_remote,
        "reasons": reasons,
        "passes": not reasons,
    }


# ---------------------------------------------------------------- units and record checks


def get_path(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def plan_units(
    intent: Mapping[str, Any],
    phase: str,
    units: Sequence[Mapping[str, Any]],
    look: int,
    rows: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Dispatchable units: key, the unit's episodes (``unit_episodes``) and its roster row."""
    out = []
    for unit in units:
        row = rows[(unit["mix"], int(unit["world_index"]))]
        R.require(int(row["world_seed"]) == int(unit["world_seed"]), "roster seed mismatch")
        out.append(
            {
                "unit": f"{phase}|{unit['mix']}|{int(unit['world_index'])}",
                "index": int(unit["unit"]),
                "shard": int(unit["worker"]),
                "row": dict(row),
                "episodes": R.unit_episodes(phase, unit, look),
            }
        )
    return out


def check_result(
    intent: Mapping[str, Any],
    intent_sha: str,
    unit: Mapping[str, Any],
    episode: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """One returned episode against the frozen intent; returns the parsed worker output."""
    eid = episode["episode_id"]
    data = str(result.get("output") or "").encode("utf-8")
    R.require(hashlib.sha256(data).hexdigest() == result.get("sha256"), f"{eid}: output bytes")
    out = json.loads(data)
    R.require(
        out.get("schema") == remote_worker.OUTPUT_SCHEMA and out.get("episode_id") == eid,
        f"{eid}: not this episode's output",
    )
    checks = out.get("checks") or {}
    expected = {
        "intent_sha256": intent_sha,
        "spec_descriptor_sha256": R.canonical_sha(intent["spec"]),
        "arm_identity_sha256": R.canonical_sha(intent["arms"][episode["arm"]]),
        "row_sha256": R.canonical_sha(dict(unit["row"])),
        "closure_digest": intent["source_closure"]["digest"],
    }
    bad = sorted(k for k, v in expected.items() if checks.get(k) != v)
    R.require(not bad, f"{eid}: worker bindings differ: {bad}")
    record = out.get("record")
    R.require(isinstance(record, Mapping), f"{eid}: no record")
    rc = intent["execution"]["record_checks"]
    R.require(
        get_path(record, rc["horizon_path"]) == rc["horizon"],
        f"{eid}: horizon at {rc['horizon_path']} is not {rc['horizon']}",
    )
    for path, value in rc["record_pins"].items():
        R.require(get_path(record, path) == value, f"{eid}: pinned {path} differs")
    stamp = out.get("platform") or {}
    R.require(bool(stamp.get("platform_id")), f"{eid}: no platform stamp")
    return out


# ---------------------------------------------------------------- remote session


@dataclass
class _Job:
    id: str
    units: List[str]
    submitted: float
    timeout_s: float
    status: str = "IN_QUEUE"
    started: Optional[float] = None
    errors: int = 0


@dataclass
class _Target:
    """What ``runpod_fanout.runner.spawn_watchdog_process`` needs from a run."""

    job: Dict[str, Any]
    deadline: float
    policy: Mapping[str, Any]
    run_dir: Path
    sp: Mapping[str, Any]

    def run_prefix(self) -> str:
        return f"{self.policy['pod_name_prefix']}{self.job['job_id']}--{self.run_dir.name}-"

    def endpoint_prefix(self) -> str:
        return f"{self.sp['endpoint_prefix']}{self.job['job_id']}--{self.run_dir.name}-"


def pack_units(
    pending: Sequence[Tuple[str, int]], slots: int, solo: Sequence[str] = ()
) -> List[str]:
    """Next job: whole units in order, one wave (episodes <= slots; the first unit always).
    A ``solo`` unit (it was in a lost batch) runs alone."""
    if not pending:
        return []
    first, n = pending[0]
    batch = [first]
    if first in solo:
        return batch
    for unit, size in pending[1:]:
        if unit in solo or n + size > slots:
            continue
        batch.append(unit)
        n += size
    return batch


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RemoteSession:
    """One endpoint's life for one label (the gate, or the identity check).

    ``open`` (refuses cleanly, nothing left behind): TLS preflight, seeding of the frozen
    commit / checkpoints / strict runtime on the private volume, the account worker quota, the
    balance floor, the shared ledger (run + worst-case reservation, which must fit the cap),
    the watchdog, the endpoint and a probe job that proves the runtime. ``run_units``
    dispatches whole units, re-dispatches lost ones and enforces the guards. ``close`` tears the
    endpoint down, releases the ledger and writes ``receipt.json``.
    """

    def __init__(
        self,
        *,
        label: str,
        intent: Mapping[str, Any],
        intent_text: str,
        intent_sha: str,
        run_dir: Path,
        budget_usd: float,
        wall_seconds: float,
        workers: int,
        rp: Any = None,
        policies_: Optional[Tuple[Mapping, Mapping, Mapping]] = None,
        ledger: Optional[SharedLedger] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        power_check: Optional[Callable[[], bool]] = None,
        lid_check: Optional[Callable[[], bool]] = None,
        spawn_watchdog: Optional[Callable[[Any], Any]] = None,
        preflight_fn: Optional[Callable[[], List[str]]] = None,
        seed_check: Optional[Callable[[], Mapping[str, Any]]] = None,
        transport: str = TRANSPORT_PRODUCTION,
    ):
        self.label = label
        self.intent = intent
        self.intent_text = intent_text
        self.intent_sha = intent_sha
        self.block = intent["execution"]
        self.run_dir = Path(run_dir)
        self.budget = float(budget_usd)
        self.wall_seconds = float(wall_seconds)
        self.rpol, self.fp, self.sp = policies_ or policies()
        self.rp = rp if rp is not None else RpClient()
        self.ledger = ledger or SharedLedger(self.fp, clock=clock)
        self.clock = clock
        self.sleep = sleep
        self.power_check = power_check or R.on_ac_power
        self.lid_check = lid_check or R.lid_open
        self.spawn_watchdog = spawn_watchdog or rmod.spawn_watchdog_process
        self.preflight_fn = preflight_fn or (lambda: rmod.preflight(serverless.SLS_PREFLIGHT_URLS))
        self.seed_check = seed_check
        self.transport = transport
        sizing = self.block["sizing"]
        self.workers = int(workers)
        self.vcpu = int(sizing["vcpu_per_worker"])
        self.slots = int(sizing["slots_per_worker"])
        self.flavors = list(sizing["flavors"])
        prefix = self.rpol["job_id_prefix_gate" if label == "gate" else "job_id_prefix_identity"]
        self.job_id = f"{prefix}{intent_sha[:12]}"
        self.run_id = f"{self.run_dir.resolve()}#{secrets.token_hex(4)}"
        self.key = f"sls:{self.endpoint_prefix()}1"
        self.endpoint_name = f"{self.endpoint_prefix()}1"
        self.endpoint_id: Optional[str] = None
        self.template_id: Optional[str] = None
        self.repo_sha: Optional[str] = None
        self.volume: Dict[str, Any] = {}
        self.created: Optional[float] = None
        self.deleted: Optional[float] = None
        self.until = 0.0
        self.rate = 0.0
        self.balance_start: Optional[float] = None
        self.t0: Optional[float] = None
        self.last_balance = 0.0
        self.last_power = 0.0
        self.balance_failures = 0
        self.spend: Dict[str, Any] = {}
        self.watchdog: Any = None
        self.awake: Any = None
        self.jobs: Dict[str, _Job] = {}
        self.events: Any = None
        self.opened = False
        self.registered = False
        self.reserved = False
        self.receipt: Optional[Dict[str, Any]] = None
        self.stop_reason: Optional[str] = None
        self.dispatch: Dict[str, Any] = {"jobs": [], "lost": [], "duplicates_verified": 0}
        self.seen: Dict[str, bytes] = {}
        self.probe_worker: Dict[str, Any] = {}

    # ------------------------------------------------------------ naming / logging
    def endpoint_prefix(self) -> str:
        return f"{self.sp['endpoint_prefix']}{self.job_id}--{self.run_dir.name}-"

    def log(self, event: str, **fields: Any) -> None:
        line = json.dumps({"utc": _utc(), "event": event, **fields}, sort_keys=True, default=str)
        if self.events is not None:
            self.events.write(line + "\n")
            self.events.flush()

    # ------------------------------------------------------------ open
    def open(self) -> None:
        R.require(not self.opened, "session already opened")
        self.run_dir.mkdir(parents=True)  # create-only: a new dir per attempt
        (self.run_dir / "endpoints").mkdir()
        self.events = (self.run_dir / "events.jsonl").open("x", encoding="utf-8")
        try:
            self._open()
        except BaseException as exc:
            self.log("open_refused", reason=str(exc)[:300])
            self.close(f"open refused: {exc}")
            if isinstance(exc, (R.StrictRunError, RunPodError, DuplicateRun)):
                raise SessionRefused(f"{self.label} remote session refused: {exc}") from exc
            raise

    def _open(self) -> None:
        problems = self.preflight_fn()
        R.require(not problems, f"TLS preflight failed: {problems}")
        seed = self.seed_check() if self.seed_check else self._seed_status()
        R.require(
            seed.get("ready") is True,
            "the private volume is not ready for this commit (owner: serverless.py seed "
            f"--handler strict --commit {self.block['repo_commit']} --confirm, then "
            f"template-create --handler strict --confirm): {seed.get('problems')}",
        )
        self.repo_sha = seed["repo_sha256"]
        self.template_id = seed["template_id"]
        self.volume = dict(seed["volume"])
        R.require(bool(self.template_id), "no serverless template for the strict runtime")
        self._wait_quota()
        balance = float(self.rp.balance())
        R.require(
            balance - float(self.fp["global_floor_usd"]) > self.budget,
            f"balance {balance:.2f} minus the floor does not cover the cap {self.budget}",
        )
        now = self.clock()
        self.balance_start, self.t0 = balance, now
        self.ledger.register_run(
            self.run_id, self.job_id, str(self.run_dir), self.budget, os.getpid()
        )
        self.registered = True
        margin = float(self.sp["price_margin"])
        self.rate = self.workers * serverless.worker_rate(self.sp, self.flavors, self.vcpu) * margin
        self.until = now + self.wall_seconds + float(self.fp["watchdog_grace_seconds"]) + 300.0
        worst = SAFETY * self.rate * (self.until - now + SLACK_SECONDS) / 3600.0
        R.require(
            worst <= self.budget + 1e-9,
            f"worst-case reservation {worst:.2f} exceeds the cap {self.budget}",
        )
        why = self.ledger.reserve_pod(
            self.run_id, self.key, self.rate, self.until, balance, kind="serverless"
        )
        R.require(why is None, f"shared RunPod ledger refused the reservation: {why}")
        self.reserved = True
        target = _Target(
            {"job_id": self.job_id}, now + self.wall_seconds, self.fp, self.run_dir, self.sp
        )
        self.watchdog = self.spawn_watchdog(target)
        self._check_watchdog(startup=True)
        # the Mac must not sleep while jobs bill: caffeinate follows the watchdog (or this run)
        self.awake = rmod.keep_awake(getattr(self.watchdog, "pid", None))
        self._create_endpoint()
        self._probe()
        self.opened = True
        self.log("opened", endpoint=self.endpoint_id, workers=self.workers, vcpu=self.vcpu)

    def _seed_status(self) -> Dict[str, Any]:
        seed = seed_status(
            self.fp, self.sp, self.block["repo_commit"], self.block["checkpoints"], REPO
        )
        reg = serverless.Registry(self.fp).read()
        seed["volume"] = reg.get("volume") or {}
        R.require(
            seed["runtime_id"] == self.block["handler"]["runtime_id"],
            "strict runtime id differs from the intent's",
        )
        return seed

    def _wait_quota(self) -> None:
        quota = int(self.sp.get("account_worker_quota") or 0)
        if quota <= 0:
            return
        t0 = self.clock()
        while True:
            free, held = serverless.read_quota(self.rp, self.sp, self.sleep)
            if int(free or 0) >= self.workers:
                return
            if self.clock() - t0 >= float(self.rpol["quota_wait_seconds"]):
                raise SessionRefused(
                    f"serverless worker quota: {free} of {quota} free (held {held}), need "
                    f"{self.workers}; nothing was created"
                )
            self.log("quota_wait", free=free, held=held, need=self.workers)
            self.sleep(float(self.sp.get("quota_poll_seconds") or 60))

    def _check_watchdog(self, startup: bool = False) -> None:
        wd = self.watchdog
        if wd is None:  # injected stub (tests, dry runs)
            return
        if startup:
            for _ in range(60):
                log = self.run_dir / "watchdog.log"
                if log.exists() and b"armed" in log.read_bytes():
                    break
                self.sleep(0.5)
            else:
                raise GuardStop("watchdog did not arm")
        if wd.poll() is not None:
            raise GuardStop(f"watchdog exited early (rc {wd.returncode})")

    def _create_endpoint(self) -> None:
        sp = dict(self.sp, idle_timeout_seconds=self.rpol["idle_timeout_seconds"])
        timeout = self._job_timeout(max(self.slots, len(R.ARMS)))
        body = serverless.endpoint_body(
            sp,
            self.endpoint_name,
            self.template_id,
            self.volume,
            self.flavors,
            self.workers,
            self.vcpu,
            min(timeout, self.until - self.clock()),
        )
        path = self.run_dir / "endpoints" / f"{self.endpoint_name}.request.json"
        path.write_text(json.dumps(body, indent=1, sort_keys=True))
        self.created = self.clock()
        try:
            out = self.rp.create_endpoint(path, confirm=True)
        except RunPodError as exc:
            out = self._find_endpoint()
            if out is None:
                self.deleted = self.clock()
                raise SessionRefused(f"endpoint create failed: {str(exc)[:200]}") from exc
        if not isinstance(out, dict) or not out.get("id"):
            out = self._find_endpoint() or {}
            if not out.get("id"):
                self.deleted = self.clock()
                raise SessionRefused("endpoint create returned no id and no endpoint exists")
        self.endpoint_id = str(out["id"])
        (self.run_dir / "endpoints" / f"{self.endpoint_name}.response.json").write_text(
            json.dumps(
                {
                    "id": self.endpoint_id,
                    "name": self.endpoint_name,
                    "rpf_until_epoch": self.until,
                    "flavors": self.flavors,
                    "workersMax": self.workers,
                    "vcpuCount": self.vcpu,
                },
                indent=1,
            )
        )
        self.ledger.annotate(self.run_id, self.key, endpoint_id=self.endpoint_id)
        self.log("endpoint_created", endpoint=self.endpoint_id, name=self.endpoint_name)

    def _find_endpoint(self) -> Optional[Dict[str, Any]]:
        for _ in range(3):
            try:
                rows = self.rp.list_endpoints()
            except RunPodError:
                self.sleep(10)
                continue
            for e in rows:
                if str(e.get("name", "")).split(" ")[0] == self.endpoint_name:
                    return e
            self.sleep(30)
        return None

    def _base_input(self, op: str) -> Dict[str, Any]:
        return {
            "op": op,
            "runtime_id": self.block["handler"]["runtime_id"],
            "commit": self.block["repo_commit"],
            "repo_sha256": self.repo_sha,
            "checkpoints": list(self.block["checkpoints"]),
        }

    def _probe(self) -> None:
        out = self.rp.sls(
            self.endpoint_id,
            "run",
            body={
                "input": self._base_input("probe"),
                "policy": {"executionTimeout": 600000, "ttl": 3600000},
            },
            confirm=True,
        )
        R.require(isinstance(out, dict) and out.get("id"), f"probe submit failed: {out}")
        jid = str(out["id"])
        t0 = self.clock()
        while True:
            st = self.rp.sls(self.endpoint_id, "status", jid) or {}
            status = str(st.get("status") or "")
            if status == "COMPLETED":
                worker = dict((st.get("output") or {}).get("worker") or {})
                R.require(
                    not (st.get("output") or {}).get("refused"),
                    f"probe refused: {(st.get('output') or {}).get('refused')}",
                )
                R.require(
                    worker.get("runtime_id") == self.block["handler"]["runtime_id"]
                    and worker.get("handler_sha256") == self.block["handler"]["sha256"],
                    "probe worker runs another runtime or handler",
                )
                self.probe_worker = worker
                self.log("probe_ok", worker=worker)
                return
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise SessionRefused(f"probe job {status}: {str(st.get('error'))[:200]}")
            if self.clock() - t0 > float(self.rpol["startup_deadline_seconds"]):
                raise SessionRefused("no serverless worker started (capacity); nothing ran")
            self.sleep(float(self.rpol["poll_seconds"]))

    # ------------------------------------------------------------ guards
    def own_spend(self, balance: float, now: float) -> Dict[str, Any]:
        """This run's spend: the balance drop since open, less what every OTHER runner's
        ledger reservation could have spent over the same interval."""
        others = 0.0
        t0 = float(self.t0 or now)
        with self.ledger.locked(write=False) as state:
            for rid, run in state["runs"].items():
                if rid == self.run_id:
                    continue
                for pod in run["pods"].values():
                    start = max(t0, float(pod.get("created") or now))
                    end = min(now, float(pod.get("until") or now))
                    if pod.get("deleted") is not None:
                        end = min(end, float(pod["deleted"]) + 60)
                    if end > start:
                        others += float(pod.get("rate") or 0.0) * (end - start) / 3600
        drop = float(self.balance_start or balance) - balance
        return {
            "balance": round(balance, 4),
            "drop": round(drop, 4),
            "others_allowance": round(others, 4),
            "own": round(drop - others, 4),
            "cap": self.budget,
        }

    def guard(self, force: bool = False) -> None:
        now = self.clock()
        if force or now - self.last_power >= float(self.rpol["power_poll_seconds"]):
            self.last_power = now
            if not self.power_check():
                raise GuardStop("on_battery: the Mac orchestrator left AC power")
            if not self.lid_check():
                raise GuardStop("lid_closed: the Mac orchestrator's lid is closed")
        self._check_watchdog()
        if force or now - self.last_balance >= float(self.rpol["balance_guard_seconds"]):
            self.last_balance = now
            try:
                balance = float(self.rp.balance())
            except (RunPodError, ValueError, TypeError) as exc:
                self.balance_failures += 1
                self.log("balance_unavailable", detail=str(exc)[:160])
                if self.balance_failures >= int(self.rpol["balance_failures_stop"]):
                    raise SpendStop(
                        f"spend cap unenforceable: the balance was unreadable "
                        f"{self.balance_failures} times in a row"
                    ) from exc
                return
            self.balance_failures = 0
            self.spend = self.own_spend(balance, now)
            self.log("balance_check", **self.spend)
            if self.spend["own"] >= self.budget:
                raise SpendStop(
                    f"spend cap reached: this run spent ${self.spend['own']:.2f} of its "
                    f"${self.budget:.2f} cap (balance-delta hard stop)"
                )

    # ------------------------------------------------------------ dispatch
    def _job_timeout(self, episodes: int) -> float:
        cfg = self.block["remote_config"]["values"]
        per = max(
            float(self.rpol["episode_timeout_floor_seconds"]),
            float(self.rpol["episode_timeout_factor"])
            * cloud_episode_seconds(cfg, self.fp, self.sp),
        )
        return math.ceil(max(1, int(episodes)) / self.slots) * per + 120.0

    def _submit(self, units: Sequence[Mapping[str, Any]], deadline: float) -> Optional[str]:
        now = self.clock()
        episodes = sum(len(u["episodes"]) for u in units)
        timeout = min(self._job_timeout(episodes), self.until - now - 60)
        if timeout < 60:
            return None
        per_episode = max(60.0, (timeout - 120) / math.ceil(episodes / self.slots))
        inp = self._base_input("units")
        inp.update(
            job_id=self.job_id,
            slots=self.slots,
            timeout_seconds=per_episode,
            deadline_epoch=min(deadline, self.until),
            numerics_env=rmod.numerics_env(self.fp),
            intent_text=self.intent_text,
            intent_sha256=self.intent_sha,
            units=[
                {"unit": u["unit"], "row": u["row"], "episodes": list(u["episodes"])} for u in units
            ],
        )
        body = {
            "input": inp,
            "policy": {
                "executionTimeout": int(timeout * 1000),
                "ttl": int(max(60.0, min(deadline, self.until) - now) * 1000),
            },
        }
        try:
            out = self.rp.sls(self.endpoint_id, "run", body=body, confirm=True)
        except RunPodError as exc:
            self.log("submit_failed", detail=str(exc)[:200])
            return None
        if not isinstance(out, dict) or not out.get("id"):
            self.log("submit_no_id", detail=str(out)[:200])
            return None
        jid = str(out["id"])
        self.jobs[jid] = _Job(jid, [u["unit"] for u in units], now, timeout)
        self._persist_jobs()
        self.log("submitted", job=jid, units=[u["unit"] for u in units], episodes=episodes)
        return jid

    def _persist_jobs(self) -> None:
        if not self.endpoint_id:
            return
        tmp = self.run_dir / "endpoints" / ".jobs.json.tmp"
        tmp.write_text(json.dumps({self.endpoint_id: sorted(self.jobs)}, sort_keys=True))
        os.replace(tmp, self.run_dir / "endpoints" / "jobs.json")

    def _cancel(self, jid: str) -> None:
        try:
            self.rp.sls(self.endpoint_id, "cancel", jid, confirm=True)
        except RunPodError:
            pass

    def observe(self, eid: str, record: Mapping[str, Any], source: str) -> None:
        """Duplicate rule: a second copy of an episode must be byte-identical."""
        data = record_bytes(record)
        prior = self.seen.get(eid)
        if prior is None:
            self.seen[eid] = data
            return
        if prior != data:
            raise RemoteAbort(f"DIVERGENT duplicate of {eid} from {source}")
        self.dispatch["duplicates_verified"] += 1

    def run_units(
        self,
        units: Sequence[Mapping[str, Any]],
        *,
        deadline: float,
        accept: Callable[[Mapping[str, Any], List[Dict[str, Any]], Mapping[str, Any]], None],
    ) -> Dict[str, Any]:
        """Dispatch ``units`` (whole) until each is accepted, the deadline passes (returns
        incomplete) or a guard / failure raises. ``accept(unit, outputs, meta)`` validates
        and publishes one complete unit (``outputs`` in the unit's episode order)."""
        R.require(self.opened, "session not opened")
        deadline = min(float(deadline), float(self.t0 or self.clock()) + self.wall_seconds)
        by_key = {u["unit"]: u for u in units}
        R.require(len(by_key) == len(units), "duplicate unit keys")
        pending = deque(u["unit"] for u in units)
        attempts: Dict[str, int] = {}
        solo: set = set()
        done: List[str] = []
        cap = max(1, math.ceil(self.workers * float(self.sp["inflight_jobs_per_worker"])))
        stopped = None
        refused = 0
        while len(done) < len(units):
            self.guard()
            if self.clock() >= deadline:
                stopped = "deadline"
                break
            while len(self.jobs) < cap and pending:
                sizes = [(k, len(by_key[k]["episodes"])) for k in pending]
                batch = pack_units(sizes, self.slots, sorted(solo))
                if self._submit([by_key[k] for k in batch], deadline) is None:
                    refused += 1
                    if refused >= int(self.rpol["submit_failures_stop"]):
                        raise RemoteAbort(f"{refused} job submissions failed in a row")
                    break
                refused = 0
                for k in batch:
                    pending.remove(k)
            self._poll(by_key, pending, attempts, solo, done, accept)
            if len(done) < len(units):
                self.sleep(float(self.rpol["poll_seconds"]))
        if stopped is not None:
            for jid in list(self.jobs):
                self._cancel(jid)
                self.jobs.pop(jid, None)
            self._persist_jobs()
        return {"done": list(done), "stopped": stopped, "attempts": dict(attempts)}

    def _lost(self, job: _Job, why: str, by_key, pending, attempts, solo) -> None:
        self.dispatch["lost"].append({"job": job.id, "units": job.units, "why": why})
        self.log("job_lost", job=job.id, units=job.units, why=why)
        for unit in reversed(job.units):  # back at the front, in their order
            self._requeue(unit, why, pending, attempts)
        if len(job.units) > 1:
            solo.update(job.units)

    def _requeue(self, unit: str, why: str, pending, attempts) -> None:
        attempts[unit] = attempts.get(unit, 0) + 1
        if attempts[unit] >= int(self.block["max_attempts_per_unit"]):
            raise RemoteAbort(f"unit {unit} lost {attempts[unit]} times (last: {why})")
        pending.appendleft(unit)

    def _poll(self, by_key, pending, attempts, solo, done, accept) -> None:
        for jid in list(self.jobs):
            job = self.jobs[jid]
            try:
                st = self.rp.sls(self.endpoint_id, "status", jid) or {}
            except RunPodError as exc:
                job.errors += 1
                if job.errors >= int(self.rpol["status_errors_lost"]):
                    self._cancel(jid)
                    self.jobs.pop(jid)
                    self._lost(
                        job, f"status unavailable: {str(exc)[:80]}", by_key, pending, attempts, solo
                    )
                continue
            job.errors = 0
            status = str(st.get("status") or "")
            now = self.clock()
            if status == "IN_PROGRESS" and job.started is None:
                job.started = now
            if status in ("IN_QUEUE", "IN_PROGRESS", ""):
                if job.started and now - job.started > job.timeout_s + 600:
                    self._cancel(jid)
                    self.jobs.pop(jid)
                    self._lost(
                        job, "overran its execution timeout", by_key, pending, attempts, solo
                    )
                continue
            self.jobs.pop(jid)
            self._persist_jobs()
            out = st.get("output")
            if status != "COMPLETED" or not isinstance(out, dict):
                self._lost(
                    job, f"job {status or 'without output'}", by_key, pending, attempts, solo
                )
                continue
            if out.get("refused"):
                if str(out["refused"]) == "deadline":
                    self._lost(job, "refused: deadline", by_key, pending, attempts, solo)
                    continue
                raise RemoteAbort(f"worker refused job {jid}: {str(out['refused'])[:300]}")
            self._collect(job, out, st, by_key, pending, attempts, solo, done, accept)

    def _collect(self, job, out, st, by_key, pending, attempts, solo, done, accept) -> None:
        worker = dict(out.get("worker") or {})
        worker_id = str(worker.get("worker_id") or st.get("workerId") or "")
        returned = {str(u.get("unit")): u for u in out.get("units") or [] if isinstance(u, dict)}
        R.require(set(returned) <= set(job.units), f"job {job.id} returned unknown units")
        entry = {"job": job.id, "worker_id": worker_id, "units": {}}
        for key in job.units:
            unit = by_key[key]
            results = {
                str(r.get("episode_id")): r
                for r in (returned.get(key) or {}).get("results") or []
                if isinstance(r, dict)
            }
            R.require(
                set(results) <= {e["episode_id"] for e in unit["episodes"]},
                f"job {job.id} returned unknown episodes for {key}",
            )
            fatal = [
                r for r in results.values() if r.get("kind") in ("episode_error", "binding_error")
            ]
            if fatal:
                first = fatal[0]
                raise RemoteAbort(
                    f"{first.get('kind')} in {first.get('episode_id')} on worker {worker_id}: "
                    f"{str(first.get('error') or first.get('stderr_tail'))[-400:]}"
                )
            outputs, failed = [], []
            for episode in unit["episodes"]:
                r = results.get(episode["episode_id"])
                if not r or r.get("kind") != "ok":
                    failed.append(episode["episode_id"])
                    continue
                parsed = check_result(self.intent, self.intent_sha, unit, episode, r)
                self.observe(episode["episode_id"], parsed["record"], f"{job.id}/{worker_id}")
                outputs.append(parsed)
            if failed:
                entry["units"][key] = {"status": "requeued", "failed": failed}
                self._requeue(key, f"worker failure in {failed[:2]}", pending, attempts)
                if len(job.units) > 1:
                    solo.add(key)
                continue
            meta = {
                "job_id": job.id,
                "worker_id": worker_id,
                "worker": worker,
                "attempt": attempts.get(key, 0) + 1,
                "endpoint": self.endpoint_name,
            }
            accept(unit, outputs, meta)
            done.append(key)
            entry["units"][key] = {"status": "accepted", "attempt": meta["attempt"]}
        self.dispatch["jobs"].append(entry)
        self.log("collected", job=job.id, worker=worker_id, units=entry["units"])

    # ------------------------------------------------------------ close
    def close(self, why: str) -> Dict[str, Any]:
        """Tear down (idempotent): endpoint, ledger, receipt. Never raises."""
        if self.receipt is not None:
            return self.receipt
        self.stop_reason = why
        leftovers: List[str] = []
        try:
            for jid in list(self.jobs):
                self._cancel(jid)
            self.jobs.clear()
            if self.endpoint_id and self.deleted is None:
                ok = serverless.teardown_endpoint(
                    self.rp,
                    self.endpoint_id,
                    [],
                    lambda m: self.log("teardown", detail=m),
                    self.sleep,
                    self.clock,
                )
                if ok:
                    self.deleted = self.clock()
            if self.created is not None and self.deleted is None:
                leftovers = serverless.sweep_endpoints(
                    self.rp,
                    self.endpoint_prefix(),
                    self.run_dir,
                    lambda m: self.log("sweep", detail=m),
                    self.sleep,
                    self.clock,
                    give_up_seconds=600,
                )
                if not leftovers:
                    self.deleted = self.clock()
        except Exception as exc:  # noqa: BLE001 - the watchdog keeps trying
            leftovers = leftovers or [f"teardown error: {exc!r}"[:200]]
        upper = 0.0
        if self.created is not None:
            end = self.deleted if self.deleted is not None else self.until
            upper = self.rate * (max(0.0, end - self.created) + 60) / 3600
        balance_end = None
        if self.balance_start is not None:
            try:
                self.sleep(float(self.rpol["close_settle_seconds"]))
                balance_end = float(self.rp.balance())
                self.spend = self.own_spend(balance_end, self.clock())
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal here
                self.log("balance_unavailable", detail=str(exc)[:160])
        try:
            if self.reserved and self.deleted is not None:
                self.ledger.release_pod(self.run_id, self.key, upper)
            if self.registered:
                self.ledger.finish_run(
                    self.run_id, success=not leftovers, episodes_complete=why == "complete"
                )
        except Exception as exc:  # noqa: BLE001
            self.log("ledger_error", detail=repr(exc)[:200])
        self.receipt = {
            "schema": RECEIPT_SCHEMA,
            "label": self.label,
            "job_id": self.job_id,
            "transport": self.transport,
            "run_dir": str(self.run_dir),
            "stop_reason": why,
            "endpoint": {
                "id": self.endpoint_id,
                "name": self.endpoint_name,
                "workers_max": self.workers,
                "vcpu_per_worker": self.vcpu,
                "created": self.created,
                "deleted": self.deleted,
                "reserved_usd_per_hr": round(self.rate, 4),
            },
            "probe_worker": self.probe_worker,
            "dispatch": self.dispatch,
            "spend": self.spend,
            "balance_start": self.balance_start,
            "balance_end": balance_end,
            "cap_usd": self.budget,
            "ledger_cost_usd_upper_bound": round(upper, 5),
            "leftover_endpoints": leftovers,
            "utc": _utc(),
        }
        if self.run_dir.is_dir():
            # the watchdog's "runner finished" signal (it sweeps, re-sweeps, then exits)
            (self.run_dir / "receipt.json").write_text(
                json.dumps(R.json_safe(self.receipt), indent=1, sort_keys=True)
            )
        if self.events is not None:
            self.log("closed", why=why, leftovers=leftovers)
            self.events.close()
            self.events = None
        if self.awake is not None and self.watchdog is None:
            self.awake.terminate()  # with a watchdog, caffeinate ends when the watchdog does
        return self.receipt


# ---------------------------------------------------------------- the gate's executor


class RemoteExecutor:
    """Runs each look segment as RunPod world-unit jobs; the orchestrator (this process) keeps
    the barrier, writes the shard markers, records and reports exactly as local workers do
    (so ``collect_prefix``, the receipts and the audit are unchanged), plus a create-only
    ``remote.json`` per segment."""

    identity = REMOTE_EXECUTOR

    def __init__(
        self,
        intent_path: Path,
        intent: Mapping[str, Any],
        spec: R.StudySpec,
        session_factory: Optional[Callable[..., RemoteSession]] = None,
    ):
        self.intent_path = Path(intent_path)
        self.intent = intent
        self.spec = spec
        self.root = self.intent_path.parent
        self.session_factory = session_factory or RemoteSession
        self.session: Optional[RemoteSession] = None
        self.summary: Optional[Dict[str, Any]] = None
        self.transport = TRANSPORT_PRODUCTION

    def preflight(self) -> None:
        text = self.intent_path.read_text(encoding="utf-8")
        block = self.intent["execution"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
        self.session = self.session_factory(
            label="gate",
            intent=self.intent,
            intent_text=text,
            intent_sha=R.sha256_file(self.intent_path),
            run_dir=self.root / REMOTE_DIR / f"gate-{stamp}-{secrets.token_hex(2)}",
            budget_usd=block["cost_cap"]["gate_usd"],
            wall_seconds=block["remote_wall_seconds"],
            workers=block["sizing"]["workers"],
        )
        self.transport = self.session.transport
        self.session.open()

    def run_segment(
        self, ctx: Any, phase: str, look: int, wall_seconds: float, worker_deadline: datetime
    ) -> Dict[str, Any]:
        began = time.monotonic()
        session = self.session
        R.require(session is not None and session.opened, "remote session not open")
        output = ctx.output
        seg = R.segment_dir(output, phase, look)
        gate = R.prior_gate(ctx.intent, output, phase, look)
        admitted = R.read_json(output / "admitted.json")
        rosters_path = output / "rosters.json"
        R.require(R.sha256_file(rosters_path) == admitted["rosters_sha256"], "roster drift")
        rows = {(r["mix"], int(r["world_index"])): r for r in R.read_json(rosters_path)[phase]}
        workers = ctx.intent["caps"]["workers"]
        shard_units = {k: R.segment_units(ctx.intent, phase, look, k) for k in range(workers)}
        planned = {
            k: [e for u in units for e in R.unit_episodes(phase, u, look)]
            for k, units in shard_units.items()
        }
        for k in range(workers):
            (seg / f"shard-{k}").mkdir()
            R.write_once(
                seg / f"shard-{k}" / "started.json",
                {
                    "phase": phase,
                    "look": look,
                    "shard": k,
                    "gate": gate,
                    "intent_sha256": ctx.intent_sha,
                    "planned_episode_ids": [e["episode_id"] for e in planned[k]],
                    "executor": self.identity,
                    "utc": R.now_utc().isoformat(),
                },
            )
        units = sorted(
            (u for k in range(workers) for u in shard_units[k]), key=lambda u: int(u["unit"])
        )
        dispatch = plan_units(ctx.intent, phase, units, look, rows)
        records_dir = output / phase / "records"
        records_dir.mkdir(exist_ok=True)
        done: Dict[int, Dict[str, str]] = {k: {} for k in range(workers)}
        published: Dict[str, Any] = {}
        spent: List[float] = []

        def accept(unit, outputs, meta) -> None:
            entries = []
            for episode, out in zip(unit["episodes"], outputs):
                entry = R.envelope(
                    ctx.intent, ctx.intent_sha, episode, out["record"], float(out["wall_seconds"])
                )
                entry["platform"] = {
                    **out["platform"],
                    "backend": REMOTE_BACKEND,
                    "worker_id": meta["worker_id"],
                    "worker_isa_flags": list(meta["worker"].get("isa_flags") or []),
                }
                entry["fanout"] = {
                    "job_id": meta["job_id"],
                    "attempt": meta["attempt"],
                    "unit_key": unit["unit"],
                    "endpoint": meta["endpoint"],
                }
                problems = R.envelope_problems(ctx.intent, ctx.intent_sha, entry, episode)
                problems += list(ctx.spec.validate_record(entry, unit["row"]))
                R.require(not problems, f"record shape: {problems}")
                entries.append(entry)
            stamps = {R.canonical_sha(e["platform"]) for e in entries}
            R.require(len(stamps) == 1, f"unit {unit['unit']} spans platforms")
            for entry in entries:
                path = records_dir / f"{entry['episode_id']}.json"
                R.require(not path.exists(), f"record {path.name} already exists (create-only)")
                done[unit["shard"]][entry["episode_id"]] = R.write_once(path, entry)
                spent.append(float(entry["wall_seconds"]))
            published[unit["unit"]] = {
                "index": unit["index"],
                "shard": unit["shard"],
                "episodes": [e["episode_id"] for e in entries],
                "job_id": meta["job_id"],
                "worker_id": meta["worker_id"],
                "attempt": meta["attempt"],
                "platform_id": entries[0]["platform"]["platform_id"],
            }

        cause = None
        jobs_before = len(session.dispatch["jobs"])
        lost_before = len(session.dispatch["lost"])
        try:
            result = session.run_units(
                dispatch, deadline=worker_deadline.timestamp(), accept=accept
            )
        except (R.Interrupted, KeyboardInterrupt):
            session.close("interrupted")
            raise
        except Exception as exc:  # noqa: BLE001 - stop spending, then fail the segment
            cause = f"remote_abort: {type(exc).__name__}: {exc}"
            session.close(cause)
            result = {"done": list(published), "stopped": "abort", "attempts": {}}
        stopped = None
        if cause is None and result["stopped"] == "deadline":
            left = (worker_deadline - R.now_utc()).total_seconds()
            stopped = f"deadline: {left:.0f}s left; {len(dispatch) - len(published)} units open"
        for k in range(workers):
            ids = [e["episode_id"] for e in planned[k]]
            R.write_once(
                seg / f"shard-{k}" / "report.json",
                {
                    "phase": phase,
                    "look": look,
                    "shard": k,
                    "slot": None,
                    "executor": self.identity,
                    "planned_episode_ids": ids,
                    "records_sha256": done[k],
                    "complete": cause is None and stopped is None and len(done[k]) == len(ids),
                    "stopped_reason": stopped if cause is None else cause,
                    "wall_seconds_total": sum(spent),
                    "utc": R.now_utc().isoformat(),
                },
            )
        R.write_once(
            seg / "remote.json",
            R.json_safe(
                {
                    "schema": SEGMENT_SCHEMA,
                    "phase": phase,
                    "look": look,
                    "intent_sha256": ctx.intent_sha,
                    "endpoint": session.endpoint_name,
                    "planned_units": [u["unit"] for u in dispatch],
                    "units": published,
                    "jobs": session.dispatch["jobs"][jobs_before:],
                    "lost": session.dispatch["lost"][lost_before:],
                    "attempts": result.get("attempts", {}),
                    "stopped": stopped,
                    "cause": cause,
                    "spend": session.spend,
                    "utc": R.now_utc().isoformat(),
                }
            ),
        )
        child_ok = cause is None
        return {
            "cause": cause,
            "elapsed_seconds": time.monotonic() - began,
            "wall_seconds": wall_seconds,
            "executor": self.identity,
            "children": [
                {
                    "command": [self.identity, phase, str(look), f"shard-{k}"],
                    "returncode": 0 if child_ok else 1,
                    "termination": "natural_exit" if child_ok else "remote_abort",
                    "confirmed_exit": True,
                    "peak_group_rss_bytes": 0,
                }
                for k in range(workers)
            ],
            "min_available_bytes": None,
        }

    def close(self, why: str) -> Dict[str, Any]:
        if self.summary is not None:
            return self.summary
        receipt = self.session.close(why) if self.session is not None else {}
        self.summary = {
            "backend": REMOTE_BACKEND,
            "transport": self.transport,
            "run_dir": str(self.session.run_dir) if self.session is not None else None,
            "stop_reason": receipt.get("stop_reason"),
            "spend": receipt.get("spend"),
            "cap_usd": receipt.get("cap_usd"),
            "leftover_endpoints": receipt.get("leftover_endpoints"),
            "receipt_sha256": (
                R.sha256_file(self.session.run_dir / "receipt.json")
                if self.session is not None and (self.session.run_dir / "receipt.json").is_file()
                else None
            ),
        }
        return self.summary


def production_remote_factory(intent_path: Path, intent: Mapping[str, Any], spec: Any):
    return RemoteExecutor(intent_path, intent, spec)


def acquire_orchestrator_lock(root: Path) -> Any:
    """The remote run parent's liveness lock (it holds no CPU slot)."""
    path = Path(root) / REMOTE_DIR / ORCHESTRATOR_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise R.StrictRunError("another process holds this run's orchestrator lock") from exc
    return handle


def release_lock(handle: Any) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def close_quietly(executor: Any, why: str) -> Optional[Dict[str, Any]]:
    close = getattr(executor, "close", None)
    if close is None:
        return None
    try:
        return close(why)
    except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome
        print(f"remote close failed: {exc!r}", file=sys.stderr)
        return {"close_error": repr(exc)[:300]}


# ---------------------------------------------------------------- identity check (condition 3)


def sample_units(intent: Mapping[str, Any], spec: R.StudySpec) -> List[Dict[str, Any]]:
    """The pre-registered identity sample as dispatchable final-phase units (look 0), with
    the roster rows ``admit_rosters`` will freeze for the same worlds (``spec.build_row``)."""
    wanted = {
        (s["mix"], int(s["world_index"])) for s in intent["execution"]["identity_check"]["sample"]
    }
    units = [u for u in R.phase_units(intent, "final") if (u["mix"], u["world_index"]) in wanted]
    R.require(len(units) == len(wanted), "identity sample outside the final bank")
    rows = {}
    for u in units:
        row = spec.build_row("final", u["mix"], u["world_index"], u["world_seed"])
        R.require(
            row["mix"] == u["mix"]
            and int(row["world_index"]) == u["world_index"]
            and int(row["world_seed"]) == u["world_seed"],
            "build_row must keep mix, world_index and world_seed",
        )
        rows[(u["mix"], u["world_index"])] = R.json_safe(row)
    return plan_units(intent, "final", units, 0, rows)


def mac_identity_digests(
    intent: Mapping[str, Any],
    spec: R.StudySpec,
    units: Sequence[Mapping[str, Any]],
    key: bytes,
    *,
    shard: Optional[int] = None,
    workers: int = R.WORKERS,
    backend: str = "local-mac",
) -> Dict[str, Any]:
    """Play the sample on this machine exactly as a local gate worker would
    (``spec.worker_setup`` once, then ``spec.episode_runner``) and keep salted digests only.
    ``shard`` k plays the sample units at positions ``i % workers == k``."""
    context = spec.worker_setup(intent)
    digests = {}
    for i, unit in enumerate(units):
        if shard is not None and i % workers != shard:
            continue
        for episode in unit["episodes"]:
            record = spec.episode_runner(dict(episode), dict(unit["row"]), context)
            digests[episode["episode_id"]] = salted_digests(key, record)
    return {"digests": digests, "platform": remote_worker.platform_stamp(backend)}


def in_process_mac_identity(
    intent_path: Path,
    intent: Mapping[str, Any],
    spec: R.StudySpec,
    units: Sequence[Mapping[str, Any]],
    key: bytes,
    idir: Path,
) -> Dict[str, Any]:
    """Dry runs and tests only: the Mac side in this process."""
    try:
        out = mac_identity_digests(intent, spec, units, key, backend="in-process")
    except Exception as exc:  # noqa: BLE001 - recorded: the check then fails (Mac fallback)
        return {"executor": "in-process", "complete": False, "error": repr(exc)[:300]}
    return {"executor": "in-process", "complete": True, "error": None, **out}


class SubprocessMacIdentity:
    """Production Mac side: two supervised ``identity-worker`` children on the gate's 2 CPU
    slots (acquired BEFORE the check starts, so busy slots refuse the check cleanly), with
    the AC and lid guards polled while they run."""

    identity = "subprocess"

    def __init__(self, slot_lock_root: Path):
        self.slot_lock_root = Path(slot_lock_root)
        self.slots: List[Any] = []

    def acquire(self) -> None:
        self.slots = R.acquire_run_slots(self.slot_lock_root, R.WORKERS)

    def release(self) -> None:
        R.release_run_slots(self.slots)
        self.slots = []

    def __call__(self, intent_path, intent, spec, units, key, idir) -> Dict[str, Any]:
        key_path = Path(idir) / ".identity-key"
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(key.hex())
        try:
            commands = [
                [
                    intent["python"],
                    "-I",
                    "-B",
                    str(HERE / "sequential_runner.py"),
                    "identity-worker",
                    "--intent",
                    str(intent_path),
                    "--shard",
                    str(k),
                    "--slot-fd",
                    str(self.slots[k].fileno()),
                    "--key-file",
                    str(key_path),
                    "--out",
                    str(Path(idir) / f"mac-shard-{k}.json"),
                ]
                for k in range(R.WORKERS)
            ]
            env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
            env.update(R.capacity_environment())
            result = R.supervise_children(
                commands,
                cwd=Path(intent["repo"]),
                env=env,
                logs=[Path(idir) / f"mac-shard-{k}.log" for k in range(R.WORKERS)],
                heartbeats=[None] * R.WORKERS,
                wall_seconds=float(intent["execution"]["identity_wall_seconds"]),
                pass_fds=[(self.slots[k].fileno(),) for k in range(R.WORKERS)],
                power_check=lambda: R.on_ac_power() and R.lid_open(),
            )
        finally:
            key_path.unlink(missing_ok=True)
        files = [Path(idir) / f"mac-shard-{k}.json" for k in range(R.WORKERS)]
        if not R.clean_supervision(result) or not all(f.is_file() for f in files):
            return {
                "executor": self.identity,
                "complete": False,
                "error": f"Mac identity workers: {result['cause']}",
                "supervision": result,
            }
        digests, stamps = {}, []
        for f in files:
            part = R.read_json(f)
            digests.update(part["digests"])
            stamps.append(part["platform"])
        return {
            "executor": self.identity,
            "complete": True,
            "error": None,
            "digests": digests,
            "platform": stamps[0],
            "platforms_agree": all(s == stamps[0] for s in stamps),
        }


def identity_worker_main(
    intent_path: Path, shard: int, slot_fd: int, key_file: Path, out: Path
) -> int:
    """Child entry of the production Mac side: parentage, slot, closure, spec; then digests."""
    intent = R.read_json(intent_path)
    started = R.read_json(Path(intent_path).parent / IDENTITY_DIR / "started.json")
    R.require(started["pid"] == os.getppid(), "identity worker is not supervised by its parent")
    R.require(started["intent_sha256"] == R.sha256_file(intent_path), "intent drift")
    R.assert_inherited_slot(slot_fd, Path(intent["slot_lock_root"]), shard + 1)
    R.require(not R.closure_drift(intent["source_closure"]), "source drift at identity worker")
    spec = R.resolve_spec(intent["spec_ref"])
    R.require(intent["spec"] == spec.descriptor(), "spec drift at identity worker")
    key = bytes.fromhex(Path(key_file).read_text().strip())
    units = sample_units(intent, spec)
    R.require(
        [u["unit"] for u in units] == started["units"], "identity sample differs from started"
    )
    result = mac_identity_digests(intent, spec, units, key, shard=shard, backend="local-mac")
    R.write_once(Path(out), {"shard": shard, **result})
    return 0


def remote_identity(
    session: RemoteSession,
    intent: Mapping[str, Any],
    intent_sha: str,
    spec: R.StudySpec,
    units: Sequence[Mapping[str, Any]],
    key: bytes,
) -> Dict[str, Any]:
    """The RunPod side on an already-open session: records are validated (bindings, horizon,
    record shape, one platform per unit) and reduced to salted digests in memory."""
    digests: Dict[str, Any] = {}
    platforms: Dict[str, Any] = {}

    def accept(unit, outputs, meta) -> None:
        stamps = {R.canonical_sha(o["platform"]) for o in outputs}
        R.require(len(stamps) == 1, f"identity unit {unit['unit']} spans platforms")
        for episode, out in zip(unit["episodes"], outputs):
            entry = R.envelope(intent, intent_sha, episode, out["record"], 0.0)
            problems = R.envelope_problems(intent, intent_sha, entry, episode)
            problems += list(spec.validate_record(entry, unit["row"]))
            R.require(not problems, f"identity record shape: {problems}")
            digests[episode["episode_id"]] = salted_digests(key, out["record"])
            platforms[episode["episode_id"]] = {
                **out["platform"],
                "worker_id": meta["worker_id"],
                "job_id": meta["job_id"],
            }

    error, stopped = None, None
    try:
        result = session.run_units(
            units, deadline=session.clock() + session.wall_seconds, accept=accept
        )
        stopped = result["stopped"]
    except (R.Interrupted, KeyboardInterrupt):
        session.close("interrupted")
        raise
    except Exception as exc:  # noqa: BLE001 - recorded: the check fails (Mac fallback)
        error = f"{type(exc).__name__}: {exc}"[:400]
    receipt = session.close("complete" if error is None and stopped is None else (error or stopped))
    complete = error is None and stopped is None
    return {
        "executor": REMOTE_EXECUTOR,
        "transport": session.transport,
        "complete": complete,
        "error": error or (None if stopped is None else f"stopped: {stopped}"),
        "digests": digests,
        "platforms": platforms,
        "receipt": {
            "run_dir": receipt.get("run_dir"),
            "spend": receipt.get("spend"),
            "cap_usd": receipt.get("cap_usd"),
            "leftover_endpoints": receipt.get("leftover_endpoints"),
            "stop_reason": receipt.get("stop_reason"),
        },
    }


def run_identity_check(
    intent_path: Path,
    *,
    spec: Optional[R.StudySpec] = None,
    mac_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    session_factory: Optional[Callable[..., RemoteSession]] = None,
    power_check: Optional[Callable[[], bool]] = None,
    lid_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Condition 3: once per intent, after ``prepare`` and before ``run``.

    Everything that can refuse (AC, lid, the Mac slots, the RunPod session: seeding, quota,
    capacity, ledger, probe) happens BEFORE ``identity_check/started.json`` exists, so a
    refusal leaves the check NOT_RUN. From that marker on, the check ends in exactly one
    create-only ``identity_check.json`` (PASSED or FAILED), or it is ABANDONED (which ``run``
    treats as FAILED).
    """
    intent_path = Path(intent_path).resolve()
    intent = R.read_json(intent_path)
    R.require(is_remote(intent), "identity-check is for template v3 (remote execution) intents")
    spec = spec if spec is not None else R.resolve_spec(intent["spec_ref"])
    R.validate_intent(intent, spec)
    R.require(intent_path == Path(intent["output_root"]) / "intent.json", "intent location")
    root = intent_path.parent
    status = R.run_status(root)
    R.require(status["state"] == "NOT_STARTED", f"the gate is {status['state']}: too late")
    state = identity_state(root)
    R.require(
        state["state"] == "NOT_RUN",
        f"identity check is {state['state']}: it runs once per intent and is never re-run",
    )
    if intent["dry_run"]:
        R.require(
            mac_runner is not None and session_factory is not None,
            "a dry-run identity check needs injected runners (never real slots or RunPod)",
        )
    else:
        R.require(
            mac_runner is None and session_factory is None,
            "a production identity check uses the subprocess Mac side and the real RunPod session",
        )
    R.require((power_check or R.on_ac_power)(), "host on battery: no identity check")
    R.require((lid_check or R.lid_open)(), "lid closed: no identity check")
    production_mac = None
    if mac_runner is None:
        production_mac = SubprocessMacIdentity(Path(intent["slot_lock_root"]))
        production_mac.acquire()
        mac_runner = production_mac
    block = intent["execution"]
    intent_text = intent_path.read_text(encoding="utf-8")
    intent_sha = R.sha256_file(intent_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    session = (session_factory or RemoteSession)(
        label="identity",
        intent=intent,
        intent_text=intent_text,
        intent_sha=intent_sha,
        run_dir=root / REMOTE_DIR / f"identity-{stamp}-{secrets.token_hex(2)}",
        budget_usd=block["cost_cap"]["identity_usd"],
        wall_seconds=block["identity_wall_seconds"],
        workers=block["sizing"]["identity_workers"],
    )
    lock = None
    try:
        session.open()  # refuses cleanly: nothing started yet
        idir = root / IDENTITY_DIR
        idir.mkdir()  # create-only: never re-run
        lock = (idir / "running.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        units = sample_units(intent, spec)
        expected = [e["episode_id"] for u in units for e in u["episodes"]]
        R.write_once(
            idir / "started.json",
            {
                "schema": IDENTITY_SCHEMA,
                "intent_sha256": intent_sha,
                "pid": os.getpid(),
                "units": [u["unit"] for u in units],
                "expected_episodes": expected,
                "rows_sha256": {u["unit"]: R.canonical_sha(u["row"]) for u in units},
                "mac_runner": getattr(mac_runner, "identity", R.runner_identity(mac_runner)),
                "remote_session": str(session.run_dir),
                "utc": R.now_utc().isoformat(),
            },
        )
        key = secrets.token_bytes(32)  # never written to identity_check.json
        mac = mac_runner(intent_path, intent, spec, units, key, idir)
        if mac.get("complete"):
            remote = remote_identity(session, intent, intent_sha, spec, units, key)
        else:
            session.close("Mac side incomplete")
            remote = {
                "executor": REMOTE_EXECUTOR,
                "complete": False,
                "error": "not run: the Mac side did not complete",
                "digests": {},
            }
        key = b""
        verdict = judge_identity(expected, mac, remote)
        result = {
            "schema": IDENTITY_SCHEMA,
            "intent_sha256": intent_sha,
            "study_id": intent["study_id"],
            "dry_run": intent["dry_run"],
            "sample": block["identity_check"]["sample"],
            "units": [u["unit"] for u in units],
            "expected_episodes": expected,
            "rows_sha256": {u["unit"]: R.canonical_sha(u["row"]) for u in units},
            "mac": {k: v for k, v in mac.items() if k != "supervision"},
            "remote": remote,
            **verdict,
            "backend_decided": REMOTE_BACKEND if verdict["passes"] else FALLBACK_BACKEND,
            "started_sha256": R.sha256_file(idir / "started.json"),
            "utc": R.now_utc().isoformat(),
        }
        R.write_once(root / IDENTITY_FILE, R.json_safe(result))
        return result
    finally:
        session.close("identity check end")
        if production_mac is not None:
            production_mac.release()
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


# ---------------------------------------------------------------- plan report (CLI)


def remote_plan_report(
    spec: R.StudySpec,
    params: Mapping[str, Any],
    config_path: Path,
    n_calibration: int,
    *,
    account: bool = False,
    rp: Any = None,
    repo: Path = REPO,
    seeding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """``remote-plan``: the projection ``prepare --remote-config`` would freeze, plus what
    still blocks it (allow-list, seeding, ratification); ``--account`` adds read-only balance,
    worker quota and ledger headroom."""
    rpol, fp, sp = policies()
    cfg = load_remote_config(Path(config_path), rpol, sp)
    plan_dict = R.plan_from_parameters(params).as_dict()
    closure = R.source_closure(spec.closure_roots, repo)
    hooks = callable(spec.remote_worker_setup) and callable(spec.remote_checkpoints)
    ckpts = sorted({str(s) for s in spec.remote_checkpoints()}) if hooks else []
    allow = jobspec.load_allowlist()
    not_allowed = [s[:12] for s in ckpts if s not in allow]
    if seeding is None:
        seeding = seed_status(fp, sp, closure["git_commit"], ckpts, repo)
    plan = plan_remote(
        mixes=spec.mixes,
        n_calibration=n_calibration,
        look_sizes=plan_dict["look_sizes"],
        cfg=cfg,
        rpol=rpol,
        fp=fp,
        sp=sp,
        seeding=seeding,
    )
    amendment = amendment_status(Path(repo) / rpol["amendment_path"])
    blockers = []
    if not hooks:
        blockers.append("spec lacks remote_worker_setup / remote_checkpoints")
    if not_allowed:
        blockers.append(f"checkpoints not on the RunPod allow-list: {not_allowed}")
    if plan["choice"] is None:
        blockers.append("no sizing fits the caps")
    elif not plan["meets_min_speedup"] and not cfg["force_below_5x"]:
        blockers.append(f"speed-up {plan['choice']['speedup']}x < {plan['min_speedup']}x")
    if not amendment["ratified"]:
        blockers.append("amendment not ratified (production prepare refuses; dry runs allowed)")
    if closure["dirty"]:
        blockers.append("source closure is dirty")
    report: Dict[str, Any] = {
        "dry_run": True,
        "platform": PLATFORM,
        "look_sizes": plan_dict["look_sizes"],
        "plan": plan,
        "seeding": dict(seeding),
        "commit": closure["git_commit"],
        "checkpoints": ckpts,
        "amendment": amendment,
        "blockers": blockers,
        "admissible": plan["choice"] is not None
        and (plan["meets_min_speedup"] or cfg["force_below_5x"])
        and hooks
        and not not_allowed,
        "owner_steps": (
            [
                f"serverless.py seed --handler strict --commit {closure['git_commit']} --confirm",
                "serverless.py template-create --handler strict --confirm",
            ]
            if not seeding.get("ready")
            else []
        ),
    }
    if account:
        rp = rp or RpClient()
        rows = list(rp.list_endpoints())
        report["account"] = {
            "balance_usd": float(rp.balance()),
            "worker_quota": serverless.quota_status(rows, sp),
            "ledger": SharedLedger(fp).peek(),
        }
    return report
