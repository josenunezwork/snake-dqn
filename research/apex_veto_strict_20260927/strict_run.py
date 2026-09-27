#!/usr/bin/env python3
"""Tier-2 strict challenge: Apex champion + free-space veto (candidate) vs the champion.

Pre-registration: ``protocol.md`` beside this file. Subcommands:

* ``prepare`` writes ``<root>/intent.json`` once: source closure and git commit, checkpoint
  sha256, candidate identity (checkpoint + wrapper + wrapper source sha256), namespaces and
  their disjointness report, deadline, caps, authorization quote, the pre-registered design
  and the final-world count N frozen from the independent Tier-1 screen pilot.
* ``run`` admits that intent once: calibration -> final -> serving (each sharded over two
  supervised worker processes, one shared CPU slot each) -> audit child -> closeout. No retry.
* ``worker`` and ``audit`` are internal child entry points launched by ``run``.

A STRICT_PASS writes a receipt only; nothing here changes a champion, default or deployment.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Mapping, Sequence  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402

SCHEMA = "apex-veto-strict/v1"
AUTHORITY = "tier-2-strict-promotion"
STUDY_ID = "apex-veto-strict-20260927"
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.md"
MIXES = dev_screen.MIXES
HORIZON = dev_screen.HORIZON
PROFILE_NAME = dev_screen.PROFILE_NAME
CHAMPION = dev_screen.CHAMPION
SERVING_MIX = "serving-selfplay"
ARMS = ("incumbent", "candidate")
STAGES = ("calibration", "final", "serving")

NAMESPACE_KEY = "worlds"
NAMESPACES = {
    "dev": ("apex-veto-strict-dev-v1", 16),
    "final": ("apex-veto-strict-final-v1", 300),
    "serving": ("apex-veto-strict-serving-v1", 50),
}
SMOKE_DOMAIN = "apex-veto-strict-smoke-v1"  # plumbing dry-runs only; never Tier-2 worlds
SCREEN_PREFIX_CHECKED = 1000  # first seeds of apex-safety-screen-v1 checked for overlap

MDE_ABSOLUTE = 20.0
NI_FRACTION = 0.03
BAND_BELOW = 0.02
BAND_ABOVE = 1.0
N_FLOOR = 40
N_MAX = 300
ALPHA = 0.05

# Final cap: pre-registered from the Tier-1 screen's measured episode times (protocol.md
# "Execution envelope"): 235 x 189.8 s of A+B rollout per world triplet / 2 workers = 22.3k s
# per worker before overhead; 45000 s keeps the overhead-inclusive projection near 57%.
CAPS = {"calibration": 1800, "final": 45000, "serving": 3600, "audit": 900}
HANDOFF_SECONDS = 120
WORKERS = 2
RSS_BYTES = 8 * 1024**3
AVAILABLE_GIB = 9.6
AVAILABLE_BYTES = math.ceil(AVAILABLE_GIB * 1024**3)
POLL_SECONDS = 1.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_STALE_SECONDS = 600.0
SLOT_TIMEOUT_SECONDS = 180.0
WORKER_STOP_MARGIN_SECONDS = 30.0
MIN_EPISODE_BUDGET_SECONDS = 45.0
PROJECTION_OVERHEAD = 1.15  # screen: 8582 s elapsed / 7590 s summed A+B rollouts = 1.13
PROJECTION_MAX_BUDGET_FRACTION = 0.70  # prepare refuses a projected final above this

SLOT_LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")
SUPERVISOR_SOURCE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "apex-scaling-h256-diagnostic/supervise.py"
)
STUDY_ARTIFACT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-strict-20260927"
)
DEFAULT_OUT_ROOT = STUDY_ARTIFACT_ROOT / "run-v1"  # the pre-registered Tier-2 root
DEFAULT_SMOKE_ROOT = STUDY_ARTIFACT_ROOT / "smoke-v1"  # pre-GO plumbing dry-run root
# The only root a non-smoke intent may use (tests point it at tmp_path).
TIER2_OUT_ROOT = DEFAULT_OUT_ROOT
SCREEN_RUN = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-safety-screen-20260926/run-v1"
)
PYTHON = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
WRAPPER_SOURCE = REPO / "src/evaluation/safety_veto.py"
INDEPENDENT_AUDIT = HERE / "strict_audit.py"
CLOSURE_ROOTS = ("src", "research/apex_safety_20260926", "research/apex_veto_strict_20260927")
CLOSURE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md")

OUTCOMES = ("STRICT_PASS", "STRICT_FAIL", "STOP_INFEASIBLE", "INCOMPLETE", "INVALID_STOP")
SERVING_REMAINING_WORK = [
    "web/backend/session.py GameSession: opt-in install of FreeSpaceVeto on the served "
    "Watch hero (vector61 ApexPolicy build path has no wrapper hook today)",
    "Play mode: install the wrapper on AI snakes (human dispatch bypasses the policy)",
    "a vector61+wrapper per-episode serving receipt schema; strict_promotion._serving "
    "accepts only raster31v3 receipts",
    "a 50-episode web serving run under that schema, with its own audit",
    "any deployment that wraps more than the one hero (every Watch snake sharing the policy, "
    "or every AI snake in Play) needs its own strict evaluation: this study measures one "
    "wrapped hero against unwrapped opponents only",
]
SERVING_STAGE_KIND = "rollout-harness self-play compatibility check"
NON_CLAIMS = [
    "no champion file, default, config or deployment change on any outcome",
    "a STRICT_PASS is a receipt only; release is a separate explicit action",
    "serving_path_qualified is false: the web serving path cannot host the wrapper yet",
    "Tier-1 screen numbers are not evidence in this study (pilot variance only)",
    "the result covers a single wrapped hero against unwrapped opponents under "
    "promotion-v2-watch-rect H5000 only; wrapping several or all snakes is not measured",
    "the serving stage is a rollout-harness self-play compatibility check through "
    "tournament_eval.rollout; it exercises no web backend or session code",
]


class StrictRunError(RuntimeError):
    """A fail-closed check in this harness."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`StrictRunError` unless ``condition`` (not disabled by -O)."""
    if not condition:
        raise StrictRunError(message)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timestamps need an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def read_json(path: Path) -> Any:
    """Read JSON, rejecting NaN/Infinity constants."""

    def reject(token: str) -> None:
        raise StrictRunError(f"non-finite JSON constant {token} in {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_durable(path: Path, value: Any) -> str:
    """Create-only, fsynced JSON write; returns the file's sha256."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return dev_screen.sha256_file(path)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(dev_screen.canonical_json(value).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- namespaces and rosters


def namespace_seeds(counts: Mapping[str, int] | None = None) -> Dict[str, List[int]]:
    """Ordered seeds of the three new namespaces (the dev_screen SHA-prefix recipe)."""
    counts = counts or {name: count for name, (_, count) in NAMESPACES.items()}
    return {
        name: [
            dev_screen.uint32_seed(NAMESPACES[name][0], NAMESPACE_KEY, i)
            for i in range(counts[name])
        ]
        for name in NAMESPACES
    }


def namespace_report(
    seeds: Mapping[str, Sequence[int]], pilot_output: Path | None
) -> Dict[str, Any]:
    """Construction-time freshness check; ``disjoint`` is the fail-closed verdict.

    Each namespace must be unique within itself and disjoint from the other two, from every
    set ``dev_screen.disjointness_report`` checks, and from the Tier-1 screen namespace
    ``apex-safety-screen-v1`` (first ``SCREEN_PREFIX_CHECKED`` seeds).
    """
    screen = set(dev_screen.screen_seeds(SCREEN_PREFIX_CHECKED))
    smoke = {dev_screen.uint32_seed(SMOKE_DOMAIN, NAMESPACE_KEY, i) for i in range(16)}
    report: Dict[str, Any] = {"namespaces": {}, "cross_namespace_overlap": {}}
    ok = True
    for name, values in seeds.items():
        unique = len(set(values)) == len(values)
        earlier = dev_screen.disjointness_report(values, pilot_output)
        screen_overlap = sorted((screen | smoke) & set(values))
        report["namespaces"][name] = {
            "domain": NAMESPACES[name][0],
            "namespace": NAMESPACE_KEY,
            "count": len(values),
            "unique": unique,
            "seeds_sha256": canonical_sha(list(values)),
            "earlier_namespaces": earlier,
            "tier1_screen_or_smoke_overlap": screen_overlap,
        }
        ok = ok and unique and bool(earlier["disjoint"]) and not screen_overlap
        ok = ok and earlier.get("strict_pilot_observed") is not None
    names = list(seeds)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(seeds[left]) & set(seeds[right]))
            report["cross_namespace_overlap"][f"{left}|{right}"] = overlap
            ok = ok and not overlap
    report["tier1_screen_domain"] = dev_screen.SCREEN_DOMAIN
    report["tier1_screen_prefix_checked"] = SCREEN_PREFIX_CHECKED
    report["smoke_domain_prefix_checked"] = {"domain": SMOKE_DOMAIN, "count": 16}
    report["registry_checked"] = False
    report["registry_note"] = "governance namespace registry not implemented; in-code check only"
    report["disjoint"] = bool(ok)
    return report


def serving_rows(seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Self-play Watch rosters: five incumbent opponents (no wrapper) per serving world."""
    return [
        {
            "mix": SERVING_MIX,
            "world_seed": int(seed),
            "slots": [
                {"slot": slot, "member_sha256": CHAMPION[1]}
                for slot in range(1, dev_screen.ROSTER_WIDTH + 1)
            ],
        }
        for seed in seeds
    ]


def build_rosters(
    dev: Sequence[int],
    final: Sequence[int],
    serving: Sequence[int],
    mixes: Sequence[str] = MIXES,
) -> Dict[str, List[Dict[str, Any]]]:
    """Materialized rosters per stage (strict balanced construction via dev_screen).

    Rows are mix-major, world-minor; only ``mixes`` are kept (smoke plays one mix). A
    prefix of an ordered bank yields the same rows as the prefix of the full bank's rows,
    because slot members rotate by world index.
    """

    def design(seeds: Sequence[int]) -> List[Dict[str, Any]]:
        if not seeds:
            return []
        return [row for row in dev_screen._design_rows(list(seeds)) if row["mix"] in mixes]

    return {"calibration": design(dev), "final": design(final), "serving": serving_rows(serving)}


def intent_rosters(intent: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Rebuild the rosters an intent pins (used by the producer and by the audit)."""
    return build_rosters(
        intent["namespaces"]["dev"]["seeds"],
        intent["final_seeds"],
        intent["namespaces"]["serving"]["seeds"],
        intent["mixes"],
    )


def stage_plan(stage: str, rows: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Deterministic ordered units per stage; a unit runs on one worker, in order.

    calibration: one incumbent episode per row; final: an (incumbent, candidate) pair per
    row; serving: one candidate episode per row. Rows are mix-major, world-minor.
    """
    arms = {"calibration": ("incumbent",), "final": ARMS, "serving": ("candidate",)}[stage]
    units = []
    for row in rows:
        units.append(
            [
                {
                    "episode_id": f"{stage}-{arm}-{row['mix']}-{row['world_seed']}",
                    "stage": stage,
                    "arm": arm,
                    "mix": row["mix"],
                    "world_seed": int(row["world_seed"]),
                }
                for arm in arms
            ]
        )
    return units


def shard_units(units: Sequence[Any], shard: int, workers: int = WORKERS) -> List[Any]:
    """Unit ``j`` goes to worker ``j % workers`` (deterministic, balanced, disjoint)."""
    require(0 <= shard < workers, "shard out of range")
    return [unit for index, unit in enumerate(units) if index % workers == shard]


# ---------------------------------------------------------------- pilot sizing


def screen_pilot_deltas(screen_run: Path) -> Dict[str, Any]:
    """Recompute the Tier-1 screen's paired B-A mass deltas from its raw records.

    Reads ``records/{A,B}-<mix>-<seed>.json`` (read-only), orders pairs by the screen's
    world index, and cross-checks against ``summary.json`` per-mix deltas.
    """
    records = Path(screen_run) / "records"
    by_key: Dict[tuple, Mapping[str, Any]] = {}
    files: Dict[str, str] = {}
    for path in sorted(records.glob("*.json")):
        entry = read_json(path)
        if entry["arm"] not in ("A", "B"):
            continue
        require(entry["hero_sha256"] == CHAMPION[1], f"screen record hero differs: {path.name}")
        by_key[(entry["arm"], entry["mix"], int(entry["world_seed"]))] = entry
        files[path.name] = dev_screen.sha256_file(path)
    deltas: Dict[str, List[float]] = {}
    for mix in MIXES:
        worlds = sorted(
            {(e["world_index"], s) for (a, m, s), e in by_key.items() if m == mix and a == "A"}
        )
        paired = [(i, s) for i, s in worlds if ("B", mix, s) in by_key]
        deltas[mix] = [
            by_key[("B", mix, s)]["record"]["mass_integral"]
            - by_key[("A", mix, s)]["record"]["mass_integral"]
            for _, s in paired
        ]
    summary_path = Path(screen_run) / "summary.json"
    summary_match = None
    if summary_path.is_file():
        summary = read_json(summary_path)
        summary_match = all(
            len(deltas[m])
            == len(summary["per_mix"][m]["primary_mass_integral"]["deltas_B_minus_A"])
            and all(
                math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                for a, b in zip(
                    deltas[m], summary["per_mix"][m]["primary_mass_integral"]["deltas_B_minus_A"]
                )
            )
            for m in MIXES
        )
    return {
        "deltas_by_mix": deltas,
        "record_files_sha256": files,
        "summary_sha256": (
            dev_screen.sha256_file(summary_path) if summary_path.is_file() else None
        ),
        "summary_deltas_match": summary_match,
        "episode_wall_seconds": {
            f"{arm}-{mix}": [
                float(e["wall_seconds"])
                for (a, m, _), e in sorted(by_key.items())
                if a == arm and m == mix
            ]
            for arm in ("A", "B")
            for mix in MIXES
        },
    }


def frozen_final_n(deltas_by_mix: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """N = max(40, paired_delta_pilot_size(deltas, {m: MDE})); feasible iff N <= N_MAX."""
    from src.scripts.eval_stats import paired_delta_pilot_size

    sizing = paired_delta_pilot_size(
        {m: list(deltas_by_mix[m]) for m in MIXES}, {m: MDE_ABSOLUTE for m in MIXES}
    )
    n = max(N_FLOOR, int(sizing["required_final_worlds"]))
    return {
        "mde_absolute_per_mix": MDE_ABSOLUTE,
        "pilot_sizing": sizing,
        "final_worlds_per_mix": n,
        "n_max": N_MAX,
        "feasible": n <= N_MAX,
    }


def runtime_projection(
    wall_seconds: Mapping[str, Sequence[float]], n_final: int, cap: float | None = None
) -> Dict[str, Any]:
    """Projected per-worker final-stage wall time from the screen's measured episode times.

    Basis: the Tier-1 screen ran one worker (torch 2 intra-op / 1 inter-op). The final plays
    ``n_final`` worlds x 3 mixes x both arms over ``WORKERS`` workers; ``PROJECTION_OVERHEAD``
    covers per-episode overhead. Two concurrent workers were never measured, so the projection
    must leave ``1 - PROJECTION_MAX_BUDGET_FRACTION`` of the worker budget for contention.
    """
    cap = CAPS["final"] if cap is None else cap
    per_mix = {}
    for mix in MIXES:
        a, b = list(wall_seconds[f"A-{mix}"]), list(wall_seconds[f"B-{mix}"])
        require(a and b, f"no measured screen wall times for {mix}")
        per_mix[mix] = {
            "incumbent_mean_seconds": sum(a) / len(a),
            "candidate_mean_seconds": sum(b) / len(b),
            "max_seconds": max(a + b),
            "episodes": len(a) + len(b),
        }
    triplet = sum(
        r["incumbent_mean_seconds"] + r["candidate_mean_seconds"] for r in per_mix.values()
    )
    rollout = n_final * triplet / WORKERS
    projected = rollout * PROJECTION_OVERHEAD
    budget = cap - WORKER_STOP_MARGIN_SECONDS
    return {
        "basis": "Tier-1 screen A/B record wall_seconds, one worker, 2 intra-op / 1 inter-op",
        "concurrency_measured": False,
        "per_mix_episode_seconds": per_mix,
        "pair_seconds_per_world_triplet": triplet,
        "rollout_seconds_per_worker": rollout,
        "overhead_factor": PROJECTION_OVERHEAD,
        "projected_seconds_per_worker": projected,
        "worker_budget_seconds": budget,
        "fraction_of_worker_budget": projected / budget,
        "max_fraction": PROJECTION_MAX_BUDGET_FRACTION,
        "within_limit": projected / budget <= PROJECTION_MAX_BUDGET_FRACTION,
    }


# ---------------------------------------------------------------- source closure and identity


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def source_closure(repo: Path = REPO, extra: Sequence[Path] = ()) -> Dict[str, Any]:
    """sha256 of every tracked or untracked-unignored runtime file under CLOSURE_ROOTS.

    ``extra`` adds absolute files outside the repo (the supervisor helpers). ``dirty`` lists
    closure paths with uncommitted changes (``git status --porcelain``).
    """
    listed = _git(
        repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", *CLOSURE_ROOTS
    ).splitlines()
    files = {
        str((repo / item).resolve()): dev_screen.sha256_file(repo / item)
        for item in sorted(set(listed))
        if item.endswith(CLOSURE_SUFFIXES) and (repo / item).is_file()
    }
    for path in extra:
        files[str(Path(path).resolve())] = dev_screen.sha256_file(Path(path))
    dirty = [
        line for line in _git(repo, "status", "--porcelain", "--", *CLOSURE_ROOTS).splitlines()
    ]
    return {
        "git_commit": _git(repo, "rev-parse", "HEAD").strip(),
        "roots": list(CLOSURE_ROOTS),
        "files": files,
        "digest": canonical_sha(files),
        "dirty": dirty,
    }


def closure_drift(closure: Mapping[str, Any]) -> List[str]:
    """Closure files whose bytes changed (or vanished) since ``prepare``."""
    drift = []
    for path, expected in closure["files"].items():
        if not Path(path).is_file() or dev_screen.sha256_file(Path(path)) != expected:
            drift.append(path)
    return drift


def wrapper_identity() -> Dict[str, Any]:
    """The candidate wrapper's identity: method, descriptor and source sha256."""
    from src.evaluation.safety_veto import VETO_METHOD, FreeSpaceVeto

    return {
        "method": VETO_METHOD,
        "descriptor": FreeSpaceVeto().descriptor(),
        "source_path": str(WRAPPER_SOURCE),
        "source_sha256": dev_screen.sha256_file(WRAPPER_SOURCE),
    }


def load_supervisor(path: Path, expected_sha256: str) -> Any:
    """Import the verified supervisor helpers (never its ``main``)."""
    require(dev_screen.sha256_file(path) == expected_sha256, "supervisor helper drift")
    spec = importlib.util.spec_from_file_location("apex_veto_strict_supervisor", path)
    require(spec is not None and spec.loader is not None, "supervisor import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- prepare


def required_seconds(from_stage: str) -> float:
    """Caps of ``from_stage`` and every later stage (audit included) plus the handoff."""
    order = [*STAGES, "audit"]
    return float(sum(CAPS[s] for s in order[order.index(from_stage) :]) + HANDOFF_SECONDS)


def check_output_root(out_root: Path, smoke: bool) -> None:
    """A Tier-2 intent uses exactly the pre-registered root; a smoke never uses a run root."""
    root = Path(out_root).resolve()
    require(dev_screen.FORBIDDEN_OUTPUT_ROOT not in Path(out_root).parts, "forbidden out root")
    require(dev_screen.FORBIDDEN_OUTPUT_ROOT not in root.parts, "forbidden out root")
    tier2 = Path(TIER2_OUT_ROOT).resolve()
    if smoke:
        require(
            root != tier2 and not root.name.startswith("run-"),
            f"a smoke dry-run may not use a Tier-2 run root ({root}); use {DEFAULT_SMOKE_ROOT}",
        )
    else:
        require(root == tier2, f"a Tier-2 intent must use the pre-registered root {tier2}")


def build_intent(
    *,
    out_root: Path,
    deadline: datetime,
    authorization_quote: str,
    screen_run: Path = SCREEN_RUN,
    pilot_output: Path | None = dev_screen.DEFAULT_PILOT_OUTPUT,
    checkpoint_dir: Path = dev_screen.DEFAULT_CHECKPOINT_DIR,
    config: Path = dev_screen.DEFAULT_CONFIG,
    supervisor_source: Path = SUPERVISOR_SOURCE,
    slot_lock_root: Path = SLOT_LOCK_ROOT,
    python: Path = PYTHON,
    smoke_frames: int | None = None,
    allow_dirty_source: bool = False,
    clock: Callable[[], datetime] = now_utc,
) -> Dict[str, Any]:
    """Assemble the write-once intent; raises on any failed construction-time check."""
    prepared = clock()
    require(bool(authorization_quote.strip()), "an authorization quote is required")
    smoke = smoke_frames is not None
    check_output_root(out_root, smoke)
    if smoke:
        require(0 < smoke_frames <= 500, "smoke is limited to 500 frames")
    else:
        require(
            deadline - prepared >= timedelta(seconds=required_seconds("calibration")),
            "deadline leaves less than the caps plus handoff",
        )
    require(dev_screen.sha256_file(config) == dev_screen.CONFIG_SHA256, "config bytes differ")
    for name, sha in dev_screen.POOL:
        require(dev_screen.sha256_file(Path(checkpoint_dir) / name) == sha, f"{name} sha drift")
    require(INDEPENDENT_AUDIT.is_file(), "independent audit strict_audit.py is missing")
    closure = source_closure(extra=[supervisor_source])
    # A plumbing smoke may run from an uncommitted tree (recorded in the closure).
    require(
        allow_dirty_source or smoke or not closure["dirty"], f"dirty source: {closure['dirty']}"
    )

    if smoke:
        # Smoke never touches a Tier-2 namespace: its one world is from its own domain.
        domains = {"dev": SMOKE_DOMAIN, "final": SMOKE_DOMAIN, "serving": SMOKE_DOMAIN}
        seeds = {
            "dev": [],
            "final": [dev_screen.uint32_seed(SMOKE_DOMAIN, NAMESPACE_KEY, 0)],
            "serving": [],
        }
        report = {"disjoint": True, "smoke": True}
        parity = {"checked": False, "reason": "smoke"}
        pilot = {"deltas_by_mix": None, "smoke": True}
        sizing = {"final_worlds_per_mix": 1, "feasible": True, "smoke": True}
        projection = {"smoke": True}
    else:
        domains = {name: domain for name, (domain, _) in NAMESPACES.items()}
        seeds = namespace_seeds()
        report = namespace_report(seeds, pilot_output)
        require(report["disjoint"], "namespaces are not fresh/disjoint")
        parity = dev_screen.roster_parity_report(pilot_output)
        require(
            parity.get("checked")
            and not parity.get("mismatches")
            and parity.get("compared") == dev_screen.EXPECTED_ROSTER_PARITY,
            "roster parity against the strict pilot failed",
        )
        pilot = screen_pilot_deltas(screen_run)
        require(pilot["summary_deltas_match"] is True, "screen records disagree with summary")
        require(all(len(pilot["deltas_by_mix"][m]) >= 2 for m in MIXES), "pilot too small")
        sizing = frozen_final_n(pilot["deltas_by_mix"])
        projection = runtime_projection(
            pilot.pop("episode_wall_seconds"), int(sizing["final_worlds_per_mix"])
        )
        require(
            not sizing["feasible"] or projection["within_limit"],
            f"projected final {projection['projected_seconds_per_worker']:.0f} s per worker "
            f"exceeds {PROJECTION_MAX_BUDGET_FRACTION:.0%} of the worker budget",
        )
    wrapper = wrapper_identity()
    n_final = int(sizing["final_worlds_per_mix"])
    final_seeds = seeds["final"][: min(n_final, len(seeds["final"]))]
    mixes = ["scripted"] if smoke else list(MIXES)
    return {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "study_id": STUDY_ID,
        "attempt_id": Path(out_root).name,
        "output_root": str(Path(out_root).resolve()),
        "prepared_utc": prepared.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "authorization": {"quote": authorization_quote, "utc": prepared.isoformat()},
        "protocol": {"path": str(PROTOCOL), "sha256": dev_screen.sha256_file(PROTOCOL)},
        "repo": str(REPO),
        "python": str(python),
        "source_closure": closure,
        "supervisor_source": {
            "path": str(Path(supervisor_source).resolve()),
            "sha256": dev_screen.sha256_file(supervisor_source),
        },
        "slot_lock_root": str(Path(slot_lock_root)),
        "config": {"path": str(Path(config).resolve()), "sha256": dev_screen.CONFIG_SHA256},
        "profile": {"name": PROFILE_NAME, "digest": dev_screen.PROFILE_DIGEST, "horizon": HORIZON},
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "checkpoint_pool": [{"name": n, "sha256": s} for n, s in dev_screen.POOL],
        "incumbent": {"checkpoint_sha256": CHAMPION[1], "name": CHAMPION[0], "wrapper": None},
        "candidate": {
            "checkpoint_sha256": CHAMPION[1],
            "name": CHAMPION[0],
            "wrapper": wrapper["method"],
            "wrapper_identity": wrapper,
            "wrapper_source_sha256": wrapper["source_sha256"],
        },
        "namespaces": {
            name: {"domain": domains[name], "namespace": NAMESPACE_KEY, "seeds": values}
            for name, values in seeds.items()
        },
        "namespace_report": report,
        "roster_parity": parity,
        "mixes": mixes,
        "design": {
            "alpha": ALPHA,
            "decision": "eval_stats.strict_promotion_decision(scripted_mix='scripted')",
            "ni_fraction_of_dev_incumbent_scripted_mean": NI_FRACTION,
            "survival_band": {"below": BAND_BELOW, "above": BAND_ABOVE},
            "mde_absolute_per_mix": MDE_ABSOLUTE,
            "tier1_screen_seen_before_mde": True,
            "outcomes": list(OUTCOMES),
        },
        "pilot": {"screen_run": str(screen_run), **pilot},
        "sizing": sizing,
        "final_seeds": final_seeds,
        "caps": {
            "stage_seconds": dict(CAPS),
            "handoff_seconds": HANDOFF_SECONDS,
            "workers": WORKERS,
            "rss_bytes": RSS_BYTES,
            "min_available_bytes": AVAILABLE_BYTES,
            "torch_intraop_threads": 2,
            "torch_interop_threads": 1,
            "poll_seconds": POLL_SECONDS,
            "term_grace_seconds": TERM_GRACE_SECONDS,
            "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
            "retry_authorized": False,
            "slot_locks": "held by the run parent from before started.json to closeout; "
            "workers share them through inherited descriptors (--slot-fd)",
            "final_runtime_projection": projection,
        },
        "audit": {
            "independent_source": {
                "path": str(INDEPENDENT_AUDIT),
                "sha256": dev_screen.sha256_file(INDEPENDENT_AUDIT),
            },
            "independent_command": [
                str(python),
                "-I",
                str(INDEPENDENT_AUDIT),
                "--root",
                str(Path(out_root).resolve()),
                "--out",
                str(Path(out_root).resolve() / "output" / "audit"),
                "--pilot-root",
                str(Path(screen_run).resolve()),
                *(["--smoke"] if smoke else []),
            ],
            "independent_pass_rule": "exit 0 and audit.json status PASS",
            "self_check": "strict_run.py audit (runs after the independent audit)",
            "smoke": "dry-run: strict_audit.py --smoke (legacy-path record rules), gating",
        },
        "smoke_frames": smoke_frames,
        "serving_path_qualified": False,
        "serving_stage_kind": SERVING_STAGE_KIND,
        "serving_stage_exercises_web_path": False,
        "serving_remaining_work": SERVING_REMAINING_WORK,
        "non_claims": NON_CLAIMS,
    }


def prepare(argv_intent: Mapping[str, Any]) -> Path:
    """Create ``out_root`` and write ``intent.json`` exactly once; returns its path."""
    out_root = Path(argv_intent["output_root"])
    require(not out_root.exists(), f"output root {out_root} already exists")
    out_root.mkdir(parents=True)
    path = out_root / "intent.json"
    write_durable(path, argv_intent)
    return path


# ---------------------------------------------------------------- worker (internal child)


def acquire_run_slots(lock_root: Path, timeout: float = SLOT_TIMEOUT_SECONDS) -> List[Any]:
    """All-or-nothing: hold every worker's CPU slot (``cpu-slot-1..WORKERS.lock``) or raise.

    The run parent calls this before ``output/`` exists (a busy slot is not an attempt) and
    holds the locks until closeout, so no other job can take a slot between stages.
    """
    handles = [(Path(lock_root) / f"cpu-slot-{k + 1}.lock").open("a+") for k in range(WORKERS)]
    started = time.monotonic()
    while True:
        acquired: List[Any] = []
        try:
            for handle in handles:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired.append(handle)
            return handles
        except BlockingIOError:
            for handle in acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if time.monotonic() - started > timeout:
                for handle in handles:
                    handle.close()
                raise StrictRunError(f"CPU slots unavailable for {timeout:.0f} s")
            time.sleep(0.1)


def release_run_slots(handles: Sequence[Any]) -> None:
    for handle in handles:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def assert_inherited_slot(fd: int, lock_root: Path, slot: int) -> None:
    """Worker side: ``fd`` is the parent's open ``cpu-slot-<slot>.lock`` and holds its lock.

    flock belongs to the open file description, which the worker shares with the parent, so
    a non-blocking LOCK_EX on it succeeds without contention. It fails if another description
    holds the lock. The worker never unlocks it (that would release the parent's lock).
    """
    target = Path(lock_root) / f"cpu-slot-{slot}.lock"
    info, expected = os.fstat(fd), target.stat()
    require(
        (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino),
        f"inherited fd {fd} is not {target.name}",
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise StrictRunError(f"CPU slot {slot} is held by another process") from exc


def episode_entry(
    unit: Mapping[str, Any],
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    shard: int,
    wall_seconds: float,
    world_index: int,
) -> Dict[str, Any]:
    """The write-once per-episode envelope around a rollout record (dev_screen shape).

    ``world_index`` is the seed's index in its namespace bank. Only candidate envelopes
    carry ``safety_veto: true``, ``wrapper`` and ``wrapper_source_sha256``.
    """
    candidate = unit["arm"] == "candidate"
    entry = {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "episode_id": unit["episode_id"],
        "stage": unit["stage"],
        "arm": unit["arm"],
        "mix": unit["mix"],
        "world_seed": int(unit["world_seed"]),
        "hero_sha256": CHAMPION[1],
        "world_index": int(world_index),
        "safety_veto": candidate,
        "wrapper": wrapper["method"] if candidate else None,
        "roster_member_sha256s": [slot["member_sha256"] for slot in row["slots"]],
        "shard": shard,
        "wall_seconds": wall_seconds,
        "record": record,
    }
    if candidate:
        entry["wrapper_source_sha256"] = wrapper["source_sha256"]
    return entry


def world_index_maps(intent: Mapping[str, Any]) -> Dict[str, Dict[int, int]]:
    """Per stage: seed -> index in its namespace bank (dev, full final bank, serving)."""
    banks = {"calibration": "dev", "final": "final", "serving": "serving"}
    return {
        stage: {int(seed): i for i, seed in enumerate(intent["namespaces"][name]["seeds"])}
        for stage, name in banks.items()
    }


def run_unit_episode(
    unit: Mapping[str, Any],
    row: Mapping[str, Any],
    lookup: Mapping[str, tuple],
    profile: Any,
    smoke_frames: int | None,
) -> Mapping[str, Any]:
    """Play one hero episode (the candidate carries the opt-in veto; nobody else does)."""
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import rollout

    hero = lookup[CHAMPION[1]]
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    veto = unit["arm"] == "candidate"
    if smoke_frames is not None:
        return rollout(hero, opponents, smoke_frames, row["world_seed"], hero_safety_veto=veto)
    return rollout(
        hero,
        opponents,
        HORIZON,
        row["world_seed"],
        profile=profile,
        world_identity=_expected_world_identity(row),
        mix_id=row["mix"],
        hero_safety_veto=veto,
    )


def _write_heartbeat(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def worker_main(
    intent_path: Path, stage: str, shard: int, stage_deadline: datetime, slot_fd: int
) -> int:
    """Internal: run one shard of one numeric stage under the parent's supervision.

    The CPU slot lock is the parent's, shared through the inherited descriptor ``slot_fd``.
    """
    require(stage in STAGES, "unknown stage")
    intent = read_json(intent_path)
    intent_sha = dev_screen.sha256_file(intent_path)
    output = Path(intent["output_root"]) / "output"
    started = read_json(output / "started.json")
    require(started["pid"] == os.getppid(), "worker is not supervised by the run parent")
    require(started["intent_sha256"] == intent_sha, "intent drift at worker start")
    admitted = read_json(output / "admitted.json")
    shard_dir = output / stage / f"shard-{shard}"
    shard_dir.mkdir()
    records_dir = output / stage / "records"
    assert_inherited_slot(slot_fd, Path(intent["slot_lock_root"]), shard + 1)
    return _worker_body(intent, admitted, stage, shard, stage_deadline, shard_dir, records_dir)


def _worker_body(
    intent: Mapping[str, Any],
    admitted: Mapping[str, Any],
    stage: str,
    shard: int,
    stage_deadline: datetime,
    shard_dir: Path,
    records_dir: Path,
) -> int:
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    dev_screen._configure_torch()
    require(
        dev_screen.sha256_file(Path(intent["config"]["path"])) == intent["config"]["sha256"],
        "config drift",
    )
    load_and_initialize_config(intent["config"]["path"])
    profile = evaluation_profile_for_name(PROFILE_NAME, HORIZON)
    require(profile.digest == intent["profile"]["digest"], "profile digest drift")
    rosters_path = Path(intent["output_root"]) / "output" / "rosters.json"
    require(dev_screen.sha256_file(rosters_path) == admitted["rosters_sha256"], "roster drift")
    rows = read_json(rosters_path)[stage]
    by_key = {(row["mix"], int(row["world_seed"])): row for row in rows}
    lookup = dev_screen.agent_lookup(admitted["checkpoint_snapshots"])
    wrapper = intent["candidate"]["wrapper_identity"]
    index = world_index_maps(intent)[stage]
    units = shard_units(stage_plan(stage, rows), shard)
    planned = [episode for unit in units for episode in unit]
    heartbeat = shard_dir / "heartbeat.json"
    done: Dict[str, str] = {}
    spent: List[float] = []
    stopped = None
    _write_heartbeat(heartbeat, {"utc": now_utc().isoformat(), "done": 0})
    for episode in planned:
        budget = max(MIN_EPISODE_BUDGET_SECONDS, 2 * sum(spent) / len(spent) if spent else 0)
        remaining = (stage_deadline - now_utc()).total_seconds()
        if remaining < budget:
            stopped = f"deadline: {remaining:.0f}s left < {budget:.0f}s budget"
            break
        row = by_key[(episode["mix"], episode["world_seed"])]
        began = time.monotonic()
        record = run_unit_episode(episode, row, lookup, profile, intent["smoke_frames"])
        elapsed = time.monotonic() - began
        spent.append(elapsed)
        entry = episode_entry(
            episode, row, record, wrapper, shard, elapsed, index[episode["world_seed"]]
        )
        done[episode["episode_id"]] = write_durable(
            records_dir / f"{episode['episode_id']}.json", entry
        )
        _write_heartbeat(heartbeat, {"utc": now_utc().isoformat(), "done": len(done)})
    write_durable(
        shard_dir / "report.json",
        {
            "schema_version": SCHEMA,
            "stage": stage,
            "shard": shard,
            "slot": shard + 1,
            "planned_episode_ids": [e["episode_id"] for e in planned],
            "records_sha256": done,
            "complete": stopped is None and len(done) == len(planned),
            "stopped_reason": stopped,
            "wall_seconds_total": sum(spent),
            "utc": now_utc().isoformat(),
        },
    )
    return 0


# ---------------------------------------------------------------- supervision (parent side)


class Interrupted(BaseException):
    """SIGTERM/SIGHUP received by the run parent."""


def _group_rss(psutil: Any, pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(child: subprocess.Popen, grace: float) -> str:
    """SIGTERM the child's process group, SIGKILL after ``grace``; always reap."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
        termination = "terminated"
    except subprocess.TimeoutExpired:
        termination = "killed"
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        termination = "killed"
    child.wait()
    return termination


def supervise_children(
    commands: Sequence[Sequence[str]],
    *,
    cwd: Path,
    env: Mapping[str, str],
    logs: Sequence[Path],
    heartbeats: Sequence[Path | None],
    wall_seconds: float,
    rss_limit_bytes: int = RSS_BYTES,
    available_floor_bytes: int = AVAILABLE_BYTES,
    poll_seconds: float = POLL_SECONDS,
    grace_seconds: float = TERM_GRACE_SECONDS,
    heartbeat_stale_seconds: float = HEARTBEAT_STALE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    pass_fds: Sequence[Sequence[int]] | None = None,
) -> Dict[str, Any]:
    """Run children concurrently, each in its own process group, under shared watchdogs.

    ``pass_fds[k]`` are descriptors child ``k`` inherits (the parent's CPU slot lock).

    Causes (first wins, all children are then stopped): ``child_failed`` (a nonzero exit
    while a sibling runs), ``wall_timeout`` (the stage cap, a
    deadline stop), ``rss_limit`` (any child's group RSS), ``available_memory_floor``,
    ``heartbeat_stale`` and ``watchdog_error``/``interrupted``. ``None`` means every child
    exited on its own; return codes are reported per child.
    """
    import psutil

    started = clock()
    children: List[subprocess.Popen] = []
    streams = []
    cause = None
    peaks = [0 for _ in commands]
    min_available = None
    interrupt: BaseException | None = None
    try:
        inherited = list(pass_fds) if pass_fds is not None else [() for _ in commands]
        require(len(inherited) == len(commands), "one pass_fds entry per child")
        for command, log, fds in zip(commands, logs, inherited):
            stream = Path(log).open("x", encoding="utf-8")
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(fds),
                )
            )
        while any(child.poll() is None for child in children):
            elapsed = clock() - started
            available = int(psutil.virtual_memory().available)
            min_available = available if min_available is None else min(min_available, available)
            for index, child in enumerate(children):
                peaks[index] = max(peaks[index], _group_rss(psutil, child.pid))
            stale = [
                beat
                for beat, child in zip(heartbeats, children)
                if beat is not None
                and child.poll() is None
                and time.time() - (Path(beat).stat().st_mtime if Path(beat).exists() else 0)
                > heartbeat_stale_seconds
                and elapsed > heartbeat_stale_seconds
            ]
            failed = any(child.poll() not in (None, 0) for child in children)
            if failed and any(child.poll() is None for child in children):
                cause = "child_failed"
            elif elapsed > wall_seconds:
                cause = "wall_timeout"
            elif max(peaks) > rss_limit_bytes:
                cause = "rss_limit"
            elif available < available_floor_bytes:
                cause = "available_memory_floor"
            elif stale:
                cause = "heartbeat_stale"
            if cause is not None:
                break
            time.sleep(poll_seconds)
    except Interrupted as exc:
        cause, interrupt = f"interrupted: {exc}", exc
    except BaseException as exc:  # noqa: BLE001 - children are still stopped below
        cause = f"watchdog_error: {type(exc).__name__}: {exc}"
    finally:
        terminations = []
        for child in children:
            if child.poll() is None or _group_alive(child.pid):
                cause = cause or "process_group_stragglers"
                terminations.append(_stop_group(child, grace_seconds))
            else:
                terminations.append("natural_exit")
        for stream in streams:
            stream.close()
    result = {
        "cause": cause,
        "elapsed_seconds": clock() - started,
        "wall_seconds": wall_seconds,
        "children": [
            {
                "command": list(command),
                "returncode": child.returncode,
                "termination": termination,
                "confirmed_exit": child.returncode is not None and not _group_alive(child.pid),
                "peak_group_rss_bytes": peak,
            }
            for command, child, termination, peak in zip(commands, children, terminations, peaks)
        ],
        "min_available_bytes": min_available,
        "rss_limit_bytes": rss_limit_bytes,
        "available_floor_bytes": available_floor_bytes,
    }
    if interrupt is not None:
        interrupt.supervision = result  # type: ignore[attr-defined]
        raise interrupt
    return result


def clean_supervision(result: Mapping[str, Any]) -> bool:
    """Every child exited naturally with code 0 and no watchdog fired."""
    return result["cause"] is None and all(
        child["termination"] == "natural_exit"
        and child["returncode"] == 0
        and child["confirmed_exit"]
        for child in result["children"]
    )


# ---------------------------------------------------------------- stage evidence and analysis


def collect_stage(stage_dir: Path, stage: str, rows: Sequence[Mapping[str, Any]]) -> Dict:
    """Read both shard reports and every record they list; fail closed on integrity.

    Returns ``{"complete", "entries" (episode_id -> envelope), "missing", "stopped"}``.
    Raises :class:`StrictRunError` on a missing report, an unplanned or extra record, or
    a record whose bytes differ from its shard report.
    """
    planned = [e["episode_id"] for unit in stage_plan(stage, rows) for e in unit]
    reports = []
    for shard in range(WORKERS):
        path = Path(stage_dir) / f"shard-{shard}" / "report.json"
        require(path.is_file(), f"{stage} shard {shard} report missing")
        reports.append(read_json(path))
    shard_planned = [i for report in reports for i in report["planned_episode_ids"]]
    require(sorted(shard_planned) == sorted(planned), f"{stage} shard plans differ from plan")
    listed: Dict[str, str] = {}
    for report in reports:
        for episode_id, sha in report["records_sha256"].items():
            require(episode_id in report["planned_episode_ids"], f"unplanned record {episode_id}")
            listed[episode_id] = sha
    records_dir = Path(stage_dir) / "records"
    on_disk = {p.stem for p in records_dir.glob("*.json")} if records_dir.is_dir() else set()
    require(on_disk == set(listed), f"{stage} records on disk differ from shard reports")
    entries = {}
    for episode_id, sha in listed.items():
        path = records_dir / f"{episode_id}.json"
        require(dev_screen.sha256_file(path) == sha, f"record bytes changed: {episode_id}")
        entries[episode_id] = read_json(path)
    missing = [i for i in planned if i not in listed]
    return {
        "complete": not missing and all(r["complete"] for r in reports),
        "entries": entries,
        "missing": missing,
        "stopped": [r["stopped_reason"] for r in reports if r["stopped_reason"]],
        "planned": len(planned),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def calibration_reference(
    entries: Mapping[str, Mapping[str, Any]], mixes: Sequence[str] = MIXES
) -> Dict[str, Any]:
    """Dev incumbent means -> absolute NI margin and survival bands (protocol.md)."""
    per_mix = {}
    for mix in mixes:
        records = [
            e["record"] for e in entries.values() if e["mix"] == mix and e["arm"] == "incumbent"
        ]
        require(len(records) >= 2, f"calibration needs >= 2 incumbent worlds for {mix}")
        per_mix[mix] = {
            "n": len(records),
            "mean_mass_integral": _mean([r["mass_integral"] for r in records]),
            "mean_survival_fraction": _mean([r["survival_fraction"] for r in records]),
        }
    scripted = per_mix["scripted"]["mean_mass_integral"] if "scripted" in per_mix else None
    delta_ni = NI_FRACTION * scripted if scripted is not None else None
    return {
        "per_mix": per_mix,
        "absolute_delta_ni": delta_ni,
        "ni_rule": f"{NI_FRACTION} x dev incumbent scripted mean mass_integral",
        "survival_bands": {
            mix: band_node(mix, per_mix[mix]["mean_survival_fraction"]) for mix in mixes
        },
    }


def band_node(mix: str, reference_mean: float) -> Dict[str, Any]:
    """One behavioral band: candidate mean survival_fraction in [ref - 0.02, ref + 1]."""
    return {
        "metric": "survival_fraction",
        "mix_scope": [mix],
        "reference_mean": reference_mean,
        "lower_offset": -BAND_BELOW,
        "upper_offset": BAND_ABOVE,
        "lower": reference_mean - BAND_BELOW,
        "upper": reference_mean + BAND_ABOVE,
    }


def paired_final(
    entries: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Per-mix candidate-minus-incumbent mass deltas in roster (world-index) order."""
    deltas: Dict[str, List[float]] = {}
    survival: Dict[str, List[float]] = {}
    for row in rows:
        mix, seed = row["mix"], int(row["world_seed"])
        incumbent = entries[f"final-incumbent-{mix}-{seed}"]["record"]
        candidate = entries[f"final-candidate-{mix}-{seed}"]["record"]
        deltas.setdefault(mix, []).append(candidate["mass_integral"] - incumbent["mass_integral"])
        survival.setdefault(mix, []).append(candidate["survival_fraction"])
    return {"deltas_by_mix": deltas, "candidate_survival": survival}


def final_decision(
    paired: Mapping[str, Any], calibration: Mapping[str, Any], mixes: Sequence[str] = MIXES
) -> Dict[str, Any]:
    """Frozen strict reducer plus behavioral bands; ``passes`` is the STRICT_PASS test."""
    from src.scripts.eval_stats import strict_promotion_decision

    deltas = {mix: list(paired["deltas_by_mix"][mix]) for mix in mixes}
    decision = strict_promotion_decision(
        deltas, scripted_mix="scripted", absolute_delta_ni=calibration["absolute_delta_ni"]
    )
    bands = {}
    for mix in mixes:
        node = dict(calibration["survival_bands"][mix])
        value = _mean(paired["candidate_survival"][mix])
        node.update({"value": value, "passes": node["lower"] <= value <= node["upper"]})
        bands[mix] = node
    bands_pass = all(row["passes"] for row in bands.values())
    return {
        "strict_promotion_decision": decision,
        "paired_deltas_by_mix": deltas,
        "behavioral_bands": bands,
        "bands_pass": bands_pass,
        "valid": bool(decision["valid"]),
        "passes": bool(decision["valid"] and decision["passes"] and bands_pass),
        "mean_delta_by_mix": {mix: _mean(deltas[mix]) for mix in mixes},
    }


def classify_outcome(
    *,
    feasible: bool,
    deadline_stop: bool,
    failure: str | None,
    audit_passed: bool | None,
    decision: Mapping[str, Any] | None,
    decisions_agree: bool | None,
) -> str:
    """Map run facts to one pre-registered outcome (protocol.md "Outcomes")."""
    if not feasible:
        return "STOP_INFEASIBLE"
    if deadline_stop:
        return "INCOMPLETE"
    if failure is not None or audit_passed is not True or decisions_agree is not True:
        return "INVALID_STOP"
    if decision is None or not decision.get("valid"):
        return "INVALID_STOP"
    return "STRICT_PASS" if decision.get("passes") else "STRICT_FAIL"


# ---------------------------------------------------------------- audit (producer self-check child)
# The audit shares record-shape code (collect_stage, strict_promotion's record validator)
# but recomputes calibration and the decision in its own code, and cross-checks the frozen
# reducer against a second plain-mean / integrated-t implementation (governance rule 3).


def _t_pdf(x: float, df: float) -> float:
    log_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return math.exp(log_norm - (df + 1) / 2 * math.log1p(x * x / df))


def audit_t_sf(t: float, df: float, steps: int = 20000) -> float:
    """Upper-tail Student-t probability by composite Simpson integration of the density."""
    if t < 0:
        return 1.0 - audit_t_sf(-t, df, steps)
    upper = min(t, 60.0)
    h = upper / steps
    total = _t_pdf(0.0, df) + _t_pdf(upper, df)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * _t_pdf(i * h, df)
    return max(0.0, 0.5 - total * h / 3)


def audit_t_isf(p: float, df: float) -> float:
    """Upper quantile by bisection on :func:`audit_t_sf`."""
    low, high = 0.0, 60.0
    for _ in range(80):
        mid = (low + high) / 2
        if audit_t_sf(mid, df, 4000) > p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def audit_one_sided(deltas: Sequence[float]) -> Dict[str, float]:
    """Plain mean, sample SD, SE and one-sided p (H1: mean > 0) for paired deltas."""
    n = len(deltas)
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    p = (0.0 if mean > 0 else 1.0 if mean < 0 else 0.5) if se == 0 else audit_t_sf(mean / se, n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "p": p}


def audit_decision(
    deltas_by_mix: Mapping[str, Sequence[float]], delta_ni: float, alpha: float = ALPHA
) -> Dict[str, Any]:
    """Second implementation of the strict rule (Holm >= 2 of 3, scripted NI)."""
    tests = {mix: audit_one_sided(list(values)) for mix, values in deltas_by_mix.items()}
    order = sorted(tests, key=lambda mix: tests[mix]["p"])
    rejected, still = {}, True
    for rank, mix in enumerate(order):
        still = still and tests[mix]["p"] <= alpha / (len(order) - rank)
        rejected[mix] = still
    scripted = tests["scripted"]
    lower = scripted["mean"] - audit_t_isf(alpha, scripted["n"] - 1) * scripted["se"]
    superiority = sum(rejected.values()) >= 2
    return {
        "tests": tests,
        "holm_rejected": rejected,
        "superiority_passes": superiority,
        "scripted_lower_bound": lower,
        "noninferiority_passes": lower > -delta_ni,
        "passes": superiority and lower > -delta_ni,
    }


def cross_check(frozen: Mapping[str, Any], second: Mapping[str, Any], delta_ni: float) -> Dict:
    """Agreement of the frozen reducer and the audit's second implementation."""
    problems = []
    sup = frozen["superiority"]
    for mix, test in second["tests"].items():
        mine = sup["per_mix"][mix]
        if not math.isclose(mine["mean_delta"], test["mean"], rel_tol=0, abs_tol=1e-9):
            problems.append(f"mean differs for {mix}")
        if mine["p_value"] is None or abs(mine["p_value"] - test["p"]) > 1e-6:
            problems.append(f"p-value differs for {mix}")
    ni = frozen["scripted_noninferiority"]
    if ni["lower_bound"] is None or abs(ni["lower_bound"] - second["scripted_lower_bound"]) > 1e-6:
        problems.append("scripted lower bound differs")
    near_boundary = abs(second["scripted_lower_bound"] + delta_ni) < 1e-6 or any(
        abs(t["p"] - ALPHA / k) < 1e-6 for t in second["tests"].values() for k in (1, 2, 3)
    )
    if frozen["passes"] != second["passes"] and not near_boundary:
        problems.append("pass/fail differs")
    return {"agree": not problems, "problems": problems, "near_boundary": near_boundary}


def audit_stage_entries(
    stage: str,
    entries: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None,
    wrapper: Mapping[str, Any],
    bank_index: Mapping[int, int],
) -> Dict[str, Any]:
    """Envelope, world identity and record-shape rules for one stage's records.

    ``bank_index`` maps each seed to its index in the stage's namespace bank.

    ``profile`` is ``{"descriptor", "digest"}``; ``None`` (smoke, legacy records) checks
    only the envelope and wrapper presence. Candidate records must carry the wrapper probe
    with the intent's descriptor; incumbent records must not carry it.
    """
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        validate_strict_world_record,
    )

    units = {e["episode_id"]: e for unit in stage_plan(stage, rows) for e in unit}
    by_key = {(row["mix"], int(row["world_seed"])): row for row in rows}
    failures: List[str] = []
    veto_match = veto_total = 0
    for episode_id, entry in sorted(entries.items()):
        unit = units.get(episode_id)
        if unit is None:
            failures.append(f"{episode_id}: not in the stage plan")
            continue
        row = by_key[(unit["mix"], unit["world_seed"])]
        expected = {
            "schema_version": SCHEMA,
            "study_id": STUDY_ID,
            "episode_id": episode_id,
            "stage": stage,
            "arm": unit["arm"],
            "mix": unit["mix"],
            "world_seed": unit["world_seed"],
            "hero_sha256": CHAMPION[1],
            "world_index": bank_index.get(unit["world_seed"]),
            "safety_veto": unit["arm"] == "candidate",
            "wrapper": wrapper["method"] if unit["arm"] == "candidate" else None,
            "wrapper_source_sha256": (
                wrapper["source_sha256"] if unit["arm"] == "candidate" else None
            ),
            "roster_member_sha256s": [slot["member_sha256"] for slot in row["slots"]],
        }
        wrong = sorted(key for key, value in expected.items() if entry.get(key) != value)
        if wrong:
            failures.append(f"{episode_id}: envelope fields differ {wrong}")
        record = entry.get("record", {})
        probes = record.get("probes", {}) if isinstance(record, Mapping) else {}
        candidate = unit["arm"] == "candidate"
        if profile is None:
            if candidate != ("safety_veto" in probes):
                failures.append(f"{episode_id}: wrapper probe presence wrong for {unit['arm']}")
            elif candidate and probes["safety_veto"].get("method") != wrapper["method"]:
                failures.append(f"{episode_id}: wrapper method differs")
        else:
            try:
                validate_strict_world_record(
                    record,
                    profile,
                    row,
                    candidate_wrapper=wrapper["descriptor"] if candidate else None,
                )
            except StrictPromotionArtifactError as exc:
                failures.append(f"{episode_id}: record shape: {exc}")
        if candidate and "safety_veto" in probes:
            veto_total += 1
            decisions = probes["safety_veto"].get("counters", {}).get("decisions")
            veto_match += int(decisions == record.get("denominators", {}).get("decision_frames"))
    return {
        "stage": stage,
        "records": len(entries),
        "failures": failures,
        "reported_not_gated": {
            "candidate_records_with_veto": veto_total,
            "veto_decisions_equal_decision_frames": veto_match,
        },
    }


def audit_calibration(entries: Mapping[str, Mapping[str, Any]], saved: Mapping[str, Any]) -> List:
    """Recompute dev means, margin and bands with plain arithmetic; list disagreements."""
    problems = []
    for mix, row in saved["per_mix"].items():
        mass = [e["record"]["mass_integral"] for e in entries.values() if e["mix"] == mix]
        surv = [e["record"]["survival_fraction"] for e in entries.values() if e["mix"] == mix]
        if len(mass) != row["n"] or not math.isclose(
            sum(mass) / len(mass), row["mean_mass_integral"], rel_tol=0, abs_tol=1e-9
        ):
            problems.append(f"calibration mass mean differs for {mix}")
        mean_surv = sum(surv) / len(surv)
        band = saved["survival_bands"][mix]
        if not (
            math.isclose(mean_surv, row["mean_survival_fraction"], rel_tol=0, abs_tol=1e-12)
            and math.isclose(band["lower"], mean_surv - BAND_BELOW, rel_tol=0, abs_tol=1e-12)
            and math.isclose(band["upper"], mean_surv + BAND_ABOVE, rel_tol=0, abs_tol=1e-12)
        ):
            problems.append(f"calibration survival mean or band differs for {mix}")
    scripted = [e["record"]["mass_integral"] for e in entries.values() if e["mix"] == "scripted"]
    margin = NI_FRACTION * sum(scripted) / len(scripted)
    if not math.isclose(margin, saved["absolute_delta_ni"], rel_tol=0, abs_tol=1e-12):
        problems.append("absolute_delta_ni differs")
    if not margin > 0:
        problems.append("absolute_delta_ni is not positive")
    return problems


def audit_final(
    entries: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independent pairing, delta, decision and band recomputation for the final stage."""
    from src.scripts.eval_stats import strict_promotion_decision

    problems: List[str] = []
    deltas: Dict[str, List[float]] = {mix: [] for mix in MIXES}
    survival: Dict[str, List[float]] = {mix: [] for mix in MIXES}
    for row in rows:
        key = f"{row['mix']}-{row['world_seed']}"
        pair = [entries.get(f"final-{arm}-{key}") for arm in ARMS]
        if any(item is None for item in pair):
            problems.append(f"unpaired final world {key}")
            continue
        incumbent, candidate = (item["record"] for item in pair)
        deltas[row["mix"]].append(candidate["mass_integral"] - incumbent["mass_integral"])
        survival[row["mix"]].append(candidate["survival_fraction"])
    if problems:
        return {"problems": problems}
    margin = calibration["absolute_delta_ni"]
    frozen = strict_promotion_decision(deltas, scripted_mix="scripted", absolute_delta_ni=margin)
    second = audit_decision(deltas, margin)
    agreement = cross_check(frozen, second, margin)
    problems.extend(agreement["problems"])
    bands = {}
    for mix in MIXES:
        value = sum(survival[mix]) / len(survival[mix])
        band = calibration["survival_bands"][mix]
        bands[mix] = band["lower"] <= value <= band["upper"]
    passes = bool(frozen["valid"] and frozen["passes"] and all(bands.values()))
    if passes != producer.get("passes"):
        problems.append("audit pass/fail differs from the producer decision")
    for mix in MIXES:
        mine = sum(deltas[mix]) / len(deltas[mix])
        theirs = producer.get("mean_delta_by_mix", {}).get(mix)
        if theirs is None or not math.isclose(mine, theirs, rel_tol=0, abs_tol=1e-9):
            problems.append(f"producer mean delta differs for {mix}")
    return {
        "problems": problems,
        "n_per_mix": {mix: len(values) for mix, values in deltas.items()},
        "second_implementation": second,
        "cross_check": agreement,
        "bands_inside": bands,
        "valid": bool(frozen["valid"]),
        "passes": passes,
    }


def audit_main(intent_path: Path, out_dir: Path) -> int:
    """Internal: audit the saved run evidence; writes ``out_dir/report.json`` once."""
    intent = read_json(intent_path)
    output = Path(intent["output_root"]) / "output"
    failures: List[str] = []
    checks: Dict[str, Any] = {}
    started = read_json(output / "started.json")
    admitted = read_json(output / "admitted.json")
    if started["intent_sha256"] != dev_screen.sha256_file(intent_path):
        failures.append("intent bytes differ from the admitted intent")
    drift = closure_drift(intent["source_closure"])
    if drift:
        failures.append(f"source closure drift: {drift}")
    for sha, path in admitted["checkpoint_snapshots"].items():
        if dev_screen.sha256_file(Path(path)) != sha:
            failures.append(f"checkpoint snapshot changed: {path}")
    rebuilt = intent_rosters(intent)
    rosters_path = output / "rosters.json"
    rosters = read_json(rosters_path)
    if dev_screen.canonical_json(rebuilt) != dev_screen.canonical_json(rosters):
        failures.append("saved rosters differ from the rebuilt rosters")
    if dev_screen.sha256_file(rosters_path) != admitted["rosters_sha256"]:
        failures.append("rosters.json bytes changed after admission")
    profile = None
    if intent["smoke_frames"] is None:
        from src.core.config_loader import load_and_initialize_config
        from src.scripts.tournament_eval import evaluation_profile_for_name

        load_and_initialize_config(intent["config"]["path"])
        resolved = evaluation_profile_for_name(PROFILE_NAME, HORIZON)
        if resolved.digest != intent["profile"]["digest"]:
            failures.append("resolved profile digest differs from the intent")
        profile = {"descriptor": resolved.descriptor(), "digest": resolved.digest}
    wrapper = intent["candidate"]["wrapper_identity"]
    stage_entries: Dict[str, Mapping[str, Any]] = {}
    for stage in STAGES:
        rows = rosters[stage]
        if not rows:
            checks[stage] = {"skipped": "no planned episodes"}
            continue
        try:
            collected = collect_stage(output / stage, stage, rows)
        except (StrictRunError, OSError, KeyError, ValueError) as exc:
            failures.append(f"{stage}: {exc}")
            continue
        if not collected["complete"]:
            failures.append(f"{stage}: incomplete ({len(collected['missing'])} missing)")
        shape = audit_stage_entries(
            stage, collected["entries"], rows, profile, wrapper, world_index_maps(intent)[stage]
        )
        failures.extend(shape["failures"])
        checks[stage] = {
            "records": shape["records"],
            "planned": collected["planned"],
            "reported_not_gated": shape["reported_not_gated"],
        }
        stage_entries[stage] = collected["entries"]
    recomputed = None
    if intent["smoke_frames"] is None and len(stage_entries) == len(STAGES):
        calibration = read_json(output / "calibration.json")
        failures.extend(audit_calibration(stage_entries["calibration"], calibration))
        producer = read_json(output / "decision.json")
        recomputed = audit_final(stage_entries["final"], rosters["final"], calibration, producer)
        failures.extend(recomputed["problems"])
        if checks["serving"]["records"] != len(rosters["serving"]):
            failures.append("serving stage does not have one record per serving world")
    elif intent["smoke_frames"] is None:
        failures.append("not every numeric stage produced evidence")
    report = {
        "schema_version": SCHEMA,
        "role": "producer self-check child (strict_audit.py is the independent audit)",
        "intent_sha256": started["intent_sha256"],
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "recomputed_final": recomputed,
        "utc": now_utc().isoformat(),
    }
    write_durable(Path(out_dir) / "report.json", dev_screen.json_safe(report))
    return 0


# ---------------------------------------------------------------- run (parent)


def validate_intent(intent: Mapping[str, Any]) -> None:
    """Reject a malformed, drifted or out-of-envelope intent before any child starts."""
    require(intent["schema_version"] == SCHEMA and intent["study_id"] == STUDY_ID, "schema")
    require(intent["caps"]["stage_seconds"] == CAPS, "stage caps differ from protocol")
    require(intent["caps"]["handoff_seconds"] == HANDOFF_SECONDS, "handoff reserve")
    require(intent["caps"]["retry_authorized"] is False, "retry must not be authorized")
    require(intent["caps"]["workers"] == WORKERS, "worker count")
    check_output_root(Path(intent["output_root"]), intent["smoke_frames"] is not None)
    parse_utc(intent["deadline_utc"])
    drift = closure_drift(intent["source_closure"])
    require(not drift, f"source closure drift: {drift}")
    require(dev_screen.sha256_file(PROTOCOL) == intent["protocol"]["sha256"], "protocol drift")
    require(intent["candidate"]["wrapper_identity"] == wrapper_identity(), "wrapper identity drift")
    if intent["smoke_frames"] is None:
        require(intent["sizing"]["final_worlds_per_mix"] >= N_FLOOR, "N below floor")
        require(
            len(intent["final_seeds"]) == min(intent["sizing"]["final_worlds_per_mix"], N_MAX),
            "final seed prefix",
        )


def _child_command(intent: Mapping[str, Any], intent_path: Path, *extra: str) -> List[str]:
    return [
        intent["python"],
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        *extra,
        "--intent",
        str(intent_path),
    ]


def _tree_hashes(directory: Path) -> Dict[str, str]:
    return {
        str(path): dev_screen.sha256_file(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.name != "heartbeat.json"
    }


def run_stages(
    intent: Mapping[str, Any],
    intent_path: Path,
    output: Path,
    state: Dict,
    slots: Sequence[Any],
) -> None:
    """Calibration -> final -> serving -> audit; records facts into ``state``.

    ``slots`` are the CPU slot locks the run parent holds; worker ``k`` inherits ``slots[k]``.
    """
    smoke = intent["smoke_frames"] is not None
    supervisor = load_supervisor(
        Path(intent["supervisor_source"]["path"]), intent["supervisor_source"]["sha256"]
    )
    if not smoke:
        require(Path(intent["slot_lock_root"]) == Path(supervisor.GLOBAL_LOCK_ROOT), "lock root")
    capacity = supervisor.capacity_environment(2)
    os.environ.update(capacity)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(capacity)
    env["SNAKE_DQN_DEVICE"] = "cpu"
    hardware = supervisor.capacity_preflight(AVAILABLE_GIB)
    snapshots = dev_screen.snapshot_checkpoints(Path(intent["checkpoint_dir"]), output)
    rosters = intent_rosters(intent)
    rosters_sha = write_durable(output / "rosters.json", rosters)
    write_durable(
        output / "admitted.json",
        {
            "utc": now_utc().isoformat(),
            "hardware": hardware,
            "rosters_sha256": rosters_sha,
            "checkpoint_snapshots": snapshots,
            "intent_sha256": state["intent_sha256"],
        },
    )
    prior = _tree_hashes(output)
    deadline = parse_utc(intent["deadline_utc"])
    calibration = None
    for stage in STAGES:
        require(_tree_hashes(output) == prior, f"output drift before {stage}")
        require(not closure_drift(intent["source_closure"]), f"source drift before {stage}")
        rows = rosters[stage]
        if not rows:
            state["stages"][stage] = {"status": "skipped", "reason": "no planned episodes"}
            continue
        remaining = (deadline - now_utc()).total_seconds()
        if not smoke and remaining < required_seconds(stage):
            state["deadline_stop"] = f"{stage}: {remaining:.0f}s left < caps + handoff"
            return
        stage_dir = output / stage
        (stage_dir / "records").mkdir(parents=True)
        worker_deadline = now_utc() + timedelta(seconds=CAPS[stage] - WORKER_STOP_MARGIN_SECONDS)
        commands = [
            _child_command(
                intent,
                intent_path,
                "worker",
                "--stage",
                stage,
                "--shard",
                str(k),
                "--stage-deadline",
                worker_deadline.isoformat(),
                "--slot-fd",
                str(slots[k].fileno()),
            )
            for k in range(WORKERS)
        ]
        write_durable(
            stage_dir / "started.json",
            {"utc": now_utc().isoformat(), "commands": commands, "cap": CAPS[stage]},
        )
        result = supervise_children(
            commands,
            cwd=Path(intent["repo"]),
            env=env,
            logs=[stage_dir / f"worker-{k}.log" for k in range(WORKERS)],
            heartbeats=[stage_dir / f"shard-{k}" / "heartbeat.json" for k in range(WORKERS)],
            wall_seconds=CAPS[stage],
            pass_fds=[(slots[k].fileno(),) for k in range(WORKERS)],
        )
        write_durable(stage_dir / "supervision.json", result)
        if result["cause"] == "wall_timeout":
            state["deadline_stop"] = f"{stage}: wall cap {CAPS[stage]} s reached"
            return
        require(clean_supervision(result), f"{stage} supervision: {result['cause']}")
        collected = collect_stage(stage_dir, stage, rows)
        if not collected["complete"]:
            require(
                all(s.startswith("deadline") for s in collected["stopped"])
                and collected["stopped"],
                f"{stage} incomplete without a deadline stop",
            )
            state["deadline_stop"] = f"{stage}: {collected['stopped']}"
            return
        state["stages"][stage] = {"status": "complete", "episodes": len(collected["entries"])}
        if stage == "calibration" and not smoke:
            calibration = calibration_reference(collected["entries"])
            require(calibration["absolute_delta_ni"] > 0, "non-positive NI margin")
            write_durable(output / "calibration.json", calibration)
        if stage == "final" and not smoke:
            paired = paired_final(collected["entries"], rows)
            state["decision"] = dev_screen.json_safe(final_decision(paired, calibration))
            write_durable(output / "decision.json", state["decision"])
        write_durable(
            stage_dir / "completed.json", {"utc": now_utc().isoformat(), **state["stages"][stage]}
        )
        prior = _tree_hashes(output)
    run_audit(intent, intent_path, output, env, state, deadline, smoke)


def producer_claim(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The producer's pre-audit outcome claim (the independent audit verifies it)."""
    decision = state.get("decision")
    if decision is None or not decision.get("valid"):
        return {"outcome": "INVALID_STOP", "stop_reason": "no valid final decision"}
    return {"outcome": "STRICT_PASS" if decision.get("passes") else "STRICT_FAIL"}


def _audit_child(
    command: Sequence[str], out_dir: Path, report_name: str, env: Mapping, cwd: Path, cap: float
) -> Dict[str, Any]:
    """Run one audit child; exit 0 = PASS, 1 = FAIL (report written), else harness error."""
    out_dir.mkdir()
    result = supervise_children(
        [list(command)],
        cwd=cwd,
        env=env,
        logs=[out_dir / "child.log"],
        heartbeats=[None],
        wall_seconds=cap,
    )
    write_durable(out_dir / "supervision.json", result)
    child = result["children"][0] if result["children"] else {}
    report_path = out_dir / report_name
    usable = (
        result["cause"] is None
        and child.get("termination") == "natural_exit"
        and child.get("confirmed_exit") is True
        and child.get("returncode") in (0, 1)
        and report_path.is_file()
    )
    return {
        "cause": result["cause"],
        "usable": usable,
        "returncode": child.get("returncode"),
        "report": read_json(report_path) if usable else None,
        "report_sha256": dev_screen.sha256_file(report_path) if usable else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_audit(intent, intent_path, output, env, state, deadline, smoke) -> None:
    """Independent audit child (strict_audit.py), then the producer self-check child.

    Both share the audit cap and both gate (a smoke dry-run runs the audit with ``--smoke``).
    The independent audit reads the saved evidence, including the producer's pre-audit
    ``producer-outcome.json`` claim (not written on smoke); the self-check (this module's
    ``audit`` subcommand) runs after it so its report is not part of the audited evidence.
    """
    remaining = (deadline - now_utc()).total_seconds()
    if not smoke and remaining < required_seconds("audit"):
        state["deadline_stop"] = f"audit: {remaining:.0f}s left < cap + handoff"
        return
    began = time.monotonic()
    cwd = Path(intent["repo"])
    if not smoke:
        state["producer_claim"] = producer_claim(state)
        write_durable(
            output / "producer-outcome.json",
            {
                **state["producer_claim"],
                "role": "producer claim before the independent audit; closeout.json is final",
            },
        )
    # Smoke runs the same child with --smoke (governance rule 5: dry-run before GO).
    independent = _audit_child(
        intent["audit"]["independent_command"],
        output / "audit",
        "audit.json",
        env,
        cwd,
        CAPS["audit"],
    )
    if independent["cause"] == "wall_timeout":
        state["deadline_stop"] = "independent audit: wall cap reached"
        return
    require(independent["usable"], f"independent audit unusable: {independent['cause']}")
    independent_passed = (
        independent["returncode"] == 0 and independent["report"]["status"] == "PASS"
    )
    state["audit_report_sha256"] = independent["report_sha256"]
    state["independent_audit_failures"] = [
        row.get("rule") for row in independent["report"].get("failures", [])
    ] or independent["report"].get("error")
    left = CAPS["audit"] - (time.monotonic() - began)
    require(left > 0, "audit cap exhausted before the self-check")
    command = _child_command(intent, intent_path, "audit", "--out", str(output / "self-check"))
    check = _audit_child(command, output / "self-check", "report.json", env, cwd, left)
    if check["cause"] == "wall_timeout":
        state["deadline_stop"] = "self-check: audit wall cap reached"
        return
    require(check["usable"], f"self-check unusable: {check['cause']}")
    report = check["report"]
    state["self_check_report_sha256"] = check["report_sha256"]
    state["audit_failures"] = report["failures"]
    state["audit_passed"] = bool(independent_passed and report["passed"])
    recomputed = report.get("recomputed_final")
    decision = state.get("decision")
    state["decisions_agree"] = bool(
        smoke
        or (
            recomputed is not None
            and decision is not None
            and recomputed.get("passes") == decision.get("passes")
            and not recomputed.get("problems")
        )
    )


def closeout(intent: Mapping[str, Any], output: Path, state: Mapping[str, Any]) -> Dict:
    """Write ``closeout.json`` (and ``receipt.json`` on STRICT_PASS) exactly once."""
    smoke = intent["smoke_frames"] is not None
    failure = state.get("failure")
    deadline_stop = state.get("deadline_stop")
    if smoke:
        outcome = (
            "INVALID_STOP"
            if failure or state.get("audit_passed") is not True
            else "INCOMPLETE" if deadline_stop else "SMOKE_NO_DECISION"
        )
    else:
        outcome = classify_outcome(
            feasible=bool(intent["sizing"]["feasible"]),
            deadline_stop=bool(deadline_stop),
            failure=failure,
            audit_passed=state.get("audit_passed"),
            decision=state.get("decision"),
            decisions_agree=state.get("decisions_agree"),
        )
    remaining = (parse_utc(intent["deadline_utc"]) - now_utc()).total_seconds()
    record = {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "attempt_id": intent["attempt_id"],
        "intent_sha256": state["intent_sha256"],
        "outcome": outcome,
        "precedence": "closeout.json takes precedence over every stage record",
        "stages": state["stages"],
        "deadline_stop": deadline_stop,
        "failure": failure,
        "audit_passed": state.get("audit_passed"),
        "audit_report_sha256": state.get("audit_report_sha256"),
        "audit_failures": state.get("audit_failures"),
        "independent_audit_failures": state.get("independent_audit_failures"),
        "self_check_report_sha256": state.get("self_check_report_sha256"),
        "producer_claim": state.get("producer_claim"),
        "decisions_agree": state.get("decisions_agree"),
        "decision": state.get("decision") if outcome in ("STRICT_PASS", "STRICT_FAIL") else None,
        "final_worlds_per_mix": intent["sizing"]["final_worlds_per_mix"],
        "remaining_seconds_at_closeout": remaining,
        "handoff_reserve_met": remaining >= HANDOFF_SECONDS,
        "serving_path_qualified": False,
        "serving_stage_kind": SERVING_STAGE_KIND,
        "serving_stage_exercises_web_path": False,
        "serving_remaining_work": SERVING_REMAINING_WORK,
        "promotion_performed": False,
        "retry_authorized": False,
        "non_claims": NON_CLAIMS,
        "utc": now_utc().isoformat(),
    }
    if outcome == "STRICT_PASS":
        record["receipt_sha256"] = write_durable(
            output / "receipt.json",
            {
                "schema_version": SCHEMA,
                "receipt": "strict-pass-receipt-only",
                "candidate": intent["candidate"],
                "incumbent": intent["incumbent"],
                "profile": intent["profile"],
                "intent_sha256": state["intent_sha256"],
                "audit_report_sha256": state["audit_report_sha256"],
                "decision": state["decision"],
                "serving_path_qualified": False,
                "serving_stage_kind": SERVING_STAGE_KIND,
                "serving_stage_exercises_web_path": False,
                "serving_remaining_work": SERVING_REMAINING_WORK,
                "non_claims": NON_CLAIMS,
            },
        )
    write_durable(output / "closeout.json", dev_screen.json_safe(record))
    return record


def run(intent_path: Path) -> Dict[str, Any]:
    """Admit the intent once and close out with exactly one outcome. Never retries."""
    intent_path = Path(intent_path).resolve()
    intent = read_json(intent_path)
    validate_intent(intent)
    output = Path(intent["output_root"]) / "output"
    require(intent_path == Path(intent["output_root"]) / "intent.json", "intent location")
    slots: List[Any] = []
    if intent["sizing"]["feasible"]:
        # A transient RAM shortage or a busy CPU slot raises here and leaves no record, so it
        # is not an attempt. The slots stay held by this process until closeout.
        supervisor = load_supervisor(
            Path(intent["supervisor_source"]["path"]), intent["supervisor_source"]["sha256"]
        )
        if intent["smoke_frames"] is None:
            require(
                Path(intent["slot_lock_root"]) == Path(supervisor.GLOBAL_LOCK_ROOT), "lock root"
            )
        supervisor.capacity_preflight(AVAILABLE_GIB)
        if output.exists():  # create-only: never resumes or retries
            raise FileExistsError(str(output))
        slots = acquire_run_slots(Path(intent["slot_lock_root"]))
    try:
        return _admitted_run(intent, intent_path, output, slots)
    finally:
        release_run_slots(slots)


def _admitted_run(
    intent: Mapping[str, Any], intent_path: Path, output: Path, slots: Sequence[Any]
) -> Dict[str, Any]:
    """From ``output/started.json`` on, every stop is final (closeout is always written)."""
    output.mkdir(exist_ok=False)
    state: Dict[str, Any] = {
        "intent_sha256": dev_screen.sha256_file(intent_path),
        "stages": {},
    }
    write_durable(
        output / "started.json",
        {
            "utc": now_utc().isoformat(),
            "pid": os.getpid(),
            "intent_sha256": state["intent_sha256"],
            "cpu_slot_locks_held": [str(handle.name) for handle in slots],
        },
    )

    def interrupt(signum: int, frame: Any) -> None:
        raise Interrupted(signal.Signals(signum).name)

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        if intent["sizing"]["feasible"]:
            run_stages(intent, intent_path, output, state, slots)
    except (Exception, Interrupted) as exc:  # noqa: BLE001 - recorded, never retried
        state["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return closeout(intent, output, state)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="write the intent once")
    prep.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=f"default {DEFAULT_OUT_ROOT}, or {DEFAULT_SMOKE_ROOT} with --smoke-frames",
    )
    prep.add_argument("--deadline-utc", required=True)
    prep.add_argument("--authorization-quote", required=True)
    prep.add_argument("--screen-run", type=Path, default=SCREEN_RUN)
    prep.add_argument("--pilot-output", type=Path, default=dev_screen.DEFAULT_PILOT_OUTPUT)
    prep.add_argument("--slot-lock-root", type=Path, default=SLOT_LOCK_ROOT)
    prep.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help="plumbing dry-run only: 2 episodes x <= 500 frames, no decision",
    )
    for name in ("run", "worker", "audit"):
        child = sub.add_parser(name)
        child.add_argument("--intent", type=Path, required=True)
        if name == "worker":
            child.add_argument("--stage", choices=STAGES, required=True)
            child.add_argument("--shard", type=int, choices=range(WORKERS), required=True)
            child.add_argument("--stage-deadline", required=True)
            child.add_argument("--slot-fd", type=int, required=True)
        if name == "audit":
            child.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        default_root = DEFAULT_SMOKE_ROOT if args.smoke_frames is not None else DEFAULT_OUT_ROOT
        intent = build_intent(
            out_root=args.out_root or default_root,
            deadline=parse_utc(args.deadline_utc),
            authorization_quote=args.authorization_quote,
            screen_run=args.screen_run,
            pilot_output=args.pilot_output,
            slot_lock_root=args.slot_lock_root,
            smoke_frames=args.smoke_frames,
        )
        path = prepare(intent)
        print(
            json.dumps(
                {
                    "intent": str(path),
                    "sha256": dev_screen.sha256_file(path),
                    "final_worlds_per_mix": intent["sizing"]["final_worlds_per_mix"],
                    "feasible": intent["sizing"]["feasible"],
                }
            )
        )
        return 0
    if args.command == "run":
        record = run(args.intent)
        print(json.dumps({"outcome": record["outcome"]}))
        return 0
    if args.command == "worker":
        return worker_main(
            args.intent.resolve(),
            args.stage,
            args.shard,
            parse_utc(args.stage_deadline),
            args.slot_fd,
        )
    return audit_main(args.intent.resolve(), args.out)


if __name__ == "__main__":
    raise SystemExit(main())
