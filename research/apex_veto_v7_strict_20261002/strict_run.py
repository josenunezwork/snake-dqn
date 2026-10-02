#!/usr/bin/env python3
"""Tier-2 strict challenge: Apex champion + v7 space preference (lambda=4) vs champion + v5.

Pre-registration: ``protocol.md`` beside this file. Adapted (copied, then parametrized)
from ``research/apex_veto_v5_strict_20261001/strict_run.py``, whose files stay bound by the
v5 STRICT_PASS receipt and are not imported. Subcommands:

* ``prepare`` writes ``<root>/intent.json`` once: source closure and git commit, checkpoint
  sha256, both arm identities (checkpoint + wrapper descriptor + wrapper source sha256s;
  the incumbent is the released v5 veto and the modules it imports, the candidate the v7
  veto at lambda=4.0 and every veto module it imports), namespaces and their disjointness
  report, deadline, caps, authorization quote, the pre-registered design and the
  final-world count N frozen from the v7 Tier-1 screen pilot (which must have decided
  RECOMMEND_STRICT_GATE with a passing self-check, from the candidate's and incumbent's
  exact source bytes, with lambda=4.0 bound through the DEV sweep summary).
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

SCHEMA = "apex-veto-v7-strict/v1"
AUTHORITY = "tier-2-strict-promotion"
STUDY_ID = "apex-veto-v7-strict-20261002"
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.md"
MIXES = dev_screen.MIXES
HORIZON = dev_screen.HORIZON
PROFILE_NAME = dev_screen.PROFILE_NAME
CHAMPION = dev_screen.CHAMPION
SERVING_MIX = "serving-selfplay"
ARMS = ("incumbent", "candidate")
STAGES = ("calibration", "final", "serving")

# Arms: both are the champion checkpoint behind a hero-only veto. The incumbent is the
# released configuration (v5, v5 STRICT_PASS + SERVING_PASS); the candidate is v7 at the
# lambda the DEV sweep selected and the v7 Tier-1 screen ran (4.0, pinned here).
CANDIDATE_LAMBDA = 4.0
INCUMBENT_METHOD = "free-space-veto/v5-boost-aware"
CANDIDATE_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"
ARM_METHODS = {"incumbent": INCUMBENT_METHOD, "candidate": CANDIDATE_METHOD}
# A key every ``veto_diagnostics`` of the arm's kind has (v5: BoostAwareCounters.to_dict;
# v7: SpacePreferenceCounters.to_dict, which nests the v5 counters under "v5").
DIAGNOSTICS_MARKER = {"incumbent": "boost_landing_vetoes", "candidate": "rerank_changes"}
V2_SOURCE = "src/evaluation/safety_veto.py"
V3_SOURCE = "src/evaluation/safety_veto_v3.py"
V5_SOURCE = "src/evaluation/safety_veto_v5.py"
V7_SOURCE = "src/evaluation/safety_veto_v7.py"
# Released incumbent source identity (web/backend/safety_veto_serving.py and the v5 web
# serving release bind the same three files): v5 plus the v2 and v3 modules it imports.
INCUMBENT_SOURCE_SHA256S = {
    V5_SOURCE: "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
    V2_SOURCE: "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    V3_SOURCE: "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
}
INCUMBENT_SOURCE_SHA256 = INCUMBENT_SOURCE_SHA256S[V5_SOURCE]
# Repo-relative veto modules each arm executes (the primary module first). v7 imports v2,
# v3 and v5 (``from src.evaluation.safety_veto{,_v3,_v5} import ...``); v5 imports v2 and
# v3; v3 imports v2. Non-veto imports (src/game, src/core) are bound by the source closure.
ARM_SOURCES = {
    "incumbent": (V5_SOURCE, V2_SOURCE, V3_SOURCE),
    "candidate": (V7_SOURCE, V2_SOURCE, V3_SOURCE, V5_SOURCE),
}
HEX_DIGITS = "0123456789abcdef"

NAMESPACE_KEY = "worlds"
NAMESPACES = {
    "dev": ("apex-veto-v7-strict-dev-v1", 16),
    "final": ("apex-veto-v7-strict-final-v1", 300),
    "serving": ("apex-veto-v7-strict-serving-v1", 50),
}
SMOKE_DOMAIN = "apex-veto-v7-strict-smoke-v1"  # plumbing dry-runs only; never Tier-2 worlds
EXCLUSION_PREFIX = 1000  # seeds per earlier domain/purpose checked (>= any bank's size)
# Every earlier SHA-prefix domain of this recipe (name -> purposes), first 1000 seeds each:
# the v2 Tier-1 screen (also the SIMD H5000 parity worlds), all v2 strict banks (v1-v3) and
# their smoke, both web serving lanes, the v3/v4/v5/v6/v7 screens and smokes (the SIMD v5
# parity replayed v5 screen worlds), the v5 strict banks and smoke, both trap-horizon
# diagnostics, the v7 DEV lambda sweep and smoke, and this study's own smoke domain.
# dev_screen.disjointness_report adds the task-aligned challenger namespaces, the strict
# pilot's observed seeds and seeds 0..999.
WEB_PURPOSES = ("watch", "play", "parity", "worlds")
EARLIER_DOMAINS: Dict[str, Sequence[str]] = {
    dev_screen.SCREEN_DOMAIN: (dev_screen.SCREEN_NAMESPACE,),
    **{
        f"apex-veto-strict-{bank}-v{version}": ("worlds",)
        for bank in ("dev", "final", "serving")
        for version in (1, 2, 3)
    },
    "apex-veto-strict-smoke-v1": ("worlds",),
    "apex-veto-web-serving-v1": WEB_PURPOSES,
    "apex-veto-web-serving-smoke-v1": WEB_PURPOSES,
    "apex-veto-v3-screen-v1": ("worlds",),
    "apex-veto-v3-screen-v2": ("worlds",),
    "apex-veto-v3-screen-smoke-v1": ("worlds",),
    "apex-veto-v4-screen-v1": ("worlds",),
    "apex-veto-v4-screen-smoke-v1": ("worlds",),
    "apex-veto-v5-screen-v1": ("worlds",),
    "apex-veto-v5-screen-smoke-v1": ("worlds",),
    "apex-veto-v5-strict-dev-v1": ("worlds",),
    "apex-veto-v5-strict-final-v1": ("worlds",),
    "apex-veto-v5-strict-serving-v1": ("worlds",),
    "apex-veto-v5-strict-smoke-v1": ("worlds",),
    "apex-veto-v5-web-serving-v1": WEB_PURPOSES,
    "apex-veto-v5-web-serving-smoke-v1": WEB_PURPOSES,
    "apex-veto-v6-screen-v1": ("worlds",),
    "apex-veto-v6-screen-smoke-v1": ("worlds",),
    "apex-veto-v7-dev-v1": ("worlds",),
    "apex-veto-v7-dev-smoke-v1": ("worlds",),
    "apex-veto-v7-screen-v1": ("worlds",),
    "apex-veto-v7-screen-smoke-v1": ("worlds",),
    "trap-horizon-dev-v1": ("worlds",),
    "trap-horizon-dev-smoke-v1": ("worlds",),
    "trap-horizon-v5-dev-v1": ("worlds",),
    "trap-horizon-v5-dev-smoke-v1": ("worlds",),
    SMOKE_DOMAIN: (NAMESPACE_KEY,),
}

# Pilot: the v7 Tier-1 screen (arms A = champion + v5, B = champion + v7 lambda=4.0; C/D
# controls), whose lambda came from the DEV sweep summary below.
PILOT_DOMAIN = "apex-veto-v7-screen-v1"
PILOT_WORLDS = 40
PILOT_REQUIRED_DECISION = "RECOMMEND_STRICT_GATE"
PILOT_ARM_METHODS = {"A": INCUMBENT_METHOD, "B": CANDIDATE_METHOD}
PILOT_COMMIT = "900385fed868a1b35f308871d211aeafc318e4c2"  # the screen's (and sweep's) commit
SWEEP_SCHEMA = "apex-veto-v7-lambda-sweep/v1"
SWEEP_ID = "apex-veto-v7-dev-v1"
SWEEP_STATUS_SELECTED = "SELECTED"

# MDE 40 absolute per mix, declared AFTER the v7 screen finished (protocol.md "Sizing"):
# its paired B-A SD (~135-153) makes MDE 20 infeasible (N = 547 > 300).
MDE_ABSOLUTE = 40.0
MDE_BASIS = (
    "chosen after the v7 Tier-1 screen finished: its paired B-A SD (135-153 per mix) "
    "sizes MDE 20 at N = 547 > Nmax 300 (infeasible); 40 gives N = 137 (MDE 25/30/35 would "
    "give 350/243/179); the screen's observed mean effect was +135 to +154 per mix, so 40 is "
    "about a quarter of it (that effect is not evidence here)"
)
NI_FRACTION = 0.03
BAND_BELOW = 0.02
BAND_ABOVE = 1.0
N_FLOOR = 40
N_MAX = 300
ALPHA = 0.05

# Caps are the v2/v5 strict packages', unchanged (protocol.md "Execution envelope"). The
# final cap is checked again at prepare against the v7 screen's measured episode times.
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
PROJECTION_OVERHEAD = 1.15  # v2 Tier-1 screen measured 1.13 (8582 s / 7590 s); kept
PROJECTION_MAX_BUDGET_FRACTION = 0.70  # prepare refuses a projected final above this

SLOT_LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")
SUPERVISOR_SOURCE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "apex-scaling-h256-diagnostic/supervise.py"
)
STUDY_ARTIFACT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-strict-20261002"
)
DEFAULT_OUT_ROOT = STUDY_ARTIFACT_ROOT / "run-v1"  # pre-registered Tier-2 root
DEFAULT_SMOKE_ROOT = STUDY_ARTIFACT_ROOT / "smoke-v1"  # pre-GO plumbing dry-run root
# The only root a non-smoke intent may use (tests point it at tmp_path).
TIER2_OUT_ROOT = DEFAULT_OUT_ROOT
SCREEN_RUN = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-screen-20261002/run-v1"
)
SWEEP_SUMMARY = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002/"
    "run-v1/summary.json"
)
PYTHON = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
INDEPENDENT_AUDIT = HERE / "strict_audit.py"
CLOSURE_ROOTS = ("src", "research/apex_safety_20260926", "research/apex_veto_v7_strict_20261002")
CLOSURE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md")

OUTCOMES = ("STRICT_PASS", "STRICT_FAIL", "STOP_INFEASIBLE", "INCOMPLETE", "INVALID_STOP")
SERVING_REMAINING_WORK = [
    "web/backend/safety_veto_serving.py serves v2 or v5 on the Watch hero "
    "(SNAKE_SERVE_VETO_VARIANT); serving v7 needs an opt-in variant that installs "
    "SpacePreferenceVeto(lambda=4.0) there",
    "a per-episode web serving receipt for the v7 descriptor and its source closure "
    "(safety_veto_v7.py, safety_veto_v5.py, safety_veto.py, safety_veto_v3.py) and a "
    "50-episode web serving run with its own audit",
    "Play mode: AI snakes would need the same install (human dispatch bypasses the policy)",
    "v7 costs more per decision than v5 (capped breadth-first area counts); the served "
    "frame budget under v7 is not measured here",
    "any deployment that wraps more than the one hero (every Watch snake sharing the policy, "
    "or every AI snake in Play) needs its own strict evaluation: this study measures one "
    "wrapped hero against unwrapped opponents only",
]
SERVING_STAGE_KIND = "rollout-harness self-play compatibility check"
NON_CLAIMS = [
    "no champion file, default, config, released veto or deployment change on any outcome",
    "a STRICT_PASS is a receipt only; replacing the released v5 veto is a separate explicit "
    "release action that also needs its own web serving qualification",
    "serving_path_qualified is false: the web serving path cannot install v7 today",
    "v7 Tier-1 screen numbers are not evidence in this study (pilot variance only); its "
    "observed effect informed the MDE choice, which is disclosed",
    "the result covers lambda=4.0 only, a single wrapped hero against unwrapped opponents "
    "under promotion-v2-watch-rect H5000 only; wrapping several or all snakes is not measured",
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


def earlier_namespaces(prefix: int = EXCLUSION_PREFIX) -> Dict[str, List[int]]:
    """``domain/purpose[0:prefix]`` -> seeds for every :data:`EARLIER_DOMAINS` entry."""
    return {
        f"{domain}/{purpose}[0:{prefix}]": [
            dev_screen.uint32_seed(domain, purpose, index) for index in range(prefix)
        ]
        for domain, purposes in EARLIER_DOMAINS.items()
        for purpose in purposes
    }


def namespace_report(
    seeds: Mapping[str, Sequence[int]], pilot_output: Path | None
) -> Dict[str, Any]:
    """Construction-time freshness check; ``disjoint`` is the fail-closed verdict.

    Each namespace must be unique within itself and disjoint from the other two, from every
    set ``dev_screen.disjointness_report`` checks, and from the first
    :data:`EXCLUSION_PREFIX` seeds of every :data:`EARLIER_DOMAINS` domain/purpose.
    """
    earlier = earlier_namespaces()
    report: Dict[str, Any] = {"namespaces": {}, "cross_namespace_overlap": {}}
    ok = True
    for name, values in seeds.items():
        unique = len(set(values)) == len(values)
        checked = dev_screen.disjointness_report(values, pilot_output)
        overlaps = {
            label: sorted(set(bank) & set(values))
            for label, bank in earlier.items()
            if set(bank) & set(values)
        }
        report["namespaces"][name] = {
            "domain": NAMESPACES[name][0],
            "namespace": NAMESPACE_KEY,
            "count": len(values),
            "unique": unique,
            "seeds_sha256": canonical_sha(list(values)),
            "earlier_namespaces": checked,
            "earlier_domain_overlap": overlaps,
        }
        ok = ok and unique and bool(checked["disjoint"]) and not overlaps
        ok = ok and checked.get("strict_pilot_observed") is not None
    names = list(seeds)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(seeds[left]) & set(seeds[right]))
            report["cross_namespace_overlap"][f"{left}|{right}"] = overlap
            ok = ok and not overlap
    report["earlier_domains_checked"] = {
        domain: list(purposes) for domain, purposes in EARLIER_DOMAINS.items()
    }
    report["earlier_domain_prefix_checked"] = EXCLUSION_PREFIX
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


def pilot_bank(count: int = PILOT_WORLDS) -> List[int]:
    """The v7 screen's world seeds (``apex-veto-v7-screen-v1|worlds|i``)."""
    return [dev_screen.uint32_seed(PILOT_DOMAIN, NAMESPACE_KEY, i) for i in range(count)]


def screen_pilot_deltas(screen_run: Path) -> Dict[str, Any]:
    """Recompute the v7 screen's paired B-A mass deltas from its raw records.

    Reads ``records/{A,B}-<mix>-<seed>.json`` (read-only; C/D controls are skipped), checks
    each record's arm identity (A = v5 veto, B = v7 veto at lambda=4.0 with the probe's
    ``space_preference_lambda`` == 4.0, champion hero, seed in the screen bank at its world
    index), orders pairs by world index, and cross-checks the deltas against
    ``summary.json``. ``screen_receipt_gate`` decides whether the pilot may be used.
    """
    records = Path(screen_run) / "records"
    bank = pilot_bank()
    by_key: Dict[tuple, Mapping[str, Any]] = {}
    files: Dict[str, str] = {}
    for path in sorted(records.glob("*.json")):
        entry = read_json(path)
        if entry["arm"] not in PILOT_ARM_METHODS:
            continue
        method = PILOT_ARM_METHODS[entry["arm"]]
        probe = entry["record"]["probes"].get("safety_veto") or {}
        require(entry["hero_sha256"] == CHAMPION[1], f"screen record hero differs: {path.name}")
        require(
            entry.get("safety_veto") is True
            and entry.get("safety_veto_method") == method
            and probe.get("method") == method,
            f"screen record veto is not arm {entry['arm']}'s ({method}): {path.name}",
        )
        if entry["arm"] == "B":
            lam = probe.get("space_preference_lambda")
            require(
                not isinstance(lam, bool) and lam == CANDIDATE_LAMBDA,
                f"screen B record veto lambda {lam!r} != {CANDIDATE_LAMBDA}: {path.name}",
            )
        index, seed = int(entry["world_index"]), int(entry["world_seed"])
        require(
            0 <= index < len(bank) and bank[index] == seed == int(entry["record"]["seed"]),
            f"screen record seed is not the v7 screen recipe: {path.name}",
        )
        by_key[(entry["arm"], entry["mix"], seed)] = entry
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


def screen_receipt_gate(screen_run: Path, pilot: Mapping[str, Any]) -> Dict[str, Any]:
    """The v7 screen may size this study only if its receipt says RECOMMEND_STRICT_GATE.

    Requires ``receipt.json`` with ``decision`` and ``screen_decision_before_self_check``
    both :data:`PILOT_REQUIRED_DECISION`, a passing self-check with no failures, the
    receipt's ``summary_sha256`` equal to ``summary.json``'s bytes, every A/B record read by
    :func:`screen_pilot_deltas` listed in ``receipt.records`` with the same sha256, and all
    40 pairs per mix present. ``passes`` is the fail-closed verdict. The lambda binding is
    :func:`lambda_binding_gate`.
    """
    path = Path(screen_run) / "receipt.json"
    problems: List[str] = []
    receipt: Mapping[str, Any] = read_json(path) if path.is_file() else {}
    if not receipt:
        problems.append("receipt.json missing")
    check = receipt.get("self_check") if isinstance(receipt.get("self_check"), dict) else {}
    if receipt.get("decision") != PILOT_REQUIRED_DECISION:
        problems.append(f"screen decision {receipt.get('decision')!r}")
    if receipt.get("screen_decision_before_self_check") != PILOT_REQUIRED_DECISION:
        problems.append("screen decision before self-check is not RECOMMEND_STRICT_GATE")
    if check.get("passes") is not True or check.get("failures"):
        problems.append("screen self-check did not pass")
    if receipt and receipt.get("summary_sha256") != pilot.get("summary_sha256"):
        problems.append("receipt summary_sha256 differs from summary.json")
    listed = {
        Path(str(row.get("path"))).name: row.get("sha256")
        for row in receipt.get("records") or []
        if isinstance(row, dict)
    }
    unlisted = sorted(
        name for name, sha in pilot["record_files_sha256"].items() if listed.get(name) != sha
    )
    if unlisted:
        problems.append(f"records not bound by the receipt: {unlisted[:5]}")
    counts = {m: len(pilot["deltas_by_mix"][m]) for m in MIXES}
    if any(count != PILOT_WORLDS for count in counts.values()):
        problems.append(f"pilot pairs per mix {counts} != {PILOT_WORLDS}")
    if pilot.get("summary_deltas_match") is not True:
        problems.append("screen records disagree with summary.json (or it is missing)")
    return {
        "receipt_path": str(path),
        "receipt_sha256": dev_screen.sha256_file(path) if path.is_file() else None,
        "decision": receipt.get("decision"),
        "required_decision": PILOT_REQUIRED_DECISION,
        "self_check_passes": check.get("passes"),
        "pairs_per_mix": counts,
        "problems": problems,
        "passes": not problems,
    }


def lambda_binding_gate(screen_run: Path, sweep_summary: Path = SWEEP_SUMMARY) -> Dict[str, Any]:
    """The screen's B arm ran lambda == :data:`CANDIDATE_LAMBDA`, bound by the DEV sweep.

    Requires (fail closed): the screen ``receipt.json`` ``lambda`` and
    ``sweep_binding.selected_lambda`` equal 4.0; ``sweep_binding.summary_path`` is
    ``sweep_summary`` and ``summary_sha256`` its bytes; that summary is a real (non-smoke)
    ``apex-veto-v7-lambda-sweep/v1`` summary of ``apex-veto-v7-dev-v1`` whose selection is
    ``SELECTED`` (``passes``) at 4.0, whose ``intent_sha256`` is the ``intent.json`` beside
    it, and whose source commit (summary and sweep intent, clean) is the screen's
    :data:`PILOT_COMMIT`; the binding's ``source_commit`` and ``intent_sha256`` agree; the
    screen ``intent.json`` (the one the receipt binds) names ``lambda=4.0`` and the sweep
    summary sha256 in its arm B description and its hypothesis.
    """
    screen_run, sweep_summary = Path(screen_run), Path(sweep_summary)
    problems: List[str] = []
    receipt_path, intent_path = screen_run / "receipt.json", screen_run / "intent.json"
    receipt: Mapping[str, Any] = read_json(receipt_path) if receipt_path.is_file() else {}
    screen: Mapping[str, Any] = read_json(intent_path) if intent_path.is_file() else {}
    binding = receipt.get("sweep_binding") if isinstance(receipt.get("sweep_binding"), dict) else {}

    def is_lambda(value: Any) -> bool:
        return not isinstance(value, bool) and value == CANDIDATE_LAMBDA

    if not is_lambda(receipt.get("lambda")):
        problems.append(f"screen receipt lambda {receipt.get('lambda')!r} != {CANDIDATE_LAMBDA}")
    if not is_lambda(binding.get("selected_lambda")):
        problems.append("screen receipt sweep_binding.selected_lambda is not 4.0")
    summary_sha = dev_screen.sha256_file(sweep_summary) if sweep_summary.is_file() else None
    if summary_sha is None:
        problems.append(f"sweep summary {sweep_summary} missing")
    if str(binding.get("summary_path")) != str(sweep_summary.resolve()):
        problems.append("screen sweep_binding.summary_path is not the pre-declared sweep summary")
    if binding.get("summary_sha256") != summary_sha:
        problems.append("screen sweep_binding.summary_sha256 differs from the sweep summary")
    summary: Mapping[str, Any] = read_json(sweep_summary) if summary_sha else {}
    sweep_intent = sweep_summary.parent / "intent.json"
    sweep_intent_sha = dev_screen.sha256_file(sweep_intent) if sweep_intent.is_file() else None
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    if summary.get("schema_version") != SWEEP_SCHEMA or summary.get("sweep_id") != SWEEP_ID:
        problems.append("sweep summary is not the v7 DEV lambda sweep")
    if summary.get("smoke") is not False:
        problems.append("sweep summary is a smoke (or does not say)")
    if selection.get("status") != SWEEP_STATUS_SELECTED or selection.get("passes") is not True:
        problems.append(f"sweep selection {selection.get('status')!r} did not pass")
    if not is_lambda(selection.get("selected_lambda")):
        problems.append(f"sweep selected lambda {selection.get('selected_lambda')!r} != 4.0")
    if sweep_intent_sha is None or summary.get("intent_sha256") != sweep_intent_sha:
        problems.append("sweep summary intent_sha256 differs from the sweep intent.json")
    sweep_git = read_json(sweep_intent).get("git") if sweep_intent_sha else None
    sweep_git = sweep_git if isinstance(sweep_git, dict) else {}
    if source.get("commit") != PILOT_COMMIT or sweep_git.get("commit") != PILOT_COMMIT:
        problems.append(f"sweep source commit is not {PILOT_COMMIT}")
    if source.get("dirty_paths") != "" or sweep_git.get("dirty_paths") != "":
        problems.append("sweep ran from a dirty tree")
    if binding.get("source_commit") != PILOT_COMMIT:
        problems.append("screen sweep_binding.source_commit is not the screen commit")
    if binding.get("intent_sha256") != summary.get("intent_sha256"):
        problems.append("screen sweep_binding.intent_sha256 differs from the sweep summary")
    arms = screen.get("arms") if isinstance(screen.get("arms"), dict) else {}
    arm_b, hypothesis = str(arms.get("B", "")), str(screen.get("hypothesis", ""))
    if CANDIDATE_METHOD not in arm_b or str(summary_sha) not in arm_b:
        problems.append("screen intent arm B does not name lambda=4.0 and the sweep summary")
    if f"lambda {CANDIDATE_LAMBDA!r}" not in hypothesis or str(summary_sha) not in hypothesis:
        problems.append("screen intent hypothesis does not name lambda 4.0 and the sweep summary")
    return {
        "candidate_lambda": CANDIDATE_LAMBDA,
        "sweep_summary_path": str(sweep_summary.resolve()),
        "sweep_summary_sha256": summary_sha,
        "sweep_intent_sha256": sweep_intent_sha,
        "screen_receipt_lambda": receipt.get("lambda"),
        "sweep_selected_lambda": selection.get("selected_lambda"),
        "problems": problems,
        "passes": not problems,
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

    Basis: the v7 Tier-1 screen ran one worker (torch 2 intra-op / 1 inter-op). The final plays
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
        "basis": "v7 screen A/B record wall_seconds, one worker, 2 intra-op / 1 inter-op",
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


def arm_identities(repo: Path = REPO) -> Dict[str, Dict[str, Any]]:
    """Each arm's wrapper identity: method, descriptor and source sha256s.

    ``source_sha256`` is the arm's primary veto module; ``source_sha256s`` maps every veto
    module the arm executes (repo-relative, :data:`ARM_SOURCES`) to its sha256. v5 imports
    v2 and v3, so the incumbent binds three modules; v7 imports v2, v3 and v5, so the
    candidate binds four. The candidate descriptor carries ``space_preference_lambda``.
    """
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto

    descriptors = {
        "incumbent": BoostAwareFreeSpaceVeto().descriptor(),
        "candidate": SpacePreferenceVeto(CANDIDATE_LAMBDA).descriptor(),
    }
    require(
        descriptors["candidate"].get("space_preference_lambda") == CANDIDATE_LAMBDA,
        "candidate veto lambda differs",
    )
    out = {}
    for arm in ARMS:
        sources = {rel: dev_screen.sha256_file(Path(repo) / rel) for rel in ARM_SOURCES[arm]}
        primary = ARM_SOURCES[arm][0]
        require(descriptors[arm]["method"] == ARM_METHODS[arm], f"{arm} veto method differs")
        out[arm] = {
            "method": ARM_METHODS[arm],
            "descriptor": descriptors[arm],
            "source_path": str(Path(repo) / primary),
            "source_sha256": sources[primary],
            "source_sha256s": sources,
        }
    return out


def git_blob_sha256(repo: Path, commit: str, rel: str) -> str | None:
    """sha256 of ``git show <commit>:<rel>`` bytes (None if git cannot produce them)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{rel}"], capture_output=True, check=False
    )
    return hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None


def screen_source_parity(
    screen_run: Path, identities: Mapping[str, Mapping[str, Any]], repo: Path = REPO
) -> Dict[str, Any]:
    """Tie the pilot's arms to the strict arms at the source level (fail closed).

    The screen's ``intent.json`` must be the one its receipt binds (``intent_sha256``), name
    the pre-declared screen commit :data:`PILOT_COMMIT` and record a clean tree
    (``git.dirty_paths == ""``). Every veto module in
    :data:`ARM_SOURCES` read at that commit (``git show <commit>:<rel>``) must hash to the
    strict arm's ``source_sha256s[rel]``, so N is sized from the variance of the same bytes.
    """
    path, receipt_path = Path(screen_run) / "intent.json", Path(screen_run) / "receipt.json"
    problems: List[str] = []
    intent: Mapping[str, Any] = read_json(path) if path.is_file() else {}
    receipt: Mapping[str, Any] = read_json(receipt_path) if receipt_path.is_file() else {}
    intent_sha = dev_screen.sha256_file(path) if path.is_file() else None
    if not intent:
        problems.append("screen intent.json missing")
    if intent_sha is None or receipt.get("intent_sha256") != intent_sha:
        problems.append("receipt intent_sha256 differs from the screen intent.json")
    git = intent.get("git") if isinstance(intent.get("git"), dict) else {}
    commit, dirty = git.get("commit"), git.get("dirty_paths")
    commit_ok = isinstance(commit, str) and len(commit) == 40 and set(commit) <= set(HEX_DIGITS)
    if not commit_ok:
        problems.append(f"screen git.commit is not a full commit: {commit!r}")
    elif commit != PILOT_COMMIT:
        problems.append(f"screen git.commit {commit} is not the pilot commit {PILOT_COMMIT}")
    if dirty != "":
        problems.append(f"screen ran from a dirty tree: {dirty!r}")
    sources: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        for rel in ARM_SOURCES[arm]:
            strict = (identities.get(arm, {}).get("source_sha256s") or {}).get(rel)
            screen = git_blob_sha256(repo, commit, rel) if commit_ok else None
            sources[rel] = {"screen_sha256": screen, "strict_sha256": strict}
            if screen is None or screen != strict:
                problems.append(f"{arm} {rel}: screen source {screen} != strict {strict}")
    return {
        "screen_intent_path": str(path),
        "screen_intent_sha256": intent_sha,
        "screen_commit": commit,
        "screen_dirty_paths": dirty,
        "sources": sources,
        "problems": problems,
        "passes": not problems,
    }


def install_incumbent_veto(hero: Any) -> Any:
    """The incumbent's hero install: the released v5 boost-aware veto (no options)."""
    from src.evaluation.safety_veto_v5 import install_boost_aware_veto

    return install_boost_aware_veto(hero)


def install_candidate_veto(hero: Any) -> Any:
    """The candidate's hero install: the v7 space-preference veto at lambda=4.0."""
    from src.evaluation.safety_veto_v7 import install_space_preference_veto

    return install_space_preference_veto(hero, CANDIDATE_LAMBDA)


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
    sweep_summary: Path = SWEEP_SUMMARY,
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
        gate = screen_receipt_gate(screen_run, pilot)
        require(gate["passes"], f"v7 screen may not size this study: {gate['problems']}")
        pilot["screen_receipt_gate"] = gate
        binding = lambda_binding_gate(screen_run, sweep_summary)
        require(binding["passes"], f"v7 screen lambda is not bound to 4.0: {binding['problems']}")
        pilot["lambda_binding"] = binding
        sizing = frozen_final_n(pilot["deltas_by_mix"])
        projection = runtime_projection(
            pilot.pop("episode_wall_seconds"), int(sizing["final_worlds_per_mix"])
        )
        require(
            not sizing["feasible"] or projection["within_limit"],
            f"projected final {projection['projected_seconds_per_worker']:.0f} s per worker "
            f"exceeds {PROJECTION_MAX_BUDGET_FRACTION:.0%} of the worker budget",
        )
    identities = arm_identities()
    require(
        identities["incumbent"]["source_sha256s"] == INCUMBENT_SOURCE_SHA256S,
        "the incumbent veto modules are not the released v5 sources",
    )
    for arm, identity in identities.items():
        for rel, sha in identity["source_sha256s"].items():
            closed = closure["files"].get(str((REPO / rel).resolve()))
            require(closed == sha, f"{arm} veto source {rel} is not in the source closure")
    if not smoke:
        # The pilot's A/B arms must have run these exact veto bytes (sized variance = ours).
        source_parity = screen_source_parity(screen_run, identities)
        require(
            source_parity["passes"],
            f"v7 screen ran other veto sources: {source_parity['problems']}",
        )
        pilot["screen_source_parity"] = source_parity
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
        **{
            arm: {
                "checkpoint_sha256": CHAMPION[1],
                "name": CHAMPION[0],
                "wrapper": identities[arm]["method"],
                "wrapper_identity": identities[arm],
                "wrapper_source_sha256": identities[arm]["source_sha256"],
                "wrapper_install": (
                    "dev_screen.hero_veto_installer(install_boost_aware_veto) around "
                    "rollout(hero_safety_veto=True)"
                    if arm == "incumbent"
                    else "dev_screen.hero_veto_installer(install_space_preference_veto, "
                    "lam=4.0) around rollout(hero_safety_veto=True)"
                ),
            }
            for arm in ARMS
        },
        "incumbent_released_source_sha256": INCUMBENT_SOURCE_SHA256,
        "incumbent_released_source_sha256s": dict(INCUMBENT_SOURCE_SHA256S),
        "candidate_lambda": CANDIDATE_LAMBDA,
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
            "mde_basis": MDE_BASIS,
            "mde_chosen_after_v7_screen": True,
            "v7_screen_complete_before_mde": True,
            "mde_infeasible_alternative": {"mde": 20.0, "required_n": 547},
            "candidate_lambda": CANDIDATE_LAMBDA,
            "pilot_commit": PILOT_COMMIT,
            "pilot_required_screen_decision": PILOT_REQUIRED_DECISION,
            "incumbent": f"champion + {INCUMBENT_METHOD} (released)",
            "candidate": f"champion + {CANDIDATE_METHOD}",
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
                "--sweep-summary",
                str(Path(sweep_summary).resolve()),
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
    wrappers: Mapping[str, Mapping[str, Any]],
    shard: int,
    wall_seconds: float,
    world_index: int,
    diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """The write-once per-episode envelope around a rollout record (dev_screen shape).

    ``world_index`` is the seed's index in its namespace bank. Both arms carry a veto:
    ``safety_veto: true``, ``wrapper`` (the arm's method), ``wrapper_source_sha256`` (the
    arm's primary veto module) and ``wrapper_source_sha256s`` (every veto module it runs).
    Both arms also carry ``veto_diagnostics`` (the incumbent's v5 counters, the candidate's
    v7 counters with nested v5 counters; reported, not gated).
    """
    wrapper = wrappers[unit["arm"]]
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
        "safety_veto": True,
        "wrapper": wrapper["method"],
        "wrapper_source_sha256": wrapper["source_sha256"],
        "wrapper_source_sha256s": dict(wrapper["source_sha256s"]),
        "roster_member_sha256s": [slot["member_sha256"] for slot in row["slots"]],
        "shard": shard,
        "wall_seconds": wall_seconds,
        "record": record,
        "veto_diagnostics": dict(diagnostics) if diagnostics is not None else None,
    }
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
) -> tuple:
    """Play one hero episode; returns ``(record, the arm's veto diagnostics)``.

    Both arms run ``rollout(hero_safety_veto=True)`` with the install routed through
    ``dev_screen.hero_veto_installer`` (the built-in vector61 guard still runs first): the
    incumbent's replaces rollout's v2 veto with v5, the candidate's with v7 at lambda=4.0.
    Opponents never carry a veto.
    """
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import rollout

    hero = lookup[CHAMPION[1]]
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    install = install_candidate_veto if unit["arm"] == "candidate" else install_incumbent_veto
    with dev_screen.hero_veto_installer(install) as installed:
        if smoke_frames is not None:
            record = rollout(
                hero, opponents, smoke_frames, row["world_seed"], hero_safety_veto=True
            )
        else:
            record = rollout(
                hero,
                opponents,
                HORIZON,
                row["world_seed"],
                profile=profile,
                world_identity=_expected_world_identity(row),
                mix_id=row["mix"],
                hero_safety_veto=True,
            )
    require(len(installed) == 1, f"the {unit['arm']} veto was not installed exactly once")
    return record, installed[0].diagnostics_record()


def _heartbeat_mono(path: Path, started: float) -> float:
    """Last heartbeat on the system monotonic clock, which pauses while the host sleeps.

    Wall-clock mtimes made a lid-close or thermal sleep look like a hung worker (run-v1,
    run-v2). A missing or unreadable beat counts from stage start, so a worker that never
    writes one is still caught; os.replace keeps a present file whole.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("mono")
    except (OSError, ValueError, AttributeError):
        return started
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else started


def on_ac_power() -> bool:
    """True when macOS reports the host drawing from AC; other platforms are not gated."""
    if sys.platform != "darwin":
        return True
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "'AC Power'" in out


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
    wrappers = {arm: intent[arm]["wrapper_identity"] for arm in ARMS}
    index = world_index_maps(intent)[stage]
    units = shard_units(stage_plan(stage, rows), shard)
    planned = [episode for unit in units for episode in unit]
    heartbeat = shard_dir / "heartbeat.json"
    done: Dict[str, str] = {}
    spent: List[float] = []
    stopped = None
    _write_heartbeat(heartbeat, {"utc": now_utc().isoformat(), "mono": time.monotonic(), "done": 0})
    for episode in planned:
        budget = max(MIN_EPISODE_BUDGET_SECONDS, 2 * sum(spent) / len(spent) if spent else 0)
        remaining = (stage_deadline - now_utc()).total_seconds()
        if remaining < budget:
            stopped = f"deadline: {remaining:.0f}s left < {budget:.0f}s budget"
            break
        row = by_key[(episode["mix"], episode["world_seed"])]
        began = time.monotonic()
        record, diagnostics = run_unit_episode(
            episode, row, lookup, profile, intent["smoke_frames"]
        )
        elapsed = time.monotonic() - began
        spent.append(elapsed)
        entry = episode_entry(
            episode,
            row,
            record,
            wrappers,
            shard,
            elapsed,
            index[episode["world_seed"]],
            diagnostics,
        )
        done[episode["episode_id"]] = write_durable(
            records_dir / f"{episode['episode_id']}.json", entry
        )
        _write_heartbeat(
            heartbeat, {"utc": now_utc().isoformat(), "mono": time.monotonic(), "done": len(done)}
        )
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
                and clock() - _heartbeat_mono(Path(beat), started) > heartbeat_stale_seconds
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
    wrappers: Mapping[str, Mapping[str, Any]],
    bank_index: Mapping[int, int],
) -> Dict[str, Any]:
    """Envelope, world identity and record-shape rules for one stage's records.

    ``bank_index`` maps each seed to its index in the stage's namespace bank. ``wrappers``
    maps arm -> the intent's wrapper identity. ``profile`` is ``{"descriptor", "digest"}``;
    ``None`` (smoke, legacy records) checks only the envelope and the probe's method. Every
    record must carry its own arm's wrapper probe (descriptor plus the seven v2 counters):
    v5 on the incumbent, v7 (lambda=4.0) on the candidate, and its own arm's
    ``veto_diagnostics`` kind (:data:`DIAGNOSTICS_MARKER`); their values are reported only.
    """
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        validate_strict_world_record,
    )

    units = {e["episode_id"]: e for unit in stage_plan(stage, rows) for e in unit}
    by_key = {(row["mix"], int(row["world_seed"])): row for row in rows}
    failures: List[str] = []
    totals = {arm: {"records_with_veto": 0, "decisions_equal_decision_frames": 0} for arm in ARMS}
    reported = {
        "incumbent_with_diagnostics": 0,
        "incumbent_boost_landing_vetoes": 0,
        "candidate_with_diagnostics": 0,
        "candidate_rerank_changes": 0,
        "candidate_rerank_changes_tail_release_driven": 0,
        "candidate_v5_boost_landing_vetoes": 0,
    }
    for episode_id, entry in sorted(entries.items()):
        unit = units.get(episode_id)
        if unit is None:
            failures.append(f"{episode_id}: not in the stage plan")
            continue
        row = by_key[(unit["mix"], unit["world_seed"])]
        wrapper = wrappers[unit["arm"]]
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
            "safety_veto": True,
            "wrapper": wrapper["method"],
            "wrapper_source_sha256": wrapper["source_sha256"],
            "wrapper_source_sha256s": dict(wrapper["source_sha256s"]),
            "roster_member_sha256s": [slot["member_sha256"] for slot in row["slots"]],
        }
        wrong = sorted(key for key, value in expected.items() if entry.get(key) != value)
        diagnostics = entry.get("veto_diagnostics")
        if not (
            isinstance(diagnostics, Mapping) and DIAGNOSTICS_MARKER[unit["arm"]] in diagnostics
        ):
            wrong.append("veto_diagnostics")
        if wrong:
            failures.append(f"{episode_id}: envelope fields differ {wrong}")
        record = entry.get("record", {})
        probes = record.get("probes", {}) if isinstance(record, Mapping) else {}
        probe = probes.get("safety_veto") if isinstance(probes, Mapping) else None
        if profile is None:
            if not isinstance(probe, Mapping) or probe.get("method") != wrapper["method"]:
                failures.append(f"{episode_id}: wrapper probe is not the {unit['arm']}'s veto")
        else:
            try:
                validate_strict_world_record(
                    record, profile, row, candidate_wrapper=wrapper["descriptor"]
                )
            except StrictPromotionArtifactError as exc:
                failures.append(f"{episode_id}: record shape: {exc}")
        if isinstance(probe, Mapping):
            totals[unit["arm"]]["records_with_veto"] += 1
            decisions = (probe.get("counters") or {}).get("decisions")
            frames = record.get("denominators", {}).get("decision_frames")
            totals[unit["arm"]]["decisions_equal_decision_frames"] += int(decisions == frames)
        if isinstance(diagnostics, Mapping):
            reported[f"{unit['arm']}_with_diagnostics"] += 1
            if unit["arm"] == "incumbent":
                landing = int(diagnostics.get("boost_landing_vetoes") or 0)
                reported["incumbent_boost_landing_vetoes"] += landing
            else:
                nested = diagnostics.get("v5") if isinstance(diagnostics.get("v5"), Mapping) else {}
                reported["candidate_rerank_changes"] += int(diagnostics.get("rerank_changes") or 0)
                reported["candidate_rerank_changes_tail_release_driven"] += int(
                    diagnostics.get("rerank_changes_tail_release_driven") or 0
                )
                landing = int(nested.get("boost_landing_vetoes") or 0)
                reported["candidate_v5_boost_landing_vetoes"] += landing
    return {
        "stage": stage,
        "records": len(entries),
        "failures": failures,
        "reported_not_gated": {"veto_counters": totals, "veto_diagnostics": reported},
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
    wrappers = {arm: intent[arm]["wrapper_identity"] for arm in ARMS}
    if wrappers != arm_identities():
        failures.append("arm wrapper identities differ from the frozen veto sources")
    if wrappers["incumbent"]["source_sha256s"] != INCUMBENT_SOURCE_SHA256S:
        failures.append("incumbent veto sources are not the released v5 sources")
    if wrappers["candidate"]["descriptor"].get("space_preference_lambda") != CANDIDATE_LAMBDA:
        failures.append("candidate veto lambda is not 4.0")
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
            stage, collected["entries"], rows, profile, wrappers, world_index_maps(intent)[stage]
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
    require(
        {arm: intent[arm]["wrapper_identity"] for arm in ARMS} == arm_identities(),
        "wrapper identity drift",
    )
    require(
        intent["incumbent"]["wrapper_source_sha256"] == INCUMBENT_SOURCE_SHA256
        and intent["incumbent"]["wrapper_identity"]["source_sha256s"] == INCUMBENT_SOURCE_SHA256S,
        "incumbent is not the released v5 veto",
    )
    require(
        intent["candidate"]["wrapper"] == CANDIDATE_METHOD
        and intent["candidate"]["wrapper_identity"]["descriptor"]["space_preference_lambda"]
        == CANDIDATE_LAMBDA,
        "candidate is not v7 at lambda=4.0",
    )
    if intent["smoke_frames"] is None:
        require(intent["sizing"]["final_worlds_per_mix"] >= N_FLOOR, "N below floor")
        binding = intent["pilot"].get("lambda_binding") or {}
        require(
            binding.get("passes") is True
            and binding.get("candidate_lambda") == CANDIDATE_LAMBDA
            and binding.get("screen_receipt_lambda") == CANDIDATE_LAMBDA
            and binding.get("sweep_selected_lambda") == CANDIDATE_LAMBDA,
            "pilot lambda_binding does not bind lambda=4.0",
        )
        parity = intent["pilot"].get("screen_source_parity") or {}
        rows = parity.get("sources") or {}
        require(
            parity.get("passes") is True
            and sorted(rows) == sorted({rel for arm in ARMS for rel in ARM_SOURCES[arm]})
            and all(
                rows[rel]["screen_sha256"]
                == rows[rel]["strict_sha256"]
                == intent[arm]["wrapper_identity"]["source_sha256s"][rel]
                for arm in ARMS
                for rel in ARM_SOURCES[arm]
            ),
            "pilot screen_source_parity does not bind the arms' veto sources",
        )
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
        if intent["smoke_frames"] is None:
            # Two Tier-2 runs died on battery sleep; refuse before any record exists.
            require(on_ac_power(), "host is on battery; plug in before a Tier-2 run")
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
    prep.add_argument("--sweep-summary", type=Path, default=SWEEP_SUMMARY)
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
            sweep_summary=args.sweep_summary,
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
