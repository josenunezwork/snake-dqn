#!/usr/bin/env python3
"""Independent audit of the Apex v7-veto strict (Tier-2) run: champion + v7 vs champion + v5.

Standard library only (``json`` + stdlib); it imports nothing from the repo, so
``python -I strict_audit.py --root <run-root> --out <run-root>/output/audit``
works from any directory. Exit 0 = PASS, 1 = FAIL. Adapted (copied, then
parametrized) from research/apex_veto_v5_strict_20261001/strict_audit.py.

It reads only the run's saved evidence under ``--root`` (plus the v7 Tier-1 screen
records, intent and receipt under ``--pilot-root``, the independent sizing pilot, and the
DEV lambda sweep summary under ``--sweep-summary`` that bound the screen's lambda) and
recomputes the decision path in its own code (governance_tiers_2026-09-26.md
"Audit scope"):

* namespaces: seeds re-derived from the recipe, unique, disjoint from every
  namespace dev_screen checks and from the first 1000 seeds of every earlier
  apex-veto-*/apex-safety-screen/trap-horizon domain (v5 strict, v6/v7 screens, v7 sweep);
* per-record identity: world identity rebuilt from the balanced roster recipe,
  hero/checkpoint sha, profile digest, learn/training flags, the arm's wrapper probe
  (v5 on the incumbent, v7 lambda=4.0 on the candidate; both arms are wrapped and both
  carry their own kind of ``veto_diagnostics``);
* arm identities bound to the source: each arm's wrapper source sha256s equal the
  source-closure hashes of the veto modules it runs (v5: safety_veto_v5.py, safety_veto.py,
  safety_veto_v3.py, pinned to the released bytes; v7: safety_veto_v7.py plus the same
  three), and every record carries its arm's sha256s and descriptor (probe minus counters);
* pilot: the v7 screen's receipt says RECOMMEND_STRICT_GATE with a passing self-check,
  and binds every A/B record the sizing reads and the screen intent; the screen ran from
  the clean pilot commit whose veto modules (``git show``) are the strict arms' source
  bytes; its lambda is 4.0 in the receipt, the sweep binding, the sweep summary's
  selection and the screen intent;
* denominators: ``scored_frames`` (and ``frames_completed`` where recorded) equal
  the 5000 horizon; every other frame counter is only checked at-most-cap;
* calibration means, absolute NI margin, survival bands; pilot sizing N (MDE 30);
* pairing (same seed + roster identity for both arms, exactly N per mix);
* one-sided paired t tests (own Student-t via the hypergeometric series of the
  incomplete beta), Holm, >=2-of-3 superiority, scripted NI lower bound, bands,
  expected outcome; then compares every producer claim it finds to 1e-9 rel.

Evidence contract (strict_run.py; the tests build runs through its own functions):
  <root>/intent.json; <root>/output/{admitted,rosters,calibration,decision,
  producer-outcome}.json; <root>/output/checkpoints/<sha256>.pth snapshots;
  <root>/output/<stage>/records/<episode_id>.json (episode_entry envelope around a
  tournament_eval.rollout record; stage calibration|final|serving, ``dev`` and
  ``development`` accepted as calibration); <root>/output/<stage>/shard-<k>/report.json
  (planned_episode_ids, records_sha256). Unit j of a stage plan (mix-major,
  world-minor; a final unit is the incumbent+candidate pair) belongs to shard j % 2.
Counters whose writer may not be installed (e.g. ``lifecycle_frames``,
``optimizer_updates`` on the rollout path) are never required, only bounded when
present. Producer claims are located by key in every non-record JSON document under
``<root>`` except ``--out`` (see ``compare_producer_claims`` / ``compare_design_claims``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

AUDIT_SCHEMA = "apex-veto-v7-strict-audit/v1"
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
# Checkpoint pool in the strict pilot / dev_screen order (roster phase depends on it).
POOL_SHA256S = (
    CHAMPION_SHA256,
    "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195",
    "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
    "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
)
ANCHOR_VERSION = "scripted-anchor/v1"
CANDIDATE_LAMBDA = 4.0
INCUMBENT_METHOD = "free-space-veto/v5-boost-aware"
CANDIDATE_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"
ARM_METHODS = {"incumbent": INCUMBENT_METHOD, "candidate": CANDIDATE_METHOD}
V2_SOURCE = "src/evaluation/safety_veto.py"
V3_SOURCE = "src/evaluation/safety_veto_v3.py"
V5_SOURCE = "src/evaluation/safety_veto_v5.py"
V7_SOURCE = "src/evaluation/safety_veto_v7.py"
# Released v5 source bytes (v5 STRICT_PASS, web serving release): v5 and what it imports.
INCUMBENT_SOURCE_SHA256S = {
    V5_SOURCE: "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
    V2_SOURCE: "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    V3_SOURCE: "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
}
INCUMBENT_SOURCE_SHA256 = INCUMBENT_SOURCE_SHA256S[V5_SOURCE]
# Repo-relative veto modules each arm executes (primary first); v7 imports v2, v3 and v5.
ARM_SOURCES = {
    "incumbent": (V5_SOURCE, V2_SOURCE, V3_SOURCE),
    "candidate": (V7_SOURCE, V2_SOURCE, V3_SOURCE, V5_SOURCE),
}
# A key every veto_diagnostics of the arm's kind has (v5 counters; v7 counters).
DIAGNOSTICS_MARKER = {"incumbent": "boost_landing_vetoes", "candidate": "rerank_changes"}
# The seven v2 counters both probes carry (writer: SafetyVetoCounters.to_dict).
VETO_COUNTERS = (
    "decisions",
    "kept_base",
    "vetoes_applied",
    "fallback_no_spacious",
    "vetoed_base_boost",
    "vetoes_to_boost",
    "vetoes_speed_switched",
)
PROFILE_NAME = "promotion-v2-watch-rect"
PROFILE_DIGEST = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
WORLD_SEED_NAMESPACE = "evaluation-world/v1"
HORIZON = 5000
MIXES = ("frozen", "scripted", "mixed")
SCRIPTED_MIX = "scripted"
ROSTER_WIDTH = 5
ALPHA = 0.05
REQUIRED_SUCCESSES = 2
MDE_ABSOLUTE = 30.0  # declared after the v7 screen (protocol.md "Sizing", disclosed)
SIZING_FLOOR = 40
SIZING_POWER = 0.8
N_MAX = 300
NI_FRACTION = 0.03
BAND_LOWER_OFFSET = -0.02
BAND_UPPER_OFFSET = 1.0
REL_TOL = 1e-9
ABS_FLOOR = 1e-12
SERVING_EPISODES = 50
SERVING_MIX = "serving-selfplay"  # Watch self-play: five unwrapped champion opponents
WORKERS = 2  # unit j of a stage plan runs on shard j % WORKERS
STAGES = ("calibration", "final", "serving")
# Directory / ``stage`` field spellings accepted for each stage.
STAGE_ALIASES = {
    "calibration": "calibration",
    "dev": "calibration",
    "development": "calibration",
    "final": "final",
    "serving": "serving",
}
# Repo-relative source listings ({path, sha256}) are outside the run root.
REPO_TOP_LEVEL = ("src", "research", "configs", "web", "tests", "docs", "saved_snakes")
OUTCOMES = ("STRICT_PASS", "STRICT_FAIL", "STOP_INFEASIBLE", "INVALID_STOP", "INCOMPLETE")
DECISION_METHOD = "strict-promotion-v1"
SIZING_METHOD = "paired-delta-t-marginal-v1"

# Seed recipe: uint32 big-endian prefix of sha256("<domain>|<namespace>|<index>").
NAMESPACE_LABEL = "worlds"
NAMESPACES = {
    "calibration": ("apex-veto-v7-strict-dev-v1", 16),
    "final": ("apex-veto-v7-strict-final-v1", N_MAX),
    "serving": ("apex-veto-v7-strict-serving-v1", SERVING_EPISODES),
}
SMOKE_DOMAIN = "apex-veto-v7-strict-smoke-v1"  # producer dry-run worlds
# Pilot: the v7 Tier-1 screen played its first 40 seeds (arms A = v5, B = v7 lambda=4.0).
PILOT_DOMAIN, PILOT_COUNT = "apex-veto-v7-screen-v1", 40
PILOT_REQUIRED_DECISION = "RECOMMEND_STRICT_GATE"
PILOT_ARM_METHODS = {"A": INCUMBENT_METHOD, "B": CANDIDATE_METHOD}
PILOT_COMMIT = "900385fed868a1b35f308871d211aeafc318e4c2"
SWEEP_SCHEMA, SWEEP_ID = "apex-veto-v7-lambda-sweep/v1", "apex-veto-v7-dev-v1"
EXCLUSION_PREFIX = 1000
WEB_PURPOSES = ("watch", "play", "parity", "worlds")
# Every earlier domain of this recipe (name -> purposes); the first 1000 seeds of each.
EARLIER_DOMAINS: Dict[str, Tuple[str, ...]] = {
    "apex-safety-screen-v1": ("worlds",),
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
    SMOKE_DOMAIN: ("worlds",),
}
CHALLENGER_DOMAIN = "task-aligned-challenger-20260924/original6102/v1"
CHALLENGER_COUNTS = {"development": 16, "shakedown": 4, "pilot": 16, "final": 120, "serving": 50}
SMALL_INTEGER_SEEDS = range(1000)
DEFAULT_PILOT_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-screen-20261002/run-v1"
)
DEFAULT_SWEEP_SUMMARY = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002/"
    "run-v1/summary.json"
)
# The repository this audit file lives in (git objects for the screen-source parity rule).
AUDIT_REPO = Path(__file__).resolve().parents[2]
ARM_ALIASES = {
    "incumbent": "incumbent",
    "A": "incumbent",
    "candidate": "candidate",
    "B": "candidate",
}


class AuditError(Exception):
    """Evidence is unreadable or structurally unusable."""


# ----------------------------------------------------------------------------- JSON


def _reject_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant {value}")


def _unique_pairs(items: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise AuditError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    """Strict JSON: no duplicate keys, no NaN/Infinity."""
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise AuditError(f"{path}: {exc}") from exc


def canonical_digest(value: Any) -> str:
    """sha256 of sorted-key compact ASCII JSON (runtime_contract.canonical_digest)."""
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def close(a: Any, b: Any) -> bool:
    """Equality at 1e-9 relative (with a 1e-12 absolute floor for values near 0)."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if not (is_real(a) and is_real(b)):
        return False
    a, b = float(a), float(b)
    diff = abs(a - b)
    return diff <= REL_TOL * max(abs(a), abs(b)) or diff <= ABS_FLOOR


def walk(value: Any, path: str = "$") -> Iterator[Tuple[str, Any]]:
    """Yield (json-path, node) for every dict/list node, depth first."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")


def find_key(docs: Mapping[str, Any], key: str) -> List[Tuple[str, Any]]:
    """Every (doc:path, value) whose dict key is ``key`` across producer docs."""
    found: List[Tuple[str, Any]] = []
    for name, doc in docs.items():
        for path, node in walk(doc):
            if isinstance(node, dict) and key in node:
                found.append((f"{name}:{path}.{key}", node[key]))
    return found


# ----------------------------------------------------------------------------- worlds


def uint32_seed(domain: str, namespace: str, index: int) -> int:
    digest = hashlib.sha256(f"{domain}|{namespace}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def stage_bank(stage: str) -> List[int]:
    domain, count = NAMESPACES[stage]
    return [uint32_seed(domain, NAMESPACE_LABEL, index) for index in range(count)]


def earlier_namespaces() -> Dict[str, List[int]]:
    """Every namespace dev_screen checks, plus the first 1000 seeds of each earlier domain."""
    out = {
        f"challenger/{name}": [uint32_seed(CHALLENGER_DOMAIN, name, i) for i in range(count)]
        for name, count in CHALLENGER_COUNTS.items()
    }
    for domain, purposes in EARLIER_DOMAINS.items():
        for purpose in purposes:
            out[f"{domain}/{purpose}"] = [
                uint32_seed(domain, purpose, i) for i in range(EXCLUSION_PREFIX)
            ]
    out["small-integers-0-999"] = list(SMALL_INTEGER_SEEDS)
    return out


def scripted_anchor_sha(name: str) -> str:
    return canonical_digest({"scripted_anchor": name, "anchor_version": ANCHOR_VERSION})


def expected_roster(mix: str, world_index: int) -> List[str]:
    """Slot members of strict_promotion.materialize_rosters for one (mix, index)."""
    if mix == SERVING_MIX:
        return [CHAMPION_SHA256] * ROSTER_WIDTH
    greedy, random_safe = scripted_anchor_sha("greedy_food"), scripted_anchor_sha("random_safe")
    pool = list(POOL_SHA256S)
    members: List[str] = []
    ordinal = 0
    for slot in range(1, ROSTER_WIDTH + 1):
        if mix == "frozen":
            members.append(pool[(world_index + slot - 1) % len(pool)])
        elif mix == "scripted":
            members.append(greedy)
        elif mix == "mixed" and slot % 2:
            members.append(pool[(world_index + ordinal) % len(pool)])
            ordinal += 1
        elif mix == "mixed":
            members.append(random_safe)
        else:
            raise AuditError(f"unknown mix {mix!r}")
    return members


def expected_world_identity(mix: str, seed: int, world_index: int) -> Dict[str, Any]:
    hashes = expected_roster(mix, world_index)
    return {
        "seed_namespace": WORLD_SEED_NAMESPACE,
        "seed": seed,
        "mix_id": mix,
        "ordered_slot_content_hashes": hashes,
        "roster_id": canonical_digest({"slots": hashes}),
    }


# ----------------------------------------------------------------------------- statistics
# Second implementation: eval_stats uses a Lentz continued fraction for the
# incomplete beta; this uses the positive-term hypergeometric power series
# I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * 2F1(a+b, 1; a+1; x), switched by symmetry
# so the series ratio stays below one. No subtraction near a small tail value.


def _ibeta_series(x: float, y: float, a: float, b: float) -> float:
    """I_x(a, b) by series; ``y`` is 1 - x computed without cancellation."""
    log_front = (
        a * math.log(x)
        + b * math.log(y)
        - math.log(a)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    term, total, n = 1.0, 1.0, 0
    while True:
        term *= (a + b + n) / (a + 1.0 + n) * x
        n += 1
        total += term
        if term <= total * 1e-17:
            break
        if n > 50_000_000:
            raise ArithmeticError("incomplete beta series did not converge")
    return math.exp(log_front) * total


def regularized_beta(x: float, y: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    if x <= (a + 1.0) / (a + b + 2.0):
        return _ibeta_series(x, y, a, b)
    return 1.0 - _ibeta_series(y, x, b, a)


def t_sf(t: float, df: int) -> float:
    """P(T > t) for Student-t with ``df`` degrees of freedom."""
    if t == 0.0:
        return 0.5
    t2 = t * t
    x, y = df / (df + t2), t2 / (df + t2)
    tail = 0.5 * regularized_beta(x, y, df / 2.0, 0.5)
    return tail if t > 0.0 else 1.0 - tail


def t_isf(p: float, df: int) -> float:
    """Positive t with P(T > t) = p (0 < p < 0.5), bisection to adjacent floats."""
    low, high = 0.0, 1.0
    while t_sf(high, df) > p:
        high *= 2.0
    for _ in range(4096):
        mid = (low + high) / 2.0
        if mid in (low, high):
            break
        if t_sf(mid, df) > p:
            low = mid
        else:
            high = mid
    return high


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def sample_sd(values: Sequence[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def paired_test(deltas: Sequence[float], alpha: float = ALPHA) -> Dict[str, Any]:
    """One-sided paired t test of H0: mean delta <= 0 (eval_stats semantics)."""
    n = len(deltas)
    out: Dict[str, Any] = {"n": n, "mean_delta": mean(deltas) if n else None, "valid": n >= 2}
    if n < 2:
        out.update({"p_value": None, "superior": False})
        return out
    sd = sample_sd(deltas)
    se = sd / math.sqrt(n)
    crit = t_isf(alpha, n - 1)
    out.update({"sample_std": sd, "standard_error": se, "critical_value": crit})
    if se == 0.0:
        mu = out["mean_delta"]
        out["t_statistic"] = None
        out["p_value"] = 0.0 if mu > 0 else (1.0 if mu < 0 else 0.5)
    else:
        out["t_statistic"] = out["mean_delta"] / se
        out["p_value"] = t_sf(out["t_statistic"], n - 1)
    out["superior"] = out["p_value"] <= alpha
    return out


def holm_step_down(p_by_mix: Mapping[str, float], alpha: float = ALPHA) -> Dict[str, Any]:
    """Holm: reject in ascending p order while p_(k) <= alpha / (m - k)."""
    order = sorted(p_by_mix, key=lambda name: (p_by_mix[name], list(p_by_mix).index(name)))
    m = len(order)
    rejected = {name: False for name in p_by_mix}
    adjusted: Dict[str, float] = {}
    running, still = 0.0, True
    for k, name in enumerate(order):
        p = p_by_mix[name]
        still = still and p <= alpha / (m - k)
        rejected[name] = still
        running = max(running, min(1.0, (m - k) * p))
        adjusted[name] = running
    return {"rejected": rejected, "adjusted_p_values": adjusted}


def strict_decision(
    deltas_by_mix: Mapping[str, Sequence[float]], absolute_delta_ni: float
) -> Dict[str, Any]:
    """Holm >=2-of-3 one-sided superiority AND scripted NI lower bound > -margin."""
    per_mix = {m: paired_test(deltas_by_mix[m]) for m in MIXES}
    lengths = {len(deltas_by_mix[m]) for m in MIXES}
    valid = all(per_mix[m]["valid"] for m in MIXES) and len(lengths) == 1
    out: Dict[str, Any] = {"per_mix": per_mix, "valid": valid}
    if not valid:
        out.update({"superiority_passes": False, "ni_passes": False, "passes": False})
        return out
    holm = holm_step_down({m: per_mix[m]["p_value"] for m in MIXES})
    successful = [m for m in MIXES if holm["rejected"][m]]
    scripted = per_mix[SCRIPTED_MIX]
    lower = scripted["mean_delta"] - scripted["critical_value"] * scripted["standard_error"]
    out.update(
        {
            "rejected": holm["rejected"],
            "adjusted_p_values": holm["adjusted_p_values"],
            "successful_mixes": successful,
            "superiority_passes": len(successful) >= REQUIRED_SUCCESSES,
            "ni_lower_bound": lower,
            "absolute_delta_ni": absolute_delta_ni,
            "ni_passes": lower > -absolute_delta_ni,
        }
    )
    out["passes"] = bool(out["superiority_passes"] and out["ni_passes"])
    return out


def required_count(delta_sd: float, mde: float) -> Tuple[int, float]:
    """eval_stats' monotone fixed-point t sizing; also returns the last raw value."""
    if delta_sd == 0.0:
        return SIZING_FLOOR, 0.0
    planning_alpha = ALPHA / len(MIXES)
    count, raw = SIZING_FLOOR, 0.0
    for _ in range(10_000):
        crit = t_isf(planning_alpha, count - 1) + t_isf(1.0 - SIZING_POWER, count - 1)
        raw = (crit * delta_sd / mde) ** 2
        nxt = max(SIZING_FLOOR, int(math.ceil(raw)))
        if nxt <= count:
            return count, raw
        count = nxt
    raise ArithmeticError("sizing did not converge")


def pilot_sizing(pilot_deltas: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    per_mix: Dict[str, Any] = {}
    for m in MIXES:
        sd = sample_sd(pilot_deltas[m])
        n, raw = required_count(sd, MDE_ABSOLUTE)
        # A ceil input within 1e-9 of an integer could flip on a last-ulp difference.
        borderline = abs(raw - round(raw)) <= 1e-9 * max(1.0, raw) and raw > SIZING_FLOOR
        per_mix[m] = {
            "n": len(pilot_deltas[m]),
            "paired_delta_std": sd,
            "mde": MDE_ABSOLUTE,
            "recommended_n": n,
            "ceil_input_borderline": borderline,
        }
    required = max(SIZING_FLOOR, max(row["recommended_n"] for row in per_mix.values()))
    return {"method": SIZING_METHOD, "per_mix": per_mix, "required_final_worlds": required}


def p_values_agree(p_audit: Any, p_producer: Any, t_statistic: Any) -> bool:
    """1e-9 relative, except |t| < 1e-4 (p within 4e-5 of 0.5) at 1e-6 absolute.

    The frozen reducer (eval_stats) forms x = df / (df + t^2), which loses
    relative precision for tiny |t|; there it differs from the exact value by up
    to ~1e-7 relative. Such p-values are ~0.5 and cannot reject at alpha 0.05.
    """
    if close(p_audit, p_producer):
        return True
    return (
        is_real(t_statistic)
        and abs(float(t_statistic)) < 1e-4
        and is_real(p_audit)
        and is_real(p_producer)
        and abs(float(p_audit) - float(p_producer)) <= 1e-6
    )


# ----------------------------------------------------------------------------- audit state


class Audit:
    """Collects rule results; any failed rule fails the audit."""

    def __init__(self) -> None:
        self.rules: List[Dict[str, Any]] = []
        self.report: Dict[str, Any] = {}

    def rule(self, name: str, ok: bool, detail: Any = None, source: str = "") -> bool:
        self.rules.append({"rule": name, "ok": bool(ok), "detail": detail, "source": source})
        return bool(ok)

    def failures(self) -> List[Dict[str, Any]]:
        return [row for row in self.rules if not row["ok"]]


def stage_of(relative: Path) -> Optional[str]:
    for part in relative.parts:
        if part in STAGE_ALIASES:
            return STAGE_ALIASES[part]
    return None


def discover(root: Path, out_dir: Path) -> Tuple[Dict[str, List[Tuple[str, Any]]], Dict[str, Any]]:
    """Split JSON evidence into per-stage episode records and producer documents."""
    records: Dict[str, List[Tuple[str, Any]]] = {stage: [] for stage in STAGES}
    records["<no-stage>"] = []
    docs: Dict[str, Any] = {}
    out_resolved = out_dir.resolve()
    for path in sorted(root.rglob("*.json")):
        resolved = path.resolve()
        if resolved == out_resolved or out_resolved in resolved.parents:
            continue
        relative = path.relative_to(root)
        if "records" in relative.parts[:-1]:
            records[stage_of(relative) or "<no-stage>"].append((str(relative), load_json(path)))
        else:
            docs[str(relative)] = load_json(path)
    return records, docs


FRAME_SKIP_SUBTREES = ("evaluation_profile",)


def _frame_counters(entry: Mapping[str, Any]) -> Iterator[Tuple[str, str, Any]]:
    """(path, key, value) for every frame-like counter outside the profile subtree."""
    for path, node in walk(entry):
        if not isinstance(node, dict) or any(f".{s}" in path for s in FRAME_SKIP_SUBTREES):
            continue
        for key, value in node.items():
            if "frames" in key or key.startswith("frame_"):
                if not isinstance(value, (dict, list)):
                    yield path, key, value


def veto_problems(entry: Mapping[str, Any], arm: str, cap: int) -> List[str]:
    """Both arms are wrapped: the arm's probe, seven v2 counters and envelope fields.

    Writers: ``tournament_eval.rollout`` attaches ``veto.record()`` (descriptor plus
    ``SafetyVetoCounters.to_dict``); strict_run.episode_entry writes the envelope and the
    arm's ``veto_diagnostics`` (``diagnostics_record()``: v5 or v7 counters).
    Exact descriptor equality with the intent is ``identity.arm_records_bound``.
    """
    problems: List[str] = []
    method = ARM_METHODS[arm]
    record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
    probes = record.get("probes") if isinstance(record.get("probes"), dict) else {}
    veto = probes.get("safety_veto")
    if entry.get("safety_veto") is not True:
        problems.append(f"{arm} safety_veto flag is not true")
    if entry.get("wrapper") != method:
        problems.append(f"entry wrapper is not the {arm}'s {method}")
    if not is_sha256(entry.get("wrapper_source_sha256")):
        problems.append("wrapper_source_sha256")
    shas = entry.get("wrapper_source_sha256s")
    if not (
        isinstance(shas, dict)
        and sorted(shas) == sorted(ARM_SOURCES[arm])
        and all(is_sha256(v) for v in shas.values())
        and shas.get(ARM_SOURCES[arm][0]) == entry.get("wrapper_source_sha256")
    ):
        problems.append("wrapper_source_sha256s")
    diagnostics = entry.get("veto_diagnostics")
    if not (isinstance(diagnostics, dict) and DIAGNOSTICS_MARKER[arm] in diagnostics):
        problems.append(f"veto_diagnostics are not the {arm}'s kind")
    if not isinstance(veto, dict):
        problems.append(f"{arm} record lacks the safety_veto probe")
        return problems
    if veto.get("method") != method:
        problems.append(f"safety_veto.method is not {method}")
    counters = veto.get("counters") if isinstance(veto.get("counters"), dict) else {}
    if sorted(counters) != sorted(VETO_COUNTERS) or not all(
        is_int(v) and 0 <= v <= cap for v in counters.values()
    ):
        problems.append("veto counters missing, extra or out of [0, cap]")
    elif counters["decisions"] != (
        counters["kept_base"] + counters["vetoes_applied"] + counters["fallback_no_spacious"]
    ):
        problems.append("veto decision counters are inconsistent")
    return problems


STAGE_ARMS = {"calibration": ("incumbent",), "final": ("incumbent", "candidate")}
STAGE_ARMS["serving"] = ("candidate",)


def validate_entry(
    stage: str, name: str, entry: Any, bank: Optional[Sequence[int]] = None
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Per-record rules: (normalized episode or None, list of problems).

    Every stage is a profiled H5000 ``tournament_eval.rollout`` (serving included, on
    the ``serving-selfplay`` roster). ``bank`` overrides the stage's seed bank (tests
    replay real Tier-1 records on their own seeds).
    """
    problems: List[str] = []

    def need(ok: bool, message: str) -> bool:
        if not ok:
            problems.append(message)
        return ok

    if not need(isinstance(entry, dict), "entry is not an object"):
        return None, problems
    record = entry.get("record")
    arm = ARM_ALIASES.get(entry.get("arm"))
    seed, mix, index = entry.get("world_seed"), entry.get("mix"), entry.get("world_index")
    need(isinstance(record, dict), "record missing")
    need(arm in STAGE_ARMS[stage], f"arm {entry.get('arm')!r} not allowed in {stage}")
    need(is_int(seed) and 0 <= seed < 2**32, "world_seed not uint32")
    need(is_int(index), "world_index missing")
    need(mix in ((SERVING_MIX,) if stage == "serving" else MIXES), f"mix {mix!r}")
    if "stage" in entry:
        declared = STAGE_ALIASES.get(entry["stage"])
        need(declared == stage, f"stage field {entry['stage']!r} != directory {stage!r}")
    need(entry.get("hero_sha256") == CHAMPION_SHA256, "hero_sha256 is not the champion")
    if problems:
        return None, problems
    if "episode_id" in entry:
        need(entry["episode_id"] == f"{stage}-{arm}-{mix}-{seed}", "episode_id")
    for path, node in walk(entry):
        if isinstance(node, dict) and "optimizer_updates" in node:
            need(node["optimizer_updates"] == 0, f"{path}.optimizer_updates != 0")
    # Frame counters: at most the horizon; only the always-charged ones must equal it.
    for path, key, value in _frame_counters(entry):
        if need(is_real(value) and 0 <= value <= HORIZON, f"{path}.{key} not in [0, cap]"):
            if key in ("frames_completed", "scored_frames"):
                need(value == HORIZON, f"{path}.{key} != {HORIZON}")
    profile = record.get("evaluation_profile")
    profile = profile if isinstance(profile, dict) else {}
    runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
    need(profile.get("learn") is False, "evaluation_profile.learn is not false")
    need(runtime.get("training") is False, "runtime.training is not false")
    need(runtime.get("mode") == "watch", "runtime.mode is not watch")
    need(record.get("evaluation_profile_digest") == PROFILE_DIGEST, "profile digest")
    need(profile.get("name") == PROFILE_NAME, "profile name")
    need(profile.get("scored_horizon") == HORIZON, "scored_horizon")
    den = record.get("denominators") if isinstance(record.get("denominators"), dict) else {}
    need(den.get("scored_frames") == HORIZON, "denominators.scored_frames missing")
    mass, surv, alive = (
        record.get("mass_integral"),
        record.get("survival_fraction"),
        den.get("alive_frames"),
    )
    need(is_real(mass) and mass >= 0, "mass_integral not a finite non-negative real")
    if need(is_real(surv) and 0 <= surv <= 1 and is_int(alive), "survival/alive_frames"):
        need(abs(surv - alive / HORIZON) <= ABS_FLOOR, "survival_fraction != alive/scored")
    problems.extend(veto_problems(entry, arm, HORIZON))
    bank = list(bank) if bank is not None else stage_bank(stage)
    if need(0 <= index < len(bank) and bank[index] == seed, "seed != bank[world_index]"):
        identity = expected_world_identity(mix, seed, index)
        need(record.get("seed") == seed, "record.seed != world_seed")
        need(record.get("world_identity") == identity, "world identity != roster recipe")
        need(
            entry.get("roster_member_sha256s") == identity["ordered_slot_content_hashes"],
            "roster_member_sha256s != world identity",
        )
    if problems:
        return None, problems
    episode = {"name": name, "stage": stage, "arm": arm, "mix": mix, "seed": seed}
    episode.update({"index": index, "record": record, "entry": entry})
    return episode, problems


# ----------------------------------------------------------------------------- stages


def validate_stage(
    audit: Audit, stage: str, raw: Sequence[Tuple[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate every record of one stage; one rule row per stage (problems listed)."""
    episodes: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    for name, entry in raw:
        episode, problems = validate_entry(stage, name, entry)
        if problems:
            bad.append({"record": name, "problems": problems})
        elif episode is not None:
            episodes.append(episode)
    keys = [(e["arm"], e["mix"], e["seed"]) for e in episodes]
    duplicates = (
        sorted({str(k) for k in keys if keys.count(k) > 1}) if len(set(keys)) != len(keys) else []
    )
    audit.rule(
        f"{stage}.records_valid",
        not bad and not duplicates,
        {
            "records": len(raw),
            "invalid": bad[:50],
            "invalid_count": len(bad),
            "duplicates": duplicates,
        },
        "tournament_eval.rollout record + producer entry wrapper (dev_screen shape)",
    )
    return episodes


def audit_namespaces(audit: Audit) -> None:
    banks = {stage: stage_bank(stage) for stage in STAGES}
    earlier = earlier_namespaces()
    detail: Dict[str, Any] = {}
    ok = True
    for stage, bank in banks.items():
        unique = len(set(bank)) == len(bank)
        overlaps = {
            name: sorted(set(bank) & set(values))
            for name, values in list(earlier.items())
            + [(f"strict/{s}", banks[s]) for s in STAGES if s != stage]
        }
        overlaps = {name: seeds for name, seeds in overlaps.items() if seeds}
        ok = ok and unique and not overlaps
        detail[stage] = {
            "domain": NAMESPACES[stage][0],
            "count": len(bank),
            "unique": unique,
            "overlaps": overlaps,
            "sha256": canonical_digest({"seeds": bank}),
        }
    audit.rule("namespaces.fresh_and_disjoint", ok, detail, "uint32 sha256 recipe (dev_screen)")


def pilot_bank() -> List[int]:
    return [uint32_seed(PILOT_DOMAIN, "worlds", i) for i in range(PILOT_COUNT)]


def load_pilot(audit: Audit, pilot_root: Path) -> Optional[Dict[str, List[float]]]:
    """Paired B-A mass deltas of the v7 screen (the independent sizing pilot).

    Rules: ``pilot.pairs`` (A = champion + v5, B = champion + v7 at lambda 4.0, seeds on the
    screen recipe, 40 pairs per mix with equal world identity) and ``pilot.screen_receipt``
    (decision RECOMMEND_STRICT_GATE before and after a passing self-check, summary bytes
    and every A/B record bound by the receipt).
    """
    directory = pilot_root / "records"
    if not directory.is_dir():
        audit.rule("pilot.available", False, str(directory), "dev_screen run_episode records")
        return None
    bank = pilot_bank()
    arms: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    hashes: Dict[str, str] = {}
    problems: List[str] = []
    for path in sorted(directory.glob("*.json")):
        entry = load_json(path)
        if entry.get("arm") not in PILOT_ARM_METHODS:
            continue  # arms C/D are the screen's determinism and replay controls
        hashes[path.name] = sha256_file(path)
        key = (entry["arm"], entry["mix"], entry["world_index"])
        record = entry["record"]
        method = PILOT_ARM_METHODS[entry["arm"]]
        probe = (record.get("probes") or {}).get("safety_veto") or {}
        if key in arms:
            problems.append(f"duplicate {key}")
        if (
            not 0 <= entry["world_index"] < len(bank)
            or bank[entry["world_index"]] != entry["world_seed"]
            or record["seed"] != entry["world_seed"]
        ):
            problems.append(f"{path.name}: seed not in the screen recipe")
        if entry["hero_sha256"] != CHAMPION_SHA256 or entry.get("safety_veto") is not True:
            problems.append(f"{path.name}: hero/veto flag")
        if entry.get("safety_veto_method") != method or probe.get("method") != method:
            problems.append(f"{path.name}: arm {entry['arm']} veto is not {method}")
        lam = probe.get("space_preference_lambda")
        if entry["arm"] == "B" and (isinstance(lam, bool) or lam != CANDIDATE_LAMBDA):
            problems.append(f"{path.name}: arm B veto lambda {lam!r} != {CANDIDATE_LAMBDA}")
        if record["denominators"]["scored_frames"] != HORIZON:
            problems.append(f"{path.name}: scored_frames")
        if record["evaluation_profile_digest"] != PROFILE_DIGEST:
            problems.append(f"{path.name}: profile digest")
        arms[key] = entry
    deltas: Dict[str, List[float]] = {}
    for mix in MIXES:
        deltas[mix] = []
        for index in range(PILOT_COUNT):
            a, b = arms.get(("A", mix, index)), arms.get(("B", mix, index))
            if a is None or b is None:
                problems.append(f"missing pair {mix}/{index}")
                continue
            if a["record"]["world_identity"] != b["record"]["world_identity"]:
                problems.append(f"world identity differs {mix}/{index}")
            deltas[mix].append(b["record"]["mass_integral"] - a["record"]["mass_integral"])
    audit.rule(
        "pilot.pairs",
        not problems,
        {
            "root": str(pilot_root),
            "problems": problems[:50],
            "pairs": {m: len(deltas[m]) for m in MIXES},
        },
        "v7 screen records (dev_screen.run_episode), arms A/B, 40 worlds per mix",
    )
    gate = screen_receipt_problems(pilot_root, hashes)
    audit.rule(
        "pilot.screen_receipt",
        not gate,
        gate[:20],
        "v7 screen receipt.json (screen.py build_receipt) and summary.json",
    )
    return deltas if not problems and not gate else None


def screen_receipt_problems(pilot_root: Path, hashes: Mapping[str, str]) -> List[str]:
    """Why the v7 screen may not size this study (empty list: it may)."""
    path = pilot_root / "receipt.json"
    if not path.is_file():
        return ["receipt.json missing"]
    receipt = load_json(path)
    problems: List[str] = []
    check = receipt.get("self_check") if isinstance(receipt.get("self_check"), dict) else {}
    if receipt.get("decision") != PILOT_REQUIRED_DECISION:
        problems.append(f"screen decision {receipt.get('decision')!r}")
    if receipt.get("screen_decision_before_self_check") != PILOT_REQUIRED_DECISION:
        problems.append("screen decision before self-check")
    if check.get("passes") is not True or check.get("failures"):
        problems.append("screen self-check did not pass")
    summary = pilot_root / "summary.json"
    if not summary.is_file() or receipt.get("summary_sha256") != sha256_file(summary):
        problems.append("receipt summary_sha256 differs from summary.json")
    listed = {
        Path(str(row.get("path"))).name: row.get("sha256")
        for row in receipt.get("records") or []
        if isinstance(row, dict)
    }
    unbound = sorted(name for name, sha in hashes.items() if listed.get(name) != sha)
    if unbound:
        problems.append(f"records not bound by the receipt: {unbound[:5]}")
    return problems


def git_blob_sha256(repo: Path, commit: str, rel: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{rel}"], capture_output=True, check=False
    )
    return hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None


def screen_source_problems(
    pilot_root: Path, intent: Mapping[str, Any], repo: Path = AUDIT_REPO
) -> List[str]:
    """Why the screen's arms are not the strict arms at the source level (empty: they are).

    Independently of the producer: the screen ``intent.json`` is the one its receipt binds,
    names the pilot commit and a clean tree; each arm's veto module read from that commit
    (``git show``) hashes to the strict intent's ``wrapper_identity.source_sha256s[rel]``;
    and the producer's recorded ``pilot.screen_source_parity`` agrees with all of it.
    """
    path, receipt_path = pilot_root / "intent.json", pilot_root / "receipt.json"
    if not path.is_file() or not receipt_path.is_file():
        return ["screen intent.json or receipt.json missing"]
    problems: List[str] = []
    screen, receipt, digest = load_json(path), load_json(receipt_path), sha256_file(path)
    if receipt.get("intent_sha256") != digest:
        problems.append("receipt intent_sha256 differs from the screen intent.json")
    git = screen.get("git") if isinstance(screen.get("git"), dict) else {}
    commit = git.get("commit")
    commit_ok = (
        isinstance(commit, str)
        and len(commit) == 40
        and all(ch in "0123456789abcdef" for ch in commit)
    )
    if not commit_ok:
        problems.append(f"screen git.commit is not a full commit: {commit!r}")
    elif commit != PILOT_COMMIT:
        problems.append(f"screen git.commit {commit} is not the pilot commit {PILOT_COMMIT}")
    if git.get("dirty_paths") != "":
        problems.append(f"screen ran from a dirty tree: {git.get('dirty_paths')!r}")
    pilot = intent.get("pilot") if isinstance(intent.get("pilot"), dict) else {}
    parity = pilot.get("screen_source_parity")
    parity = parity if isinstance(parity, dict) else {}
    rows = parity.get("sources") if isinstance(parity.get("sources"), dict) else {}
    if parity.get("passes") is not True or parity.get("problems"):
        problems.append("intent pilot.screen_source_parity does not pass")
    if parity.get("screen_commit") != commit or parity.get("screen_intent_sha256") != digest:
        problems.append("intent pilot.screen_source_parity names another screen intent/commit")
    for arm in ARM_METHODS:
        identity = _arm_node(intent, arm).get("wrapper_identity")
        shas = identity.get("source_sha256s") if isinstance(identity, dict) else None
        shas = shas if isinstance(shas, dict) else {}
        for rel in ARM_SOURCES[arm]:
            strict = shas.get(rel)
            mine = git_blob_sha256(repo, commit, rel) if commit_ok else None
            row = rows.get(rel) if isinstance(rows.get(rel), dict) else {}
            if not is_sha256(strict) or mine != strict:
                problems.append(f"{arm} {rel}: screen-commit source {mine} != strict {strict}")
            if row.get("screen_sha256") != mine or row.get("strict_sha256") != strict:
                problems.append(f"{arm} {rel}: recorded parity differs from the audit's")
    return problems


def audit_screen_source_parity(audit: Audit, pilot_root: Path, intent: Mapping[str, Any]) -> None:
    problems = screen_source_problems(pilot_root, intent)
    audit.rule(
        "pilot.screen_source_parity",
        not problems,
        problems[:20],
        "screen intent.json git.{commit,dirty_paths} + git show <commit>:<veto module>",
    )


def _is_lambda(value: Any) -> bool:
    return is_real(value) and value == CANDIDATE_LAMBDA


def lambda_binding_problems(
    pilot_root: Path, sweep_summary: Path, intent: Mapping[str, Any]
) -> List[str]:
    """Why the screen's B arm is not bound to lambda 4.0 (empty list: it is).

    Independently of the producer: the screen receipt's ``lambda`` and
    ``sweep_binding.selected_lambda`` are 4.0, the binding names ``sweep_summary`` and its
    bytes; that sweep summary is a real v7 DEV sweep summary (not a smoke) whose selection
    passed at 4.0, whose ``intent_sha256`` is the intent beside it and whose source commit
    (summary and sweep intent) is the clean pilot commit; the screen intent's arm B and
    hypothesis name lambda 4.0 and the sweep summary's sha256; the strict intent's recorded
    ``pilot.lambda_binding`` agrees.
    """
    receipt_path, screen_path = pilot_root / "receipt.json", pilot_root / "intent.json"
    if not receipt_path.is_file() or not screen_path.is_file():
        return ["screen receipt.json or intent.json missing"]
    if not sweep_summary.is_file() or not (sweep_summary.parent / "intent.json").is_file():
        return [f"sweep summary {sweep_summary} or its intent.json missing"]
    problems: List[str] = []
    receipt, screen = load_json(receipt_path), load_json(screen_path)
    summary, digest = load_json(sweep_summary), sha256_file(sweep_summary)
    sweep_intent_path = sweep_summary.parent / "intent.json"
    sweep_intent, sweep_intent_sha = load_json(sweep_intent_path), sha256_file(sweep_intent_path)
    binding = receipt.get("sweep_binding") if isinstance(receipt.get("sweep_binding"), dict) else {}
    if not _is_lambda(receipt.get("lambda")) or not _is_lambda(binding.get("selected_lambda")):
        problems.append("screen receipt lambda / sweep_binding.selected_lambda is not 4.0")
    if binding.get("summary_path") != str(sweep_summary.resolve()):
        problems.append("screen sweep_binding names another sweep summary")
    if binding.get("summary_sha256") != digest:
        problems.append("screen sweep_binding.summary_sha256 differs from the sweep summary bytes")
    if binding.get("source_commit") != PILOT_COMMIT:
        problems.append("screen sweep_binding.source_commit is not the pilot commit")
    if binding.get("intent_sha256") != summary.get("intent_sha256"):
        problems.append("screen sweep_binding.intent_sha256 differs from the sweep summary")
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    git = sweep_intent.get("git") if isinstance(sweep_intent.get("git"), dict) else {}
    if summary.get("schema_version") != SWEEP_SCHEMA or summary.get("sweep_id") != SWEEP_ID:
        problems.append("sweep summary is not the v7 DEV lambda sweep")
    if summary.get("smoke") is not False:
        problems.append("sweep summary is a smoke")
    if selection.get("status") != "SELECTED" or selection.get("passes") is not True:
        problems.append("sweep selection did not pass")
    if not _is_lambda(selection.get("selected_lambda")):
        problems.append("sweep selected lambda is not 4.0")
    if summary.get("intent_sha256") != sweep_intent_sha:
        problems.append("sweep summary intent_sha256 differs from its intent.json")
    if source.get("commit") != PILOT_COMMIT or git.get("commit") != PILOT_COMMIT:
        problems.append("sweep source commit is not the pilot commit")
    if source.get("dirty_paths") != "" or git.get("dirty_paths") != "":
        problems.append("sweep ran from a dirty tree")
    arms = screen.get("arms") if isinstance(screen.get("arms"), dict) else {}
    arm_b, hypothesis = str(arms.get("B", "")), str(screen.get("hypothesis", ""))
    if CANDIDATE_METHOD not in arm_b or digest not in arm_b:
        problems.append("screen intent arm B does not name lambda=4.0 and the sweep summary")
    if "lambda 4.0" not in hypothesis or digest not in hypothesis:
        problems.append("screen intent hypothesis does not name lambda 4.0 and the sweep summary")
    pilot = intent.get("pilot") if isinstance(intent.get("pilot"), dict) else {}
    recorded = pilot.get("lambda_binding") if isinstance(pilot.get("lambda_binding"), dict) else {}
    if (
        recorded.get("passes") is not True
        or recorded.get("problems")
        or recorded.get("sweep_summary_sha256") != digest
        or recorded.get("sweep_intent_sha256") != sweep_intent_sha
        or not _is_lambda(recorded.get("candidate_lambda"))
    ):
        problems.append("intent pilot.lambda_binding does not match the audit's")
    return problems


def audit_lambda_binding(
    audit: Audit, pilot_root: Path, sweep_summary: Path, intent: Mapping[str, Any]
) -> None:
    problems = lambda_binding_problems(pilot_root, sweep_summary, intent)
    audit.rule(
        "pilot.lambda_binding",
        not problems,
        problems[:20],
        "screen receipt lambda + sweep_binding, sweep summary selection, screen intent arm B",
    )


def audit_calibration(audit: Audit, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    domain, count = NAMESPACES["calibration"]
    bank = stage_bank("calibration")
    expected = {(m, bank[i]) for m in MIXES for i in range(count)}
    got = {(e["mix"], e["seed"]) for e in episodes}
    audit.rule("calibration.worlds_subset", got <= expected, sorted(map(str, got - expected))[:20])
    out: Dict[str, Any] = {"episodes": len(episodes), "complete": got == expected}
    if not out["complete"]:
        return out
    means = {
        m: {
            "mass_integral": mean(
                [e["record"]["mass_integral"] for e in episodes if e["mix"] == m]
            ),
            "survival_fraction": mean(
                [e["record"]["survival_fraction"] for e in episodes if e["mix"] == m]
            ),
        }
        for m in MIXES
    }
    scripted = means[SCRIPTED_MIX]["mass_integral"]
    out.update(
        {
            "reference_means": means,
            "absolute_delta_ni": NI_FRACTION * scripted,
            "margin_valid": scripted > 0,
            "bands": {
                m: {
                    "lower": means[m]["survival_fraction"] + BAND_LOWER_OFFSET,
                    "upper": means[m]["survival_fraction"] + BAND_UPPER_OFFSET,
                }
                for m in MIXES
            },
        }
    )
    return out


def audit_final(
    audit: Audit,
    episodes: Sequence[Mapping[str, Any]],
    n_final: int,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    bank = stage_bank("final")[: max(0, min(n_final, N_MAX))]
    by_key = {(e["arm"], e["mix"], e["seed"]): e for e in episodes}
    outside = sorted(str(k) for k in by_key if k[2] not in bank)
    audit.rule(
        "final.worlds_in_frozen_prefix", not outside, {"n": n_final, "outside": outside[:20]}
    )
    unpaired = []
    for arm, mix, seed in by_key:
        other = by_key.get(("candidate" if arm == "incumbent" else "incumbent", mix, seed))
        if (
            other is not None
            and other["record"]["world_identity"]
            != by_key[(arm, mix, seed)]["record"]["world_identity"]
        ):
            unpaired.append(f"{mix}/{seed}")
    audit.rule("final.pair_identity", not unpaired, unpaired[:20], "record.world_identity")
    complete = bool(bank) and all(
        (arm, m, s) in by_key for arm in ("incumbent", "candidate") for m in MIXES for s in bank
    )
    out: Dict[str, Any] = {"episodes": len(episodes), "complete": complete, "n_final": len(bank)}
    audit.rule(
        "final.requires_calibration",
        not episodes or bool(calibration.get("complete")),
        {"final_episodes": len(episodes)},
    )
    if not (complete and calibration.get("complete") and calibration.get("margin_valid")):
        return out
    deltas = {
        m: [
            by_key[("candidate", m, s)]["record"]["mass_integral"]
            - by_key[("incumbent", m, s)]["record"]["mass_integral"]
            for s in bank
        ]
        for m in MIXES
    }
    decision = strict_decision(deltas, calibration["absolute_delta_ni"])
    measurements = {}
    for m in MIXES:
        value = mean([by_key[("candidate", m, s)]["record"]["survival_fraction"] for s in bank])
        band = calibration["bands"][m]
        measurements[m] = {
            "value": value,
            "lower": band["lower"],
            "upper": band["upper"],
            "passes": band["lower"] <= value <= band["upper"],
        }
    out.update(
        {
            "paired_deltas_by_mix": deltas,
            "decision": decision,
            "behavioral_measurements": measurements,
            "bands_pass": all(row["passes"] for row in measurements.values()),
        }
    )
    return out


def audit_serving(audit: Audit, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bank = stage_bank("serving")
    seeds = [e["seed"] for e in episodes]
    audit.rule(
        "serving.worlds",
        len(seeds) == len(set(seeds)) and set(seeds) <= set(bank),
        {"episodes": len(seeds)},
        "serving namespace apex-veto-v7-strict-serving-v1, serving-selfplay roster",
    )
    return {"episodes": len(seeds), "complete": set(seeds) == set(bank)}


def expected_outcomes(
    n_final: Optional[int],
    calibration: Mapping[str, Any],
    final: Mapping[str, Any],
    serving: Mapping[str, Any],
) -> List[str]:
    """Outcomes consistent with the evidence (INVALID_STOP is checked separately).

    Stage order is calibration -> final -> serving -> audit; a deadline stop anywhere
    before the audit (serving included) is INCOMPLETE, so STRICT_PASS / STRICT_FAIL
    need every stage complete.
    """
    if n_final is None:
        return []
    if n_final > N_MAX:
        return ["STOP_INFEASIBLE"]
    if calibration.get("complete") and not calibration.get("margin_valid"):
        return ["INVALID_STOP"]
    if "decision" not in final or not serving.get("complete"):
        return ["INCOMPLETE", "INVALID_STOP"]
    passes = final["decision"]["valid"] and final["decision"]["passes"] and final["bands_pass"]
    return ["STRICT_PASS" if passes else "STRICT_FAIL"]


def expected_plan(stage: str, n_final: int) -> List[List[str]]:
    """Episode-id units in the producer's order: mix-major, world-minor (strict_run)."""
    if stage == "serving":
        return [[f"serving-candidate-{SERVING_MIX}-{s}"] for s in stage_bank("serving")]
    bank = stage_bank(stage) if stage == "calibration" else stage_bank("final")[:n_final]
    arms = STAGE_ARMS[stage]
    return [[f"{stage}-{arm}-{m}-{s}" for arm in arms] for m in MIXES for s in bank]


def audit_shards(
    audit: Audit,
    root: Path,
    docs: Mapping[str, Any],
    raw: Mapping[str, Sequence[Tuple[str, Any]]],
    n_final: Optional[int],
    plans: Optional[Mapping[str, List[List[str]]]] = None,
) -> None:
    """Shard reports: plan, deterministic j % 2 assignment, record hashes, no extras.

    ``plans`` replaces :func:`expected_plan` per stage (the smoke dry-run's own plan).
    """
    problems: List[str] = []
    for stage in STAGES:
        reports = {
            name: doc
            for name, doc in docs.items()
            if isinstance(doc, dict)
            and "planned_episode_ids" in doc
            and "records_sha256" in doc
            and STAGE_ALIASES.get(doc.get("stage")) == stage
        }
        on_disk = {name: entry for name, entry in raw[stage]}
        if not reports and not on_disk:
            continue
        if not reports:
            problems.append(f"{stage}: records without shard reports")
            continue
        units = plans[stage] if plans is not None else expected_plan(stage, n_final or 0)
        listed: Dict[str, str] = {}
        seen_shards = set()
        for name, report in sorted(reports.items()):
            shard = report.get("shard")
            seen_shards.add(shard)
            want = [i for j, unit in enumerate(units) if j % WORKERS == shard for i in unit]
            if report["planned_episode_ids"] != want:
                problems.append(f"{name}: planned ids differ from the j % {WORKERS} plan")
            records_dir = (root / name).parent.parent / "records"
            for episode_id, sha in report["records_sha256"].items():
                path = records_dir / f"{episode_id}.json"
                if episode_id not in want:
                    problems.append(f"{name}: unplanned record {episode_id}")
                elif not path.is_file():
                    problems.append(f"{name}: missing record {episode_id}")
                elif sha256_file(path) != sha:
                    problems.append(f"{name}: record bytes changed {episode_id}")
                else:
                    entry = on_disk.get(str(path.relative_to(root)), {})
                    if isinstance(entry, dict) and entry.get("shard", shard) != shard:
                        problems.append(f"{name}: record {episode_id} names another shard")
                listed[str(path.relative_to(root))] = episode_id
        if seen_shards != set(range(WORKERS)):
            problems.append(f"{stage}: shard reports {sorted(map(str, seen_shards))}")
        extra = sorted(set(on_disk) - set(listed))
        if extra:
            problems.append(f"{stage}: records not listed by any shard report {extra[:5]}")
    audit.rule(
        "shards.plan_assignment_and_hashes",
        not problems,
        problems[:30],
        "strict_run worker shard reports (planned_episode_ids, records_sha256)",
    )


# ----------------------------------------------------------------------------- producer claims


def _nodes(docs: Mapping[str, Any], predicate: Any) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (f"{name}:{path}", node)
        for name, doc in docs.items()
        for path, node in walk(doc)
        if isinstance(node, dict) and predicate(node)
    ]


def _compare_statistics(where: str, claim: Mapping[str, Any], mine: Mapping[str, Any]) -> List[str]:
    """Field-by-field comparison with eval_stats.strict_promotion_decision output."""
    diffs: List[str] = []

    def diff(label: str, ok: bool) -> None:
        if not ok:
            diffs.append(f"{where}: {label}")

    sup = claim.get("superiority") if isinstance(claim.get("superiority"), dict) else {}
    ni = claim.get("scripted_noninferiority")
    ni = ni if isinstance(ni, dict) else {}
    diff("valid", claim.get("valid") is mine["valid"])
    diff("passes", claim.get("passes") is mine["passes"])
    diff("superiority.valid", sup.get("valid") is mine["valid"])
    diff("superiority.passes", sup.get("passes") is mine["superiority_passes"])
    diff("noninferiority.passes", ni.get("passes") is mine["ni_passes"])
    per_mix = sup.get("per_mix") if isinstance(sup.get("per_mix"), dict) else {}
    for m in MIXES:
        row, own = per_mix.get(m) or {}, mine["per_mix"][m]
        diff(f"{m}.n", row.get("n") == own["n"])
        for key in ("mean_delta", "sample_std", "standard_error", "critical_value", "t_statistic"):
            diff(f"{m}.{key}", close(row.get(key), own.get(key)))
        diff(
            f"{m}.p_value",
            p_values_agree(own["p_value"], row.get("p_value"), own.get("t_statistic")),
        )
        diff(f"{m}.superior", row.get("superior") is own["superior"])
    if mine["valid"]:
        adjusted = sup.get("adjusted_p_values") or {}
        rejected = sup.get("rejected") or {}
        for m in MIXES:
            t_stat = mine["per_mix"][m].get("t_statistic")
            ok = p_values_agree(mine["adjusted_p_values"][m], adjusted.get(m), t_stat)
            diff(f"{m}.adjusted_p", ok)
            diff(f"{m}.rejected", rejected.get(m) is mine["rejected"][m])
        diff(
            "successful_mixes",
            sorted(sup.get("successful_mixes") or []) == sorted(mine["successful_mixes"]),
        )
        diff("ni.lower_bound", close(ni.get("lower_bound"), mine["ni_lower_bound"]))
        diff("ni.absolute_delta_ni", close(ni.get("absolute_delta_ni"), mine["absolute_delta_ni"]))
        diff("ni.n", ni.get("n") == mine["per_mix"][SCRIPTED_MIX]["n"])
    return diffs


def compare_producer_claims(
    audit: Audit,
    docs: Mapping[str, Any],
    sizing: Optional[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    n_final = sizing["required_final_worlds"] if sizing else None
    claims = [v for _, v in find_key(docs, "required_final_worlds")]
    audit.rule(
        "claims.required_final_worlds",
        bool(claims) and all(v == n_final for v in claims),
        {"audit": n_final, "producer": claims},
        "eval_stats.paired_delta_pilot_size on the screen deltas, MDE 30",
    )
    diffs: List[str] = []
    for where, node in _nodes(docs, lambda n: n.get("method") == SIZING_METHOD):
        rows = node.get("per_mix") if isinstance(node.get("per_mix"), dict) else {}
        for m in MIXES:
            row, own = rows.get(m) or {}, sizing["per_mix"][m] if sizing else {}
            for key in ("n", "recommended_n"):
                diffs += [f"{where}.{m}.{key}"] if row.get(key) != own.get(key) else []
            for key in ("paired_delta_std", "mde"):
                diffs += [f"{where}.{m}.{key}"] if not close(row.get(key), own.get(key)) else []
    for where, value in find_key(docs, "mde_by_mix"):
        if not (isinstance(value, dict) and all(close(value.get(m), MDE_ABSOLUTE) for m in MIXES)):
            diffs.append(where)
    for where, value in find_key(docs, "mde_absolute_per_mix"):
        diffs += [] if close(value, MDE_ABSOLUTE) else [where]
    for key in ("candidate_lambda", "space_preference_lambda"):
        for where, value in find_key(docs, key):
            diffs += [] if is_real(value) and value == CANDIDATE_LAMBDA else [where]
    audit.rule("claims.sizing", not diffs, diffs[:30])

    margin = calibration.get("absolute_delta_ni")
    claims = find_key(docs, "absolute_delta_ni")
    audit.rule(
        "claims.absolute_delta_ni",
        (bool(claims) or not calibration.get("complete"))
        and all(close(v, margin) for _, v in claims),
        {"audit": margin, "producer": [v for _, v in claims]},
        "0.03 x mean incumbent scripted mass_integral over the 16 dev worlds",
    )
    bands = calibration.get("bands") or {}
    band_diffs: List[str] = []
    seen = set()
    band_like = _nodes(
        docs,
        lambda n: n.get("metric") == "survival_fraction"
        and "lower" in n
        and "upper" in n
        and isinstance(n.get("mix_scope"), list),
    )
    for where, node in band_like:
        scope = node["mix_scope"]
        mix = scope[0] if len(scope) == 1 and scope[0] in MIXES else None
        if mix is None or mix not in bands:
            band_diffs.append(f"{where}: unverifiable band scope {scope}")
            continue
        seen.add(mix)
        if not (
            close(node["lower"], bands[mix]["lower"]) and close(node["upper"], bands[mix]["upper"])
        ):
            band_diffs.append(f"{where}: bounds")
        for key, expected in (
            ("lower_offset", BAND_LOWER_OFFSET),
            ("upper_offset", BAND_UPPER_OFFSET),
        ):
            if key in node and not close(node[key], expected):
                band_diffs.append(f"{where}: {key}")
        if "value" in node:
            own = (final.get("behavioral_measurements") or {}).get(mix)
            if (
                own is None
                or not close(node["value"], own["value"])
                or node.get("passes") is not own["passes"]
            ):
                band_diffs.append(f"{where}: measurement")
    if calibration.get("complete") and seen != set(MIXES):
        band_diffs.append(f"band definitions found for {sorted(seen)} only")
    audit.rule(
        "claims.behavioral_bands", not band_diffs, band_diffs[:30], "reference mean + offsets"
    )

    stats = _nodes(docs, lambda n: n.get("decision_method") == DECISION_METHOD)
    stat_diffs: List[str] = []
    mine = final.get("decision")
    for where, node in stats:
        if mine is None:
            if node.get("passes") is not False:
                stat_diffs.append(f"{where}: passing statistics without complete final evidence")
            continue
        stat_diffs += _compare_statistics(where, node, mine)
    if mine is not None and not stats:
        stat_diffs.append("no strict-promotion-v1 statistics found in producer documents")
    for where, value in find_key(docs, "paired_deltas_by_mix"):
        own = final.get("paired_deltas_by_mix")
        ok = (
            own is not None
            and isinstance(value, dict)
            and all(
                isinstance(value.get(m), list)
                and len(value[m]) == len(own[m])
                and all(close(a, b) for a, b in zip(value[m], own[m]))
                for m in MIXES
            )
        )
        stat_diffs += [] if ok else [where]
    audit.rule("claims.statistics", not stat_diffs, stat_diffs[:40], "own t/Holm/NI recomputation")


def compare_design_claims(
    audit: Audit,
    docs: Mapping[str, Any],
    pilot: Optional[Mapping[str, Sequence[float]]],
    sizing: Optional[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    """Intent / calibration / decision claims beyond the strict statistics."""
    diffs: List[str] = []
    n_final = sizing["required_final_worlds"] if sizing else None
    for where, node in _nodes(docs, lambda n: "final_worlds_per_mix" in n):
        if node["final_worlds_per_mix"] != n_final:
            diffs.append(f"{where}.final_worlds_per_mix")
        if "feasible" in node and node["feasible"] is not (
            n_final is not None and n_final <= N_MAX
        ):
            diffs.append(f"{where}.feasible")
    for where, seeds in find_key(docs, "final_seeds"):
        if n_final is None or seeds != stage_bank("final")[: min(n_final, N_MAX)]:
            diffs.append(where)
    stage_of_namespace = {"dev": "calibration", "final": "final", "serving": "serving"}
    for where, value in find_key(docs, "namespaces"):
        rows = value.items() if isinstance(value, dict) else []
        for name, row in rows:
            stage = stage_of_namespace.get(name)
            if isinstance(row, dict) and "seeds" in row:
                if stage is None or row["seeds"] != stage_bank(stage):
                    diffs.append(f"{where}.{name}.seeds")
                if (
                    stage is not None
                    and row.get("domain", NAMESPACES[stage][0]) != NAMESPACES[stage][0]
                ):
                    diffs.append(f"{where}.{name}.domain")
    for where, node in _nodes(docs, lambda n: isinstance(n.get("pilot"), dict)):
        claimed = node["pilot"].get("deltas_by_mix")
        if claimed is not None and not (
            pilot is not None
            and isinstance(claimed, dict)
            and all(
                len(claimed.get(m) or []) == len(pilot[m])
                and all(close(a, b) for a, b in zip(claimed[m], pilot[m]))
                for m in MIXES
            )
        ):
            diffs.append(f"{where}.pilot.deltas_by_mix")
    means = calibration.get("reference_means") or {}
    for where, value in find_key(docs, "per_mix"):
        if not isinstance(value, dict):
            continue
        for mix, row in value.items():
            if not (isinstance(row, dict) and "mean_mass_integral" in row):
                continue
            own = means.get(mix)
            if (
                own is None
                or not close(row["mean_mass_integral"], own["mass_integral"])
                or not close(row.get("mean_survival_fraction"), own["survival_fraction"])
                or row.get("n", NAMESPACES["calibration"][1]) != NAMESPACES["calibration"][1]
            ):
                diffs.append(f"{where}.{mix}")
    for where, node in _nodes(docs, lambda n: "reference_mean" in n and "mix_scope" in n):
        scope = node["mix_scope"]
        own = means.get(scope[0]) if isinstance(scope, list) and len(scope) == 1 else None
        if own is None or not close(node["reference_mean"], own["survival_fraction"]):
            diffs.append(f"{where}.reference_mean")
    mine = final.get("decision")
    for where, node in _nodes(docs, lambda n: "strict_promotion_decision" in n):
        if mine is None:
            diffs.append(f"{where}: decision without complete final evidence")
            continue
        overall = bool(mine["valid"] and mine["passes"] and final["bands_pass"])
        if node.get("bands_pass") is not final["bands_pass"]:
            diffs.append(f"{where}.bands_pass")
        if node.get("valid") is not mine["valid"] or node.get("passes") is not overall:
            diffs.append(f"{where}.valid/passes")
        claimed = node.get("mean_delta_by_mix") or {}
        for m in MIXES:
            if m in claimed and not close(claimed[m], mine["per_mix"][m]["mean_delta"]):
                diffs.append(f"{where}.mean_delta_by_mix.{m}")
    audit.rule("claims.design_and_calibration", not diffs, diffs[:30], "intent, calibration.json")


def audit_outcome(audit: Audit, docs: Mapping[str, Any], expected: Sequence[str]) -> Optional[str]:
    values = [
        (where, value)
        for key in ("outcome", "decision")
        for where, value in find_key(docs, key)
        if isinstance(value, str) and value in OUTCOMES
    ]
    distinct = sorted({value for _, value in values})
    claimed = distinct[0] if len(distinct) == 1 else None
    reasons = [
        v
        for key in ("stop_reason", "invalid_reason", "stopped_reason", "failure")
        for _, v in find_key(docs, key)
        if isinstance(v, str) and v
    ]
    ok = claimed in expected or (claimed == "INVALID_STOP" and bool(reasons))
    audit.rule(
        "claims.outcome",
        ok,
        {"producer": distinct, "audit_expected": list(expected), "stop_reasons": reasons[:5]},
        "expected_outcomes(): sizing cap, calibration, completeness, strict decision + bands",
    )
    return claimed


def _arm_node(intent: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    node = intent.get(arm)
    return node if isinstance(node, dict) else {}


def audit_identity(
    audit: Audit,
    docs: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any],
) -> None:
    """Arm identities across producer documents and records.

    The incumbent's wrapper source sha256s are the released v5 bytes of v5, v2 and v3
    (pinned here); the candidate's descriptor is lambda 4.0; every ``wrapper_source_sha256``
    in the producer documents is one of the two arms' values; every ``wrapper_identity``
    node names a known method with that method's source sha; every record's envelope sha
    is its own arm's.
    """
    claimed = {arm: _arm_node(intent, arm).get("wrapper_source_sha256") for arm in ARM_METHODS}
    incumbent_identity = _arm_node(intent, "incumbent").get("wrapper_identity")
    incumbent_identity = incumbent_identity if isinstance(incumbent_identity, dict) else {}
    candidate_identity = _arm_node(intent, "candidate").get("wrapper_identity")
    candidate_identity = candidate_identity if isinstance(candidate_identity, dict) else {}
    descriptor = candidate_identity.get("descriptor")
    lam = descriptor.get("space_preference_lambda") if isinstance(descriptor, dict) else None
    released_ok = incumbent_identity.get("source_sha256s") == INCUMBENT_SOURCE_SHA256S
    lambda_ok = is_real(lam) and lam == CANDIDATE_LAMBDA
    by_method = {ARM_METHODS[arm]: sha for arm, sha in claimed.items()}
    values = {v for _, v in find_key(docs, "wrapper_source_sha256")}
    identities = [v for _, v in find_key(docs, "wrapper_identity") if isinstance(v, dict)]
    identity_ok = bool(identities) and all(
        v.get("method") in by_method and v.get("source_sha256") == by_method[v.get("method")]
        for v in identities
    )
    wrong_records = sorted(
        str(e["entry"].get("episode_id", e["entry"].get("world_seed")))
        for e in episodes
        if e["entry"].get("wrapper_source_sha256")
        != claimed.get(ARM_ALIASES.get(e["entry"].get("arm"), ""))
    )
    audit.rule(
        "identity.wrapper_source_sha256",
        claimed["incumbent"] == INCUMBENT_SOURCE_SHA256
        and released_ok
        and lambda_ok
        and is_sha256(claimed["candidate"])
        and claimed["candidate"] != claimed["incumbent"]
        and values <= set(claimed.values())
        and identity_ok
        and not wrong_records,
        {
            "intent": claimed,
            "released_incumbent": INCUMBENT_SOURCE_SHA256S,
            "incumbent_sources_released": released_ok,
            "candidate_lambda": lam,
            "producer": sorted(map(str, values)),
            "wrapper_identity_consistent": identity_ok,
            "records_with_other_arm_sha": wrong_records[:20],
        },
        "arm identities {checkpoint, wrapper, wrapper source sha256s} in the intent",
    )
    wrong = [
        f"intent.{arm}"
        for arm, method in ARM_METHODS.items()
        if _arm_node(intent, arm).get("wrapper") != method
        or _arm_node(intent, arm).get("checkpoint_sha256") != CHAMPION_SHA256
    ]
    methods = set(ARM_METHODS.values())
    wrong += [
        w
        for w, n in _nodes(docs, lambda n: n.get("wrapper") in methods)
        if n.get("checkpoint_sha256", CHAMPION_SHA256) != CHAMPION_SHA256
    ]
    wrong += [w for w, v in find_key(docs, "hero_sha256") if v != CHAMPION_SHA256]
    audit.rule(
        "identity.checkpoint_and_wrapper",
        not wrong,
        {"mismatches": wrong[:20]},
        "intent incumbent (champion + v5) / candidate (champion + v7); hero_sha256 fields",
    )


def audit_arm_binding(
    audit: Audit, intent: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> None:
    """Bind both arm identities to the frozen source and to every record.

    Rules: (1) ``identity.wrapper_source_bound_to_closure``: for each arm, the intent's
    ``wrapper_identity.source_sha256s`` covers exactly the arm's veto modules and each
    equals the source-closure hash of that module; ``wrapper_source_sha256`` is the
    primary module's. (2) ``identity.arm_records_bound``: every record carries its arm's
    ``wrapper_source_sha256(s)`` and a ``probes.safety_veto`` equal (without ``counters``)
    to its arm's descriptor.
    """
    closure = intent.get("source_closure") if isinstance(intent.get("source_closure"), dict) else {}
    files = closure.get("files") if isinstance(closure.get("files"), dict) else {}
    detail: Dict[str, Any] = {}
    ok = True
    identities: Dict[str, Dict[str, Any]] = {}
    for arm, method in ARM_METHODS.items():
        node = _arm_node(intent, arm)
        identity = node.get("wrapper_identity")
        identity = identity if isinstance(identity, dict) else {}
        identities[arm] = identity
        shas = identity.get("source_sha256s")
        shas = shas if isinstance(shas, dict) else {}
        mismatched = []
        for rel in ARM_SOURCES[arm]:
            matches = [sha for path, sha in files.items() if str(path).endswith("/" + rel)]
            if len(matches) != 1 or shas.get(rel) != matches[0] or not is_sha256(matches[0]):
                mismatched.append(rel)
        primary = shas.get(ARM_SOURCES[arm][0])
        arm_ok = (
            not mismatched
            and sorted(shas) == sorted(ARM_SOURCES[arm])
            and node.get("wrapper_source_sha256") == primary
            and identity.get("source_sha256") == primary
            and identity.get("method") == method
        )
        ok = ok and arm_ok
        detail[arm] = {"mismatched": mismatched, "claimed": node.get("wrapper_source_sha256")}
    audit.rule(
        "identity.wrapper_source_bound_to_closure",
        ok,
        detail,
        "intent.source_closure.files[...src/evaluation/safety_veto{,_v3,_v5,_v7}.py]",
    )
    descriptors = {arm: identities[arm].get("descriptor") for arm in ARM_METHODS}
    descriptor_ok = all(
        isinstance(d, dict) and d.get("method") == ARM_METHODS[arm]
        for arm, d in descriptors.items()
    )
    unbound: List[str] = []
    for entry in entries:
        arm = ARM_ALIASES.get(entry.get("arm"))
        if arm not in ARM_METHODS:
            continue
        identity = identities[arm]
        record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
        probes = record.get("probes") if isinstance(record.get("probes"), dict) else {}
        veto = probes.get("safety_veto") if isinstance(probes.get("safety_veto"), dict) else {}
        where = str(entry.get("episode_id", entry.get("world_seed")))
        if entry.get("wrapper_source_sha256") != identity.get("source_sha256"):
            unbound.append(f"{where}: wrapper_source_sha256")
        if entry.get("wrapper_source_sha256s") != identity.get("source_sha256s"):
            unbound.append(f"{where}: wrapper_source_sha256s")
        if {k: v for k, v in veto.items() if k != "counters"} != descriptors[arm]:
            unbound.append(f"{where}: safety_veto descriptor")
    audit.rule(
        "identity.arm_records_bound",
        descriptor_ok and not unbound,
        {"descriptors": descriptors, "unbound": unbound[:20], "unbound_count": len(unbound)},
        "record envelopes vs intent.{incumbent,candidate}.wrapper_identity",
    )


def audit_counters(audit: Audit, docs: Mapping[str, Any]) -> None:
    bad = [w for w, v in find_key(docs, "optimizer_updates") if v != 0]
    over: List[str] = []
    for where, node in _nodes(
        docs, lambda n: isinstance(n.get("caps"), dict) and isinstance(n.get("counters"), dict)
    ):
        for key, value in node["counters"].items():
            cap = node["caps"].get(key)
            if is_real(value) and is_real(cap) and value > cap:
                over.append(f"{where}.counters.{key} {value} > {cap}")
    audit.rule(
        "counters.optimizer_updates_zero_and_at_most_cap",
        not bad and not over,
        {"optimizer_updates_nonzero": bad[:20], "over_cap": over[:20]},
        "producer physical counters where present (never required)",
    )


def audit_artifacts(
    audit: Audit, root: Path, docs: Mapping[str, Any], episodes_played: bool
) -> Dict[str, Any]:
    """Re-hash listed {path, sha256} files, snapshot maps and every .pth under the root.

    Rollouts load only the snapshots (dev_screen.snapshot_checkpoints), so once any
    episode exists the four pool checkpoints must be present and intact.
    """
    problems: List[str] = []
    outside: List[str] = []
    for where, node in _nodes(
        docs, lambda n: isinstance(n.get("path"), str) and is_sha256(n.get("sha256"))
    ):
        doc_dir = (root / where.split(":", 1)[0]).parent
        raw = Path(node["path"])
        options = [raw] if raw.is_absolute() else [doc_dir / raw, root / raw]
        options = [
            p
            for p in options
            if p.resolve() == root.resolve() or root.resolve() in p.resolve().parents
        ]
        target = next((p for p in options if p.is_file()), None)
        if target is not None:
            if sha256_file(target) != node["sha256"]:
                problems.append(f"{where}: sha256 mismatch {node['path']}")
        elif not options or (not raw.is_absolute() and raw.parts[0] in REPO_TOP_LEVEL):
            outside.append(node["path"])
        else:
            problems.append(f"{where}: missing {node['path']}")
    snapshots = {}
    for path in sorted(root.rglob("*.pth")):
        digest = sha256_file(path)
        snapshots[str(path.relative_to(root))] = digest
        if is_sha256(path.stem) and path.stem != digest:
            problems.append(f"snapshot {path.name} hashes to {digest}")
    missing_pool = sorted(set(POOL_SHA256S) - set(snapshots.values()))
    if episodes_played and missing_pool:
        problems.append(f"pool checkpoint snapshots missing: {missing_pool}")
    for where, value in find_key(docs, "checkpoint_snapshots"):
        for sha, raw_path in (value.items() if isinstance(value, dict) else []):
            target = Path(str(raw_path))
            # Resolve the directory only: a snapshot file may itself be a symlink.
            parent = target.parent.resolve() if target.is_absolute() else None
            inside = parent is not None and (parent == root or root in parent.parents)
            if not inside:
                outside.append(str(raw_path))
            elif not target.is_file() or sha256_file(target) != sha:
                problems.append(f"{where}: snapshot {sha} missing or changed")
    audit.rule(
        "artifacts.hashes",
        not problems,
        {"problems": problems[:30], "outside_root_not_checked": outside[:20]},
        "producer artifact listings {path, sha256}; checkpoint snapshots named by sha256",
    )
    return {
        "snapshots": snapshots,
        "champion_snapshot_present": CHAMPION_SHA256 in snapshots.values(),
    }


def evidence_manifest(root: Path, out_dir: Path) -> Dict[str, str]:
    out_resolved = out_dir.resolve()
    manifest = {}
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if path.is_file() and resolved != out_resolved and out_resolved not in resolved.parents:
            if path.suffix in (".json", ".jsonl", ".md"):
                manifest[str(path.relative_to(root))] = sha256_file(path)
    return manifest


# ----------------------------------------------------------------------------- smoke dry-run
# Governance rule 5: the producer and this audit run end to end on a smoke output before GO.
# Smoke records come from the legacy truncated rollout (<= 500 frames: no evaluation
# profile, no denominators) on one scripted world of the smoke namespace, so the H5000
# record, sizing, calibration and decision rules do not apply. The envelope, roster,
# wrapper identity, shard plan and hashes, counters and artifact rules do, and no document
# may claim a Tier-2 outcome. Selected only by ``--smoke``, and only for a smoke intent.

SMOKE_MAX_FRAMES = 500
SMOKE_DECISION_DOCS = ("calibration.json", "decision.json", "producer-outcome.json", "receipt.json")


def smoke_seed() -> int:
    return uint32_seed(SMOKE_DOMAIN, NAMESPACE_LABEL, 0)


def smoke_plan(seed: int) -> Dict[str, List[List[str]]]:
    """strict_run's smoke plan: one scripted final pair, no calibration or serving."""
    pair = [f"final-{arm}-{SCRIPTED_MIX}-{seed}" for arm in STAGE_ARMS["final"]]
    return {"calibration": [], "final": [pair], "serving": []}


def validate_smoke_entry(entry: Any, seed: int, frames: int) -> List[str]:
    """Envelope, roster and wrapper rules for one legacy-path smoke record."""
    problems: List[str] = []

    def need(ok: bool, message: str) -> bool:
        if not ok:
            problems.append(message)
        return ok

    if not need(isinstance(entry, dict) and isinstance(entry.get("record"), dict), "no record"):
        return problems
    record, arm = entry["record"], entry.get("arm")
    need(arm in STAGE_ARMS["final"], f"arm {arm!r}")
    need(entry.get("stage") == "final" and entry.get("mix") == SCRIPTED_MIX, "stage/mix")
    need(entry.get("world_seed") == seed and record.get("seed") == seed, "not the smoke world")
    need(entry.get("world_index") == 0, "world_index is not 0")
    need(entry.get("episode_id") == f"final-{arm}-{SCRIPTED_MIX}-{seed}", "episode_id")
    need(entry.get("hero_sha256") == CHAMPION_SHA256, "hero_sha256 is not the champion")
    need(entry.get("roster_member_sha256s") == expected_roster(SCRIPTED_MIX, 0), "roster")
    for path, node in walk(entry):
        if isinstance(node, dict) and "optimizer_updates" in node:
            need(node["optimizer_updates"] == 0, f"{path}.optimizer_updates != 0")
    for path, key, value in _frame_counters(entry):
        need(is_real(value) and 0 <= value <= frames, f"{path}.{key} not in [0, {frames}]")
    mass, surv = record.get("mass_integral"), record.get("survival_fraction")
    need(is_real(mass) and mass >= 0, "mass_integral not a finite non-negative real")
    need(is_real(surv) and 0 <= surv <= 1, "survival_fraction not in [0, 1]")
    if arm in ARM_METHODS:
        problems.extend(veto_problems(entry, arm, frames))
    return problems


def run_smoke_audit(root: Path, out_dir: Path) -> Dict[str, Any]:
    """The pre-GO dry-run audit of a ``prepare --smoke-frames`` run (see block comment)."""
    audit = Audit()
    raw, docs = discover(root, out_dir)
    intent = docs.get("intent.json") if isinstance(docs.get("intent.json"), dict) else {}
    seed, frames = smoke_seed(), intent.get("smoke_frames")
    namespaces = intent.get("namespaces") if isinstance(intent.get("namespaces"), dict) else {}
    final_ns = namespaces.get("final") if isinstance(namespaces.get("final"), dict) else {}
    empty = all(
        isinstance(namespaces.get(name), dict) and namespaces[name].get("seeds") == []
        for name in ("dev", "serving")
    )
    smoke_ok = is_int(frames) and 0 < frames <= SMOKE_MAX_FRAMES
    audit.rule(
        "smoke.intent",
        smoke_ok
        and intent.get("mixes") == [SCRIPTED_MIX]
        and intent.get("final_seeds") == [seed]
        and final_ns.get("domain") == SMOKE_DOMAIN
        and final_ns.get("seeds") == [seed]
        and empty,
        {"smoke_frames": frames, "final_seeds": intent.get("final_seeds"), "seed": seed},
        "strict_run.build_intent(smoke_frames=...)",
    )
    frames = frames if smoke_ok else 0
    audit.rule(
        "records.inside_stage_dirs", not raw["<no-stage>"], [n for n, _ in raw["<no-stage>"]][:20]
    )
    audit.rule(
        "smoke.no_tier2_stages",
        not raw["calibration"] and not raw["serving"],
        {"calibration": len(raw["calibration"]), "serving": len(raw["serving"])},
    )
    bad = []
    for name, entry in raw["final"]:
        problems = validate_smoke_entry(entry, seed, frames)
        if problems:
            bad.append({"record": name, "problems": problems})
    arms = sorted(str(e.get("arm")) for _, e in raw["final"] if isinstance(e, dict))
    audit.rule(
        "smoke.records_valid",
        not bad and arms == ["candidate", "incumbent"],
        {"records": len(raw["final"]), "arms": arms, "invalid": bad[:10]},
        "tournament_eval.rollout legacy record + strict_run.episode_entry envelope",
    )
    audit_shards(audit, root, docs, raw, None, plans=smoke_plan(seed))
    claims = [
        f"{where}={value}"
        for key in ("outcome", "decision")
        for where, value in find_key(docs, key)
        if isinstance(value, str) and value in OUTCOMES
    ]
    decision_docs = sorted(n for n in docs if Path(n).name in SMOKE_DECISION_DOCS)
    audit.rule(
        "smoke.no_decision_claims",
        not claims and not decision_docs,
        {"outcome_claims": claims[:10], "decision_documents": decision_docs},
        "a smoke run never decides (closeout SMOKE_NO_DECISION is written after the audit)",
    )
    smoke_entries = [e for _, e in raw["final"] if isinstance(e, dict)]
    audit_identity(audit, docs, [{"entry": e} for e in smoke_entries], intent)
    audit_arm_binding(audit, intent, smoke_entries)
    audit_counters(audit, docs)
    artifacts = audit_artifacts(audit, root, docs, bool(raw["final"]))
    failures = audit.failures()
    return {
        "schema_version": AUDIT_SCHEMA,
        "mode": "smoke",
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_frames": frames,
        "reported_not_gated": {"checkpoint_snapshots": artifacts, "stage_documents": sorted(docs)},
        "failures": failures,
        "rules": audit.rules,
        "evidence_sha256": evidence_manifest(root, out_dir),
        "non_claims": {"promotion": False, "decision": False, "release_authorized": False},
    }


# ----------------------------------------------------------------------------- driver


def run_audit(
    root: Path, out_dir: Path, pilot_root: Path, sweep_summary: Path = DEFAULT_SWEEP_SUMMARY
) -> Dict[str, Any]:
    audit = Audit()
    raw, docs = discover(root, out_dir)
    intent = docs.get("intent.json") if isinstance(docs.get("intent.json"), dict) else {}
    audit.rule(
        "mode.not_smoke",
        intent.get("smoke_frames") is None,
        {"smoke_frames": intent.get("smoke_frames")},
        "a smoke intent is audited only by --smoke",
    )
    audit.rule(
        "records.inside_stage_dirs", not raw["<no-stage>"], [n for n, _ in raw["<no-stage>"]][:20]
    )
    audit_namespaces(audit)
    pilot = load_pilot(audit, pilot_root)
    audit_screen_source_parity(audit, pilot_root, intent)
    audit_lambda_binding(audit, pilot_root, sweep_summary, intent)
    sizing = pilot_sizing(pilot) if pilot else None
    n_final = sizing["required_final_worlds"] if sizing else None
    audit.rule(
        "sizing.not_borderline",
        sizing is not None
        and not any(r["ceil_input_borderline"] for r in sizing["per_mix"].values()),
        sizing,
    )
    episodes = {stage: validate_stage(audit, stage, raw[stage]) for stage in STAGES}
    calibration = audit_calibration(audit, episodes["calibration"])
    if n_final is not None and n_final > N_MAX:
        audit.rule(
            "final.absent_when_infeasible", not raw["final"], {"final_records": len(raw["final"])}
        )
    final = audit_final(audit, episodes["final"], n_final or 0, calibration)
    serving = audit_serving(audit, episodes["serving"])
    compare_producer_claims(audit, docs, sizing, calibration, final)
    compare_design_claims(audit, docs, pilot, sizing, calibration, final)
    audit_shards(audit, root, docs, raw, n_final)
    expected = expected_outcomes(n_final, calibration, final, serving)
    claimed = audit_outcome(audit, docs, expected)
    all_episodes = [e for stage in STAGES for e in episodes[stage]]
    audit_identity(audit, docs, all_episodes, intent)
    # Binding covers every record found, including ones another rule already rejected.
    raw_entries = [entry for stage in STAGES for _, entry in raw[stage] if isinstance(entry, dict)]
    audit_arm_binding(audit, intent, raw_entries)
    audit_counters(audit, docs)
    artifacts = audit_artifacts(audit, root, docs, bool(all_episodes))
    veto_consistency = {
        arm: sum(
            1
            for e in all_episodes
            if e["arm"] == arm
            and e["record"]["probes"]["safety_veto"]["counters"].get("decisions")
            == (e["record"].get("denominators") or {}).get("decision_frames")
        )
        for arm in ARM_METHODS
    }

    def diag_sum(arm: str, key: str, nested: Optional[str] = None) -> int:
        total = 0
        for e in all_episodes:
            node = e["entry"].get("veto_diagnostics") if e["arm"] == arm else None
            node = node.get(nested) if isinstance(node, dict) and nested else node
            total += int(node.get(key) or 0) if isinstance(node, dict) else 0
        return total

    diagnostics_reported = {
        "incumbent_boost_landing_vetoes": diag_sum("incumbent", "boost_landing_vetoes"),
        "candidate_rerank_changes": diag_sum("candidate", "rerank_changes"),
        "candidate_rerank_changes_tail_release_driven": diag_sum(
            "candidate", "rerank_changes_tail_release_driven"
        ),
        "candidate_v5_boost_landing_vetoes": diag_sum("candidate", "boost_landing_vetoes", "v5"),
    }
    failures = audit.failures()
    final_view = {k: v for k, v in final.items() if k != "paired_deltas_by_mix"}
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "pilot_root": str(pilot_root),
        "sweep_summary": str(sweep_summary),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "producer_outcome": claimed,
        "audit_expected_outcomes": expected,
        "sizing": sizing,
        "calibration": calibration,
        "final": final_view,
        "paired_deltas_by_mix": final.get("paired_deltas_by_mix"),
        "serving": serving,
        "reported_not_gated": {
            "veto_decisions_equal_decision_frames_by_arm": veto_consistency,
            "veto_diagnostics": diagnostics_reported,
            "checkpoint_snapshots": artifacts,
            "stage_documents": sorted(docs),
        },
        "failures": failures,
        "rules": audit.rules,
        "evidence_sha256": evidence_manifest(root, out_dir),
        "non_claims": {
            "promotion": False,
            "champion_or_default_change": False,
            "release_authorized": False,
            "serving_path_qualified_by_audit": False,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_create_only(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="run root (read-only)")
    parser.add_argument("--out", required=True, type=Path, help="audit output dir")
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--sweep-summary", type=Path, default=DEFAULT_SWEEP_SUMMARY)
    parser.add_argument(
        "--smoke", action="store_true", help="pre-GO dry-run audit of a smoke intent only"
    )
    args = parser.parse_args(argv)
    root, out_dir = args.root.resolve(), args.out.resolve()
    try:
        if not root.is_dir():
            raise AuditError(f"--root {root} is not a directory")
        if args.smoke:
            result = run_smoke_audit(root, out_dir)
        else:
            result = run_audit(
                root, out_dir, args.pilot_root.resolve(), args.sweep_summary.resolve()
            )
    except (AuditError, KeyError, TypeError, ValueError, ArithmeticError, OSError) as exc:
        result = {
            "schema_version": AUDIT_SCHEMA,
            "status": "FAIL",
            "root": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "non_claims": {"promotion": False, "release_authorized": False},
        }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_create_only(out_dir / "audit.json", result)
    except OSError as exc:
        print(json.dumps({"status": "FAIL", "error": f"cannot write audit: {exc}"}))
        return 1
    summary = {
        "status": result["status"],
        "producer_outcome": result.get("producer_outcome"),
        "failures": [row["rule"] for row in result.get("failures", [])] or result.get("error"),
        "audit": str(out_dir / "audit.json"),
    }
    print(json.dumps(summary))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
