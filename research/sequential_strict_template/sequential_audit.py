#!/usr/bin/env python3
"""Independent audit of a group-sequential strict run (sequential_runner.py evidence).

Standard library only: it imports nothing from the repo (in particular not
``src.evaluation.sequential_gate`` or ``screen_stats``), so
``python -I sequential_audit.py --root <run-root> --out <dir>`` works from any directory.
Exit 0 = PASS, 1 = FAIL (``<out>/audit.json`` is written create-only either way).

What it recomputes in its own code (amendment "Audit requirements"):

* **Boundaries.**  Lan-DeMets O'Brien-Fleming spending and the z-scale boundaries by its
  own recursive integration (Brownian-motion score scale, composite Simpson on a uniform
  grid, bisection), for efficacy at ``family_alpha / n_mixes`` and NI at ``ni_alpha``; they
  must match the intent's frozen plan within 2e-4.  Also look sizes, fractions, nominal
  levels, alpha spent, the plan sha256 and the interleaving counts.
* **Calibration** (delta_NI, band bounds) from the calibration records, and the
  **skew check** verdict from the saved probe outputs against the amendment thresholds
  (the resampling itself needs numpy and is not re-run; its outputs are hash-bound).
* **Every look** up to the stop from the raw records: per-mix paired t (own incomplete
  beta), crossing at the nominal level, NI lower bound, conditional power and futility,
  bands with the ``band_check`` margin, the qualifying look and the decision sequence, the
  validity rule and the stop/continue action; each must agree with its look receipt.
* **Prefix integrity.**  The records on disk are exactly the stop look's unit prefix
  (no record beyond any worker's count at the stop), every record sits in the segment of
  its unit, and every segment's shard ``started.json`` binds the receipt that allowed it
  (look 0: calibration and the passing skew check) -- create-only order, not wall clock.
* **Reducer cross-check**: a plain mean and t over each prefix against the receipts.
* **Outcome**: the expected outcome against ``producer-outcome.json``, ``decision.json``
  and (post-hoc) ``closeout.json``; a ``dry_run`` intent may only yield ``DRY_RUN_*`` and
  never a ``receipt.json``.
* **Provenance**: unless the intent is ``dry_run``, FAIL if ``started.json`` records an
  in-process executor, an injected skew or audit runner or ``allow_dirty``, or a segment's
  supervision shows the in-process executor.  No resume markers may exist.
* **Pre-registration documents**: protocol, OC simulation report and look-1 band-cost
  report still have their frozen sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AUDIT_SCHEMA = "sequential-strict-audit/v1"
METHOD = "strict-sequential-obf-bonferroni-v1"
TEMPLATE_VERSION = "sequential-strict-template/v1"
ARMS = ("incumbent", "candidate")
BOUNDARY_TOLERANCE = 2e-4
REL_TOLERANCE = 1e-6
SKEW_EFFICACY_FACTOR = 1.2
SKEW_NI_LIMIT = 0.06
TERMINAL = ("STOP_PASS", "STOP_FAIL_BANDS", "FINAL_PASS", "FINAL_FAIL")
PASSING = ("STOP_PASS", "FINAL_PASS")
DECIDED = ("STRICT_PASS", "STRICT_FAIL", "DRY_RUN_PASS", "DRY_RUN_FAIL")
PRODUCTION_EXECUTOR = "subprocess"
PRODUCTION_SKEW_RUNNER = "subprocess_skew_runner"
PRODUCTION_AUDIT_RUNNER = "subprocess_audit_runner"
PREREGISTRATION = ("protocol", "oc_report", "band_cost_report")
GRID_POINTS = 401  # odd (Simpson)
GRID_SD = 10.0
_N = NormalDist()


class AuditError(Exception):
    """Evidence that cannot be read at all."""


# ---------------------------------------------------------------- io


def _reject_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant {value}")


def _unique_pairs(items: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise AuditError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)


def canonical_sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: Any, b: Any, rel: float = REL_TOLERANCE, abs_tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)


# ---------------------------------------------------------------- numerics (own code)


def norm_sf(x: float) -> float:
    return _N.cdf(-x)


def obf_spend(t: float, alpha: float) -> float:
    """Lan-DeMets O'Brien-Fleming: ``2 (1 - Phi(z_{1-alpha/2} / sqrt(t)))``."""
    if t <= 0.0:
        return 0.0
    return 2.0 * norm_sf(_N.inv_cdf(1.0 - alpha / 2.0) / math.sqrt(t))


def _simpson(h: float, m: int) -> List[float]:
    w = [2.0 if i % 2 == 0 else 4.0 for i in range(m)]
    w[0] = w[-1] = 1.0
    return [x * h / 3.0 for x in w]


@lru_cache(maxsize=32)
def obf_boundaries(fractions: Tuple[float, ...], alpha: float) -> Tuple[float, ...]:
    """Upper boundaries ``c_k`` (z scale) whose first-crossing probabilities under the null
    equal the OBF spending increments.  Score scale ``S(t) = Z(t) sqrt(t)`` is Brownian
    motion; the sub-density of the not-yet-crossed process is carried on a Simpson grid.
    """
    spent = [obf_spend(t, alpha) for t in fractions]
    increments = [b - a for a, b in zip([0.0] + spent[:-1], spent)]
    bounds: List[float] = []
    grid: List[float] = []
    mass: List[float] = []
    previous = 0.0
    for k, t in enumerate(fractions):
        root_t = math.sqrt(t)
        root_step = math.sqrt(t - previous)

        def crossing(c: float) -> float:
            if k == 0:
                return norm_sf(c)
            upper = c * root_t
            return sum(m * norm_sf((upper - s) / root_step) for s, m in zip(grid, mass))

        low, high = -10.0, 40.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if crossing(mid) > increments[k]:
                low = mid
            else:
                high = mid
        c = 0.5 * (low + high)
        bounds.append(c)
        top = c * root_t
        bottom = min(-GRID_SD * root_t, top - 1e-9)
        h = (top - bottom) / (GRID_POINTS - 1)
        new_grid = [bottom + i * h for i in range(GRID_POINTS)]
        weights = _simpson(h, GRID_POINTS)
        if k == 0:
            density = [
                math.exp(-0.5 * (s / root_t) ** 2) / (root_t * math.sqrt(2 * math.pi))
                for s in new_grid
            ]
        else:
            norm = 1.0 / (root_step * math.sqrt(2 * math.pi))
            density = [
                norm
                * sum(m * math.exp(-0.5 * ((s - g) / root_step) ** 2) for g, m in zip(grid, mass))
                for s in new_grid
            ]
        grid = new_grid
        mass = [w * d for w, d in zip(weights, density)]
        previous = t
    return tuple(bounds)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_isf(p: float, df: int) -> float:
    """Upper ``p`` quantile of Student t by bisection (``0 < p < 0.5``)."""
    low, high = 0.0, 1.0
    while t_sf(high, df) > p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if t_sf(mid, df) > p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def paired(deltas: Sequence[float]) -> Dict[str, Any]:
    n = len(deltas)
    mean = math.fsum(deltas) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0.0:
        p = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
        t = None
    else:
        t = mean / se
        p = t_sf(t, n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "p": p}


def z_from_p(p: float) -> float:
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return -math.inf
    return -_N.inv_cdf(p)


def conditional_power(z: float, f: float, final_c: float, drift: float) -> float:
    if f >= 1.0:
        return 1.0 if z >= final_c else 0.0
    if math.isinf(z):
        return 1.0 if z > 0 else 0.0
    rest = 1.0 - f
    return _N.cdf((z * math.sqrt(f) + drift * rest - final_c) / math.sqrt(rest))


def band(values: Sequence[float], lower: float, upper: float, n_final: int, z: float) -> Dict:
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    margin = z * sd * max(0.0, 1.0 / math.sqrt(n) - 1.0 / math.sqrt(n_final))
    return {
        "n": n,
        "mean": mean,
        "margin": margin,
        "passes": lower + margin <= mean <= upper - margin,
    }


def uint32_seed(domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{domain}|worlds|{index}".encode("utf-8")).digest()
    return int(struct.unpack(">I", digest[:4])[0])


# ---------------------------------------------------------------- audit state


class Audit:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, rule: str, passes: bool, detail: Any = None) -> bool:
        self.checks.append({"rule": rule, "passes": bool(passes), "detail": detail})
        return bool(passes)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if not c["passes"]]


# ---------------------------------------------------------------- plan and design


def audit_plan(audit: Audit, intent: Mapping[str, Any]) -> None:
    plan, params = intent["plan"], intent["plan_parameters"]
    mixes = params["mixes"]
    audit.add("plan.method", intent["method"] == METHOD and plan["method"] == METHOD)
    same = all(
        plan[key] == params[key]
        for key in (
            "n_max",
            "mde",
            "mixes",
            "scripted_mix",
            "family_alpha",
            "ni_alpha",
            "futility_cp",
            "required_successes",
            "futility_policy",
            "band_policy",
            "band_margin_z",
        )
    )
    same = same and close(plan["efficacy_alpha_per_mix"], params["family_alpha"] / len(mixes))
    same = same and intent["spec"]["mixes"] == mixes
    same = same and plan["spending"] == "lan_demets_obrien_fleming"
    same = same and plan["multiplicity"] == "bonferroni_across_mixes"
    audit.add("plan.parameters", same)
    n_max = params["n_max"]
    sizes = [int(math.ceil(f * n_max - 1e-9)) for f in params["fractions"]]
    fractions = tuple(n / n_max for n in sizes)
    audit.add(
        "plan.look_sizes",
        plan["look_sizes"] == sizes
        and len(plan["fractions"]) == len(sizes)
        and all(close(a, b, 1e-12) for a, b in zip(plan["fractions"], fractions)),
        {"expected": sizes, "intent": plan["look_sizes"]},
    )
    for name, alpha, key in (
        ("efficacy", params["family_alpha"] / len(mixes), "efficacy"),
        ("ni", params["ni_alpha"], "ni"),
    ):
        mine = obf_boundaries(fractions, alpha)
        theirs = plan[f"{key}_boundaries"]
        audit.add(
            f"plan.{name}_boundaries",
            len(mine) == len(theirs)
            and all(abs(a - b) <= BOUNDARY_TOLERANCE for a, b in zip(mine, theirs)),
            {"recomputed": list(mine), "intent": theirs},
        )
        audit.add(
            f"plan.{name}_nominal_levels",
            all(close(p, norm_sf(c), 1e-9) for p, c in zip(plan[f"{key}_nominal_p"], theirs)),
        )
        spend = [obf_spend(t, alpha) for t in fractions]
        audit.add(
            f"plan.{name}_alpha_spent",
            all(abs(a - b) <= 1e-6 for a, b in zip(plan[f"{key}_alpha_spent"], spend)),
        )
    audit.add("plan.sha256", canonical_sha(plan) == intent["plan_sha256"])
    workers = intent["caps"]["workers"]
    counts = []
    for look, size in enumerate(sizes):
        total = size * len(mixes)
        counts.append(
            {
                "look": look,
                "worlds_per_mix": size,
                "units": total,
                "per_worker": [(total - w + workers - 1) // workers for w in range(workers)],
            }
        )
    audit.add(
        "interleaving.worker_look_counts", intent["interleaving"]["worker_look_counts"] == counts
    )
    audit.add(
        "plan.futility_action",
        intent["futility_action"] == "stop"
        or (intent["futility_action"] == "continue" and plan["futility_policy"] == "overridable"),
    )


def audit_banks(audit: Audit, intent: Mapping[str, Any]) -> None:
    spaces = intent["spec"]["namespaces"]
    banks = intent["banks"]
    recipe = banks["final"] == [
        uint32_seed(spaces["final"], i) for i in range(intent["plan"]["n_max"])
    ] and banks["calibration"] == [
        uint32_seed(spaces["calibration"], i) for i in range(len(banks["calibration"]))
    ]
    audit.add("banks.recipe", recipe)
    allseeds = banks["final"] + banks["calibration"]
    audit.add("banks.unique_and_disjoint", len(set(allseeds)) == len(allseeds))


# ---------------------------------------------------------------- units


def units(intent: Mapping[str, Any], phase: str) -> List[Dict[str, Any]]:
    mixes = intent["spec"]["mixes"]
    workers = intent["caps"]["workers"]
    count = intent["plan"]["n_max"] if phase == "final" else len(intent["banks"]["calibration"])
    bank = intent["banks"][phase]
    out = []
    for world in range(count):
        for mix in mixes:
            u = len(out)
            out.append(
                {
                    "unit": u,
                    "world_index": world,
                    "mix": mix,
                    "worker": u % workers,
                    "world_seed": bank[world],
                }
            )
    return out


def eid(phase: str, arm: str, mix: str, world: int) -> str:
    return f"{phase}-{arm}-{mix}-w{world:05d}"


def unit_ids(phase: str, unit: Mapping[str, Any]) -> List[Tuple[str, str]]:
    arms = ARMS if phase == "final" else ("incumbent",)
    return [(eid(phase, arm, unit["mix"], unit["world_index"]), arm) for arm in arms]


def load_records(audit: Audit, intent: Mapping[str, Any], output: Path, phase: str) -> Dict:
    """Every record file of ``phase`` keyed by id, with its envelope checked."""
    folder = output / phase / "records"
    plan_units = {
        i: (unit, arm) for unit in units(intent, phase) for i, arm in unit_ids(phase, unit)
    }
    arm_sha = {arm: canonical_sha(intent["arms"][arm]) for arm in ARMS}
    metrics = [intent["spec"]["primary_metric"]] + [b["metric"] for b in intent["spec"]["bands"]]
    entries, problems = {}, []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        entry = load_json(path)
        key = path.stem
        if key not in plan_units:
            problems.append(f"{key}: not a planned episode")
            entries[key] = entry
            continue
        unit, arm = plan_units[key]
        expect = {
            "episode_id": key,
            "phase": phase,
            "arm": arm,
            "mix": unit["mix"],
            "world_index": unit["world_index"],
            "world_seed": unit["world_seed"],
            "unit": unit["unit"],
            "worker": unit["worker"],
            "study_id": intent["study_id"],
            "arm_identity_sha256": arm_sha[arm],
        }
        for field, value in expect.items():
            if entry.get(field) != value:
                problems.append(f"{key}: {field}")
        record = entry.get("record") or {}
        for metric in metrics:
            value = record.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{key}: metric {metric}")
            elif not math.isfinite(value):
                problems.append(f"{key}: metric {metric} non-finite")
        entries[key] = entry
    audit.add(f"{phase}.record_envelopes", not problems, problems[:20])
    return entries


# ---------------------------------------------------------------- calibration and skew


def audit_calibration(
    audit: Audit, intent: Mapping[str, Any], output: Path, entries: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    spec = intent["spec"]
    expected = {
        i for unit in units(intent, "calibration") for i, _ in unit_ids("calibration", unit)
    }
    audit.add("calibration.records_complete", set(entries) == expected)
    path = output / "calibration.json"
    if not path.is_file():
        audit.add("calibration.present", False)
        return None
    saved = load_json(path)
    metrics = sorted({spec["primary_metric"], *(b["metric"] for b in spec["bands"])})
    means: Dict[str, Dict[str, float]] = {}
    for mix in spec["mixes"]:
        rows = [
            e["record"]
            for e in entries.values()
            if e.get("mix") == mix and e.get("arm") == "incumbent"
        ]
        means[mix] = {m: math.fsum(r[m] for r in rows) / len(rows) for m in metrics} if rows else {}
    ok = True
    try:
        delta_ni = spec["ni_fraction"] * means[spec["scripted_mix"]][spec["primary_metric"]]
        ok = close(saved["absolute_delta_ni"], delta_ni, 1e-9) and delta_ni > 0
        for mine, theirs in zip(spec["bands"], saved["bands"]):
            ref = means[mine["mix"]][mine["metric"]]
            ok = ok and theirs["metric"] == mine["metric"] and theirs["mix"] == mine["mix"]
            ok = ok and close(theirs["lower"], ref + mine["lower_offset"], 1e-9)
            ok = ok and close(theirs["upper"], ref + mine["upper_offset"], 1e-9)
        ok = ok and len(saved["bands"]) == len(spec["bands"])
    except (KeyError, TypeError, ZeroDivisionError):
        ok = False
    audit.add("calibration.recomputed", ok)
    return saved


def audit_skew(
    audit: Audit, intent: Mapping[str, Any], output: Path, calibration: Optional[Mapping]
) -> Optional[bool]:
    path = output / "skew_check.json"
    if not path.is_file():
        audit.add("skew.present", False)
        return None
    receipt = load_json(path)
    skew = intent["skew_check"]
    plan = intent["plan"]
    limits_ok = close(
        skew["limits"]["efficacy_any_look"], SKEW_EFFICACY_FACTOR * plan["efficacy_alpha_per_mix"]
    ) and close(skew["limits"]["ni_any_look_scripted"], SKEW_NI_LIMIT)
    audit.add("skew.limits", limits_ok and receipt["limits"] == skew["limits"])
    input_ok = Path(skew["input_path"]).is_file() and (
        sha256_file(Path(skew["input_path"])) == skew["input_sha256"]
    )
    audit.add("skew.input_bound", input_ok)
    saved = load_json(Path(skew["input_path"])) if input_ok else {}
    verdicts, problems = {}, []
    for mix in intent["spec"]["mixes"]:
        row = receipt["files"].get(mix, {})
        try:
            if sha256_file(Path(row["output_path"])) != row["output_sha256"]:
                problems.append(f"{mix}: probe output bytes")
            if sha256_file(Path(row["deltas_path"])) != row["deltas_sha256"]:
                problems.append(f"{mix}: probe input bytes")
            if load_json(Path(row["deltas_path"])) != saved.get(mix):
                problems.append(f"{mix}: probe input is not the frozen saved deltas")
            result = load_json(Path(row["output_path"]))["result"]
        except (KeyError, AuditError, OSError) as exc:
            problems.append(f"{mix}: {exc}")
            verdicts[mix] = False
            continue
        efficacy = result["efficacy_null"]["any_look"]
        ni = result["ni_null"]["any_look"]
        good = (
            efficacy <= skew["limits"]["efficacy_any_look"]
            and (
                mix != intent["spec"]["scripted_mix"]
                or ni <= skew["limits"]["ni_any_look_scripted"]
            )
            and result["look_sizes"] == plan["look_sizes"]
            and result["n_saved"] == skew["input_counts"][mix]
            and result["efficacy_null"]["reps"] == skew["reps"]
        )
        verdicts[mix] = good
        if receipt["per_mix"][mix]["passes"] != good:
            problems.append(f"{mix}: verdict differs")
    audit.add("skew.outputs_bound", not problems, problems)
    passes = all(verdicts.values())
    audit.add("skew.verdict", receipt["passes"] == passes, {"recomputed": passes})
    if calibration is not None:
        audit.add(
            "skew.delta_ni_frozen",
            close(receipt["delta_ni"], calibration["absolute_delta_ni"], 1e-12)
            and receipt["calibration_sha256"] == sha256_file(output / "calibration.json"),
        )
    audit.add(
        "skew.remedy",
        receipt.get("remedy_on_fail") == skew["remedy_on_fail"] == "stop_and_escalate",
    )
    return passes


# ---------------------------------------------------------------- look replay


def replay_looks(
    intent: Mapping[str, Any],
    calibration: Mapping[str, Any],
    entries: Mapping[str, Any],
    last: int,
) -> List[Dict[str, Any]]:
    """Decision history 0..last from the raw records, in this module's own code."""
    plan = intent["plan"]
    spec = intent["spec"]
    mixes = spec["mixes"]
    metric = spec["primary_metric"]
    sizes = plan["look_sizes"]
    n_looks = len(sizes)
    delta_ni = calibration["absolute_delta_ni"]
    deltas = {
        mix: [
            entries[eid("final", "candidate", mix, w)]["record"][metric]
            - entries[eid("final", "incumbent", mix, w)]["record"][metric]
            for w in range(sizes[last])
        ]
        for mix in mixes
    }
    crossing: Dict[str, Optional[int]] = {m: None for m in mixes}
    ni_look: Optional[int] = None
    history = []
    for k in range(last + 1):
        n = sizes[k]
        per_mix = {}
        for mix in mixes:
            stats = paired(deltas[mix][:n])
            crossed_now = stats["p"] <= plan["efficacy_nominal_p"][k]
            if crossing[mix] is None and crossed_now:
                crossing[mix] = k
            cp, futile = None, False
            if k < n_looks - 1 and crossing[mix] is None and stats["sd"]:
                drift = plan["mde"] * math.sqrt(plan["n_max"]) / stats["sd"]
                cp = conditional_power(
                    z_from_p(stats["p"]),
                    plan["fractions"][k],
                    plan["efficacy_boundaries"][-1],
                    drift,
                )
                futile = cp < plan["futility_cp"]
            per_mix[mix] = {
                **stats,
                "crossed": crossing[mix] is not None,
                "cp": cp,
                "futile": futile,
            }
        ni_stats = paired(deltas[plan["scripted_mix"]][:n])
        lower = ni_stats["mean"] - t_isf(plan["ni_nominal_p"][k], n - 1) * ni_stats["se"]
        if ni_look is None and lower > -delta_ni:
            ni_look = k
        bands = []
        for spec_band in calibration["bands"]:
            values = [
                entries[eid("final", "candidate", spec_band["mix"], w)]["record"][
                    spec_band["metric"]
                ]
                for w in range(n)
            ]
            bands.append(
                band(
                    values,
                    spec_band["lower"],
                    spec_band["upper"],
                    plan["n_max"],
                    plan["band_margin_z"],
                )
            )
        bands_pass = all(b["passes"] for b in bands)
        successes = sum(1 for m in mixes if crossing[m] is not None)
        futile = sum(1 for m in mixes if per_mix[m]["futile"])
        qualifies = successes >= plan["required_successes"] and ni_look is not None
        if k == n_looks - 1:
            decision = "FINAL_PASS" if qualifies and bands_pass else "FINAL_FAIL"
        elif qualifies:
            decision = "STOP_PASS" if bands_pass else "STOP_FAIL_BANDS"
        elif len(mixes) - futile < plan["required_successes"]:
            decision = "STOP_FUTILE"
        else:
            decision = "CONTINUE"
        earlier = [h["decision"] for h in history]
        final_stops = [d for d in earlier if d in ("STOP_PASS", "STOP_FAIL_BANDS")]
        overrides = [d for d in earlier if d == "STOP_FUTILE"]
        valid = not final_stops and not (overrides and plan["futility_policy"] != "overridable")
        if decision in TERMINAL:
            action = "stop"
        elif decision == "STOP_FUTILE":
            action = "continue" if intent["futility_action"] == "continue" else "stop"
        else:
            action = "continue"
        history.append(
            {
                "look": k,
                "decision": decision,
                "valid": valid,
                "passes": bool(valid and decision in PASSING),
                "action": action,
                "bands_pass": bands_pass,
                "ni_lower": lower,
                "per_mix": per_mix,
                "deltas_digest": canonical_sha({m: deltas[m][:n] for m in mixes}),
            }
        )
    return history


def compare_receipt(
    audit_rows: List[str], k: int, mine: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    for key in ("decision", "valid", "passes", "action", "deltas_digest"):
        if mine[key] != receipt.get(key):
            audit_rows.append(f"look {k}: {key} {receipt.get(key)!r} != recomputed {mine[key]!r}")
    if receipt.get("bands_pass_by_look", [None])[-1] != mine["bands_pass"]:
        audit_rows.append(f"look {k}: bands verdict")
    decision = receipt.get("sequential_decision", {})
    for mix, row in mine["per_mix"].items():
        theirs = decision.get("per_mix", {}).get(mix, {})
        if not close(theirs.get("p_value"), row["p"], 1e-6, 1e-15):
            audit_rows.append(f"look {k} {mix}: p-value")
        if not close(theirs.get("mean_delta"), row["mean"], 1e-9):
            audit_rows.append(f"look {k} {mix}: mean (reducer cross-check)")
        if row["t"] is not None and not close(theirs.get("t_statistic"), row["t"], 1e-9):
            audit_rows.append(f"look {k} {mix}: t (reducer cross-check)")
        if not close(theirs.get("conditional_power"), row["cp"], 1e-6, 1e-12):
            audit_rows.append(f"look {k} {mix}: conditional power")
        if theirs.get("futile") != row["futile"] or theirs.get("crossed_so_far") != row["crossed"]:
            audit_rows.append(f"look {k} {mix}: crossing/futility flags")
    ni = decision.get("scripted_noninferiority", {})
    if not close(ni.get("lower_bound"), mine["ni_lower"], 1e-6, 1e-9):
        audit_rows.append(f"look {k}: NI lower bound")


def segment_bounds(intent: Mapping[str, Any]) -> List[List[int]]:
    """Cumulative per-worker unit counts at each look."""
    return [row["per_worker"] for row in intent["interleaving"]["worker_look_counts"]]


def audit_final(
    audit: Audit,
    intent: Mapping[str, Any],
    output: Path,
    calibration: Optional[Mapping[str, Any]],
    skew_passed: Optional[bool],
) -> Dict[str, Any]:
    entries = load_records(audit, intent, output, "final")
    looks_dir = output / "looks"
    receipts: List[Dict[str, Any]] = []
    k = 0
    while (looks_dir / f"look-{k}.json").is_file():
        receipts.append(load_json(looks_dir / f"look-{k}.json"))
        k += 1
    on_disk = sorted(p.name for p in looks_dir.glob("*.json")) if looks_dir.is_dir() else []
    audit.add(
        "looks.contiguous", on_disk == [f"look-{j}.json" for j in range(len(receipts))], on_disk
    )
    result: Dict[str, Any] = {"stop_look": None, "decision": None, "receipt": None}
    if skew_passed is not True:
        segments = output / "final" / "segments"
        audit.add(
            "final.nothing_without_skew_pass",
            not entries and not receipts and not (segments.exists() and any(segments.iterdir())),
        )
        return result
    all_units = units(intent, "final")
    workers = intent["caps"]["workers"]
    bounds = segment_bounds(intent)
    stop = len(receipts) - 1
    # prefix integrity: what exists must be exactly a look prefix (the last receipt's, or a
    # partial next segment when the run never reached a receipt for it)
    if stop >= 0 and receipts[-1]["action"] == "stop":
        limit_look = stop
    else:  # no stopping receipt (incomplete run): at most the next segment may exist
        limit_look = min(stop + 1, len(bounds) - 1)
    beyond, by_worker = [], [0] * workers
    shard_pos: Dict[int, int] = {}
    seen = [0] * workers
    for unit in all_units:
        shard_pos[unit["unit"]] = seen[unit["worker"]]
        seen[unit["worker"]] += 1
    unit_of = {i: unit for unit in all_units for i, _ in unit_ids("final", unit)}
    wrong_segment = []
    for key, entry in entries.items():
        unit = unit_of.get(key)
        if unit is None:
            beyond.append(key)
            continue
        pos = shard_pos[unit["unit"]]
        if pos >= bounds[limit_look][unit["worker"]]:
            beyond.append(key)
        segment = next(j for j, b in enumerate(bounds) if pos < b[unit["worker"]])
        if entry.get("look") != segment:
            wrong_segment.append(key)
        by_worker[unit["worker"]] = max(by_worker[unit["worker"]], pos + 1)
    audit.add("final.no_record_beyond_stop", not beyond, sorted(beyond)[:20])
    audit.add("final.records_in_their_segment", not wrong_segment, sorted(wrong_segment)[:20])
    if stop >= 0 and receipts[-1]["action"] == "stop":
        expected = {
            i
            for unit in all_units[: intent["interleaving"]["worker_look_counts"][stop]["units"]]
            for i, _ in unit_ids("final", unit)
        }
        audit.add(
            "final.prefix_exact_at_stop",
            set(entries) == expected and by_worker == bounds[stop],
            {"per_worker": by_worker, "expected": bounds[stop]},
        )
    audit_segments(audit, intent, output, receipts)
    if stop < 0 or calibration is None:
        audit.add("looks.present", stop >= 0)
        return result
    try:
        history = replay_looks(intent, calibration, entries, stop)
    except (KeyError, ZeroDivisionError, TypeError) as exc:
        audit.add("looks.replay", False, f"cannot replay: {exc}")
        return result
    rows: List[str] = []
    intent_sha = sha256_file(output.parent / "intent.json")
    for j, receipt in enumerate(receipts):
        compare_receipt(rows, j, history[j], receipt)
        if receipt.get("look") != j or receipt.get("n_per_mix") != intent["plan"]["look_sizes"][j]:
            rows.append(f"look {j}: index or size")
        if (
            receipt.get("intent_sha256") != intent_sha
            or receipt.get("plan_sha256") != intent["plan_sha256"]
        ):
            rows.append(f"look {j}: intent/plan binding")
        prior = sha256_file(looks_dir / f"look-{j - 1}.json") if j else None
        if receipt.get("prior_look_receipt_sha256") != prior:
            rows.append(f"look {j}: receipt chain")
        prefix = {
            i
            for unit in all_units[: intent["interleaving"]["worker_look_counts"][j]["units"]]
            for i, _ in unit_ids("final", unit)
        }
        listed = receipt.get("records_sha256", {})
        if set(listed) != prefix:
            rows.append(f"look {j}: receipt records differ from the prefix")
        for key, sha in listed.items():
            path = output / "final" / "records" / f"{key}.json"
            if not path.is_file() or sha256_file(path) != sha:
                rows.append(f"look {j}: record bytes {key}")
        if j < stop and history[j]["action"] != "continue":
            rows.append(f"look {j}: run continued past a stop")
    audit.add("looks.replay", not rows, rows[:40])
    last = receipts[-1]
    result.update(
        {"stop_look": stop, "decision": last["decision"], "receipt": last, "history": history}
    )
    return result


def audit_segments(
    audit: Audit, intent: Mapping[str, Any], output: Path, receipts: Sequence[Mapping]
) -> None:
    """Each segment's shard ``started.json`` binds the receipt that allowed it to start."""
    rows = []
    root = output / "final" / "segments"
    calibration = output / "calibration.json"
    skew = output / "skew_check.json"
    for look_dir in sorted(root.glob("look-*")) if root.is_dir() else []:
        look = int(look_dir.name.split("-", 1)[1])
        if look > len(receipts):
            rows.append(f"segment look {look} with no receipt for look {look - 1}")
            continue
        expected = {
            "calibration_sha256": sha256_file(calibration) if calibration.is_file() else None
        }
        if look == 0:
            expected["skew_check_sha256"] = sha256_file(skew) if skew.is_file() else None
        else:
            prior = output / "looks" / f"look-{look - 1}.json"
            expected["prior_look_receipt_sha256"] = sha256_file(prior)
            if receipts[look - 1].get("action") != "continue":
                rows.append(f"segment look {look} started after a stop at look {look - 1}")
        for started in sorted(look_dir.glob("shard-*/started.json")):
            if load_json(started).get("gate") != expected:
                rows.append(f"{started.relative_to(output)}: gate binding")
    for j, receipt in enumerate(receipts):
        for rel, sha in receipt.get("segment_reports_sha256", {}).items():
            path = output / rel
            if not path.is_file() or sha256_file(path) != sha:
                rows.append(f"look {j}: segment report {rel}")
    audit.add("final.segment_order", not rows, rows[:20])


# ---------------------------------------------------------------- outcome


def audit_outcome(
    audit: Audit, intent: Mapping[str, Any], output: Path, skew: Optional[bool], final: Mapping
) -> Dict[str, Any]:
    receipt = final.get("receipt")
    prefix = "DRY_RUN_" if intent.get("dry_run") is True else ""
    if skew is False:
        expected = f"{prefix}SKEW_CHECK_FAILED"
    elif receipt is not None and receipt["action"] == "stop" and receipt["valid"]:
        verdict = "PASS" if receipt["passes"] else "FAIL"
        expected = f"DRY_RUN_{verdict}" if prefix else f"STRICT_{verdict}"
    else:
        expected = None
    decided = expected in DECIDED
    claim_path = output / "producer-outcome.json"
    if claim_path.is_file():
        claim = load_json(claim_path)
        audit.add(
            "outcome.producer_claim",
            claim.get("outcome") == expected
            and claim.get("stop_look") == (final["stop_look"] if decided else None)
            and claim.get("decision") == (final["decision"] if decided else None),
            {"claim": claim, "expected": expected},
        )
    decision_path = output / "decision.json"
    if decision_path.is_file() and receipt is not None:
        saved = load_json(decision_path)
        audit.add(
            "outcome.decision_json",
            saved.get("stop_look") == final["stop_look"]
            and saved.get("decision") == receipt["decision"]
            and saved.get("passes") == receipt["passes"],
        )
    receipt_file = output / "receipt.json"
    if intent.get("dry_run") is True:
        audit.add("outcome.dry_run_has_no_receipt", not receipt_file.exists())
    closeout = output / "closeout.json"
    if closeout.is_file():
        saved = load_json(closeout)
        outcome = saved.get("outcome")
        consistent = (
            outcome in ("INVALID_STOP", "INCOMPLETE", "STOP_INFEASIBLE") or outcome == expected
        )
        if outcome not in ("INVALID_STOP", "INCOMPLETE", "STOP_INFEASIBLE"):
            consistent = consistent and saved.get("audit_passed") is True
        if intent.get("dry_run") is True:
            consistent = consistent and not str(outcome).startswith("STRICT_")
        audit.add("outcome.closeout", consistent, {"closeout": outcome, "expected": expected})
        audit.add(
            "outcome.pass_receipt_only_on_pass",
            receipt_file.is_file() == (outcome == "STRICT_PASS"),
        )
    return {
        "expected_outcome": expected,
        "stop_look": final["stop_look"] if decided else None,
        "decision": final["decision"] if decided else None,
    }


def provenance_problems(
    intent: Mapping[str, Any],
    started: Mapping[str, Any],
    supervisions: Sequence[Mapping[str, Any]],
    skew_receipt: Optional[Mapping[str, Any]],
    audit_runner: Optional[Mapping[str, Any]],
) -> List[str]:
    """A non-dry-run intent must have run with the production executor and runners, a clean
    source closure and the global slot lock root recorded in its intent."""
    if intent.get("dry_run") is True:
        return []
    problems = []
    prov = started.get("provenance") or {}
    expected = {
        "executor": PRODUCTION_EXECUTOR,
        "skew_runner": PRODUCTION_SKEW_RUNNER,
        "audit_runner": PRODUCTION_AUDIT_RUNNER,
        "allow_dirty": False,
        "dry_run": False,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            problems.append(f"started.json provenance {key}={prov.get(key)!r}")
    if intent.get("allow_dirty") is not False or intent["source_closure"].get("dirty"):
        problems.append("dirty source closure on a production intent")
    for row in supervisions:
        if row.get("executor") is not None or any(
            "worker" not in child.get("command", []) for child in row.get("children", [])
        ):
            problems.append("a segment did not run through the subprocess executor")
    if skew_receipt is not None and skew_receipt.get("skew_runner") != PRODUCTION_SKEW_RUNNER:
        problems.append(f"skew runner {skew_receipt.get('skew_runner')!r}")
    if audit_runner is not None and audit_runner.get("identity") != PRODUCTION_AUDIT_RUNNER:
        problems.append(f"audit runner {audit_runner.get('identity')!r}")
    return problems


def audit_intent_binding(audit: Audit, root: Path, intent: Mapping[str, Any]) -> None:
    output = root / "output"
    sha = sha256_file(root / "intent.json")
    started = output / "started.json"
    audit.add(
        "intent.sha256_bound",
        started.is_file() and load_json(started).get("intent_sha256") == sha,
    )
    audit.add(
        "intent.template",
        intent.get("template_version") == TEMPLATE_VERSION
        and intent["caps"]["retry_authorized"] is False
        and Path(intent["output_root"]).resolve() == root.resolve(),
    )
    audit.add(
        "intent.no_resume",
        intent["caps"].get("resume_authorized") is False and not (output / "resumes").exists(),
    )
    rows = []
    for name, row in sorted(intent["preregistration"].items()):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            rows.append(name)
    audit.add(
        "intent.preregistration_documents",
        not rows and set(intent["preregistration"]) == set(PREREGISTRATION),
        rows,
    )
    supervisions = [load_json(p) for p in sorted(output.glob("*/segments/look-*/supervision.json"))]
    skew = output / "skew_check.json"
    runner = output / "audit" / "runner.json"
    problems = provenance_problems(
        intent,
        load_json(started) if started.is_file() else {},
        supervisions,
        load_json(skew) if skew.is_file() else None,
        load_json(runner) if runner.is_file() else None,
    )
    audit.add("provenance.production_or_dry_run", not problems, problems)


def run_audit(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    output = root / "output"
    audit = Audit()
    recomputed: Dict[str, Any] = {}
    error = None
    try:
        intent = load_json(root / "intent.json")
        audit_intent_binding(audit, root, intent)
        audit_plan(audit, intent)
        audit_banks(audit, intent)
        cal_entries = load_records(audit, intent, output, "calibration")
        calibration = audit_calibration(audit, intent, output, cal_entries)
        skew = audit_skew(audit, intent, output, calibration)
        final = audit_final(audit, intent, output, calibration, skew)
        recomputed = audit_outcome(audit, intent, output, skew, final)
    except (AuditError, KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        audit.add("audit.evidence_readable", False, error)
    return {
        "schema_version": AUDIT_SCHEMA,
        "root": str(root),
        "status": "PASS" if not audit.failures and error is None else "FAIL",
        "checks": audit.checks,
        "failures": audit.failures,
        "error": error,
        "recomputed": recomputed,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_create_only(path: Path, value: Any) -> None:
    data = json.dumps(_json_safe(value), sort_keys=True, indent=2, default=str) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    write_create_only(args.out / "audit.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
