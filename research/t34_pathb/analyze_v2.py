"""Fail-closed, fixed-N Amendment 2 paced-retention analysis.

Terminal paired worlds are independent within disjoint mix strata. Product
sessions are the clusters. Artifact loading binds every row to its registration;
no missing evidence, deadline fallback, or failed bar is silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.scripts.eval_stats import student_t_isf, student_t_sf

MIXES = ("scripted", "frozen", "mixed")
# The registered identity list retains the unpaced controller, but the Mac
# efficacy execution has only these arms.  Unpaced observations are loaded from
# independent Colab exports below and are never inputs to ``decide``.
ACTIVE_ARMS = ("paced",)
MAC_ARMS = ("bare", *ACTIVE_ARMS)
ARMS = ("bare", "paced", "unpaced")
HORIZON = 5000
UNPACED_WORLD_SCHEMA = "t34-unpaced-world/v1"
UNPACED_SHARD_SCHEMA = "t34-unpaced-shard/v1"
DESCRIPTIVE_ERRORS = (OSError, KeyError, TypeError, ValueError, ArithmeticError, RuntimeError)


class InvalidRetention(ValueError):
    """Immutable inputs cannot support a scientific decision."""


def _finite(
    value: Any, name: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidRetention(f"{name} must be finite")
    result = float(value)
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise InvalidRetention(f"{name} is out of range")
    return result


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    number = _finite(value, name, minimum, maximum)
    if int(number) != number:
        raise InvalidRetention(f"{name} must be an integer")
    return int(number)


def _interval(mean: float, se: float, df: float, tail: float, n: int) -> dict[str, Any]:
    margin = student_t_isf(tail, df) * se
    return {
        "n": n,
        "df": df,
        "mean": mean,
        "se": se,
        "lower": mean - margin,
        "upper": mean + margin,
        "tail_alpha": tail,
    }


def _ci(values: Iterable[float], tail: float) -> dict[str, Any]:
    xs = [_finite(value, "observation") for value in values]
    if len(xs) < 2:
        raise InvalidRetention("at least two independent observations required")
    return _interval(
        statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs)), len(xs) - 1, tail, len(xs)
    )


def _stratified(groups: Mapping[str, list[float]], tail: float) -> dict[str, Any]:
    """Equal-mix effect and independent-stratum Welch-Satterthwaite uncertainty.

    P7 disjoint seeds remove cross-mix covariance. Pairing arbitrary row indices
    would invent covariance and replace 144 independent worlds with 48 clusters.
    """
    variances = [statistics.variance(groups[mix]) / len(groups[mix]) / 9 for mix in MIXES]
    variance = sum(variances)
    denominator = sum(v * v / (len(groups[mix]) - 1) for mix, v in zip(MIXES, variances))
    df = variance * variance / denominator if denominator else sum(map(len, groups.values())) - 3
    return _interval(
        statistics.fmean(statistics.fmean(groups[mix]) for mix in MIXES),
        math.sqrt(variance),
        df,
        tail,
        sum(map(len, groups.values())),
    )


def _one_sided_p(ci: Mapping[str, Any]) -> float:
    if ci["se"] == 0:
        return 0.0 if ci["mean"] > 0 else 1.0
    return student_t_sf(ci["mean"] / ci["se"], ci["df"])


def _validate_arm(arm: Mapping[str, Any], kind: str) -> None:
    _finite(arm["mass_integral"], "mass", 0)
    survival = _finite(arm["survival_fraction"], "survival", 0, 1)
    no_death = _integer(arm["no_death"], "no_death", 0, 1)
    deaths = _integer(arm["deaths"], "deaths", 0, HORIZON)
    decisions = _integer(arm["decisions"], "decisions", 1, HORIZON)
    boosts = _integer(arm["boost_decisions"], "boost decisions", 0, decisions)
    if no_death != int(deaths == 0) or (no_death and survival != 1):
        raise InvalidRetention("no-death, death and survival counters do not reconcile")
    if kind == "terminal" and deaths > 1:
        raise InvalidRetention("terminal hero cannot respawn")
    if deaths > decisions or not math.isclose(
        _finite(arm["deaths_per_1k"], "death rate", 0), deaths * 1000 / decisions, abs_tol=1e-10
    ):
        raise InvalidRetention("death counters do not reconcile")
    if not math.isclose(
        _finite(arm["boost_share"], "boost share", 0, 1), boosts / decisions, abs_tol=1e-12
    ):
        raise InvalidRetention("boost counters do not reconcile")
    counts = arm["action_counts"]
    if not isinstance(counts, list) or len(counts) != 6:
        raise InvalidRetention("missing six-action counts")
    counts = [_integer(x, "action count", 0, decisions) for x in counts]
    if (
        sum(counts) != decisions
        or sum(counts[3:]) != boosts
        or not math.isclose(
            _finite(arm["one_action_share"], "one-action share", 0, 1),
            max(counts) / decisions,
            abs_tol=1e-12,
        )
    ):
        raise InvalidRetention("action counts do not reconcile")
    faults = arm["unconditional_faults"]
    if not isinstance(faults, list) or any(not isinstance(x, str) or not x for x in faults):
        raise InvalidRetention("missing or malformed unconditional-fault evidence")
    _integer(arm["overruns"], "overruns", 0, HORIZON)
    median = _finite(arm["median_length"], "median length", 0)
    _finite(arm["p95_length"], "p95 length", median)
    if kind == "product":
        large = _integer(arm["deaths_length_ge_800"], "large-body deaths", 0, deaths)
        late = _integer(arm["late_deaths"], "late deaths", 0, deaths)
        overlap = _integer(arm["late_large_overlap"], "late-large overlap", 0, min(large, late))
        if large + late - overlap > deaths:
            raise InvalidRetention("late-large death counters do not reconcile")


def _validate_records(records: list[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    counts = {mix: 0 for mix in MIXES}
    products = 0
    for row in records:
        kind, mix = row["kind"], row["mix"]
        if kind not in ("terminal", "product") or mix not in MIXES:
            raise InvalidRetention("unknown record kind or mix")
        if (
            row["horizon"] != HORIZON
            or row["identity_status"] != "PASS"
            or row["parity_status"] != "PASS"
        ):
            raise InvalidRetention("horizon, parity or identity failure")
        seed = _integer(row["seed"], "seed")
        if seed in seen:
            raise InvalidRetention(
                "duplicate seed; all mix and product namespaces must be disjoint"
            )
        seen.add(seed)
        if set(row["arms"]) != set(MAC_ARMS):
            raise InvalidRetention("Mac record is missing or has an unexpected arm")
        for arm in row["arms"].values():
            _validate_arm(arm, kind)
        if kind == "terminal":
            counts[mix] += 1
        else:
            if mix != "mixed":
                raise InvalidRetention("product sessions must be mixed Watch6")
            products += 1
    if set(counts.values()) != {48} or products != 32:
        raise InvalidRetention("fixed N requires 48 terminal worlds/mix and 32 product sessions")


def skill_summary(records: Iterable[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """Calculate contract skill bars vs bare B8 s3, with P7 paired-world units."""
    if arm not in (*ACTIVE_ARMS, "unpaced"):
        raise InvalidRetention("unknown active arm")
    groups = {
        mix: [row for row in records if row["kind"] == "terminal" and row["mix"] == mix]
        for mix in MIXES
    }
    if any(len(rows) != 48 for rows in groups.values()):
        raise InvalidRetention("requires 48 paired terminal worlds per mix")
    metrics = {}
    for metric in ("mass_integral", "survival_fraction", "no_death"):
        metrics[metric] = {
            mix: [row["arms"][arm][metric] - row["arms"]["bare"][metric] for row in groups[mix]]
            for mix in MIXES
        }
    mass = _stratified(metrics["mass_integral"], 0.10)
    alive = _stratified(metrics["survival_fraction"], 0.05)
    per_mix = {
        mix: {
            "mass": _ci(metrics["mass_integral"][mix], 0.10),
            "alive": _ci(metrics["survival_fraction"][mix], 0.05),
            "no_death": _ci(metrics["no_death"][mix], 0.10),
            "absolute_alive_fraction": statistics.fmean(
                row["arms"][arm]["survival_fraction"] for row in groups[mix]
            ),
        }
        for mix in MIXES
    }
    rows = [row for group in groups.values() for row in group]
    boost = sum(row["arms"][arm]["boost_decisions"] for row in rows) / sum(
        row["arms"][arm]["decisions"] for row in rows
    )
    clauses = {
        "mass_gain_ge_40": mass["mean"] >= 40,
        "mass_lb90_positive": mass["lower"] > 0,
        "survival_guard_a_pooled": alive["lower"] > -0.05,
        "survival_guard_a_each_mix": all(per_mix[mix]["alive"]["lower"] > -0.075 for mix in MIXES),
        "survival_guard_a_absolute_floor": all(
            per_mix[mix]["absolute_alive_fraction"] >= 0.30 for mix in MIXES
        ),
        "survival_guard_b_scripted": per_mix["scripted"]["no_death"]["mean"] >= 0.03,
        "boost_share": 0.005 <= boost <= 0.10,
    }
    return {
        "arm": arm,
        "mass": mass,
        "pooled_alive": alive,
        "per_mix": per_mix,
        "independent_unit": "paired_world",
        "boost_share": boost,
        "mass_p_one_sided": _one_sided_p(mass),
        "clauses": clauses,
        "pass": all(clauses.values()),
    }


def product_summary(records: Iterable[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """Paired session CIs, without treating lives or ticks as replicates."""
    rows = [row for row in records if row["kind"] == "product"]
    if len(rows) != 32:
        raise InvalidRetention("requires 32 paired product sessions")
    deaths, mass = [], []
    for row in rows:
        candidate, bare = row["arms"][arm], row["arms"]["bare"]
        deaths.append(candidate["deaths_per_1k"] - bare["deaths_per_1k"])
        base = _finite(bare["mass_integral"], "bare product mass", 0)
        if base == 0:
            raise InvalidRetention("relative product mass undefined at zero bare mass")
        mass.append((candidate["mass_integral"] - base) / base)
    death_ci, mass_ci = _ci(deaths, 0.025), _ci(mass, 0.025)
    clauses = {
        "death_rate_upper95_negative": death_ci["upper"] < 0,
        "relative_mass_lower95_ge_minus005": mass_ci["lower"] >= -0.05,
    }
    return {
        "independent_unit": "paired_session",
        "death_rate_delta95": death_ci,
        "relative_mass_delta95": mass_ci,
        "clauses": clauses,
        "pass": all(clauses.values()),
        "length_diagnostics": [
            {
                "seed": row["seed"],
                "median_length": row["arms"][arm]["median_length"],
                "p95_length": row["arms"][arm]["p95_length"],
                "deaths_length_ge_800": row["arms"][arm]["deaths_length_ge_800"],
                "late_deaths": row["arms"][arm]["late_deaths"],
                "late_large_overlap": row["arms"][arm]["late_large_overlap"],
            }
            for row in rows
        ],
    }


def timing_summary(timing: Mapping[str, Any]) -> dict[str, Any]:
    """Judge Watch6 probes, retaining the registered cumulative contention allowance."""
    probes = timing["probes"]
    if set(probes) != {f"{arm}-{fps}" for arm in ("bare", "paced") for fps in (12, 24)}:
        raise InvalidRetention("timing requires ON/OFF at both 12 and 24 fps")
    for key, probe in probes.items():
        fps = int(key.split("-")[1])
        ticks = _integer(probe["ticks"], "timing ticks", 1)
        if probe["fps"] != fps or type(probe["contended"]) is not bool:
            raise InvalidRetention("timing profile mismatch")
        _integer(probe["overruns"], "timing overruns", 0, ticks)
        for metric in ("update_ms", "full_tick_ms"):
            distribution = probe[metric]
            if distribution["n"] != ticks:
                raise InvalidRetention("timing sample count mismatch")
            p95 = _finite(distribution["p95"], "p95", 0)
            p99 = _finite(distribution["p99"], "p99", p95)
            _finite(distribution["max"], "max", p99)
        _finite(probe["achieved_fps"], "achieved fps", 0, fps * 1.01)
        if probe["achieved_fps"] == 0:
            raise InvalidRetention("zero delivered cadence")
    clauses = {
        # The 12 fps p95 remains in the immutable probe report as a diagnostic.
        # Contract §4 sets the update-p95 limit only for the 24 fps Watch6 probe.
        "update_p95_le_11688ms_24": probes["paced-24"]["update_ms"]["p95"] <= 11.688,
        "zero_overruns_12": probes["paced-12"]["overruns"] == 0,
        "zero_overruns_24": probes["paced-24"]["overruns"] == 0,
        "cadence_deterioration_le_5pct": all(
            probes[f"paced-{fps}"]["achieved_fps"] >= 0.95 * probes[f"bare-{fps}"]["achieved_fps"]
            for fps in (12, 24)
        ),
    }
    return {
        "probes": probes,
        "clauses": clauses,
        "pass": all(clauses.values()),
        "contended": any(probe["contended"] for probe in probes.values()),
    }


def _uncertain_skill(skill: Mapping[str, Any]) -> bool:
    """Exactly one failed skill category, with its confidence interval overlapping the bar."""
    failed = {key for key, value in skill["clauses"].items() if not value}
    # Raw and Holm mass significance are one bar, not independent misses.
    if "mass_holm_positive" in failed:
        failed.discard("mass_lb90_positive")
    if len(failed) != 1:
        return False
    key = next(iter(failed))
    if key == "mass_gain_ge_40":
        return skill["mass"]["upper"] >= 40
    if key in ("mass_lb90_positive", "mass_holm_positive"):
        return skill["mass_holm_interval"]["upper"] > 0
    if key == "survival_guard_a_pooled":
        return skill["pooled_alive"]["upper"] > -0.05
    if key == "survival_guard_a_each_mix":
        return all(item["alive"]["upper"] > -0.075 for item in skill["per_mix"].values())
    if key == "survival_guard_b_scripted":
        return skill["per_mix"]["scripted"]["no_death"]["upper"] >= 0.03
    return False


def _inconclusive(reason: str) -> dict[str, Any]:
    return {
        "schema": "t34-pathb-v2-decision/v1",
        "verdict": "INCONCLUSIVE",
        "reason": reason,
        "release_authorized": False,
    }


def decide(
    records: Iterable[Mapping[str, Any]], timing: Mapping[str, Any], *, complete: bool = True
) -> dict[str, Any]:
    """Fixed look: integrity, faults, uncertainty exception, scientific KILL/GO.

    Incomplete data do not support an efficacy decision. Recorded unconditional
    faults and registered timing-probe overruns still kill immediately, including
    partial runs. Paced efficacy ticks run bare after a missed deadline; those
    overruns are retained as diagnostics and do not constitute timing failures.
    """
    records = list(records)
    try:
        faults = []
        timing_overrun = False
        paced_efficacy_overruns = 0
        for row in records:
            for name in MAC_ARMS:
                evidence = row.get("arms", {}).get(name, {})
                arm_faults = evidence.get("unconditional_faults", [])
                if not isinstance(arm_faults, list) or any(
                    not isinstance(fault, str) or not fault for fault in arm_faults
                ):
                    raise InvalidRetention("malformed unconditional-fault evidence")
                faults.extend(arm_faults)
            # Paced efficacy misses execute bare fallback and remain diagnostic.
            # Unpaced latency is descriptive as well.
            paced_efficacy_overruns += _integer(
                row.get("arms", {}).get("paced", {}).get("overruns", 0), "overruns"
            )
        for fps in (12, 24):
            timing_overrun |= (
                _integer(
                    timing.get("probes", {}).get(f"paced-{fps}", {}).get("overruns", 0),
                    "timing overruns",
                )
                > 0
            )
    except (InvalidRetention, TypeError, ValueError, AttributeError) as exc:
        return _inconclusive(f"malformed fault evidence: {exc}")
    if faults or timing_overrun:
        return {
            "schema": "t34-pathb-v2-decision/v1",
            "verdict": "KILL",
            "reason": "unconditional_fault_or_timing_overrun",
            "faults": faults,
            "release_authorized": False,
        }
    if not complete:
        return _inconclusive("missing rows, cap reached, or producer did not complete")
    try:
        _validate_records(records)
        paced = skill_summary(records, "paced")
        # Pre-data amendment: preserve the former two-controller family even
        # though unpaced is descriptive and executes elsewhere.  Applying the
        # fixed worst-case Holm threshold avoids relaxing the paced bar after
        # observing (or failing to observe) any Colab export.
        paced["mass_holm_rejected"] = paced["mass_p_one_sided"] <= 0.05
        paced["mass_holm_adjusted_p"] = min(1.0, 2.0 * paced["mass_p_one_sided"])
        paced["mass_holm_interval"] = _interval(
            paced["mass"]["mean"], paced["mass"]["se"], paced["mass"]["df"], 0.05, 144
        )
        paced["clauses"]["mass_holm_positive"] = paced["mass_holm_rejected"]
        paced["pass"] = all(paced["clauses"].values())
        product, timing_report = product_summary(records, "paced"), timing_summary(timing)
        failed_paced_bars = {key for key, passed in paced["clauses"].items() if not passed}
        mass_bars = {"mass_gain_ge_40", "mass_lb90_positive", "mass_holm_positive"}
        near_zero_mass_only = (
            paced["mass"]["mean"] >= 40
            and abs(paced["mass"]["lower"]) <= 10
            and bool(failed_paced_bars & mass_bars)
            and failed_paced_bars <= mass_bars
            and product["pass"]
        )
        timing_contention = timing.get("host_summary", {}).get("contended_fraction")
        if timing_report["contended"] and (
            type(timing_contention) not in (int, float)
            or not math.isfinite(timing_contention)
            or timing_contention > 0.02
        ):
            verdict, reason = "INCONCLUSIVE", "timing contention exceeded registered allowance"
        elif not timing_report["pass"]:
            verdict, reason = "KILL", "paced timing bar failed"
        elif near_zero_mass_only or (product["pass"] and _uncertain_skill(paced)):
            verdict, reason = "INCONCLUSIVE", "registered near-boundary uncertainty exception"
        elif not paced["pass"]:
            verdict, reason = "KILL", "paced skill bars failed"
        elif product["pass"]:
            verdict, reason = "GO", "all registered delivered-controller bars passed"
        else:
            verdict, reason = "INCONCLUSIVE", "product bars not established"
        return {
            "schema": "t34-pathb-v2-decision/v1",
            "verdict": verdict,
            "reason": reason,
            "paced": paced,
            "product": product,
            "timing": timing_report,
            "paced_efficacy_overruns_diagnostic": paced_efficacy_overruns,
            "multiplicity": {
                "method": "Holm fixed worst-case threshold",
                "registered_hypotheses": 2,
                "paced_p_one_sided_threshold": 0.05,
                "looks": 1,
                "family_alpha": 0.10,
                "unpaced_role": "separate descriptive Colab input; never relaxes paced threshold",
            },
            "client_latency": "NOT_MEASURED",
            "serving_qualification": "NOT_MEASURED",
            "release_authorized": False,
        }
    except (
        InvalidRetention,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        return _inconclusive(f"identity_or_completeness: {type(exc).__name__}: {exc}")


def _load(path: Path) -> dict[str, Any]:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise InvalidRetention(f"duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise InvalidRetention(f"non-finite JSON {value}")

    result = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(result, dict):
        raise InvalidRetention("expected JSON object")
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_export_sidecars(path: Path, digest: str) -> None:
    sha_path = path.with_name(f"{path.name}.sha256")
    complete_path = path.with_name(f"{path.name}.complete")
    if sha_path.read_text() != f"{digest}  {path.name}\n":
        raise InvalidRetention("export sha256 sidecar does not bind its payload")
    if complete_path.read_text() != "complete\n":
        raise InvalidRetention("export completion marker is missing or malformed")


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidRetention(f"{name} must be a sha256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise InvalidRetention(f"{name} must be a sha256 hex digest") from exc
    return value.lower()


def _prefix_trace(trace: Any, label: str) -> tuple[str, list[str], bool]:
    if not isinstance(trace, Mapping):
        raise InvalidRetention(f"{label} bare prefix trace is missing")
    initial = _sha256_text(trace.get("initial_sha256"), f"{label} initial prefix")
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise InvalidRetention(f"{label} prefix steps are missing")
    return (
        initial,
        [_sha256_text(step, f"{label} prefix step") for step in steps],
        trace.get("ended_by_intervention") is True,
    )


def _prefix_divergence(mac_trace: Any, colab_trace: Any) -> dict[str, Any]:
    """Compare only the physical bare prefix shared before either intervention."""
    mac_initial, mac_steps, mac_intervened = _prefix_trace(mac_trace, "Mac")
    colab_initial, colab_steps, colab_intervened = _prefix_trace(colab_trace, "Colab")
    if mac_initial != colab_initial:
        return {
            "status": "DIVERGED",
            "first_divergence_tick": 0,
            "shared_steps": 0,
            "mac_steps": len(mac_steps),
            "colab_steps": len(colab_steps),
            "mac_ended_by_intervention": mac_intervened,
            "colab_ended_by_intervention": colab_intervened,
        }
    shared = min(len(mac_steps), len(colab_steps))
    for index in range(shared):
        if mac_steps[index] != colab_steps[index]:
            return {
                "status": "DIVERGED",
                "first_divergence_tick": index + 1,
                "shared_steps": index,
                "mac_steps": len(mac_steps),
                "colab_steps": len(colab_steps),
                "mac_ended_by_intervention": mac_intervened,
                "colab_ended_by_intervention": colab_intervened,
            }
    return {
        "status": "MATCHED_SHARED_PREFIX",
        "first_divergence_tick": None,
        "shared_steps": shared,
        "mac_steps": len(mac_steps),
        "colab_steps": len(colab_steps),
        "mac_ended_by_intervention": mac_intervened,
        "colab_ended_by_intervention": colab_intervened,
    }


def _expected_worlds(registration: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (kind, _integer(row["seed"], "registered seed")): row
        for kind, key in (("terminal", "terminal_rows"), ("product", "product_rows"))
        for row in registration[key]
    }


def _registered_bank_sha(registration: Mapping[str, Any]) -> str:
    value = registration.get("bank_sha256")
    if value is None and isinstance(registration.get("bank"), Mapping):
        value = registration["bank"].get("sha256")
    return _sha256_text(value, "registered bank")


def load_unpaced_descriptive(
    registration: Mapping[str, Any],
    registration_sha256: str,
    mac_records: Iterable[Mapping[str, Any]],
    shard_paths: Iterable[Path],
) -> dict[str, Any]:
    """Load optional Colab exports without granting them any decision authority.

    A bad or absent shard stays visible as a descriptive missing row.  This
    routine deliberately does not raise to the efficacy analyzer: changing a
    Colab transport outcome must not change fault precedence or GO/KILL/
    INCONCLUSIVE from the Mac-registered observations.
    """
    expected = _expected_worlds(registration)
    mac_by_key = {(row["kind"], _integer(row["seed"], "Mac seed")): row for row in mac_records}
    result: dict[str, Any] = {
        "role": "descriptive_only",
        "schema": UNPACED_SHARD_SCHEMA,
        "status": "MISSING",
        "rows": [],
        "missing_rows": [],
        "errors": [],
        "summary": {"expected": len(expected), "valid": 0, "missing_or_invalid": len(expected)},
    }
    seen: set[tuple[str, int]] = set()
    duplicate_keys: set[tuple[str, int]] = set()
    exports: dict[tuple[str, int], Mapping[str, Any]] = {}
    try:
        bank_sha256 = _registered_bank_sha(registration)
        source_commit = registration["source_commit"]
        if not isinstance(source_commit, str) or len(source_commit) != 40:
            raise InvalidRetention("registered source commit is malformed")
    except (InvalidRetention, KeyError) as exc:
        result["errors"].append(f"registration binding unavailable: {exc}")
        result["missing_rows"] = [
            {"kind": kind, "seed": seed, "reason": "registration_binding_unavailable"}
            for kind, seed in sorted(expected)
        ]
        return result

    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        try:
            _require_export_sidecars(shard_path, _sha256_file(shard_path))
            shard = _load(shard_path)
            if (
                shard.get("schema") != UNPACED_SHARD_SCHEMA
                or shard.get("status") != "COMPLETE"
                or shard.get("registration_sha256") != registration_sha256
                or shard.get("source_commit") != source_commit
                or shard.get("bank_sha256") != bank_sha256
            ):
                raise InvalidRetention("shard identity or completion status mismatch")
            entries = shard.get("worlds")
            if not isinstance(entries, list):
                raise InvalidRetention("shard worlds are missing")
            for entry in entries:
                if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                    raise InvalidRetention("malformed shard world entry")
                payload_path = Path(entry["path"])
                if payload_path.is_absolute() or ".." in payload_path.parts:
                    raise InvalidRetention("world export path must stay below its shard index")
                payload_path = (shard_path.parent / payload_path).resolve()
                if not payload_path.is_relative_to(shard_path.parent.resolve()):
                    raise InvalidRetention("world export path escapes its shard index")
                digest = _sha256_text(entry.get("sha256"), "shard world digest")
                if not payload_path.is_file() or _sha256_file(payload_path) != digest:
                    raise InvalidRetention("world export is missing or hash mismatched")
                _require_export_sidecars(payload_path, digest)
                payload = _load(payload_path)
                kind = payload.get("kind")
                seed = payload.get("seed")
                if not isinstance(kind, str):
                    raise InvalidRetention("world export kind is malformed")
                key = (kind, _integer(seed, "unpaced seed"))
                if key in seen:
                    duplicate_keys.add(key)
                    exports.pop(key, None)
                    continue
                seen.add(key)
                if (
                    payload.get("schema") != UNPACED_WORLD_SCHEMA
                    or payload.get("registration_sha256") != registration_sha256
                    or payload.get("source_commit") != source_commit
                    or payload.get("bank_sha256") != bank_sha256
                    or payload.get("arm") != "unpaced"
                    or payload.get("horizon") != HORIZON
                    or key not in expected
                    or payload.get("world") != expected[key]
                    or payload.get("mix") != expected[key].get("mix", "mixed")
                    or not isinstance(payload.get("metrics"), Mapping)
                ):
                    raise InvalidRetention("world export identity mismatch")
                metrics = payload["metrics"]
                _validate_arm(metrics, kind)
                if (
                    metrics.get("identity_status") != "PASS"
                    or metrics.get("parity_status") != "PASS"
                    or metrics.get("unconditional_faults")
                ):
                    raise InvalidRetention("world export fault, parity, or identity failure")
                exports[key] = payload
        except DESCRIPTIVE_ERRORS as exc:
            result["errors"].append(f"{shard_path}: {type(exc).__name__}: {exc}")

    for key in sorted(expected):
        payload = exports.get(key)
        if payload is None:
            result["missing_rows"].append(
                {
                    "kind": key[0],
                    "seed": key[1],
                    "reason": "duplicate_export" if key in duplicate_keys else "not_exported",
                }
            )
            continue
        row: dict[str, Any] = {"kind": key[0], "seed": key[1], "status": "VALID"}
        mac = mac_by_key.get(key)
        if mac is None:
            row.update({"status": "INVALID", "reason": "Mac row is unavailable"})
        else:
            try:
                row["bare_prefix_divergence"] = _prefix_divergence(
                    mac["arms"]["bare"]["bare_prefix_trace"],
                    payload["metrics"]["bare_prefix_trace"],
                )
            except DESCRIPTIVE_ERRORS as exc:
                row.update({"status": "INVALID", "reason": f"prefix trace: {exc}"})
        result["rows"].append(row)
    valid = sum(row["status"] == "VALID" for row in result["rows"])
    missing = len(expected) - valid
    result["summary"] = {"expected": len(expected), "valid": valid, "missing_or_invalid": missing}
    result["status"] = "COMPLETE" if valid == len(expected) else ("PARTIAL" if valid else "MISSING")
    if result["status"] == "COMPLETE":
        descriptive_records = []
        for key in sorted(expected):
            mac = mac_by_key[key]
            descriptive_records.append(
                {
                    **mac,
                    "arms": {"bare": mac["arms"]["bare"], "unpaced": exports[key]["metrics"]},
                }
            )
        try:
            result["statistics"] = {
                "skill": skill_summary(descriptive_records, "unpaced"),
                "product": product_summary(descriptive_records, "unpaced"),
            }
        except DESCRIPTIVE_ERRORS as exc:
            result["errors"].append(f"descriptive statistics invalid: {type(exc).__name__}: {exc}")
            result["statistics"] = {"status": "INVALID", "reason": str(exc)}
    else:
        result["statistics"] = {"status": "NOT_COMPUTED_INCOMPLETE_EXPORTS"}
    return result


def analyze_path(root: Path, unpaced_shards: Iterable[Path] = ()) -> dict[str, Any]:
    """Fence the entire analysis at the detached ratified source revision."""
    from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard

    root = Path(root)
    try:
        registration = _load(root / "registration.json")
        with FrozenWorktreeGuard(
            Path(__file__).resolve().parents[2], registration["source_commit"]
        ):
            report = _analyze_path(root)
            # This attachment is intentionally after the Mac decision and does
            # not inspect its verdict or mutate its fault evidence.
            try:
                registration = _load(root / "registration.json")
                mac_records = [_load(path) for path in sorted((root / "records").glob("*.json"))]
                report["unpaced_descriptive"] = load_unpaced_descriptive(
                    registration,
                    _sha256_file(root / "registration.json"),
                    mac_records,
                    unpaced_shards,
                )
            except DESCRIPTIVE_ERRORS as exc:
                report["unpaced_descriptive"] = {
                    "role": "descriptive_only",
                    "status": "MISSING",
                    "errors": [f"descriptive loader unavailable: {type(exc).__name__}: {exc}"],
                }
            return report
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        return _inconclusive(f"identity_or_completeness: {type(exc).__name__}: {exc}")


def _analyze_path(root: Path) -> dict[str, Any]:
    """Load immutable artifacts and verify registered identity before calculation."""
    root = Path(root)
    try:
        if (root / "source_failure.json").exists():
            return _inconclusive("source fence failed during execution or analysis")
        registration = _load(root / "registration.json")
        from research.t34_pathb.retention_v2 import require_registration

        require_registration(registration, _load(root / "precheck.json"))
        registration_hash = hashlib.sha256((root / "registration.json").read_bytes()).hexdigest()
        if (root / "incomplete.json").exists():
            partial = _load(root / "incomplete.json")
            faults = partial.get("unconditional_faults", [])
            if not isinstance(faults, list) or any(
                not isinstance(fault, str) or not fault for fault in faults
            ):
                raise InvalidRetention("malformed partial-run fault evidence")
            if faults:
                return {
                    "schema": "t34-pathb-v2-decision/v1",
                    "verdict": "KILL",
                    "reason": "unconditional_fault",
                    "faults": faults,
                    "release_authorized": False,
                }
        expected = {
            (kind, row["seed"]): row
            for kind, key in (("terminal", "terminal_rows"), ("product", "product_rows"))
            for row in registration[key]
        }
        records = [_load(path) for path in sorted((root / "records").glob("*.json"))]
        for row in records:
            key = (row["kind"], row["seed"])
            if (
                key not in expected
                or row["registration_sha256"] != registration_hash
                or row["world"] != expected[key]
            ):
                raise InvalidRetention("record differs from registered world or registration hash")
            if row["mix"] != expected[key].get("mix", "mixed"):
                raise InvalidRetention("record mix differs from registered world")
        timing = _load(root / "timing.json") if (root / "timing.json").exists() else {}
        early = decide(records, timing, complete=False)
        if early["verdict"] == "KILL":
            return early
        complete = _load(root / "complete.json")
        if (
            complete["status"] != "complete"
            or complete["registration_sha256"] != registration_hash
            or complete["identity_status"] != "PASS"
            or complete["parity_status"] != "PASS"
        ):
            raise InvalidRetention("completion receipt identity mismatch")
        if (
            _finite(complete["elapsed_seconds"], "elapsed seconds", 0) >= 40 * 3600
            or _finite(complete["cpu_seconds"], "CPU seconds", 0) >= 60 * 3600
        ):
            return _inconclusive("registered elapsed or CPU cap reached")
        if (root / "analysis_accounting.json").exists():
            accounting = _load(root / "analysis_accounting.json")
            if (
                _finite(accounting["elapsed_seconds"], "analysis elapsed seconds", 0) >= 40 * 3600
                or _finite(accounting["cpu_seconds"], "analysis CPU seconds", 0) >= 60 * 3600
            ):
                return _inconclusive("registered cap reached through analysis")
        summary = complete.get("host_summary")
        if not isinstance(summary, Mapping):
            raise InvalidRetention("missing cumulative host contention accounting")
        if _finite(summary["contended_fraction"], "host contention fraction", 0) > 0.02:
            return _inconclusive("cumulative host contention exceeded two percent")
        from research.t34_pathb.source_binding_v2 import verify_frozen_worktree

        verify_frozen_worktree(Path(__file__).resolve().parents[2], registration["source_commit"])
        return decide(records, timing, complete=not (root / "incomplete.json").exists())
    except (
        InvalidRetention,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        return _inconclusive(f"identity_or_completeness: {type(exc).__name__}: {exc}")
