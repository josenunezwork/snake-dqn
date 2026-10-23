"""Immutable R8/R9 decision analyzer for the F4 residual-dose registration.

This module consumes sealed, already-produced receipts.  It never opens labels,
checkpoints, or simulator state, and it has no dispatch capability.  The input
format deliberately contains raw paired gameplay rows rather than precomputed
claims: the analyzer recomputes every decision-bearing statistic at its one
registered look.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.bl_ppo.exports import export_json
from src.evaluation.sequential_gate import pooled_band_check
from src.scripts.eval_stats import holm, student_t_isf, student_t_sf

SCHEMA = "bl-residual-f4-analyzer/v1"
INPUT_SCHEMA = "bl-residual-f4-analysis-input/v1"
H5000_ADMISSION_SCHEMA = "bl-residual-f4-h5000-go/v1"
MIXES = ("scripted", "frozen", "mixed")
N_PER_MIX = 128
REGISTRATION_SHA256 = "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65"
LABELS = (
    "INVALID",
    "INCOMPLETE",
    "INSUFFICIENT",
    "KILL_FIT",
    "STOP_FUNDING",
    "STOP_ADMISSION",
    "KILL_DOSE_CLOSED",
    "KILL_DOSE_FUTILE",
    "GO_TOP_DOSE_DEV",
    "INCONCLUSIVE",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def canonical_receipt(value: Mapping[str, Any]) -> str:
    """Return canonical bytes suitable for a create-only analysis receipt."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two paired worlds required")
    average = _mean(values)
    return math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)


def _one_sided(mean: float, variance: float, df: float) -> tuple[float, float, float]:
    se = math.sqrt(variance)
    if se == 0.0:
        p = 0.0 if mean > 0.0 else (1.0 if mean < 0.0 else 0.5)
        return mean, mean, p
    critical = student_t_isf(0.10, df)
    statistic = mean / se
    return mean - critical * se, mean + critical * se, student_t_sf(statistic, df)


def welch_stratified(deltas: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """R8 equal-mix Welch estimator for exactly 128 paired worlds per mix."""
    if tuple(deltas) != MIXES:
        raise ValueError("mix order must be scripted, frozen, mixed")
    if any(len(deltas[mix]) != N_PER_MIX for mix in MIXES):
        raise ValueError("R8 requires exactly 128 paired worlds per mix")
    parsed = {mix: [_finite(v, f"{mix} delta") for v in deltas[mix]] for mix in MIXES}
    means = {mix: _mean(parsed[mix]) for mix in MIXES}
    variances = {mix: _sample_variance(parsed[mix]) for mix in MIXES}
    contributions = {mix: variances[mix] / (9.0 * N_PER_MIX) for mix in MIXES}
    variance = math.fsum(contributions.values())
    if variance == 0.0:
        # The limiting t statistic is well defined; retain a JSON-safe finite
        # df because receipts are canonical JSON and must never contain Infinity.
        df = float(N_PER_MIX - 1)
        lower, upper, p_value = _one_sided(_mean(list(means.values())), 0.0, 127.0)
    else:
        df = math.fsum(contributions.values()) ** 2 / math.fsum(
            contribution**2 / (N_PER_MIX - 1) for contribution in contributions.values()
        )
        lower, upper, p_value = _one_sided(_mean(list(means.values())), variance, df)
    per_mix = {}
    for mix in MIXES:
        lower_m, upper_m, p_m = _one_sided(means[mix], variances[mix] / N_PER_MIX, N_PER_MIX - 1)
        per_mix[mix] = {
            "mean": means[mix],
            "sample_variance": variances[mix],
            "lower90": lower_m,
            "upper90": upper_m,
            "p_value": p_m,
        }
    return {
        "mean": _mean(list(means.values())),
        "variance": variance,
        "standard_error": math.sqrt(variance),
        "df": df,
        "lower90": lower,
        "upper90": upper,
        "p_value": p_value,
        "per_mix": per_mix,
    }


def _arm(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(name), Mapping):
        raise ValueError(f"row lacks {name} arm")
    return arms[name]


def _metric(arm: Mapping[str, Any], name: str) -> float:
    return _finite(arm.get(name), name)


def _survival(arm: Mapping[str, Any]) -> float:
    # Gameplay's live producer calls this ``alive_frame_fraction``; accepting
    # ``survival_fraction`` supports independently sealed versions of the same receipt.
    return _metric(
        arm, "survival_fraction" if "survival_fraction" in arm else "alive_frame_fraction"
    )


def _boost(row: Mapping[str, Any], arm: str) -> float:
    shares = row.get("boost_share")
    if isinstance(shares, Mapping) and arm in shares:
        return _finite(shares[arm], f"{arm} boost_share")
    return _metric(_arm(row, arm), "boost_share")


def _validate_trace(arm: Mapping[str, Any], name: str) -> tuple[float, int]:
    """Verify the delivered trace and derive, rather than trust, boost share."""
    trace, decisions = arm.get("action_trace"), arm.get("decisions")
    if (
        not isinstance(trace, list)
        or type(decisions) is not int
        or decisions != len(trace)
        or decisions <= 0
    ):
        raise ValueError("missing or mismatched delivered action trace")
    boosts = 0
    for entry in trace:
        if not isinstance(entry, Mapping) or type(entry.get("served_action")) is not int:
            raise ValueError("malformed delivered action trace")
        action = entry["served_action"]
        if not 0 <= action < 6:
            raise ValueError("illegal delivered action trace")
        boosts += action >= 3
    share = boosts / decisions
    if not math.isclose(_metric(arm, "boost_share"), share, abs_tol=1e-15, rel_tol=0.0):
        raise ValueError(f"{name} boost share disagrees with delivered trace")
    return share, decisions


def _current_recipe_digest() -> str:
    """Compute the bound recipe identity at analysis time, never trust a claim."""
    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import canonical_sha

    return canonical_sha(dataclasses.asdict(FitRecipe()))


def source_sha256() -> str:
    """Digest these immutable analyzer bytes for the result lineage."""
    with Path(__file__).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_identity(payload: Mapping[str, Any]) -> list[str]:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        return ["identity_missing"]
    errors: list[str] = []
    for key in ("recipe_sha256", "lineage_sha256", "seed_audit_sha256", "parity_sha256"):
        try:
            _digest(identity.get(key), key)
        except ValueError:
            errors.append(f"bad_{key}")
    if identity.get("registration_sha256") != REGISTRATION_SHA256:
        errors.append("registration_sha256_mismatch")
    if identity.get("analyzer_sha256") != source_sha256():
        errors.append("analyzer_source_sha256_mismatch")
    try:
        if identity.get("recipe_sha256") != _current_recipe_digest():
            errors.append("recipe_sha256_mismatch")
    except (ImportError, AttributeError, TypeError, ValueError):
        errors.append("recipe_digest_unavailable")
    controllers = identity.get("controller_sha256")
    if not isinstance(controllers, Mapping):
        errors.append("controller_identity_missing")
    else:
        # B8 exists on every path.  Candidate digests are required only when
        # their sealed artifact was actually produced; a D256 stop cannot know
        # a future D1024/D4096 checkpoint identity.
        required_arms = {"B8"}
        if isinstance(payload.get("gameplay"), Mapping):
            required_arms.update(("D1024", "D4096"))
        if isinstance(payload.get("probe"), Mapping) or isinstance(payload.get("watch6"), Mapping):
            required_arms.add("D4096")
        for arm in required_arms:
            try:
                _digest(controllers.get(arm), f"controller_{arm}")
            except ValueError:
                errors.append(f"bad_controller_{arm}")
        for arm, value in controllers.items():
            if arm not in {"B8", "D1024", "D4096"}:
                errors.append("foreign_controller_identity")
            else:
                try:
                    _digest(value, f"controller_{arm}")
                except ValueError:
                    errors.append(f"bad_controller_{arm}")
        if (
            controllers.get("B8")
            != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
        ):
            errors.append("B8_controller_identity_mismatch")
    if identity.get("sealed") is not True:
        errors.append("unsealed_output")
    # These are hashes of the exact sealed receipt objects fed below.  They
    # prevent substituting a syntactically valid, but different, gameplay or
    # product receipt between validation and the one look.
    gameplay = payload.get("gameplay")
    if gameplay is not None and (
        not isinstance(gameplay, Mapping) or identity.get("gameplay_sha256") != _sha(gameplay)
    ):
        errors.append("gameplay_lineage_mismatch")
    elif isinstance(gameplay, Mapping):
        if gameplay.get("controller_sha256") != controllers:
            errors.append("gameplay_controller_identity_mismatch")
    watch = payload.get("watch6")
    if watch is not None and (
        not isinstance(watch, Mapping) or identity.get("watch6_sha256") != _sha(watch)
    ):
        errors.append("watch6_lineage_mismatch")
    expected_pair = (
        {arm: controllers[arm] for arm in ("B8", "D4096")}
        if isinstance(controllers, Mapping) and all(arm in controllers for arm in ("B8", "D4096"))
        else None
    )
    probe = payload.get("probe")
    if probe is not None and (
        not isinstance(probe, Mapping) or probe.get("controller_sha256") != expected_pair
    ):
        errors.append("probe_controller_identity_mismatch")
    if (
        watch is not None
        and isinstance(watch, Mapping)
        and watch.get("controller_sha256") != expected_pair
    ):
        errors.append("watch6_controller_identity_mismatch")
    # Any already-materialized unit participates in INVALID precedence, even
    # when an earlier scientific rule would have stopped the remaining path.
    if isinstance(watch, Mapping) and "sessions" in watch:
        sessions = watch["sessions"]
        if not isinstance(sessions, list):
            errors.append("Watch6_sessions_malformed")
        else:
            seen_sessions: set[str] = set()
            for session in sessions:
                session_id = session.get("session_id") if isinstance(session, Mapping) else None
                if not isinstance(session_id, str) or not session_id:
                    errors.append("Watch6_session_identity_malformed")
                elif session_id in seen_sessions:
                    errors.append("duplicate_Watch6_session")
                else:
                    seen_sessions.add(session_id)
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append("evidence_missing")
    else:
        for name, digest_key in (
            ("lineage", "lineage_sha256"),
            ("seed_audit", "seed_audit_sha256"),
            ("parity", "parity_sha256"),
        ):
            receipt = evidence.get(name)
            if not isinstance(receipt, Mapping) or receipt.get("sha256") != identity.get(
                digest_key
            ):
                errors.append(f"{name}_binding_mismatch")
            elif receipt.get("status") not in ("CLEAR", "PASS"):
                errors.append(f"{name}_not_passed")
    return errors


def _not_run(payload: Mapping[str, Any], unit: str) -> bool:
    rows = payload.get("not_run_by_rule", [])
    reasons = payload.get("not_run_by_rule_reason", {})
    return (
        isinstance(rows, list)
        and unit in rows
        and isinstance(reasons, Mapping)
        and reasons.get(unit)
        in {
            "KILL_FIT",
            "STOP_FUNDING",
            "STOP_ADMISSION",
            "INSUFFICIENT",
            "KILL_DOSE_CLOSED",
            "KILL_DOSE_FUTILE",
            "INCONCLUSIVE",
        }
    )


def _path_status(payload: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Return the pre-gameplay R9 result, preserving its registered precedence."""
    stage = payload.get("staging", {})
    if not isinstance(stage, Mapping):
        return "INVALID", ["staging_malformed"]
    gameplay = payload.get("gameplay")
    if isinstance(gameplay, Mapping):
        faults = gameplay.get("known_controller_fault", {})
        if not isinstance(faults, Mapping):
            return "INVALID", ["known_controller_fault_malformed"]
        if any(
            isinstance(value, Mapping)
            and value.get("kind") == "KILL_FIT"
            and isinstance(value.get("fault"), str)
            for value in faults.values()
        ):
            return "KILL_FIT", ["controller_fault"]
        if any(not isinstance(value, Mapping) for value in faults.values()):
            return "INVALID", ["known_controller_fault_malformed"]
    # Validate every materialized unit first, even if a later unit is not
    # required because an earlier registered rule stopped the path.
    for dose in ("D256", "D1024", "D4096"):
        row = stage.get(dose)
        if row is not None and (
            not isinstance(row, Mapping)
            or not isinstance(row.get("selected_support"), bool)
            or not isinstance(row.get("unconditional_fault"), bool)
        ):
            return "INVALID", [f"malformed_stage_{dose}"]

    def dose_row(dose: str) -> Mapping[str, Any] | None:
        row = stage.get(dose)
        return row if isinstance(row, Mapping) else None

    d256 = dose_row("D256")
    if d256 is None:
        return "INCOMPLETE", ["missing_D256"]
    if not d256["selected_support"]:
        return "INSUFFICIENT", ["D256"]
    if d256["unconditional_fault"]:
        return "KILL_FIT", ["D256"]
    d1024 = dose_row("D1024")
    if d1024 is None:
        return "INCOMPLETE", ["missing_D1024"]
    if not d1024["selected_support"]:
        return "INSUFFICIENT", ["D1024"]
    if d1024["unconditional_fault"]:
        return "KILL_FIT", ["D1024"]

    funding = stage.get("funding")
    if funding is None:
        return "INCOMPLETE", ["missing_funding_receipt"]
    if not isinstance(funding, Mapping) or funding.get("verdict") not in (
        "PASS_D4096_FUNDING",
        "PASS",
        "STOP_FUNDING",
        "F2_NON_STRIKE",
    ):
        return "INVALID", ["funding_receipt_malformed"]
    if funding["verdict"] == "STOP_FUNDING":
        return "STOP_FUNDING", ["funding_verdict"]
    projection = stage.get("projection")
    if projection is None:
        return "INCOMPLETE", ["missing_projection_receipt"]
    if not isinstance(projection, Mapping) or not isinstance(
        projection.get("qualified_placement"), bool
    ):
        return "INVALID", ["projection_receipt_malformed"]
    if not projection["qualified_placement"]:
        return "INSUFFICIENT", ["projection"]
    top_stage = dose_row("D4096")
    if top_stage is None:
        return "INCOMPLETE", ["missing_D4096"]
    if not top_stage["selected_support"]:
        return "INSUFFICIENT", ["D4096"]
    faults = ["D4096"] if top_stage["unconditional_fault"] else []
    if isinstance(top_stage, Mapping) and top_stage.get("fit_drift_fault") is True:
        faults.append("D4096_fit_drift")
    if isinstance(top_stage, Mapping) and "reserve_retention" in top_stage:
        try:
            if _finite(top_stage["reserve_retention"], "reserve_retention") < 0.70:
                faults.append("D4096_reserve_retention")
        except ValueError:
            return "INVALID", ["malformed_D4096_reserve_retention"]
    if faults:
        return "KILL_FIT", faults
    probe = payload.get("probe")
    if probe is None:
        return "INCOMPLETE", ["missing_probe_receipt"]
    if not isinstance(probe, Mapping) or probe.get("schema") != "bl-residual-gameplay-probe/v1":
        return "INVALID", ["probe_receipt_malformed"]
    if not isinstance(probe.get("pass"), bool) or not isinstance(probe.get("tripwires"), Mapping):
        return "INVALID", ["probe_receipt_malformed"]
    if not all(isinstance(value, bool) for value in probe["tripwires"].values()):
        return "INVALID", ["probe_receipt_malformed"]
    # Producer records pass predicates (rather than fault flags): each named
    # predicate must be true and the aggregate receipt must agree.
    if not probe["pass"] or not all(probe["tripwires"].values()):
        return "KILL_FIT", ["D4096_probe_tripwire"]
    collapse = probe.get("action_collapse")
    if not isinstance(collapse, Mapping) or not all(
        isinstance(value, Mapping) and isinstance(value.get("pass"), bool)
        for value in collapse.values()
    ):
        return "INVALID", ["probe_action_collapse_malformed"]
    if not all(value["pass"] for value in collapse.values()):
        return "KILL_FIT", ["D4096_probe_action_collapse"]
    admission = stage.get("admission")
    if admission is None:
        return "INCOMPLETE", ["missing_admission_receipt"]
    if not isinstance(admission, Mapping) or not isinstance(
        admission.get("harm_screen_pass"), bool
    ):
        return "INVALID", ["admission_receipt_malformed"]
    if isinstance(admission, Mapping) and admission.get("harm_screen_pass") is False:
        return "STOP_ADMISSION", ["harm_screen"]
    return None, []


def _observations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gameplay = payload.get("gameplay")
    if not isinstance(gameplay, Mapping) or not isinstance(gameplay.get("observations"), list):
        raise ValueError("gameplay observations missing")
    rows = gameplay["observations"]
    tripwires = gameplay.get("trajectory_tripwires")
    if not isinstance(tripwires, Mapping) or not all(
        isinstance(tripwires.get(arm), bool) for arm in ("D1024", "D4096")
    ):
        raise ValueError("malformed trajectory tripwires")
    if any(tripwires[arm] for arm in ("D1024", "D4096")):
        raise ValueError("trajectory tripwire")
    identities: set[str] = set()
    by_mix = {mix: [] for mix in MIXES}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("gameplay row malformed")
        world_id, mix = row.get("world_id"), row.get("mix")
        if (
            not isinstance(world_id, str)
            or not world_id
            or world_id in identities
            or mix not in by_mix
        ):
            raise ValueError("foreign or duplicate gameplay world")
        identities.add(world_id)
        for arm in ("B8", "D1024", "D4096"):
            value = _arm(row, arm)
            _metric(value, "mass_integral")
            survival = _survival(value)
            if not 0.0 <= survival <= 1.0:
                raise ValueError("survival fraction outside [0, 1]")
            if arm in ("B8", "D4096"):
                _boolean(value.get("no_death_episode"), f"{arm} no_death_episode")
        for arm in ("B8", "D1024", "D4096"):
            _validate_trace(_arm(row, arm), arm)
            trace = _arm(row, arm)["action_trace"]
            for start in range(0, len(trace) - 100_000 + 1, 100_000):
                window = trace[start : start + 100_000]
                counts = [
                    sum(item["served_action"] == action for item in window) for action in range(6)
                ]
                if max(counts) / len(window) > 0.90:
                    raise ValueError("trajectory tripwire")
        _boost(row, "D4096")
        by_mix[mix].append(row)
    if {mix: len(by_mix[mix]) for mix in MIXES} != {mix: N_PER_MIX for mix in MIXES}:
        raise ValueError("gameplay bank must have 128 rows per mix")
    return rows


def _screen_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = {mix: [row for row in rows if row["mix"] == mix] for mix in MIXES}
    top = {
        mix: [
            _metric(_arm(row, "D4096"), "mass_integral") - _metric(_arm(row, "B8"), "mass_integral")
            for row in grouped[mix]
        ]
        for mix in MIXES
    }
    dose = {
        mix: [
            _metric(_arm(row, "D4096"), "mass_integral")
            - _metric(_arm(row, "D1024"), "mass_integral")
            for row in grouped[mix]
        ]
        for mix in MIXES
    }
    lower = {
        mix: [
            _metric(_arm(row, "D1024"), "mass_integral") - _metric(_arm(row, "B8"), "mass_integral")
            for row in grouped[mix]
        ]
        for mix in MIXES
    }
    top_stats, dose_stats, lower_stats = (
        welch_stratified(top),
        welch_stratified(dose),
        welch_stratified(lower),
    )
    rejected, adjusted = holm([top_stats["p_value"], dose_stats["p_value"]], alpha=0.10)
    candidate = {mix: [_survival(_arm(row, "D4096")) for row in grouped[mix]] for mix in MIXES}
    incumbent = {mix: [_survival(_arm(row, "B8")) for row in grouped[mix]] for mix in MIXES}
    band = pooled_band_check(
        candidate, incumbent, pooled_margin=0.05, mix_margin=0.075, nominal_p=0.05
    )
    lower_candidate = {
        mix: [_survival(_arm(row, "D1024")) for row in grouped[mix]] for mix in MIXES
    }
    lower_band = pooled_band_check(
        lower_candidate, incumbent, pooled_margin=0.05, mix_margin=0.075, nominal_p=0.05
    )
    scripted = grouped["scripted"]
    no_death_delta = _mean(
        [
            float(_arm(row, "D4096")["no_death_episode"])
            - float(_arm(row, "B8")["no_death_episode"])
            for row in scripted
        ]
    )
    # R8 specifies the share over all decisions, so world shares may not be
    # averaged: terminal worlds have different delivered-decision counts.
    trace_totals = [_validate_trace(_arm(row, "D4096"), "D4096") for row in rows]
    boost_decisions = math.fsum(share * decisions for share, decisions in trace_totals)
    total_decisions = math.fsum(decisions for _, decisions in trace_totals)
    boost_share = boost_decisions / total_decisions
    top_point = top_stats["mean"] >= 40.0
    dose_point = dose_stats["mean"] >= 20.0
    top_guard = all(top_stats["per_mix"][mix]["upper90"] >= 0.0 for mix in MIXES)
    guards = {
        "top_per_mix_upper90_nonnegative": top_guard,
        "survival_band_v2": bool(band["passes"]),
        "scripted_no_death_delta": no_death_delta >= 0.03,
        "boost_share": 0.005 <= boost_share <= 0.10,
    }
    top_pass = bool(rejected[0] and top_point and all(guards.values()))
    dose_pass = bool(rejected[1] and dose_point)
    return {
        "top": top_stats,
        "dose": dose_stats,
        "D1024_vs_B8": {
            "contrast": lower_stats,
            "guards": {
                "top_per_mix_upper90_nonnegative": all(
                    lower_stats["per_mix"][mix]["upper90"] >= 0.0 for mix in MIXES
                ),
                "survival_band_v2": bool(lower_band["passes"]),
            },
            "survival_band": lower_band,
        },
        "holm": {
            "alpha": 0.10,
            "rejected": {"top": rejected[0], "dose": rejected[1]},
            "adjusted_p_values": {"top": adjusted[0], "dose": adjusted[1]},
        },
        "point_bars": {"top": top_point, "dose": dose_point},
        "guards": guards,
        "survival_band": band,
        "scripted_no_death_delta": no_death_delta,
        "boost_share": boost_share,
        "top_go_bars_pass": top_pass,
        "dose_go_bars_pass": dose_pass,
    }


def _product(payload: Mapping[str, Any]) -> dict[str, Any]:
    watch = payload.get("watch6")
    if not isinstance(watch, Mapping) or not isinstance(watch.get("sessions"), list):
        raise ValueError("Watch6 sessions missing")
    sessions = watch["sessions"]
    if len(sessions) != 128:
        raise ValueError("Watch6 requires exactly 128 sessions")
    seen: set[str] = set()
    death_rates: list[tuple[float, float]] = []
    masses: list[tuple[float, float]] = []
    for row in sessions:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("session_id"), str)
            or row["session_id"] in seen
        ):
            raise ValueError("foreign or duplicate Watch6 session")
        seen.add(row["session_id"])
        b8, top = _arm(row, "B8"), _arm(row, "D4096")
        b8_deaths, top_deaths = _metric(b8, "deaths"), _metric(top, "deaths")
        b8_decisions, top_decisions = _metric(b8, "hero_decisions"), _metric(top, "hero_decisions")
        if b8_decisions <= 0 or top_decisions <= 0:
            raise ValueError("Watch6 hero decisions must be positive")
        if b8_deaths < 0 or top_deaths < 0:
            raise ValueError("Watch6 deaths must be nonnegative")
        death_rates.append((b8_deaths / b8_decisions, top_deaths / top_decisions))
        b8_session_mass, top_session_mass = _metric(b8, "mass_integral"), _metric(
            top, "mass_integral"
        )
        if b8_session_mass < 0:
            raise ValueError("Watch6 B8 mass must be nonnegative")
        masses.append((b8_session_mass, top_session_mass))
    rng = np.random.default_rng(3403)
    indexes = rng.integers(0, len(sessions), size=(10_000, len(sessions)))
    b8_rates, top_rates = np.asarray(death_rates).T
    b8_mass, top_mass = np.asarray(masses).T
    # Bootstrap the registered aggregate counts / aggregate denominators, not
    # an average of per-session rates.
    b8_deaths = np.asarray([_metric(_arm(row, "B8"), "deaths") for row in sessions])
    top_deaths = np.asarray([_metric(_arm(row, "D4096"), "deaths") for row in sessions])
    b8_decisions = np.asarray([_metric(_arm(row, "B8"), "hero_decisions") for row in sessions])
    top_decisions = np.asarray([_metric(_arm(row, "D4096"), "hero_decisions") for row in sessions])
    top_rate = top_deaths[indexes].sum(axis=1) / top_decisions[indexes].sum(axis=1)
    b8_rate = b8_deaths[indexes].sum(axis=1) / b8_decisions[indexes].sum(axis=1)
    mass_top, mass_b8 = top_mass[indexes].mean(axis=1), b8_mass[indexes].mean(axis=1)
    valid_r, valid_q = b8_rate > 0.0, mass_b8 > 0.0
    ratios_r = top_rate[valid_r] / b8_rate[valid_r] - 1.0
    mass_ratio_raw = mass_top[valid_q] / mass_b8[valid_q]
    ratios_q = mass_ratio_raw - 1.0
    r_ci = np.percentile(ratios_r, [2.5, 97.5]).tolist() if len(ratios_r) else [None, None]
    q_ci = np.percentile(ratios_q, [2.5, 97.5]).tolist() if len(ratios_q) else [None, None]
    complete_bootstrap = bool(valid_r.all() and valid_q.all())
    point_r = (
        float(
            (top_deaths.sum() / top_decisions.sum()) / (b8_deaths.sum() / b8_decisions.sum()) - 1.0
        )
        if b8_deaths.sum() > 0
        else None
    )
    point_q = float(top_mass.mean() / b8_mass.mean() - 1.0) if b8_mass.mean() > 0 else None
    return {
        "bootstrap": {"seed": 3403, "resamples": 10_000},
        "death_ratio": {"point": point_r, "ci95": r_ci, "undefined_draws": int((~valid_r).sum())},
        "mass_ratio": {"point": point_q, "ci95": q_ci, "undefined_draws": int((~valid_q).sum())},
        # Compare the unshifted ratio to 0.95 so exact mathematical equality
        # does not turn into a floating-point -0.050000000000000044 failure.
        "passes": bool(
            complete_bootstrap
            and r_ci[1] < 0.0
            and float(np.percentile(mass_ratio_raw, 2.5)) >= 0.95
        ),
    }


def _structured_fault(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("kind") == "KILL_FIT" and isinstance(value.get("fault"), str):
        return True
    return any(
        isinstance(item, Mapping) and item.get("kind") == "KILL_FIT" for item in value.values()
    )


def h5000_admission(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create the bounded H5000-to-Watch6 admission artifact, never an R9 label.

    The Watch6 runner consumes this receipt's source and controller bindings. It
    only says whether every registered H5000 bar passed; it has no product data
    and therefore cannot be interpreted as a final scientific result.
    """
    base = {
        "schema": H5000_ADMISSION_SCHEMA,
        "registration_sha256": REGISTRATION_SHA256,
        "analyzer_sha256": source_sha256(),
        "scientific_result": False,
    }
    invalid = [] if payload.get("schema") == INPUT_SCHEMA else ["input_schema_mismatch"]
    invalid.extend(_validate_identity(payload))
    if invalid:
        return {**base, "status": "REFUSED", "reasons": invalid, "all_go_bars_pass": False}
    label, reasons = _path_status(payload)
    if label is not None:
        return {
            **base,
            "status": "NOT_RUN_BY_RULE",
            "reasons": [label, *reasons],
            "all_go_bars_pass": False,
        }
    try:
        screen = _screen_stats(_observations(payload))
    except ValueError as exc:
        return {**base, "status": "REFUSED", "reasons": [str(exc)], "all_go_bars_pass": False}
    passes = bool(screen["top_go_bars_pass"] and screen["dose_go_bars_pass"])
    return {
        **base,
        "status": "PASS_WATCH6" if passes else "NOT_RUN_BY_RULE",
        "reasons": [] if passes else ["H5000_GO_bars_not_all_passed"],
        "all_go_bars_pass": passes,
        "screen_input_sha256": _sha(payload["gameplay"]),
        "controller_sha256": dict(payload["identity"]["controller_sha256"]),
        "recipe_sha256": payload["identity"]["recipe_sha256"],
        "screen": screen,
    }


def analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the complete R9 precedence table once to a sealed F4 ledger."""
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "registration_sha256": REGISTRATION_SHA256,
        "analyzer_sha256": source_sha256(),
    }
    invalid = [] if payload.get("schema") == INPUT_SCHEMA else ["input_schema_mismatch"]
    invalid.extend(_validate_identity(payload))
    if invalid:
        return {**result, "label": "INVALID", "reasons": invalid, "flag": None}
    probe_value = payload.get("probe")
    if isinstance(probe_value, Mapping) and _structured_fault(
        probe_value.get("known_controller_fault", {})
    ):
        return {**result, "label": "KILL_FIT", "reasons": ["probe_controller_fault"], "flag": None}
    # Foreign/duplicate materialized gameplay is an INVALID ledger even when
    # a later path rule would otherwise end at an earlier staging label.
    if payload.get("gameplay") is not None:
        try:
            _observations(payload)
        except ValueError as exc:
            if any(word in str(exc) for word in ("foreign", "duplicate", "malformed")):
                return {**result, "label": "INVALID", "reasons": [str(exc)], "flag": None}
    label, reasons = _path_status(payload)
    if label is not None:
        return {**result, "label": label, "reasons": reasons, "flag": None}
    try:
        rows = _observations(payload)
    except ValueError as exc:
        kind = (
            "INVALID"
            if any(word in str(exc) for word in ("foreign", "duplicate", "malformed"))
            else "KILL_FIT" if "tripwire" in str(exc) else "INCOMPLETE"
        )
        return {**result, "label": kind, "reasons": [str(exc)], "flag": None}
    screen = _screen_stats(rows)
    # The closure bounds are deliberately unadjusted and precede every GO bar.
    pre_flag = (
        "TOP_PASSED_DOSE_FAILED"
        if screen["top_go_bars_pass"] and not screen["dose_go_bars_pass"]
        else None
    )
    if screen["dose"]["upper90"] <= 0.0:
        return {
            **result,
            "label": "KILL_DOSE_CLOSED",
            "reasons": ["dose_upper90_nonpositive"],
            "screen": screen,
            "flag": pre_flag,
        }
    if screen["dose"]["upper90"] < 20.0:
        return {
            **result,
            "label": "KILL_DOSE_FUTILE",
            "reasons": ["dose_upper90_below_20"],
            "screen": screen,
            "flag": pre_flag,
        }
    flag = pre_flag
    h5000_go = bool(screen["top_go_bars_pass"] and screen["dose_go_bars_pass"])
    if not h5000_go:
        return {
            **result,
            "label": "INCONCLUSIVE",
            "reasons": ["H5000_GO_bars_not_all_passed"],
            "screen": screen,
            "flag": flag,
            "watch6": "NOT_RUN_BY_RULE",
        }
    admission = payload.get("h5000_admission")
    expected_admission = {
        "schema": H5000_ADMISSION_SCHEMA,
        "status": "PASS_WATCH6",
        "all_go_bars_pass": True,
        "screen_input_sha256": _sha(payload["gameplay"]),
        "controller_sha256": payload["identity"]["controller_sha256"],
        "recipe_sha256": payload["identity"]["recipe_sha256"],
        "analyzer_sha256": source_sha256(),
    }
    if (
        not isinstance(admission, Mapping)
        or any(admission.get(key) != value for key, value in expected_admission.items())
        or payload["identity"].get("h5000_admission_sha256") != _sha(admission)
    ):
        return {
            **result,
            "label": "INCOMPLETE",
            "reasons": ["sealed_H5000_admission_receipt_required"],
            "screen": screen,
            "flag": flag,
        }
    try:
        watch_value = payload.get("watch6")
        if isinstance(watch_value, Mapping) and _structured_fault(
            watch_value.get("known_controller_fault", {})
        ):
            return {
                **result,
                "label": "KILL_FIT",
                "reasons": ["Watch6_controller_fault"],
                "screen": screen,
                "flag": flag,
            }
        if isinstance(watch_value, Mapping):
            collapse = watch_value.get("action_collapse")
            if isinstance(collapse, Mapping) and any(
                isinstance(value, Mapping) and value.get("pass") is False
                for value in collapse.values()
            ):
                return {
                    **result,
                    "label": "KILL_FIT",
                    "reasons": ["Watch6_action_collapse"],
                    "screen": screen,
                    "flag": flag,
                }
        product = _product(payload)
    except ValueError as exc:
        return {
            **result,
            "label": (
                "INVALID"
                if any(word in str(exc) for word in ("foreign", "duplicate", "malformed"))
                else "INCOMPLETE"
            ),
            "reasons": [str(exc)],
            "screen": screen,
            "flag": flag,
        }
    label = "GO_TOP_DOSE_DEV" if product["passes"] else "INCONCLUSIVE"
    return {
        **result,
        "label": label,
        "reasons": [] if product["passes"] else ["Watch6_product_bars_failed"],
        "screen": screen,
        "product": product,
        "flag": flag,
    }


def main() -> None:
    """Create a single canonical receipt; refuse replacing an existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("final", "h5000-admission"), default="final")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("analysis output must be create-only")
    value = json.loads(args.input.read_text())
    output = h5000_admission(value) if args.kind == "h5000-admission" else analyze(value)
    output["input_sha256"] = _sha(value)
    output["source_sha256"] = source_sha256()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_json(args.output, output)


if __name__ == "__main__":
    main()
