"""Pure R6/R7/R9 rules for the adopted F5 W-versus-C study.

The module intentionally has no path, checkpoint, label, dispatch, or simulator
API.  A stage runner may pass only its already sealed/reduced records here.  In
particular, these functions are safe to exercise with synthetic fixtures before
any F5 scientific payload is opened.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.evaluation.sequential_gate import pooled_band_check
from src.scripts.eval_stats import holm, student_t_isf, student_t_sf

MIXES = ("scripted", "frozen", "mixed")
N_PER_MIX = 128
BOOTSTRAP_SEED = 3403
BOOTSTRAP_RESAMPLES = 10_000
# R7 says "original absolute floor/implementation".  The retained survival
# band-v2 ratification specifies the original .30 floor; do not substitute an
# F4-local convention.
SURVIVAL_ABSOLUTE_FLOOR = 0.30
GROUPS = (
    ("archive/watch6", 64),
    ("archive/scripted", 22),
    ("archive/frozen", 21),
    ("archive/mixed", 21),
    ("probe/scripted", 11),
    ("probe/frozen", 11),
    ("probe/mixed", 10),
    ("gameplay/scripted", 128),
    ("gameplay/frozen", 128),
    ("gameplay/mixed", 128),
    ("product/watch6", 128),
    ("timing/watch6", 3),
    ("timing/scripted", 3),
    ("timing/frozen", 3),
    ("timing/mixed", 3),
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _ordered(value: Any, name: str) -> Decimal | Fraction | float | int:
    """Accept the Decimal/Fraction R10c boundary fixtures without rounding them."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return math.fsum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two paired worlds required")
    average = _mean(values)
    return math.fsum((item - average) ** 2 for item in values) / (len(values) - 1)


def select_candidate(
    confirmed: Any = None,
    unknown: Any = None,
    confirmed_low: Any = None,
    *,
    complete: bool = True,
    valid: bool = True,
) -> str:
    """Implement R4's aggregate-only selector without inspecting row outcomes."""
    if not valid:
        return "INVALID"
    if not complete or None in (confirmed, unknown, confirmed_low):
        return "HOLD_DIAGNOSTIC"
    values = (confirmed, unknown, confirmed_low)
    if any(type(value) is not int or value < 0 for value in values) or confirmed_low > confirmed:
        return "INVALID"
    if confirmed + unknown == 0:
        return "HOLD_NO_TARGET"
    if unknown > 0 and unknown >= confirmed:
        return "COV"
    return "P" if 2 * confirmed_low >= confirmed else "W"


def margin_bin(value: Any) -> int:
    """Return the R4 exact Decimal margin bin; reject nonfinite/negative values."""
    try:
        margin = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid margin") from exc
    if not margin.is_finite() or margin < 0:
        raise ValueError("invalid margin")
    if margin < Decimal(".01"):
        return 0
    if margin < Decimal(".1"):
        return 1
    if margin < Decimal(1):
        return 2
    return 3


def conservative_harm(
    changed: bool, selected_exact: bool, bare_exact: bool, selected_dies: bool, bare_survives: bool
) -> bool:
    """F4 conservative ordinary-harm predicate retained verbatim by R6."""
    return bool(
        changed and (not selected_exact or not bare_exact or (selected_dies and bare_survives))
    )


def cov_status(
    required: Iterable[Any],
    completed: Iterable[Any],
    *,
    original_exact_unchanged: bool = True,
    strata_unchanged: bool = True,
) -> str:
    """Classify the selected COV completion without a fallback route."""
    required_set, completed_list = set(required), list(completed)
    if not original_exact_unchanged or not strata_unchanged:
        return "INVALID"
    if len(required_set) > 46_080:
        return "HOLD_COVERAGE_IMPLEMENTATION"
    if not required_set:
        return "HOLD_NO_TARGET"
    completed_set = set(completed_list)
    if len(completed_list) != len(completed_set) or not completed_set <= required_set:
        return "INVALID"
    return "COMPLETE" if completed_set == required_set else "INCOMPLETE"


def adoption_order(
    adopt: Any,
    completed: Any,
    published: Any,
    exposed: Any,
    selected: Any,
    selection_pub: Any,
    work: Any,
) -> bool:
    """R10a causal ordering predicate; values need only be mutually ordered."""
    return bool(
        adopt < exposed <= selected <= selection_pub < work and completed <= published < selected
    )


def ordinary_phase(parent: str) -> int:
    """R6a counter-hash rejection sampler, deterministic per parent without redraw."""
    if not isinstance(parent, str) or not parent:
        raise ValueError("parent must be nonempty")
    limit = (1 << 256) - (1 << 256) % 200
    for counter in range(1_000_000):
        raw = f"bl-f5-predat-v2/20261025|ordinary-phase|{parent}|{counter}"
        integer = int.from_bytes(sha256(raw.encode()).digest(), "big")
        if integer < limit:
            return integer % 200
    raise RuntimeError("HOLD: counter guard")


def allocate_round(round_index: int) -> list[tuple[int, str, int]]:
    """Materialize the whole R6a allocation round; never replace one world."""
    if type(round_index) is not int or not 0 <= round_index < 1000:
        raise ValueError("round must be an integer in [0, 1000)")
    rows = []
    for group, count in GROUPS:
        for index in range(count):
            key = f"bl-f5-predat-v2/20261025|r={round_index}/{group}|{index}"
            seed = int.from_bytes(sha256(key.encode()).digest()[:4], "big")
            parent = f"bl-f5-predat-v2/20261025/r={round_index}/{group}/{index:04d}"
            rows.append((len(rows), parent, seed))
    return rows


def first_clear_round(
    blocked_seeds: set[int], blocked_parents: set[str], *, audit_complete: bool = True
) -> tuple[int | str, list[tuple[int, str, int]] | None]:
    """Choose only a complete collision-free allocation round."""
    if not audit_complete:
        return "HOLD", None
    for round_index in range(1000):
        rows = allocate_round(round_index)
        seeds, parents = [row[2] for row in rows], {row[1] for row in rows}
        if (
            len(set(seeds)) == len(seeds)
            and not set(seeds) & blocked_seeds
            and not parents & blocked_parents
        ):
            return round_index, rows
    return "HOLD", None


def support_status(
    *, rescue_parents: int, natural_rows: int, scheduled_census_complete: bool
) -> str | None:
    """Return R6 support failure, keeping incomplete census above small support."""
    if type(rescue_parents) is not int or type(natural_rows) is not int:
        return "INVALID"
    if rescue_parents < 0 or natural_rows < 0:
        return "INVALID"
    if not scheduled_census_complete:
        return "INCOMPLETE"
    return "INSUFFICIENT" if rescue_parents < 32 or natural_rows < 1000 else None


def action_collapse_flag(actions: Sequence[int]) -> bool:
    """True for any sliding 100k-decision window with share strictly above 90%."""
    if any(type(action) is not int or not 0 <= action < 6 for action in actions):
        raise ValueError("illegal action")
    width = 100_000
    if len(actions) < width:
        return False
    counts = [0] * 6
    for action in actions[:width]:
        counts[action] += 1
    if max(counts) > 90_000:
        return True
    for index in range(width, len(actions)):
        counts[actions[index - width]] -= 1
        counts[actions[index]] += 1
        if max(counts) > 90_000:
            return True
    return False


def fit_drift(
    disagreements: Sequence[float],
    boost_share: float,
    bare_boost_share: float,
    decisive_retention: float,
) -> bool:
    """Apply the P7 per-arm drift limits using unrounded comparisons."""
    if len(disagreements) != 10:
        raise ValueError("P7 requires ten nonempty rank-decile disagreements")
    values = [_ordered(value, "disagreement") for value in disagreements]
    boost, bare, retention = (
        _ordered(boost_share, "boost share"),
        _ordered(bare_boost_share, "bare boost share"),
        _ordered(decisive_retention, "decisive retention"),
    )
    # Native float fixtures intentionally use native float endpoints (``.05``
    # and ``.70`` pass); Decimal/Fraction fixtures retain their exact endpoint.
    disagreement = any(
        value > (0.05 if isinstance(value, float) else Fraction(1, 20)) for value in values
    )
    boost_limit = (
        1.5 * bare if isinstance(boost, float) or isinstance(bare, float) else Fraction(3, 2) * bare
    )
    retention_floor = 0.70 if isinstance(retention, float) else Fraction(7, 10)
    return disagreement or boost > boost_limit or retention < retention_floor


def rank_decile_disagreements(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Compute P7's ten rank-decile disagreement shares from reduced probe rows.

    Each row is one candidate-visited decision with a nonnegative length,
    bank index and decision index plus a boolean ``disagrees`` field.  Sorting
    by the frozen tuple prevents ties from moving a failing observation.
    """
    if len(rows) < 10:
        raise ValueError("P7 needs ten nonempty rank deciles")
    parsed: list[tuple[float, int, int, bool]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe row must be a mapping")
        length = _finite(row.get("length"), "length")
        bank_index, decision, disagrees = (
            row.get("bank_index"),
            row.get("decision"),
            row.get("disagrees"),
        )
        if (
            length < 0
            or type(bank_index) is not int
            or bank_index < 0
            or type(decision) is not int
            or decision < 0
        ):
            raise ValueError("invalid P7 rank key")
        if type(disagrees) is not bool:
            raise ValueError("P7 disagreement must be boolean")
        parsed.append((length, bank_index, decision, disagrees))
    if len({(length, bank, decision) for length, bank, decision, _ in parsed}) != len(parsed):
        raise ValueError("duplicate P7 decision")
    buckets: list[list[bool]] = [[] for _ in range(10)]
    for rank, (_, _, _, disagrees) in enumerate(sorted(parsed)):
        buckets[(10 * rank) // len(parsed)].append(disagrees)
    if any(not bucket for bucket in buckets):
        raise ValueError("P7 rank decile empty")
    return tuple(sum(bucket) / len(bucket) for bucket in buckets)


def r6_adapter(
    *,
    t_drift: bool,
    c_drift: bool,
    t_unconditional_fault: bool = False,
    c_unconditional_fault: bool = False,
    rescue_parents: int = 32,
    natural_rows: int = 1000,
    scheduled_census_complete: bool = True,
    t_retention: float = 0.70,
    c_retention: float = 0.70,
    t_harm: int = 0,
    invalid: bool = False,
    incomplete: bool = False,
) -> dict[str, Any]:
    """Translate sealed R6 summaries into retained flags and stage status.

    A C-only P7 failure is deliberately a matched-study ``KILL_FIT``.
    """
    flags: set[str] = set()
    affected: list[str] = []
    if invalid:
        flags.add("INVALID")
    if incomplete:
        flags.add("INCOMPLETE")
    support = support_status(
        rescue_parents=rescue_parents,
        natural_rows=natural_rows,
        scheduled_census_complete=scheduled_census_complete,
    )
    if support is not None:
        flags.add(support)
    for arm, drift, fault in (
        ("T", t_drift, t_unconditional_fault),
        ("C", c_drift, c_unconditional_fault),
    ):
        if drift or fault:
            flags.add("KILL_FIT")
            affected.append(arm)
    t_value, c_value = _finite(t_retention, "T retention"), _finite(c_retention, "C retention")
    if t_value < 0.70:
        flags.add("KILL_FIT")
        if "T" not in affected:
            affected.append("T")
    if c_value < 0.70:
        flags.add("KILL_FIT")
        if "C" not in affected:
            affected.append("C")
    if t_value >= 0.70 and c_value >= 0.70 and t_value < c_value:
        flags.add("KILL_RETENTION")
    if type(t_harm) is not int or t_harm < 0:
        flags.add("INVALID")
    elif t_harm > 0:
        flags.add("STOP_ADMISSION")
    primary = r9_disposition(flags)
    status = "PASS_OFFLINE_OPEN_S3" if primary == "GO_F5_DEV" else primary
    reason_flags = set(flags)
    if t_drift or t_unconditional_fault or t_value < 0.70:
        reason_flags.add("T_P7_FIT_DRIFT")
    if c_drift or c_unconditional_fault or c_value < 0.70:
        reason_flags.add("C_P7_FIT_DRIFT")
    return {
        "flags": sorted(flags),
        "reason_flags": sorted(reason_flags),
        "affected_fits": affected,
        "status": status,
        "interpretation": exploratory_screening(status, sorted(flags)),
    }


def welch_equal_mix(deltas: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """R7 equal-mix Welch statistic for the frozen 128 worlds per mix."""
    if tuple(deltas) != MIXES or any(len(deltas[mix]) != N_PER_MIX for mix in MIXES):
        raise ValueError("R7 requires ordered scripted, frozen, mixed arrays of 128 deltas")
    parsed = {mix: [_finite(item, f"{mix} delta") for item in deltas[mix]] for mix in MIXES}
    means = {mix: _mean(parsed[mix]) for mix in MIXES}
    variances = {mix: _variance(parsed[mix]) for mix in MIXES}
    pieces = {mix: variances[mix] / (9 * N_PER_MIX) for mix in MIXES}
    variance = math.fsum(pieces.values())
    mean = _mean(list(means.values()))
    if variance == 0.0:
        lower = upper = mean
        p_value = 0.0 if mean > 0 else 1.0
        df = float(N_PER_MIX - 1)
    else:
        df = math.fsum(pieces.values()) ** 2 / math.fsum(
            piece**2 / 127 for piece in pieces.values()
        )
        se = math.sqrt(variance)
        critical = student_t_isf(0.10, df)
        lower, upper = mean - critical * se, mean + critical * se
        p_value = student_t_sf(mean / se, df)
    per_mix = {}
    for mix in MIXES:
        se = math.sqrt(variances[mix] / N_PER_MIX)
        if se == 0:
            low = high = means[mix]
        else:
            critical = student_t_isf(0.10, 127)
            low, high = means[mix] - critical * se, means[mix] + critical * se
        per_mix[mix] = {
            "mean": means[mix],
            "variance": variances[mix],
            "lower90": low,
            "upper90": high,
        }
    return {
        "mean": mean,
        "variance": variance,
        "standard_error": math.sqrt(variance),
        "df": df,
        "lower90": lower,
        "upper90": upper,
        "p_value": p_value,
        "per_mix": per_mix,
    }


def r7_adapter(
    *,
    top: Mapping[str, Sequence[float]],
    change: Mapping[str, Sequence[float]],
    t_survival: Mapping[str, Sequence[float]],
    b8_survival: Mapping[str, Sequence[float]],
    scripted_no_death_gain: float,
    delivered_boost_share: float,
) -> dict[str, Any]:
    """Compute both fixed R7 contrasts, Holm family, and all H5000 guards."""
    top_stats, change_stats = welch_equal_mix(top), welch_equal_mix(change)
    rejected, adjusted = holm([top_stats["p_value"], change_stats["p_value"]], alpha=0.10)
    survival = pooled_band_check(
        t_survival,
        b8_survival,
        pooled_margin=0.05,
        mix_margin=0.075,
        nominal_p=0.05,
        floor=SURVIVAL_ABSOLUTE_FLOOR,
    )
    mix_guard = all(top_stats["per_mix"][mix]["upper90"] >= 0.0 for mix in MIXES)
    gain, boost = _finite(scripted_no_death_gain, "scripted gain"), _finite(
        delivered_boost_share, "boost share"
    )
    flags: set[str] = set()
    guards = {
        "top_per_mix_upper90_nonnegative": mix_guard,
        "survival_band_v2": bool(survival["passes"]),
        "scripted_no_death_gain_ge_0_03": gain >= 0.03,
        "boost_in_closed_interval": 0.005 <= boost <= 0.10,
    }
    if not (mix_guard and survival["passes"] and guards["scripted_no_death_gain_ge_0_03"]):
        flags.add("KILL_GAMEPLAY_GUARD")
    # The boost bar is a non-kill advancement bar, even when it is the sole failure.
    top_pass = rejected[0] and top_stats["mean"] >= 40 and all(guards.values())
    change_pass = rejected[1] and change_stats["mean"] >= 20
    if change_stats["upper90"] <= 0:
        flags.add("KILL_CHANGE_CLOSED")
    elif change_stats["upper90"] < 20:
        flags.add("KILL_CHANGE_FUTILE")
    primary = r9_disposition(
        flags, ub_change=change_stats["upper90"], all_pass=bool(top_pass and change_pass)
    )
    # S3 merely authorizes the separately fixed product look; R9's terminal
    # GO_F5_DEV is possible only after product has passed.
    status = "PASS_H5000_OPEN_S4" if primary == "GO_F5_DEV" else primary
    reason_flags = set(flags)
    if not mix_guard:
        reason_flags.add("top_per_mix_upper90_negative")
    if not survival["passes"]:
        reason_flags.add("survival_band_v2_failed")
    if not guards["scripted_no_death_gain_ge_0_03"]:
        reason_flags.add("scripted_no_death_gain_below_0_03")
    if not guards["boost_in_closed_interval"]:
        reason_flags.add("boost_outside_closed_interval")
    if not rejected[0]:
        reason_flags.add("top_holm_failed")
    if top_stats["mean"] < 40:
        reason_flags.add("top_point_below_40")
    if not rejected[1]:
        reason_flags.add("change_holm_failed")
    if change_stats["mean"] < 20:
        reason_flags.add("change_point_below_20")
    return {
        "top": top_stats,
        "change": change_stats,
        "holm": {"rejected": rejected, "adjusted_p_values": adjusted},
        "survival": survival,
        "guards": guards,
        "flags": sorted(flags),
        "reason_flags": sorted(reason_flags),
        "all_h5000_bars_pass": bool(top_pass and change_pass),
        "status": status,
        "interpretation": exploratory_screening(status, sorted(flags)),
    }


def survival_operating_characteristic(
    *, effect: float, trials: int = 256, worlds_per_mix: int = N_PER_MIX, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Synthetic pre-data OC for the exact R7 pooled survival implementation.

    It deliberately generates no game states: each trial is a bounded numeric
    paired-survival fixture and is passed through the same ``pooled_band_check``
    used by :func:`r7_adapter`.  This verifies deterministic structural
    behavior, not a power claim for F5.
    """
    if (
        type(trials) is not int
        or trials <= 0
        or type(worlds_per_mix) is not int
        or worlds_per_mix < 2
    ):
        raise ValueError("synthetic OC needs positive trials and at least two worlds")
    shift = _finite(effect, "survival effect")
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(trials):
        incumbent = {mix: rng.uniform(0.35, 0.85, worlds_per_mix).tolist() for mix in MIXES}
        candidate = {
            mix: np.clip(
                np.asarray(incumbent[mix]) + shift + rng.normal(0, 0.03, worlds_per_mix), 0, 1
            ).tolist()
            for mix in MIXES
        }
        passes += bool(
            pooled_band_check(
                candidate,
                incumbent,
                pooled_margin=0.05,
                mix_margin=0.075,
                nominal_p=0.05,
                floor=SURVIVAL_ABSOLUTE_FLOOR,
            )["passes"]
        )
    return {
        "seed": seed,
        "trials": trials,
        "worlds_per_mix": worlds_per_mix,
        "effect": shift,
        "pass_count": passes,
        "pass_rate": passes / trials,
        "synthetic_only": True,
        "confirmatory": False,
        "release": False,
    }


def product_bootstrap(sessions: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Paired-session 10k percentile bootstrap; undefined draws are insufficient."""
    if len(sessions) != 128:
        raise ValueError("R7 product requires exactly 128 paired sessions")
    required = ("t_deaths", "t_decisions", "b8_deaths", "b8_decisions", "t_mass", "b8_mass")
    rows = [[_finite(row.get(key), key) for key in required] for row in sessions]
    if any(
        row[1] <= 0 or row[3] <= 0 or row[0] < 0 or row[2] < 0 or row[4] < 0 or row[5] < 0
        for row in rows
    ):
        raise ValueError("invalid product numerator or denominator")
    values = np.asarray(rows, dtype=float)
    indexes = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, 128, size=(BOOTSTRAP_RESAMPLES, 128)
    )
    drawn = values[indexes]
    b8_rate = drawn[:, :, 2].sum(1) / drawn[:, :, 3].sum(1)
    t_rate = drawn[:, :, 0].sum(1) / drawn[:, :, 1].sum(1)
    b8_mass, t_mass = drawn[:, :, 5].mean(1), drawn[:, :, 4].mean(1)
    valid = (b8_rate > 0) & (b8_mass > 0)
    if not bool(valid.all()):
        return {
            "status": "INSUFFICIENT",
            "undefined_draws": int((~valid).sum()),
            "passes": False,
            "interpretation": exploratory_screening("INSUFFICIENT", ["undefined_bootstrap_ratio"]),
        }
    ratio_r, mass_ratio = t_rate / b8_rate - 1, t_mass / b8_mass
    ratio_q = mass_ratio - 1
    r_low, r_high = np.percentile(ratio_r, [2.5, 97.5])
    q_low, q_high = np.percentile(ratio_q, [2.5, 97.5])
    # Compare the unshifted ratio so exact .95 cannot become
    # -0.050000000000000044 after subtraction and fail an inclusive bar.
    passes = bool(r_high < 0 and float(np.percentile(mass_ratio, 2.5)) >= 0.95)
    status = "PASS_PRODUCT" if passes else "KILL_PRODUCT"
    return {
        "status": status,
        "undefined_draws": 0,
        "death_ratio_ci95": (float(r_low), float(r_high)),
        "mass_ratio_ci95": (float(q_low), float(q_high)),
        "passes": passes,
        "interpretation": exploratory_screening(status, [] if passes else ["product_bar_failed"]),
    }


def r9_disposition(flags: Iterable[str], *, ub_change: float = 20.0, all_pass: bool = True) -> str:
    """Return the R9 primary disposition while callers retain every supplied flag."""
    items = set(flags)
    for label in ("INVALID", "INCOMPLETE", "KILL_FIT", "INSUFFICIENT"):
        if label in items:
            return label
    for label in ("HOLD_DIAGNOSTIC", "HOLD_NO_TARGET", "HOLD_COVERAGE_IMPLEMENTATION", "HOLD"):
        if label in items:
            return label
    for label in ("KILL_RETENTION", "STOP_ADMISSION", "KILL_GAMEPLAY_GUARD"):
        if label in items:
            return label
    bound = _ordered(ub_change, "change upper90")
    if bound <= 0:
        return "KILL_CHANGE_CLOSED"
    if bound < 20:
        return "KILL_CHANGE_FUTILE"
    if "KILL_PRODUCT" in items:
        return "KILL_PRODUCT"
    return "GO_F5_DEV" if all_pass else "INCONCLUSIVE"


def exploratory_screening(status: str, reasons: Iterable[str] = ()) -> dict[str, Any]:
    """Make the non-confirmatory/non-release boundary explicit in every adapter output."""
    return {
        "classification": "EXPLORATORY_SCREENING",
        "status": status,
        "reasons": list(reasons),
        "confirmatory": False,
        "release": False,
        "confirmatory_efficacy_claim": False,
        "release_authorized": False,
    }


def final_lifecycle(*, s2_status: str, s3_status: str, s4_status: str) -> dict[str, Any]:
    """Issue GO_F5_DEV only after each separately fixed stage has passed."""
    stages = (s2_status, s3_status, s4_status)
    required = ("PASS_OFFLINE_OPEN_S3", "PASS_H5000_OPEN_S4", "PASS_PRODUCT")
    terminal = {
        status
        for status in stages
        if status.startswith(("INVALID", "INCOMPLETE", "KILL_", "INSUFFICIENT", "HOLD", "STOP_"))
    }
    status = "GO_F5_DEV" if stages == required else r9_disposition(terminal, all_pass=False)
    return {
        "status": status,
        "stage_statuses": {"S2": s2_status, "S3": s3_status, "S4": s4_status},
        "interpretation": exploratory_screening(status),
    }
