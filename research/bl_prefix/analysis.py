"""Fail-closed calculations for the E6 and E7 development protocols.

The producer writes one row per continuation.  This module deliberately does
not infer from continuation rows: RNG streams are first averaged within a
source world and mixes are then given equal weight.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from src.scripts.eval_stats import student_t_isf

MIXES = ("scripted", "frozen", "mixed")
_Z90 = 1.2815515655446004


class IncompleteResult(ValueError):
    """A required input/result row is absent or inconsistent."""


def _one_sided_wilson(successes: int, total: int) -> dict[str, float | int]:
    """One-sided 90% Wilson display for the frozen E7 point rule."""
    if total <= 0 or not 0 <= successes <= total:
        raise IncompleteResult("invalid Wilson count")
    point = successes / total
    denominator = 1.0 + _Z90**2 / total
    center = (point + _Z90**2 / (2 * total)) / denominator
    radius = _Z90 * math.sqrt(point * (1 - point) / total + _Z90**2 / (4 * total**2)) / denominator
    return {
        "count": successes,
        "total": total,
        "point": point,
        "lb90": max(0.0, center - radius),
        "ub90": min(1.0, center + radius),
    }


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IncompleteResult(f"{name} must be finite")
    return float(value)


def _welch(parts: Mapping[str, list[float]]) -> dict[str, float | int]:
    if set(parts) != set(MIXES) or any(len(parts[mix]) < 2 for mix in MIXES):
        raise IncompleteResult("each mix needs at least two distinct source worlds")
    means = {mix: math.fsum(values) / len(values) for mix, values in parts.items()}
    variances = {
        mix: math.fsum((value - means[mix]) ** 2 for value in values) / (len(values) - 1)
        for mix, values in parts.items()
    }
    # Equal mix weighting.  Independent source worlds are the sole inferential units.
    terms = {mix: variances[mix] / len(parts[mix]) / 9.0 for mix in MIXES}
    se2 = math.fsum(terms.values())
    mean = math.fsum(means.values()) / 3.0
    if se2 == 0.0:
        return {"mean": mean, "se": 0.0, "df": None, "lb90": mean, "ub90": mean, "mix_means": means}
    df = se2**2 / math.fsum((term**2) / (len(parts[mix]) - 1) for mix, term in terms.items())
    critical = student_t_isf(0.1, df)
    se = math.sqrt(se2)
    return {
        "mean": mean,
        "se": se,
        "df": df,
        "lb90": mean - critical * se,
        "ub90": mean + critical * se,
        "mix_means": means,
    }


def _epd_welch(strata: Mapping[str, Mapping[str, list[float]]]) -> dict[str, Any]:
    """Equal-weight six independent (stratum, mix) cells; no arbitrary pairing."""
    terms: list[tuple[float, int, str, str]] = []
    means: dict[str, dict[str, float]] = {}
    for stratum in ("early", "ordinary"):
        means[stratum] = {}
        for mix in MIXES:
            values = strata[stratum][mix]
            if len(values) < 2:
                raise IncompleteResult("EPD requires two worlds per stratum/mix")
            average = math.fsum(values) / len(values)
            variance = math.fsum((x - average) ** 2 for x in values) / (len(values) - 1)
            means[stratum][mix] = average
            terms.append((variance / len(values) / 36.0, len(values) - 1, stratum, mix))
    estimate = math.fsum(means[s][m] for s in means for m in MIXES) / 6.0
    variance = math.fsum(term for term, *_ in terms)
    if variance == 0.0:
        return {
            "mean": estimate,
            "se": 0.0,
            "df": None,
            "lb90": estimate,
            "ub90": estimate,
            "cell_means": means,
        }
    df = variance**2 / math.fsum(term**2 / degrees for term, degrees, *_ in terms)
    width = student_t_isf(0.1, df) * math.sqrt(variance)
    return {
        "mean": estimate,
        "se": math.sqrt(variance),
        "df": df,
        "lb90": estimate - width,
        "ub90": estimate + width,
        "cell_means": means,
    }


def e6_summary(
    rows: Iterable[Mapping[str, Any]],
    controller: str,
    *,
    expected_early: Iterable[str] | None = None,
    expected_ordinary: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute the registered E6 controller-versus-B8 primary statistic."""
    grouped: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("phase") != "primary" or row.get("controller") not in {controller, "B8"}:
            continue
        key = (
            str(row.get("stratum")),
            str(row.get("mix")),
            str(row.get("source_world")),
            int(row.get("stream", -1)),
        )
        if key[0] not in {"early", "ordinary"} or key[1] not in MIXES or key[3] not in (0, 1):
            raise IncompleteResult("malformed E6 continuation identity")
        label = str(row["controller"])
        if label in grouped[key]:
            raise IncompleteResult("duplicate E6 continuation row")
        grouped[key][label] = _number(row.get("score"), "E6 score")
    worlds: dict[str, dict[str, list[float]]] = {
        "early": defaultdict(list),
        "ordinary": defaultdict(list),
    }
    by_root: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, values in grouped.items():
        if set(values) != {controller, "B8"}:
            raise IncompleteResult("missing E6 controller/B8 row")
        by_root[key[:3]].append(values[controller] - values["B8"])
    for (stratum, mix, _source), deltas in by_root.items():
        if len(deltas) != 2:
            raise IncompleteResult("E6 requires two future RNG streams per root")
        worlds[stratum][mix].append(math.fsum(deltas) / 2.0)
    if set(worlds["early"]) != set(MIXES) or set(worlds["ordinary"]) != set(MIXES):
        raise IncompleteResult("missing E6 stratum or mix")
    expected = {"early": set(expected_early or ()), "ordinary": set(expected_ordinary or ())}
    observed = {
        stratum: set(source for (stratum2, _mix, source) in by_root if stratum2 == stratum)
        for stratum in ("early", "ordinary")
    }
    if any(expected[stratum] and observed[stratum] != expected[stratum] for stratum in expected):
        raise IncompleteResult("E6 rows do not cover the exact audited bank")
    if expected_early is None and len(observed["early"]) != 128:
        raise IncompleteResult("scientific E6 requires 128 early source worlds")
    if expected_ordinary is None and len(observed["ordinary"]) != 128:
        raise IncompleteResult("scientific E6 requires 128 ordinary source worlds")
    early, ordinary = _welch(worlds["early"]), _welch(worlds["ordinary"])
    epd = _epd_welch(worlds)
    if early["ub90"] <= 0.0:
        decision = "WRONG_MECHANISM"
    elif early["lb90"] > 0.0 and ordinary["mean"] >= -2.0:
        decision = "NEEDS_SCALE"
    else:
        decision = "INCONCLUSIVE"
    return {
        "controller": controller,
        "early": early,
        "ordinary": ordinary,
        "epd": epd,
        "decision": decision,
        "world_counts": {s: {m: len(v) for m, v in worlds[s].items()} for s in worlds},
    }


def e7_summary(
    rows: Iterable[Mapping[str, Any]], expected_sources: Iterable[str]
) -> dict[str, Any]:
    """Apply E7's confirmation rule, once per source world and per ancestor."""
    expected_list = list(expected_sources)
    expected = set(expected_list)
    if not expected or len(expected) != len(expected_list):
        raise IncompleteResult("E7 needs nonempty unique audited source worlds")
    found: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source, ancestor = str(row.get("source_world")), int(row.get("ancestor", -1))
        if source not in expected or ancestor not in (100, 150, 200):
            raise IncompleteResult("unexpected E7 source/ancestor")
        found[source, ancestor].append(row)
    confirmed: dict[str, dict[int, bool]] = {source: {} for source in expected}
    productive: dict[str, dict[int, bool]] = {source: {} for source in expected}
    fallback_total = 0
    for source in expected:
        for ancestor in (100, 150, 200):
            entries = found[source, ancestor]
            if len(entries) != 2 or {int(x.get("stream", -1)) for x in entries} != {0, 1}:
                raise IncompleteResult("E7 requires exactly two confirmation rows")
            good = True
            productive_good = True
            for entry in entries:
                delta = _number(entry.get("candidate_score"), "candidate score") - _number(
                    entry.get("b8_score"), "B8 score"
                )
                if (
                    type(entry.get("candidate_survival")) is not bool
                    or type(entry.get("b8_survival")) is not bool
                ):
                    raise IncompleteResult("E7 survival must be explicit booleans")
                fallback = entry.get("fallback_count")
                if type(fallback) is not int or fallback < 0:
                    raise IncompleteResult("E7 fallback count must be a nonnegative integer")
                survival = entry["candidate_survival"] >= entry["b8_survival"]
                good = good and delta >= 20.0 and survival
                # This is deliberately stricter than the frozen E7 historical
                # display: an all-die delay cannot enter an archive admission.
                if entry["b8_survival"]:
                    productive_good = False
                else:
                    death = entry.get("b8_death_frame")
                    if type(death) is not int or not 0 <= death <= 5000:
                        raise IncompleteResult("E7 productive confirmation requires B8 death frame")
                    last_value = _number(
                        entry.get("b8_last_living_length"), "E7 B8 last living length"
                    )
                    tail_value = _number(
                        entry.get("candidate_tail_mean_length"), "E7 candidate tail mean length"
                    )
                    productive_good = (
                        productive_good
                        and entry["candidate_survival"]
                        and tail_value >= 0.9 * last_value
                    )
                fallback_total += fallback
            confirmed[source][ancestor] = good
            productive[source][ancestor] = good and productive_good
    per_ancestor = {
        str(a): sum(values[a] for values in confirmed.values()) / len(expected)
        for a in (100, 150, 200)
    }
    union = sum(any(values.values()) for values in confirmed.values()) / len(expected)
    productive_by_ancestor = {
        str(a): _one_sided_wilson(sum(values[a] for values in productive.values()), len(expected))
        for a in (100, 150, 200)
    }
    productive_union = _one_sided_wilson(
        sum(any(values.values()) for values in productive.values()), len(expected)
    )
    return {
        "sources": len(expected),
        "confirmed_fraction": union,
        "by_ancestor": per_ancestor,
        "confirmed_wilson_lb90": _one_sided_wilson(
            sum(any(values.values()) for values in confirmed.values()), len(expected)
        ),
        "productive_by_ancestor": productive_by_ancestor,
        "productive_wilson_lb90": productive_union,
        # E10 supplies q, C, and tau.  This hard false prevents the old point
        # rule from admitting a delay-only prefix before that bridge is scored.
        "productive_transfer_admissible": False,
        "fallback_count": fallback_total,
        "decision": "TRANSFER_ADMISSIBLE" if union >= 0.20 else "FRONTIER_SCARCE",
    }
