"""E10's population bridge, productive-confirmation, and finite-menu readouts.

This is a reporting-only module.  It intentionally refuses to turn an
unmeasured eligible source into a zero-valued envelope observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix.analysis import MIXES, IncompleteResult, _number, e6_summary, e7_summary
from src.scripts.eval_stats import student_t_isf

HORIZON = 5000
# Reset length is 1 and the simulator can add at most one food segment per
# frame, so mean post-step mass is sum(1+t, t=1..5000)/5000 = 2501.5.
PHYSICAL_SCORE_CEILING = 2501.5
_Z90 = 1.2815515655446004


def _one_sided_wilson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise IncompleteResult("invalid Wilson count")
    p = successes / total
    den = 1.0 + _Z90**2 / total
    center = (p + _Z90**2 / (2 * total)) / den
    radius = _Z90 * math.sqrt(p * (1 - p) / total + _Z90**2 / (4 * total**2)) / den
    return {
        "count": successes,
        "total": total,
        "point": p,
        "lb90": max(0.0, center - radius),
        "ub90": min(1.0, center + radius),
    }


def _anytime_hoeffding(successes: int, total: int) -> dict[str, float | int]:
    """A sensitivity interval valid under quota stopping by a union bound."""
    if total <= 0:
        raise IncompleteResult("invalid anytime count")
    alpha_n = 0.1 / (total * (total + 1))
    radius = math.sqrt(math.log(1.0 / alpha_n) / (2.0 * total))
    point = successes / total
    return {
        "count": successes,
        "total": total,
        "point": point,
        "alpha_n": alpha_n,
        "lb90": max(0.0, point - radius),
        "ub90": min(1.0, point + radius),
    }


def _histogram(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    bins: dict[str, int] = {
        f"{frame_bin} x {length_bin}": 0
        for frame_bin in ("2000-2999", "3000-3999", "4000-4999")
        for length_bin in ("700-999", "1000-1199", ">=1200")
    }
    bins.update({"outside_bin": 0, "survivor": 0})
    for row in rows:
        if bool(row.get("survived_h5000")):
            bins["survivor"] += 1
            continue
        frame, length = row.get("death_frame"), row.get("last_living_length")
        if type(frame) is not int or type(length) is not int:
            raise IncompleteResult("death ledger requires integer frame and length")
        fbin = next(
            (
                x
                for x in ("2000-2999", "3000-3999", "4000-4999")
                if int(x[:4]) <= frame <= int(x[5:])
            ),
            None,
        )
        lbin = (
            "700-999"
            if 700 <= length <= 999
            else "1000-1199" if 1000 <= length <= 1199 else ">=1200" if length >= 1200 else None
        )
        bins[f"{fbin} x {lbin}" if fbin and lbin else "outside_bin"] += 1
    return dict(sorted(bins.items()))


def _training_by_source(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    found: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        source = str(row.get("source_world", ""))
        ancestor, index = row.get("ancestor"), row.get("training_index")
        if (
            not source
            or type(ancestor) is not int
            or ancestor not in (100, 150, 200)
            or type(index) is not int
            or not 0 <= index < 8
        ):
            raise IncompleteResult("malformed E7 training outcome identity")
        key = source, ancestor, index
        if key in seen:
            raise IncompleteResult("duplicate E7 training outcome")
        seen.add(key)
        delta = _number(row.get("paired_delta"), "E7 paired delta")
        if abs(delta) > PHYSICAL_SCORE_CEILING:
            raise IncompleteResult("E7 paired delta exceeds the physical score ceiling")
        found[source].append(row)
    return found


def _two_phase_upper(
    parts: Mapping[str, list[float]], q_rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Addendum's q*mu Cauchy delta-method bound for quota-selected sources."""
    terms, means, theta = [], {}, 0.0
    for mix in MIXES:
        values, N = parts[mix], len(q_rows[mix])
        k, e = len(values), sum(bool(row["eligible"]) for row in q_rows[mix])
        if k < 2 or N < 2:
            raise IncompleteResult("two-phase U needs two screened and selected sources per mix")
        q, mu = e / N, math.fsum(values) / k
        vq = q * (1 - q) / (N - 1)
        vmu = math.fsum((x - mu) ** 2 for x in values) / (k - 1) / k
        vm = (abs(mu) * math.sqrt(vq) + q * math.sqrt(vmu)) ** 2
        means[mix] = q * mu
        theta += means[mix] / 3
        terms.append((vm / 9, min(N - 1, k - 1)))
    variance = math.fsum(term for term, _df in terms)
    if variance == 0.0:
        return {
            "mean": theta,
            "se": 0.0,
            "df": None,
            "lb90": theta,
            "ub90": theta,
            "mix_means": means,
        }
    df = variance**2 / math.fsum(term**2 / degree for term, degree in terms)
    se = math.sqrt(variance)
    return {
        "mean": theta,
        "se": se,
        "df": df,
        "lb90": theta - student_t_isf(0.1, df) * se,
        "ub90": min(PHYSICAL_SCORE_CEILING, theta + student_t_isf(0.1, df) * se),
        "mix_means": means,
    }


def _population_sensitivity(
    parts: Mapping[str, list[float]], by_mix: Mapping[str, list[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Joint bounded-mean CS sensitivity and compatible finite-ledger range."""
    joint_upper, q_lower, q_upper, finite = [], [], [], {}
    for mix in MIXES:
        N, k = len(by_mix[mix]), len(parts[mix])
        e = sum(row["eligible"] for row in by_mix[mix])
        q, mu = e / N, math.fsum(parts[mix]) / k
        # Six one-sided bounds, each alpha=.1/6; q bounds spend over all N.
        q_radius = math.sqrt(math.log(60 * N * (N + 1)) / (2 * N))
        mu_radius = PHYSICAL_SCORE_CEILING * math.sqrt(math.log(60) / (2 * k))
        joint_upper.append(min(1.0, q + q_radius) * min(PHYSICAL_SCORE_CEILING, mu + mu_radius))
        q_lower.append(max(0.0, q - q_radius) * mu)
        q_upper.append(min(1.0, q + q_radius) * mu)
        finite[mix] = {
            "lower": math.fsum(parts[mix]) / N,
            "upper": (math.fsum(parts[mix]) + (e - k) * PHYSICAL_SCORE_CEILING) / N,
            "unmeasured_eligible": e - k,
        }
    return {
        "joint_stopping_aware_ub90": math.fsum(joint_upper) / 3,
        "prevalence_only_conditional_on_measured_mu": {
            "lower": math.fsum(q_lower) / 3,
            "upper": math.fsum(q_upper) / 3,
        },
        "finite_ledger_compatible_range": finite,
        "label_uses": "working Student-t bound; this sensitivity does not replace it",
        "assumption": (
            "first-quota selection independent of prefix outcomes; "
            "eligible sources exchangeable within mix"
        ),
    }


def _productive(confirmations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in confirmations:
        source, ancestor, stream = (
            str(row.get("source_world", "")),
            row.get("ancestor"),
            row.get("stream"),
        )
        if not source or ancestor not in (100, 150, 200) or stream not in (0, 1):
            raise IncompleteResult("malformed E7 confirmation identity")
        if stream in groups[source, ancestor]:
            raise IncompleteResult("duplicate E7 confirmation")
        groups[source, ancestor][stream] = row
    result: dict[str, dict[int, tuple[bool, float]]] = defaultdict(dict)
    for key, streams in groups.items():
        if set(streams) != {0, 1}:
            raise IncompleteResult("productive E7 requires both confirmation streams")
        tail_values: list[float] = []
        good = True
        for row in streams.values():
            if (
                type(row.get("b8_survival")) is not bool
                or type(row.get("candidate_survival")) is not bool
            ):
                raise IncompleteResult("productive confirmation requires explicit survival")
            if row["b8_survival"] or not row["candidate_survival"]:
                good = False
                continue
            death, last, tail = (
                row.get("b8_death_frame"),
                row.get("b8_last_living_length"),
                row.get("candidate_tail_mean_length"),
            )
            if type(death) is not int or not 0 <= death <= HORIZON:
                raise IncompleteResult("productive confirmation requires B8 death frame")
            last_value, tail_value = _number(last, "B8 last length"), _number(
                tail, "candidate tail mean"
            )
            delta = _number(row.get("candidate_score"), "candidate score") - _number(
                row.get("b8_score"), "B8 score"
            )
            good = (
                good
                and delta >= 20.0
                and row["candidate_survival"]
                and tail_value >= 0.9 * last_value
            )
            tail_values.append(tail_value * (HORIZON - death + 1) / HORIZON)
        result[key[0]][key[1]] = (good, min(tail_values) if tail_values else 0.0)
    return result


def e10_summary(
    collection_ledger: Iterable[Mapping[str, Any]],
    early_by_arm: Mapping[str, Mapping[str, float]],
    training_outcomes: Iterable[Mapping[str, Any]],
    confirmations: Iterable[Mapping[str, Any]],
    *,
    collateral: float | None = None,
    physical_score_ceiling: float | None = PHYSICAL_SCORE_CEILING,
) -> dict[str, Any]:
    """Produce all prospectively registered E10 readouts from immutable exports."""
    ledger = list(collection_ledger)
    confirmations = list(confirmations)
    if not ledger:
        raise IncompleteResult("E10 requires a nonempty collection ledger")
    by_mix: dict[str, list[Mapping[str, Any]]] = {mix: [] for mix in MIXES}
    identities: set[str] = set()
    for row in ledger:
        source, mix = str(row.get("source_world", "")), str(row.get("mix", ""))
        if (
            not source
            or source in identities
            or mix not in by_mix
            or type(row.get("eligible")) is not bool
            or type(row.get("selected")) is not bool
        ):
            raise IncompleteResult("malformed collection ledger")
        if row.get("sealer") not in {"hero", "opponent", "none", "unresolved"}:
            raise IncompleteResult("unknown sealer classification")
        identities.add(source)
        by_mix[mix].append(row)
    training = _training_by_source(training_outcomes)
    productive = _productive(confirmations)
    selected_ids = {str(row["source_world"]) for row in ledger if row["selected"]}
    if any(row["selected"] and not row["eligible"] for row in ledger):
        raise IncompleteResult("selected source must be eligible")
    if set(training) - selected_ids or set(productive) - selected_ids:
        raise IncompleteResult("E10 outcomes name an unselected or unknown source")
    q, histograms, sealers, coverage = {}, {}, {}, {}
    envelope_parts: dict[str, list[float]] = {mix: [] for mix in MIXES}
    complete = True
    for mix, rows in by_mix.items():
        eligible = [row for row in rows if row["eligible"]]
        selected = [row for row in eligible if row["selected"]]
        all24 = []
        confirmation_covered = []
        for row in selected:
            outcomes = training.get(str(row["source_world"]), [])
            complete_outcome = len(outcomes) == 24
            if complete_outcome:
                all24.append(row)
            if str(row["source_world"]) in productive:
                confirmation_covered.append(row)
            if complete_outcome:
                envelope_parts[mix].append(
                    max(0.0, max(_number(x["paired_delta"], "paired delta") for x in outcomes))
                )
        selected_missing = len(selected) - sum(row in all24 for row in selected)
        missing = selected_missing
        complete = complete and missing == 0
        q[mix] = {
            "wilson": _one_sided_wilson(len(eligible), len(rows)),
            "stopping_aware": _anytime_hoeffding(len(eligible), len(rows)),
        }
        histograms[mix] = _histogram(rows)
        sealer_counts = {
            label: sum(str(row.get("sealer")) == label for row in rows)
            for label in ("hero", "opponent", "none", "unresolved")
        }
        sealers[mix] = {
            "counts": sealer_counts,
            "resolved": len(rows) - sealer_counts["unresolved"],
            "total": len(rows),
        }
        coverage[mix] = {
            "screened": len(rows),
            "ineligible": len(rows) - len(eligible),
            "eligible": len(eligible),
            "selected": len(selected),
            "eligible_unmeasured_by_design": len(eligible) - len(selected),
            "all_24_outcome_covered": len(all24),
            "confirmation_covered": sum(
                len(productive.get(str(row["source_world"]), {})) == 3 for row in selected
            ),
            "eligible_missing": missing,
        }
    bridge: dict[str, Any] = {}
    for arm, means in early_by_arm.items():
        if set(means) != set(MIXES):
            raise IncompleteResult("E10 bridge needs mean early delta for every mix")
        values = {
            mix: q[mix]["wilson"]["point"] * _number(means[mix], "early mean") for mix in MIXES
        }
        bridge[arm] = {"by_mix": values, "equal_mix_mean": math.fsum(values.values()) / len(MIXES)}
    selected_sources = {
        str(row["source_world"]) for rows in by_mix.values() for row in rows if row["selected"]
    }
    confirmation_counts: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in confirmations:
        source, ancestor, stream = (
            str(row.get("source_world", "")),
            row.get("ancestor"),
            row.get("stream"),
        )
        if source in selected_sources and ancestor in (100, 150, 200) and stream in (0, 1):
            confirmation_counts[source].add((ancestor, stream))
    confirmations_complete = all(
        len(confirmation_counts[source]) == 6 for source in selected_sources
    )
    envelope: dict[str, Any]
    if complete and confirmations_complete:
        statistic = _two_phase_upper(envelope_parts, by_mix)
        # E10's population estimate is q_m times the selected eligible mean;
        # unselected eligible worlds remain visibly unmeasured by design.
        confirmed_parts: dict[str, list[float]] = {mix: [] for mix in MIXES}
        confirmed_by_source: dict[str, dict[int, dict[int, Mapping[str, Any]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for row in confirmations:
            confirmed_by_source[str(row.get("source_world"))][int(row.get("ancestor", -1))][
                int(row.get("stream", -1))
            ] = row
        for mix, rows in by_mix.items():
            for row in rows:
                if not row["selected"]:
                    continue
                choices = []
                for streams in confirmed_by_source.get(str(row["source_world"]), {}).values():
                    if set(streams) != {0, 1}:
                        continue
                    deltas = [
                        _number(x.get("candidate_score"), "candidate score")
                        - _number(x.get("b8_score"), "B8 score")
                        for x in streams.values()
                    ]
                    frozen = all(
                        delta >= 20.0 and x.get("candidate_survival") >= x.get("b8_survival")
                        for delta, x in zip(deltas, streams.values())
                    )
                    if frozen:
                        choices.append(min(deltas))
                confirmed_parts[mix].append(max(0.0, max(choices, default=0.0)))
        envelope = {
            "status": "COMPLETE",
            "statistic": statistic,
            "confirmed_only": _two_phase_upper(confirmed_parts, by_mix),
            "sensitivity": _population_sensitivity(envelope_parts, by_mix),
            "decision": "FINITE_MENU_INSUFFICIENT" if statistic["ub90"] < 40.0 else "INCONCLUSIVE",
        }
    else:
        # The storage capacity is not a logical-length bound.  This ceiling is
        # derived from the one-food-increment-per-frame H5000 simulator bound.
        if physical_score_ceiling is None:
            missing_max = None
        else:
            ceiling = _number(physical_score_ceiling, "physical score ceiling")
            if ceiling < 0.0:
                raise IncompleteResult("physical score ceiling must be nonnegative")
            missing_max = {
                mix: coverage[mix]["eligible_missing"] * ceiling / max(1, coverage[mix]["screened"])
                for mix in MIXES
            }
        envelope = {
            "status": "INCOMPLETE",
            "decision": "INCOMPLETE",
            "missing_eligible_upper_sensitivity": missing_max,
            "reason": (
                "selected training outcomes or confirmations are missing; "
                "missing observations are not imputed as zero"
            ),
        }
    productive_rows: dict[str, Any] = {}
    if collateral is not None:
        collateral = _number(collateral, "natural-world collateral")
        if collateral < 0:
            raise IncompleteResult("natural-world collateral must be nonnegative")
    C_values = (0.0, 5.9, 10.0) if collateral is None else (collateral,)
    for mix, rows in by_mix.items():
        sources = [str(r["source_world"]) for r in rows if r["selected"]]
        candidates = []
        for source in sources:
            successful = [tail for ok, tail in productive.get(source, {}).values() if ok]
            if successful:
                candidates.append(max(successful))
        rate = _one_sided_wilson(len(candidates), len(sources)) if sources else None
        q_point = q[mix]["wilson"]["point"]
        gross = q_point * math.fsum(candidates) / len(sources) if sources else 0.0
        productive_rows[mix] = {
            "productive_sources": len(candidates),
            "selected_sources": len(sources),
            "natural_productive_fraction": (
                q_point * len(candidates) / len(sources) if sources else 0.0
            ),
            "productive_wilson": rate,
            "gross_supply": gross,
            "tau_required": {str(C): None if gross == 0 else (40.0 + C) / gross for C in C_values},
            "tau_feasible": {str(C): gross > 0 and (40 + C) / gross <= 1 for C in C_values},
        }
    productive_strata: dict[str, dict[str, dict[str, int]]] = {mix: {} for mix in MIXES}
    productive_ancestor: dict[str, dict[str, dict[str, int]]] = {mix: {} for mix in MIXES}
    for mix in MIXES:
        for sealer in ("hero", "opponent", "none", "unresolved"):
            sources = [
                str(r["source_world"])
                for r in by_mix[mix]
                if r["selected"] and r["sealer"] == sealer
            ]
            count = sum(any(ok for ok, _tail in productive.get(s, {}).values()) for s in sources)
            productive_strata[mix][sealer] = {
                "selected": len(sources),
                "productive": count,
                "wilson": _one_sided_wilson(count, len(sources)) if sources else None,
            }
        for ancestor in (100, 150, 200):
            sources = [str(r["source_world"]) for r in by_mix[mix] if r["selected"]]
            count = sum(productive.get(s, {}).get(ancestor, (False, 0.0))[0] for s in sources)
            productive_ancestor[mix][str(ancestor)] = {
                "selected": len(sources),
                "productive": count,
                "wilson": _one_sided_wilson(count, len(sources)) if sources else None,
            }
    pooled_gross = math.fsum(productive_rows[mix]["gross_supply"] for mix in MIXES) / len(MIXES)
    tau = {str(C): None if pooled_gross == 0 else (40.0 + C) / pooled_gross for C in C_values}
    historical = e7_summary(confirmations, sorted(selected_ids)) if confirmations_complete else None
    admission_by_C = {
        str(C): bool(
            envelope["status"] == "COMPLETE"
            and historical
            and historical["decision"] == "TRANSFER_ADMISSIBLE"
            and all(productive_rows[mix]["natural_productive_fraction"] >= 0.163 for mix in MIXES)
            and tau[str(C)] is not None
            and tau[str(C)] <= 1.0
            and envelope["decision"] != "FINITE_MENU_INSUFFICIENT"
        )
        for C in C_values
    }
    # Planning collateral scenarios are necessary-condition displays only.
    admission = collateral is not None and admission_by_C[str(float(collateral))]

    return {
        "schema": "bl-prefix-e10-results/v1",
        "q_by_mix": q,
        "death_histogram": histograms,
        "sealer_strata": sealers,
        "coverage": coverage,
        "bridge": bridge,
        "envelope": envelope,
        "productive_confirmations": productive_rows,
        "productive_by_ancestor": productive_ancestor,
        "productive_by_sealer": productive_strata,
        "productive_equal_mix_gross_supply": pooled_gross,
        "tau_required_equal_mix": tau,
        "required_rescue_fraction": 0.163,
        "historical_e7": historical,
        "archive_transfer_admissible": admission,
        "archive_transfer_admissible_by_C": admission_by_C,
        "collateral_evidence": (
            "natural-world measured input"
            if collateral is not None
            else "planning sensitivity only; no measured natural collateral"
        ),
    }


def _caution_summary(
    bank: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Describe both continuation modes and the exact ledger by death stratum."""
    sources = {str(row["source_world"]): row for row in bank["collection_ledger"]}
    death_for_root = {
        str(row["source_world"]): str(row.get("matched_death_world", row["source_world"]))
        for row in bank["roots"]
    }

    def identity(row: Mapping[str, Any]) -> tuple:
        return tuple(row[key] for key in ("stratum", "mix", "source_world", "stream", "phase"))

    baselines = {
        identity(row): _number(row["score"], "B8 score")
        for row in rows
        if row["controller"] == "B8"
    }
    groups: dict[tuple, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    fields = (
        "both_alive_delta",
        "candidate_only_mass",
        "b8_only_mass",
        "common_alive_prefix_growth",
    )
    for row in rows:
        if row["controller"] == "B8":
            continue
        ledger = row["ledger"]
        for field in fields:
            _number(ledger[field], field)
        delta = ledger["both_alive_delta"] + ledger["candidate_only_mass"] - ledger["b8_only_mass"]
        if not math.isclose(delta, row["score"] - baselines[identity(row)], abs_tol=1e-8):
            raise IncompleteResult("E6 ledger does not reconcile to paired score")
        death = sources[death_for_root[str(row["source_world"])]]
        frame, length = death["death_frame"], death["last_living_length"]
        fbin = f"{(frame//1000)*1000}-{(frame//1000)*1000+999}"
        lbin = "700-999" if length < 1000 else "1000-1199" if length < 1200 else ">=1200"
        key = (row["controller"], row["phase"], row["stratum"], row["mix"], fbin, lbin)
        groups[key][str(row["source_world"])].append(ledger)
    result = []
    for key, worlds in sorted(groups.items()):
        if any(len(entries) != 2 for entries in worlds.values()):
            raise IncompleteResult("E6 caution ledger requires two streams per world")
        averages = {
            field: math.fsum(
                math.fsum(entry[field] for entry in entries) / 2 for entries in worlds.values()
            )
            / len(worlds)
            for field in fields
        }
        result.append(
            dict(
                zip(
                    (
                        "controller",
                        "phase",
                        "stratum",
                        "mix",
                        "b8_death_frame_bin",
                        "b8_last_length_bin",
                    ),
                    key,
                ),
                worlds=len(worlds),
                **averages,
            )
        )
    return result


def _load_complete(path: Path, schema: str) -> Mapping[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != schema
        or value.get("preflight") is True
        or value.get("scientific") is not True
        or value.get("status") not in {None, "COMPLETE"}
    ):
        raise IncompleteResult(f"E10 refuses non-scientific or incomplete {path.name}")
    return value


def run(bank_path: Path, e6_path: Path, e7_path: Path, out: Path) -> dict[str, Any]:
    """Read only verified E6/E7 exports and publish E10's immutable result."""
    try:
        bank = _load_complete(bank_path, "bl-prefix-bank/v1")
        e6 = _load_complete(e6_path, "bl-prefix-e6-results/v1")
        e7 = _load_complete(e7_path, "bl-prefix-e7-results/v1")
        bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        for value in (e6, e7):
            if (
                value.get("bank_sha256") != bank_sha
                or value.get("audit_sha256") != bank.get("audit_sha256")
                or value.get("b8_sha256") != bank.get("b8", {}).get("sha256")
            ):
                raise IncompleteResult(
                    "E10 input provenance does not bind the shared bank, audit, and B8"
                )
        expected_early = [
            str(row["source_world"]) for row in bank["roots"] if row["stratum"] == "early"
        ]
        expected_ordinary = [
            str(row["source_world"]) for row in bank["roots"] if row["stratum"] == "ordinary"
        ]
        if len(expected_early) != 128 or len(expected_ordinary) != 128:
            raise IncompleteResult("scientific E10 requires the exact 128 early/ordinary bank")
        early = {
            arm: e6_summary(
                e6["rows"], arm, expected_early=expected_early, expected_ordinary=expected_ordinary
            )["early"]["mix_means"]
            for arm in ("R", "F")
        }
        result = e10_summary(
            bank["collection_ledger"], early, e7["training_outcomes"], e7["confirmations"]
        )
        result["e6_caution_ledger_by_death_stratum"] = _caution_summary(bank, e6["rows"])
        result.update(
            {
                "preflight": False,
                "scientific": True,
                "status": result["envelope"]["status"],
                "inputs": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (bank_path, e6_path, e7_path)
                },
            }
        )
    except (KeyError, OSError, ValueError, IncompleteResult) as exc:
        result = {
            "schema": "bl-prefix-e10-results/v1",
            "preflight": False,
            "scientific": False,
            "status": "INCOMPLETE",
            "reason": str(exc),
        }
    export_json(out / "e10-results.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--e6", type=Path, required=True)
    parser.add_argument("--e7", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bank, args.e6, args.e7, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
