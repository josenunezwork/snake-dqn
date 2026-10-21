"""Fail-closed analysis for immutable L2 dense-ceiling shards."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.bl_served_search.ceiling import (
    ARM,
    B8_SHA256,
    HORIZON,
    MIX_COUNTS,
    ROW_SCHEMA,
    SCHEMA,
    digest_file,
    digest_value,
    validate_bank,
)
from research.bl_served_search.chunks import (
    CHUNK_COUNT,
    chunk_rows,
    validate_chunk_rows,
    world_ids_sha256,
)
from research.bl_served_search.exports import export_json, verify_export
from src.scripts.eval_stats import student_t_isf


def _finite(value: Any, *, boolean: bool = False) -> float:
    if isinstance(value, bool):
        if boolean:
            return float(value)
        raise ValueError("boolean is not a numeric metric")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("nonfinite metric")
    return float(value)


def one_sided_90(values: Iterable[float]) -> dict[str, float | int]:
    """World-clustered Student-t point estimate and one-sided 90% bounds."""
    values = [_finite(value) for value in values]
    n = len(values)
    if n < 2:
        raise ValueError("a clustered statistic needs at least two paired worlds")
    estimate = sum(values) / n
    sample_variance = sum((value - estimate) ** 2 for value in values) / (n - 1)
    se = math.sqrt(sample_variance / n)
    critical = student_t_isf(0.10, n - 1) if se else 0.0
    return {
        "n": n,
        "df": n - 1,
        "estimate": estimate,
        "sample_variance": sample_variance,
        "standard_error": se,
        "critical_one_sided_90": critical,
        "lower90": estimate - critical * se,
        "upper90": estimate + critical * se,
    }


def pooled_equivalent(per_mix: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    """Equal-mix estimate with the preregistered Welch-Satterthwaite variance."""
    terms = [float(per_mix[mix]["standard_error"]) ** 2 for mix in MIX_COUNTS]
    estimate = sum(float(per_mix[mix]["estimate"]) for mix in MIX_COUNTS) / len(MIX_COUNTS)
    variance = sum(terms) / len(MIX_COUNTS) ** 2
    if variance == 0:
        df = math.inf
        critical = 0.0
    else:
        denominator = sum(
            (term / len(MIX_COUNTS) ** 2) ** 2 / float(per_mix[mix]["df"])
            for term, mix in zip(terms, MIX_COUNTS)
            if term > 0
        )
        df = variance**2 / denominator if denominator else math.inf
        critical = student_t_isf(0.10, df) if math.isfinite(df) else 1.2815515655446004
    se = math.sqrt(variance)
    return {
        "estimate": estimate,
        "standard_error": se,
        "df": df if math.isfinite(df) else None,
        "lower90": estimate - critical * se,
        "upper90": estimate + critical * se,
    }


def transfer_status(per_mix: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    """Report the prospective L4 transfer budget clause without relabelling L2.

    This is intentionally a placement/funding status.  It cannot change the
    scientific L2 decision, and L2 exports mark it non-authoritative because
    the clause applies to a future paced L4 study.
    """
    missing = set(MIX_COUNTS) - set(per_mix)
    if missing:
        raise ValueError(f"missing transfer mixes: {sorted(missing)}")
    points = {mix: _finite(per_mix[mix].get("estimate")) for mix in MIX_COUNTS}
    upper90 = {mix: _finite(per_mix[mix].get("upper90")) for mix in MIX_COUNTS}
    passed = (
        points["frozen"] > 0.0
        and points["mixed"] > 0.0
        and all(value >= 0.0 for value in upper90.values())
    )
    return {
        "label": "TRANSFER_READY" if passed else "TRANSFER_UNRESOLVED",
        "frozen_point_positive": points["frozen"] > 0.0,
        "mixed_point_positive": points["mixed"] > 0.0,
        "no_per_mix_upper90_below_zero": all(value >= 0.0 for value in upper90.values()),
        "points": points,
        "upper90": upper90,
    }


def _row_error(row: dict[str, Any], expected: dict[str, Any]) -> str | None:
    if row.get("schema") != ROW_SCHEMA:
        return "invalid row schema"
    for key in ("world_id", "world_seed", "mix", "horizon"):
        if row.get(key) != expected.get(key):
            return f"row {key} disagrees with bank"
    if row.get("plan") != expected.get("plan"):
        # Runtime receipts use resolved checkpoint paths so compare the stable
        # portable plan payload after replacing only its specifications.
        plan = row.get("plan")
        if not isinstance(plan, dict) or {
            key: value for key, value in plan.items() if key != "specs"
        } != {key: value for key, value in expected["plan"].items() if key != "specs"}:
            return "row roster plan disagrees with bank"
    for arm in ("baseline", "dense"):
        value = row.get(arm)
        if not isinstance(value, dict):
            return f"missing {arm} result"
        for metric in ("mass_integral", "alive_frame_fraction", "no_death_episode"):
            try:
                metric_value = _finite(value.get(metric), boolean=metric == "no_death_episode")
            except ValueError:
                return f"invalid {arm}.{metric}"
            if (
                metric in ("alive_frame_fraction", "no_death_episode")
                and not 0 <= metric_value <= 1
            ):
                return f"invalid {arm}.{metric} range"
        if int(value.get("decisions", 0)) not in range(1, HORIZON + 1):
            return f"invalid {arm} decision count"
    paired = row.get("paired_delta")
    if not isinstance(paired, dict):
        return "missing paired delta"
    for metric in ("mass_integral", "alive_frame_fraction", "no_death_episode"):
        observed = _finite(paired.get(metric))
        expected_delta = _finite(
            row["dense"][metric], boolean=metric == "no_death_episode"
        ) - _finite(row["baseline"][metric], boolean=metric == "no_death_episode")
        if not math.isclose(observed, expected_delta, abs_tol=1e-12, rel_tol=0.0):
            return f"paired delta mismatch: {metric}"
    ledger = row.get("mass_ledger")
    if not isinstance(ledger, dict):
        return "missing mass ledger"
    names = (
        "common_alive_productivity_delta",
        "dense_only_survival_mass",
        "baseline_only_survival_mass",
    )
    try:
        ledger_total = sum(_finite(ledger.get(name)) for name in names)
    except ValueError:
        return "invalid mass ledger"
    if not math.isclose(ledger_total, _finite(ledger.get("total")), abs_tol=1e-12):
        return "mass ledger does not sum"
    if not math.isclose(ledger_total, _finite(paired["mass_integral"]), abs_tol=1e-12):
        return "mass ledger does not reconcile"
    trajectories = row.get("trajectories")
    if not isinstance(trajectories, dict):
        return "missing trajectories"
    baseline_trace, dense_trace = trajectories.get("baseline_mass"), trajectories.get("dense_mass")
    if not isinstance(baseline_trace, list) or not isinstance(dense_trace, list):
        return "malformed trajectories"
    if len(baseline_trace) != HORIZON or len(dense_trace) != HORIZON:
        return "incomplete trajectories"
    try:
        for trace in (baseline_trace, dense_trace):
            if any(type(value) is not int or value < 0 for value in trace):
                return "invalid trajectory mass"
            first_dead = next((index for index, value in enumerate(trace) if value == 0), HORIZON)
            if any(value != 0 for value in trace[first_dead:]):
                return "nonterminal trajectory death"
        baseline_mass = sum(baseline_trace) / HORIZON
        dense_mass = sum(dense_trace) / HORIZON
        trace_delta = dense_mass - baseline_mass
    except ValueError:
        return "invalid trajectory mass"
    if not math.isclose(baseline_mass, _finite(row["baseline"]["mass_integral"]), abs_tol=1e-12):
        return "baseline trajectory does not reconcile"
    if not math.isclose(dense_mass, _finite(row["dense"]["mass_integral"]), abs_tol=1e-12):
        return "dense trajectory does not reconcile"
    for arm, trace in (("baseline", baseline_trace), ("dense", dense_trace)):
        trace_alive_fraction = sum(value > 0 for value in trace) / HORIZON
        trace_no_death = not any(value == 0 for value in trace)
        try:
            reported_alive_fraction = _finite(row[arm]["alive_frame_fraction"])
            reported_no_death = _finite(row[arm]["no_death_episode"], boolean=True)
        except ValueError:
            return f"invalid {arm} survival endpoint"
        if not math.isclose(trace_alive_fraction, reported_alive_fraction, abs_tol=1e-12):
            return f"{arm} alive fraction does not reconcile to trajectory"
        if trace_no_death != bool(reported_no_death):
            return f"{arm} no-death endpoint does not reconcile to trajectory"
    if not math.isclose(trace_delta, _finite(paired["mass_integral"]), abs_tol=1e-12):
        return "trajectory mass does not reconcile"
    trace_components = {
        "common_alive_productivity_delta": sum(
            dense - baseline
            for baseline, dense in zip(baseline_trace, dense_trace)
            if baseline and dense
        )
        / HORIZON,
        "dense_only_survival_mass": sum(
            dense for baseline, dense in zip(baseline_trace, dense_trace) if not baseline and dense
        )
        / HORIZON,
        "baseline_only_survival_mass": -sum(
            baseline
            for baseline, dense in zip(baseline_trace, dense_trace)
            if baseline and not dense
        )
        / HORIZON,
    }
    if any(
        not math.isclose(_finite(ledger[name]), value, abs_tol=1e-12)
        for name, value in trace_components.items()
    ):
        return "mass ledger components do not reconcile to trajectories"
    world_ledger = row.get("world_ledger")
    if not isinstance(world_ledger, dict):
        return "missing world ledger"
    required = (
        "productive_rescue",
        "d300_only_escape",
        "re_death",
        "delay_only_changes",
        "newly_caused_death",
        "common_alive_growth_loss",
        "natural_world_denominator",
        "baseline_died",
        "b8_death_frame",
        "b8_death_length",
        "b8_death_cause",
        "baseline_sealer",
        "dense_sealer",
        "interventions",
    )
    if any(name not in world_ledger for name in required):
        return "incomplete world ledger"
    if world_ledger["natural_world_denominator"] != 1:
        return "wrong natural-world denominator"
    if world_ledger["baseline_sealer"] not in {"hero", "opponent", "none", "unresolved"}:
        return "invalid baseline sealer"
    if world_ledger["dense_sealer"] not in {"hero", "opponent", "none", "unresolved"}:
        return "invalid dense sealer"
    interventions = world_ledger["interventions"]
    if not isinstance(interventions, dict):
        return "missing interventions ledger"
    try:
        proposed, completed = (interventions[name] for name in ("proposed", "completed"))
        consumed = interventions["consumed_before_deadline"]
    except (KeyError, TypeError, ValueError):
        return "invalid interventions ledger"
    if (
        type(proposed) is not int
        or type(completed) is not int
        or min(proposed, completed) < 0
        or not completed <= proposed
    ):
        return "unreconciled interventions ledger"
    if consumed is not None and (type(consumed) is not int or consumed < 0 or consumed > completed):
        return "unreconciled consumed-before-deadline ledger"
    if (
        consumed is None
        and interventions.get("consumed_before_deadline_availability") != "UNAVAILABLE_UNPACED_L2"
    ):
        return "missing consumed-before-deadline availability"
    baseline_died = bool(world_ledger["baseline_died"])
    if type(world_ledger["baseline_died"]) is not bool:
        return "invalid baseline death indicator"
    observed_baseline_death = any(value == 0 for value in baseline_trace)
    if baseline_died != observed_baseline_death:
        return "baseline death ledger disagrees with trajectory"
    first_baseline_death = next(
        (index for index, value in enumerate(baseline_trace) if value == 0), None
    )
    if baseline_died:
        if (
            type(world_ledger["b8_death_frame"]) is not int
            or world_ledger["b8_death_frame"] != first_baseline_death
            or type(world_ledger["b8_death_length"]) is not int
            or world_ledger["b8_death_length"] <= 0
        ):
            return "B8 death frame/length ledger disagrees with trajectory"
        if (
            first_baseline_death > 0
            and world_ledger["b8_death_length"] != baseline_trace[first_baseline_death - 1]
        ):
            return "B8 death length does not match the last living baseline frame"
    elif any(
        world_ledger[name] is not None
        for name in ("b8_death_frame", "b8_death_length", "b8_death_cause")
    ):
        return "surviving B8 row has death ledger"
    dense_survived = not any(value == 0 for value in dense_trace)
    tail = dense_trace[first_baseline_death:] if first_baseline_death is not None else []
    expected_productive = bool(
        baseline_died
        and dense_survived
        and tail
        and sum(tail) / len(tail) >= 0.9 * int(world_ledger["b8_death_length"])
    )
    if type(world_ledger["productive_rescue"]) is not bool or (
        world_ledger["productive_rescue"] != expected_productive
    ):
        return "productive rescue ledger disagrees with trajectories"
    if bool(world_ledger["newly_caused_death"]) != (
        not observed_baseline_death and any(value == 0 for value in dense_trace)
    ):
        return "new-death ledger disagrees with trajectories"
    expected_growth_loss = (
        sum(
            max(0, base - dense)
            for base, dense in zip(baseline_trace, dense_trace)
            if base and dense
        )
        / HORIZON
    )
    try:
        growth_loss = _finite(world_ledger["common_alive_growth_loss"])
    except ValueError:
        return "invalid common-alive growth loss"
    if not math.isclose(growth_loss, expected_growth_loss, abs_tol=1e-12):
        return "common-alive growth loss does not reconcile"
    if int(world_ledger["delay_only_changes"]) != int(row["dense"].get("delay_only_changes", -1)):
        return "delay-only ledger disagrees with dense receipt"
    if proposed != int(row["dense"].get("proposed_interventions", -1)) or completed != int(
        row["dense"].get("completed_interventions", -1)
    ):
        return "interventions ledger disagrees with dense receipt"
    return None


def load_rows(bank_path: Path, shard_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read exports and require exactly one successful result per bank world."""
    bank = json.loads(bank_path.read_text())
    expected_rows = validate_bank(bank)
    expected = {str(row["world_id"]): row for row in expected_rows}
    errors, rows, seen, seen_chunks = [], [], set(), set()
    bank_sha = digest_file(bank_path)
    for path in shard_paths:
        try:
            verify_export(path)
            shard = json.loads(path.read_text())
            identity = shard.get("identity", {})
            if shard.get("schema") != SCHEMA or shard.get("status") != "completed":
                raise ValueError("bad shard status/schema")
            if (
                identity.get("bank_sha256") != bank_sha
                or not shard.get("clean_exit")
                or identity.get("arm") != ARM
                or identity.get("b8_sha256") != B8_SHA256
                or identity.get("horizon") != HORIZON
                or identity.get("source_manifest")
                != __import__(
                    "research.bl_served_search.ceiling", fromlist=["source_manifest"]
                ).source_manifest()
            ):
                raise ValueError("shard is not bound to this complete bank")
            chunk_index = identity.get("chunk_index")
            if type(chunk_index) is not int or identity.get("chunk_count") != CHUNK_COUNT:
                raise ValueError("missing immutable L2' chunk identity")
            if chunk_index in seen_chunks or not 0 <= chunk_index < CHUNK_COUNT:
                raise ValueError("duplicate or invalid chunk receipt")
            expected_chunk = chunk_rows(expected_rows, chunk_index)
            validate_chunk_rows(expected_chunk)
            if identity.get("chunk_world_ids_sha256") != world_ids_sha256(expected_chunk):
                raise ValueError("chunk world identity disagrees with bank")
            from research.bl_served_search.ceiling import validate_runtime_parity

            validate_runtime_parity(
                identity.get("runtime_parity"),
                bank_sha256=bank_sha,
                source=identity["source_manifest"],
                require_local_runtime=False,
            )
            shard_rows = shard.get("rows")
            if not isinstance(shard_rows, list) or shard.get("rows_sha256") != digest_value(
                shard_rows
            ):
                raise ValueError("row digest mismatch")
            if [str(row.get("world_id")) for row in shard_rows] != [
                str(row["world_id"]) for row in expected_chunk
            ]:
                raise ValueError("chunk rows are not the registered ordered chunk")
            for row in shard_rows:
                world_id = str(row.get("world_id"))
                if world_id in seen or world_id not in expected:
                    raise ValueError("duplicate or foreign world row")
                error = _row_error(row, expected[world_id])
                if error:
                    raise ValueError(f"{world_id}: {error}")
                seen.add(world_id)
                rows.append(row)
            seen_chunks.add(chunk_index)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing selection worlds: {len(missing)}")
    missing_chunks = sorted(set(range(CHUNK_COUNT)) - seen_chunks)
    if missing_chunks:
        errors.append(f"missing terminal L2' chunks: {missing_chunks}")
    return rows, errors


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the preregistered readout from complete, validated world pairs."""
    by_mix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mix[str(row["mix"])].append(row)
    metrics: dict[str, Any] = {}
    for metric in ("mass_integral", "alive_frame_fraction", "no_death_episode"):
        per_mix = {
            mix: one_sided_90(row["paired_delta"][metric] for row in by_mix[mix])
            for mix in MIX_COUNTS
        }
        metrics[metric] = {"per_mix": per_mix, "pooled_equivalent": pooled_equivalent(per_mix)}
    mass = metrics["mass_integral"]
    frozen_mix = mass["per_mix"]["frozen"]
    mixed_mix = mass["per_mix"]["mixed"]
    pooled = mass["pooled_equivalent"]
    prospective_l4_transfer = transfer_status(mass["per_mix"])
    if float(pooled["upper90"]) < 40.0:
        verdict = "PACKAGE_KILL"
    elif (
        float(frozen_mix["lower90"]) > 0.0
        and float(mixed_mix["lower90"]) > 0.0
        and float(pooled["estimate"]) >= 40.0
    ):
        verdict = "FUND_MAC_SYSTEMS"
    else:
        verdict = "INCONCLUSIVE"
    ledger = {}
    changes = {}
    world_ledger = {}
    for mix in MIX_COUNTS:
        values = by_mix[mix]
        ledger[mix] = {
            arm: sum(float(row[arm]["mass_integral"]) for row in values)
            for arm in ("baseline", "dense")
        }
        ledger[mix]["paired_delta_sum"] = sum(
            float(row["paired_delta"]["mass_integral"]) for row in values
        )
        for name in (
            "common_alive_productivity_delta",
            "dense_only_survival_mass",
            "baseline_only_survival_mass",
            "total",
        ):
            ledger[mix][name] = sum(float(row["mass_ledger"][name]) for row in values)
        decisions = sum(int(row["dense"]["decisions"]) for row in values)
        changes[mix] = {
            "decisions": decisions,
            "action_changes": sum(int(row["dense"]["action_changes"]) for row in values),
            "changes_per_1000_decisions": 1000
            * sum(int(row["dense"]["action_changes"]) for row in values)
            / max(1, decisions),
            "horizon_surviving_changes": sum(
                int(row["dense"]["horizon_surviving_changes"]) for row in values
            ),
            "delay_only_changes": sum(int(row["dense"]["delay_only_changes"]) for row in values),
            "residual_changes": sum(int(row["dense"]["residual_changes"]) for row in values),
        }
        sealers = {name: 0 for name in ("hero", "opponent", "none", "unresolved")}
        for row in values:
            sealers[str(row["world_ledger"]["baseline_sealer"])] += 1
        world_ledger[mix] = {
            "productive_rescues": sum(
                bool(row["world_ledger"]["productive_rescue"]) for row in values
            ),
            "d300_only_escapes": sum(
                bool(row["world_ledger"]["d300_only_escape"]) for row in values
            ),
            "re_deaths": sum(bool(row["world_ledger"]["re_death"]) for row in values),
            "delay_only_changes": sum(
                int(row["world_ledger"]["delay_only_changes"]) for row in values
            ),
            "newly_caused_deaths": sum(
                bool(row["world_ledger"]["newly_caused_death"]) for row in values
            ),
            "common_alive_growth_loss": sum(
                float(row["world_ledger"]["common_alive_growth_loss"]) for row in values
            ),
            "natural_world_denominator": sum(
                int(row["world_ledger"]["natural_world_denominator"]) for row in values
            ),
            "baseline_deaths": sum(bool(row["world_ledger"]["baseline_died"]) for row in values),
            "baseline_sealer_strata": sealers,
            "interventions": {
                "proposed": sum(
                    int(row["world_ledger"]["interventions"]["proposed"]) for row in values
                ),
                "completed": sum(
                    int(row["world_ledger"]["interventions"]["completed"]) for row in values
                ),
                "consumed_before_deadline": None,
                "consumed_before_deadline_availability": "UNAVAILABLE_UNPACED_L2",
            },
        }
    return {
        "status": "completed",
        "development_only": True,
        "selection_worlds": len(rows),
        "metrics": metrics,
        "mass_ledger": ledger,
        "changes": changes,
        "world_ledger": world_ledger,
        "prospective_l4_transfer": {
            **prospective_l4_transfer,
            "authoritative_for_l2": False,
            "scope": "future L4 budget status; does not alter L2 scientific label",
        },
        "decision": {"label": verdict, "frozen_kill_ub90": pooled["upper90"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(
        Path(str(args.out) + suffix).exists() for suffix in (".sha256", ".complete")
    ):
        parser.error("analysis output is create-only and already exists")
    rows, errors = load_rows(args.bank, args.shard)
    if errors:
        payload = {
            "schema": "bl-served-search-l2-analysis/v1",
            # Receipt failure, timeout, fault or an absent chunk means the
            # registered one-look bank never opened.  Emit no partial metrics.
            "status": "INCONCLUSIVE",
            "decision": {"label": "INCONCLUSIVE", "reason": "terminal chunk set incomplete"},
            "errors": errors,
        }
    else:
        payload = {"schema": "bl-served-search-l2-analysis/v1", **analyze(rows)}
    export_json(args.out, payload)


if __name__ == "__main__":
    main()
