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
    return None


def load_rows(bank_path: Path, shard_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read exports and require exactly one successful result per bank world."""
    bank = json.loads(bank_path.read_text())
    expected_rows = validate_bank(bank)
    expected = {str(row["world_id"]): row for row in expected_rows}
    errors, rows, seen = [], [], set()
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
            for row in shard_rows:
                world_id = str(row.get("world_id"))
                if world_id in seen or world_id not in expected:
                    raise ValueError("duplicate or foreign world row")
                error = _row_error(row, expected[world_id])
                if error:
                    raise ValueError(f"{world_id}: {error}")
                seen.add(world_id)
                rows.append(row)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing selection worlds: {len(missing)}")
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
    if float(pooled["upper90"]) < 40.0:
        verdict = "KILL_SERVED_SEARCH_PUSH"
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
    return {
        "status": "completed",
        "development_only": True,
        "selection_worlds": len(rows),
        "metrics": metrics,
        "mass_ledger": ledger,
        "changes": changes,
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
            "status": "INCOMPLETE",
            "errors": errors,
        }
    else:
        payload = {"schema": "bl-served-search-l2-analysis/v1", **analyze(rows)}
    export_json(args.out, payload)


if __name__ == "__main__":
    main()
