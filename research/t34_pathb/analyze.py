"""Create-only, fail-closed analysis of the frozen T34 development bank.

Worlds are paired observations. Independent bank blocks, with equal weighting of
three terminal mixes, are the uncertainty units. Frame rates are never pooled.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

from src.scripts.eval_stats import student_t_isf

REPO = Path(__file__).resolve().parents[2]
MIXES = ("scripted", "frozen", "mixed")
ARMS = ("bare", "async", "sync")
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


class InvalidBank(ValueError):
    """A missing or malformed input cannot support a development decision."""


def _number(
    value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBank(f"{name}: expected a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise InvalidBank(f"{name}: out of range or nonfinite")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    result = _number(value, name, minimum=minimum)
    if int(result) != result:
        raise InvalidBank(f"{name}: expected an integer")
    return int(result)


def _load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise InvalidBank(f"{path.name}: nonfinite JSON {value}")

    def unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise InvalidBank(f"{path.name}: duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys
    )
    if not isinstance(value, dict):
        raise InvalidBank(f"{path.name}: expected an object")
    return value


def _ci(values: Iterable[float], alpha: float = 0.10) -> dict[str, Any]:
    """Student-t bounds; .10 gives LB90, .025 a two-sided 95% interval."""
    xs = [_number(value, "CI observation") for value in values]
    if len(xs) < 2:
        return {"status": "NOT_ESTIMABLE", "n": len(xs)}
    mean = statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    margin = student_t_isf(alpha, len(xs) - 1) * se
    return {
        "status": "OK",
        "n": len(xs),
        "df": len(xs) - 1,
        "mean": mean,
        "se": se,
        "lower": mean - margin,
        "upper": mean + margin,
        "one_sided_confidence": 1 - alpha,
    }


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    if not xs:
        raise InvalidBank("empty registered metric group")
    return statistics.fmean(xs)


def _verify_identity(intent: dict[str, Any]) -> None:
    """Bind current inputs and committed producer bytes, not merely row counts."""
    commit = intent["source_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise InvalidBank("invalid source commit")
    protocol = REPO / "research/t34_pathb/PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != intent["protocol_sha256"]:
        raise InvalidBank("protocol hash differs")
    sources = intent["source_files"]
    if not isinstance(sources, dict) or not sources:
        raise InvalidBank("source_files missing")
    sources = {**sources, "research/t34_pathb/PROTOCOL.md": intent["protocol_sha256"]}
    for name, digest in sources.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()):
            raise InvalidBank("source path escapes repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise InvalidBank(f"current source hash differs: {name}")
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        if hashlib.sha256(committed).hexdigest() != digest:
            raise InvalidBank(f"source commit hash differs: {name}")
    checkpoints = intent["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise InvalidBank("checkpoint inventory missing")
    for item in checkpoints:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise InvalidBank("checkpoint hash differs")
    if intent["checkpoint"]["sha256"] != B8_SHA256:
        raise InvalidBank("hero checkpoint is not fixed B8 s3")
    if intent["checkpoint"] not in checkpoints:
        raise InvalidBank("hero missing from checkpoint inventory")


def _coordinates(intent: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    settings = intent["settings"]
    if settings["fps"] != [12, 24] or settings["workers"] != 1:
        raise InvalidBank("expected 12/24 fps and exactly one worker")
    horizon = _integer(settings["horizon"], "horizon", minimum=1)
    if type(settings["smoke"]) is not bool or (not settings["smoke"] and horizon != 5000):
        raise InvalidBank("non-smoke requires terminal H5000")
    blocks = intent["bank"]["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise InvalidBank("empty bank")
    expected: dict[tuple[Any, ...], dict[str, Any]] = {}
    block_ids, seeds = set(), set()
    for block in blocks:
        block_id = _integer(block["block"], "block")
        if block_id in block_ids:
            raise InvalidBank("duplicate independent block")
        block_ids.add(block_id)
        for kind in ("terminal", "product"):
            rows = block[kind]
            if not rows and not settings["smoke"]:
                raise InvalidBank(f"block {block_id}: missing {kind} worlds")
            if (
                kind == "terminal"
                and not settings["smoke"]
                and {row["mix"] for row in rows} != set(MIXES)
            ):
                raise InvalidBank(f"block {block_id}: missing terminal mix")
            for row in rows:
                seed = _integer(row["seed"], "seed")
                if seed in seeds:
                    raise InvalidBank("duplicate world seed")
                seeds.add(seed)
                mix = row["mix"]
                if mix not in MIXES or (kind == "product" and mix != "mixed"):
                    raise InvalidBank("unexpected world mix")
                for fps in settings["fps"]:
                    expected[(kind, block_id, seed, fps, mix)] = row
    if not expected:
        raise InvalidBank("empty registered bank")
    if _integer(intent["expected_records"], "expected_records") != len(expected):
        raise InvalidBank("manifest record count differs from coordinates")
    return expected


def _validate_arm(arm: dict[str, Any], kind: str, horizon: int) -> None:
    _number(arm["mass_integral"], "mass_integral", minimum=0)
    decisions = _integer(arm["decisions"], "decisions", minimum=1)
    if decisions > horizon:
        raise InvalidBank("decisions exceed horizon")
    boosts = _integer(arm["boost_decisions"], "boost_decisions")
    if boosts > decisions or not math.isclose(
        _number(arm["boost_share"], "boost_share", minimum=0, maximum=1),
        boosts / decisions,
        abs_tol=1e-12,
    ):
        raise InvalidBank("boost counters do not reconcile")
    if kind == "terminal":
        survival = _number(arm["survival_fraction"], "survival_fraction", minimum=0, maximum=1)
        no_death = _integer(arm["no_death"], "no_death")
        if no_death not in (0, 1) or (no_death == 1 and survival != 1):
            raise InvalidBank("terminal no-death/survival mismatch")
    else:
        deaths = _integer(arm["deaths"], "deaths")
        if deaths > decisions or not math.isclose(
            _number(arm["deaths_per_1k"], "deaths_per_1k", minimum=0),
            deaths * 1000 / decisions,
            abs_tol=1e-10,
        ):
            raise InvalidBank("death counters do not reconcile")
        median = _number(arm["median_length"], "median_length", minimum=0)
        if _number(arm["p95_length"], "p95_length", minimum=0) < median:
            raise InvalidBank("length quantiles unordered")
        large = _integer(arm["deaths_length_ge_800"], "deaths_length_ge_800")
        late = _integer(arm["late_deaths"], "late_deaths")
        overlap = _integer(arm["late_large_overlap"], "late_large_overlap")
        if (
            max(large, late) > deaths
            or overlap > min(large, late)
            or large + late - overlap > deaths
        ):
            raise InvalidBank("death diagnostic counters inconsistent")


def _read_records(root: Path, intent: dict[str, Any], expected: dict) -> list[dict[str, Any]]:
    paths = sorted((root / "records").glob("*.json"))
    if len(paths) != len(expected):
        raise InvalidBank(f"record count {len(paths)} differs from {len(expected)}")
    seen, records = set(), []
    for path in paths:
        row = _load(path)
        mix = row["mix"] if row["kind"] == "terminal" else row.get("mix", "mixed")
        key = (
            row["kind"],
            _integer(row["block"], "block"),
            _integer(row["seed"], "seed"),
            _integer(row["fps"], "fps"),
            mix,
        )
        if key not in expected or key in seen or row["status"] != "complete":
            raise InvalidBank(f"unexpected, duplicate, or incomplete row: {path.name}")
        if row["schema"] != f"t34-pathb-{row['kind']}/v1":
            raise InvalidBank("unexpected record schema")
        if set(row["arms"]) != set(ARMS):
            raise InvalidBank("missing or unexpected arm")
        for arm in row["arms"].values():
            _validate_arm(arm, row["kind"], intent["settings"]["horizon"])
        seen.add(key)
        records.append(row)
    return records


def _skill(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    blocks = sorted({row["block"] for row in rows})
    by_mix, by_block = {}, []

    def delta(group: list[dict[str, Any]], metric: str) -> float:
        return _mean(row["arms"][arm][metric] - row["arms"]["bare"][metric] for row in group)

    for block in blocks:
        effects = {}
        for mix in MIXES:
            group = [row for row in rows if row["block"] == block and row["mix"] == mix]
            effects[mix] = {
                "mass_gain": delta(group, "mass_integral"),
                "no_death_gain": delta(group, "no_death"),
                "alive_fraction_gain": delta(group, "survival_fraction"),
                "absolute_no_death": _mean(row["arms"][arm]["no_death"] for row in group),
            }
        by_block.append(
            {
                "block": block,
                "mixes": effects,
                "equal_mix_mass_gain": _mean(e["mass_gain"] for e in effects.values()),
                "equal_mix_no_death_gain": _mean(e["no_death_gain"] for e in effects.values()),
            }
        )
    mass = _ci(block["equal_mix_mass_gain"] for block in by_block)
    for mix in MIXES:
        by_mix[mix] = {
            metric: _mean(block["mixes"][mix][metric] for block in by_block)
            for metric in ("mass_gain", "no_death_gain", "alive_fraction_gain", "absolute_no_death")
        }
    decisions = sum(row["arms"][arm]["decisions"] for row in rows)
    boost_share = sum(row["arms"][arm]["boost_decisions"] for row in rows) / decisions
    clauses = {
        "mass_gain_ge_40": mass.get("mean", -math.inf) >= 40,
        "mass_lb90_positive": mass.get("lower", -math.inf) > 0,
        "scripted_no_death_gain_ge_003": by_mix["scripted"]["no_death_gain"] >= 0.03,
        "pooled_no_death_noninferiority": _mean(b["equal_mix_no_death_gain"] for b in by_block)
        >= -0.05,
        "each_mix_no_death_floor": all(m["no_death_gain"] >= -0.075 for m in by_mix.values()),
        "each_mix_absolute_no_death_ge_030": all(
            m["absolute_no_death"] >= 0.30 for m in by_mix.values()
        ),
        "each_mix_alive_fraction_guard": all(
            m["alive_fraction_gain"] >= -0.03 for m in by_mix.values()
        ),
        "each_block_scripted_alive_fraction_guard": all(
            b["mixes"]["scripted"]["alive_fraction_gain"] >= -0.15 for b in by_block
        ),
        "each_mix_absolute_no_death_ge_050": all(
            m["absolute_no_death"] >= 0.50 for m in by_mix.values()
        ),
        "boost_share_in_range": 0.005 <= boost_share <= 0.10,
    }
    return {
        "mass_gain_lb90": mass,
        "independent_unit": "bank_block",
        "blocks": by_block,
        "mixes": by_mix,
        "boost_share": boost_share,
        "clauses": clauses,
        "pass": all(clauses.values()),
    }


def _product(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    effects = []
    for block in sorted({row["block"] for row in rows}):
        group = [row for row in rows if row["block"] == block]
        relative = []
        for row in group:
            base = row["arms"]["bare"]["mass_integral"]
            if base <= 0:
                raise InvalidBank("zero bare mass makes paired relative mass NOT_ESTIMABLE")
            relative.append((row["arms"][arm]["mass_integral"] - base) / base)
        effects.append(
            {
                "block": block,
                "death_rate_delta": _mean(
                    row["arms"][arm]["deaths_per_1k"] - row["arms"]["bare"]["deaths_per_1k"]
                    for row in group
                ),
                "relative_mass_delta": _mean(relative),
                "bare_death_rate": _mean(row["arms"]["bare"]["deaths_per_1k"] for row in group),
            }
        )
    deaths = _ci((b["death_rate_delta"] for b in effects), 0.025)
    mass = _ci((b["relative_mass_delta"] for b in effects), 0.025)
    base_rate = _mean(b["bare_death_rate"] for b in effects)
    reduction = (
        -_mean(b["death_rate_delta"] for b in effects) / base_rate if base_rate > 0 else None
    )
    clauses = {
        "death_rate_upper95_negative": deaths.get("upper", math.inf) < 0,
        "relative_mass_lower95_ge_minus005": mass.get("lower", -math.inf) >= -0.05,
    }
    diagnostics = [
        {
            "block": row["block"],
            "seed": row["seed"],
            **{
                key: row["arms"][arm][key]
                for key in (
                    "median_length",
                    "p95_length",
                    "deaths_length_ge_800",
                    "late_deaths",
                    "late_large_overlap",
                )
            },
        }
        for row in rows
    ]
    return {
        "death_rate_delta95": deaths,
        "relative_mass_delta95": mass,
        "blocks": effects,
        "relative_death_reduction": reduction,
        "target_20_percent_diagnostic": (
            reduction >= 0.20 if reduction is not None else "NOT_ESTIMABLE"
        ),
        "world_length_and_death_diagnostics": diagnostics,
        "clauses": clauses,
        "pass": all(clauses.values()),
    }


def _percentile(xs: list[float], q: float) -> float:
    ordered = sorted(xs)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _distribution(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"status": "NOT_MEASURED", "n": 0}
    return {
        "status": "OK",
        "n": len(xs),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "max": max(xs),
    }


def _timing(root: Path, intent: dict[str, Any]) -> dict[str, Any]:
    expected = {f"{mode}-{fps}.json" for mode in ("bare", "async") for fps in (12, 24)}
    paths = list((root / "timing").glob("*.json"))
    if {path.name for path in paths} != expected:
        raise InvalidBank("missing or unexpected timing rows")
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    sample_count = _integer(intent["settings"]["timing_samples"], "timing_samples", minimum=1)
    for path in paths:
        row = _load(path)
        mode, fps = row["mode"], _integer(row["fps"], "timing fps")
        if path.name != f"{mode}-{fps}.json" or row["status"] != "complete":
            raise InvalidBank("timing coordinate mismatch")
        if row["actual_speed"] != fps or row["speed_ceiling"] != 24:
            raise InvalidBank("timing speed/ceiling mismatch")
        samples = row["samples"]
        if len(samples) != sample_count:
            raise InvalidBank("timing sample count mismatch")
        updates, ticks, long_ticks, contention = [], [], [], []
        for sample in samples:
            update = _number(sample["update_ms"], "update_ms", minimum=0)
            tick = _number(sample["tick_ms"], "tick_ms", minimum=0)
            length = _integer(sample["hero_length"], "hero_length")
            if type(sample["worker_busy"]) is not bool or tick < update:
                raise InvalidBank("timing sample inconsistent")
            updates.append(update)
            ticks.append(tick)
            if length >= 600:
                long_ticks.append(tick)
            if sample["worker_busy"]:
                contention.append(tick)
        elapsed = _number(row["elapsed_seconds"], "elapsed_seconds", minimum=0)
        if elapsed <= 0 or elapsed + 1e-9 < sum(ticks) / 1000 + sample_count / fps:
            raise InvalidBank("timing elapsed does not include real processing-plus-sleep pacing")
        summaries[(mode, fps)] = {
            "update_ms": _distribution(updates),
            "tick_ms": _distribution(ticks),
            "natural_long_body_tick_ms": _distribution(long_ticks),
            "worker_contention_tick_ms": _distribution(contention),
            "overruns": sum(t >= 1000 / fps for t in ticks),
            "achieved_fps": sample_count / elapsed,
        }
    report = {}
    for fps in (12, 24):
        bare, assisted = summaries[("bare", fps)], summaries[("async", fps)]
        cadence_change = assisted["achieved_fps"] / bare["achieved_fps"] - 1
        base_p95 = bare["tick_ms"]["p95"]
        if base_p95 <= 0:
            raise InvalidBank("zero baseline full-tick latency")
        proxy_change = assisted["tick_ms"]["p95"] / base_p95 - 1
        clauses = {
            "update_p95_le_11688ms": assisted["update_ms"]["p95"] <= 11.688,
            "zero_overruns": bare["overruns"] == assisted["overruns"] == 0,
            "cadence_deterioration_le_5pct": cadence_change >= -0.05,
            "full_tick_p95_proxy_deterioration_le_10pct": proxy_change <= 0.10,
        }
        report[str(fps)] = {
            "bare": bare,
            "async": assisted,
            "cadence_change": cadence_change,
            "full_tick_p95_proxy_change": proxy_change,
            "clauses": clauses,
            "pass": all(clauses.values()),
        }
    return report


def analyze(root: Path) -> dict[str, Any]:
    """Analyze one complete fixed bank; never fill gaps or infer unmeasured clients."""
    report: dict[str, Any] = {
        "schema": "t34-pathb-analysis/v1",
        "complete": False,
        "development_only": True,
        "client_latency": "NOT_MEASURED",
        "serving_qualification": "NOT_MEASURED",
        "release_authorized": False,
    }
    try:
        root = Path(root)
        intent = _load(root / "manifest.json")
        if intent["schema"] != "t34-pathb-manifest/v1":
            raise InvalidBank("manifest schema mismatch")
        _verify_identity(intent)
        expected = _coordinates(intent)
        rows = _read_records(root, intent, expected)
        report.update(expected_records=len(expected), records=len(rows), complete=True)
        if intent["settings"]["smoke"]:
            report.update(
                decision="SMOKE_ONLY",
                clauses="NOT_ESTIMABLE",
                skill="NOT_ESTIMABLE",
                product="NOT_ESTIMABLE",
                timing="PLUMBING_ONLY",
            )
            return report
        efficacy = {}
        for fps in (12, 24):
            terminal = [r for r in rows if r["kind"] == "terminal" and r["fps"] == fps]
            product = [r for r in rows if r["kind"] == "product" and r["fps"] == fps]
            efficacy[str(fps)] = {
                "async": {
                    "skill": _skill(terminal, "async"),
                    "product": _product(product, "async"),
                },
                "sync_unlimited_descriptive": {
                    "skill": _skill(terminal, "sync"),
                    "product": _product(product, "sync"),
                },
            }
        report["efficacy_by_fps"] = efficacy
        report["timing_by_fps"] = _timing(root, intent)
        passed = all(
            value["async"]["skill"]["pass"] and value["async"]["product"]["pass"]
            for value in efficacy.values()
        )
        passed = passed and all(value["pass"] for value in report["timing_by_fps"].values())
        report["decision"] = "DEVELOPMENT_BARS_PASS" if passed else "DEVELOPMENT_BARS_FAIL"
        report["contract_acceptance"] = "INCOMPLETE_CLIENT_LATENCY_NOT_MEASURED"
    except (
        InvalidBank,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        ArithmeticError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        report.update(
            decision="INCOMPLETE", complete=False, errors=[str(exc)], clauses="NOT_ESTIMABLE"
        )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.out)
    target = args.out / "analysis.json"
    with target.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"analysis": str(target), "decision": report["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
