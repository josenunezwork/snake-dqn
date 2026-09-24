"""Audit only this new study's receipts and saved outputs, without model imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

COUNTERS = (
    "native_frames",
    "hero_transitions",
    "model_forwards",
    "checkpoint_loads",
    "optimizer_updates",
)


def check(value: bool, message: str) -> None:
    """Fail independently of assert optimization."""
    if not value:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    """Read only explicitly named JSON records."""

    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite JSON: {value}")

    result = json.loads(path.read_text(), parse_constant=invalid)
    check(isinstance(result, dict), f"object required: {path}")
    return result


def sha(path: Path) -> str:
    """Hash authenticated new output bytes without loading model tensors."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pulse(heartbeat: Path, phase: str) -> None:
    """Publish atomic heartbeat metadata with an unambiguous argument name."""
    temporary = heartbeat.with_suffix(".tmp")
    temporary.write_text(json.dumps({"monotonic": time.monotonic(), "phase": phase}))
    temporary.replace(heartbeat)


def load_evaluator(intent: dict[str, Any]) -> Any:
    """Import only the frozen stdlib saved-record validator and reducer."""
    path = Path(intent["evaluation_source"])
    check(sha(path) == intent["source_sha256"][str(path)], "evaluation source drift")
    spec = importlib.util.spec_from_file_location("task_aligned_audit_evaluation", path)
    check(spec is not None and spec.loader is not None, "evaluation source spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_frame_metrics(case: dict[str, Any], horizon: int) -> None:
    """Independently derive metrics from saved physical frame columns."""
    columns = case["frame_columns"]
    check(
        set(columns) == {"alive", "mass", "ambient_food", "boost", "encounter"},
        "frame-column schema",
    )
    length = len(columns["alive"])
    check(
        0 < length <= horizon and all(len(values) == length for values in columns.values()),
        "frame-column lengths",
    )
    for key in ("alive", "boost", "encounter"):
        check(all(type(value) is bool for value in columns[key]), f"boolean column: {key}")
    check(
        all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in columns["mass"]
        ),
        "mass column",
    )
    check(
        all(type(value) is int and value >= 0 for value in columns["ambient_food"]), "food column"
    )
    check(all(columns["alive"][:-1]), "frames after hero death")
    check(columns["alive"][-1] or columns["mass"][-1] == 0, "dead mass not zero")
    expected = {
        "observed_frames": length,
        "living_mass_sum": sum(columns["mass"]),
        "mass_integral": sum(columns["mass"]) / horizon,
        "ambient_food": sum(columns["ambient_food"]),
        "survival_fraction": sum(columns["alive"]) / horizon,
        "endpoint": int(columns["alive"][-1]),
        "boost_fraction": sum(columns["boost"]) / horizon,
        "encounter_frames": sum(columns["encounter"][16:]),
    }
    for key, value in expected.items():
        check(case[key] == value, f"independent frame reduction: {key}")


def check_cell_counters(cell: dict[str, Any]) -> None:
    """Reconcile episode work, initial loads, and shared batched model calls."""
    initial = cell["initial_counters"]
    shared = cell.get("shared_counters", dict.fromkeys(COUNTERS, 0))
    for counts in (initial, shared, cell["counters"]):
        check(set(counts) == set(COUNTERS), "cell counter schema")
        check(
            all(type(value) is int and value >= 0 for value in counts.values()), "cell counter type"
        )
    check(
        all(initial[key] == 0 for key in COUNTERS if key != "checkpoint_loads"),
        "non-load work outside episode stream",
    )
    check(
        all(shared[key] == 0 for key in COUNTERS if key != "model_forwards"),
        "shared work must be batched forward calls only",
    )
    for key in COUNTERS:
        case_values = [case["counters"][key] for case in cell["cases"]]
        check(all(type(value) is int and value >= 0 for value in case_values), "case counter type")
        check(
            sum(case_values) + initial[key] + shared[key] == cell["counters"][key],
            "independent batched cell counter sum",
        )


def inspect(
    intent: dict[str, Any], intent_path: Path, root: Path, heartbeat: Path
) -> dict[str, Any]:
    """Verify complete prefix; the controller authenticates this audit's own receipt."""
    identity = sha(intent_path)
    check(root.resolve() == Path(intent["output_root"]).resolve(), "output root mismatch")
    check(not (root / "failure.json").exists(), "study already failed")
    check(read(root / "started.json")["intent_sha256"] == identity, "root intent identity")
    stages = intent["stages"]
    check(
        stages[0]["kind"] == "qualification" and stages[-1]["kind"] == "audit",
        "qualification/audit ordering",
    )
    check(len({s["id"] for s in stages}) == len(stages), "stage IDs not unique")
    totals = {key: 0 for key in COUNTERS}
    training_totals = {key: 0 for key in COUNTERS}
    elapsed = 0.0
    previous_end = None
    audited = []
    selected_dose = None
    cells = []
    analysis = None
    evaluator = load_evaluator(intent)
    evaluation_spec = None
    training_worlds = set()
    ledger_lineages = set()
    check(intent["evaluation"]["training_world_seeds"] == [], "frozen seed ledger not empty")
    for index, stage in enumerate(stages[:-1]):
        pulse(heartbeat, f"receipt:{stage['id']}")
        directory = root / stage["id"]
        started = read(directory / "started.json")
        admitted = read(directory / "admitted.json")
        completed = read(directory / "completed.json")
        supervisor = read(directory / "supervisor.json")
        report_path = directory / "output/report.json"
        report = read(report_path)
        check(
            started["intent_sha256"] == completed["intent_sha256"] == identity,
            "stage intent identity",
        )
        check(started["stage_id"] == completed["stage_id"] == stage["id"], "stage identity")
        values = {
            "root": str(root),
            "out": str(directory / "output"),
            "intent": str(intent_path),
            "heartbeat": str(directory / "output/heartbeat.json"),
            "stage_id": stage["id"],
        }
        check(started["argv"] == [item.format_map(values) for item in stage["argv"]], "argv drift")
        required = sum(s["cap_seconds"] for s in stages[index:]) + 120
        check(
            started["cap_seconds"] == stage["cap_seconds"]
            and started["remaining_required_seconds"] == required,
            "budget identity",
        )
        begin, admission, end = (
            datetime.fromisoformat(x["utc"]) for x in (started, admitted, completed)
        )
        check(
            begin <= admission <= end and (previous_end is None or previous_end <= begin),
            "stage timing/order",
        )
        check(
            (datetime.fromisoformat(intent["deadline"]) - admission).total_seconds() >= required,
            "post-lock deadline admission",
        )
        previous_end = end
        result = supervisor["result"]
        check(
            result["cause"] is None
            and result["confirmed_exit"] is True
            and result["termination"] == "natural_exit"
            and result["observed_returncode"] == 0,
            "non-successful or ambiguous child",
        )
        seconds = result["elapsed_seconds"]
        check(
            type(seconds) in (int, float)
            and math.isfinite(seconds)
            and 0 <= seconds <= stage["cap_seconds"],
            "elapsed cap",
        )
        elapsed += seconds
        check(
            supervisor["limits"]["rss_bytes"] == 8 * 1024**3
            and supervisor["limits"]["mps_driver_bytes"] == 24 * 1024**3
            and supervisor["limits"]["available_bytes"] == math.ceil(9.6 * 1024**3),
            "resource guard identity",
        )
        check(
            completed["supervisor_sha256"] == sha(directory / "supervisor.json")
            and completed["report_sha256"] == sha(report_path),
            "receipt hash mismatch",
        )
        check(
            report["schema_version"] == 1
            and report["study_id"] == intent["study_id"]
            and report["stage_id"] == stage["id"]
            and report["status"] == "PASS",
            "report contract",
        )
        check(report["counters"] == completed["counters"], "counter receipt mismatch")
        for key, limit in stage["counter_caps"].items():
            value = report["counters"][key]
            check(type(value) is int and 0 <= value <= limit, f"counter cap: {key}")
        for key in COUNTERS:
            totals[key] += report["counters"][key]
            if stage["kind"] == "training":
                training_totals[key] += report["counters"][key]
        if stage["kind"] == "training":
            check(report["counters"]["hero_transitions"] <= 10_000_000, "lineage ceiling")
            check(selected_dose is not None, "training before qualified dose")
            check(
                report["lineage"] == stage["lineage"]
                and report["selected_rollouts"] == selected_dose["selected_rollouts"]
                and report["completed_rollouts"] == selected_dose["selected_rollouts"]
                and report["profile_rollouts"] == selected_dose["profile_rollouts"]
                and report["dose_sha256"] == selected_dose["dose_sha256"],
                "training dose identity",
            )
        for key, expected in stage.get("expected_report_fields", {}).items():
            check(report.get(key) == expected, f"required output field: {key}")
        if stage["kind"] == "qualification":
            check(
                report["runtime_contracts_pass"] is True and report["fixed_science_fits"] is True,
                "qualification did not admit fixed science",
            )
            projection = report["estimated_science_seconds"]
            check(
                type(projection) in (int, float)
                and math.isfinite(projection)
                and 0
                < projection
                <= sum(s["cap_seconds"] for s in stages if s["kind"] == "training"),
                "throughput projection",
            )
            training = [s for s in stages if s["kind"] == "training"]
            cycle, overhead = report["conservative_cycle_seconds"], report["fixed_overhead_seconds"]
            check(
                type(cycle) in (int, float) and math.isfinite(cycle) and cycle > 0,
                "invalid cycle timing",
            )
            check(
                type(overhead) in (int, float) and math.isfinite(overhead) and overhead >= 0,
                "invalid fixed overhead",
            )
            check(report["shared_probe_lineage"] == 2026096101, "timing probe lineage")
            cap = min(s["cap_seconds"] for s in training)
            count = min(39060, int((cap / 2 - overhead) / cycle) * 5)
            check(count >= 7815 and report["selected_rollouts"] == count, "dose selection rule")
            forecast = 2 * (overhead + count / 5 * cycle)
            check(
                report["forecast_per_lineage_seconds"] == forecast and projection == 3 * forecast,
                "qualification forecast",
            )
            selected_dose = {
                "selected_rollouts": count,
                "conservative_cycle_seconds": cycle,
                "fixed_overhead_seconds": overhead,
                "shared_probe_lineage": 2026096101,
                "profile_rollouts": {
                    "solo": count // 5,
                    "S2": count * 2 // 5,
                    "S6": count * 2 // 5,
                },
            }
            encoded = json.dumps(
                selected_dose, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            selected_dose["dose_sha256"] = hashlib.sha256(encoded).hexdigest()
            check(
                selected_dose == read(root / "qualified-dose.json")
                and selected_dose["dose_sha256"] == report["dose_sha256"],
                "dose seal",
            )
        expected_artifacts = {}
        for item in report["artifacts"]:
            pulse(heartbeat, f"artifact:{stage['id']}")
            path = (directory / "output" / item["path"]).resolve()
            check(path.is_relative_to((directory / "output").resolve()), "artifact escape")
            check(path.is_file() and sha(path) == item["sha256"], "artifact hash mismatch")
            expected_artifacts[str(path)] = item["sha256"]
        check(expected_artifacts == completed["artifacts"], "artifact inventory mismatch")
        if stage["kind"] == "training":
            ledger_path = (directory / "output/training-world-seeds.json").resolve()
            check(str(ledger_path) in expected_artifacts, "unsealed training-world ledger")
            worlds = read(ledger_path)["worlds"]
            check(set(worlds) == {"solo", "s2", "s6"}, "training-world profile inventory")
            runtime_seed = intent["runtime_seeds"][str(stage["lineage"])]
            for profile, environments in worlds.items():
                check(
                    isinstance(environments, list) and len(environments) == 16,
                    "training-world environment inventory",
                )
                for env_index, episodes in enumerate(environments):
                    pulse(heartbeat, f"training_worlds:{stage['lineage']}:{profile}:{env_index}")
                    check(
                        isinstance(episodes, list) and bool(episodes), "empty training-world ledger"
                    )
                    check(
                        all(type(seed) is int and seed >= 0 for seed in episodes),
                        "invalid training-world seed",
                    )
                    for episode_index, seed in enumerate(episodes):
                        namespace = f"{profile}/world/{env_index}/{episode_index}"
                        expected_seed = int.from_bytes(
                            hashlib.sha256(
                                f"task-aligned/v1/{runtime_seed}/{namespace}".encode()
                            ).digest()[:8],
                            "big",
                        )
                        check(seed == expected_seed, "training-world namespace identity")
                    training_worlds.update(episodes)
            check(stage["lineage"] not in ledger_lineages, "duplicate training-world lineage")
            ledger_lineages.add(stage["lineage"])
        if stage["kind"] == "evaluation":
            expected_lineages = {s["lineage"] for s in stages if s["kind"] == "training"}
            check(
                len(expected_lineages) == 3 and ledger_lineages == expected_lineages,
                "evaluation before all three sealed training ledgers",
            )
            if evaluation_spec is None:
                evaluation_spec = dict(intent["evaluation"])
                evaluation_spec["training_world_seeds"] = sorted(training_worlds)
                evaluator.validate_spec(evaluation_spec)
            spec_path = (directory / "output" / report["evaluation_spec"]).resolve()
            check(str(spec_path) in expected_artifacts, "unsealed evaluation spec")
            check(read(spec_path) == evaluation_spec, "generated evaluation spec mismatch")
            batch = []
            for name in report["evaluation_reports"]:
                path = (directory / "output" / name).resolve()
                check(str(path) in expected_artifacts, "evaluation cell absent from artifacts")
                cell = read(path)
                evaluator.validate_report(evaluation_spec, cell)
                check_cell_counters(cell)
                for case in cell["cases"]:
                    check_frame_metrics(case, evaluation_spec["horizon"])
                batch.append(cell)
                cells.append(cell)
            check(bool(batch), "empty evaluation batch")
            for key in COUNTERS:
                check(
                    sum(cell["counters"][key] for cell in batch) == report["counters"][key],
                    "evaluation batch counter sum",
                )
        if stage["kind"] == "analysis":
            check(analysis is None, "duplicate analysis")
            analysis = report
        audited.append({"stage_id": stage["id"], "report_sha256": sha(report_path)})
    check(training_totals["hero_transitions"] <= 30_000_000, "training transition ceiling")
    check(elapsed + stages[-1]["cap_seconds"] <= 86400, "24-hour numerical ceiling")
    pulse(heartbeat, "saved_decision_reduction")
    recomputed = evaluator.analyze_reports(evaluation_spec, cells)
    check(analysis is not None, "missing analysis")
    check(analysis["evaluation_spec"] == evaluation_spec, "analysis evaluation spec mismatch")
    for key, expected in recomputed.items():
        if key not in {"stage_id", "artifacts"}:
            check(analysis.get(key) == expected, f"analysis reduction mismatch: {key}")
    return {
        "audited_stages": audited,
        "audited_counters": totals,
        "audited_training_counters": training_totals,
        "audited_elapsed_seconds": elapsed,
        "intent_sha256": identity,
        "own_receipt_requires_controller_reconciliation": True,
        "scientific_reduction": "saved_records_shared_frozen_reducer",
        "frame_metric_reduction": "independent_saved_column_reduction",
        "promotion_eligible": False,
    }


def main() -> None:
    """Write a create-only audit report for this prospective study."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intent", "root", "out", "heartbeat"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage-id", required=True)
    args = parser.parse_args()
    intent_path = args.intent.resolve()
    intent = read(intent_path)
    check(args.stage_id == intent["stages"][-1]["id"], "audit stage identity")
    pulse(args.heartbeat, "audit_started")
    detail = inspect(intent, intent_path, args.root.resolve(), args.heartbeat)
    report = {
        "schema_version": 1,
        "study_id": intent["study_id"],
        "stage_id": args.stage_id,
        "status": "PASS",
        "kind": "receipt_audit",
        "counters": {k: 0 for k in COUNTERS},
        "artifacts": [],
        **detail,
    }
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pulse(args.heartbeat, "audit_complete")


if __name__ == "__main__":
    main()
