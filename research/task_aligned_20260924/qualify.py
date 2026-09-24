"""Finite, discarded qualification for the newly admitted task-aligned package.

Importing this module does no numerical work. The sole public numerical entry
point is called by stage_cli after create-only controller admission. Neither
schema fixtures nor a model load may run as an informal preflight.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

LINEAGES = (2026096101, 2026096102, 2026096103)
CYCLE = ("solo", "S2", "S2", "S6", "S6")
PROBE_CYCLES = 30
PROBE_ROLLOUTS = 150
MIN_ROLLOUTS = 7_815
MAX_ROLLOUTS = 39_060
SCIENCE_SECONDS = 25_200
QUALIFICATION_SECONDS = 1_800


def require(condition: bool, message: str) -> None:
    """Keep qualification failures active under optimized Python."""
    if not condition:
        raise RuntimeError(message)


def select_dose(cycle_seconds: list[float], fixed_seconds: float) -> dict[str, Any]:
    """Select by synchronized hardware timing alone, using the slowest full cycle.

    The complete parent restoration and checkpoint-output overhead is supplied in
    fixed_seconds. Doubling applies to both it and the projected rollout work.
    No gameplay, reward, loss or parameter value is an input to this decision.
    """
    require(len(cycle_seconds) == PROBE_CYCLES, "exactly 30 timed cycles required")
    require(
        all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in cycle_seconds),
        "invalid synchronized cycle timing",
    )
    require(
        type(fixed_seconds) in (int, float) and math.isfinite(fixed_seconds) and fixed_seconds >= 0,
        "invalid fixed overhead timing",
    )
    conservative_cycle = max(cycle_seconds)
    affordable_cycles = math.floor((SCIENCE_SECONDS / 2 - fixed_seconds) / conservative_cycle)
    selected = max(0, min(MAX_ROLLOUTS // 5, affordable_cycles)) * 5
    forecast_per_seed = 2 * (fixed_seconds + selected / 5 * conservative_cycle)
    return {
        "selected_rollouts": selected,
        "fixed_science_fits": MIN_ROLLOUTS <= selected <= MAX_ROLLOUTS,
        "estimated_science_seconds": 3 * forecast_per_seed,
        "forecast_per_lineage_seconds": forecast_per_seed,
        "conservative_cycle_seconds": conservative_cycle,
        "fixed_overhead_seconds": fixed_seconds,
        "forecast_safety_factor": 2,
        "timing_rule": "twice_fixed_overhead_plus_slowest_complete_five_rollout_cycle",
        "scheduled_hero_slots_per_lineage": selected * 16 * 16,
    }


def _reject(call: Callable[[], Any], label: str) -> str:
    """Require an explicit schema failure; unexpected exceptions remain failures."""
    try:
        call()
    except (ValueError, RuntimeError, FileExistsError):
        return label
    raise RuntimeError(f"schema fixture accepted malformed evidence: {label}")


def _synthetic_reports(evaluation: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct 768 algebraic records; never simulate a physical game."""
    policies = [
        {"role": role, "lineage": lineage, "checkpoint_sha256": "a" * 64}
        for role in ("parent", "candidate")
        for lineage in LINEAGES
    ]
    policies.extend(
        {"role": "anchor", "name": name, "lineage": None} for name in ("greedy_food", "random_safe")
    )
    result = []
    for policy in policies:
        for profile in spec["profiles"]:
            cases = []
            mass = 12.0 if policy["role"] == "candidate" and profile != "solo" else 10.0
            for seed in spec["world_seeds"][profile]:
                descriptor = {
                    "world_seed": seed,
                    "hero": {"fixture": True},
                    "opponents": list(spec["profiles"][profile]),
                    "food": [],
                    "rng_sha256": "b" * 64,
                }
                counters = dict.fromkeys(evaluation.COUNTERS, 0)
                counters.update(native_frames=17, hero_transitions=17)
                cases.append(
                    {
                        "world_seed": seed,
                        "profile": profile,
                        "horizon": 3000,
                        "observed_frames": 17,
                        "initial_descriptor": descriptor,
                        "initial_sha256": evaluation.canonical_hash(descriptor),
                        "living_mass_sum": mass * 3000,
                        "mass_integral": mass,
                        "ambient_food": 10,
                        "survival_fraction": 16 / 3000,
                        "endpoint": 0,
                        "boost_fraction": 0.0,
                        "encounter_frames": 0 if profile == "solo" else 1,
                        "counters": counters,
                        "frame_columns": {
                            "alive": [True] * 16 + [False],
                            "mass": [mass * 3000 / 16] * 16 + [0.0],
                            "ambient_food": [10] + [0] * 16,
                            "boost": [False] * 17,
                            "encounter": [profile != "solo"] * 17,
                        },
                    }
                )
            totals = {
                key: sum(row["counters"][key] for row in cases) for key in evaluation.COUNTERS
            }
            result.append(
                {
                    "schema_version": 1,
                    "study_id": spec["study_id"],
                    "stage_id": "fixture",
                    "status": "PASS",
                    "kind": "policy_evaluation",
                    "spec_sha256": evaluation.canonical_hash(spec),
                    "contract_sha256": spec["contract_sha256"],
                    "policy": policy,
                    "profile": profile,
                    "cases": cases,
                    "cases_sha256": evaluation.canonical_hash(cases),
                    "counters": totals,
                    "initial_counters": dict.fromkeys(evaluation.COUNTERS, 0),
                    "summary": {
                        key: statistics.mean(row[key] for row in cases)
                        for key in evaluation.METRICS
                    },
                    "artifacts": [],
                    "promotion_eligible": False,
                }
            )
    return result


def _refresh(evaluation: Any, reports: list[dict[str, Any]]) -> None:
    """Rehash only intentionally changed synthetic evidence."""
    for report in reports:
        for row in report["cases"]:
            row["frame_columns"]["mass"] = [row["mass_integral"] * 3000 / 16] * 16 + [0.0]
            row["frame_columns"]["encounter"] = [bool(row["encounter_frames"])] * 17
        report["cases_sha256"] = evaluation.canonical_hash(report["cases"])
        report["summary"] = {
            key: statistics.mean(row[key] for row in report["cases"]) for key in evaluation.METRICS
        }


def reporter_fixtures(evaluation: Any, spec: dict[str, Any], out: Path) -> dict[str, Any]:
    """Exercise saved-evidence decisions and rejection boundaries in a temp tree."""
    reports = _synthetic_reports(evaluation, spec)
    checks = []
    with tempfile.TemporaryDirectory(prefix="qualification-schema-", dir=out) as temporary:
        target = Path(temporary) / "decision.json"
        result = evaluation.analyze_reports(spec, reports, target)
        require(result["decision"] == "ADVANCE_PACKAGE_SCREEN", "synthetic pass decision")
        require(json.loads(target.read_text()) == result, "saved decision roundtrip")
        checks.append("complete_saved_report_roundtrip")
        batched = copy.deepcopy(reports[0])
        batched["shared_counters"] = dict.fromkeys(evaluation.COUNTERS, 0)
        batched["shared_counters"]["model_forwards"] = 2
        batched["counters"]["model_forwards"] = 2
        evaluation.validate_report(spec, batched)
        checks.append("shared_batched_forward_accounting")
        batched["shared_counters"]["native_frames"] = 1
        checks.append(
            _reject(
                lambda: evaluation.validate_report(spec, batched), "shared_native_frames_rejected"
            )
        )
        checks.append(
            _reject(lambda: evaluation.analyze_reports(spec, reports, target), "create_only_report")
        )
        for scenario, expected in (
            ("retention", "NEGATIVE_RELIABILITY_SCREEN"),
            ("futility", "NEGATIVE_PRACTICAL_GAIN_SCREEN"),
            ("uncertain", "INCONCLUSIVE_SCREEN"),
            ("exposure", "INCONCLUSIVE_EXPOSURE"),
        ):
            changed = copy.deepcopy(reports)
            for report in changed:
                for index, row in enumerate(report["cases"]):
                    if scenario == "exposure":
                        row["encounter_frames"] = 0
                    if report["policy"]["role"] != "candidate":
                        continue
                    if scenario == "retention" and report["profile"] == "solo":
                        row["mass_integral"] = 8.0
                    elif scenario == "futility" and report["profile"] != "solo":
                        row["mass_integral"] = 10.0
                    elif scenario == "uncertain" and report["profile"] != "solo":
                        row["mass_integral"] = 10.5 + (-2 if index % 2 else 2)
                    row["living_mass_sum"] = row["mass_integral"] * 3000
            _refresh(evaluation, changed)
            actual = evaluation.analyze_reports(spec, changed)["decision"]
            require(actual == expected, f"synthetic {scenario} decision: {actual}")
            checks.append(scenario)
        checks.append(
            _reject(lambda: evaluation.analyze_reports(spec, reports[:-1]), "missing_cell")
        )
        checks.append(
            _reject(
                lambda: evaluation.analyze_reports(spec, reports + reports[:1]), "duplicate_cell"
            )
        )
        changed = copy.deepcopy(reports)
        changed[0]["cases"][0]["initial_descriptor"]["hero"]["fixture"] = False
        changed[0]["cases"][0]["initial_sha256"] = evaluation.canonical_hash(
            changed[0]["cases"][0]["initial_descriptor"]
        )
        _refresh(evaluation, changed)
        checks.append(
            _reject(lambda: evaluation.analyze_reports(spec, changed), "mismatched_initial_pair")
        )
        changed = copy.deepcopy(reports)
        changed[0]["counters"]["optimizer_updates"] = 1
        checks.append(
            _reject(
                lambda: evaluation.analyze_reports(spec, changed), "evaluation_optimizer_update"
            )
        )
        changed = copy.deepcopy(reports)
        changed[0]["cases"][0]["mass_integral"] = float("nan")
        checks.append(_reject(lambda: evaluation.analyze_reports(spec, changed), "nonfinite_case"))
        changed = copy.deepcopy(reports)
        changed[0]["cases"] = changed[0]["cases"][::-1]
        _refresh(evaluation, changed)
        checks.append(
            _reject(lambda: evaluation.analyze_reports(spec, changed), "reordered_worlds")
        )
    return {
        "checks": checks,
        "algebraic_fixture_cells": 24,
        "algebraic_fixture_world_rows": 768,
        "physical_worlds": 0,
    }


def operations_fixtures(
    runner: Any, auditor: Any, evaluation: Any, spec: dict[str, Any], out: Path
) -> dict[str, Any]:
    """Run the real receipt audit over a synthetic complete saved-evidence tree."""
    from datetime import datetime, timedelta, timezone

    def write(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))

    checks = []
    with tempfile.TemporaryDirectory(prefix="qualification-receipts-", dir=out) as temporary:
        root = Path(temporary)
        counters = dict.fromkeys(runner.COUNTERS, 0)
        stage_specs = [("qualification", "qualification", 1800)]
        stage_specs += [(f"training-{seed}", "training", SCIENCE_SECONDS) for seed in LINEAGES]
        stage_specs += [
            ("evaluation", "evaluation", 900),
            ("analysis", "analysis", 600),
            ("audit", "audit", 900),
        ]
        stages = [
            {
                "id": ident,
                "kind": kind,
                "cap_seconds": cap,
                "argv": ["fixture-python", "fixture-source", "{out}"],
                "counter_caps": {key: 3_000_000 for key in counters},
            }
            for ident, kind, cap in stage_specs
        ]
        for stage, lineage in zip(stages[1:4], LINEAGES):
            stage["lineage"] = lineage
        runtime_seeds = {str(lineage): 900000 + i for i, lineage in enumerate(LINEAGES)}
        ledgers = {
            str(lineage): {
                profile: [
                    [
                        int.from_bytes(
                            hashlib.sha256(
                                (
                                    f"task-aligned/v1/{runtime_seeds[str(lineage)]}/"
                                    f"{profile}/world/{env}/0"
                                ).encode()
                            ).digest()[:8],
                            "big",
                        )
                    ]
                    for env in range(16)
                ]
                for profile in ("solo", "s2", "s6")
            }
            for lineage in LINEAGES
        }
        spec = copy.deepcopy(spec)
        spec["training_world_seeds"] = sorted(
            {
                seed
                for worlds in ledgers.values()
                for environments in worlds.values()
                for episodes in environments
                for seed in episodes
            }
        )
        template = copy.deepcopy(spec)
        template["training_world_seeds"] = []
        evaluation_source = str(Path(evaluation.__file__).resolve())
        spec_path = root / "spec.json"
        write(spec_path, spec)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        intent = {
            "schema_version": 1,
            "study_id": spec["study_id"],
            "stages": stages,
            "output_root": str(root),
            "evaluation_source": evaluation_source,
            "evaluation": template,
            "runtime_seeds": runtime_seeds,
            "source_sha256": {
                evaluation_source: auditor.sha(Path(evaluation_source)),
                str(spec_path): auditor.sha(spec_path),
            },
            "deadline": (epoch + timedelta(days=1)).isoformat(),
        }
        intent_path = root / "intent.json"
        write(intent_path, intent)
        identity = auditor.sha(intent_path)
        write(root / "started.json", {"intent_sha256": identity})
        dose = select_dose([1.0] * 30, 2.0)
        payload = {
            key: dose[key]
            for key in ("selected_rollouts", "conservative_cycle_seconds", "fixed_overhead_seconds")
        }
        payload.update(
            shared_probe_lineage=LINEAGES[0],
            profile_rollouts={
                "solo": dose["selected_rollouts"] // 5,
                "S2": dose["selected_rollouts"] * 2 // 5,
                "S6": dose["selected_rollouts"] * 2 // 5,
            },
        )
        payload["dose_sha256"] = evaluation.canonical_hash(payload)
        write(root / "qualified-dose.json", payload)
        cells = _synthetic_reports(evaluation, spec)
        for index, stage in enumerate(stages[:-1]):
            directory = root / stage["id"]
            output = directory / "output"
            output.mkdir(parents=True)
            report = {
                "schema_version": 1,
                "study_id": spec["study_id"],
                "stage_id": stage["id"],
                "status": "PASS",
                "counters": counters.copy(),
                "artifacts": [],
            }
            if stage["kind"] == "qualification":
                report.update(
                    dose,
                    **{key: value for key, value in payload.items() if key not in dose},
                    runtime_contracts_pass=True,
                )
            elif stage["kind"] == "training":
                report.update(
                    lineage=stage["lineage"],
                    **payload,
                    completed_rollouts=payload["selected_rollouts"],
                )
                ledger_path = output / "training-world-seeds.json"
                write(ledger_path, {"worlds": ledgers[str(stage["lineage"])]})
                report["artifacts"].append(
                    {"path": ledger_path.name, "sha256": auditor.sha(ledger_path)}
                )
            elif stage["kind"] == "evaluation":
                report["evaluation_reports"] = []
                write(output / "evaluation-spec.json", spec)
                report["evaluation_spec"] = "evaluation-spec.json"
                report["artifacts"].append(
                    {
                        "path": "evaluation-spec.json",
                        "sha256": auditor.sha(output / "evaluation-spec.json"),
                    }
                )
                for number, cell in enumerate(cells):
                    name = f"cell-{number}.json"
                    write(output / name, cell)
                    report["evaluation_reports"].append(name)
                    report["artifacts"].append({"path": name, "sha256": auditor.sha(output / name)})
                report["counters"] = {
                    key: sum(cell["counters"][key] for cell in cells) for key in counters
                }
            elif stage["kind"] == "analysis":
                report.update(evaluation.analyze_reports(spec, cells), stage_id=stage["id"])
                report["evaluation_spec"] = spec
            write(output / "report.json", report)
            values = {
                "root": str(root),
                "out": str(output),
                "intent": str(intent_path),
                "heartbeat": str(output / "heartbeat.json"),
                "stage_id": stage["id"],
            }
            start = epoch + timedelta(seconds=3 * index)
            write(
                directory / "started.json",
                {
                    "intent_sha256": identity,
                    "stage_id": stage["id"],
                    "utc": start.isoformat(),
                    "argv": [arg.format_map(values) for arg in stage["argv"]],
                    "cap_seconds": stage["cap_seconds"],
                    "remaining_required_seconds": sum(s["cap_seconds"] for s in stages[index:])
                    + 120,
                },
            )
            write(directory / "admitted.json", {"utc": (start + timedelta(seconds=1)).isoformat()})
            write(
                directory / "supervisor.json",
                {
                    "result": {
                        "cause": None,
                        "confirmed_exit": True,
                        "termination": "natural_exit",
                        "observed_returncode": 0,
                        "elapsed_seconds": 1,
                    },
                    "limits": {
                        "rss_bytes": 8 * 1024**3,
                        "mps_driver_bytes": 24 * 1024**3,
                        "available_bytes": math.ceil(9.6 * 1024**3),
                    },
                },
            )
            write(
                directory / "completed.json",
                {
                    "intent_sha256": identity,
                    "stage_id": stage["id"],
                    "utc": (start + timedelta(seconds=2)).isoformat(),
                    "counters": report["counters"],
                    "report_sha256": auditor.sha(output / "report.json"),
                    "supervisor_sha256": auditor.sha(directory / "supervisor.json"),
                    "artifacts": {
                        str(output / item["path"]): item["sha256"] for item in report["artifacts"]
                    },
                },
            )

        def call() -> dict[str, Any]:
            return auditor.inspect(intent, intent_path, root, root / "heartbeat.json")

        require(len(call()["audited_stages"]) == len(stages) - 1, "full audit fixture")
        checks.append("complete_saved_evidence_audit")
        report_path = root / "qualification/output/report.json"
        report = runner.report_ok(report_path, intent, stages[0])
        require(runner.qualify_dose(report, intent) == payload, "runner dose/audit agreement")
        checks.append("runner_report_and_dose_agreement")
        corrupt_path = root / "qualification/supervisor.json"
        original = corrupt_path.read_text()
        damaged = json.loads(original)
        damaged["result"]["termination"] = "forced_stop"
        write(corrupt_path, damaged)
        checks.append(_reject(call, "audit_requires_natural_exit"))
        corrupt_path.write_text(original)
        original = report_path.read_text()
        report_path.write_text(original + " ")
        checks.append(_reject(call, "audit_requires_unchanged_report_hash"))
        report_path.write_text(original)
        bad = copy.deepcopy(report)
        bad["counters"]["native_frames"] = stages[0]["counter_caps"]["native_frames"] + 1
        write(report_path, bad)
        checks.append(
            _reject(lambda: runner.report_ok(report_path, intent, stages[0]), "runner_counter_cap")
        )
        report_path.write_text(original)
        bad = copy.deepcopy(report)
        bad["selected_rollouts"] -= 5
        checks.append(_reject(lambda: runner.qualify_dose(bad, intent), "nonmaximal_dose"))
    return {"checks": checks, "fixture_saved_stages": 6, "physical_operations": 0}


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def frozen_wrapper_fixture(adapter: Any, trainer: Any) -> dict[str, Any]:
    """Compare six synthetic inputs against the explicit restricted frozen network."""
    import torch

    from src.model.raster_network import RasterDuelingNetwork

    device = trainer.device
    shapes = {"tactical": (6, 9, 31, 31), "strategic": (6, 3, 25, 25), "scalars": (6, 26)}
    obs = {
        key: torch.linspace(-0.5, 0.5, math.prod(shape), device=device).reshape(shape)
        for key, shape in shapes.items()
    }
    original = {key: value.clone() for key, value in obs.items()}
    # This is the evaluation wrapper's independent mathematical reference, not
    # a second restoration or a reference to an old experiment implementation.
    with torch.random.fork_rng(devices=[]):
        frozen = RasterDuelingNetwork().to(device).eval()
    frozen.load_state_dict(trainer.network.state_dict(), strict=True)
    restricted = {key: torch.zeros_like(value) for key, value in obs.items()}
    restricted["tactical"][:, :6] = obs["tactical"][:, :6]
    restricted["scalars"][:, [0, 1, 12, 13, 14]] = obs["scalars"][:, [0, 1, 12, 13, 14]]
    with torch.no_grad():
        actual = trainer.network(obs)
        expected = frozen(restricted) * 0.1
    trainer.counters.model_forwards += 1
    trainer.counters.model_forward_rows += 6
    require(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), "frozen Q wrapper parity")
    masks = torch.eye(6, dtype=torch.bool, device=device)
    masks |= torch.roll(masks, 1, 1)
    actual_actions = actual.masked_fill(~masks, -torch.inf).argmax(1)
    expected_actions = expected.masked_fill(~masks, -torch.inf).argmax(1)
    require(torch.equal(actual_actions, expected_actions), "frozen masked action parity")
    require(all(torch.equal(obs[key], original[key]) for key in obs), "input mutated")
    return {
        "synthetic_observations": 6,
        "q_max_abs_error": float((actual - expected).abs().max()),
        "masked_action_parity": True,
        "input_immutable": True,
    }


def lifecycle_fixture(adapter: Any, trainer: Any, seed: int) -> dict[str, Any]:
    """One E16/S6 fixture with 17 physical lane frames and no optimizer updates."""
    import pickle

    import numpy as np
    import torch

    class ForceStraight:
        def random(self) -> float:
            return 0.0

        def integers(self, size: int) -> int:
            # In this fixture all three normal directions are resolved legal;
            # normal straight is entry one. Fail if the fixture is misarranged.
            require(size >= 3, "fixture normal-action menu")
            return 1

    runtime = adapter.make_runtime("s6", trainer.parent_config, seed)
    trainer.counters.episode_starts += 16
    sim = runtime.sim
    require(sim.S == 6 and sim.E == 16, "physical S6 roster")
    require(all(len(sim.get_bodies(0, slot)) > 0 for slot in range(6)), "six actual bodies")
    require(sim.get_resolved_action_mask().shape == (16, 6, 6), "all six masks")
    # Fourteen lanes are at an existing cap and remain frozen throughout.
    sim.frame[:] = adapter.TRAIN_HORIZON
    sim.frame[0] = 0
    sim.frame[1] = adapter.TRAIN_HORIZON - 16
    sim.bodies[0, 0, 0] = (0, 0)
    sim.direction[0, 0] = 3
    sim.bodies[0, 1, 0] = (0, 1)
    # Lane one's hero has at least 16 unblocked straight moves. Opponents are
    # parked on another row; one respawns during Watch preparation.
    width = trainer.parent_config["game_width"] // trainer.parent_config["segment_size"]
    height = trainer.parent_config["game_height"] // trainer.parent_config["segment_size"]
    require(width > 20 and height > 20, "fixture arena too small")
    sim.bodies[1, 0, 0] = (2, height // 2)
    sim.direction[1, 0] = 1
    for slot in range(1, 6):
        sim.bodies[1, slot, 0] = (width - 2 - 2 * slot, 2)
        sim.direction[1, slot] = 3
    sim.alive[0, 5] = False
    sim.respawn_timer[0, 5] = 1
    sim.alive[1, 1:] = False
    sim.respawn_timer[1, 1:] = 10000
    runtime.action_rngs = [ForceStraight() for _ in range(16)]
    frozen_names = (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "frame",
    )
    inactive_before = {key: getattr(sim, key)[2:].copy() for key in frozen_names}
    rng_before = pickle.dumps(sim._rngs[2:])
    # A disposable prepared preview must leave even RNG and food untouched.
    before = pickle.dumps(sim)
    preview, preview_mask = adapter.prepared_preview(
        sim, np.arange(16) < 2, trainer.parent_config, trainer.device
    )
    trainer.counters.preview_calls += 1
    require(pickle.dumps(sim) == before, "prepared preview mutated live simulator")
    require(preview_mask.shape == (16, 6), "one hero mask per physical world")
    before_counts = trainer.counters.as_dict()
    roll = adapter.collect_rollout(trainer, runtime)
    require(roll["valid"].shape == (16, 16, 1), "single hero tensor layout")
    require(
        bool(roll["dones"][0, 0, 0]) and int(roll["valid"][:, 0].sum()) == 1,
        "hero death must terminate immediately",
    )
    require(not bool(sim.alive[0, 0]), "dead hero respawned")
    require(
        bool(sim.alive[1, 0]) and int(sim.frame[1]) == adapter.TRAIN_HORIZON,
        "cap fixture hero must survive to cap",
    )
    require(
        not bool(roll["dones"][15, 1, 0]) and int(roll["valid"][:, 1].sum()) == 16,
        "live cap marked terminal or lost valid rows",
    )
    require(
        bool(sim.alive[0, 5]) or int(sim.respawn_timer[0, 5]) > 1,
        "opponent did not respawn during Watch preparation",
    )
    require(
        all(np.array_equal(getattr(sim, key)[2:], value) for key, value in inactive_before.items())
        and pickle.dumps(sim._rngs[2:]) == rng_before,
        "inactive lane changed",
    )
    require(not runtime.episode_ids.any(), "collector reset inside rollout")
    # Compare the actual final cap observation against an independently prepared
    # disposable successor; death has no legal bootstrap obligation.
    expected_obs, expected_mask = adapter.prepared_preview(
        sim, sim.get_alive()[:, 0], trainer.parent_config, trainer.device
    )
    trainer.counters.preview_calls += 1
    require(
        all(torch.equal(roll["final_obs"][key][:, 0], expected_obs[key]) for key in expected_obs),
        "live cap successor phase differs",
    )
    require(torch.equal(roll["next_mask"][-1, :, 0], expected_mask), "cap successor mask differs")
    targets = trainer._compute_targets(roll)
    require(
        float(targets[0, 0, 0])
        == float(torch.as_tensor(roll["rewards"][0, 0, 0], dtype=torch.float32)),
        "death bootstrapped",
    )
    with torch.no_grad():
        q = trainer.network(expected_obs)
    boot = q[1].masked_fill(~expected_mask[1], -torch.inf).max()
    expected_target = float(roll["rewards"][-1, 1, 0]) + trainer.cfg.gamma * boot
    require(
        torch.allclose(targets[-1, 1, 0], expected_target, rtol=1e-6, atol=1e-7),
        "live cap lost masked bootstrap",
    )
    delta = {key: value - before_counts[key] for key, value in trainer.counters.as_dict().items()}
    require(
        delta["native_frames"] == 17 and delta["hero_transitions"] == 17,
        "physical fixture accounting",
    )
    require(delta["optimizer_updates"] == 0, "fixture performed optimization")
    return {
        "checks": [
            "six_actual_bodies_masks_one_hero",
            "death_terminal",
            "live_cap_bootstrap",
            "watch_opponent_respawn",
            "preview_immutable",
            "inactive_lane_frozen",
            "no_midrollout_reset",
            "prepared_successor_parity",
        ],
        "counters": delta,
        "episode_starts": 16,
    }


def run_qualification(intent: dict[str, Any], out: Path, heartbeat: Any) -> dict[str, Any]:
    """Run the one admitted qualification; a failed fixture or fit ends the package."""
    from research.task_aligned_20260924 import adapter, audit, evaluation, runner

    started = time.monotonic()
    require(intent.get("admitted") is True, "qualification requires admitted intent")
    require(set(intent["parents"]) == {str(seed) for seed in LINEAGES}, "all three parent bindings")
    device = intent["device"]
    seed = intent["qualification_seed"]
    require(type(seed) is int and seed not in LINEAGES, "explicit disposable runtime seed")
    fixture_spec = copy.deepcopy(intent["evaluation"])
    # These are algebraic schema records, never generated training worlds.
    held = {value for seeds in fixture_spec["world_seeds"].values() for value in seeds}
    fixture_training_seed = next(value for value in range(1000) if value not in held)
    fixture_spec["training_world_seeds"] = [fixture_training_seed]
    heartbeat.progress = {"phase": "qualification_schema_fixtures"}
    schema = reporter_fixtures(evaluation, fixture_spec, out)
    operations = operations_fixtures(runner, audit, evaluation, fixture_spec, out)
    trainer = None
    source = None
    count_objects = []
    restorations = []
    setup_seconds = []
    try:
        for lineage in LINEAGES:
            heartbeat.progress = {"phase": "qualification_parent_restore", "lineage": lineage}
            _sync(device)
            load_started = time.monotonic()
            bound = intent["parents"][str(lineage)]
            parent_spec = adapter.ParentSpec(
                lineage,
                Path(bound["report_path"]),
                bound["report_sha256"],
                Path(bound["checkpoint_path"]),
                bound["checkpoint_sha256"],
            )
            parent = adapter.load_parent(parent_spec, device)
            current = adapter.build_trainer(
                parent, seed, device, rollouts=PROBE_ROLLOUTS, initialize_runtimes=False
            )
            count_objects.append(current.counters)
            expected_model = parent.metadata["network_state_digest"]
            expected_adam = parent.metadata["optimizer_state_digest"]
            require(
                adapter.state_digest(current.network.state_dict()) == expected_model,
                "restored network changed",
            )
            require(
                adapter.state_digest(current.optimizer.state_dict()) == expected_adam,
                "restored Adam changed",
            )
            require(
                adapter.state_digest(current._last_known_good_state["network"]) == expected_model
                and adapter.state_digest(current._last_known_good_state["optimizer"])
                == expected_adam,
                "numeric recovery snapshot is not the restored parent",
            )
            require(
                current.update_idx == 4608 and current.agent_steps == parent.native["agent_steps"],
                "restored learner clocks differ",
            )
            parity = frozen_wrapper_fixture(adapter, current)
            _sync(device)
            setup_seconds.append(time.monotonic() - load_started)
            restorations.append(
                {
                    "lineage": lineage,
                    "checkpoint_sha256": bound["checkpoint_sha256"],
                    "network_state_digest": expected_model,
                    "optimizer_state_digest": expected_adam,
                    "update_counter": current.update_idx,
                    "agent_steps": current.agent_steps,
                    "full_state_restored": True,
                    "recovery_snapshot_restored": True,
                    "wrapper": parity,
                }
            )
            if lineage == LINEAGES[0]:
                trainer, source = current, parent
            else:
                current.close()
                del current, parent
        require(trainer is not None and source is not None, "missing disposable source 6101")
        import torch

        prediction = torch.tensor([2.0, -2.0], requires_grad=True)
        loss = trainer._sgd.__func__.__globals__["F"].smooth_l1_loss(
            prediction, torch.zeros_like(prediction)
        )
        loss.backward()
        require(
            float(loss) == 2.0 and torch.equal(prediction.grad, torch.tensor([1.0, -1.0])),
            "inherited half-MSE value/gradient differs",
        )
        heartbeat.progress = {"phase": "qualification_physical_lifecycle"}
        lifecycle = lifecycle_fixture(adapter, trainer, seed)
        # A single round trip exercises the new checkpoint boundary, before any
        # timing update. Its fourth deserialization is counted separately.
        with tempfile.TemporaryDirectory(prefix="qualification-roundtrip-", dir=out) as temporary:
            path = Path(temporary) / "discarded-state.pth"
            _sync(device)
            saving = time.monotonic()
            trainer.save_checkpoint(path)
            _sync(device)
            save_seconds = time.monotonic() - saving
            restored = adapter.load_candidate(
                path, adapter.file_sha256(path), device, allow_partial=True
            )
            count_objects.append(restored.counters)
            require(
                adapter.state_digest(restored.native["dqn_state_dict"])
                == adapter.state_digest(trainer.network.state_dict()),
                "roundtrip network mismatch",
            )
            require(
                adapter.state_digest(restored.native["optimizer_state_dict"])
                == adapter.state_digest(trainer.optimizer.state_dict()),
                "roundtrip Adam mismatch",
            )
            require(
                restored.native["update_counter"] == trainer.update_idx
                and restored.native["agent_steps"] == trainer.agent_steps,
                "roundtrip clocks mismatch",
            )
            del restored
        _sync(device)
        initialize_started = time.monotonic()
        trainer.initialize_runtimes()
        _sync(device)
        initialize_seconds = time.monotonic() - initialize_started
        before_probe = trainer.counters.as_dict()
        model_identity, optimizer_identity = id(trainer.network), id(trainer.optimizer)
        cycle_seconds = []
        probe_rows = []
        for cycle in range(PROBE_CYCLES):
            _sync(device)
            cycle_started = time.monotonic()
            for profile in adapter.SCHEDULE:
                heartbeat.progress = {
                    "phase": "qualification_probe",
                    "cycle": cycle,
                    "profile": profile,
                    "rollouts": trainer.new_rollouts,
                }
                row = trainer.update()
                require(row["profile"] == profile, "mixed-runtime cycle changed")
                require(
                    id(trainer.network) == model_identity
                    and id(trainer.optimizer) == optimizer_identity,
                    "task switch replaced shared learner",
                )
                probe_rows.append(row)
            _sync(device)
            cycle_seconds.append(time.monotonic() - cycle_started)
        probe_counts = {
            key: value - before_probe[key] for key, value in trainer.counters.as_dict().items()
        }
        require(
            probe_counts["completed_rollouts"] == 150 and probe_counts["optimizer_updates"] == 150,
            "probe must perform exactly 150 complete one-epoch updates",
        )
        require(
            0 < probe_counts["hero_transitions"] <= 38400
            and probe_counts["sampled_transition_draws"] == probe_counts["hero_transitions"],
            "probe exact coverage/row cap",
        )
        require(
            trainer.update_idx == 4758
            and all(float(state["step"]) == 4758 for state in trainer.optimizer.state.values()),
            "shared learner/Adam clocks did not advance exactly150",
        )
        fixed_seconds = max(setup_seconds) + initialize_seconds + 3 * save_seconds
        selected = select_dose(cycle_seconds, fixed_seconds)
        # The serving probe uses the still-frozen parent's tensor copy, not this
        # disposable learner endpoint, and runs only the three fixed profiles.
        heartbeat.progress = {"phase": "qualification_evaluation_throughput"}
        evaluation_probe = adapter.qualification_evaluation_probe(
            source, intent["evaluation"], seed, out, heartbeat
        )
        count_objects.append(evaluation_probe.pop("physical_counters"))
        require(
            evaluation_probe["contract_sha256"] == intent["evaluation"]["contract_sha256"],
            "serving contract differs from frozen evaluation template",
        )
        require(evaluation_probe["fits"] is True, "fixed evaluation stages do not fit")
        total = {
            key: sum(getattr(count, key) for count in count_objects)
            for key in adapter.PhysicalCounters.__dataclass_fields__
        }
        stage = next(row for row in intent["stages"] if row["kind"] == "qualification")
        for key, cap in {**stage["counter_caps"], **intent["qualification_limits"]}.items():
            require(
                type(total[key]) is int and 0 <= total[key] <= cap,
                f"qualification physical cap: {key}",
            )
        payload = {
            key: selected[key]
            for key in ("selected_rollouts", "conservative_cycle_seconds", "fixed_overhead_seconds")
        }
        count = selected["selected_rollouts"]
        payload.update(
            shared_probe_lineage=LINEAGES[0],
            profile_rollouts={"solo": count // 5, "S2": count * 2 // 5, "S6": count * 2 // 5},
        )
        payload["dose_sha256"] = evaluation.canonical_hash(payload)
        detail = {
            "parent_restorations": restorations,
            "schema_fixtures": schema,
            "operations_fixtures": operations,
            "lifecycle_fixture": lifecycle,
            "probe_source_lineage": LINEAGES[0],
            "probe_cycles": 30,
            "probe_rollouts": 150,
            "probe_counters": probe_counts,
            "probe_profile_rollouts": {"solo": 30, "S2": 60, "S6": 60},
            "probe_cycle_seconds": cycle_seconds,
            "probe_telemetry": probe_rows,
            "evaluation_probe": evaluation_probe,
            "counters": total,
            "qualification_state_discarded": True,
            "roundtrip_full_state_pass": True,
            "half_mse_value_gradient_pass": True,
            "shared_model_adam_clocks_pass": True,
            "fixed_overhead": {
                "max_parent_setup_seconds": max(setup_seconds),
                "runtime_initialization_seconds": initialize_seconds,
                "one_save_seconds": save_seconds,
                "reserved_science_saves": 3,
            },
            "elapsed_seconds": time.monotonic() - started,
            **selected,
            **payload,
        }
        runner.durable(out / "qualification-evidence.json", detail)
        require(detail["elapsed_seconds"] <= QUALIFICATION_SECONDS, "qualification wall cap")
        require(selected["fixed_science_fits"], "RESOURCE_INFEASIBLE: fewer than 7815 rollouts fit")
        return {
            **selected,
            **payload,
            "runtime_contracts_pass": True,
            "evaluation_fits": True,
            "counters": total,
            "artifacts": [
                {
                    "path": "qualification-evidence.json",
                    "sha256": runner.digest(out / "qualification-evidence.json"),
                },
                {
                    "path": "evaluation-throughput.json",
                    "sha256": runner.digest(out / "evaluation-throughput.json"),
                },
                *[
                    {
                        "path": f"serialization-fixture-{index}.json",
                        "sha256": runner.digest(out / f"serialization-fixture-{index}.json"),
                    }
                    for index in range(3)
                ],
            ],
        }
    finally:
        if trainer is not None:
            trainer.close()
