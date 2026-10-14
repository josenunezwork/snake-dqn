"""One admitted continuation stage; no informal smoke, retries, or old-stage mains."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import statistics
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.task_aligned_20260924 import adapter
from research.task_aligned_continuation_20260924 import evaluation, learner, runner

read = runner.read
save = runner.durable


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_caps(counts: dict[str, int], caps: dict[str, int]) -> None:
    for key, cap in caps.items():
        check(type(counts.get(key)) is int and 0 <= counts[key] <= cap, f"Physical cap: {key}")


class Heartbeat:
    """Keep supervision live during numerical work and serialization."""

    def __init__(self, path: Path, device: str) -> None:
        self.path, self.device = path, device
        self.progress: dict[str, Any] = {"phase": "starting"}
        self.numerical_ready = False
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                driver = None
                if self.device == "mps" and self.numerical_ready:
                    driver = int(sys.modules["torch"].mps.driver_allocated_memory())
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "monotonic": time.monotonic(),
                            "mps_driver_bytes": driver,
                            **self.progress,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                temporary.replace(self.path)
            except BaseException as exc:
                self.error = exc
                return
            self.stop.wait(2)

    def __enter__(self) -> Heartbeat:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if not args[0] and self.error:
            raise RuntimeError("Heartbeat writer failed") from self.error


def numerical_setup(device: str) -> None:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    check(device in {"cpu", "mps"}, "Undeclared numerical device")
    if device == "mps":
        check(torch.backends.mps.is_available(), "Frozen MPS device unavailable")
        torch.mps.set_per_process_memory_fraction(
            min(1.0, 24 * 1024**3 / torch.mps.recommended_max_memory())
        )


def synthetic_fixtures(intent: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """Four small new-wrapper records; no games, model calls, or old fixture drivers."""
    spec = copy.deepcopy(intent["evaluation"])
    held = {seed for rows in spec["world_seeds"].values() for seed in rows}
    dummy_seed = next(seed for seed in range(1000) if seed not in held)
    spec["training_world_seeds"] = [dummy_seed]
    spec["excluded_world_seeds"] = [dummy_seed]
    spec["policy_sha256"] = {
        "parent": learner.SOURCE_SHA256,
        "candidate": evaluation.canonical_hash({"synthetic_child": True}),
    }
    evaluation.validate_spec(spec)
    results = []

    def reject(call: Any) -> str:
        try:
            call()
        except ValueError as exc:
            return str(exc)
        raise RuntimeError("Synthetic malformed record was accepted")

    def record(index: int, name: str, evidence: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "synthetic_fixture_only": True,
            "physical_counters": dict.fromkeys(runner.COUNTERS, 0),
            "name": name,
            "status": "PASS",
            **evidence,
        }
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode()
        check(len(encoded) + 1 <= 1024**2, "Synthetic fixture exceeds1MiB")
        path = out / f"synthetic-fixture-{index}.json"
        save(path, payload)
        results.append(
            {
                "name": name,
                "path": path.name,
                "bytes": len(encoded) + 1,
                "sha256": runner.digest(path),
                "pass": True,
            }
        )

    old_scope = copy.deepcopy(spec)
    old_scope["lineages"] = [2026096101, 2026096102, 2026096103]
    wrong_endpoint = copy.deepcopy(spec)
    wrong_endpoint["candidate_update"] = 35613
    record(
        1,
        "new_scope_and_endpoint",
        {
            "old_scope_rejected": reject(lambda: evaluation.validate_spec(old_scope)),
            "source_as_child_rejected": reject(lambda: evaluation.validate_spec(wrong_endpoint)),
        },
    )

    calls = []

    def forbidden(requests: Any) -> Any:
        calls.append(requests)
        raise RuntimeError("Rejected identity reached synthetic callback")

    zero = dict.fromkeys(evaluation.COUNTERS, 0)
    runtime = evaluation.EvaluationRuntime(
        lambda request: forbidden([request]),
        spec["contract_sha256"],
        zero.copy(),
        run_batch=forbidden,
    )
    bad = {
        "role": "candidate",
        "lineage": learner.SOURCE_SEED,
        "checkpoint_sha256": learner.SOURCE_SHA256,
    }
    error = reject(
        lambda: evaluation.evaluate_cell(spec, bad, "solo", runtime, out / "must-not-exist.json")
    )
    check(
        not calls and not (out / "must-not-exist.json").exists(),
        "Policy preflight leaked to callback",
    )
    record(2, "sealed_identity_precedes_callback", {"rejection": error, "callback_calls": 0})

    def cell(role: str, identity: Any, profile: str, mass: float) -> dict[str, Any]:
        policy = {"role": role, "lineage": None if role == "anchor" else learner.SOURCE_SEED}
        if role == "anchor":
            policy["name"] = identity
        else:
            policy["checkpoint_sha256"] = spec["policy_sha256"][role]
        cases = []
        for seed in spec["world_seeds"][profile]:
            desc = {
                "world_seed": seed,
                "hero": {"synthetic": True},
                "food": [],
                "opponents": [{"synthetic": True} for _ in evaluation.PROFILES[profile]],
                "rng_sha256": evaluation.canonical_hash({"synthetic": seed}),
            }
            frames = [
                evaluation.FrameMetrics(True, mass, 1, False, False),
                evaluation.FrameMetrics(False, 0.0, 0, False, False),
            ]
            counts = {**zero, "native_frames": 2, "hero_transitions": 2}
            request = evaluation.EpisodeRequest(
                seed, profile, tuple(evaluation.PROFILES[profile]), 3000
            )
            cases.append(
                evaluation.primitives._reduce_episode(
                    request, evaluation.EpisodeTrace(frames, desc, counts)
                )
            )
        initial = {**zero, "checkpoint_loads": int(role != "anchor" and profile == "solo")}
        shared = {**zero, "model_forwards": 2 if role != "anchor" else 0}
        report = {
            "schema_version": 1,
            "study_id": evaluation.STUDY_ID,
            "stage_id": "synthetic",
            "status": "PASS",
            "kind": "policy_evaluation",
            "spec_sha256": evaluation.canonical_hash(spec),
            "contract_sha256": spec["contract_sha256"],
            "policy": policy,
            "profile": profile,
            "cases": cases,
            "cases_sha256": evaluation.canonical_hash(cases),
            "initial_counters": initial,
            "shared_counters": shared,
            "counters": {
                key: initial[key] + shared[key] + sum(row["counters"][key] for row in cases)
                for key in zero
            },
            "summary": {
                key: statistics.mean(row[key] for row in cases) for key in evaluation.METRICS
            },
            "artifacts": [],
            "promotion_eligible": False,
        }
        evaluation.validate_report(spec, report)
        return report

    positive = cell("parent", learner.SOURCE_SEED, "solo", 30.0)
    check(positive["cases"][0]["mass_integral"] == 30 / 3000, "Wrong total-horizon denominator")
    broken = copy.deepcopy(positive)
    broken["cases"][0]["mass_integral"] = 30 / 2
    broken["cases_sha256"] = evaluation.canonical_hash(broken["cases"])
    record(
        3,
        "nonzero_death_case_arithmetic",
        {
            "valid_cell": positive,
            "corruption_rejected": reject(lambda: evaluation.validate_report(spec, broken)),
        },
    )

    cells = [
        cell(role, identity, profile, 15.0 if role == "candidate" and profile != "solo" else 30.0)
        for role, identity in (
            ("parent", learner.SOURCE_SEED),
            ("candidate", learner.SOURCE_SEED),
            ("anchor", "greedy_food"),
            ("anchor", "random_safe"),
        )
        for profile in evaluation.PROFILES
    ]
    decision = evaluation.analyze_reports(spec, cells)
    check(
        decision["decision"] == "NEGATIVE_CONTINUATION_SCREEN"
        and decision["baseline_exposure_pass"] is False,
        "New harm precedence differs",
    )
    record(
        4,
        "negative_harm_precedes_absent_exposure",
        {
            "spec": spec,
            "cell_hashes": [evaluation.canonical_hash(row) for row in cells],
            "synthetic_case_count": 384,
            "decision": decision,
            "missing_cell_rejected": reject(lambda: evaluation.analyze_reports(spec, cells[:-1])),
        },
    )
    return results


def check_source_config(source: Any, intent: dict[str, Any]) -> str:
    expected = intent["evaluation"]["source_config"]
    check(source.config == expected, "Loaded6102 source config differs from frozen intent")
    check(
        evaluation.canonical_hash(adapter.evaluation_contract(source.config))
        == intent["evaluation"]["contract_sha256"],
        "Loaded6102 serving contract differs",
    )
    return evaluation.canonical_hash(source.config)


def qualification(
    intent: dict[str, Any], stage: dict[str, Any], out: Path, heartbeat: Heartbeat
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    heartbeat.progress = {"phase": "new_synthetic_schema_fixtures"}
    fixtures = synthetic_fixtures(intent, out)
    heartbeat.progress = {"phase": "source_full_state_restore"}
    source = learner.load_source(device=stage["device"])
    source_config_sha256 = check_source_config(source, intent)
    trainer = learner.build_continuation(
        source, intent["qualification_seed"], stage["device"], qualification=True
    )
    try:
        restoration = trainer.restoration_evidence()
        check(set(trainer.runtimes) == set(adapter.ROSTERS), "Physical task roster differs")
        for name, runtime in trainer.runtimes.items():
            sim = runtime.sim
            check(
                sim.E == 16
                and sim.S == adapter.ROSTERS[name]
                and not sim.train_mode
                and sim.allow_respawn
                and sim.cfg.body_storage_capacity == 2448,
                "Qualified physical runtime configuration differs",
            )
        cycles = []
        with (out / "disposable-training.jsonl").open("x") as log:
            for cycle in range(2):
                torch.mps.synchronize()
                began = time.monotonic()
                for index in range(5):
                    row = trainer.update()
                    check_caps(trainer.counters.as_dict(), stage["counter_caps"])
                    log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    heartbeat.progress = {
                        "phase": "disposable_update",
                        "completed_rollouts": row["new_rollouts"],
                        "counters": trainer.counters.as_dict(),
                    }
                torch.mps.synchronize()
                cycles.append(time.monotonic() - began)
            log.flush()
            os.fsync(log.fileno())
        state = trainer.snapshot_state()
        path = out / "disposable-state.pth"
        heartbeat.progress = {"phase": "disposable_roundtrip"}
        with path.open("xb") as handle:
            torch.save(state, handle)
        loaded = learner.load_qualification_snapshot(path, runner.digest(path), device="cpu")
        compare_keys = (
            "update_counter",
            "agent_steps",
            "additional_agent_steps",
            "source_checkpoint",
            "action_collapse_streak",
            "action_collapse_evidence_samples",
            "action_collapse_raw_action_mode",
        )
        check(
            all(loaded.native[key] == state[key] for key in compare_keys),
            "Roundtrip clocks/tripwire differ",
        )
        check(
            adapter.state_digest(loaded.model.state_dict())
            == adapter.state_digest(state["dqn_state_dict"]),
            "Roundtrip network values differ",
        )
        check(
            adapter.state_digest(loaded.native["optimizer_state_dict"])
            == adapter.state_digest(state["optimizer_state_dict"]),
            "Roundtrip full Adam values differ",
        )
        counts = {
            key: trainer.counters.as_dict()[key] + loaded.counters.as_dict()[key]
            for key in trainer.counters.as_dict()
        }
        check_caps(counts, stage["counter_caps"])
        noncycle_overhead = time.monotonic() - started - sum(cycles)
        timing_rule = "2*(noncycle_overhead+startup_cycle+6201*repeated_cycle)"
        forecast = 2 * (noncycle_overhead + cycles[0] + 6201 * cycles[1])
        fits = forecast <= 25200
        evidence = {
            "source_restoration": restoration,
            "roundtrip_compared_fields": list(compare_keys),
            "roundtrip_full_adam": True,
            "cycle_seconds": cycles,
            "timing_rule": timing_rule,
            "startup_cycle_seconds": cycles[0],
            "repeated_cycle_seconds": cycles[1],
            "noncycle_overhead_seconds": noncycle_overhead,
            "fixed_overhead_seconds": noncycle_overhead,
            "synthetic_fixtures": fixtures,
            "discarded_updates": 10,
            "forecast_seconds": forecast,
        }
        save(out / "qualification-evidence.json", evidence)
        return {
            "kind": "qualification",
            "full_state_restore_pass": True,
            "roundtrip_pass": True,
            "runtime_contracts_pass": True,
            "fixed_science_fits": fits,
            "forecast_per_lineage_seconds": forecast,
            "estimated_science_seconds": forecast,
            "timing_rule": timing_rule,
            "startup_cycle_seconds": cycles[0],
            "repeated_cycle_seconds": cycles[1],
            "noncycle_overhead_seconds": noncycle_overhead,
            "counters": counts,
            "synthetic_fixture_count": 4,
            "source_checkpoint": learner.source_binding(),
            "source_config_sha256": source_config_sha256,
            "promotion_eligible": False,
        }
    finally:
        trainer.close()


def train(
    intent: dict[str, Any], stage: dict[str, Any], root: Path, out: Path, heartbeat: Heartbeat
) -> dict[str, Any]:
    check(
        read(root / "qualification/output/report.json")["fixed_science_fits"] is True,
        "Fixed-dose forecast not qualified",
    )
    source = learner.load_source(device=stage["device"])
    source_config_sha256 = check_source_config(source, intent)
    trainer = learner.build_continuation(
        source, intent["runtime_seeds"][str(learner.SOURCE_SEED)], device=stage["device"]
    )
    started = time.monotonic()
    try:
        with (out / "training.jsonl").open("x") as log:
            for index in range(learner.SCIENCE_ROLLOUTS):
                row = trainer.update()
                check_caps(trainer.counters.as_dict(), stage["counter_caps"])
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                heartbeat.progress = {
                    "phase": "training",
                    "lineage": learner.SOURCE_SEED,
                    "completed_rollouts": index + 1,
                    "selected_rollouts": learner.SCIENCE_ROLLOUTS,
                    "counters": trainer.counters.as_dict(),
                }
                if (index + 1) % 50 == 0:
                    log.flush()
            log.flush()
            os.fsync(log.fileno())
        path = out / "checkpoint_rollout_31005.pth"
        heartbeat.progress = {"phase": "saving_final_child", "counters": trainer.counters.as_dict()}
        trainer.save_checkpoint(path)
        endpoint = {
            "completed_rollouts": 31005,
            "global_update": 66618,
            "path": path.name,
            "sha256": runner.digest(path),
        }
        worlds = {
            name: [
                [
                    adapter._seed(runtime.run_seed, f"{name}/world/{env}/{episode}")
                    for episode in range(int(runtime.episode_ids[env]) + 1)
                ]
                for env in range(16)
            ]
            for name, runtime in trainer.runtimes.items()
        }
        save(
            out / "training-world-seeds.json",
            {"worlds": worlds, "scope": "this continuation's actual episode namespaces only"},
        )
        return {
            "kind": "training",
            "lineage": learner.SOURCE_SEED,
            "source_checkpoint": learner.source_binding(),
            "source_config_sha256": source_config_sha256,
            "task_contract": adapter.task_contract(),
            "selected_rollouts": 31005,
            "completed_rollouts": trainer.new_rollouts,
            "scheduled_hero_slots": 7937280,
            "starting_update": 35613,
            "final_update": trainer.update_idx,
            "starting_agent_steps": trainer.starting_agent_steps,
            "final_agent_steps": trainer.agent_steps,
            "profile_rollouts": {
                "solo": trainer.per_profile["solo"]["rollouts"],
                "S2": trainer.per_profile["s2"]["rollouts"],
                "S6": trainer.per_profile["s6"]["rollouts"],
            },
            "per_profile": copy.deepcopy(trainer.per_profile),
            "snapshots": [endpoint],
            "final_checkpoint": endpoint,
            "counters": trainer.counters.as_dict(),
            "elapsed_seconds": time.monotonic() - started,
            "promotion_eligible": False,
        }
    finally:
        trainer.close()


def eval_spec(intent: dict[str, Any], root: Path) -> dict[str, Any]:
    out = root / "train_2026096102/output"
    endpoint = read(out / "report.json")["final_checkpoint"]
    ledger = read(out / "training-world-seeds.json")["worlds"]
    spec = copy.deepcopy(intent["evaluation"])
    spec["training_world_seeds"] = sorted(
        {seed for lanes in ledger.values() for episodes in lanes for seed in episodes}
    )
    spec["policy_sha256"] = {"parent": learner.SOURCE_SHA256, "candidate": endpoint["sha256"]}
    evaluation.validate_spec(spec)
    return spec


def evaluate(
    intent: dict[str, Any], stage: dict[str, Any], root: Path, out: Path, heartbeat: Heartbeat
) -> dict[str, Any]:
    spec = eval_spec(intent, root)
    check(stage["device"] == "cpu", "Serving device must remain CPU")
    role = stage["role"]
    endpoint = read(root / "train_2026096102/output/report.json")["final_checkpoint"]
    policies = (
        [
            {"role": "anchor", "name": name, "lineage": None}
            for name in ("greedy_food", "random_safe")
        ]
        if role == "anchor"
        else [
            {
                "role": role,
                "lineage": learner.SOURCE_SEED,
                "checkpoint_sha256": spec["policy_sha256"][role],
            }
        ]
    )
    save(out / "evaluation-spec.json", spec)
    counts = dict.fromkeys(adapter.PhysicalCounters.__dataclass_fields__, 0)
    files = []
    for policy in policies:
        model, anchor = None, None
        if role == "parent":
            loaded = learner.load_source(device="cpu")
        elif role == "candidate":
            loaded = learner.load_endpoint(
                root / "train_2026096102/output" / endpoint["path"],
                endpoint["sha256"],
                device="cpu",
            )
        elif role == "anchor":
            anchor = policy["name"]
            loaded = None
        else:
            raise ValueError("Unknown evaluation role")
        if loaded is None:
            config, meter = spec["source_config"], adapter.PhysicalCounters()
        else:
            model, config, meter = loaded.model, loaded.config, loaded.counters
            check(config == spec["source_config"], "Serving source config differs")
            check(
                loaded.metadata["sha256"] == policy["checkpoint_sha256"], "Serving identity differs"
            )
        contract = evaluation.canonical_hash(adapter.evaluation_contract(config))
        check(contract == spec["contract_sha256"], "Serving contract differs")
        runtime = evaluation.EvaluationRuntime(
            run_episode=lambda request: adapter.run_episode(
                model, config, request, anchor=anchor, device="cpu"
            ),
            run_batch=lambda requests: adapter.run_batch_episodes(
                model, config, requests, anchor=anchor, device="cpu", meter=meter
            ),
            contract_sha256=contract,
            initial_counters={key: int(getattr(meter, key)) for key in evaluation.COUNTERS},
        )
        observed_core = dict.fromkeys(evaluation.COUNTERS, 0)
        for index, profile in enumerate(evaluation.PROFILES):
            heartbeat.progress = {
                "phase": "evaluation",
                "role": role,
                "identity": policy.get("name", learner.SOURCE_SEED),
                "profile": profile,
            }
            if index:
                runtime.initial_counters = dict.fromkeys(evaluation.COUNTERS, 0)
            path = out / f"{policy.get('name', str(learner.SOURCE_SEED))}-{profile}.json"
            cell = evaluation.evaluate_cell(spec, policy, profile, runtime, path)
            files.append(path.name)
            for key in observed_core:
                observed_core[key] += cell["counters"][key]
            check(
                all(observed_core[key] == getattr(meter, key) for key in observed_core),
                "Evaluation saved versus physical core counters differ",
            )
            check_caps(
                {key: counts[key] + value for key, value in meter.as_dict().items()},
                stage["counter_caps"],
            )
        for key, value in meter.as_dict().items():
            counts[key] += value
        del runtime, model, loaded
        gc.collect()
    return {
        "kind": "evaluation_batch",
        "role": role,
        "lineage": stage.get("lineage"),
        "evaluation_reports": files,
        "evaluation_spec": "evaluation-spec.json",
        "counters": counts,
        "promotion_eligible": False,
    }


def analyze(intent: dict[str, Any], root: Path, out: Path) -> dict[str, Any]:
    reports, spec = [], None
    for stage in intent["stages"]:
        if stage["kind"] != "evaluation":
            continue
        directory = root / stage["id"] / "output"
        report = read(directory / "report.json")
        current = read(directory / report["evaluation_spec"])
        check(spec is None or spec == current, "Evaluation spec changed across jobs")
        spec = current
        reports.extend(read(directory / name) for name in report["evaluation_reports"])
    result = evaluation.analyze_reports(spec, reports)
    save(out / "decision.json", result)
    from research.task_aligned_continuation_20260924.reporting import build_presentation

    presentation = out / "presentation"
    files = build_presentation(
        spec,
        {**result, "cell_reports": reports},
        root / "train_2026096102/output/training.jsonl",
        presentation,
    )
    result["presentation_files"] = [str(path.relative_to(out)) for path in files]
    result["evaluation_spec"] = spec
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intent", "root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--stage-id", required=True)
    args = parser.parse_args()
    intent = read(args.intent)
    stage = next(row for row in intent["stages"] if row["id"] == args.stage_id)
    check(
        intent["study_id"] == learner.STUDY_ID
        and intent["source_checkpoint"] == learner.source_binding(),
        "Continuation study/source binding differs",
    )
    out = args.out.resolve()
    check(out == (args.root / stage["id"] / "output").resolve(), "Stage output boundary differs")
    check(not (out / "report.json").exists(), "Stage already has a report")
    heartbeat_path = args.heartbeat or out / "heartbeat.json"
    with Heartbeat(heartbeat_path, stage.get("device", "cpu")) as heartbeat:
        try:
            if stage["kind"] in {"qualification", "training", "evaluation"}:
                numerical_setup(stage["device"])
                heartbeat.numerical_ready = True
            if stage["kind"] == "qualification":
                result = qualification(intent, stage, out, heartbeat)
            elif stage["kind"] == "training":
                result = train(intent, stage, args.root, out, heartbeat)
            elif stage["kind"] == "evaluation":
                result = evaluate(intent, stage, args.root, out, heartbeat)
            elif stage["kind"] == "analysis":
                result = analyze(intent, args.root, out)
            else:
                raise ValueError("Independent audit owns its own entrypoint")
            result.update(
                schema_version=1, study_id=learner.STUDY_ID, stage_id=stage["id"], status="PASS"
            )
            check_caps(result["counters"], stage["counter_caps"])
            result["artifacts"] = [
                {"path": str(path.relative_to(out)), "sha256": runner.digest(path)}
                for path in sorted(out.rglob("*"))
                if path.is_file() and path.name not in {"heartbeat.json", "heartbeat.tmp"}
            ]
            save(out / "report.json", result)
        except BaseException as exc:
            save(
                out / "failure.json",
                {
                    "status": "INVALID_STOP",
                    "stage_id": stage["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_authorized": False,
                },
            )
            raise


if __name__ == "__main__":
    main()
