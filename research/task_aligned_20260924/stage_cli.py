"""Single admitted-stage entry point; never launch this as an informal smoke."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.task_aligned_20260924 import adapter, evaluation, runner


def read(path: Path) -> dict[str, Any]:
    return runner.read(path)


def save(path: Path, value: dict[str, Any]) -> None:
    runner.durable(path, value)


def core_counts(values: dict[str, int]) -> dict[str, int]:
    return {name: int(values.get(name, 0)) for name in runner.COUNTERS}


class Heartbeat:
    """Publish physical-resource observations even during slow artifact writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stop = threading.Event()
        self.progress: dict[str, Any] = {"phase": "starting"}
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                torch = sys.modules.get("torch")
                driver = None
                if torch is not None and torch.backends.mps.is_available():
                    driver = int(torch.mps.driver_allocated_memory())
                payload = {
                    "monotonic": time.monotonic(),
                    "mps_driver_bytes": driver,
                    **self.progress,
                }
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
                temporary.replace(self.path)
            except BaseException as exc:
                self.error = exc
                return
            self.stop.wait(2.0)

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if not args[0] and self.error:
            raise RuntimeError("heartbeat writer failed") from self.error


def numerical_setup(device: str) -> None:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("frozen MPS device unavailable")
        torch.mps.set_per_process_memory_fraction(
            min(1.0, 24 * 1024**3 / torch.mps.recommended_max_memory())
        )


def parent_binding(intent: dict[str, Any], lineage: int) -> adapter.ParentSpec:
    bound = intent["parents"][str(lineage)]
    return adapter.ParentSpec(
        lineage,
        Path(bound["report_path"]),
        bound["report_sha256"],
        Path(bound["checkpoint_path"]),
        bound["checkpoint_sha256"],
    )


def check_caps(counters: dict[str, int], caps: dict[str, int]) -> None:
    for name, cap in caps.items():
        if name not in counters or not 0 <= counters[name] <= cap:
            raise RuntimeError(f"physical cap or missing counter: {name}")


def train(
    intent: dict[str, Any], stage: dict[str, Any], root: Path, out: Path, heartbeat: Heartbeat
) -> dict[str, Any]:
    qualification = read(root / "qualification/output/report.json")
    qualified = read(root / "qualified-dose.json")
    if qualified != runner.qualify_dose(qualification, intent):
        raise RuntimeError("qualified dose seal differs")
    dose = qualified["selected_rollouts"]
    lineage = stage["lineage"]
    loaded = adapter.load_parent(parent_binding(intent, lineage), intent["device"])
    trainer = adapter.build_trainer(
        loaded, intent["runtime_seeds"][str(lineage)], intent["device"], dose
    )
    marks = sorted({(dose // 20) * 5, (dose // 10) * 5, dose})
    snapshots = []
    started = time.monotonic()
    try:
        with (out / "training.jsonl").open("x") as log:
            for index in range(dose):
                row = trainer.update()
                check_caps(trainer.counters.as_dict(), stage["counter_caps"])
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                heartbeat.progress = {
                    "phase": "training",
                    "lineage": lineage,
                    "completed_rollouts": index + 1,
                    "selected_rollouts": dose,
                    "counters": trainer.counters.as_dict(),
                }
                if (index + 1) % 50 == 0:
                    log.flush()
                if index + 1 in marks:
                    path = out / f"checkpoint_rollout_{index + 1}.pth"
                    trainer.save_checkpoint(path)
                    snapshots.append(
                        {
                            "completed_rollouts": index + 1,
                            "global_update": trainer.update_idx,
                            "path": path.name,
                            "sha256": runner.digest(path),
                        }
                    )
            log.flush()
            os.fsync(log.fileno())
        if trainer.new_rollouts != dose:
            raise RuntimeError("fixed endpoint not reached")
        worlds = {
            name: [
                [
                    adapter._seed(runtime.run_seed, f"{name}/world/{env}/{episode}")
                    for episode in range(int(runtime.episode_ids[env]) + 1)
                ]
                for env in range(adapter.ENVIRONMENTS)
            ]
            for name, runtime in trainer.runtimes.items()
        }
        save(
            out / "training-world-seeds.json",
            {"worlds": worlds, "scope": "this study's actual episode namespaces only"},
        )
        return {
            "kind": "training",
            "lineage": lineage,
            "selected_rollouts": dose,
            "completed_rollouts": trainer.new_rollouts,
            "dose_sha256": qualified["dose_sha256"],
            "profile_rollouts": {
                "solo": trainer.per_profile["solo"]["rollouts"],
                "S2": trainer.per_profile["s2"]["rollouts"],
                "S6": trainer.per_profile["s6"]["rollouts"],
            },
            "scheduled_hero_slots": dose * 256,
            "starting_update": 4608,
            "final_update": trainer.update_idx,
            "starting_agent_steps": trainer.starting_agent_steps,
            "final_agent_steps": trainer.agent_steps,
            "per_profile": trainer.per_profile,
            "snapshots": snapshots,
            "final_checkpoint": snapshots[-1],
            "counters": trainer.counters.as_dict(),
            "elapsed_seconds": time.monotonic() - started,
            "task_contract": adapter.task_contract(),
            "promotion_eligible": False,
        }
    finally:
        trainer.close()


def eval_spec(intent: dict[str, Any], root: Path) -> dict[str, Any]:
    spec = dict(intent["evaluation"])
    worlds = set()
    for lineage in adapter.PARENT_SEEDS:
        report = read(root / f"train_{lineage}/output/report.json")
        endpoint = report["final_checkpoint"]
        path = root / f"train_{lineage}/output" / endpoint["path"]
        if runner.digest(path) != endpoint["sha256"]:
            raise RuntimeError("final endpoint seal changed before evaluation")
        used = read(root / f"train_{lineage}/output/training-world-seeds.json")["worlds"]
        worlds.update(seed for envs in used.values() for episodes in envs for seed in episodes)
    spec["training_world_seeds"] = sorted(worlds)
    evaluation.validate_spec(spec)
    return spec


def evaluate(
    intent: dict[str, Any], stage: dict[str, Any], root: Path, out: Path, heartbeat: Heartbeat
) -> dict[str, Any]:
    spec = eval_spec(intent, root)
    role = stage["role"]
    policies = []
    if role == "anchor":
        policies = [
            {"role": "anchor", "name": name, "lineage": None}
            for name in ("greedy_food", "random_safe")
        ]
    elif role == "parent":
        lineage = stage["lineage"]
        bound = intent["parents"][str(lineage)]
        policies = [
            {
                "role": role,
                "lineage": lineage,
                "checkpoint_sha256": bound["checkpoint_sha256"],
                "checkpoint_path": bound["checkpoint_path"],
                "parent_binding": bound,
            }
        ]
    else:
        lineage = stage["lineage"]
        report = read(root / f"train_{lineage}/output/report.json")
        endpoint = report["final_checkpoint"]
        policies = [
            {
                "role": role,
                "lineage": lineage,
                "checkpoint_sha256": endpoint["sha256"],
                "checkpoint_path": str(root / f"train_{lineage}/output" / endpoint["path"]),
            }
        ]
    files = []
    counts = dict.fromkeys(runner.COUNTERS, 0)
    save(out / "evaluation-spec.json", spec)
    for policy in policies:
        runtime = adapter.evaluation_runtime(spec, policy, device=stage["device"])
        for index, profile in enumerate(spec["profiles"]):
            heartbeat.progress = {
                "phase": "evaluation",
                "role": role,
                "identity": policy.get("lineage") or policy.get("name"),
                "profile": profile,
            }
            if index:
                runtime.initial_counters = dict.fromkeys(evaluation.COUNTERS, 0)
            path = out / f"{policy.get('name', str(policy.get('lineage')))}-{profile}.json"
            report = evaluation.evaluate_cell(spec, policy, profile, runtime, path)
            files.append(path.name)
            for key in counts:
                counts[key] += report["counters"][key]
            check_caps(counts, stage["counter_caps"])
        del runtime
        gc.collect()
    return {
        "kind": "evaluation_batch",
        "evaluation_reports": files,
        "evaluation_spec": "evaluation-spec.json",
        "counters": counts,
        "role": role,
        "lineage": stage.get("lineage"),
        "promotion_eligible": False,
    }


def analyze(intent: dict[str, Any], root: Path, out: Path) -> dict[str, Any]:
    reports = []
    spec = None
    for stage in intent["stages"]:
        if stage["kind"] != "evaluation":
            continue
        directory = root / stage["id"] / "output"
        report = read(directory / "report.json")
        this_spec = read(directory / report["evaluation_spec"])
        if spec is not None and this_spec != spec:
            raise RuntimeError("evaluation spec changed between jobs")
        spec = this_spec
        reports.extend(read(directory / name) for name in report["evaluation_reports"])
    result = evaluation.analyze_reports(spec, reports)
    save(out / "decision.json", result)
    from research.task_aligned_20260924.reporting import render_reports

    render_reports(
        spec,
        reports,
        result,
        {
            lineage: root / f"train_{lineage}/output/training.jsonl"
            for lineage in adapter.PARENT_SEEDS
        },
        out / "presentation",
    )
    result["presentation_manifest"] = "presentation/presentation.json"
    result["evaluation_spec"] = spec
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-id", required=True)
    args = parser.parse_args()
    intent = read(args.intent)
    stage = next(row for row in intent["stages"] if row["id"] == args.stage_id)
    out = args.out.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("stage report already exists")
    with Heartbeat(out / "heartbeat.json") as heartbeat:
        try:
            if stage["kind"] in {"qualification", "training", "evaluation"}:
                numerical_setup(stage.get("device", intent["device"]))
            if stage["kind"] == "qualification":
                from research.task_aligned_20260924.qualify import run_qualification

                result = run_qualification(intent, out, heartbeat)
            elif stage["kind"] == "training":
                result = train(intent, stage, args.root, out, heartbeat)
            elif stage["kind"] == "evaluation":
                result = evaluate(intent, stage, args.root, out, heartbeat)
            elif stage["kind"] == "analysis":
                result = analyze(intent, args.root, out)
            else:
                raise RuntimeError("audit uses its independent entry point")
            result.update(
                schema_version=1, study_id=intent["study_id"], stage_id=stage["id"], status="PASS"
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
